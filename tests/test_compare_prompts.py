"""Tests for the contrastive prompt evaluation (#192 PR2, D036): src/evaluation/ and
scripts/compare_prompts.py. No live API: labelers are fakes returning canned results."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from scripts import compare_prompts
from src.data_collection.models import (
    SKIP_INCOMPLETE_JUDGMENT,
    SKIP_NO_ESG_CATEGORIES,
    SKIP_NO_ESG_CONTENT,
)
from src.evaluation import comparator
from src.evaluation.comparator import (
    FAILED,
    MOST_BRANDS,
    NO_CATEGORY_STRATA,
    SAMPLE,
    SKIPPED_JUDGED,
    SKIPPED_UNJUDGED,
    ArticleRun,
    EvalArticle,
    Expectation,
    disposition_class,
    matching_prompt_example,
    prompt_examples,
    record_reply,
    run_articles,
    summarize,
    SideSettings,
)
from src.evaluation.diff_report import render_markdown
from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED
from src.labeling.labeler import LabelingResult
from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse
from src.labeling.config import llm_cost_usd

MAX_TOKENS = 2000
HAIKU_45 = "claude-haiku-4-5-20251001"
HAIKU_55 = "claude-haiku-5-5"
SIDE = SideSettings(model=HAIKU_45, max_tokens=MAX_TOKENS)
SIDES = {"baseline": SIDE, "candidate": SIDE}


def _esg(brand, sentiment=1):
    """A sportswear brand with the environmental category applying."""
    categories = {
        c: CategoryLabel(applies=False, sentiment=None, evidence=[])
        for c in ("social", "governance", "digital_transformation")
    }
    categories["environmental"] = CategoryLabel(applies=True, sentiment=sentiment, evidence=["e"])
    return BrandAnalysis(brand=brand, is_sportswear_brand=True, categories=categories, reasoning=f"{brand} esg")


def _no_esg(brand):
    return BrandAnalysis(brand=brand, is_sportswear_brand=True, categories={}, reasoning=f"{brand} product")


def _not_sportswear(brand):
    return BrandAnalysis(
        brand=brand, is_sportswear_brand=False, not_sportswear_reason="the animal", reasoning="no"
    )


def _ok(analyses, output_tokens=100, input_tokens=1000, stop_reason="end_turn"):
    return LabelingResult(
        success=True,
        response=LabelingResponse(brand_analyses=analyses, article_summary="s"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
    )


def _reply(analyses, brands=("Nike",), **kw):
    return record_reply(list(brands), _ok(analyses, **kw))


def _article(article_id="a1", brands=("Nike",), **kw):
    return EvalArticle(article_id=article_id, title=f"T {article_id}", content="x" * 300, brands=list(brands), **kw)


class FakeLabeler:
    """Returns queued results in order and records each call."""

    def __init__(self, results, model=HAIKU_45):
        self.results = list(results)
        self.calls = []
        self.model = model

    def label_article(self, title, content, brands, published_at=None, source_name=None):
        self.calls.append((title, tuple(brands), source_name))
        return self.results.pop(0)


class TestDispositionClass:
    @pytest.mark.parametrize(
        "status,reason,expected",
        [
            (LABELED, None, LABELED),
            (FALSE_POSITIVE, "x", FALSE_POSITIVE),
            (SKIPPED, SKIP_NO_ESG_CATEGORIES, SKIPPED_JUDGED),
            (SKIPPED, SKIP_NO_ESG_CONTENT, SKIPPED_UNJUDGED),
            (SKIPPED, SKIP_INCOMPLETE_JUDGMENT, SKIPPED_UNJUDGED),
        ],
    )
    def test_skips_split_by_whether_the_reply_judged(self, status, reason, expected):
        assert disposition_class(status, reason) == expected

    def test_unknown_skip_reason_raises(self):
        with pytest.raises(ValueError):
            disposition_class(SKIPPED, "something new")


class TestRecordReply:
    def test_parse_failure_is_a_failure_with_output_tokens(self):
        r = record_reply(["Nike"], LabelingResult(success=False, error="Failed to parse", output_tokens=40))
        assert (r.outcome, r.parse_failure) == (FAILED, True)

    def test_api_error_is_not_a_parse_failure(self):
        r = record_reply(["Nike"], LabelingResult(success=False, error="rate limit"))
        assert (r.outcome, r.parse_failure) == (FAILED, False)

    def test_outcome_verdicts_and_compliance(self):
        r = _reply([_not_sportswear("Puma")], brands=("Puma", "Nike"))
        assert r.outcome == SKIPPED_UNJUDGED and r.reason == SKIP_INCOMPLETE_JUDGMENT
        assert r.verdicts == {"puma": False}
        assert r.omitted == ("Nike",) and not r.complete
        assert "the animal" in r.explanations["puma"]

    def test_duplicates_are_counted_and_sportswear_wins_a_conflict(self):
        r = _reply([_esg("Nike"), _not_sportswear("nike"), _esg("Nike")])
        assert r.duplicates == 2
        assert r.verdicts == {"nike": True}
        assert r.outcome == LABELED and set(r.decisions) == {"nike"}

    def test_judged_skip(self):
        r = _reply([_no_esg("Nike")])
        assert r.outcome == SKIPPED_JUDGED and r.complete and r.decisions == {}


class TestPromptExamples:
    TEXT = 'A sale listing ("Skechers slippers \'perfect for cold months\' reduced to £19") or "short"'

    def test_quoted_examples_long_enough_to_be_titles(self):
        assert prompt_examples(self.TEXT) == ["Skechers slippers 'perfect for cold months' reduced to £19"]

    def test_title_matching_an_example(self):
        examples = prompt_examples(self.TEXT)
        title = "  Comfy  Skechers slippers  perfect for cold month reduced to £19"
        assert matching_prompt_example(title, examples) == examples[0]
        assert matching_prompt_example("Nike Air Max Metro Releases October 2026", examples) is None


class TestWorstCaseCost:
    def test_one_call_is_a_generous_input_estimate_and_a_full_output(self):
        article = EvalArticle(article_id="a", title="t", content="x" * 300, brands=["Nike"])
        # (3000 prompt chars + 300 content chars) / 3 chars per token
        assert comparator.worst_case_call_cost(article, 3000, SIDE) == pytest.approx(
            llm_cost_usd(1100, 2000, HAIKU_45)
        )

    def test_one_call_is_priced_at_its_sides_model_and_cap(self):
        article = EvalArticle(article_id="a", title="t", content="x" * 300, brands=["Nike"])
        side = SideSettings(model=HAIKU_55, max_tokens=8000)
        # Haiku 5.5's tokenizer: (3000 + 300) / 2 chars per token
        assert comparator.worst_case_call_cost(article, 3000, side) == pytest.approx(
            llm_cost_usd(1650, 8000, HAIKU_55)
        )

    def test_run_bound_covers_both_versions_and_every_repeat(self):
        a = EvalArticle(article_id="a", title="t", content="x" * 300, brands=["Nike"], repeats=3)
        calls = [(3000, SIDE), (6000, SIDE)]
        expected = 3 * (comparator.worst_case_call_cost(a, 3000, SIDE) + comparator.worst_case_call_cost(a, 6000, SIDE))
        assert comparator.worst_case_cost([a], calls) == pytest.approx(expected)


class TestRunArticles:
    def test_both_versions_back_to_back_once_per_repeat(self):
        baseline = FakeLabeler([_ok([_esg("Nike")])] * 3)
        candidate = FakeLabeler([_ok([_no_esg("Nike")])] * 3)
        articles = [_article("a", repeats=2), _article("b", source_name="Src")]
        runs, stopped = run_articles(
            articles, baseline, candidate, max_cost_usd=100, pair_bound=lambda a: 0.0
        )
        assert not stopped
        assert [len(r.baseline) for r in runs] == [2, 1] and [len(r.candidate) for r in runs] == [2, 1]
        assert candidate.calls[-1] == ("T b", ("Nike",), "Src")

    def test_cap_stops_before_a_pair_that_could_exceed_it(self):
        reply = _ok([_esg("Nike")], input_tokens=1_000_000, output_tokens=0)  # $1.00 each
        baseline, candidate = FakeLabeler([reply] * 5), FakeLabeler([reply] * 5)
        runs, stopped = run_articles(
            [_article("a"), _article("b"), _article("c")],
            baseline,
            candidate,
            max_cost_usd=3.5,
            pair_bound=lambda a: 2.0,
        )
        assert stopped
        # spent 2.0 after one pair; 2.0 + 2.0 > 3.5, so the second pair never starts
        assert len(baseline.calls) == 1 and len(candidate.calls) == 1
        assert [r.article.article_id for r in runs] == ["a"]

    def test_cap_is_checked_before_every_repeat_and_keeps_the_partial_case(self):
        reply = _ok([_esg("Nike")], input_tokens=1_000_000, output_tokens=0)  # $1.00 each
        baseline, candidate = FakeLabeler([reply] * 3), FakeLabeler([reply] * 3)
        runs, stopped = run_articles(
            [_article("a", repeats=3)], baseline, candidate, max_cost_usd=3.5, pair_bound=lambda a: 2.0
        )
        assert stopped and len(baseline.calls) == 1
        assert len(runs) == 1 and len(runs[0].baseline) == 1 and len(runs[0].candidate) == 1

    def test_cap_below_the_first_pair_makes_no_call(self):
        baseline, candidate = FakeLabeler([]), FakeLabeler([])
        runs, stopped = run_articles(
            [_article()], baseline, candidate, max_cost_usd=1.0, pair_bound=lambda a: 1.5
        )
        assert stopped and runs == [] and baseline.calls == [] and candidate.calls == []


def _run(article, baseline, candidate):
    return ArticleRun(article, baseline=list(baseline), candidate=list(candidate))


class TestSummarize:
    def test_newly_labeled_rate_is_over_baseline_non_labeled_by_class(self):
        runs = [
            _run(_article("1", strata=(SAMPLE,)), [_reply([_not_sportswear("Nike")])], [_reply([_esg("Nike")])]),
            _run(_article("2", strata=(SAMPLE,)), [_reply([])], [_reply([_esg("Nike")])]),
            _run(_article("3", strata=(SAMPLE,)), [_reply([_no_esg("Nike")])], [_reply([_no_esg("Nike")])]),
            _run(_article("4", strata=(SAMPLE,)), [_reply([_esg("Nike")])], [_reply([_no_esg("Nike")])]),
            # outside the sample: counted in "all" only
            _run(_article("5"), [_reply([])], [_reply([_esg("Nike")])]),
        ]
        s = summarize(runs, **SIDES)
        sample = s["newly_labeled_sample"]
        assert (sample["newly_labeled"], sample["denominator"]) == (2, 3)
        assert sample["by_baseline_class"][FALSE_POSITIVE] == {"denominator": 1, "newly_labeled": 1}
        assert sample["by_baseline_class"][SKIPPED_UNJUDGED] == {"denominator": 1, "newly_labeled": 1}
        assert sample["by_baseline_class"][SKIPPED_JUDGED] == {"denominator": 1, "newly_labeled": 0}
        assert (sample["dropped_labeled"], sample["baseline_labeled"]) == (1, 1)
        assert s["newly_labeled_all"]["newly_labeled"] == 3
        assert len(s["non_labeled_to_labeled"]) == 3 and len(s["labeled_to_non_labeled"]) == 1

    def test_four_class_transitions_and_the_unjudged_split(self):
        runs = [
            _run(_article("1"), [_reply([])], [_reply([_no_esg("Nike")])]),
            _run(_article("2"), [_reply([])], [_reply([_not_sportswear("Nike")])]),
        ]
        s = summarize(runs, **SIDES)
        assert s["transitions"] == {
            f"{SKIPPED_UNJUDGED} -> {FALSE_POSITIVE}": 1,
            f"{SKIPPED_UNJUDGED} -> {SKIPPED_JUDGED}": 1,
        }
        assert s["baseline_unjudged_split"] == {SKIPPED_JUDGED: 1, FALSE_POSITIVE: 1}
        assert s["baseline"]["unjudged_rate"] == 1.0 and s["candidate"]["unjudged_rate"] == 0.0
        assert s["brands_newly_judged"] == 2

    def test_fp_flips_category_flips_and_sentiment(self):
        runs = [
            _run(
                _article("1", brands=("Nike", "Puma")),
                [_reply([_esg("Nike", 1), _esg("Puma")], brands=("Nike", "Puma"))],
                [_reply([_esg("Nike", -1), _not_sportswear("Puma")], brands=("Nike", "Puma"))],
            )
        ]
        s = summarize(runs, **SIDES)
        assert [(f["brand"], f["baseline"], f["candidate"]) for f in s["fp_flips"]] == [("puma", True, False)]
        # Puma is labeled on one side only: an outcome change, not a category flip
        assert s["category_flips"] == 0 and s["brands_labeled_by_one_side"] == 1
        assert s["sentiment_agreement"] == 0.0  # Nike environmental: +1 vs -1

    def test_category_flip_on_a_brand_both_sides_labeled(self):
        social = _esg("Nike")
        social.categories["social"] = CategoryLabel(applies=True, sentiment=1, evidence=["e"])
        runs = [_run(_article("1"), [_reply([_esg("Nike")])], [_reply([social])])]
        s = summarize(runs, **SIDES)
        assert s["category_flips"] == 1 and s["brands_labeled_by_one_side"] == 0

    def test_compliance_and_duplicates_per_version(self):
        runs = [
            _run(
                _article("1", brands=("Nike", "Puma")),
                [_reply([_esg("Nike")], brands=("Nike", "Puma"))],
                [_reply([_esg("Nike"), _no_esg("Puma"), _no_esg("puma")], brands=("Nike", "Puma"))],
            )
        ]
        s = summarize(runs, **SIDES)
        assert s["baseline"]["compliance"] == 0.0 and s["baseline"]["incomplete_replies"] == 1
        assert s["candidate"]["compliance"] == 1.0
        assert s["candidate"]["duplicate_entries"] == 1 and s["candidate"]["replies_with_duplicates"] == 1

    def test_tokens_truncation_near_limit_and_most_brands_stratum(self):
        runs = [
            _run(_article("1", strata=(MOST_BRANDS,)), [_reply([_esg("Nike")], output_tokens=1900)],
                 [_reply([_esg("Nike")], output_tokens=MAX_TOKENS, stop_reason="max_tokens")]),
            _run(_article("2"), [_reply([_esg("Nike")], output_tokens=100)],
                 [_reply([_esg("Nike")], output_tokens=300)]),
        ]
        s = summarize(runs, **SIDES)
        assert s["candidate"]["tokens"]["truncated"] == 1 and s["candidate"]["tokens"]["near_limit"] == 0
        assert s["baseline"]["tokens"]["near_limit"] == 1 and s["baseline"]["tokens"]["truncated"] == 0
        assert s["candidate"]["tokens_most_brands"]["output_max"] == MAX_TOKENS
        assert s["candidate"]["tokens_most_brands"]["calls"] == 1
        assert s["candidate"]["tokens"]["cost_usd"] == pytest.approx(llm_cost_usd(2000, 2300, HAIKU_45))

    def test_truncation_is_the_stop_reason_not_the_output_count(self):
        """Output tokens include thinking, so a reply can stop at the cap below it, and a count
        at the cap is not a truncation unless the API says so."""
        runs = [
            _run(_article("1"), [_reply([_esg("Nike")], output_tokens=MAX_TOKENS)],
                 [_reply([_esg("Nike")], output_tokens=500, stop_reason="max_tokens")]),
        ]
        s = summarize(runs, **SIDES)
        assert s["baseline"]["tokens"]["truncated"] == 0 and s["baseline"]["tokens"]["near_limit"] == 1
        assert s["candidate"]["tokens"]["truncated"] == 1 and s["candidate"]["tokens"]["near_limit"] == 0

    def test_each_side_is_priced_at_its_own_model(self):
        runs = [_run(_article("1"), [_reply([_esg("Nike")])], [_reply([_esg("Nike")])])]
        s = summarize(runs, baseline=SIDE, candidate=SideSettings(model=HAIKU_55, max_tokens=8000))
        assert s["baseline"]["tokens"]["cost_usd"] == pytest.approx(llm_cost_usd(1000, 100, HAIKU_45))
        assert s["candidate"]["tokens"]["cost_usd"] == pytest.approx(llm_cost_usd(1000, 100, HAIKU_55))

    def test_refusals_are_counted_apart_from_parse_failures(self):
        refused = LabelingResult(
            success=False, error="declined", error_type="refusal", output_tokens=10, stop_reason="refusal"
        )
        runs = [_run(_article("1"), [_reply([_esg("Nike")])], [record_reply(["Nike"], refused)])]
        s = summarize(runs, **SIDES)
        assert s["candidate"]["refusals"] == 1 and s["candidate"]["parse_failures"] == 0
        assert s["candidate"]["outcomes"][FAILED] == 1

    def test_product_accuracy_buckets_per_case(self):
        expect = Expectation("Nike", True)
        runs = [
            # baseline: one reply judged Nike sportswear, one left it out -> correct
            # candidate: one judged true, one judged false -> wrong
            _run(
                _article("1", repeats=2, expected=expect),
                [_reply([_esg("Nike")]), _reply([])],
                [_reply([_no_esg("Nike")]), _reply([_not_sportswear("Nike")])],
            ),
            # baseline left the brand out (as v1.10.0 instructs) -> not judged, not a miss
            _run(_article("2", expected=expect), [_reply([])], [_reply([_no_esg("Nike")])]),
            # its title is a prompt example, so it is reported but not scored
            _run(
                _article("3", expected=expect, prompt_example="the example"),
                [_reply([_not_sportswear("Nike")])],
                [_reply([_not_sportswear("Nike")])],
            ),
        ]
        acc = summarize(runs, **SIDES)["product_accuracy"]
        b, c = acc["baseline"], acc["candidate"]
        assert (b["correct"], b["wrong"], b["not_judged"], b["rate"]) == (1, 0, 1, 1.0)
        assert (c["correct"], c["wrong"], c["not_judged"], c["rate"]) == (1, 1, 0, 0.5)
        assert c["misses"][0]["article_id"] == "1" and c["misses"][0]["got"] == [True, False]
        assert [x["article_id"] for x in b["not_judged_cases"]] == ["2"]
        s = summarize(runs, **SIDES)
        assert [x["article_id"] for x in s["prompt_example_cases"]] == ["3"]

    def test_stability_consistency_and_completeness(self):
        runs = [
            _run(_article("same", repeats=2), [_reply([])] * 2, [_reply([_no_esg("Nike")])] * 2),
            _run(_article("varies", repeats=2), [_reply([_esg("Nike")]), _reply([])], [_reply([])] * 2),
            # cut short by the spend cap: one of three repeats ran
            _run(_article("short", repeats=3), [_reply([])], [_reply([])]),
            _run(_article("single"), [_reply([])], [_reply([])]),
        ]
        stab = {s["article_id"]: s for s in summarize(runs, **SIDES)["stability"]}
        assert set(stab) == {"same", "varies", "short"}
        assert stab["same"]["baseline_consistent"] and stab["same"]["candidate_consistent"]
        assert not stab["varies"]["baseline_consistent"] and stab["varies"]["candidate_consistent"]
        assert stab["short"]["complete"] is False
        assert not stab["short"]["baseline_consistent"] and not stab["short"]["candidate_consistent"]

    def test_transitions_use_the_first_repeat(self):
        runs = [_run(_article("1", repeats=2), [_reply([]), _reply([_esg("Nike")])],
                     [_reply([_esg("Nike")]), _reply([])])]
        s = summarize(runs, **SIDES)
        assert s["transitions"] == {f"{SKIPPED_UNJUDGED} -> {LABELED}": 1}

    def test_failed_replies_are_kept_out_of_rates(self):
        parse_failure = record_reply(["Nike"], LabelingResult(success=False, error="parse", output_tokens=50))
        runs = [
            _run(_article("1", strata=(SAMPLE,)), [_reply([])], [parse_failure]),
            _run(_article("2", strata=(SAMPLE,)), [parse_failure], [_reply([_esg("Nike")])]),
            _run(_article("3", strata=(SAMPLE,)), [_reply([])], [_reply([_esg("Nike")])]),
        ]
        s = summarize(runs, **SIDES)
        assert s["baseline"]["parse_failures"] == 1 and s["candidate"]["parse_failures"] == 1
        assert s["baseline"]["outcomes"][FAILED] == 1
        nl = s["newly_labeled_sample"]
        assert (nl["newly_labeled"], nl["denominator"]) == (1, 1)
        assert [x["article_id"] for x in s["non_labeled_to_labeled"]] == ["3"]
        assert s["labeled_to_non_labeled"] == []
        assert s["brands_newly_judged"] == 1  # article 2's baseline failed, so it is not counted
        assert s["candidate"]["compliance"] == 1.0  # over the two answered replies
        assert s["candidate"]["tokens"]["calls"] == 3

    def test_a_failed_candidate_reply_is_not_a_labeled_to_non_labeled_transition(self):
        parse_failure = record_reply(["Nike"], LabelingResult(success=False, error="parse", output_tokens=50))
        runs = [
            _run(_article("failed"), [_reply([_esg("Nike")])], [parse_failure]),
            _run(_article("dropped"), [_reply([_esg("Nike")])], [_reply([_no_esg("Nike")])]),
        ]
        s = summarize(runs, **SIDES)
        assert [x["article_id"] for x in s["labeled_to_non_labeled"]] == ["dropped"]

    def test_newly_judged_counts_sent_brands_only(self):
        runs = [_run(_article("1"), [_reply([])], [_reply([_no_esg("Nike"), _no_esg("Vans")])])]
        s = summarize(runs, **SIDES)
        assert s["brands_newly_judged"] == 1
        assert s["candidate"]["replies_with_unsent_brands"] == 1

    def test_compliance_uses_the_pipelines_definition(self):
        unset = BrandAnalysis(brand="Converse", reasoning="mentioned")  # no verdict stated
        runs = [_run(
            _article("1", brands=("Nike", "Converse")),
            [_reply([_esg("Nike")], brands=("Nike", "Converse"))],
            [_reply([_esg("Nike"), unset, _no_esg("Converse")], brands=("Nike", "Converse"))],
        )]
        s = summarize(runs, **SIDES)
        # the baseline left Converse out; the candidate judged it, beside an entry with no verdict
        assert s["baseline"]["incomplete_replies"] == 1 and s["baseline"]["compliance"] == 0.0
        assert s["candidate"]["incomplete_replies"] == 0 and s["candidate"]["compliance"] == 1.0
        assert s["candidate"]["replies_with_unset_entries"] == 1
        bare = _reply([unset], brands=("Converse",))
        assert bare.omitted == ("Converse",) and not bare.complete

    def test_noise_floor_is_sample_rows_stored_labeled_under_the_baseline(self):
        sample = (SAMPLE,)
        runs = [
            _run(_article("1", strata=sample, stored_status=LABELED, stored_under_baseline=True),
                 [_reply([_esg("Nike")])], [_reply([_esg("Nike")])]),
            _run(_article("2", strata=sample, stored_status=LABELED, stored_under_baseline=True),
                 [_reply([])], [_reply([_esg("Nike")])]),
            # a named case stored under the baseline is not in the floor
            _run(_article("3", stored_status=LABELED, stored_under_baseline=True), [_reply([])], [_reply([])]),
            _run(_article("4", strata=sample, stored_status=LABELED), [_reply([])], [_reply([])]),
        ]
        floor = summarize(runs, **SIDES)["noise_floor"]
        assert (floor["still_labeled"], floor["articles"]) == (1, 2)

    def test_report_renders(self):
        runs = [_run(_article("1", strata=(SAMPLE,)), [_reply([])], [_reply([_esg("Nike")])])]
        config = {
            "baseline": "v1.10.0", "candidate": "v1.11.0",
            "baseline_model": HAIKU_45, "candidate_model": HAIKU_45,
            "baseline_max_tokens": MAX_TOKENS, "candidate_max_tokens": MAX_TOKENS,
            "effort": {"baseline": None, "candidate": None},
            "baseline_hashes": "a/b", "candidate_hashes": "c/d", "stopped_early": False,
        }
        md = render_markdown(summarize(runs, **SIDES), config)
        assert f"v1.10.0 on {HAIKU_45} (baseline) vs v1.11.0 on {HAIKU_45} (candidate)" in md
        assert "effort not sent" in md
        assert "### Non-labeled -> labeled (1)" in md
        assert "1 of 1 baseline non-labeled replies labeled by the candidate" in md


def _typed(brand, content_type, applying=("environmental",)):
    """A v1.12.0 brand analysis: the content type derives the verdict and the gate (D037)."""
    categories = {
        c: {"applies": c in applying, "sentiment": 1 if c in applying else None, "evidence": []}
        for c in ("environmental", "social", "governance", "digital_transformation")
    }
    return BrandAnalysis.model_validate({
        "brand": brand, "reasoning": f"{brand} {content_type}", "content_type": content_type,
        "categories": categories, "confidence": 0.9,
    })


class TestContentTypeReport:
    """What the report adds for a content-type candidate (#219, D037)."""

    def test_record_reply_keeps_types_kept_categories_and_gated_ones(self):
        r = _reply([_typed("Nike", "product"), _typed("Puma", "business")], brands=("Nike", "Puma"))
        assert r.content_types == {"nike": "product", "puma": "business"}
        assert r.gated == {"nike": ("environmental",)}
        assert r.categories == {"nike": (), "puma": ("environmental",)}

    def test_a_reply_without_types_records_none(self):
        r = _reply([_esg("Nike")])
        assert r.content_types == {} and r.gated == {}
        assert r.categories == {"nike": ("environmental",)}

    def test_summary_counts_types_and_lists_the_gate(self):
        runs = [
            _run(_article("1"), [_reply([_esg("Nike")])], [_reply([_typed("Nike", "financial", ("governance",))])]),
            _run(_article("2"), [_reply([])], [_reply([_typed("Nike", "retail_only", ())])]),
        ]
        s = summarize(runs, **SIDES)
        assert s["candidate"]["content_types"] == {"financial": 1, "retail_only": 1}
        assert s["baseline"]["content_types"] == {}
        [gated] = s["gated_categories"]["candidate"]
        assert (gated["article_id"], gated["content_type"], gated["categories"]) == ("1", "financial", ["governance"])
        assert s["gated_categories"]["baseline"] == []

    def test_categories_on_product_and_financial_cases_are_listed(self):
        expect = Expectation("Nike", True)
        runs = [
            _run(_article("1", strata=("product_true",), expected=expect), [_reply([_esg("Nike")])], [_reply([_typed("Nike", "product")])]),
            _run(_article("2", strata=("financial",), expected=expect), [_reply([_esg("Nike")])], [_reply([_typed("Nike", "business")])]),
            # a case outside the Product and financial strata is not checked
            _run(_article("3", strata=("stability",), expected=expect), [_reply([_esg("Nike")])], [_reply([_esg("Nike")])]),
            # a no-category case that names no brand is checked on every brand
            _run(
                _article("4", strata=("no_category",), brands=("Nike", "Puma")),
                [_reply([_esg("Nike"), _no_esg("Puma")], brands=("Nike", "Puma"))],
                [_reply([_typed("Nike", "mention_only"), _typed("Puma", "mention_only")], brands=("Nike", "Puma"))],
            ),
        ]
        cats = summarize(runs, **SIDES)["case_categories"]
        assert [(x["article_id"], x["brand"]) for x in cats["baseline"]] == [("1", "Nike"), ("2", "Nike"), ("4", "nike")]
        assert [(x["article_id"], x["categories"]) for x in cats["candidate"]] == [("2", ["environmental"])]

    def test_expected_categories_are_checked_per_reply(self):
        social = Expectation("Nike", True, ("social",))
        runs = [
            # matched: the brand gets exactly the ruled category
            _run(_article("1", expected=social), [_reply([_typed("Nike", "business", ("social",))])],
                 [_reply([_typed("Nike", "business", ("social",))])]),
            # a reply that leaves the brand out, and a content type that keeps no category
            _run(_article("2", expected=social), [_reply([])], [_reply([_typed("Nike", "product", ("social",))])]),
            # an extra category is a miss too
            _run(_article("3", expected=social), [_reply([_typed("Nike", "business", ("social",))])],
                 [_reply([_typed("Nike", "business", ("social", "governance"))])]),
            # a case that rules no categories is not checked
            _run(_article("4", expected=Expectation("Nike", True)), [_reply([_esg("Nike")])], [_reply([_esg("Nike")])]),
        ]
        cats = summarize(runs, **SIDES)["expected_categories"]
        assert (cats["baseline"]["matched"], cats["baseline"]["cases"]) == (2, 3)
        assert [m["article_id"] for m in cats["baseline"]["misses"]] == ["2"]
        assert (cats["candidate"]["matched"], cats["candidate"]["cases"]) == (1, 3)
        two, three = cats["candidate"]["misses"]
        assert (two["content_type"], two["missing"], two["extra"]) == ("product", ["social"], [])
        assert (three["missing"], three["extra"]) == ([], ["governance"])

    def test_case_file_reads_expected_categories_and_refuses_unknown_ones(self):
        case = {"article_id": "a", "expected": {"brand": "Deckers", "is_sportswear_brand": True, "categories": ["social"]}}
        specs = compare_prompts.parse_case_file({"cases": [case]})
        assert specs["a"].expected == Expectation("Deckers", True, ("social",))
        case["expected"]["categories"] = ["sociall"]
        with pytest.raises(ValueError, match="unknown expected categories"):
            compare_prompts.parse_case_file({"cases": [case]})

    def test_accuracy_reports_the_always_true_score(self):
        runs = [
            _run(_article("1", expected=Expectation("Nike", True)), [_reply([_no_esg("Nike")])], [_reply([_no_esg("Nike")])]),
            _run(_article("2", expected=Expectation("Nike", False)), [_reply([_no_esg("Nike")])], [_reply([_not_sportswear("Nike")])]),
            _run(_article("3", expected=Expectation("Nike", False)), [_reply([])], [_reply([_no_esg("Nike")])]),
        ]
        acc = summarize(runs, **SIDES)["product_accuracy"]
        # over judged cases only: the baseline judged two, one expected true
        assert (acc["baseline"]["always_true"], acc["baseline"]["always_true_rate"]) == (1, 0.5)
        assert (acc["candidate"]["always_true"], acc["candidate"]["correct"]) == (1, 2)

    def test_report_renders_the_new_sections(self):
        runs = [_run(
            _article("1", strata=("product_true",), expected=Expectation("Nike", True)),
            [_reply([_esg("Nike")])],
            [_reply([_typed("Nike", "product")])],
        )]
        config = {
            "baseline": "v1.10.0", "candidate": "v1.12.0",
            "baseline_model": HAIKU_55, "candidate_model": HAIKU_55,
            "baseline_max_tokens": 8000, "candidate_max_tokens": 8000,
            "effort": {"baseline": "medium", "candidate": "medium"},
            "baseline_hashes": "a/b", "candidate_hashes": "c/d/schema e", "stopped_early": False,
        }
        md = render_markdown(summarize(runs, **SIDES), config)
        assert "| Content types stated (brand entries) | none | product 1 |" in md
        assert "answering `true` for every judged case scores 1, 100.0%" in md
        assert "## Categories on Product, financial and no-category cases" in md
        assert "## Categories the content type dropped (D037 gate)" in md
        assert "## Expected categories (cases that rule them)" in md
        assert "- candidate: 0 of 0 cases matched" in md
        assert "T 1 (`1`), nike (product): environmental. Nike product" in md

    def test_prompt_hashes_name_the_schema_when_there_is_one(self):
        from src.labeling.prompt_manager import PromptManager

        manager = PromptManager()
        assert compare_prompts.prompt_hashes(manager.load_version("v1.11.0")) == "e3a32befdb40/0ede675fa4b0"
        assert compare_prompts.prompt_hashes(manager.load_version("v1.12.0")).endswith("/schema 96064689a1f3")


def _contrast(b_version, c_version, b_side, c_side):
    compare_prompts.check_contrast(
        {"baseline": b_version, "candidate": c_version}, {"baseline": b_side, "candidate": c_side}
    )


H55_LOW = SideSettings(HAIKU_55, 8000, "low")
H55_MEDIUM = SideSettings(HAIKU_55, 8000, "medium")


class TestScriptGuards:
    def test_equal_sides_are_refused(self):
        with pytest.raises(compare_prompts.EvalSetupError, match="same version"):
            _contrast("v1.10.0", "v1.10.0", SIDE, SIDE)

    @pytest.mark.parametrize("c_side", [H55_MEDIUM, SideSettings(HAIKU_55, 8000, "low")])
    def test_sides_differing_in_prompt_and_call_are_refused(self, c_side):
        b_side = SIDE if c_side is H55_MEDIUM else H55_MEDIUM
        with pytest.raises(compare_prompts.EvalSetupError, match="vary one at a time"):
            _contrast("v1.10.0", "v1.11.0", b_side, c_side)

    @pytest.mark.parametrize(
        "versions,sides",
        [
            (("v1.10.0", "v1.11.0"), (SIDE, SIDE)),
            (("v1.11.0", "v1.11.0"), (SIDE, H55_MEDIUM)),
            (("v1.11.0", "v1.11.0"), (H55_LOW, H55_MEDIUM)),
        ],
    )
    def test_sides_differing_in_one_are_accepted(self, versions, sides):
        _contrast(*versions, *sides)

    def test_an_alias_and_its_dated_snapshot_are_one_model(self):
        """Two names for one model would report its self-disagreement as a model difference."""
        with pytest.raises(compare_prompts.EvalSetupError, match="same version"):
            _contrast("v1.11.0", "v1.11.0", SIDE, SideSettings("claude-haiku-4-5", MAX_TOKENS))

    def test_a_model_without_a_spec_is_refused(self):
        with pytest.raises(compare_prompts.EvalSetupError, match="No pricing"):
            compare_prompts.side_settings("claude-unknown-1", None, "medium")

    def test_an_unknown_effort_is_refused_for_a_model_that_takes_effort(self):
        with pytest.raises(compare_prompts.EvalSetupError, match="Effort must be one of"):
            compare_prompts.side_settings(HAIKU_55, None, "extreme")

    def test_side_settings_record_effort_only_where_the_model_takes_it(self):
        assert compare_prompts.side_settings(HAIKU_45, None, "low") == SideSettings(HAIKU_45, 2000, None)
        assert compare_prompts.side_settings(HAIKU_55, 4000, "low") == SideSettings(HAIKU_55, 4000, "low")

    def test_one_model_at_two_efforts_without_effort_support_is_the_same_call(self):
        """Haiku 4.5 is sent no effort, so asking for two efforts compares a setup with itself."""
        b = compare_prompts.side_settings(HAIKU_45, None, "low")
        c = compare_prompts.side_settings(HAIKU_45, None, "high")
        with pytest.raises(compare_prompts.EvalSetupError, match="same version"):
            _contrast("v1.11.0", "v1.11.0", b, c)

    def test_unloadable_version_is_refused(self):
        manager = MagicMock()
        manager.load_version.side_effect = ValueError("Unknown prompt version: v9")
        with pytest.raises(compare_prompts.EvalSetupError, match="does not load"):
            compare_prompts.check_prompt_versions("v1.10.0", "v9", manager)

    def test_a_manager_that_loads_another_version_is_refused(self):
        manager = MagicMock()
        manager.load_version.side_effect = lambda v: MagicMock(version="v1.10.0")
        with pytest.raises(compare_prompts.EvalSetupError, match="loaded v1.10.0"):
            compare_prompts.check_prompt_versions("v1.9.0", "v1.11.0", manager)

    def test_labeler_that_fell_back_to_the_hardcoded_prompt_is_refused(self):
        labeler = MagicMock(loaded_prompt=None, prompt_version="legacy", model=HAIKU_45)
        with pytest.raises(compare_prompts.EvalSetupError, match="legacy"):
            compare_prompts.check_labeler(labeler, "v1.11.0", SIDE)

    def test_labeler_with_another_version_is_refused(self):
        labeler = MagicMock(loaded_prompt=object(), prompt_version="v1.10.0", model=HAIKU_45)
        with pytest.raises(compare_prompts.EvalSetupError):
            compare_prompts.check_labeler(labeler, "v1.11.0", SIDE)
        compare_prompts.check_labeler(
            MagicMock(loaded_prompt=object(), prompt_version="v1.11.0", model=HAIKU_45), "v1.11.0", SIDE
        )

    def test_labeler_with_another_model_is_refused(self):
        labeler = MagicMock(loaded_prompt=object(), prompt_version="v1.11.0", model=HAIKU_45)
        with pytest.raises(compare_prompts.EvalSetupError, match="runs"):
            compare_prompts.check_labeler(labeler, "v1.11.0", H55_MEDIUM)

    def test_labeler_with_another_effort_is_refused(self):
        labeler = MagicMock(loaded_prompt=object(), prompt_version="v1.11.0", model=HAIKU_55, effort="medium")
        with pytest.raises(compare_prompts.EvalSetupError, match="effort"):
            compare_prompts.check_labeler(labeler, "v1.11.0", H55_LOW)


class TestSelection:
    from datetime import datetime as _dt, timezone as _tz

    NOW = _dt(2026, 10, 20, tzinfo=_tz.utc)

    @pytest.mark.parametrize(
        "source,status,error,expected",
        [
            ("llm", "labeled", None, True),
            ("unjudged", "skipped", "No ESG content found", True),
            ("unjudged", "skipped", "Insufficient content", False),
            ("unjudged", "skipped", "No brands mentioned", False),
            ("pre_classifier", "labeled", None, False),
            ("fp_classifier", "false_positive", "x", False),
            ("human", "false_positive", "x", False),
        ],
    )
    def test_pool_is_replies_of_the_labeling_llm(self, source, status, error, expected):
        assert compare_prompts.replied_on_by_llm(source, status, error) is expected

    def test_human_ruled_and_pre_llm_rows_leave_the_pool(self):
        from src.data_collection.provenance import HUMAN_RULINGS

        ruled = next(k for k, v in HUMAN_RULINGS.items() if v == "false_positive")
        rows = [
            ("llm-1", "labeled", None, ["Nike"], self.NOW, None, "continued_to_llm", None),
            # without the ruling this row derives as `llm` (an FP prediction exists) and would be in the pool
            (ruled, "false_positive", "Puma: the animal", ["Puma"], self.NOW, None, "continued_to_llm", None),
            ("short", "skipped", "Insufficient content", ["Nike"], self.NOW, None, None, None),
            ("nobrands", "labeled", None, [], self.NOW, None, "continued_to_llm", None),
        ]
        assert [c.article_id for c in compare_prompts.candidates_from_rows(rows)] == ["llm-1"]
        # an unruled id with the identical row is an LLM reply, so only the ruling removed it
        rows[1] = ("unruled", *rows[1][1:])
        assert [c.article_id for c in compare_prompts.candidates_from_rows(rows)] == ["llm-1", "unruled"]
        # the ruled id whose status no longer matches its ruling is an LLM reply
        rows[1] = (ruled, "labeled", None, ["Puma"], self.NOW, None, "continued_to_llm", None)
        assert [c.article_id for c in compare_prompts.candidates_from_rows(rows)] == ["llm-1", ruled]

    def test_a_date_without_a_timezone_is_read_as_utc(self):
        from datetime import datetime, timedelta, timezone

        naive_published = datetime(2026, 10, 1, 12, 0)
        naive_created = datetime(2026, 9, 30, 8, 0)
        rows = [
            ("pub", "labeled", None, ["Nike"], naive_published, None, "continued_to_llm", None),
            ("made", "labeled", None, ["Nike"], None, naive_created, "continued_to_llm", None),
        ]
        cands = compare_prompts.candidates_from_rows(rows)
        assert [c.dated for c in cands] == [
            naive_published.replace(tzinfo=timezone.utc),
            naive_created.replace(tzinfo=timezone.utc),
        ]
        # comparable with the aware window start the script builds
        assert sorted(compare_prompts.pick_sample(cands, 5, 42, self.NOW - timedelta(days=180))) == ["made", "pub"]

    def test_most_brands_ranks_descending_with_id_ties(self):
        c = compare_prompts.Candidate
        cands = [c("b", 3, None), c("a", 3, None), c("c", 9, None), c("d", 1, self.NOW)]
        assert compare_prompts.pick_most_brands(cands, 3) == ["c", "a", "b"]

    def test_sample_is_seeded_and_windowed(self):
        from datetime import timedelta

        c = compare_prompts.Candidate
        old = self.NOW - timedelta(days=400)
        cands = [c(f"in{i}", 1, self.NOW) for i in range(20)] + [c("old", 1, old), c("undated", 1, None)]
        since = self.NOW - timedelta(days=180)
        first = compare_prompts.pick_sample(cands, 5, 42, since)
        assert first == compare_prompts.pick_sample(list(reversed(cands)), 5, 42, since)
        assert first != compare_prompts.pick_sample(cands, 5, 7, since)
        everything = compare_prompts.pick_sample(cands, 100, 42, since)
        assert len(everything) == 20 and "old" not in everything and "undated" not in everything

    def _row(self, **kw):
        row = dict(id="a1", title="Some title", full_content="x" * 200, description=None,
                   brands_mentioned=["Nike", "Brooks Running"], published_at=None,
                   source_name="S", labeling_status="labeled")
        row.update(kw)
        return MagicMock(**row)

    def test_build_drops_what_the_pipeline_would_not_send(self):
        spec = compare_prompts.Spec()
        assert compare_prompts.build_eval_article(self._row(full_content="short"), spec, False, []) is None
        assert compare_prompts.build_eval_article(self._row(brands_mentioned=[]), spec, False, []) is None

    def test_build_refuses_an_expected_brand_the_article_does_not_list(self):
        spec = compare_prompts.Spec(expected=Expectation("Brooks", False))
        with pytest.raises(compare_prompts.EvalSetupError, match="Brooks"):
            compare_prompts.build_eval_article(self._row(), spec, False, [])
        ok = compare_prompts.Spec(expected=Expectation("brooks running", False))
        assert compare_prompts.build_eval_article(self._row(), ok, False, []).expected == ok.expected

    def test_build_prefers_the_case_files_prompt_example_and_requires_labeled_for_the_floor(self):
        examples = ["Some title"]
        named = compare_prompts.Spec(prompt_example="named example")
        built = compare_prompts.build_eval_article(self._row(), named, True, examples)
        assert built.prompt_example == "named example" and built.stored_under_baseline
        auto = compare_prompts.build_eval_article(self._row(), compare_prompts.Spec(), False, examples)
        assert auto.prompt_example == "Some title" and not auto.stored_under_baseline
        skipped = compare_prompts.build_eval_article(self._row(labeling_status="skipped"), named, True, [])
        assert not skipped.stored_under_baseline


class TestNoiseFloorModel:
    ROWS = [
        ("a", "claude-haiku-4-5-20251001"),
        ("b", "claude-haiku-4-5"),
        ("c", HAIKU_55),
        ("d", "claude-haiku-5-5-preview"),
        ("e", "claude-3-5-sonnet-20240620"),
        ("f", None),
    ]

    @pytest.mark.parametrize(
        "model,expected",
        [("claude-haiku-4-5", {"a", "b"}), (HAIKU_45, {"a", "b"}), (HAIKU_55, {"c"})],
    )
    def test_stored_models_are_matched_by_table_key(self, model, expected):
        """A variant id, or a model the table no longer lists, is not the baseline model."""
        assert compare_prompts.labeled_by_model(self.ROWS, model) == expected

    def test_load_articles_puts_only_the_baseline_prompt_and_model_in_the_floor(self):
        from src.data_collection.models import Article, BrandLabel

        def row(article_id):
            return MagicMock(id=article_id, title="Some title", full_content="x" * 200,
                             description=None, brands_mentioned=["Nike"], published_at=None,
                             source_name="S", labeling_status="labeled")

        articles_q, labels_q = MagicMock(), MagicMock()
        articles_q.filter.return_value.all.return_value = [row("a1"), row("a2")]
        labels_q.filter.return_value.distinct.return_value = [("a1", HAIKU_45), ("a2", HAIKU_55)]
        session = MagicMock()
        session.query.side_effect = lambda *cols: articles_q if cols[0] is Article else labels_q

        specs = {"a1": compare_prompts.Spec(), "a2": compare_prompts.Spec()}
        built = compare_prompts.load_articles(session, specs, "v1.10.0", HAIKU_45, [])

        assert {a.article_id: a.stored_under_baseline for a in built} == {"a1": True, "a2": False}
        # Compared by identity: == on a mapped column builds SQL instead of comparing
        label_cols = [c.args for c in session.query.call_args_list if c.args[0] is not Article]
        assert len(label_cols) == 1
        assert label_cols[0][0] is BrandLabel.article_id and label_cols[0][1] is BrandLabel.model_version
        clauses = labels_q.filter.call_args.args
        assert any(
            str(c.left) == "brand_labels.prompt_version" and c.right.value == "v1.10.0"
            for c in clauses
        )


class TestCaseFile:
    def test_parse_and_merge(self):
        specs = compare_prompts.parse_case_file({
            "cases": [
                {"article_id": "a", "repeats": 3, "strata": ["stability"],
                 "expected": {"brand": "Puma", "is_sportswear_brand": False}},
                {"article_id": "b", "prompt_example": "Reebok Angel Reese 1 releases November 14"},
            ]
        })
        assert specs["a"].repeats == 3 and specs["a"].expected == Expectation("Puma", False)
        assert specs["a"].strata == {"cases", "stability"}
        assert specs["b"].prompt_example == "Reebok Angel Reese 1 releases November 14"
        compare_prompts.merge_spec(specs, "a", SAMPLE)
        compare_prompts.merge_spec(specs, "c", SAMPLE)
        assert specs["a"].strata == {"cases", "stability", SAMPLE} and specs["a"].repeats == 3
        assert specs["c"].strata == {SAMPLE}

    def test_committed_case_file_parses(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "prompts/labeling/eval_cases/product_and_stability.json"
        specs = compare_prompts.parse_case_file(json.loads(path.read_text()))
        assert sum(1 for s in specs.values() if "stability" in s.strata) == 7
        assert all(s.repeats == 3 for s in specs.values() if "stability" in s.strata)
        product = [s for s in specs.values() if s.strata & {"product_true", "product_false"}]
        assert {s.expected.is_sportswear_brand for s in product} == {True, False}

    # Each expected verdict was read from its article by a person; one changes only with this list
    EXPECTED = {
        "3a637c91-2fd3-4f8e-86be-f1ea45780808": ("Adidas", True),
        "dcf616c4-7fb5-47b2-b430-6175902ba489": ("Puma", False),
        "7118e53b-f6c0-4d7f-8877-cb62aec7f350": ("New Balance", True),
        "cfdf8b6f-861d-4b75-95f0-17c2f91c7d1f": ("Hoka", True),
        "6672e27c-6607-4880-a5cc-30700a5b34df": ("Mammut", True),
        "d3b1df33-2bf6-40f0-9775-ce4ec6782c6d": ("Nike", True),
        "f928bd32-2bc8-44f1-a658-e9bde970f924": ("Nike", True),
        "ec94386e-4263-45e0-b431-2e11e4adb905": ("Nike", True),
        "edf23d74-19c6-4435-b3d5-e7e5b34c74ad": ("Lululemon", True),
        "9a2e5cf2-be8f-46cb-b503-d2b4dfbb6403": ("Reebok", True),
        "c8f6ba97-85f3-4b5f-8552-43c2a2a2bb66": ("Adidas", False),
        "442282ca-90ae-4811-a7f5-1026bc8fcf75": ("Skechers", True),
        "02ffd32e-cca0-476d-88f8-1486c790f421": ("Lululemon", False),
        "94a563b7-2c67-4c74-9884-66f7a4f35ee8": ("Converse", True),
        "5812c8dc-58a9-4c53-b75e-394e9844322e": ("Lululemon", False),
        "b7eea904-4747-4974-a958-abd78ab0df32": ("Puma", False),
        "e7968869-86d2-4692-b06c-071d7a9b1de4": ("The North Face", False),
        "63f9fd9b-a7c8-4e49-8baf-52291f925d44": ("Adidas", False),
        "6d6195b9-5b71-478f-bc38-63e5743381b6": ("Nike", False),
        "de535470-9a93-413a-a69b-e8ca7b6d0028": ("Nike", False),
        "2d4ff51c-d1e9-49e9-bb48-48561bbdd06c": ("Nike", True),
        "eb5aa27c-2fc8-4db4-a5b7-a202b116c4ea": ("Timberland", False),
        "bcc52f23-6048-40e3-ab75-ed701468a5a2": ("Skechers", True),
        "6ec5fd75-5e7a-4e06-b53f-58b4326cddba": ("Brooks Running", True),
        "3c93143f-4a9c-44f0-845b-3c7822450cfa": ("Patagonia", True),
        "efa10606-e4c9-4f33-9c7c-eaa46140e8b8": ("Merrell", True),
        "ed1e5bad-d330-43f9-bb87-d102c9691054": ("Reebok", True),
        # #219: #118's corrections and #215 run 2's financial cases, owner-ruled 2026-10-22
        "a7eeaa13-de2d-4ed6-83ce-64f74a4bbda0": ("Nike", False),
        "b8891fbc-7d8a-40b2-9b9b-527d9d373858": ("Adidas", False),
        "12396c87-b2a4-4563-ac1b-d2923202afe2": ("Nike", True),
        "2df21233-ef21-4ec7-87b7-44334f90ac19": ("Nike", True),
        "1ba1c2a2-ac8a-4077-ac44-11feae1a08ec": ("Nike", True),
        "23c30b60-9671-4613-a93a-893d14830a14": ("Nike", True),
        "a058da6e-8656-4dc7-901a-fc1d56d4b745": ("Nike", True),
        "ea428de7-23a8-4004-80a7-5323c26e2f1a": ("Nike", True),
        "9e973f0c-458b-4d3d-bf2a-55f3e3439149": ("Nike", True),
        "eded2885-014c-45d1-8164-4bd6605a0618": ("Mountain Hardwear", True),
        "3b4de2e6-78d4-47b9-bea1-fe298967a6d0": ("Nike", True),
        "7dccbf22-de0e-4e60-acb5-f562b551fafc": ("Adidas", True),
        "cfe9e317-aed2-4c28-9c7b-ca9450d14e59": ("Adidas", True),
        "89330b08-e205-48b3-a233-ecce8b03b2d3": ("Cotopaxi", True),
        "cdb26552-8643-4007-a403-4f0451447e2f": ("Deckers", True),
        "3fee9d78-c7a8-4936-8ff0-2b14edadbcce": ("Nike", True),
        "a39e8011-0d4b-4c77-b7c9-c4b8857d7f72": ("Nike", True),
        "e434b12d-eb59-4406-bd26-f3bad800aaf1": ("Nike", False),
        "19832789-ca96-4c74-ba7a-7df0453c4874": ("Nike", True),
        # Owner-ruled financial 2026-10-22; re-ruled Social -1 by #234 (EXPECTED_CATEGORIES)
        "fbefe459-5ddf-4e05-b092-1bbc31b823cb": ("Puma", True),
    }
    # The cases that rule the brand's categories
    EXPECTED_CATEGORIES = {
        "cdb26552-8643-4007-a403-4f0451447e2f": ("social",),
        "fbefe459-5ddf-4e05-b092-1bbc31b823cb": ("social",),
    }

    def test_expected_verdicts_are_pinned(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "prompts/labeling/eval_cases/product_and_stability.json"
        specs = compare_prompts.parse_case_file(json.loads(path.read_text()))
        got = {
            k: (s.expected.brand, s.expected.is_sportswear_brand) for k, s in specs.items() if s.expected
        }
        assert got == self.EXPECTED

    def test_expected_categories_are_pinned(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "prompts/labeling/eval_cases/product_and_stability.json"
        specs = compare_prompts.parse_case_file(json.loads(path.read_text()))
        got = {k: tuple(s.expected.categories) for k, s in specs.items() if s.expected and s.expected.categories}
        assert got == self.EXPECTED_CATEGORIES
        # A case that rules a category cannot also sit in a stratum that expects none
        for k in got:
            assert not NO_CATEGORY_STRATA & set(specs[k].strata), k

    @pytest.mark.parametrize("version", ["v1.11.0", "v1.12.0"])
    def test_every_prompt_example_case_names_a_quote_in_the_candidate_prompt(self, version):
        """The cases kept out of accuracy are kept out by a quote the candidate prompt really has."""
        import json
        from pathlib import Path

        from src.labeling.prompt_manager import PromptManager

        root = Path(__file__).resolve().parents[1]
        specs = compare_prompts.parse_case_file(
            json.loads((root / "prompts/labeling/eval_cases/product_and_stability.json").read_text())
        )
        version = PromptManager(prompts_dir=root / "prompts" / "labeling").load_version(version)
        quotes = prompt_examples(version.system_prompt + version.user_prompt)
        flagged = {k: s for k, s in specs.items() if "prompt_example_source" in s.strata}
        assert flagged
        for article_id, spec in flagged.items():
            assert spec.prompt_example in quotes, article_id


@contextmanager
def _session():
    yield MagicMock()


class TestMain:
    def _patches(self, articles):
        return (
            patch.object(compare_prompts.db, "get_session", _session),
            patch.object(compare_prompts, "select_specs"),
            patch.object(compare_prompts, "load_articles", return_value=articles),
        )

    def test_dry_run_builds_no_labeler(self):
        labeler_cls = MagicMock(side_effect=AssertionError("no labeler in a dry run"))
        a, b, c = self._patches([_article()])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", labeler_cls):
            assert compare_prompts.main(["--candidate", "v1.11.0", "--baseline", "v1.10.0", "--dry-run"]) == 0
        labeler_cls.assert_not_called()

    def test_run_refuses_a_fallback_labeler_before_any_call(self, tmp_path):
        labeler = MagicMock(loaded_prompt=None, prompt_version="legacy")
        a, b, c = self._patches([_article()])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", return_value=labeler):
            code = compare_prompts.main(
                ["--candidate", "v1.11.0", "--baseline", "v1.10.0", "--output", str(tmp_path / "r")]
            )
        assert code == 2
        labeler.label_article.assert_not_called()

    @pytest.mark.parametrize("override,expected", [(None, "v1.10.0"), ("v1.9.0", "v1.9.0")])
    def test_default_baseline_is_the_prompt_the_pipeline_runs(self, override, expected):
        seen = []

        def stop(baseline, candidate, manager):
            seen.append(baseline)
            raise compare_prompts.EvalSetupError("stop")

        settings = MagicMock(prompt_version=override, labeling_model=HAIKU_45, labeling_effort="medium")
        manager = MagicMock()
        manager.get_production_version.return_value = "v1.10.0"
        with patch.object(compare_prompts, "labeling_settings", settings), \
             patch.object(compare_prompts, "PromptManager", return_value=manager), \
             patch.object(compare_prompts, "check_prompt_versions", side_effect=stop):
            assert compare_prompts.main(["--candidate", "v1.11.0", "--dry-run"]) == 2
        assert seen == [expected]

    def test_equal_versions_exit_before_the_database(self):
        session = MagicMock(side_effect=AssertionError("no database"))
        with patch.object(compare_prompts.db, "get_session", session), \
             patch.object(compare_prompts, "ArticleLabeler", MagicMock(side_effect=AssertionError)):
            assert compare_prompts.main(["--baseline", "v1.10.0", "--candidate", "v1.10.0"]) == 2
        session.assert_not_called()

    def test_run_over_the_cap_is_refused_before_any_labeler(self):
        labeler_cls = MagicMock(side_effect=AssertionError("refused before any labeler"))
        a, b, c = self._patches([_article()])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", labeler_cls):
            args = ["--candidate", "v1.11.0", "--baseline", "v1.10.0", "--max-cost", "0.001"]
            assert compare_prompts.main(args) == 2
            assert compare_prompts.main([*args, "--dry-run"]) == 0
        labeler_cls.assert_not_called()

    def test_only_the_candidate_falling_back_is_refused(self, tmp_path):
        good = MagicMock(loaded_prompt=object(), prompt_version="v1.10.0", model=HAIKU_45)
        fallback = MagicMock(loaded_prompt=None, prompt_version="legacy", model=HAIKU_45)
        a, b, c = self._patches([_article()])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", side_effect=[good, fallback]):
            code = compare_prompts.main(["--candidate", "v1.11.0", "--baseline", "v1.10.0"])
        assert code == 2
        good.label_article.assert_not_called()
        fallback.label_article.assert_not_called()

    def test_run_passes_the_cap_and_a_bound_over_both_versions(self, tmp_path):
        article = _article()
        captured = {}

        def fake_run(articles, baseline, candidate, *, max_cost_usd, pair_bound, on_pair=None):
            captured.update(max_cost_usd=max_cost_usd, bound=pair_bound(article))
            return [], False

        good = lambda **kw: MagicMock(  # noqa: E731
            loaded_prompt=object(), prompt_version=kw["prompt_version"], model=kw["model"]
        )
        a, b, c = self._patches([article])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", side_effect=good), \
             patch.object(compare_prompts, "run_articles", side_effect=fake_run):
            compare_prompts.main(["--candidate", "v1.11.0", "--baseline", "v1.10.0", "--max-cost", "7",
                                  "--model", HAIKU_45, "--output", str(tmp_path / "r")])
        from src.labeling.config import TARGET_SPORTSWEAR_BRANDS
        from src.labeling.prompt_manager import PromptManager

        m = PromptManager()
        chars = [
            len(m.get_formatted_system_prompt(version=v, brands=list(TARGET_SPORTSWEAR_BRANDS)))
            + len(m.load_version(v).user_prompt)
            for v in ("v1.10.0", "v1.11.0")
        ]
        expected = sum(comparator.worst_case_call_cost(article, n, SIDE) for n in chars)
        assert captured == {"max_cost_usd": 7.0, "bound": pytest.approx(expected)}

    def test_run_writes_both_reports(self, tmp_path):
        def make(model, max_tokens, prompt_version, effort):
            reply = [_esg("Nike")] if prompt_version == "v1.11.0" else []
            fake = FakeLabeler([_ok(reply)], model=model)
            fake.loaded_prompt, fake.prompt_version, fake.effort = object(), prompt_version, effort
            return fake

        a, b, c = self._patches([_article(strata=(SAMPLE,))])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", side_effect=make):
            code = compare_prompts.main(
                ["--candidate", "v1.11.0", "--baseline", "v1.10.0", "--max-cost", "5",
                 "--output", str(tmp_path / "v1.10.0_vs_v1.11.0")]
            )
        assert code == 0
        assert (tmp_path / "v1.10.0_vs_v1.11.0.json").exists()
        assert "Non-labeled -> labeled (1)" in (tmp_path / "v1.10.0_vs_v1.11.0.md").read_text()

    def test_model_contrast_holds_the_prompt_and_varies_the_model(self, tmp_path):
        built = []

        def make(model, max_tokens, prompt_version, effort):
            built.append((model, max_tokens, prompt_version, effort))
            fake = FakeLabeler([_ok([_esg("Nike")])], model=model)
            fake.loaded_prompt, fake.prompt_version, fake.effort = object(), prompt_version, effort
            return fake

        a, b, c = self._patches([_article(strata=(SAMPLE,))])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", side_effect=make):
            code = compare_prompts.main(
                ["--baseline", "v1.11.0", "--baseline-model", HAIKU_45, "--candidate-model", HAIKU_55,
                 "--effort", "low", "--max-cost", "5", "--output", str(tmp_path / "models")]
            )
        assert code == 0
        # Each side gets its model's own cap when --max-tokens is not given
        # Effort goes only to the side whose model takes it
        assert built == [(HAIKU_45, 2000, "v1.11.0", None), (HAIKU_55, 8000, "v1.11.0", "low")]
        import json

        config = json.loads((tmp_path / "models.json").read_text())["config"]
        assert config["effort"] == {"baseline": None, "candidate": "low"}
        md = (tmp_path / "models.md").read_text()
        assert f"v1.11.0 on {HAIKU_45} (baseline) vs v1.11.0 on {HAIKU_55} (effort low) (candidate)" in md
        # The sample drew no article stored under the baseline, so the floor says it is empty
        assert "Noise floor: empty" in md

    def test_effort_contrast_holds_the_prompt_and_model(self, tmp_path):
        built = []

        def make(model, max_tokens, prompt_version, effort):
            built.append(effort)
            fake = FakeLabeler([_ok([_esg("Nike")])], model=model)
            fake.loaded_prompt, fake.prompt_version, fake.effort = object(), prompt_version, effort
            return fake

        a, b, c = self._patches([_article(strata=(SAMPLE,))])
        with a, b, c, patch.object(compare_prompts, "ArticleLabeler", side_effect=make):
            code = compare_prompts.main(
                ["--baseline", "v1.11.0", "--model", HAIKU_55, "--baseline-effort", "medium",
                 "--candidate-effort", "low", "--max-cost", "5", "--output", str(tmp_path / "effort")]
            )
        assert code == 0 and built == ["medium", "low"]
