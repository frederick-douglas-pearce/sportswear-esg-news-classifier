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
)
from src.evaluation.diff_report import render_markdown
from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED
from src.labeling.labeler import LabelingResult
from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse
from src.labeling.pricing import llm_cost_usd

MAX_TOKENS = 2000


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


def _ok(analyses, output_tokens=100, input_tokens=1000):
    return LabelingResult(
        success=True,
        response=LabelingResponse(brand_analyses=analyses, article_summary="s"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _reply(analyses, brands=("Nike",), **kw):
    return record_reply(list(brands), _ok(analyses, **kw))


def _article(article_id="a1", brands=("Nike",), **kw):
    return EvalArticle(article_id=article_id, title=f"T {article_id}", content="x" * 300, brands=list(brands), **kw)


class FakeLabeler:
    """Returns queued results in order and records each call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

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
        s = summarize(runs, max_tokens=MAX_TOKENS)
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
        s = summarize(runs, max_tokens=MAX_TOKENS)
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
        s = summarize(runs, max_tokens=MAX_TOKENS)
        assert [(f["brand"], f["baseline"], f["candidate"]) for f in s["fp_flips"]] == [("puma", True, False)]
        assert s["category_flips"] == 1  # Puma's environmental label is gone
        assert s["sentiment_agreement"] == 0.0  # Nike environmental: +1 vs -1

    def test_compliance_and_duplicates_per_version(self):
        runs = [
            _run(
                _article("1", brands=("Nike", "Puma")),
                [_reply([_esg("Nike")], brands=("Nike", "Puma"))],
                [_reply([_esg("Nike"), _no_esg("Puma"), _no_esg("puma")], brands=("Nike", "Puma"))],
            )
        ]
        s = summarize(runs, max_tokens=MAX_TOKENS)
        assert s["baseline"]["compliance"] == 0.0 and s["baseline"]["incomplete_replies"] == 1
        assert s["candidate"]["compliance"] == 1.0
        assert s["candidate"]["duplicate_entries"] == 1 and s["candidate"]["replies_with_duplicates"] == 1

    def test_tokens_truncation_near_limit_and_most_brands_stratum(self):
        runs = [
            _run(_article("1", strata=(MOST_BRANDS,)), [_reply([_esg("Nike")], output_tokens=1900)],
                 [_reply([_esg("Nike")], output_tokens=MAX_TOKENS)]),
            _run(_article("2"), [_reply([_esg("Nike")], output_tokens=100)],
                 [_reply([_esg("Nike")], output_tokens=300)]),
        ]
        s = summarize(runs, max_tokens=MAX_TOKENS)
        assert s["candidate"]["tokens"]["truncated"] == 1 and s["candidate"]["tokens"]["near_limit"] == 0
        assert s["baseline"]["tokens"]["near_limit"] == 1 and s["baseline"]["tokens"]["truncated"] == 0
        assert s["candidate"]["tokens_most_brands"]["output_max"] == MAX_TOKENS
        assert s["candidate"]["tokens_most_brands"]["calls"] == 1
        assert s["candidate"]["tokens"]["cost_usd"] == pytest.approx(llm_cost_usd(2000, 2300))

    def test_stability_accuracy_and_prompt_example_exclusion(self):
        expect = Expectation("Nike", True)
        runs = [
            _run(
                _article("1", repeats=2, expected=expect),
                [_reply([_esg("Nike")]), _reply([])],
                [_reply([_no_esg("Nike")]), _reply([_not_sportswear("Nike")])],
            ),
            # its title is a prompt example, so it is reported but not scored
            _run(
                _article("2", expected=expect, prompt_example="the example"),
                [_reply([_not_sportswear("Nike")])],
                [_reply([_not_sportswear("Nike")])],
            ),
        ]
        s = summarize(runs, max_tokens=MAX_TOKENS)
        stab = s["stability"][0]
        assert stab["baseline_consistent"] is False and stab["candidate_consistent"] is False
        acc = s["product_accuracy"]["candidate"]
        assert (acc["correct"], acc["scored"]) == (1, 2)
        assert acc["misses"][0]["got"] is False
        assert [c["article_id"] for c in s["prompt_example_cases"]] == ["2"]

    def test_noise_floor_uses_only_rows_stored_under_the_baseline(self):
        runs = [
            _run(_article("1", stored_status=LABELED, stored_under_baseline=True),
                 [_reply([_esg("Nike")])], [_reply([_esg("Nike")])]),
            _run(_article("2", stored_status=LABELED, stored_under_baseline=True),
                 [_reply([])], [_reply([_esg("Nike")])]),
            _run(_article("3", stored_status=LABELED), [_reply([])], [_reply([])]),
        ]
        floor = summarize(runs, max_tokens=MAX_TOKENS)["noise_floor"]
        assert (floor["agree_with_stored"], floor["articles"]) == (1, 2)

    def test_report_renders(self):
        runs = [_run(_article("1", strata=(SAMPLE,)), [_reply([])], [_reply([_esg("Nike")])])]
        config = {
            "baseline": "v1.10.0", "candidate": "v1.11.0", "model": "m", "max_tokens": MAX_TOKENS,
            "baseline_hashes": "a/b", "candidate_hashes": "c/d", "stopped_early": False,
        }
        md = render_markdown(summarize(runs, max_tokens=MAX_TOKENS), config)
        assert "v1.10.0 (baseline) vs v1.11.0 (candidate)" in md
        assert "### Non-labeled -> labeled (1)" in md
        assert "1 of 1 baseline non-labeled replies labeled by the candidate" in md


class TestScriptGuards:
    def test_equal_versions_are_refused(self):
        with pytest.raises(compare_prompts.EvalSetupError, match="same version"):
            compare_prompts.check_prompt_versions("v1.10.0", "v1.10.0", MagicMock())

    def test_unloadable_version_is_refused(self):
        manager = MagicMock()
        manager.load_version.side_effect = ValueError("Unknown prompt version: v9")
        with pytest.raises(compare_prompts.EvalSetupError, match="does not load"):
            compare_prompts.check_prompt_versions("v1.10.0", "v9", manager)

    def test_labeler_that_fell_back_to_the_hardcoded_prompt_is_refused(self):
        labeler = MagicMock(loaded_prompt=None, prompt_version="legacy")
        with pytest.raises(compare_prompts.EvalSetupError, match="legacy"):
            compare_prompts.check_labeler(labeler, "v1.11.0")

    def test_labeler_with_another_version_is_refused(self):
        labeler = MagicMock(loaded_prompt=object(), prompt_version="v1.10.0")
        with pytest.raises(compare_prompts.EvalSetupError):
            compare_prompts.check_labeler(labeler, "v1.11.0")
        compare_prompts.check_labeler(MagicMock(loaded_prompt=object(), prompt_version="v1.11.0"), "v1.11.0")


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

    def test_run_writes_both_reports(self, tmp_path):
        def make(model, max_tokens, prompt_version):
            reply = [_esg("Nike")] if prompt_version == "v1.11.0" else []
            fake = FakeLabeler([_ok(reply)])
            fake.loaded_prompt, fake.prompt_version = object(), prompt_version
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


def test_comparator_imports_no_database_or_api_client():
    """The comparator stays pure; the script owns the database and the API client."""
    source = open(comparator.__file__).read()
    assert "database" not in source and "anthropic" not in source.lower()
