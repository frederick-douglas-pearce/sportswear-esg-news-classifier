"""Tests for the #192 backfill (D039): the cohort, how a pass ends, and the script's refusals and cap."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from scripts import backfill_relabel as script
from scripts import export_training_data as export_script
from src.data_collection import models, provenance
from src.data_collection.provenance import (
    ARM_FALSE_POSITIVE,
    ARM_JUDGED_SKIP,
    ARM_UNJUDGED,
    EXCLUDED_DETERMINISTIC,
    EXCLUDED_HAS_JUDGMENTS,
    EXCLUDED_HUMAN,
    EXCLUDED_HUMAN_LABELS,
    EXCLUDED_INSUFFICIENT_CONTENT,
    EXCLUDED_NO_BRANDS,
    EXCLUDED_POSSIBLE_HAND_MOVE,
    EXCLUDED_SKIPPED_AFTER_BOUND,
    FP_CLASSIFIER,
    HUMAN,
    HUMAN_RULINGS,
    LLM,
    NOT_A_CANDIDATE,
    PRE_CLASSIFIER,
    RULE,
    UNATTRIBUTED,
    UNJUDGED,
    BackfillDecision,
    BackfillRow,
    backfill_arm,
    build_backfill_cohort,
    derive_label_source,
)
from src.labeling.labeler import LabelerMismatchError, check_labeler, prompt_input_chars

UTC = timezone.utc
CUTOFF = datetime(2026, 10, 23, 13, 30, tzinfo=UTC)  # the production version's first run
BOUND = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)  # the first tagged run's start
BEFORE = CUTOFF - timedelta(days=30)
BETWEEN = CUTOFF + timedelta(days=1)
AFTER = BOUND + timedelta(hours=1)
LLM_FP_REASON = "Puma: other_entity: Puma here is the animal"
FP_SKIP_REASON = "Likely false positive (low risk): probability 0.120 < threshold 0.53"
RULED_FP = next(i for i, s in HUMAN_RULINGS.items() if s == "false_positive")
RULED_SKIPPED = next(i for i, s in HUMAN_RULINGS.items() if s == "skipped")
EVERY_SKIP_REASON = sorted(models.UNJUDGED_SKIP_REASONS | models.JUDGED_SKIP_REASONS)


def decide(**overrides) -> BackfillDecision:
    """backfill_arm on an arm-U row that every check passes, with overrides."""
    kwargs = dict(
        article_id=str(uuid4()),
        labeling_status="skipped",
        labeling_error=models.SKIP_NO_ESG_CONTENT,
        source=UNJUDGED,
        skipped_at=BEFORE,
        brand_count=1,
        sufficient_content=True,
        has_judgments=False,
        human_labels=False,
        judged_skip_cutoff=CUTOFF,
        bound=BOUND,
    )
    kwargs.update(overrides)
    return backfill_arm(**kwargs)


FP = dict(labeling_status="false_positive", labeling_error=LLM_FP_REASON, source=LLM, skipped_at=None)
JUDGED = dict(labeling_error=models.SKIP_NO_ESG_CATEGORIES, source=LLM, skipped_at=BETWEEN)


class TestBackfillArm:
    @pytest.mark.parametrize("reason", sorted(provenance.LLM_REACHABLE_UNJUDGED_REASONS))
    def test_unjudged_skip_at_a_reason_a_reply_writes(self, reason):
        assert decide(labeling_error=reason) == BackfillDecision(ARM_UNJUDGED)

    def test_the_reachable_reasons_are_the_unjudged_ones_a_reply_writes(self):
        assert provenance.LLM_REACHABLE_UNJUDGED_REASONS == {
            models.SKIP_NO_ESG_CONTENT, models.SKIP_NO_SPORTSWEAR_BRAND, models.SKIP_INCOMPLETE_JUDGMENT,
        }

    @pytest.mark.parametrize("reason", [models.SKIP_INSUFFICIENT_CONTENT, models.SKIP_NO_BRANDS])
    def test_unjudged_skip_a_pipeline_check_writes_is_left_out(self, reason):
        assert decide(labeling_error=reason) == BackfillDecision(None, EXCLUDED_DETERMINISTIC)

    @pytest.mark.parametrize("source", [LLM, PRE_CLASSIFIER])
    def test_llm_decided_false_positive(self, source):
        assert decide(**{**FP, "source": source}) == BackfillDecision(ARM_FALSE_POSITIVE)

    def test_a_classifier_skip_reason_on_an_llm_sourced_negative_is_not_a_hand_move(self):
        """The FP classifier writes that reason with this status, so it is in the pipeline's form."""
        assert decide(**{**FP, "labeling_error": FP_SKIP_REASON}).arm == ARM_FALSE_POSITIVE

    @pytest.mark.parametrize("source", [FP_CLASSIFIER, RULE])
    def test_classifier_only_and_keyword_sweep_negatives_are_not_candidates(self, source):
        assert decide(**{**FP, "source": source}) == NOT_A_CANDIDATE

    @pytest.mark.parametrize(
        "reason", [None, *EVERY_SKIP_REASON, f"{models.DEDUP_REASON_PREFIX} (similar to article x)"]
    )
    def test_a_negative_with_a_reason_no_llm_path_writes_is_a_possible_hand_move(self, reason):
        assert decide(**{**FP, "labeling_error": reason}) == BackfillDecision(
            None, EXCLUDED_POSSIBLE_HAND_MOVE
        )

    def test_dedup_prefix_is_the_text_the_pipeline_writes(self):
        assert models.DEDUP_REASON_PREFIX == "Duplicate title"

    @pytest.mark.parametrize("source", [LLM, PRE_CLASSIFIER])
    def test_judged_skip_from_the_cutoff(self, source):
        assert decide(**{**JUDGED, "source": source, "skipped_at": CUTOFF}).arm == ARM_JUDGED_SKIP
        assert decide(**{**JUDGED, "source": source, "skipped_at": CUTOFF - timedelta(microseconds=1)}) == (
            NOT_A_CANDIDATE
        )
        assert decide(**{**JUDGED, "source": source, "skipped_at": None}) == NOT_A_CANDIDATE

    def test_no_judged_skip_arm_without_a_cutoff(self):
        assert decide(**JUDGED, judged_skip_cutoff=None) == NOT_A_CANDIDATE

    def test_a_naive_skipped_at_is_utc(self):
        naive = CUTOFF.replace(tzinfo=None)
        assert decide(**{**JUDGED, "skipped_at": naive}).arm == ARM_JUDGED_SKIP
        assert decide(**{**JUDGED, "skipped_at": naive - timedelta(seconds=1)}) == NOT_A_CANDIDATE

    @pytest.mark.parametrize("status", ["labeled", "failed", "pending", "deduplicated"])
    def test_other_statuses_are_not_candidates(self, status):
        assert decide(labeling_status=status, source=LLM) == NOT_A_CANDIDATE

    def test_an_unattributed_skip_is_not_a_candidate(self):
        assert decide(labeling_error="moved by hand", source=UNATTRIBUTED) == NOT_A_CANDIDATE

    @pytest.mark.parametrize(
        "row",
        [
            # a ruling whose status has since changed derives as an ordinary source
            dict(article_id=RULED_FP),
            dict(article_id=RULED_SKIPPED, **FP),
            dict(article_id=RULED_FP, **FP),
            # a manual mark, or a ruling still in force, derives as human
            dict(**{**FP, "source": HUMAN}),
            dict(source=HUMAN),
        ],
    )
    def test_human_ruled_rows_are_left_out_whatever_their_status(self, row):
        assert decide(**row) == BackfillDecision(None, EXCLUDED_HUMAN)

    @pytest.mark.parametrize("arm_row", [{}, FP, JUDGED])
    def test_each_arm_needs_no_human_label_brands_content_and_no_judgment(self, arm_row):
        assert decide(**arm_row, human_labels=True).excluded == EXCLUDED_HUMAN_LABELS
        assert decide(**arm_row, brand_count=0).excluded == EXCLUDED_NO_BRANDS
        assert decide(**arm_row, sufficient_content=False).excluded == EXCLUDED_INSUFFICIENT_CONTENT
        assert decide(**arm_row, has_judgments=True).excluded == EXCLUDED_HAS_JUDGMENTS
        assert decide(**arm_row).arm is not None

    @pytest.mark.parametrize("arm_row", [{}, JUDGED])
    def test_a_skip_at_or_after_the_bound_is_not_selected(self, arm_row):
        assert decide(**{**arm_row, "skipped_at": BOUND}).excluded == EXCLUDED_SKIPPED_AFTER_BOUND
        assert decide(**{**arm_row, "skipped_at": BOUND - timedelta(microseconds=1)}).arm is not None

    def test_a_null_skipped_at_is_before_the_bound(self):
        assert decide(skipped_at=None) == BackfillDecision(ARM_UNJUDGED)

    def test_a_negative_is_not_time_bounded(self):
        """A false_positive write records no timestamp, so a stale skipped_at decides nothing."""
        assert decide(**{**FP, "skipped_at": AFTER}).arm == ARM_FALSE_POSITIVE


def row(status="skipped", reason=models.SKIP_NO_ESG_CONTENT, *, article_id=None, fp_action=None,
        fp_skip_reason=None, skipped_at=BEFORE, created_at=BEFORE, brands=1, chars=500):
    return BackfillRow(
        article_id=article_id or str(uuid4()),
        labeling_status=status,
        labeling_error=reason,
        fp_action=fp_action,
        fp_skip_reason=fp_skip_reason,
        skipped_at=skipped_at,
        created_at=created_at,
        brand_count=brands,
        content_chars=chars,
        sufficient_content=chars >= 100,
    )


def cohort_of(rows, judged=(), human_labeled=(), labeled=(), bound=BOUND):
    return build_backfill_cohort(
        sorted(rows, key=lambda r: r.article_id),
        judged_ids=set(judged),
        human_label_ids=set(human_labeled),
        label_ids=set(labeled),
        judged_skip_cutoff=CUTOFF,
        bound=bound,
    )


class TestBuildBackfillCohort:
    def test_selection_prior_and_report(self):
        u = row(chars=700, created_at=datetime(2026, 4, 2, tzinfo=UTC))
        f_llm = row("false_positive", LLM_FP_REASON, fp_action=models.FP_ACTION_CONTINUED_TO_LLM, skipped_at=None)
        f_pre = row("false_positive", LLM_FP_REASON, skipped_at=None)
        j = row(reason=models.SKIP_NO_ESG_CATEGORIES, skipped_at=BETWEEN)
        hand = row("false_positive", None)
        classifier_only = row(
            "false_positive", FP_SKIP_REASON, fp_action=models.FP_ACTION_SKIPPED_LLM, fp_skip_reason=FP_SKIP_REASON
        )
        # a ruling still in force derives as human through human_verified_ids
        ruled = row("false_positive", LLM_FP_REASON, article_id=RULED_FP)
        human_labeled = row()
        cohort = cohort_of(
            [u, f_llm, f_pre, j, hand, classifier_only, ruled, human_labeled],
            human_labeled=[human_labeled.article_id],
            labeled=[f_pre.article_id, human_labeled.article_id],
        )

        assert cohort.selected[ARM_UNJUDGED] == [u.article_id]
        assert cohort.selected[ARM_FALSE_POSITIVE] == sorted([f_llm.article_id, f_pre.article_id])
        assert cohort.selected[ARM_JUDGED_SKIP] == [j.article_id]
        assert cohort.ids == [u.article_id, *sorted([f_llm.article_id, f_pre.article_id]), j.article_id]
        assert cohort.prior[u.article_id] == ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED]
        assert cohort.prior[f_pre.article_id] == ["false_positive", LLM_FP_REASON, ARM_FALSE_POSITIVE]
        assert cohort.content_chars[u.article_id] == 700
        assert cohort.excluded == {EXCLUDED_POSSIBLE_HAND_MOVE: 1, EXCLUDED_HUMAN: 1, EXCLUDED_HUMAN_LABELS: 1}
        assert cohort.by_reason[(ARM_FALSE_POSITIVE, LLM)] == 1
        assert cohort.by_reason[(ARM_FALSE_POSITIVE, PRE_CLASSIFIER)] == 1
        assert cohort.by_reason[(ARM_UNJUDGED, models.SKIP_NO_ESG_CONTENT)] == 1
        assert cohort.by_month[(ARM_UNJUDGED, "2026-04")] == 1
        # the human-labeled article is not selected, so only f_pre counts
        assert cohort.stale_labels == 1


class TestOnePassEnds:
    """A pass ends because every outcome but a transient failure leaves the cohort (B1)."""

    def test_only_the_transient_row_is_selected_again(self):
        ids = {k: str(uuid4()) for k in ("labeled", "judged", "unjudged", "negative", "failed", "transient")}
        before = [
            row(article_id=ids["labeled"]),
            row(article_id=ids["judged"]),
            row(article_id=ids["unjudged"], skipped_at=None),
            row("false_positive", LLM_FP_REASON, article_id=ids["negative"], skipped_at=None),
            row(article_id=ids["failed"]),
            row(article_id=ids["transient"]),
        ]
        first = cohort_of(before)
        assert sorted(first.ids) == sorted(ids.values())

        # What the pass writes. An LLM-decided outcome writes brand judgments (D038), a skip sets
        # skipped_at to the time of the write, and a transient failure writes nothing (#222)
        after = [
            row("labeled", None, article_id=ids["labeled"]),
            row(reason=models.SKIP_NO_ESG_CATEGORIES, article_id=ids["judged"], skipped_at=AFTER),
            row(article_id=ids["unjudged"], skipped_at=AFTER),
            row("false_positive", "Puma: other_entity: the animal", article_id=ids["negative"], skipped_at=None),
            row("failed", "Model declined the request: cyber", article_id=ids["failed"]),
            before[-1],
        ]
        judged = [ids["labeled"], ids["judged"], ids["negative"]]
        assert cohort_of(after, judged=judged).ids == [ids["transient"]]
        # The bound is what lets the still-unjudged row go: measured from a later start, it returns
        assert ids["unjudged"] in cohort_of(after, judged=judged, bound=AFTER + timedelta(hours=1)).ids

    def test_backfill_outcomes_derive_as_their_new_source(self):
        """No FP prediction is written, so a row keeps its era's source (review Q3)."""
        assert derive_label_source("labeled", None, None) == PRE_CLASSIFIER
        assert derive_label_source("labeled", None, models.FP_ACTION_CONTINUED_TO_LLM) == LLM
        assert derive_label_source("skipped", models.SKIP_NO_ESG_CATEGORIES, None) == PRE_CLASSIFIER
        assert derive_label_source("false_positive", LLM_FP_REASON, None) == PRE_CLASSIFIER
        assert derive_label_source("false_positive", LLM_FP_REASON, models.FP_ACTION_CONTINUED_TO_LLM) == LLM
        # an older classifier skip no longer decides the row once its reason is replaced
        assert derive_label_source(
            "false_positive", LLM_FP_REASON, models.FP_ACTION_SKIPPED_LLM, FP_SKIP_REASON
        ) == LLM
        assert derive_label_source("skipped", models.SKIP_NO_ESG_CONTENT, None) == UNJUDGED


class TestExportQueryMove:
    def test_the_export_reads_provenances_query(self):
        assert export_script.fp_export_query is provenance.fp_export_query
        assert not hasattr(script, "fp_export_query")


class TestSharedLabelerHelpers:
    LABELER = dict(loaded_prompt=object(), prompt_version="v1.12.0", model="claude-haiku-5-5",
                   effort="medium", max_tokens=8000)

    def test_max_tokens_is_checked_when_given(self):
        labeler = MagicMock(**{**self.LABELER, "max_tokens": 4000})
        with pytest.raises(LabelerMismatchError, match="max_tokens"):
            check_labeler(labeler, "v1.12.0", model="claude-haiku-5-5", effort="medium", max_tokens=8000)
        check_labeler(labeler, "v1.12.0", model="claude-haiku-5-5", effort="medium")

    def test_prompt_input_chars_counts_system_user_and_schema(self):
        import json

        from src.labeling.config import TARGET_SPORTSWEAR_BRANDS
        from src.labeling.prompt_manager import PromptManager

        manager = PromptManager()
        for version in ("v1.12.0", "v1.10.0"):
            loaded = manager.load_version(version)
            expected = (
                len(manager.get_formatted_system_prompt(version=version, brands=list(TARGET_SPORTSWEAR_BRANDS)))
                + len(loaded.user_prompt)
                + (len(json.dumps(loaded.reply_schema)) if loaded.reply_schema else 0)
            )
            assert prompt_input_chars(manager, version) == expected
        assert manager.load_version("v1.12.0").reply_schema  # the schema term is exercised

    def test_the_comparator_bound_is_the_shared_one(self):
        from src.evaluation import comparator
        from src.evaluation.comparator import EvalArticle, SideSettings
        from src.labeling.config import worst_case_call_cost_usd

        article = EvalArticle(article_id="a", title="t", content="x" * 333, brands=["Nike"])
        side = SideSettings(model="claude-haiku-5-5", max_tokens=8000)
        with patch.object(comparator, "worst_case_call_cost_usd", wraps=worst_case_call_cost_usd) as shared:
            figure = comparator.worst_case_call_cost(article, 4000, side)
        shared.assert_called_once_with(4000, 333, 8000, "claude-haiku-5-5")
        assert figure == worst_case_call_cost_usd(4000, 333, 8000, "claude-haiku-5-5")


# --- the script ---------------------------------------------------------------------------

RECS = {"model": "claude-haiku-5-5", "max_tokens": 8000, "effort": "medium"}
GOOD_LABELER = dict(loaded_prompt=object(), prompt_version="v1.12.0", model="claude-haiku-5-5",
                    effort="medium", max_tokens=8000)


def tagged_run(cost=None, worst=None, completed=True):
    return SimpleNamespace(
        id=uuid4(),
        started_at=BOUND,
        completed_at=AFTER if completed else None,
        estimated_cost_usd=cost,
        config={"run_tag": script.RUN_TAG, "run_config": {"batch_worst_case_usd": worst}},
    )


@pytest.fixture
def env():
    """main() with no database, no API and no real labeler; each piece is a mock to assert on."""
    ids = sorted(str(uuid4()) for _ in range(10))
    cohort = provenance.BackfillCohort(judged_skip_cutoff=CUTOFF, bound=BOUND)
    cohort.selected[ARM_UNJUDGED] = ids
    for i in ids:
        cohort.prior[i] = ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED]
        cohort.content_chars[i] = 500
    manager = MagicMock()
    manager.get_production_version.return_value = "v1.12.0"
    manager.get_version_info.return_value = {"model_recommendations": dict(RECS)}
    labeler = MagicMock(**GOOD_LABELER)
    m = SimpleNamespace(ids=ids, cohort=cohort, manager=manager, labeler=labeler)
    with patch.object(script, "db") as m.db, \
         patch.object(script, "PromptManager", return_value=manager), \
         patch.object(script, "labeling_settings", SimpleNamespace(prompt_version=None)), \
         patch.object(script, "build_labeler", return_value=labeler) as m.build_labeler, \
         patch.object(script, "prompt_input_chars", return_value=3000) as m.prompt_input_chars, \
         patch.object(script, "worst_case_call_cost_usd", return_value=0.1), \
         patch.object(script, "first_run_started_at", return_value=CUTOFF), \
         patch.object(script, "tagged_runs", return_value=[]) as m.tagged_runs, \
         patch.object(script, "observed_cost_per_call", return_value=0.002) as m.observed, \
         patch.object(script.provenance, "select_backfill_cohort", return_value=cohort) as m.select, \
         patch.object(script, "backfill_report", return_value={
             "runs": 0, "transitions": {}, "content_types": {}, "failed_by_arm": {}, "still_unjudged": []
         }), \
         patch.object(script, "LabelingPipeline") as m.pipeline_cls:
        m.label_articles = m.pipeline_cls.return_value.label_articles
        yield m


class TestScriptRefusals:
    """Every refusal comes before any labeler call."""

    def assert_refused(self, env, argv=("--max-cost", "100")):
        assert script.main(list(argv)) == script.EXIT_REFUSED
        env.pipeline_cls.assert_not_called()
        env.labeler.label_article.assert_not_called()

    @pytest.mark.parametrize(
        "labeler",
        [
            {"loaded_prompt": None, "prompt_version": "legacy"},  # fell back to the hardcoded prompt
            {"prompt_version": "v1.10.0"},
            {"model": "claude-haiku-4-5-20251001"},
            {"effort": "low"},
            {"max_tokens": 4000},
        ],
    )
    def test_a_labeler_unlike_the_registry_entry(self, env, labeler):
        for name, value in labeler.items():
            setattr(env.labeler, name, value)
        self.assert_refused(env)

    def test_a_configured_version_that_is_not_production(self, env):
        with patch.object(script, "labeling_settings", SimpleNamespace(prompt_version="v1.11.0")):
            self.assert_refused(env)
        env.build_labeler.assert_not_called()

    def test_a_registry_entry_without_recommendations(self, env):
        env.manager.get_version_info.return_value = {}
        self.assert_refused(env)

    def test_the_labeler_is_built_for_production_and_checked_against_the_registry(self, env):
        assert script.main(["--max-cost", "100"]) == script.EXIT_DONE
        env.build_labeler.assert_called_once_with("v1.12.0")
        env.prompt_input_chars.assert_called_once_with(env.manager, "v1.12.0")

    def test_a_run_needs_max_cost(self, env):
        self.assert_refused(env, argv=())

    def test_the_first_batch_over_the_cap(self, env):
        # the first batch is all 10 articles, at 0.1 each
        self.assert_refused(env, argv=("--max-cost", "0.99"))

    def test_spend_so_far_counts_toward_the_first_batch(self, env):
        env.tagged_runs.return_value = [tagged_run(cost=1.0)]
        self.assert_refused(env, argv=("--max-cost", "1.99"))
        assert script.main(["--max-cost", "2.01"]) == script.EXIT_DONE

    def test_an_observed_cost_over_the_bound(self, env):
        env.observed.return_value = 0.11
        self.assert_refused(env, argv=("--dry-run",))


class TestScriptRun:
    def test_dry_run_makes_no_call_and_no_write(self, env):
        assert script.main(["--dry-run"]) == script.EXIT_DONE
        env.pipeline_cls.assert_not_called()
        env.label_articles.assert_not_called()
        session = env.db.get_session.return_value.__enter__.return_value
        session.add.assert_not_called()
        session.commit.assert_not_called()

    def test_each_batch_is_labeled_through_the_bypassed_pipeline_with_its_prior_values(self, env):
        assert script.main(["--max-cost", "100", "--batch-size", "4"]) == script.EXIT_DONE
        env.pipeline_cls.assert_called_once_with(labeler=env.labeler)
        calls = env.label_articles.call_args_list
        assert [len(c.kwargs["article_ids"]) for c in calls] == [4, 4, 2]
        for c, batch in zip(calls, [env.ids[0:4], env.ids[4:8], env.ids[8:]]):
            assert c.kwargs["article_ids"] == [UUID(i) for i in batch]
            assert (c.kwargs["prefilter"], c.kwargs["dedupe_titles"]) == (False, False)
            assert c.kwargs["run_tag"] == "192-backfill"
            run_config = c.kwargs["run_config"]
            assert run_config["prior"] == {i: ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED] for i in batch}
            assert run_config["batch_worst_case_usd"] == pytest.approx(0.1 * len(batch))
            assert run_config["bound"] == BOUND.isoformat()
            assert run_config["judged_skip_cutoff"] == CUTOFF.isoformat()

    def test_limit(self, env):
        script.main(["--max-cost", "100", "--limit", "3"])
        assert [c.kwargs["article_ids"] for c in env.label_articles.call_args_list] == [
            [UUID(i) for i in env.ids[:3]]
        ]

    def test_the_cap_stops_between_batches_counting_an_unfinished_run_at_its_worst_case(self, env):
        runs = []
        env.tagged_runs.side_effect = lambda *a: list(runs)

        def label(**kwargs):
            # the batch's run is cut off: no completed_at, so no recorded cost
            runs.append(tagged_run(worst=kwargs["run_config"]["batch_worst_case_usd"], completed=False))

        env.label_articles.side_effect = label
        # batches of 5 at 0.1 each: the second would bring the worst case to 1.0
        assert script.main(["--max-cost", "0.9", "--batch-size", "5"]) == script.EXIT_STOPPED_AT_CAP
        assert env.label_articles.call_count == 1

    def test_finished_runs_count_at_their_recorded_cost(self, env):
        runs = []
        env.tagged_runs.side_effect = lambda *a: list(runs)
        env.label_articles.side_effect = lambda **kw: runs.append(tagged_run(cost=0.05))
        assert script.main(["--max-cost", "0.65", "--batch-size", "5"]) == script.EXIT_DONE
        assert env.label_articles.call_count == 2

    def test_the_bound_is_the_first_tagged_runs_start(self, env):
        first = tagged_run(cost=0.0)
        env.tagged_runs.return_value = [first, tagged_run(cost=0.0)]
        script.main(["--dry-run"])
        assert env.select.call_args.kwargs == {"judged_skip_cutoff": CUTOFF, "bound": first.started_at}

    def test_without_a_tagged_run_the_bound_is_this_invocations_start(self, env):
        before = datetime.now(UTC)
        script.main(["--dry-run"])
        bound = env.select.call_args.kwargs["bound"]
        assert before <= bound <= datetime.now(UTC)


class TestSpendOf:
    def test_an_unfinished_run_without_a_recorded_worst_case_is_refused(self):
        with pytest.raises(script.BackfillRefused, match="unfinished"):
            script.spend_of([tagged_run(completed=False)])

    def test_sum(self):
        spend = script.spend_of([tagged_run(cost=0.25), tagged_run(worst=0.5, completed=False), tagged_run()])
        assert spend.usd == pytest.approx(0.75)
        assert [w for _, w in spend.unfinished] == [0.5]


# --- against PostgreSQL -----------------------------------------------------------------------

requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@requires_postgres
class TestAgainstPostgres:
    @pytest.fixture
    def database(self):
        from src.data_collection.database import Database

        # Never DATABASE_URL: it may point at production
        database = Database(
            database_url=os.environ.get(
                "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news_test"
            )
        )
        database.init_db()
        created = SimpleNamespace(articles=[], runs=[])
        yield database, created
        with database.get_session() as session:
            session.query(models.BrandJudgment).filter(
                models.BrandJudgment.labeling_run_id.in_(created.runs)
            ).delete(synchronize_session=False)
            session.query(models.Article).filter(models.Article.id.in_(created.articles)).delete(
                synchronize_session=False
            )
            session.query(models.LabelingRun).filter(models.LabelingRun.id.in_(created.runs)).delete(
                synchronize_session=False
            )

    def _article(self, session, created, status, reason, *, skipped_at=None, scrape="success",
                 brands=("Puma",), content="content " * 30):
        article = models.Article(
            article_id=f"backfill-{uuid4()}", url=f"https://example.com/{uuid4()}", title=f"backfill test {uuid4()}",
            full_content=content, brands_mentioned=list(brands), scrape_status=scrape,
            labeling_status=status, labeling_error=reason, skipped_at=skipped_at,
        )
        session.add(article)
        session.flush()
        created.articles.append(article.id)
        return article

    def _run(self, session, created, **fields):
        run = models.LabelingRun(**fields)
        session.add(run)
        session.flush()
        created.runs.append(run.id)
        return run

    def test_the_real_cohort_query(self, database):
        db, created = database
        with db.get_session() as session:
            u = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=BEFORE)
            f = self._article(session, created, "false_positive", LLM_FP_REASON)
            session.add(models.ClassifierPrediction(
                article_id=f.id, classifier_type="fp", probability=0.9, prediction=True, threshold_used=0.5,
                action_taken=models.FP_ACTION_CONTINUED_TO_LLM,
            ))
            hand = self._article(session, created, "false_positive", models.SKIP_NO_ESG_CONTENT)
            j = self._article(session, created, "skipped", models.SKIP_NO_ESG_CATEGORIES, skipped_at=BETWEEN)
            judged = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=BEFORE)
            run = self._run(session, created, prompt_version="v1.12.0")
            session.add(models.BrandJudgment(
                article_id=judged.id, labeling_run_id=run.id, brand="Puma", content_type="mention_only",
                reasoning="r", gated_categories=[], prompt_version="v1.12.0", model="claude-haiku-5-5",
            ))
            reviewed = self._article(session, created, "false_positive", LLM_FP_REASON)
            session.add(models.BrandLabel(article_id=reviewed.id, brand="Puma", labeled_by="claude-sonnet",
                                          human_reviewed=True))
            stale = self._article(session, created, "false_positive", LLM_FP_REASON)
            session.add(models.BrandLabel(article_id=stale.id, brand="Puma", labeled_by="claude-sonnet"))
            short = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, content="short")
            late = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=AFTER)
            unscraped = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, scrape="failed")
            session.commit()

            cohort = provenance.select_backfill_cohort(session, judged_skip_cutoff=CUTOFF, bound=BOUND)
            ids = {name: str(a.id) for name, a in {
                "u": u, "f": f, "j": j, "stale": stale, "hand": hand, "judged": judged,
                "reviewed": reviewed, "short": short, "late": late, "unscraped": unscraped,
            }.items()}

        ours = {str(i) for i in created.articles}
        selected = {arm: [i for i in arm_ids if i in ours] for arm, arm_ids in cohort.selected.items()}
        assert selected == {
            ARM_UNJUDGED: [ids["u"]],
            ARM_FALSE_POSITIVE: sorted([ids["f"], ids["stale"]]),
            ARM_JUDGED_SKIP: [ids["j"]],
        }
        for name in ("hand", "judged", "reviewed", "short", "late", "unscraped"):
            assert ids[name] not in cohort.prior
        assert cohort.content_chars[ids["u"]] == len("content " * 30)
        assert cohort.excluded[EXCLUDED_POSSIBLE_HAND_MOVE] >= 1
        assert cohort.excluded[provenance.EXCLUDED_NOT_SCRAPED] >= 1
        assert cohort.stale_labels >= 1

    def test_tagged_run_queries_and_the_report(self, database):
        db, created = database
        with db.get_session() as session:
            settings = script.RunSettings("v1.12.0", "claude-haiku-5-5", "medium", 8000)
            untagged = self._run(
                session, created, prompt_version="v1.12.0", started_at=CUTOFF, completed_at=CUTOFF,
                llm_calls_made=4, estimated_cost_usd=0.008, config={"model": "claude-haiku-5-5"},
            )
            u = self._article(session, created, "skipped", models.SKIP_NO_ESG_CATEGORIES, skipped_at=AFTER)
            f = self._article(session, created, "failed", "Model declined the request: x")
            tagged = self._run(
                session, created, prompt_version="v1.12.0", started_at=BOUND, completed_at=AFTER,
                llm_calls_made=2, estimated_cost_usd=0.5,
                config={"model": "claude-haiku-5-5", "run_tag": script.RUN_TAG, "run_config": {
                    "batch_worst_case_usd": 1.0,
                    "prior": {str(u.id): ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED],
                              str(f.id): ["false_positive", LLM_FP_REASON, ARM_FALSE_POSITIVE]},
                }},
            )
            unfinished = self._run(
                session, created, prompt_version="v1.12.0", started_at=AFTER,
                config={"run_tag": script.RUN_TAG, "run_config": {"batch_worst_case_usd": 0.75}},
            )
            session.add(models.BrandJudgment(
                article_id=u.id, labeling_run_id=tagged.id, brand="Puma", content_type="product",
                reasoning="r", gated_categories=[], prompt_version="v1.12.0", model="claude-haiku-5-5",
            ))
            session.commit()

            runs = [r for r in script.tagged_runs(session) if r.id in created.runs]
            assert [r.id for r in runs] == [tagged.id, unfinished.id]
            assert script.spend_of(runs).usd == pytest.approx(1.25)
            # untagged runs only; tagged ones from other tests could not be told apart, so the
            # first-run query is checked against our run's start
            assert script.first_run_started_at(session, "v1.12.0") <= untagged.started_at
            observed = script.observed_cost_per_call(session, settings)
            assert observed is not None

            with patch.object(script, "tagged_runs", return_value=runs):
                report = script.backfill_report(session)
        assert report["transitions"] == {
            (ARM_UNJUDGED, "skipped", "skipped", PRE_CLASSIFIER): 1,
            (ARM_FALSE_POSITIVE, "false_positive", "failed", None): 1,
        }
        assert report["failed_by_arm"] == {ARM_FALSE_POSITIVE: 1}
        assert report["content_types"] == {"product": 1}
        assert report["still_unjudged"] == []
