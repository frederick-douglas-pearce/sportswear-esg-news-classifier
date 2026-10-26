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
        brands=["Puma"],
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
    def test_unjudged_skip_at_a_reason_a_reply_writes_or_wrote(self, reason):
        assert decide(labeling_error=reason) == BackfillDecision(ARM_UNJUDGED)

    def test_the_reachable_reasons_are_the_unjudged_ones_a_reply_writes_or_wrote(self):
        assert provenance.LLM_REACHABLE_UNJUDGED_REASONS == {
            models.SKIP_NO_ESG_CONTENT, models.SKIP_NO_SPORTSWEAR_BRAND, models.SKIP_INCOMPLETE_JUDGMENT,
        }

    @pytest.mark.parametrize("reason", [models.SKIP_INSUFFICIENT_CONTENT, models.SKIP_NO_BRANDS])
    def test_unjudged_skip_a_pipeline_check_writes_is_left_out(self, reason):
        assert decide(labeling_error=reason) == BackfillDecision(None, EXCLUDED_DETERMINISTIC)

    @pytest.mark.parametrize("source", [LLM, PRE_CLASSIFIER])
    def test_llm_decided_false_positive(self, source):
        assert decide(**{**FP, "source": source}) == BackfillDecision(ARM_FALSE_POSITIVE)

    def test_a_classifier_skip_reason_on_an_llm_sourced_negative_is_a_possible_hand_move(self):
        """Not a reply's form, so fail-closed leaves it out (D039 point 3)."""
        assert decide(**{**FP, "labeling_error": FP_SKIP_REASON}) == BackfillDecision(
            None, EXCLUDED_POSSIBLE_HAND_MOVE
        )

    @pytest.mark.parametrize("source", [FP_CLASSIFIER, RULE])
    def test_classifier_only_and_keyword_sweep_negatives_are_not_candidates(self, source):
        assert decide(**{**FP, "source": source}) == NOT_A_CANDIDATE

    @pytest.mark.parametrize(
        "reason",
        [
            None,
            "",
            *EVERY_SKIP_REASON,
            f"{models.DEDUP_REASON_PREFIX} (similar to article x)",
            "Fixed: Anta brand was a Santa substring",
            "Asics: other_entity: not this article's brand",
            "Puma - the animal",
            "Puma",  # the first part is a sent brand, with no ": " after it
            "Puma; Nike: the swoosh",
        ],
    )
    def test_a_negative_whose_reason_is_not_a_replys_form_is_a_possible_hand_move(self, reason):
        assert decide(**{**FP, "labeling_error": reason}) == BackfillDecision(
            None, EXCLUDED_POSSIBLE_HAND_MOVE
        )

    def test_a_move_from_failed_keeps_the_labelers_error_and_is_a_possible_hand_move(self):
        """Each reason the pipeline writes with `failed`, from the code that writes it."""
        from anthropic import AuthenticationError, BadRequestError
        from httpx import Request, Response

        from src.labeling.labeler import ArticleLabeler, classify_api_error

        request = Request("POST", "https://api.anthropic.com/v1/messages")
        api_errors = [
            BadRequestError("credit balance is too low", response=Response(400, request=request), body=None),
            AuthenticationError("invalid x-api-key", response=Response(401, request=request), body=None),
        ]
        labeler = ArticleLabeler(api_key="test", prompt_version="v1.12.0")
        replies = [
            SimpleNamespace(text="", input_tokens=1, output_tokens=1, stop_reason="refusal", refusal_category="cyber"),
            SimpleNamespace(text="not json", input_tokens=1, output_tokens=1, stop_reason="max_tokens", refusal_category=None),
            SimpleNamespace(text="not json", input_tokens=1, output_tokens=1, stop_reason="end_turn", refusal_category=None),
        ]
        errors = [classify_api_error(e)[1] for e in api_errors]
        for reply in replies:
            with patch.object(labeler, "_call_api_with_retry", return_value=reply):
                errors.append(labeler.label_article("Puma", "content " * 30, ["Puma"]).error)
        assert errors[2].startswith(models.FAILED_REFUSAL_PREFIX)
        assert errors[3].startswith(models.FAILED_MAX_TOKENS_PREFIX)
        assert errors[4] == "Failed to parse LLM response"
        for error in errors:
            assert decide(**{**FP, "labeling_error": error}).excluded == EXCLUDED_POSSIBLE_HAND_MOVE, error

    @pytest.mark.parametrize(
        ("reason", "brands"),
        [
            ("Puma: x; Nike: y", ["Nike", "Puma"]),
            ("puma : x", ["Puma"]),  # matched as normalize_brand matches
            (" PUMA: x", ["Puma"]),
        ],
    )
    def test_a_replys_form_for_a_sent_brand_is_selected(self, reason, brands):
        assert decide(**{**FP, "labeling_error": reason, "brands": brands}).arm == ARM_FALSE_POSITIVE

    def test_the_reason_dispose_reply_writes_is_selected(self):
        from src.labeling.disposition import dispose_reply
        from src.labeling.models import BrandAnalysis

        brands = ["Puma", "Vans"]
        disposition = dispose_reply(brands, [
            BrandAnalysis(brand=b, is_sportswear_brand=False, not_sportswear_reason="another meaning")
            for b in brands
        ])
        assert disposition.status == "false_positive"
        assert decide(**{**FP, "labeling_error": disposition.reason, "brands": brands}).arm == ARM_FALSE_POSITIVE

    def test_the_brand_key_is_normalize_brands(self):
        from src.labeling.models import normalize_brand

        for name in ("Puma", " PUMA ", "Under Armour", "ASICS"):
            assert provenance._brand_key(name) == normalize_brand(name)

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
        assert decide(**arm_row, brands=[]).excluded == EXCLUDED_NO_BRANDS
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
        fp_skip_reason=None, skipped_at=BEFORE, created_at=BEFORE, brands=("Puma",), chars=500):
    return BackfillRow(
        article_id=article_id or str(uuid4()),
        labeling_status=status,
        labeling_error=reason,
        fp_action=fp_action,
        fp_skip_reason=fp_skip_reason,
        skipped_at=skipped_at,
        created_at=created_at,
        brands=list(brands),
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
        assert cohort.left_out == {
            hand.article_id: EXCLUDED_POSSIBLE_HAND_MOVE,
            ruled.article_id: EXCLUDED_HUMAN,
            human_labeled.article_id: EXCLUDED_HUMAN_LABELS,
        }
        assert cohort.hand_move_reasons == {"": 1}
        assert cohort.by_reason[(ARM_FALSE_POSITIVE, LLM)] == 1
        assert cohort.by_reason[(ARM_FALSE_POSITIVE, PRE_CLASSIFIER)] == 1
        assert cohort.by_reason[(ARM_UNJUDGED, models.SKIP_NO_ESG_CONTENT)] == 1
        assert cohort.by_month[(ARM_UNJUDGED, "2026-04")] == 1
        # the human-labeled article is not selected, so only f_pre counts
        assert cohort.stale_labels == 1

    def test_hand_move_reasons_are_tallied_by_their_start(self):
        long = "Fixed: " + "x" * 80
        cohort = cohort_of([row("false_positive", long, skipped_at=None) for _ in range(2)])
        assert cohort.hand_move_reasons == {long[: provenance.HAND_MOVE_REASON_CHARS]: 2}


class TestRecheckBatch:
    """Each batch is decided again before it is labeled (D039 point 5)."""

    def test_a_row_changed_since_selection_is_dropped(self):
        rows = [row() for _ in range(4)]
        cohort = cohort_of(rows)
        moved, reviewed, gone, same = (r.article_id for r in sorted(rows, key=lambda r: r.article_id))
        fresh = {r.article_id: r for r in rows}
        # a human moved one by hand, reviewed another's labels, and a third left the cohort's rows
        fresh[moved] = row("false_positive", "Fixed: not Puma", article_id=moved)
        del fresh[gone]
        fresh_cohort = cohort_of(list(fresh.values()), human_labeled=[reviewed])
        with patch.object(provenance, "_cohort_from_db", return_value=fresh_cohort) as read:
            kept, dropped = provenance.recheck_backfill_batch(MagicMock(), cohort, cohort.ids)
        assert read.call_args.kwargs == {
            "judged_skip_cutoff": CUTOFF, "bound": BOUND, "ids": cohort.ids,
        }
        assert kept == [same]
        assert dropped == {moved: EXCLUDED_POSSIBLE_HAND_MOVE, reviewed: EXCLUDED_HUMAN_LABELS, gone: "changed"}

    def test_a_row_whose_reason_changed_is_dropped(self):
        original = row()
        cohort = cohort_of([original])
        changed = row(reason=models.SKIP_INCOMPLETE_JUDGMENT, article_id=original.article_id)
        with patch.object(provenance, "_cohort_from_db", return_value=cohort_of([changed])):
            assert provenance.recheck_backfill_batch(MagicMock(), cohort, cohort.ids) == (
                [], {original.article_id: "changed"}
            )


class TestOnePassEnds:
    """A pass ends because every outcome but a transient failure, a call that failed with no
    reply, or a raised exception leaves the cohort."""

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
        """No FP prediction is written, so a row keeps its era's source (D039 point 4)."""
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


    def test_a_skip_write_moves_skipped_at_to_the_write(self):
        """The bound ends a pass only because a skip written again refreshes skipped_at."""
        from src.labeling.database import LabelingDatabase

        stored = SimpleNamespace(
            labeling_status="skipped", labeling_error=models.SKIP_NO_ESG_CONTENT, skipped_at=BEFORE, labeled_at=None
        )
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = stored
        before = datetime.now(UTC)
        LabelingDatabase(MagicMock()).update_article_labeling_status(
            session, uuid4(), "skipped", models.SKIP_NO_ESG_CONTENT
        )
        assert before <= stored.skipped_at <= datetime.now(UTC)


class TestExportQueryMove:
    def test_the_export_reads_provenances_query(self):
        assert export_script.fp_export_query is provenance.fp_export_query
        assert not hasattr(script, "fp_export_query")

    @staticmethod
    def _before_the_move(session, since=None):
        """fp_export_query's body as it was in scripts/export_training_data.py."""
        from sqlalchemy import and_

        from src.data_collection.models import Article

        base_filter = Article.scrape_status == "success"
        if since:
            base_filter = and_(base_filter, Article.created_at >= since)
        latest = provenance.latest_prediction_subquery("fp")
        return (
            session.query(Article, latest.c.action_taken, latest.c.skip_reason, latest.c.created_at)
            .outerjoin(latest, latest.c.article_id == Article.id)
            .filter(base_filter)
            .filter(Article.labeling_status.in_([*provenance.FP_POSITIVE_STATUSES, provenance.FP_NEGATIVE_STATUS]))
            .order_by(Article.id)
        )

    @pytest.mark.parametrize("since", [None, BEFORE])
    def test_the_moved_query_compiles_to_the_same_sql(self, since):
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.orm import Session

        def sql(query):
            return str(query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))

        session = Session()
        moved = sql(provenance.fp_export_query(session, since))
        assert moved == sql(self._before_the_move(session, since))
        assert "ORDER BY articles.id" in moved


class TestSharedLabelerHelpers:
    LABELER = dict(loaded_prompt=object(), prompt_version="v1.12.0", model="claude-haiku-5-5",
                   effort="medium", max_tokens=8000)

    def test_a_labeler_with_no_loaded_prompt_is_refused_whatever_version_it_reports(self):
        labeler = MagicMock(**{**self.LABELER, "loaded_prompt": None})
        with pytest.raises(LabelerMismatchError, match="v1.12.0"):
            check_labeler(labeler, "v1.12.0", model="claude-haiku-5-5", effort="medium")

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
BUILD_LABELER = script.build_labeler  # the real one, before the env fixture patches it
GOOD_LABELER = dict(loaded_prompt=object(), prompt_version="v1.12.0", model="claude-haiku-5-5",
                    effort="medium", max_tokens=8000)


def tagged_run(cost=None, worst=None, completed=True, started_at=BOUND, bound=BOUND):
    return SimpleNamespace(
        id=uuid4(),
        started_at=started_at,
        completed_at=AFTER if completed else None,
        estimated_cost_usd=cost,
        config={"run_tag": script.RUN_TAG, "run_config": {
            "batch_worst_case_usd": worst, "bound": bound.isoformat() if bound else None,
        }},
    )


def stats(**counts):
    from src.labeling.pipeline import LabelingStats

    return LabelingStats(**counts)


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
         patch.object(script, "worst_case_call_cost_usd", return_value=0.1) as m.worst_case, \
         patch.object(script, "first_run_started_at", return_value=CUTOFF), \
         patch.object(script, "tagged_runs", return_value=[]) as m.tagged_runs, \
         patch.object(script, "observed_cost_per_call", return_value=0.002) as m.observed, \
         patch.object(script.provenance, "select_backfill_cohort", return_value=cohort) as m.select, \
         patch.object(script.provenance, "recheck_backfill_batch",
                      side_effect=lambda session, cohort, ids: (list(ids), {})) as m.recheck, \
         patch.object(script, "backfill_report", return_value={
             "runs": 0, "transitions": {}, "content_types": {}, "failed_by_arm": {},
             "unjudged_left_out": [], "unjudged_retried": [],
         }), \
         patch.object(script, "LabelingPipeline") as m.pipeline_cls:
        m.label_articles = m.pipeline_cls.return_value.label_articles
        m.label_articles.return_value = stats()
        yield m


class TestScriptRefusals:
    """Each refusal here comes before the pipeline is built, so nothing is sent to the API."""

    def assert_refused(self, env, argv=("--max-cost", "100")):
        assert script.main(list(argv)) == script.EXIT_REFUSED
        env.pipeline_cls.assert_not_called()

    @pytest.mark.parametrize(
        "argv",
        [("--limit", "0"), ("--limit", "-1"), ("--batch-size", "0")],
    )
    def test_a_limit_or_batch_size_below_one(self, env, argv):
        self.assert_refused(env, argv=("--max-cost", "100", *argv))
        env.select.assert_not_called()

    def test_an_unfinished_tagged_run_without_a_worst_case(self, env):
        env.tagged_runs.return_value = [tagged_run(completed=False)]
        self.assert_refused(env)

    def test_a_tagged_run_without_a_bound(self, env):
        env.tagged_runs.return_value = [tagged_run(cost=0.0, bound=None)]
        self.assert_refused(env)

    def test_a_labeler_that_cannot_be_built(self, env):
        """A missing API key raises ValueError in ArticleLabeler: a refusal, not exit 1."""
        env.build_labeler.side_effect = BUILD_LABELER
        with patch.object(script, "ArticleLabeler", side_effect=ValueError("ANTHROPIC_API_KEY is required")):
            self.assert_refused(env)

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

    @pytest.mark.parametrize("missing", [None, "model", "max_tokens"])
    def test_a_registry_entry_without_recommendations(self, env, missing):
        recs = {} if missing is None else {k: v for k, v in RECS.items() if k != missing}
        env.manager.get_version_info.return_value = {"model_recommendations": recs}
        self.assert_refused(env)

    def test_the_labeler_is_built_for_the_production_version(self, env):
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
        assert script.main(["--dry-run", "--max-cost", "100"]) == script.EXIT_DONE
        env.pipeline_cls.assert_not_called()
        env.select.assert_called_once()

    def test_each_articles_worst_case_uses_its_content_and_the_registry_settings(self, env):
        env.cohort.content_chars[env.ids[0]] = 1234
        script.main(["--dry-run"])
        calls = env.worst_case.call_args_list
        assert len(calls) == len(env.ids)
        assert calls[0].args == (3000, 1234, 8000, "claude-haiku-5-5")
        assert {c.args for c in calls[1:]} == {(3000, 500, 8000, "claude-haiku-5-5")}

    def test_each_batch_is_labeled_through_the_bypassed_pipeline_with_its_prior_values(self, env):
        assert script.main(["--max-cost", "100", "--batch-size", "4"]) == script.EXIT_DONE
        env.pipeline_cls.assert_called_once_with(labeler=env.labeler)
        calls = env.label_articles.call_args_list
        assert [len(c.kwargs["article_ids"]) for c in calls] == [4, 4, 2]
        for c, batch in zip(calls, [env.ids[0:4], env.ids[4:8], env.ids[8:]]):
            assert c.kwargs["article_ids"] == [UUID(i) for i in batch]
            assert (c.kwargs["prefilter"], c.kwargs["dedupe_titles"]) == (False, False)
            assert c.kwargs["keep_status_without_reply"] is True
            assert c.kwargs["run_tag"] == "192-backfill"
            run_config = c.kwargs["run_config"]
            assert run_config["prior"] == {i: ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED] for i in batch}
            assert run_config["batch_worst_case_usd"] == pytest.approx(0.1 * len(batch))
            assert run_config["bound"] == BOUND.isoformat()
            assert run_config["judged_skip_cutoff"] == CUTOFF.isoformat()
            assert run_config["dropped"] == {}
        assert [c.args[2] for c in env.recheck.call_args_list] == [env.ids[0:4], env.ids[4:8], env.ids[8:]]

    def test_a_row_the_recheck_drops_is_not_labeled_and_is_recorded(self, env):
        dropped = {env.ids[1]: "changed"}
        env.recheck.side_effect = lambda session, cohort, ids: (
            [i for i in ids if i not in dropped], {i: w for i, w in dropped.items() if i in ids}
        )
        assert script.main(["--max-cost", "100", "--batch-size", "4"]) == script.EXIT_DONE
        first = env.label_articles.call_args_list[0].kwargs
        assert first["article_ids"] == [UUID(i) for i in (env.ids[0], env.ids[2], env.ids[3])]
        assert first["run_config"]["dropped"] == dropped
        assert env.ids[1] not in first["run_config"]["prior"]
        assert first["run_config"]["batch_worst_case_usd"] == pytest.approx(0.3)

    def test_an_empty_cohort_has_nothing_to_relabel(self, env, capsys):
        env.cohort.selected[ARM_UNJUDGED] = []
        assert script.main(["--max-cost", "100"]) == script.EXIT_DONE
        env.pipeline_cls.assert_not_called()
        assert "Nothing to relabel" in capsys.readouterr().out

    def test_a_batch_the_recheck_empties_is_not_run(self, env):
        env.recheck.side_effect = lambda session, cohort, ids: ([], {i: "changed" for i in ids})
        assert script.main(["--max-cost", "100"]) == script.EXIT_DONE
        env.label_articles.assert_not_called()

    def test_a_batch_left_by_an_error_that_is_not_transient_stops_the_invocation(self, env):
        env.label_articles.return_value = stats(
            articles_left_pending=4, articles_left_non_transient=4, errors_by_type={"api_error": 4},
            errors=["Article x [api_error]: Bad request (400): credit balance is too low"],
        )
        assert script.main(["--max-cost", "100", "--batch-size", "4"]) == script.EXIT_ABORTED
        assert env.label_articles.call_count == 1

    def test_a_transient_failure_does_not_stop_the_invocation(self, env):
        env.label_articles.return_value = stats(articles_left_pending=4, errors_by_type={"rate_limit": 4})
        assert script.main(["--max-cost", "100", "--batch-size", "4"]) == script.EXIT_DONE
        assert env.label_articles.call_count == 3

    def test_the_exit_codes_are_distinct_and_leave_1_to_an_uncaught_error(self):
        codes = [script.EXIT_DONE, script.EXIT_REFUSED, script.EXIT_STOPPED_AT_CAP, script.EXIT_ABORTED]
        assert codes == [0, 2, 3, 4]

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
            return stats()

        env.label_articles.side_effect = label
        # batches of 5 at 0.1 each: the second would bring the worst case to 1.0
        assert script.main(["--max-cost", "0.9", "--batch-size", "5"]) == script.EXIT_STOPPED_AT_CAP
        assert env.label_articles.call_count == 1

    def test_the_first_batch_over_the_cap_at_its_recheck_is_a_refusal(self, env):
        """Another invocation's spend can grow between main's check and the first batch."""
        env.tagged_runs.side_effect = [[], [tagged_run(cost=0.5)]]
        # main sees 0 + 1.0 <= 1.2; the first batch sees 0.5 + 1.0
        assert script.main(["--max-cost", "1.2"]) == script.EXIT_REFUSED
        env.label_articles.assert_not_called()

    def test_finished_runs_count_at_their_recorded_cost(self, env):
        runs = []
        env.tagged_runs.side_effect = lambda *a: list(runs)
        env.label_articles.side_effect = lambda **kw: runs.append(tagged_run(cost=0.05)) or stats()
        assert script.main(["--max-cost", "0.65", "--batch-size", "5"]) == script.EXIT_DONE
        assert env.label_articles.call_count == 2

    def test_the_bound_is_the_one_the_first_tagged_run_recorded(self, env):
        # the first run started after the bound it recorded (the cohort took time to build), and
        # a later run recorded another
        first = tagged_run(cost=0.0, started_at=BOUND + timedelta(minutes=5), bound=BOUND)
        later = tagged_run(cost=0.0, started_at=AFTER, bound=AFTER)
        env.tagged_runs.return_value = [first, later]
        script.main(["--dry-run"])
        assert env.select.call_args.kwargs == {"judged_skip_cutoff": CUTOFF, "bound": BOUND}

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
            from_failed = self._article(session, created, "false_positive", "Failed to parse LLM response")
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
                "u": u, "f": f, "j": j, "stale": stale, "hand": hand, "from_failed": from_failed,
                "judged": judged, "reviewed": reviewed, "short": short, "late": late,
                "unscraped": unscraped,
            }.items()}

        ours = {str(i) for i in created.articles}
        selected = {arm: [i for i in arm_ids if i in ours] for arm, arm_ids in cohort.selected.items()}
        assert selected == {
            ARM_UNJUDGED: [ids["u"]],
            ARM_FALSE_POSITIVE: sorted([ids["f"], ids["stale"]]),
            ARM_JUDGED_SKIP: [ids["j"]],
        }
        for name in ("hand", "from_failed", "judged", "reviewed", "short", "late", "unscraped"):
            assert ids[name] not in cohort.prior
        assert {cohort.left_out[ids[n]] for n in ("hand", "from_failed")} == {EXCLUDED_POSSIBLE_HAND_MOVE}
        assert cohort.left_out[ids["judged"]] == EXCLUDED_HAS_JUDGMENTS
        assert cohort.left_out[ids["reviewed"]] == EXCLUDED_HUMAN_LABELS
        assert cohort.left_out[ids["short"]] == EXCLUDED_INSUFFICIENT_CONTENT
        assert cohort.left_out[ids["late"]] == EXCLUDED_SKIPPED_AFTER_BOUND
        assert cohort.content_chars[ids["u"]] == len("content " * 30)
        assert cohort.excluded[provenance.EXCLUDED_NOT_SCRAPED] >= 1
        assert cohort.stale_labels >= 1

    def test_the_recheck_reads_the_batchs_rows_again(self, database):
        db, created = database
        with db.get_session() as session:
            moved, reviewed, same = (
                self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=BEFORE)
                for _ in range(3)
            )
            session.commit()
            ids = sorted(str(a.id) for a in (moved, reviewed, same))
            cohort = provenance.select_backfill_cohort(session, judged_skip_cutoff=CUTOFF, bound=BOUND)
            assert all(i in cohort.prior for i in ids)

            # during the run, a human moves one by hand and reviews another's labels
            moved.labeling_status, moved.labeling_error = "false_positive", "Fixed: not the brand"
            session.add(models.BrandLabel(article_id=reviewed.id, brand="Puma", labeled_by="human"))
            session.commit()
            kept, dropped = provenance.recheck_backfill_batch(session, cohort, ids)
            names = {str(moved.id): "moved", str(reviewed.id): "reviewed", str(same.id): "same"}

        assert [names[i] for i in kept] == ["same"]
        assert {names[i]: why for i, why in dropped.items()} == {
            "moved": EXCLUDED_POSSIBLE_HAND_MOVE, "reviewed": EXCLUDED_HUMAN_LABELS,
        }

    def test_tagged_run_queries_and_the_report(self, database):
        db, created = database
        # a version no other test's runs use, so every run under it is ours
        version = f"vt-{uuid4().hex[:12]}"
        model = "claude-haiku-5-5"
        with db.get_session() as session:
            settings = script.RunSettings(version, model, "medium", 8000)
            untagged = self._run(
                session, created, prompt_version=version, started_at=CUTOFF, completed_at=CUTOFF,
                llm_calls_made=4, estimated_cost_usd=0.008, config={"model": model},
            )
            # None of these may set the cutoff or the observed cost: a tagged run that started
            # first, an unfinished untagged run, and an untagged run on another model
            self._run(
                session, created, prompt_version=version, started_at=BEFORE, completed_at=BEFORE,
                llm_calls_made=1, estimated_cost_usd=9.0, config={"model": model, "run_tag": "another-tag"},
            )
            self._run(
                session, created, prompt_version=version, started_at=BETWEEN,
                llm_calls_made=1, estimated_cost_usd=7.0, config={"model": model},
            )
            self._run(
                session, created, prompt_version=version, started_at=BETWEEN, completed_at=BETWEEN,
                llm_calls_made=1, estimated_cost_usd=5.0, config={"model": "claude-haiku-4-5-20251001"},
            )
            u = self._article(session, created, "skipped", models.SKIP_NO_ESG_CATEGORIES, skipped_at=AFTER)
            f = self._article(session, created, "failed", "Model declined the request: x")
            left = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=AFTER)
            retried = self._article(session, created, "skipped", models.SKIP_NO_ESG_CONTENT, skipped_at=BEFORE)
            unjudged = ["skipped", models.SKIP_NO_ESG_CONTENT, ARM_UNJUDGED]
            # Inserted before the run that started first, so the order is the query's. It records
            # u again with other values (moved by hand between invocations): u's first record wins
            unfinished = self._run(
                session, created, prompt_version=version, started_at=AFTER,
                config={"run_tag": script.RUN_TAG, "run_config": {
                    "batch_worst_case_usd": 0.75, "bound": BOUND.isoformat(),
                    "prior": {str(u.id): ["false_positive", LLM_FP_REASON, ARM_FALSE_POSITIVE]},
                }},
            )
            tagged = self._run(
                session, created, prompt_version=version, started_at=BOUND, completed_at=AFTER,
                llm_calls_made=2, estimated_cost_usd=0.5,
                config={"model": model, "run_tag": script.RUN_TAG, "run_config": {
                    "batch_worst_case_usd": 1.0, "bound": BOUND.isoformat(),
                    "prior": {str(u.id): unjudged, str(left.id): unjudged, str(retried.id): unjudged,
                              str(f.id): ["false_positive", LLM_FP_REASON, ARM_FALSE_POSITIVE]},
                }},
            )
            session.add(models.BrandJudgment(
                article_id=u.id, labeling_run_id=tagged.id, brand="Puma", content_type="product",
                reasoning="r", gated_categories=[], prompt_version="v1.12.0", model="claude-haiku-5-5",
            ))
            session.commit()

            runs = [r for r in script.tagged_runs(session) if r.id in created.runs]
            assert [r.id for r in runs] == [tagged.id, unfinished.id]
            assert script.spend_of(runs).usd == pytest.approx(1.25)
            assert script.stored_bound(runs) == BOUND
            assert script.first_run_started_at(session, version) == untagged.started_at
            assert script.observed_cost_per_call(session, settings) == pytest.approx(0.008 / 4)

            with patch.object(script, "tagged_runs", return_value=runs):
                report = script.backfill_report(session)
            left_id, retried_id = str(left.id), str(retried.id)
        assert report["transitions"] == {
            (ARM_UNJUDGED, "skipped", "skipped", PRE_CLASSIFIER): 1,
            (ARM_UNJUDGED, "skipped", "skipped", UNJUDGED): 2,
            (ARM_FALSE_POSITIVE, "false_positive", "failed", None): 1,
        }
        assert report["failed_by_arm"] == {ARM_FALSE_POSITIVE: 1}
        assert report["content_types"] == {"product": 1}
        assert report["unjudged_left_out"] == [left_id]
        assert report["unjudged_retried"] == [retried_id]
