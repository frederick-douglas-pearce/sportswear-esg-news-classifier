"""Tests for the labeling script's exit-code contract (issue #81).

Labeling consumes the rows it selects, so the exit code has to distinguish
"nothing worked, try again" from "some of it worked, do not run me again".
Getting this wrong is what let a retry overwrite a real run's results with zeros
on 2026-09-05.
"""

from scripts.label_articles import exit_code_for
from src.labeling.exit_codes import (
    EXIT_FAILURE,
    EXIT_PARTIAL_FAILURE,
    EXIT_SUCCESS,
    NON_RETRYABLE_EXIT_CODES,
)
from src.labeling.pipeline import LabelingStats


class TestExitCodeFor:
    """Tests for exit_code_for."""

    def test_clean_run_succeeds(self):
        stats = LabelingStats(articles_processed=13, articles_labeled=5, articles_skipped=8)
        assert exit_code_for(stats) == EXIT_SUCCESS

    def test_every_article_raised_is_retryable(self):
        """Raised articles never got a status, so they are still pending."""
        stats = LabelingStats(
            articles_processed=0,
            articles_failed=3,
            articles_left_pending=3,
            errors=["connection refused"] * 3,
        )
        assert exit_code_for(stats) == EXIT_FAILURE

    def test_every_article_failed_terminally_is_not_retryable(self):
        """All responses unparseable (issue #82): spent, not retryable.

        `articles_processed` counts these — they were marked `failed` and will
        never be re-queued — so subtracting `articles_failed` from it would
        wrongly conclude the batch is untouched and burn three retries on an
        empty queue, whose clean exit then reports success.
        """
        stats = LabelingStats(
            articles_processed=3,
            articles_failed=3,
            articles_left_pending=0,
            errors=["Failed to parse LLM response"] * 3,
        )
        assert exit_code_for(stats) == EXIT_PARTIAL_FAILURE

    def test_mixed_success_and_raised_stays_retryable(self):
        """Half consumed, half still pending — the retry has real work to do."""
        stats = LabelingStats(
            articles_processed=6,
            articles_labeled=6,
            articles_failed=6,
            articles_left_pending=6,
            errors=["connection refused"] * 6,
        )
        assert exit_code_for(stats) == EXIT_FAILURE

    def test_empty_batch_with_an_error_is_retryable(self):
        stats = LabelingStats(articles_processed=0, errors=["database unavailable"])
        assert exit_code_for(stats) == EXIT_FAILURE

    def test_20260905_shape_is_partial(self):
        """The real incident: 12 processed, 1 failed, 1 deduplicated."""
        stats = LabelingStats(
            articles_processed=12,
            articles_labeled=5,
            articles_skipped=6,
            articles_failed=1,
            articles_deduplicated=1,
            errors=["Article e6ae677f [unknown]: Failed to parse LLM response"],
        )
        assert exit_code_for(stats) == EXIT_PARTIAL_FAILURE

    def test_one_success_is_enough_to_be_partial(self):
        stats = LabelingStats(
            articles_processed=2, articles_failed=1, errors=["parse failure"]
        )
        assert exit_code_for(stats) == EXIT_PARTIAL_FAILURE

    def test_deduplication_alone_counts_as_work_done(self):
        """Deduplicated articles left `pending` too, so the batch is still spent."""
        stats = LabelingStats(
            articles_processed=1,
            articles_failed=1,
            articles_deduplicated=2,
            errors=["parse failure"],
        )
        assert exit_code_for(stats) == EXIT_PARTIAL_FAILURE


class TestExitCodeContract:
    """The codes the runner keys off must stay distinct and correctly classified."""

    def test_codes_are_distinct(self):
        assert len({EXIT_SUCCESS, EXIT_FAILURE, EXIT_PARTIAL_FAILURE}) == 3

    def test_only_partial_is_non_retryable(self):
        assert NON_RETRYABLE_EXIT_CODES == frozenset({EXIT_PARTIAL_FAILURE})

    def test_success_is_zero(self):
        """Shell and cron treat 0 as success; nothing else may claim it."""
        assert EXIT_SUCCESS == 0
        assert EXIT_FAILURE != 0
        assert EXIT_PARTIAL_FAILURE != 0
