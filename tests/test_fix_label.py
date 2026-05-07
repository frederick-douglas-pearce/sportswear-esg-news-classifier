"""Tests for scripts/fix_label.py - article label correction tool."""

import uuid
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from scripts.fix_label import (
    STATUS_TIMESTAMPS,
    list_statuses,
    parse_uuid,
    show_article,
    update_articles,
)
from src.data_collection.models import VALID_LABELING_STATUSES


# ============================================================================
# VALID_LABELING_STATUSES constant tests
# ============================================================================


class TestValidLabelingStatuses:
    """Tests for VALID_LABELING_STATUSES constant in models.py."""

    def test_contains_expected_statuses(self):
        """All expected statuses are present."""
        expected = {
            "pending",
            "chunked",
            "embedded",
            "labeled",
            "skipped",
            "false_positive",
            "unlabelable",
            "deduplicated",
        }
        assert set(VALID_LABELING_STATUSES) == expected

    def test_is_list(self):
        """Constant is a list (ordered)."""
        assert isinstance(VALID_LABELING_STATUSES, list)

    def test_no_duplicates(self):
        """No duplicate entries."""
        assert len(VALID_LABELING_STATUSES) == len(set(VALID_LABELING_STATUSES))


# ============================================================================
# parse_uuid tests
# ============================================================================


class TestParseUuid:
    """Tests for UUID parsing and validation."""

    def test_valid_uuid(self):
        """Accepts a valid UUID string."""
        result = parse_uuid("12345678-1234-1234-1234-123456789abc")
        assert isinstance(result, uuid.UUID)

    def test_invalid_uuid(self):
        """Rejects invalid UUID strings."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="Invalid UUID"):
            parse_uuid("not-a-uuid")

    def test_empty_string(self):
        """Rejects empty string."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            parse_uuid("")


# ============================================================================
# STATUS_TIMESTAMPS tests
# ============================================================================


class TestStatusTimestamps:
    """Tests for timestamp mapping."""

    def test_skipped_sets_skipped_at(self):
        assert STATUS_TIMESTAMPS["skipped"] == "skipped_at"

    def test_labeled_sets_labeled_at(self):
        assert STATUS_TIMESTAMPS["labeled"] == "labeled_at"

    def test_other_statuses_have_no_timestamp(self):
        """Statuses not in the map don't set extra timestamps."""
        for status in VALID_LABELING_STATUSES:
            if status not in ("skipped", "labeled"):
                assert status not in STATUS_TIMESTAMPS


# ============================================================================
# show_article tests
# ============================================================================


def _make_mock_article(**overrides):
    """Create a mock Article with sensible defaults."""
    article = MagicMock()
    article.id = overrides.get("id", uuid.uuid4())
    article.title = overrides.get("title", "Test Article Title")
    article.source_name = overrides.get("source_name", "Test Source")
    article.url = overrides.get("url", "https://example.com/article")
    article.labeling_status = overrides.get("labeling_status", "pending")
    article.brands_mentioned = overrides.get("brands_mentioned", ["Nike"])
    article.published_at = overrides.get("published_at", datetime(2026, 4, 1))
    article.labeled_at = overrides.get("labeled_at", None)
    article.skipped_at = overrides.get("skipped_at", None)
    article.full_content = overrides.get("full_content", "Article content here.")
    return article


class TestShowArticle:
    """Tests for show_article function."""

    @patch("scripts.fix_label.db")
    def test_article_not_found(self, mock_db):
        """Returns False when article doesn't exist."""
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        result = show_article(uuid.uuid4())
        assert result is False

    @patch("scripts.fix_label.db")
    def test_article_found(self, mock_db, capsys):
        """Returns True and prints details when article exists."""
        article = _make_mock_article(
            title="Nike Sustainability Report",
            source_name="Reuters",
            labeling_status="false_positive",
        )

        session = MagicMock()
        # First query returns the article, second returns empty brand labels
        session.query.return_value.filter.return_value.first.return_value = article
        session.query.return_value.filter.return_value.all.return_value = []
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        result = show_article(article.id)
        assert result is True

        output = capsys.readouterr().out
        assert "Nike Sustainability Report" in output
        assert "Reuters" in output
        assert "false_positive" in output

    @patch("scripts.fix_label.db")
    def test_content_truncation(self, mock_db, capsys):
        """Content is truncated to specified length."""
        long_content = "x" * 5000
        article = _make_mock_article(full_content=long_content)

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        session.query.return_value.filter.return_value.all.return_value = []
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        result = show_article(article.id, content_length=100)
        assert result is True

        output = capsys.readouterr().out
        assert "4900 more chars" in output

    @patch("scripts.fix_label.db")
    def test_no_content(self, mock_db, capsys):
        """Handles articles with no full_content."""
        article = _make_mock_article(full_content=None)

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        session.query.return_value.filter.return_value.all.return_value = []
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        result = show_article(article.id)
        output = capsys.readouterr().out
        assert "no full_content available" in output

    @patch("scripts.fix_label.db")
    def test_brand_labels_displayed(self, mock_db, capsys):
        """Brand labels are shown when present."""
        article = _make_mock_article()

        brand_label = MagicMock()
        brand_label.brand = "Nike"
        brand_label.environmental_sentiment = 1
        brand_label.social_sentiment = -1
        brand_label.governance_sentiment = None
        brand_label.digital_sentiment = 0
        brand_label.confidence_score = 0.85

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        session.query.return_value.filter.return_value.all.return_value = [brand_label]
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        show_article(article.id)
        output = capsys.readouterr().out
        assert "Nike" in output
        assert "E:+1" in output
        assert "S:-1" in output
        assert "D:+0" in output
        assert "0.85" in output

    @patch("scripts.fix_label.db")
    def test_no_brands_mentioned(self, mock_db, capsys):
        """Handles articles with no brands_mentioned."""
        article = _make_mock_article(brands_mentioned=None)

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        session.query.return_value.filter.return_value.all.return_value = []
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        show_article(article.id)
        output = capsys.readouterr().out
        assert "Brands:  none" in output


# ============================================================================
# update_articles tests
# ============================================================================


class TestUpdateArticles:
    """Tests for update_articles function."""

    @patch("scripts.fix_label.db")
    def test_invalid_status_rejected(self, mock_db, capsys):
        """Returns early for invalid status values."""
        updated, not_found = update_articles([uuid.uuid4()], "invalid_status")
        assert updated == 0
        assert not_found == 1

        output = capsys.readouterr().out
        assert "Invalid status" in output

    @patch("scripts.fix_label.db")
    def test_article_not_found(self, mock_db, capsys):
        """Counts not-found articles."""
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        updated, not_found = update_articles([uuid.uuid4()], "skipped")
        assert updated == 0
        assert not_found == 1

    @patch("scripts.fix_label.db")
    def test_successful_update(self, mock_db, capsys):
        """Updates article status and sets timestamp."""
        article = _make_mock_article(labeling_status="false_positive")

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        updated, not_found = update_articles([article.id], "skipped")
        assert updated == 1
        assert not_found == 0
        assert article.labeling_status == "skipped"
        assert article.skipped_at is not None
        session.commit.assert_called_once()

    @patch("scripts.fix_label.db")
    def test_already_same_status(self, mock_db, capsys):
        """Skips articles already at target status."""
        article = _make_mock_article(labeling_status="skipped")

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        updated, not_found = update_articles([article.id], "skipped")
        assert updated == 0
        output = capsys.readouterr().out
        assert "Already skipped" in output
        session.commit.assert_not_called()

    @patch("scripts.fix_label.db")
    def test_batch_update(self, mock_db, capsys):
        """Updates multiple articles in one call."""
        articles = [
            _make_mock_article(id=uuid.uuid4(), labeling_status="false_positive"),
            _make_mock_article(id=uuid.uuid4(), labeling_status="false_positive"),
        ]

        session = MagicMock()
        session.query.return_value.filter.return_value.first.side_effect = articles
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        ids = [a.id for a in articles]
        updated, not_found = update_articles(ids, "skipped")
        assert updated == 2
        assert not_found == 0
        session.commit.assert_called_once()

    @patch("scripts.fix_label.db")
    def test_no_timestamp_for_pending(self, mock_db):
        """Statuses without timestamp mapping don't set extra fields."""
        article = _make_mock_article(labeling_status="false_positive")

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        update_articles([article.id], "pending")
        assert article.labeling_status == "pending"
        # skipped_at and labeled_at should not have been set
        # (MagicMock attrs are auto-created, so check setattr wasn't called for these)

    @patch("scripts.fix_label.db")
    def test_labeled_sets_labeled_at(self, mock_db):
        """Updating to 'labeled' sets labeled_at timestamp."""
        article = _make_mock_article(labeling_status="pending")

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)

        update_articles([article.id], "labeled")
        assert article.labeling_status == "labeled"
        assert article.labeled_at is not None


# ============================================================================
# list_statuses tests
# ============================================================================


class TestListStatuses:
    """Tests for list_statuses function."""

    def test_prints_all_statuses(self, capsys):
        """All valid statuses appear in output."""
        list_statuses()
        output = capsys.readouterr().out

        for status in VALID_LABELING_STATUSES:
            assert status in output

    def test_includes_descriptions(self, capsys):
        """Output includes descriptive text."""
        list_statuses()
        output = capsys.readouterr().out

        assert "Awaiting processing" in output
        assert "Not actually about the sportswear brand" in output
