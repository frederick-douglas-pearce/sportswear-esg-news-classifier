"""Tests for scorecard history database operations.

These tests require PostgreSQL.
They are skipped by default and can be run with:
    RUN_DB_TESTS=1 pytest tests/test_scorecard_database.py -v

To use a specific test database:
    TEST_DATABASE_URL=postgresql://... RUN_DB_TESTS=1 pytest tests/test_scorecard_database.py -v
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

from src.data_collection.database import Database
from src.data_collection.models import Base, ScorecardBrandScore, ScorecardSnapshot
from src.scorecard.database import ScorecardDatabase


def should_run_db_tests():
    """Check if database tests should run."""
    return os.environ.get("RUN_DB_TESTS") == "1"


requires_postgres = pytest.mark.skipif(
    not should_run_db_tests(),
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@pytest.fixture
def test_db():
    """Create a test database connection."""
    database_url = os.environ.get(
        "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news_test"
    )

    db = Database(database_url=database_url)
    db.init_db()

    yield db

    # Cleanup
    with db.get_session() as session:
        session.query(ScorecardBrandScore).delete()
        session.query(ScorecardSnapshot).delete()


@pytest.fixture
def scorecard_db(test_db):
    """Create a ScorecardDatabase instance."""
    return ScorecardDatabase(database=test_db)


@pytest.fixture
def sample_scorecard_data():
    """Create sample scorecard data matching the export format."""
    return {
        "period_days": 14,
        "period_start": "2026-01-27",
        "period_end": "2026-02-10",
        "articles_in_period": 100,
        "articles_after_dedup": 85,
        "duplicates_removed": 15,
        "similarity_threshold": 0.70,
        "require_label_match": True,
        "top_brands": [
            {
                "brand": "Patagonia",
                "total": 12,
                "environmental": 8,
                "social": 3,
                "governance": 1,
                "digital_transformation": 0,
                "article_count": 5,
                "medal": "gold",
            },
            {
                "brand": "Nike",
                "total": 8,
                "environmental": 4,
                "social": 2,
                "governance": 1,
                "digital_transformation": 1,
                "article_count": 10,
                "medal": "silver",
            },
            {
                "brand": "Adidas",
                "total": 6,
                "environmental": 2,
                "social": 3,
                "governance": 1,
                "digital_transformation": 0,
                "article_count": 8,
                "medal": "bronze",
            },
        ],
        "bottom_brands": [
            {
                "brand": "Reebok",
                "total": -3,
                "environmental": -1,
                "social": -2,
                "governance": 0,
                "digital_transformation": 0,
                "article_count": 2,
            },
        ],
        "all_brand_scores": [
            {
                "brand": "Patagonia",
                "total": 12,
                "environmental": 8,
                "social": 3,
                "governance": 1,
                "digital_transformation": 0,
                "article_count": 5,
            },
            {
                "brand": "Nike",
                "total": 8,
                "environmental": 4,
                "social": 2,
                "governance": 1,
                "digital_transformation": 1,
                "article_count": 10,
            },
            {
                "brand": "Adidas",
                "total": 6,
                "environmental": 2,
                "social": 3,
                "governance": 1,
                "digital_transformation": 0,
                "article_count": 8,
            },
            {
                "brand": "Puma",
                "total": 2,
                "environmental": 1,
                "social": 0,
                "governance": 1,
                "digital_transformation": 0,
                "article_count": 3,
            },
            {
                "brand": "Under Armour",
                "total": 0,
                "environmental": 0,
                "social": 0,
                "governance": 0,
                "digital_transformation": 0,
                "article_count": 1,
            },
            {
                "brand": "Reebok",
                "total": -3,
                "environmental": -1,
                "social": -2,
                "governance": 0,
                "digital_transformation": 0,
                "article_count": 2,
            },
        ],
    }


@requires_postgres
class TestSaveScorecard:
    """Tests for saving scorecard snapshots."""

    def test_save_new_snapshot(self, test_db, scorecard_db, sample_scorecard_data):
        """Should save a new scorecard snapshot with brand scores."""
        with test_db.get_session() as session:
            snapshot = scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

            assert snapshot.id is not None
            assert snapshot.period_days == 14
            assert snapshot.articles_in_period == 100
            assert snapshot.articles_after_dedup == 85
            assert snapshot.duplicates_removed == 15
            assert snapshot.similarity_threshold == 0.70
            assert snapshot.require_label_match is True

            # Capture id before session closes
            snapshot_id = snapshot.id

        # Verify brand scores were saved
        with test_db.get_session() as session:
            scores = (
                session.query(ScorecardBrandScore)
                .filter(ScorecardBrandScore.snapshot_id == snapshot_id)
                .order_by(ScorecardBrandScore.rank_position)
                .all()
            )

            assert len(scores) == 6

            # Check top performer
            assert scores[0].brand == "Patagonia"
            assert scores[0].total == 12
            assert scores[0].environmental == 8
            assert scores[0].rank_position == 1
            assert scores[0].is_top_performer is True
            assert scores[0].medal == "gold"

            # Check bottom performer
            reebok = next(s for s in scores if s.brand == "Reebok")
            assert reebok.total == -3
            assert reebok.is_bottom_performer is True
            assert reebok.medal is None

    def test_upsert_replaces_existing(self, test_db, scorecard_db, sample_scorecard_data):
        """Should replace existing snapshot for same period_days and period_end."""
        # Save first time
        with test_db.get_session() as session:
            snapshot1 = scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)
            snapshot1_id = snapshot1.id

        # Modify data
        sample_scorecard_data["articles_in_period"] = 150
        sample_scorecard_data["all_brand_scores"][0]["total"] = 20  # Patagonia

        # Save again (same period)
        with test_db.get_session() as session:
            snapshot2 = scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

            # Should have same ID (updated, not new)
            assert snapshot2.id == snapshot1_id
            assert snapshot2.articles_in_period == 150

        # Verify scores were replaced
        with test_db.get_session() as session:
            scores = (
                session.query(ScorecardBrandScore)
                .filter(ScorecardBrandScore.snapshot_id == snapshot1_id)
                .all()
            )

            patagonia = next(s for s in scores if s.brand == "Patagonia")
            assert patagonia.total == 20

    def test_different_periods_create_separate_snapshots(
        self, test_db, scorecard_db, sample_scorecard_data
    ):
        """Should create separate snapshots for different period_end dates."""
        # Save first snapshot
        with test_db.get_session() as session:
            snapshot1 = scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)
            snapshot1_id = snapshot1.id

        # Change period_end to next day
        sample_scorecard_data["period_start"] = "2026-01-28"
        sample_scorecard_data["period_end"] = "2026-02-11"

        # Save second snapshot
        with test_db.get_session() as session:
            snapshot2 = scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)
            snapshot2_id = snapshot2.id

            # Should be different snapshots
            assert snapshot1_id != snapshot2_id

        # Verify both exist
        with test_db.get_session() as session:
            count = session.query(ScorecardSnapshot).count()
            assert count == 2


@requires_postgres
class TestQueryScorecard:
    """Tests for querying scorecard history."""

    def test_get_snapshot_by_date(self, test_db, scorecard_db, sample_scorecard_data):
        """Should retrieve snapshot by date and period."""
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            snapshot = scorecard_db.get_snapshot_by_date(
                session, date(2026, 2, 10), period_days=14
            )

            assert snapshot is not None
            assert snapshot.period_days == 14
            assert snapshot.articles_in_period == 100

    def test_get_snapshot_by_date_not_found(self, test_db, scorecard_db):
        """Should return None for non-existent date."""
        with test_db.get_session() as session:
            snapshot = scorecard_db.get_snapshot_by_date(
                session, date(2026, 1, 1), period_days=14
            )
            assert snapshot is None

    def test_get_latest_snapshot(self, test_db, scorecard_db, sample_scorecard_data):
        """Should return the most recent snapshot."""
        # Create multiple snapshots
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        sample_scorecard_data["period_start"] = "2026-01-28"
        sample_scorecard_data["period_end"] = "2026-02-11"

        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            latest = scorecard_db.get_latest_snapshot(session, period_days=14)

            assert latest is not None
            assert latest.period_end == date(2026, 2, 11)

    def test_get_brand_score_history(self, test_db, scorecard_db, sample_scorecard_data):
        """Should return historical scores for a brand."""
        # Create two snapshots
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        sample_scorecard_data["period_start"] = "2026-01-28"
        sample_scorecard_data["period_end"] = "2026-02-11"
        sample_scorecard_data["all_brand_scores"][1]["total"] = 10  # Nike

        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            history = scorecard_db.get_brand_score_history(
                session, "Nike", days=30, period_days=14
            )

            assert len(history) == 2
            assert history[0]["date"] == "2026-02-10"
            assert history[0]["total"] == 8
            assert history[1]["date"] == "2026-02-11"
            assert history[1]["total"] == 10

    def test_get_all_brand_scores_for_date(
        self, test_db, scorecard_db, sample_scorecard_data
    ):
        """Should return all brand scores for a specific date."""
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            scores = scorecard_db.get_all_brand_scores_for_date(
                session, date(2026, 2, 10), period_days=14
            )

            assert len(scores) == 6
            assert scores[0]["brand"] == "Patagonia"
            assert scores[0]["rank_position"] == 1
            assert scores[0]["is_top_performer"] is True

    def test_get_medal_history(self, test_db, scorecard_db, sample_scorecard_data):
        """Should return medal history for a brand."""
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            history = scorecard_db.get_medal_history(
                session, "Nike", days=90, period_days=14
            )

            assert len(history) == 1
            assert history[0]["medal"] == "silver"
            assert history[0]["total"] == 8


@requires_postgres
class TestStatistics:
    """Tests for statistical queries."""

    def test_get_snapshot_count(self, test_db, scorecard_db, sample_scorecard_data):
        """Should return the count of snapshots."""
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            count = scorecard_db.get_snapshot_count(session, period_days=14)
            assert count == 1

    def test_get_deduplication_stats(self, test_db, scorecard_db, sample_scorecard_data):
        """Should return deduplication statistics."""
        with test_db.get_session() as session:
            scorecard_db.save_scorecard_snapshot(session, sample_scorecard_data)

        with test_db.get_session() as session:
            stats = scorecard_db.get_deduplication_stats(
                session, days=30, period_days=14
            )

            assert stats["snapshot_count"] == 1
            assert stats["total_articles"] == 100
            assert stats["total_duplicates"] == 15
            assert stats["dedup_rate"] == 15.0

    def test_get_deduplication_stats_empty(self, test_db, scorecard_db):
        """Should return zeros when no snapshots exist."""
        with test_db.get_session() as session:
            stats = scorecard_db.get_deduplication_stats(
                session, days=30, period_days=14
            )

            assert stats["snapshot_count"] == 0
            assert stats["total_articles"] == 0
            assert stats["total_duplicates"] == 0
            assert stats["dedup_rate"] == 0.0
