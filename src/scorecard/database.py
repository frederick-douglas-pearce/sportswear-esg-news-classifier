"""Database operations for scorecard history."""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.data_collection.config import settings
from src.data_collection.database import Database, db
from src.data_collection.models import ScorecardBrandScore, ScorecardSnapshot

logger = logging.getLogger(__name__)


class ScorecardDatabase:
    """Database operations for scorecard history."""

    def __init__(self, database: Database | None = None):
        """Initialize with a database connection."""
        self.db = database or db

    def save_scorecard_snapshot(
        self,
        session: Session,
        scorecard_data: dict[str, Any],
    ) -> ScorecardSnapshot:
        """Save a scorecard snapshot to the database.

        Uses upsert semantics: if a snapshot for the same (period_days, period_end)
        exists, it will be replaced. This allows safe re-runs on the same day.

        Args:
            session: Database session
            scorecard_data: Scorecard data from calculate_brand_scores(), containing:
                - period_days: int
                - period_start: str (ISO date)
                - period_end: str (ISO date)
                - articles_in_period: int
                - articles_after_dedup: int
                - duplicates_removed: int
                - all_brand_scores: list of brand score dicts
                - top_brands: list (used to identify top performers)
                - bottom_brands: list (used to identify bottom performers)

        Returns:
            Created or updated ScorecardSnapshot object
        """
        # Parse dates
        period_start = datetime.strptime(scorecard_data["period_start"], "%Y-%m-%d").date()
        period_end = datetime.strptime(scorecard_data["period_end"], "%Y-%m-%d").date()
        period_days = scorecard_data["period_days"]

        # Extract settings (use defaults if not present)
        similarity_threshold = scorecard_data.get(
            "similarity_threshold", settings.dedup_similarity_threshold
        )
        require_label_match = scorecard_data.get("require_label_match", True)

        # Check for existing snapshot
        existing = (
            session.query(ScorecardSnapshot)
            .filter(
                and_(
                    ScorecardSnapshot.period_days == period_days,
                    ScorecardSnapshot.period_end == period_end,
                )
            )
            .first()
        )

        if existing:
            logger.info(
                f"Updating existing snapshot for period_days={period_days}, "
                f"period_end={period_end}"
            )
            # Delete existing brand scores (will be re-created)
            session.query(ScorecardBrandScore).filter(
                ScorecardBrandScore.snapshot_id == existing.id
            ).delete()

            # Update snapshot metadata
            existing.period_start = period_start
            existing.articles_in_period = scorecard_data["articles_in_period"]
            existing.articles_after_dedup = scorecard_data["articles_after_dedup"]
            existing.duplicates_removed = scorecard_data["duplicates_removed"]
            existing.similarity_threshold = similarity_threshold
            existing.require_label_match = require_label_match
            existing.created_at = datetime.now(timezone.utc)
            snapshot = existing
        else:
            # Create new snapshot
            snapshot = ScorecardSnapshot(
                period_days=period_days,
                period_start=period_start,
                period_end=period_end,
                articles_in_period=scorecard_data["articles_in_period"],
                articles_after_dedup=scorecard_data["articles_after_dedup"],
                duplicates_removed=scorecard_data["duplicates_removed"],
                similarity_threshold=similarity_threshold,
                require_label_match=require_label_match,
            )
            session.add(snapshot)
            session.flush()  # Get the snapshot ID

        # Build sets for top/bottom performer lookup
        top_brand_names = {b["brand"] for b in scorecard_data.get("top_brands", [])}
        bottom_brand_names = {b["brand"] for b in scorecard_data.get("bottom_brands", [])}

        # Create a medal lookup from top_brands
        medal_lookup = {}
        for brand_data in scorecard_data.get("top_brands", []):
            if brand_data.get("medal"):
                medal_lookup[brand_data["brand"]] = brand_data["medal"]

        # Save brand scores
        all_scores = scorecard_data.get("all_brand_scores", [])
        for rank, brand_data in enumerate(all_scores, start=1):
            brand_name = brand_data["brand"]
            brand_score = ScorecardBrandScore(
                snapshot_id=snapshot.id,
                brand=brand_name,
                environmental=brand_data["environmental"],
                social=brand_data["social"],
                governance=brand_data["governance"],
                digital_transformation=brand_data["digital_transformation"],
                total=brand_data["total"],
                article_count=brand_data["article_count"],
                rank_position=rank,
                is_top_performer=brand_name in top_brand_names,
                is_bottom_performer=brand_name in bottom_brand_names,
                medal=medal_lookup.get(brand_name),
            )
            session.add(brand_score)

        session.flush()
        logger.info(
            f"Saved scorecard snapshot: {len(all_scores)} brands, "
            f"period={period_start} to {period_end}"
        )
        return snapshot

    def get_snapshot_by_date(
        self,
        session: Session,
        period_end: date,
        period_days: int = 14,
    ) -> ScorecardSnapshot | None:
        """Get a snapshot for a specific date and period.

        Args:
            session: Database session
            period_end: The end date of the period
            period_days: The rolling window size (default: 14)

        Returns:
            ScorecardSnapshot or None if not found
        """
        return (
            session.query(ScorecardSnapshot)
            .filter(
                and_(
                    ScorecardSnapshot.period_days == period_days,
                    ScorecardSnapshot.period_end == period_end,
                )
            )
            .first()
        )

    def get_latest_snapshot(
        self,
        session: Session,
        period_days: int = 14,
    ) -> ScorecardSnapshot | None:
        """Get the most recent snapshot for a given period length.

        Args:
            session: Database session
            period_days: The rolling window size (default: 14)

        Returns:
            Most recent ScorecardSnapshot or None if none exist
        """
        return (
            session.query(ScorecardSnapshot)
            .filter(ScorecardSnapshot.period_days == period_days)
            .order_by(ScorecardSnapshot.period_end.desc())
            .first()
        )

    def get_brand_score_history(
        self,
        session: Session,
        brand: str,
        days: int = 30,
        period_days: int = 14,
    ) -> list[dict[str, Any]]:
        """Get historical scores for a specific brand.

        Args:
            session: Database session
            brand: Brand name
            days: Number of days of history to retrieve
            period_days: The rolling window size (default: 14)

        Returns:
            List of dicts with date and score information
        """
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)

        results = (
            session.query(ScorecardBrandScore, ScorecardSnapshot)
            .join(ScorecardSnapshot)
            .filter(
                and_(
                    ScorecardBrandScore.brand == brand,
                    ScorecardSnapshot.period_days == period_days,
                    ScorecardSnapshot.period_end >= cutoff,
                )
            )
            .order_by(ScorecardSnapshot.period_end.asc())
            .all()
        )

        return [
            {
                "date": snapshot.period_end.isoformat(),
                "total": score.total,
                "environmental": score.environmental,
                "social": score.social,
                "governance": score.governance,
                "digital_transformation": score.digital_transformation,
                "rank_position": score.rank_position,
                "article_count": score.article_count,
                "is_top_performer": score.is_top_performer,
                "is_bottom_performer": score.is_bottom_performer,
                "medal": score.medal,
            }
            for score, snapshot in results
        ]

    def get_all_brand_scores_for_date(
        self,
        session: Session,
        period_end: date,
        period_days: int = 14,
    ) -> list[dict[str, Any]]:
        """Get all brand scores for a specific date.

        Args:
            session: Database session
            period_end: The end date of the period
            period_days: The rolling window size (default: 14)

        Returns:
            List of brand score dicts ordered by rank
        """
        snapshot = self.get_snapshot_by_date(session, period_end, period_days)
        if not snapshot:
            return []

        scores = (
            session.query(ScorecardBrandScore)
            .filter(ScorecardBrandScore.snapshot_id == snapshot.id)
            .order_by(ScorecardBrandScore.rank_position)
            .all()
        )

        return [
            {
                "brand": s.brand,
                "total": s.total,
                "environmental": s.environmental,
                "social": s.social,
                "governance": s.governance,
                "digital_transformation": s.digital_transformation,
                "article_count": s.article_count,
                "rank_position": s.rank_position,
                "is_top_performer": s.is_top_performer,
                "is_bottom_performer": s.is_bottom_performer,
                "medal": s.medal,
            }
            for s in scores
        ]

    def get_medal_history(
        self,
        session: Session,
        brand: str,
        days: int = 90,
        period_days: int = 14,
    ) -> list[dict[str, Any]]:
        """Get medal earning history for a brand.

        Args:
            session: Database session
            brand: Brand name
            days: Number of days of history
            period_days: The rolling window size (default: 14)

        Returns:
            List of dicts with dates when brand earned medals
        """
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)

        results = (
            session.query(ScorecardBrandScore, ScorecardSnapshot)
            .join(ScorecardSnapshot)
            .filter(
                and_(
                    ScorecardBrandScore.brand == brand,
                    ScorecardBrandScore.medal.isnot(None),
                    ScorecardSnapshot.period_days == period_days,
                    ScorecardSnapshot.period_end >= cutoff,
                )
            )
            .order_by(ScorecardSnapshot.period_end.desc())
            .all()
        )

        return [
            {
                "date": snapshot.period_end.isoformat(),
                "medal": score.medal,
                "total": score.total,
                "rank_position": score.rank_position,
            }
            for score, snapshot in results
        ]

    def get_snapshot_count(self, session: Session, period_days: int = 14) -> int:
        """Get the total number of snapshots for a period length.

        Args:
            session: Database session
            period_days: The rolling window size (default: 14)

        Returns:
            Number of snapshots
        """
        return (
            session.query(ScorecardSnapshot)
            .filter(ScorecardSnapshot.period_days == period_days)
            .count()
        )

    def get_deduplication_stats(
        self,
        session: Session,
        days: int = 30,
        period_days: int = 14,
    ) -> dict[str, Any]:
        """Get deduplication statistics over a period.

        Args:
            session: Database session
            days: Number of days of history
            period_days: The rolling window size (default: 14)

        Returns:
            Dict with deduplication statistics
        """
        from sqlalchemy import func

        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)

        results = (
            session.query(
                func.count(ScorecardSnapshot.id).label("snapshot_count"),
                func.sum(ScorecardSnapshot.articles_in_period).label("total_articles"),
                func.sum(ScorecardSnapshot.duplicates_removed).label("total_duplicates"),
                func.avg(ScorecardSnapshot.duplicates_removed).label("avg_duplicates_per_day"),
            )
            .filter(
                and_(
                    ScorecardSnapshot.period_days == period_days,
                    ScorecardSnapshot.period_end >= cutoff,
                )
            )
            .first()
        )

        if not results or results.snapshot_count == 0:
            return {
                "snapshot_count": 0,
                "total_articles": 0,
                "total_duplicates": 0,
                "avg_duplicates_per_day": 0.0,
                "dedup_rate": 0.0,
            }

        total_articles = results.total_articles or 0
        total_duplicates = results.total_duplicates or 0
        dedup_rate = (
            (total_duplicates / total_articles * 100) if total_articles > 0 else 0.0
        )

        return {
            "snapshot_count": results.snapshot_count,
            "total_articles": total_articles,
            "total_duplicates": total_duplicates,
            "avg_duplicates_per_day": float(results.avg_duplicates_per_day or 0),
            "dedup_rate": dedup_rate,
        }


# Default instance
scorecard_db = ScorecardDatabase()
