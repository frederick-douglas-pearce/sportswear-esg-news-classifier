#!/usr/bin/env python3
"""
GDELT Historical Backfill Script

Collects 3 months of historical data from GDELT in weekly batches.
This avoids overwhelming the API and allows for resumable collection.

Usage:
    python scripts/gdelt_backfill.py [OPTIONS]

Options:
    --months N          Number of months to backfill (default: 3)
    --start-from DATE   Resume from a specific date (YYYY-MM-DD)
    --dry-run           Don't save to database, just show what would be done
    --max-calls N       Maximum API calls per batch (default: 50)
    --scrape-limit N    Maximum articles to scrape per batch (default: 100)
    --verbose, -v       Enable verbose logging

Examples:
    # Run full 3-month backfill
    python scripts/gdelt_backfill.py

    # Test first batch only
    python scripts/gdelt_backfill.py --dry-run --max-calls 5

    # Resume from a specific date
    python scripts/gdelt_backfill.py --start-from 2025-11-01

    # Backfill only 1 month
    python scripts/gdelt_backfill.py --months 1
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.collector import NewsCollector
from src.data_collection.database import db


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill historical GDELT data in weekly batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Number of months to backfill (default: 3, max supported by GDELT)",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Resume from a specific date (YYYY-MM-DD), collects from this date forward",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database, just show what would be done",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=50,
        help="Maximum API calls per batch (default: 50)",
    )
    parser.add_argument(
        "--scrape-limit",
        type=int,
        default=100,
        help="Maximum articles to scrape per batch (default: 100)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def generate_weekly_batches(
    start_date: datetime, end_date: datetime
) -> list[tuple[datetime, datetime]]:
    """
    Generate weekly date ranges from start to end.

    Returns list of (week_start, week_end) tuples.
    """
    batches = []
    current_start = start_date

    while current_start < end_date:
        week_end = min(current_start + timedelta(days=7), end_date)
        batches.append((current_start, week_end))
        current_start = week_end

    return batches


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting GDELT Historical Backfill")

    # Calculate date range
    end_date = datetime.now()

    if args.start_from:
        start_date = datetime.strptime(args.start_from, "%Y-%m-%d")
        logger.info(f"Resuming from {args.start_from}")
    else:
        start_date = end_date - timedelta(days=args.months * 30)
        logger.info(f"Backfilling {args.months} months of data")

    # Validate date range (GDELT only supports ~3 months)
    max_lookback = end_date - timedelta(days=90)
    if start_date < max_lookback:
        logger.warning(
            f"GDELT only supports ~3 months of history. "
            f"Adjusting start date from {start_date.date()} to {max_lookback.date()}"
        )
        start_date = max_lookback

    # Generate weekly batches
    batches = generate_weekly_batches(start_date, end_date)
    logger.info(f"Generated {len(batches)} weekly batches")

    # Initialize database
    db.init_db()

    # Process each batch
    total_stats = {
        "api_calls": 0,
        "articles_fetched": 0,
        "articles_duplicates": 0,
        "articles_duplicate_title": 0,
        "articles_no_brand": 0,
        "articles_scraped": 0,
        "articles_scrape_failed": 0,
    }

    collector = NewsCollector(source="gdelt")

    for i, (batch_start, batch_end) in enumerate(batches, 1):
        logger.info(
            f"Processing batch {i}/{len(batches)}: "
            f"{batch_start.date()} to {batch_end.date()}"
        )

        try:
            stats = collector.collect_daily_news(
                max_calls=args.max_calls,
                scrape_limit=args.scrape_limit,
                dry_run=args.dry_run,
                brand_only=True,
                start_datetime=batch_start,
                end_datetime=batch_end,
            )

            # Accumulate stats
            total_stats["api_calls"] += stats.api_calls
            total_stats["articles_fetched"] += stats.articles_fetched
            total_stats["articles_duplicates"] += stats.articles_duplicates
            total_stats["articles_duplicate_title"] += stats.articles_duplicate_title
            total_stats["articles_no_brand"] += stats.articles_no_brand
            total_stats["articles_scraped"] += stats.articles_scraped
            total_stats["articles_scrape_failed"] += stats.articles_scrape_failed

            logger.info(
                f"Batch {i} complete: {stats.articles_fetched} new articles, "
                f"{stats.articles_scraped} scraped"
            )

            # Reset API call counter for next batch
            collector.api_client.api_calls_made = 0

        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            logger.info(f"To resume, run with: --start-from {batch_start.date()}")
            return 1

    # Print summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total API calls: {total_stats['api_calls']}")
    logger.info(f"Total new articles: {total_stats['articles_fetched']}")
    logger.info(f"Total duplicate IDs: {total_stats['articles_duplicates']}")
    logger.info(f"Total duplicate titles: {total_stats['articles_duplicate_title']}")
    logger.info(f"Total filtered (no brand): {total_stats['articles_no_brand']}")
    logger.info(f"Total articles scraped: {total_stats['articles_scraped']}")
    logger.info(f"Total scrape failures: {total_stats['articles_scrape_failed']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
