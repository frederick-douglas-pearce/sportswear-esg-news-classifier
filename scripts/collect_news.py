#!/usr/bin/env python3
"""
ESG News Collection Script

Usage:
    python scripts/collect_news.py [OPTIONS]

Options:
    --source SOURCE     API source: 'newsdata' or 'gdelt' (default: newsdata)
    --dry-run           Don't save to database, just show what would be done
    --max-calls N       Maximum API calls to make (default: 200)
    --scrape-only       Only scrape pending articles, skip API collection
    --scrape-limit N    Maximum articles to scrape (default: 100)
    --with-keywords     Search with brand + keyword combinations (old behavior)
    --verbose, -v       Enable verbose logging

Examples:
    # Run full daily collection using NewsData.io (default)
    python scripts/collect_news.py

    # Collect from GDELT (free, no API key needed, 3 months history)
    python scripts/collect_news.py --source gdelt

    # Test GDELT without saving (dry run)
    python scripts/collect_news.py --source gdelt --dry-run --max-calls 3

    # Use keyword-filtered queries
    python scripts/collect_news.py --with-keywords

    # Only scrape pending articles
    python scripts/collect_news.py --scrape-only --scrape-limit 50
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.collector import NewsCollector
from src.data_collection.config import settings
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
        description="Collect ESG news articles for sportswear brands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["newsdata", "gdelt"],
        default="newsdata",
        help="API source: 'newsdata' (requires API key) or 'gdelt' (free, 3 months history)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database, just show what would be done",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=settings.max_api_calls_per_day,
        help=f"Maximum API calls to make (default: {settings.max_api_calls_per_day})",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape pending articles, skip API collection",
    )
    parser.add_argument(
        "--scrape-limit",
        type=int,
        default=100,
        help="Maximum articles to scrape (default: 100)",
    )
    parser.add_argument(
        "--with-keywords",
        action="store_true",
        help="Search with brand + keyword combinations (old behavior, default: brand-only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting ESG News Collection")

    # Check for API key only when using NewsData.io
    if args.source == "newsdata" and not settings.newsdata_api_key and not args.scrape_only:
        logger.error("NEWSDATA_API_KEY not set. Please set it in .env file.")
        logger.info("Tip: Use --source gdelt to collect from GDELT (no API key needed)")
        return 1

    try:
        collector = NewsCollector(source=args.source)

        brand_only = not args.with_keywords

        if args.scrape_only:
            logger.info("Running scrape-only mode")
            stats = collector.scrape_pending_articles(
                limit=args.scrape_limit,
                dry_run=args.dry_run,
            )
        else:
            mode = "brand-only" if brand_only else "with keywords"
            logger.info(f"Running {args.source.upper()} collection in {mode} mode")
            stats = collector.collect_daily_news(
                max_calls=args.max_calls,
                scrape_limit=args.scrape_limit,
                dry_run=args.dry_run,
                brand_only=brand_only,
            )

        logger.info(f"Collection complete:")
        logger.info(f"  API calls made: {stats.api_calls}")
        logger.info(f"  New articles: {stats.articles_fetched}")
        logger.info(f"  Duplicate IDs skipped: {stats.articles_duplicates}")
        logger.info(f"  Duplicate titles skipped: {stats.articles_duplicate_title}")
        logger.info(f"  Filtered (no brand): {stats.articles_no_brand}")
        logger.info(f"  Articles scraped: {stats.articles_scraped}")
        logger.info(f"  Scrape failures: {stats.articles_scrape_failed}")

        if stats.errors:
            logger.warning(f"  Errors encountered: {len(stats.errors)}")
            for error in stats.errors[:5]:
                logger.warning(f"    - {error}")
            if len(stats.errors) > 5:
                logger.warning(f"    ... and {len(stats.errors) - 5} more")
            return 2

        return 0

    except Exception as e:
        logger.exception(f"Collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
