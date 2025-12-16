#!/usr/bin/env python3
"""
Cleanup script to remove non-English articles from the database.

Uses langdetect to identify articles that are not in English and removes them.

Usage:
    python scripts/cleanup_non_english.py [OPTIONS]

Options:
    --dry-run       Show what would be deleted without actually deleting
    --verbose, -v   Show details of each article found
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langdetect import LangDetectException, detect

from src.data_collection.database import db


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_language(text: str, min_length: int = 100) -> str | None:
    """
    Detect language of text.

    Returns language code (e.g., 'en', 'de', 'it') or None if detection fails.
    """
    if not text or len(text) < min_length:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove non-English articles from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show details of each article found",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Scanning for non-English articles...")

    db.init_db()

    non_english_articles = []

    with db.get_session() as session:
        # Get all articles with content
        from src.data_collection.models import Article

        articles = (
            session.query(Article)
            .filter(Article.full_content.isnot(None))
            .filter(Article.scrape_status == "success")
            .all()
        )

        logger.info(f"Checking {len(articles)} articles with content...")

        for article in articles:
            # Use first 500 chars for faster detection
            sample = article.full_content[:500] if article.full_content else ""
            lang = detect_language(sample)

            if lang and lang != "en":
                non_english_articles.append(
                    {
                        "id": article.id,
                        "article_id": article.article_id,
                        "title": article.title[:80] if article.title else "No title",
                        "language": lang,
                        "db_language": article.language,
                    }
                )
                if args.verbose:
                    logger.info(
                        f"Found {lang}: {article.title[:60]}..."
                    )

    if not non_english_articles:
        logger.info("No non-English articles found.")
        return 0

    logger.info(f"\nFound {len(non_english_articles)} non-English articles:")

    # Group by language
    by_language: dict[str, int] = {}
    for article in non_english_articles:
        lang = article["language"]
        by_language[lang] = by_language.get(lang, 0) + 1

    for lang, count in sorted(by_language.items(), key=lambda x: -x[1]):
        logger.info(f"  {lang}: {count} articles")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would delete the above articles.")
        logger.info("Run without --dry-run to actually delete them.")
        return 0

    # Delete the articles
    logger.info("\nDeleting non-English articles...")

    with db.get_session() as session:
        from src.data_collection.models import Article

        deleted = 0
        for article_info in non_english_articles:
            article = session.query(Article).filter(
                Article.id == article_info["id"]
            ).first()
            if article:
                session.delete(article)
                deleted += 1

        session.commit()
        logger.info(f"Deleted {deleted} non-English articles.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
