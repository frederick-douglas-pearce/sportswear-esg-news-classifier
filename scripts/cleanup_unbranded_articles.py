#!/usr/bin/env python3
"""
Database Cleanup Script - Remove articles not related to sportswear brands.

This script:
1. Re-analyzes all articles to detect brand mentions in their text
2. Updates the brands_mentioned field for articles that have brands
3. Deletes articles that don't mention any tracked brands

Usage:
    python scripts/cleanup_unbranded_articles.py [OPTIONS]

Options:
    --dry-run       Show what would be done without making changes
    --verbose, -v   Enable verbose logging

Examples:
    # Preview what would be deleted (dry run)
    python scripts/cleanup_unbranded_articles.py --dry-run

    # Run the actual cleanup
    python scripts/cleanup_unbranded_articles.py
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.config import BRANDS
from src.data_collection.database import db


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_brands(text: str) -> list[str]:
    """
    Extract mentioned brands from article text using word boundary matching.

    This prevents false positives like:
    - 'Anta' matching 'Santa' or 'amenities'
    - 'ASICS' matching 'basic'
    - 'Fila' matching 'Philadelphia'
    """
    if not text:
        return []
    text_lower = text.lower()
    found_brands = []
    for brand in BRANDS:
        # Use word boundary regex to match whole words only
        # \b matches word boundaries (start/end of word)
        pattern = r'\b' + re.escape(brand.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_brands.append(brand)
    return found_brands


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up articles not related to sportswear brands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
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
    logger.info("Starting database cleanup")
    logger.info(f"Tracking {len(BRANDS)} brands: {', '.join(BRANDS[:5])}...")

    db.init_db()

    with db.get_session() as session:
        # Get all articles
        from src.data_collection.models import Article

        articles = session.query(Article).all()
        logger.info(f"Found {len(articles)} total articles")

        updated_count = 0
        to_delete = []

        for article in articles:
            # Combine all text fields for brand detection
            combined_text = " ".join(filter(None, [
                article.title,
                article.description,
                article.full_content,
            ]))

            # Detect brands
            brands = extract_brands(combined_text)

            if brands:
                # Update brands_mentioned if different
                current_brands = set(article.brands_mentioned or [])
                new_brands = set(brands)

                if current_brands != new_brands:
                    if args.dry_run:
                        logger.debug(
                            f"Would update brands for '{article.title[:50]}...': "
                            f"{list(current_brands)} -> {list(new_brands)}"
                        )
                    else:
                        article.brands_mentioned = brands
                    updated_count += 1
            else:
                # No brands found - mark for deletion
                to_delete.append(article)
                logger.debug(f"No brands in: {article.title[:60]}...")

        logger.info(f"Articles to update brands: {updated_count}")
        logger.info(f"Articles to delete (no brands): {len(to_delete)}")

        if args.dry_run:
            logger.info("[DRY RUN] No changes made")
            logger.info("\nSample articles that would be deleted:")
            for article in to_delete[:10]:
                logger.info(f"  - {article.title[:70]}...")
            if len(to_delete) > 10:
                logger.info(f"  ... and {len(to_delete) - 10} more")
        else:
            # Commit brand updates
            session.commit()
            logger.info(f"Updated brands for {updated_count} articles")

            # Delete unbranded articles
            for article in to_delete:
                session.delete(article)
            session.commit()
            logger.info(f"Deleted {len(to_delete)} unbranded articles")

        # Final counts
        remaining = session.query(Article).count()
        logger.info(f"Articles remaining in database: {remaining}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
