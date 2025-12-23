#!/usr/bin/env python3
"""Script to identify and clean up false positive brand matches.

This script helps find articles that mention brand names like "Puma", "Patagonia",
or "Columbia" but are not actually about the sportswear companies.

Usage:
    # List potential false positives (preview)
    python scripts/cleanup_false_positives.py --list

    # Mark specific articles as false positives
    python scripts/cleanup_false_positives.py --mark-false-positive UUID1 UUID2

    # Delete articles marked as false positive
    python scripts/cleanup_false_positives.py --delete-false-positives

    # Auto-detect false positives by title keywords and mark them
    python scripts/cleanup_false_positives.py --auto-detect --dry-run
    python scripts/cleanup_false_positives.py --auto-detect
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import cast, or_
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import String

from src.data_collection.database import db
from src.data_collection.models import Article, ArticleChunk, BrandLabel, LabelEvidence


# Keywords that indicate false positive matches
FALSE_POSITIVE_PATTERNS = {
    "Puma": [
        r"\bpuma\s+(animal|wildcat|cougar|mountain\s+lion|cat)\b",
        r"\bford\s+puma\b",
        r"\bpuma\s+(exploration|mining|gold|drill)\b",
        r"\b(wild|hunting|predator|prey).*puma\b",
        r"\bpuma.*(penguin|wildlife|patagonia\s+region)\b",
    ],
    "Patagonia": [
        r"\bpatagonia\s+(region|argentina|chile|trek|hike|glacier|national\s+park)\b",
        r"\b(torres\s+del\s+paine|tierra\s+del\s+fuego)\b",
        r"\bpatagonia.*(penguin|wildlife|puma|guanaco|condor)\b",
        r"\b(trip|travel|expedition|voyage)\s+to\s+patagonia\b",
        r"\bsouth(ern)?\s+patagonia\b",
        r"\bsailing\s+(in|to|through)\s+patagonia\b",
        r"\btop\s+gear.*patagonia\b",
    ],
    "Columbia": [
        r"\bcolumbia\s+(university|river|country|pictures|records)\b",
        r"\bdistrict\s+of\s+columbia\b",
        r"\bbritish\s+columbia\b",  # This could be either, context matters
    ],
    "Vans": [
        # EU legislation / automotive policy
        r"\bcars\s+and\s+vans\b",
        r"\bco2\s+(emission|legislation|standard).*vans\b",
        r"\bvans?\s+(fleet|delivery|cargo|commercial|transit)\b",
        r"\belectric\s+vans?\b",
        r"\b(ev|battery|hybrid)\s+vans?\b",
        # Stolen vehicles
        r"\bstolen\s+vans?\b",
        r"\bvans?\s+(stolen|theft|burglary)\b",
        # Camper/motor vehicles
        r"\bcamper\s+vans?\b",
        r"\bvw\s+vans?\b",
        r"\bford\s+transit\b",
    ],
    "Decathlon": [
        # Investment/VC firms (not the sporting goods retailer)
        r"\bdecathlon\s+(capital|management|partners|fund|ventures)\b",
        r"\bdecathlon\s+(investment|portfolio|equity)\b",
        r"\bfunding\s+(from|by)\s+decathlon\b",
    ],
    "Black Diamond": [
        # Power/utility company
        r"\bblack\s+diamond\s+(power|energy|electric|utility)\b",
        r"\bblack\s+diamond.*(psc|commission|ratepayer)\b",
        # Event name (not equipment company)
        r"\bblack\s+diamond\s+(weekend|event|conference)\b",
    ],
    "New Balance": [
        # Political/diplomatic phrase
        r"\bnew\s+balance\s+of\s+(power|influence|force)\b",
        # Children's balance bikes
        r"\bnew\s+balance\s+bikes?\b",
        r"\bbalance\s+bikes?\s+for\s+(kids|children|kindergarten)\b",
    ],
    "Anta": [
        # Indian political constituency
        r"\banta\s+(assembly|constituency|bypoll|seat)\b",
        # Financial company (NASDAQ: ANTA)
        r"\bantalpha\b",
    ],
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def list_potential_false_positives(session, limit: int = 50) -> list[Article]:
    """List articles that might be false positives based on title/content patterns."""
    # Get articles with potentially ambiguous brand names
    ambiguous_brands = list(FALSE_POSITIVE_PATTERNS.keys())

    potential_fps = []

    for brand in ambiguous_brands:
        # Use PostgreSQL array containment operator @>
        articles = (
            session.query(Article)
            .filter(Article.brands_mentioned.op("@>")(cast([brand], ARRAY(String))))
            .filter(Article.labeling_status.in_(["pending", "labeled", "skipped"]))
            .order_by(Article.created_at.desc())
            .limit(limit)
            .all()
        )

        for article in articles:
            # Check title against false positive patterns
            title_lower = article.title.lower() if article.title else ""
            content_lower = (article.full_content or article.description or "").lower()

            for pattern in FALSE_POSITIVE_PATTERNS.get(brand, []):
                if re.search(pattern, title_lower, re.IGNORECASE) or re.search(
                    pattern, content_lower[:2000], re.IGNORECASE
                ):
                    potential_fps.append((article, brand, pattern))
                    break

    return potential_fps


def mark_articles_as_false_positive(
    session, article_ids: list[UUID], reason: str = "Manually marked as false positive"
) -> int:
    """Mark specific articles as false positives."""
    count = 0
    for article_id in article_ids:
        article = session.query(Article).filter(Article.id == article_id).first()
        if article:
            article.labeling_status = "false_positive"
            article.labeling_error = reason
            count += 1
            print(f"  Marked as false positive: {article.title[:60]}...")
    session.commit()
    return count


def delete_false_positive_articles(session, dry_run: bool = False) -> int:
    """Delete articles marked as false_positive and their related data."""
    articles = (
        session.query(Article)
        .filter(Article.labeling_status == "false_positive")
        .all()
    )

    if not articles:
        print("No articles marked as false_positive found.")
        return 0

    print(f"\nFound {len(articles)} articles marked as false_positive:")
    for article in articles[:20]:
        print(f"  - {article.title[:70]}...")
    if len(articles) > 20:
        print(f"  ... and {len(articles) - 20} more")

    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(articles)} articles and related data.")
        return len(articles)

    # Delete related data first (cascades should handle this, but be explicit)
    article_ids = [a.id for a in articles]

    # Delete evidence for brand labels of these articles
    brand_labels = (
        session.query(BrandLabel)
        .filter(BrandLabel.article_id.in_(article_ids))
        .all()
    )
    label_ids = [bl.id for bl in brand_labels]

    if label_ids:
        session.query(LabelEvidence).filter(
            LabelEvidence.brand_label_id.in_(label_ids)
        ).delete(synchronize_session=False)

    # Delete brand labels
    session.query(BrandLabel).filter(
        BrandLabel.article_id.in_(article_ids)
    ).delete(synchronize_session=False)

    # Delete chunks
    session.query(ArticleChunk).filter(
        ArticleChunk.article_id.in_(article_ids)
    ).delete(synchronize_session=False)

    # Delete articles
    session.query(Article).filter(Article.id.in_(article_ids)).delete(
        synchronize_session=False
    )

    session.commit()
    print(f"\nDeleted {len(articles)} false positive articles and related data.")
    return len(articles)


def auto_detect_false_positives(session, dry_run: bool = False) -> int:
    """Automatically detect and mark false positives based on patterns."""
    potential_fps = list_potential_false_positives(session, limit=500)

    if not potential_fps:
        print("No potential false positives detected.")
        return 0

    print(f"\nDetected {len(potential_fps)} potential false positives:")

    # Group by article to avoid duplicates
    seen_ids = set()
    unique_fps = []
    for article, brand, pattern in potential_fps:
        if article.id not in seen_ids:
            seen_ids.add(article.id)
            unique_fps.append((article, brand, pattern))

    for article, brand, pattern in unique_fps[:30]:
        print(f"\n  [{brand}] {article.title[:65]}...")
        print(f"    Pattern: {pattern}")
        print(f"    Status: {article.labeling_status}")

    if len(unique_fps) > 30:
        print(f"\n  ... and {len(unique_fps) - 30} more")

    if dry_run:
        print(f"\n[DRY RUN] Would mark {len(unique_fps)} articles as false_positive.")
        return len(unique_fps)

    # Mark them as false positives
    for article, brand, pattern in unique_fps:
        article.labeling_status = "false_positive"
        article.labeling_error = f"Auto-detected: {brand} matched pattern '{pattern}'"

    session.commit()
    print(f"\nMarked {len(unique_fps)} articles as false_positive.")
    return len(unique_fps)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Identify and clean up false positive brand matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List potential false positives based on title patterns",
    )
    parser.add_argument(
        "--mark-false-positive",
        nargs="+",
        metavar="UUID",
        help="Mark specific article UUIDs as false positives",
    )
    parser.add_argument(
        "--delete-false-positives",
        action="store_true",
        help="Delete all articles marked as false_positive",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Automatically detect and mark false positives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    db.init_db()

    with db.get_session() as session:
        if args.list:
            print("Scanning for potential false positives...\n")
            potential_fps = list_potential_false_positives(session)

            if not potential_fps:
                print("No potential false positives found.")
                return 0

            print(f"Found {len(potential_fps)} potential false positives:\n")
            for article, brand, pattern in potential_fps:
                print(f"[{brand}] {article.id}")
                print(f"  Title: {article.title[:70]}...")
                print(f"  Pattern: {pattern}")
                print(f"  Status: {article.labeling_status}")
                print()

        elif args.mark_false_positive:
            article_ids = [UUID(aid) for aid in args.mark_false_positive]
            count = mark_articles_as_false_positive(session, article_ids)
            print(f"\nMarked {count} articles as false_positive.")

        elif args.delete_false_positives:
            count = delete_false_positive_articles(session, dry_run=args.dry_run)
            return 0 if count >= 0 else 1

        elif args.auto_detect:
            count = auto_detect_false_positives(session, dry_run=args.dry_run)
            return 0 if count >= 0 else 1

        else:
            # Default: show summary
            total = session.query(Article).count()
            false_pos = (
                session.query(Article)
                .filter(Article.labeling_status == "false_positive")
                .count()
            )

            print("=== False Positive Summary ===")
            print(f"Total articles:       {total}")
            print(f"Marked false_positive: {false_pos}")
            print()
            print("Use --list to scan for potential false positives")
            print("Use --auto-detect to automatically detect and mark them")
            print("Use --delete-false-positives to remove marked articles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
