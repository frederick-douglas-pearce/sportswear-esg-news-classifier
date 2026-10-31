#!/usr/bin/env python3
"""CLI tool for viewing and correcting article labeling status.

Used after /review-labels to verify flagged articles and apply corrections.

Usage:
    # Show article details and content preview
    python scripts/fix_label.py show <article-id>

    # Update one or more articles to a new status
    python scripts/fix_label.py update <id> [<id>...] --status skipped

    # A human ruling: store the status's human marker as the reason (or clear the reason)
    python scripts/fix_label.py update <id> --status skipped --manual
    python scripts/fix_label.py update <id> --status unlabelable --clear-reason

    # List valid labeling statuses
    python scripts/fix_label.py statuses

See also: queries/article_queries.sql, queries/labeling_queries.sql
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.database import db
from src.data_collection.models import (
    FP_REASON_MANUAL,
    MANUAL_SKIP_REASON,
    VALID_LABELING_STATUSES,
    Article,
    BrandLabel,
)

logger = logging.getLogger(__name__)

# Statuses that set specific timestamp fields on transition
STATUS_TIMESTAMPS = {
    "skipped": "skipped_at",
    "labeled": "labeled_at",
}

# The human marker --manual stores as the reason for each status it takes; a labeled row's marker
# is its human-reviewed brand label, so its reason is cleared (#234)
MANUAL_REASONS = {
    "false_positive": FP_REASON_MANUAL,
    "skipped": MANUAL_SKIP_REASON,
    "labeled": None,
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_uuid(value: str) -> UUID:
    """Parse and validate a UUID string."""
    try:
        return UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid UUID: {value}")


def show_article(article_id: UUID, content_length: int = 2000) -> bool:
    """Display article details and content preview.

    Returns True if article was found, False otherwise.
    """
    with db.get_session() as session:
        article = session.query(Article).filter(Article.id == article_id).first()
        if not article:
            print(f"Article not found: {article_id}")
            return False

        print("=" * 70)
        print(f"Title:   {article.title}")
        print(f"Source:  {article.source_name}")
        print(f"URL:     {article.url}")
        print(f"Status:  {article.labeling_status}")
        print(f"Reason:  {article.labeling_error if article.labeling_error is not None else 'none'}")
        print(f"Brands:  {', '.join(article.brands_mentioned) if article.brands_mentioned else 'none'}")

        if article.published_at:
            print(f"Published: {article.published_at.strftime('%Y-%m-%d')}")
        if article.labeled_at:
            print(f"Labeled:   {article.labeled_at.strftime('%Y-%m-%d %H:%M')}")
        if article.skipped_at:
            print(f"Skipped:   {article.skipped_at.strftime('%Y-%m-%d %H:%M')}")

        # Brand labels
        labels = (
            session.query(BrandLabel)
            .filter(BrandLabel.article_id == article_id)
            .all()
        )
        if labels:
            print(f"\nBrand Labels ({len(labels)}):")
            for bl in labels:
                sentiments = []
                if bl.environmental_sentiment is not None:
                    sentiments.append(f"E:{bl.environmental_sentiment:+d}")
                if bl.social_sentiment is not None:
                    sentiments.append(f"S:{bl.social_sentiment:+d}")
                if bl.governance_sentiment is not None:
                    sentiments.append(f"G:{bl.governance_sentiment:+d}")
                if bl.digital_sentiment is not None:
                    sentiments.append(f"D:{bl.digital_sentiment:+d}")
                print(f"  {bl.brand}: {' '.join(sentiments)} (confidence: {bl.confidence_score})")

        # Content preview
        if article.full_content:
            preview = article.full_content[:content_length]
            if len(article.full_content) > content_length:
                preview += f"\n... [{len(article.full_content) - content_length} more chars]"
            print(f"\nContent ({len(article.full_content)} chars):")
            print("-" * 70)
            print(preview)
        else:
            print("\nContent: [no full_content available]")

        print("=" * 70)
        return True


def update_articles(
    article_ids: list[UUID],
    new_status: str,
    manual: bool = False,
    clear_reason: bool = False,
) -> tuple[int, int]:
    """Update labeling status for one or more articles.

    By default the stored reason (``labeling_error``) is kept, so a hand move stays detectable by
    the reason it left behind (``possible_hand_move``). ``manual`` replaces it with the status's
    human marker instead (``MANUAL_REASONS``), which ``derive_label_source`` reads as ``human``;
    ``clear_reason`` empties it (#234). Leaving ``skipped`` clears ``skipped_at``, and a row not at
    ``skipped`` that still carries one is updated to clear it. An article already at
    ``new_status`` is still updated when its reason or ``skipped_at`` changes.

    Returns (updated_count, not_found_count).
    """
    if manual and clear_reason:
        raise ValueError("Pass manual or clear_reason, not both")
    if manual and new_status not in MANUAL_REASONS:
        raise ValueError(
            f"No human marker for status {new_status!r}; --manual takes {', '.join(MANUAL_REASONS)}"
        )

    if new_status not in VALID_LABELING_STATUSES:
        print(f"Invalid status: {new_status}")
        print(f"Valid statuses: {', '.join(VALID_LABELING_STATUSES)}")
        return 0, len(article_ids)

    now = datetime.now(timezone.utc)
    updated = 0
    not_found = 0

    with db.get_session() as session:
        for article_id in article_ids:
            article = session.query(Article).filter(Article.id == article_id).first()
            if not article:
                print(f"  Not found: {article_id}")
                not_found += 1
                continue

            old_status = article.labeling_status
            old_reason = article.labeling_error
            if manual:
                new_reason = MANUAL_REASONS[new_status]
            elif clear_reason:
                new_reason = None
            else:
                new_reason = old_reason
            stale_skipped_at = new_status != "skipped" and article.skipped_at is not None
            if old_status == new_status and old_reason == new_reason and not stale_skipped_at:
                print(f"  Already {new_status}: {article.title[:60]}")
                continue

            print(f"  {old_status} -> {new_status}: {article.title[:60]}")
            if old_status != new_status:
                article.labeling_status = new_status

                # Set appropriate timestamp
                timestamp_field = STATUS_TIMESTAMPS.get(new_status)
                if timestamp_field:
                    setattr(article, timestamp_field, now)
            if stale_skipped_at:
                article.skipped_at = None
                print("    skipped_at cleared")
            if old_reason != new_reason:
                article.labeling_error = new_reason
                print(f"    reason: {old_reason!r} -> {new_reason!r}")
            updated += 1

        if updated > 0:
            session.commit()

    return updated, not_found


def list_statuses() -> None:
    """Print valid labeling statuses with descriptions."""
    descriptions = {
        "pending": "Awaiting processing",
        "chunked": "Legacy: chunked, awaiting labeling (the pipeline no longer writes it, #222)",
        "embedded": "Legacy: embedded, awaiting labeling (the pipeline no longer writes it, #222)",
        "labeled": "Successfully labeled with ESG categories",
        "skipped": "No ESG content for a sportswear brand, or not established (no LLM reply, or one that did not settle the brand); see labeling_error (UNJUDGED_SKIP_REASONS)",
        "false_positive": "Not actually about the sportswear brand (e.g., 'Vans' = vehicles)",
        "unlabelable": "Could not be labeled (insufficient content, language issues)",
        "deduplicated": "Duplicate of another article",
    }
    print("Valid labeling statuses:")
    print("-" * 70)
    for status in VALID_LABELING_STATUSES:
        desc = descriptions.get(status, "")
        print(f"  {status:<16} {desc}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="View and correct article labeling status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show article details
    %(prog)s show 12345678-1234-1234-1234-123456789abc

    # Update articles to skipped
    %(prog)s update ID1 ID2 --status skipped

    # Record a human ruling with the status's human marker as the reason
    %(prog)s update ID --status skipped --manual

    # List valid statuses
    %(prog)s statuses
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # show subcommand
    show_parser = subparsers.add_parser("show", help="Show article details and content preview")
    show_parser.add_argument("article_id", type=parse_uuid, help="Article UUID")
    show_parser.add_argument(
        "--content-length",
        type=int,
        default=2000,
        help="Max characters of content to show (default: 2000)",
    )

    # update subcommand
    update_parser = subparsers.add_parser("update", help="Update article labeling status")
    update_parser.add_argument(
        "article_ids",
        type=parse_uuid,
        nargs="+",
        help="One or more article UUIDs",
    )
    update_parser.add_argument(
        "--status",
        required=True,
        choices=VALID_LABELING_STATUSES,
        help="New labeling status",
    )
    reason_group = update_parser.add_mutually_exclusive_group()
    reason_group.add_argument(
        "--manual",
        action="store_true",
        help="Store the status's human marker as the reason (false_positive, skipped, labeled)",
    )
    reason_group.add_argument(
        "--clear-reason",
        action="store_true",
        help="Empty the stored reason (labeling_error)",
    )

    # statuses subcommand
    subparsers.add_parser("statuses", help="List valid labeling statuses")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    if args.command == "show":
        found = show_article(args.article_id, args.content_length)
        sys.exit(0 if found else 1)

    elif args.command == "update":
        updated, not_found = update_articles(
            args.article_ids, args.status, manual=args.manual, clear_reason=args.clear_reason
        )
        print(f"\nUpdated: {updated}, Not found: {not_found}")
        sys.exit(0 if not_found == 0 else 1)

    elif args.command == "statuses":
        list_statuses()


if __name__ == "__main__":
    main()
