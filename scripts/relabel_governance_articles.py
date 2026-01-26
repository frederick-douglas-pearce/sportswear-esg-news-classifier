#!/usr/bin/env python
"""Re-label governance-only articles that are likely mislabeled as financial news.

This script identifies articles that were labeled with only the Governance category
but lack ESG keywords, suggesting they are financial/business news rather than
true ESG governance content.

Usage:
    # Analyze articles and show statistics (dry run)
    uv run python scripts/relabel_governance_articles.py --analyze

    # Skip articles that are clearly non-ESG (no ESG keywords, financial patterns)
    uv run python scripts/relabel_governance_articles.py --skip-non-esg --dry-run
    uv run python scripts/relabel_governance_articles.py --skip-non-esg

    # Re-label borderline articles with v1.5.0 prompt
    uv run python scripts/relabel_governance_articles.py --relabel --dry-run
    uv run python scripts/relabel_governance_articles.py --relabel --batch-size 10
"""

import argparse
import logging
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

from src.labeling.esg_filter import check_esg_keywords

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


# Patterns for LEGITIMATE ESG governance content (checked in article TEXT only)
# These patterns identify real governance issues that should be KEPT
REAL_GOVERNANCE_CONTENT = [
    # Ethics and compliance violations
    r"\bSEC\s+investigation\b",
    r"\baccounting\s+(scandal|fraud|manipulation|irregularit)\w*\b",
    r"\bfraud\b",
    r"\bwhistleblower\b",
    r"\bdata\s+breach\b",
    r"\bcyber\s*attack\b",
    # Activist investor / proxy fights
    r"\bactivist\s+investor\b",
    r"\bproxy\s+(fight|battle|contest)\b",
    r"\bboard\s+(shakeup|overhaul|shake-up|shake\s+up)\b",
    r"\bElliott\s+(management|investment)\b",
    r"\bChip\s+Wilson\b",
    r"\bnominate\w*\s+\w+\s+director\b",
    r"\bdirector\s+candidate\b",
    # Board oversight issues
    r"\bboard\s+oversight\s+fail\w*\b",
    r"\bsuccession\s+planning\s+fail\w*\b",
    # ESG specific content
    r"\bESG\s+(report|disclosure|strategy)\b",
    r"\bsustainability\s+report\b",
    r"\bcarbon\s+(emission|footprint|neutral|target)\b",
    r"\bnet[\s-]?zero\b",
    r"\bclimate\s+(target|goal|commitment)\b",
    r"\bscience[\s-]based\s+target\b",
]


def get_governance_only_articles(session) -> list[tuple]:
    """Get articles that have only Governance labels (no E, S, or D)."""
    result = session.execute(
        text("""
            SELECT
                a.id,
                a.title,
                a.full_content,
                bl.reasoning,
                bl.prompt_version
            FROM brand_labels bl
            JOIN articles a ON a.id = bl.article_id
            WHERE a.labeling_status = 'labeled'
              AND bl.governance = true
              AND bl.environmental = false
              AND bl.social = false
              AND bl.digital_transformation = false
            ORDER BY a.created_at DESC
        """)
    )
    return list(result)


def classify_article(title: str, content: str, reasoning: str) -> dict:
    """Classify an article as ESG or non-ESG based on content analysis.

    Uses validated classification criteria:
    1. Check article TEXT (not reasoning) for real governance patterns
    2. Check for ESG keywords in article content
    3. Require multiple ESG keyword categories for weak matches

    Returns classification of 'keep' (legitimate ESG) or 'skip' (non-ESG).
    """
    article_text = f"{title or ''} {content or ''}"

    # Check for real governance content in article text only
    has_real_governance = any(
        re.search(pattern, article_text, re.IGNORECASE)
        for pattern in REAL_GOVERNANCE_CONTENT
    )

    # Check ESG keywords in article content
    esg_result = check_esg_keywords(article_text)

    # Classification logic:
    # KEEP if: real governance patterns found OR multiple ESG keyword categories
    # SKIP otherwise (financial news, routine business, etc.)
    if has_real_governance:
        classification = "keep"
        keep_reason = "Contains legitimate governance content"
    elif esg_result.has_esg_keywords and len(esg_result.esg_keywords_found) >= 2:
        classification = "keep"
        keep_reason = f"Multiple ESG keywords: {list(esg_result.esg_keywords_found.keys())}"
    else:
        classification = "skip"
        keep_reason = None

    return {
        "classification": classification,
        "keep_reason": keep_reason,
        "has_esg_keywords": esg_result.has_esg_keywords,
        "esg_keywords_found": esg_result.esg_keywords_found,
        "has_real_governance": has_real_governance,
        "recommendation": esg_result.recommendation,
    }


def analyze_articles(session) -> dict:
    """Analyze all governance-only articles and return statistics."""
    articles = get_governance_only_articles(session)

    stats = {
        "total": len(articles),
        "keep": [],
        "skip": [],
    }

    for article in articles:
        article_id, title, content, reasoning, prompt_version = article
        result = classify_article(title, content, reasoning)
        stats[result["classification"]].append({
            "id": article_id,
            "title": title,
            "keep_reason": result.get("keep_reason"),
            "has_real_governance": result["has_real_governance"],
            "esg_keywords_found": result["esg_keywords_found"],
        })

    return stats


def skip_non_esg_articles(session, dry_run: bool = True) -> int:
    """Mark non-ESG articles as skipped.

    Uses validated classification: articles without real governance patterns
    and without multiple ESG keyword categories are marked as skipped.
    """
    articles = get_governance_only_articles(session)
    skipped = 0

    for article in articles:
        article_id, title, content, reasoning, prompt_version = article
        result = classify_article(title, content, reasoning)

        # Skip articles classified as non-ESG
        if result["classification"] == "skip":
            if dry_run:
                logger.info(f"[DRY RUN] Would skip: {title[:60]}...")
                if result["esg_keywords_found"]:
                    logger.info(f"  ESG keywords: {list(result['esg_keywords_found'].keys())}")
            else:
                # Update article status to skipped
                session.execute(
                    text("""
                        UPDATE articles
                        SET labeling_status = 'skipped',
                            skipped_at = :now
                        WHERE id = :article_id
                    """),
                    {"article_id": article_id, "now": datetime.now(timezone.utc)},
                )

                # Update brand_labels to set all categories to false
                session.execute(
                    text("""
                        UPDATE brand_labels
                        SET governance = false,
                            governance_sentiment = null
                        WHERE article_id = :article_id
                    """),
                    {"article_id": article_id},
                )

                logger.info(f"Skipped: {title[:60]}...")

            skipped += 1

    if not dry_run:
        session.commit()

    return skipped


def main():
    parser = argparse.ArgumentParser(
        description="Re-label governance-only articles that are likely mislabeled"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze articles and show statistics",
    )
    parser.add_argument(
        "--skip-non-esg",
        action="store_true",
        help="Mark non-ESG articles as skipped",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Re-label uncertain articles with v1.5.0 prompt (not yet implemented)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for re-labeling (default: 10)",
    )

    args = parser.parse_args()

    if not any([args.analyze, args.skip_non_esg, args.relabel]):
        parser.print_help()
        return

    # Create database connection
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    engine = create_engine(database_url)

    with engine.connect() as session:
        if args.analyze:
            logger.info("=" * 80)
            logger.info("GOVERNANCE-ONLY ARTICLE ANALYSIS (Validated Classification)")
            logger.info("=" * 80)

            stats = analyze_articles(session)

            logger.info(f"\nTotal governance-only articles: {stats['total']}")
            logger.info("-" * 60)
            logger.info(f"Articles to KEEP (legitimate ESG): {len(stats['keep'])}")
            logger.info(f"Articles to SKIP (non-ESG): {len(stats['skip'])}")
            skip_rate = len(stats['skip']) / stats['total'] * 100 if stats['total'] else 0
            logger.info(f"Skip rate: {skip_rate:.1f}%")

            logger.info(f"\n{'='*60}")
            logger.info(f"RECOMMENDATION: {len(stats['skip'])} articles can be marked as skipped")
            logger.info(f"{'='*60}")

            if stats['skip']:
                logger.info("\nSample articles to SKIP (non-ESG):")
                for item in stats['skip'][:10]:
                    logger.info(f"  - {item['title'][:65]}...")

            if stats['keep']:
                logger.info("\nSample articles to KEEP (legitimate ESG):")
                for item in stats['keep'][:10]:
                    logger.info(f"  - {item['title'][:65]}...")
                    logger.info(f"    Reason: {item['keep_reason']}")

        if args.skip_non_esg:
            logger.info("=" * 80)
            logger.info("SKIPPING NON-ESG ARTICLES")
            logger.info("=" * 80)

            skipped = skip_non_esg_articles(session, dry_run=args.dry_run)

            logger.info(f"\n{'='*60}")
            if args.dry_run:
                logger.info(f"[DRY RUN] Would skip {skipped} articles")
            else:
                logger.info(f"Skipped {skipped} articles")
            logger.info(f"{'='*60}")

        if args.relabel:
            logger.info("=" * 80)
            logger.info("RE-LABELING WITH v1.5.0 PROMPT")
            logger.info("=" * 80)
            logger.info("This feature is not yet implemented.")
            logger.info("For now, use the standard labeling script:")
            logger.info("  uv run python scripts/label_articles.py --batch-size 10")


if __name__ == "__main__":
    main()
