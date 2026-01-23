#!/usr/bin/env python3
"""CLI tool for reviewing and correcting article labels.

Usage:
    # Review suspect environmental labels (product articles)
    uv run python scripts/review_labels.py --suspect-env

    # Review all environmental labels
    uv run python scripts/review_labels.py --category environmental

    # Review specific article by ID
    uv run python scripts/review_labels.py --article-id UUID

    # Dry run (don't save changes)
    uv run python scripts/review_labels.py --suspect-env --dry-run
"""

import argparse
import os
import sys
import textwrap
from datetime import datetime, timezone
from uuid import UUID

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    """Create database engine."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    return create_engine(database_url)


def wrap_text(text_str: str, width: int = 80, indent: str = "    ") -> str:
    """Wrap text for display."""
    if not text_str:
        return ""
    lines = textwrap.wrap(text_str, width=width - len(indent))
    return "\n".join(indent + line for line in lines)


def get_suspect_environmental_articles(engine):
    """Get articles that look like product reviews but have environmental labels."""
    query = text("""
        SELECT DISTINCT a.id, a.title, a.source_name, a.url, a.labeled_at,
               LEFT(a.full_content, 500) as content_preview
        FROM articles a
        JOIN brand_labels bl ON bl.article_id = a.id
        WHERE a.labeling_status = 'labeled'
          AND bl.environmental = true
          AND (
            a.title ILIKE '%review%'
            OR a.title ILIKE '%best %'
            OR a.title ILIKE '%top %'
            OR a.title ILIKE '% vs %'
            OR a.title ILIKE '%buying guide%'
            OR a.title ILIKE '%gift guide%'
            OR a.title ILIKE '%gear %'
            OR a.title ILIKE '%stylish%'
            OR a.title ILIKE '%functional%'
            OR a.source_name ILIKE '%t3%'
            OR a.source_name ILIKE '%techradar%'
            OR a.source_name ILIKE '%tomsguide%'
            OR a.source_name ILIKE '%wirecutter%'
          )
        ORDER BY a.labeled_at DESC
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        return result.fetchall()


def get_article_labels_and_evidence(engine, article_id: UUID):
    """Get all labels and evidence for an article."""
    labels_query = text("""
        SELECT bl.id, bl.brand, bl.environmental, bl.social, bl.governance,
               bl.digital_transformation, bl.environmental_sentiment,
               bl.social_sentiment, bl.governance_sentiment, bl.reasoning
        FROM brand_labels bl
        WHERE bl.article_id = :article_id
    """)

    evidence_query = text("""
        SELECT le.category, le.excerpt, le.relevance_score, le.rerank_score
        FROM label_evidence le
        JOIN brand_labels bl ON le.brand_label_id = bl.id
        WHERE bl.article_id = :article_id
        ORDER BY le.category, le.rerank_score DESC NULLS LAST
    """)

    with engine.connect() as conn:
        labels = conn.execute(labels_query, {"article_id": article_id}).fetchall()
        evidence = conn.execute(evidence_query, {"article_id": article_id}).fetchall()

    return labels, evidence


def display_article_for_review(article, labels, evidence):
    """Display article details for review."""
    print("\n" + "=" * 80)
    print(f"ARTICLE: {article.title}")
    print("=" * 80)
    print(f"Source: {article.source_name}")
    print(f"URL: {article.url}")
    print(f"Labeled at: {article.labeled_at}")

    if article.content_preview:
        print(f"\nContent preview:")
        print(wrap_text(article.content_preview + "..."))

    print("\n" + "-" * 40)
    print("LABELS:")
    print("-" * 40)

    for label in labels:
        categories = []
        if label.environmental:
            sent = {1: "+", 0: "~", -1: "-"}.get(label.environmental_sentiment, "?")
            categories.append(f"Environmental ({sent})")
        if label.social:
            sent = {1: "+", 0: "~", -1: "-"}.get(label.social_sentiment, "?")
            categories.append(f"Social ({sent})")
        if label.governance:
            sent = {1: "+", 0: "~", -1: "-"}.get(label.governance_sentiment, "?")
            categories.append(f"Governance ({sent})")
        if label.digital_transformation:
            categories.append("Digital")

        print(f"\n  Brand: {label.brand}")
        print(f"  Categories: {', '.join(categories) if categories else 'None'}")
        if label.reasoning:
            print(f"  Reasoning: {label.reasoning[:200]}...")

    print("\n" + "-" * 40)
    print("EVIDENCE:")
    print("-" * 40)

    current_category = None
    for ev in evidence:
        if ev.category != current_category:
            current_category = ev.category
            print(f"\n  [{current_category.upper()}]")

        score_str = f"rerank={ev.rerank_score:.3f}" if ev.rerank_score else f"relevance={ev.relevance_score:.3f}"
        print(f"    ({score_str})")
        print(wrap_text(f'"{ev.excerpt}"', indent="      "))


def update_article_status(engine, article_id: UUID, new_status: str, dry_run: bool = False):
    """Update article labeling status."""
    if dry_run:
        print(f"\n[DRY RUN] Would update article {article_id} to status: {new_status}")
        return

    with engine.begin() as conn:
        if new_status == "skipped":
            conn.execute(text("""
                UPDATE articles
                SET labeling_status = 'skipped',
                    skipped_at = :now
                WHERE id = :article_id
            """), {"article_id": article_id, "now": datetime.now(timezone.utc)})
        else:
            conn.execute(text("""
                UPDATE articles
                SET labeling_status = :status
                WHERE id = :article_id
            """), {"article_id": article_id, "status": new_status})

        # Optionally delete the brand labels if marking as skipped
        if new_status == "skipped":
            conn.execute(text("""
                DELETE FROM label_evidence
                WHERE brand_label_id IN (
                    SELECT id FROM brand_labels WHERE article_id = :article_id
                )
            """), {"article_id": article_id})
            conn.execute(text("""
                DELETE FROM brand_labels WHERE article_id = :article_id
            """), {"article_id": article_id})

    print(f"\n✓ Updated article to: {new_status}")


def review_articles(articles, engine, dry_run: bool = False):
    """Interactive review of articles."""
    total = len(articles)
    corrections = {"skipped": 0, "kept": 0, "quit": 0}

    print(f"\nFound {total} articles to review.")
    print("For each article, enter:")
    print("  [s] Skip (mark as product article, remove labels)")
    print("  [k] Keep (labels are correct)")
    print("  [q] Quit review")
    print()

    for i, article in enumerate(articles, 1):
        print(f"\n{'#' * 80}")
        print(f"# Article {i}/{total}")
        print(f"{'#' * 80}")

        labels, evidence = get_article_labels_and_evidence(engine, article.id)
        display_article_for_review(article, labels, evidence)

        while True:
            choice = input("\n[s]kip / [k]eep / [q]uit? ").strip().lower()

            if choice == 's':
                update_article_status(engine, article.id, "skipped", dry_run)
                corrections["skipped"] += 1
                break
            elif choice == 'k':
                print("✓ Keeping labels as-is")
                corrections["kept"] += 1
                break
            elif choice == 'q':
                corrections["quit"] = total - i
                print(f"\nQuitting. {total - i} articles remaining.")
                print(f"\nSummary: {corrections['skipped']} skipped, {corrections['kept']} kept")
                return corrections
            else:
                print("Invalid choice. Enter 's', 'k', or 'q'.")

    print(f"\n{'=' * 80}")
    print("REVIEW COMPLETE")
    print(f"{'=' * 80}")
    print(f"Skipped (corrected): {corrections['skipped']}")
    print(f"Kept (correct): {corrections['kept']}")

    return corrections


def main():
    parser = argparse.ArgumentParser(description="Review and correct article labels")
    parser.add_argument("--suspect-env", action="store_true",
                        help="Review suspect environmental labels (product articles)")
    parser.add_argument("--category", type=str,
                        help="Review all articles with this category label")
    parser.add_argument("--article-id", type=str,
                        help="Review specific article by ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't save changes to database")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list articles, don't review interactively")

    args = parser.parse_args()

    if not any([args.suspect_env, args.category, args.article_id]):
        parser.print_help()
        sys.exit(1)

    engine = get_engine()

    if args.suspect_env:
        articles = get_suspect_environmental_articles(engine)

        if args.list_only:
            print(f"\nFound {len(articles)} suspect articles:\n")
            for i, a in enumerate(articles, 1):
                print(f"{i}. [{a.source_name}] {a.title[:70]}...")
                print(f"   ID: {a.id}")
            return

        if not articles:
            print("No suspect articles found.")
            return

        review_articles(articles, engine, args.dry_run)

    elif args.article_id:
        article_id = UUID(args.article_id)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, title, source_name, url, labeled_at,
                       LEFT(full_content, 500) as content_preview
                FROM articles WHERE id = :id
            """), {"id": article_id})
            article = result.fetchone()

        if not article:
            print(f"Article not found: {args.article_id}")
            return

        labels, evidence = get_article_labels_and_evidence(engine, article_id)
        display_article_for_review(article, labels, evidence)

        if not args.list_only:
            choice = input("\n[s]kip / [k]eep? ").strip().lower()
            if choice == 's':
                update_article_status(engine, article_id, "skipped", args.dry_run)


if __name__ == "__main__":
    main()
