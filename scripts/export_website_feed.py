#!/usr/bin/env python3
"""Export ESG news data for website display.

Generates JSON and Atom feeds from labeled articles for display on
Jekyll/al-folio static website.

Usage:
    # Export JSON for Jekyll _data folder
    uv run python scripts/export_website_feed.py --format json -o ~/website/_data/esg_news.json

    # Export Atom feed for RSS subscribers
    uv run python scripts/export_website_feed.py --format atom -o ~/website/assets/feeds/esg_news.atom

    # Export both formats
    uv run python scripts/export_website_feed.py --format both \
        --json-output ~/website/_data/esg_news.json \
        --atom-output ~/website/assets/feeds/esg_news.atom

    # Limit number of articles
    uv run python scripts/export_website_feed.py --format json --limit 100

    # Preview without writing files
    uv run python scripts/export_website_feed.py --format json --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import joinedload

from src.data_collection.database import db
from src.data_collection.models import Article, BrandLabel, LabelEvidence


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_sentiment_label(sentiment: int | None) -> str | None:
    """Convert sentiment value to label."""
    if sentiment is None:
        return None
    return {-1: "negative", 0: "neutral", 1: "positive"}.get(sentiment)


def query_labeled_articles(session, limit: int | None = None) -> list[Article]:
    """Query labeled articles with all relationships loaded.

    Args:
        session: Database session
        limit: Maximum number of articles to return

    Returns:
        List of Article objects with brand_labels and evidence loaded
    """
    query = (
        session.query(Article)
        .filter(Article.labeling_status == "labeled")
        .options(
            joinedload(Article.brand_labels).joinedload(BrandLabel.evidence)
        )
        .order_by(Article.published_at.desc())
    )

    if limit:
        query = query.limit(limit)

    return query.all()


def format_article_for_json(article: Article) -> dict[str, Any]:
    """Format an article for JSON export.

    Args:
        article: Article object with loaded relationships

    Returns:
        Dictionary matching the JSON schema
    """
    # Collect all unique brands and categories for this article
    all_brands = set()
    all_categories = set()
    brand_details = []

    for label in article.brand_labels:
        all_brands.add(label.brand)

        # Build category info
        categories = {}
        for cat_name, cat_field, sent_field in [
            ("environmental", "environmental", "environmental_sentiment"),
            ("social", "social", "social_sentiment"),
            ("governance", "governance", "governance_sentiment"),
            ("digital_transformation", "digital_transformation", "digital_sentiment"),
        ]:
            applies = getattr(label, cat_field, False) or False
            sentiment = getattr(label, sent_field, None)

            if applies:
                all_categories.add(cat_name)

            categories[cat_name] = {
                "applies": applies,
                "sentiment": sentiment,
                "sentiment_label": get_sentiment_label(sentiment) if applies else None,
            }

        # Build evidence list
        evidence_list = []
        for ev in label.evidence:
            evidence_list.append({
                "category": ev.category,
                "excerpt": ev.excerpt,
                "relevance_score": ev.relevance_score,
            })

        brand_details.append({
            "brand": label.brand,
            "categories": categories,
            "evidence": evidence_list,
            "confidence": label.confidence_score,
            "reasoning": label.reasoning,
        })

    # Format published date
    published_at = article.published_at
    if published_at and published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)

    return {
        "id": str(article.id),
        "title": article.title,
        "url": article.url,
        "source_name": article.source_name,
        "published_at": published_at.isoformat() if published_at else None,
        "published_date": published_at.strftime("%Y-%m-%d") if published_at else None,
        "brands": sorted(all_brands),
        "categories": sorted(all_categories),
        "brand_details": brand_details,
    }


def export_to_json(articles: list[Article]) -> dict[str, Any]:
    """Export articles to JSON format.

    Args:
        articles: List of Article objects

    Returns:
        Dictionary with full JSON structure
    """
    # Collect all unique brands and categories across all articles
    all_brands = set()
    all_categories = set()

    formatted_articles = []
    for article in articles:
        formatted = format_article_for_json(article)
        formatted_articles.append(formatted)
        all_brands.update(formatted["brands"])
        all_categories.update(formatted["categories"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_articles": len(formatted_articles),
        "brands": sorted(all_brands),
        "categories": sorted(all_categories),
        "articles": formatted_articles,
    }


def export_to_atom(articles: list[Article], base_url: str) -> str:
    """Export articles to Atom feed format.

    Args:
        articles: List of Article objects
        base_url: Base URL for the website

    Returns:
        Atom feed as XML string
    """
    from feedgen.feed import FeedGenerator

    fg = FeedGenerator()
    fg.id(f"{base_url}/esg-news/")
    fg.title("ESG Sportswear News")
    fg.subtitle("Environmental, Social, and Governance news for sportswear brands")
    fg.author({"name": "ESG News Classifier", "uri": base_url})
    fg.link(href=f"{base_url}/esg-news/", rel="alternate")
    fg.link(href=f"{base_url}/assets/feeds/esg_news.atom", rel="self")
    fg.language("en")
    fg.updated(datetime.now(timezone.utc))

    for article in articles:
        fe = fg.add_entry()
        fe.id(str(article.id))
        fe.title(article.title)
        fe.link(href=article.url)

        # Build summary from brands and categories
        brands = set()
        categories = set()
        for label in article.brand_labels:
            brands.add(label.brand)
            if label.environmental:
                categories.add("Environmental")
            if label.social:
                categories.add("Social")
            if label.governance:
                categories.add("Governance")
            if label.digital_transformation:
                categories.add("Digital Transformation")

        summary_parts = []
        if brands:
            summary_parts.append(f"Brands: {', '.join(sorted(brands))}")
        if categories:
            summary_parts.append(f"Categories: {', '.join(sorted(categories))}")
        if article.source_name:
            summary_parts.append(f"Source: {article.source_name}")

        fe.summary(" | ".join(summary_parts))

        # Set dates
        published_at = article.published_at
        if published_at:
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
            fe.published(published_at)
            fe.updated(published_at)

        # Add categories as tags
        for brand in brands:
            fe.category(term=brand, label=brand)
        for cat in categories:
            fe.category(term=cat.lower().replace(" ", "_"), label=cat)

    return fg.atom_str(pretty=True).decode("utf-8")


def write_json(data: dict, filepath: Path) -> None:
    """Write JSON data to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def write_atom(content: str, filepath: Path) -> None:
    """Write Atom feed to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def print_summary(articles: list[Article], json_data: dict | None = None) -> None:
    """Print export summary."""
    print(f"\n{'=' * 60}")
    print("ESG NEWS EXPORT SUMMARY")
    print("=" * 60)
    print(f"\nTotal articles: {len(articles)}")

    if json_data:
        print(f"Unique brands: {len(json_data['brands'])}")
        print(f"Categories: {', '.join(json_data['categories'])}")

        # Brand distribution
        from collections import Counter
        brand_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()

        for article in json_data["articles"]:
            for brand in article["brands"]:
                brand_counts[brand] += 1
            for cat in article["categories"]:
                category_counts[cat] += 1

        print("\nTop 10 brands:")
        for brand, count in brand_counts.most_common(10):
            print(f"  {brand}: {count} articles")

        print("\nCategory distribution:")
        for cat, count in category_counts.most_common():
            print(f"  {cat}: {count} articles")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export ESG news for website display",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["json", "atom", "both"],
        help="Output format: json, atom, or both",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (for single format exports)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="JSON output file path (for --format both)",
    )
    parser.add_argument(
        "--atom-output",
        type=str,
        help="Atom output file path (for --format both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to export",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://frederick-douglas-pearce.github.io",
        help="Base URL for the website (for Atom feed links)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview export without writing files",
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

    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.format == "both":
        if not args.json_output or not args.atom_output:
            logger.error("--json-output and --atom-output required when --format=both")
            return 1
    elif not args.output and not args.dry_run:
        logger.error("-o/--output required unless --dry-run is specified")
        return 1

    # Initialize database
    db.init_db()

    # Query articles and export within session context
    # (relationships require active session)
    logger.info("Querying labeled articles...")
    with db.get_session() as session:
        articles = query_labeled_articles(session, limit=args.limit)

        if not articles:
            logger.warning("No labeled articles found")
            return 0

        logger.info(f"Found {len(articles)} labeled articles")

        # Export based on format (must be within session for lazy-loaded relationships)
        json_data = None
        atom_content = None

        if args.format in ("json", "both"):
            logger.info("Generating JSON export...")
            json_data = export_to_json(articles)

        if args.format in ("atom", "both"):
            logger.info("Generating Atom feed...")
            atom_content = export_to_atom(articles, args.base_url)

    # Write files outside session
    if json_data and not args.dry_run:
        output_path = Path(args.json_output if args.format == "both" else args.output)
        write_json(json_data, output_path)
        print(f"JSON export written to: {output_path}")

    if atom_content and not args.dry_run:
        output_path = Path(args.atom_output if args.format == "both" else args.output)
        write_atom(atom_content, output_path)
        print(f"Atom feed written to: {output_path}")

    # Print summary
    print_summary([], json_data)

    if args.dry_run:
        print("\n[DRY RUN] No files written")

        # Show sample JSON output
        if json_data and json_data["articles"]:
            print("\nSample article JSON:")
            print(json.dumps(json_data["articles"][0], indent=2, default=str)[:2000])

    return 0


if __name__ == "__main__":
    sys.exit(main())
