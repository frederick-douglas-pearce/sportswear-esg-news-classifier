#!/usr/bin/env python3
"""Export training data for ML classifiers.

Supports three classifier use cases:
1. False Positive (FP) Classifier: Is the brand mention about sportswear?
2. ESG Pre-filter: Does the article contain ESG content?
3. ESG Classifier: Multi-label ESG category classification

Usage:
    # Export false positive classifier data
    python scripts/export_training_data.py --dataset fp

    # Export ESG pre-filter data
    python scripts/export_training_data.py --dataset esg-prefilter

    # Export full ESG classifier data
    python scripts/export_training_data.py --dataset esg-labels

    # Export only new data since a date
    python scripts/export_training_data.py --dataset fp --since 2025-01-01

    # Export to specific file
    python scripts/export_training_data.py --dataset fp -o data/fp_data.jsonl
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import and_, or_

from src.data_collection.database import db
from src.data_collection.models import Article, BrandLabel


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def export_false_positive_data(
    session, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Export data for false positive brand classifier.

    Returns one record per article with:
    - article_id, title, content, brands (list), is_sportswear (1/0)

    Positive class (is_sportswear=1): Articles with labeling_status='labeled' or 'skipped'
        - labeled: confirmed sportswear with ESG content
        - skipped: confirmed sportswear but no ESG content
    Negative class (is_sportswear=0): Articles with labeling_status='false_positive'

    This matches the behavior of Claude's labeling pipeline which receives
    an article + brands_mentioned and determines if it's about sportswear.
    """
    records = []

    # Query filters
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)

    # Positive examples: labeled and skipped articles (both confirmed sportswear)
    # - labeled: has ESG content
    # - skipped: no ESG content, but still about sportswear brand
    sportswear_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status.in_(["labeled", "skipped"]))
        .all()
    )

    for article in sportswear_articles:
        records.append({
            "article_id": str(article.id),
            "title": article.title,
            "content": article.full_content or article.description or "",
            "brands": article.brands_mentioned or [],
            "source_name": article.source_name,
            "category": article.category or [],
            "is_sportswear": 1,
            "source": article.labeling_status,
        })

    # Negative examples: false positive articles
    fp_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status == "false_positive")
        .all()
    )

    for article in fp_articles:
        records.append({
            "article_id": str(article.id),
            "title": article.title,
            "content": article.full_content or article.description or "",
            "brands": article.brands_mentioned or [],
            "source_name": article.source_name,
            "category": article.category or [],
            "is_sportswear": 0,
            "source": "false_positive",
            "fp_reason": article.labeling_error,
        })

    return records


def export_esg_prefilter_data(
    session, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Export data for ESG pre-filter classifier.

    Returns records with:
    - article_id, title, content, has_esg (1/0)

    Positive class (has_esg=1): Articles with labeling_status='labeled'
    Negative class (has_esg=0): Articles with labeling_status='skipped'
    """
    records = []

    # Query filters
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)

    # Positive examples: labeled articles (have ESG content)
    labeled_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status == "labeled")
        .all()
    )

    for article in labeled_articles:
        records.append({
            "article_id": str(article.id),
            "title": article.title,
            "content": article.full_content or article.description or "",
            "brands": article.brands_mentioned or [],
            "source_name": article.source_name,
            "category": article.category or [],
            "has_esg": 1,
            "source": "labeled",
        })

    # Negative examples: skipped articles (no ESG content)
    skipped_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status == "skipped")
        .all()
    )

    for article in skipped_articles:
        records.append({
            "article_id": str(article.id),
            "title": article.title,
            "content": article.full_content or article.description or "",
            "brands": article.brands_mentioned or [],
            "source_name": article.source_name,
            "category": article.category or [],
            "has_esg": 0,
            "source": "skipped",
            "skip_reason": article.labeling_error,
        })

    return records


def export_esg_labels_data(
    session, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Export data for full ESG multi-label classifier.

    Returns records with:
    - article_id, title, content, brand
    - environmental, social, governance, digital_transformation (1/0)
    - environmental_sentiment, social_sentiment, etc. (-1/0/1)
    """
    records = []

    # Query filters
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)

    # Get all brand labels with their articles
    labeled_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status == "labeled")
        .all()
    )

    for article in labeled_articles:
        brand_labels = (
            session.query(BrandLabel)
            .filter(BrandLabel.article_id == article.id)
            .all()
        )

        for label in brand_labels:
            records.append({
                "article_id": str(article.id),
                "title": article.title,
                "content": article.full_content or article.description or "",
                "brand": label.brand,
                # Binary labels
                "environmental": 1 if label.environmental else 0,
                "social": 1 if label.social else 0,
                "governance": 1 if label.governance else 0,
                "digital_transformation": 1 if label.digital_transformation else 0,
                # Sentiment labels (-1, 0, 1, or None)
                "environmental_sentiment": label.environmental_sentiment,
                "social_sentiment": label.social_sentiment,
                "governance_sentiment": label.governance_sentiment,
                "digital_sentiment": label.digital_sentiment,
                # Metadata
                "confidence": label.confidence_score,
                "reasoning": label.reasoning,
                "model_version": label.model_version,
            })

    return records


def write_jsonl(records: list[dict], filepath: Path) -> None:
    """Write records to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def print_stats(records: list[dict], dataset: str) -> None:
    """Print dataset statistics."""
    print(f"\n=== {dataset.upper()} Dataset Statistics ===")
    print(f"Total records: {len(records)}")

    if dataset == "fp":
        sportswear = sum(1 for r in records if r["is_sportswear"] == 1)
        not_sportswear = sum(1 for r in records if r["is_sportswear"] == 0)
        print(f"Sportswear articles (positive): {sportswear}")
        print(f"False positive articles (negative): {not_sportswear}")
        print(f"Class ratio: {sportswear}:{not_sportswear}")

        # Brand distribution for false positives
        from collections import Counter
        fp_brands: Counter[str] = Counter()
        for r in records:
            if r["is_sportswear"] == 0:
                for brand in r["brands"]:
                    fp_brands[brand] += 1
        print("\nTop brands in false positive articles:")
        for brand, count in fp_brands.most_common(10):
            print(f"  {brand}: {count}")

    elif dataset == "esg-prefilter":
        has_esg = sum(1 for r in records if r["has_esg"] == 1)
        no_esg = sum(1 for r in records if r["has_esg"] == 0)
        print(f"Has ESG (positive): {has_esg}")
        print(f"No ESG (negative): {no_esg}")
        print(f"Class ratio: {has_esg}:{no_esg}")

    elif dataset == "esg-labels":
        env = sum(1 for r in records if r["environmental"] == 1)
        soc = sum(1 for r in records if r["social"] == 1)
        gov = sum(1 for r in records if r["governance"] == 1)
        dig = sum(1 for r in records if r["digital_transformation"] == 1)
        print(f"Environmental: {env}")
        print(f"Social: {soc}")
        print(f"Governance: {gov}")
        print(f"Digital Transformation: {dig}")

        # Brand distribution
        from collections import Counter
        brands = Counter(r["brand"] for r in records)
        print("\nTop brands:")
        for brand, count in brands.most_common(10):
            print(f"  {brand}: {count}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export training data for ML classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fp", "esg-prefilter", "esg-labels"],
        help="Dataset to export: fp (false positive), esg-prefilter, or esg-labels",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: data/<dataset>_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Only export data created since this date (YYYY-MM-DD)",
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

    # Parse since date if provided
    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d")
        logger.info(f"Filtering data since {args.since}")

    # Initialize database
    db.init_db()

    # Export data based on dataset type
    with db.get_session() as session:
        if args.dataset == "fp":
            records = export_false_positive_data(session, since)
        elif args.dataset == "esg-prefilter":
            records = export_esg_prefilter_data(session, since)
        elif args.dataset == "esg-labels":
            records = export_esg_labels_data(session, since)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            return 1

    if not records:
        logger.warning("No records found to export")
        return 0

    # Print statistics
    print_stats(records, args.dataset)

    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path("data")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"{args.dataset}_{timestamp}.jsonl"

    write_jsonl(records, output_path)
    print(f"\nExported {len(records)} records to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
