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

    # FP: put back rows excluded by default (each flag restores one cohort)
    python scripts/export_training_data.py --dataset fp --include-classifier-labels
    python scripts/export_training_data.py --dataset fp --include-unjudged-skips
    python scripts/export_training_data.py --dataset fp --include-rule-labels

    # FP: owner-attested tier - also keep classifier-only negatives skipped before a date
    python scripts/export_training_data.py --dataset fp --attested-before YYYY-MM-DD

    # Export to specific file
    python scripts/export_training_data.py --dataset fp -o data/fp_data.jsonl
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import and_, or_

from src.data_collection import provenance
from src.data_collection.database import db
from src.data_collection.models import Article, BrandLabel
from src.data_collection.provenance import derive_label_source, human_verified_ids


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class FPCandidate:
    """One article the FP export considers, with its latest FP prediction."""

    article_id: str
    title: str
    content: str
    brands: list[str]
    source_name: str | None
    category: list[str]
    labeling_status: str
    labeling_error: str | None
    fp_action: str | None  # action_taken of the latest FP prediction, None if there is none
    fp_action_at: datetime | None  # created_at of that prediction


@dataclass
class FPExportReport:
    """What the FP export left out, and the composition of what it wrote."""

    excluded_classifier_only: int = 0
    excluded_classifier_only_by_month: Counter = field(default_factory=Counter)
    excluded_unjudged: int = 0
    excluded_unjudged_by_reason: Counter = field(default_factory=Counter)
    excluded_rule: int = 0
    attested_included: int = 0
    attested_before: date | None = None
    positives_by_source: Counter = field(default_factory=Counter)
    negatives_by_source: Counter = field(default_factory=Counter)


class FPExport(NamedTuple):
    records: list[dict[str, Any]]
    report: FPExportReport


def _month(moment: datetime | None) -> str:
    """UTC year-month of a timestamp; a naive timestamp is taken as UTC."""
    if moment is None:
        return "unknown"
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m")


def _before(moment: datetime | None, cutoff: date) -> bool:
    if moment is None:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment < datetime.combine(cutoff, datetime.min.time(), tzinfo=timezone.utc)


def build_fp_dataset(
    candidates: list[FPCandidate],
    include_classifier_labels: bool = False,
    include_unjudged_skips: bool = False,
    include_rule_labels: bool = False,
    attested_before: date | None = None,
) -> FPExport:
    """Build FP training records from candidates, applying the provenance exclusions.

    Every record carries ``label_source`` (see ``src.data_collection.provenance``). By default,
    classifier-only negatives, unjudged skips and keyword-sweep negatives are left out and counted.
    Each ``include_*`` flag puts one cohort back. ``attested_before`` puts back only classifier-only
    negatives whose FP skip predates that date (UTC), marked with ``attested_cutoff``; it does not
    change their ``label_source``.
    """
    report = FPExportReport(attested_before=attested_before)
    records: list[dict[str, Any]] = []
    verified = human_verified_ids((c.article_id, c.labeling_status) for c in candidates)

    for c in candidates:
        source = derive_label_source(
            c.labeling_status,
            c.labeling_error,
            c.fp_action,
            human_verified=c.article_id in verified,
            article_id=c.article_id,
        )
        attested = False
        if source == provenance.FP_CLASSIFIER and not include_classifier_labels:
            if attested_before is not None and _before(c.fp_action_at, attested_before):
                attested = True
                report.attested_included += 1
            else:
                report.excluded_classifier_only += 1
                report.excluded_classifier_only_by_month[_month(c.fp_action_at)] += 1
                continue
        if source == provenance.UNJUDGED and not include_unjudged_skips:
            report.excluded_unjudged += 1
            report.excluded_unjudged_by_reason[c.labeling_error] += 1
            continue
        if source == provenance.RULE and not include_rule_labels:
            report.excluded_rule += 1
            continue

        is_negative = c.labeling_status == provenance.FP_NEGATIVE_STATUS
        record = {
            "article_id": c.article_id,
            "title": c.title,
            "content": c.content,
            "brands": c.brands,
            "source_name": c.source_name,
            "category": c.category,
            "is_sportswear": 0 if is_negative else 1,
            "source": c.labeling_status,
            "label_source": source,
        }
        if is_negative:
            record["fp_reason"] = c.labeling_error
            report.negatives_by_source[source] += 1
        else:
            report.positives_by_source[source] += 1
        if attested:
            record["attested_cutoff"] = attested_before.isoformat()
        records.append(record)

    return FPExport(records, report)


def export_false_positive_data(
    session,
    since: datetime | None = None,
    include_classifier_labels: bool = False,
    include_unjudged_skips: bool = False,
    include_rule_labels: bool = False,
    attested_before: date | None = None,
) -> FPExport:
    """Export data for the false positive brand classifier.

    Positive class (is_sportswear=1): labeling_status 'labeled' or 'skipped'.
    Negative class (is_sportswear=0): labeling_status 'false_positive'.

    Rows whose label nobody but the FP classifier decided, skips that never judged whether a
    brand is sportswear, and keyword-sweep negatives are excluded by default (#154); see
    ``build_fp_dataset``. Every split the notebooks and ``train.py`` make is cut from this export,
    so the exclusion applies to evaluation as well as training.
    """
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)

    latest = provenance.latest_prediction_subquery("fp")
    rows = (
        session.query(Article, latest.c.action_taken, latest.c.created_at)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(base_filter)
        .filter(
            Article.labeling_status.in_(
                [*provenance.FP_POSITIVE_STATUSES, provenance.FP_NEGATIVE_STATUS]
            )
        )
        .all()
    )
    candidates = [
        FPCandidate(
            article_id=str(article.id),
            title=article.title,
            content=article.full_content or article.description or "",
            brands=article.brands_mentioned or [],
            source_name=article.source_name,
            category=article.category or [],
            labeling_status=article.labeling_status,
            labeling_error=article.labeling_error,
            fp_action=fp_action,
            fp_action_at=fp_action_at,
        )
        for article, fp_action, fp_action_at in rows
    ]
    return build_fp_dataset(
        candidates,
        include_classifier_labels=include_classifier_labels,
        include_unjudged_skips=include_unjudged_skips,
        include_rule_labels=include_rule_labels,
        attested_before=attested_before,
    )


def export_esg_prefilter_data(
    session, since: datetime | None = None
) -> list[dict[str, Any]]:
    """Export data for ESG pre-filter classifier.

    Returns records with:
    - article_id, title, content, has_esg (1/0)

    Positive class (has_esg=1): Articles with labeling_status='labeled' AND at least
        one brand_label with an ESG category (environmental, social, governance,
        or digital_transformation = True)
    Negative class (has_esg=0): Articles with labeling_status='skipped' OR
        labeling_status='labeled' but no ESG categories assigned
    """
    records = []

    # Query filters
    base_filter = Article.scrape_status == "success"
    if since:
        base_filter = and_(base_filter, Article.created_at >= since)

    # Get labeled articles
    labeled_articles = (
        session.query(Article)
        .filter(base_filter)
        .filter(Article.labeling_status == "labeled")
        .all()
    )

    for article in labeled_articles:
        # Check if article has any ESG categories assigned
        has_esg_categories = (
            session.query(BrandLabel)
            .filter(BrandLabel.article_id == article.id)
            .filter(
                or_(
                    BrandLabel.environmental == True,
                    BrandLabel.social == True,
                    BrandLabel.governance == True,
                    BrandLabel.digital_transformation == True,
                )
            )
            .first()
            is not None
        )

        records.append({
            "article_id": str(article.id),
            "title": article.title,
            "content": article.full_content or article.description or "",
            "brands": article.brands_mentioned or [],
            "source_name": article.source_name,
            "category": article.category or [],
            "has_esg": 1 if has_esg_categories else 0,
            "source": "labeled" if has_esg_categories else "labeled_no_esg",
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


def print_fp_report(report: FPExportReport) -> None:
    """Print what the FP export left out and the composition of what it wrote."""
    print("\nPositives by label source:")
    for source, count in sorted(report.positives_by_source.items()):
        print(f"  {source}: {count}")
    print("Negatives by label source:")
    for source, count in sorted(report.negatives_by_source.items()):
        print(f"  {source}: {count}")

    print(f"\nExcluded classifier-only negatives: {report.excluded_classifier_only}")
    for month, count in sorted(report.excluded_classifier_only_by_month.items()):
        print(f"  skipped by the FP classifier in {month}: {count}")
    print(f"Excluded unjudged skips: {report.excluded_unjudged}")
    for reason, count in report.excluded_unjudged_by_reason.most_common():
        print(f"  {reason}: {count}")
    print(f"Excluded keyword-sweep negatives: {report.excluded_rule}")
    if report.attested_before is not None:
        print(
            f"Owner-attested tier (classifier-only negatives skipped before "
            f"{report.attested_before.isoformat()}, not per-article human review): "
            f"{report.attested_included} kept"
        )


def print_stats(records: list[dict], dataset: str, report: FPExportReport | None = None) -> None:
    """Print dataset statistics."""
    print(f"\n=== {dataset.upper()} Dataset Statistics ===")
    print(f"Total records: {len(records)}")

    if dataset == "fp":
        sportswear = sum(1 for r in records if r["is_sportswear"] == 1)
        not_sportswear = sum(1 for r in records if r["is_sportswear"] == 0)
        print(f"Sportswear articles (positive): {sportswear}")
        print(f"False positive articles (negative): {not_sportswear}")
        print(f"Class ratio: {sportswear}:{not_sportswear}")
        if report is not None:
            print_fp_report(report)

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
        # Breakdown by source
        from_labeled = sum(1 for r in records if r["source"] == "labeled")
        from_labeled_no_esg = sum(1 for r in records if r["source"] == "labeled_no_esg")
        from_skipped = sum(1 for r in records if r["source"] == "skipped")
        print(f"Has ESG (positive): {has_esg}")
        print(f"  - from labeled: {from_labeled}")
        print(f"No ESG (negative): {no_esg}")
        print(f"  - from labeled_no_esg: {from_labeled_no_esg}")
        print(f"  - from skipped: {from_skipped}")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    fp_group = parser.add_argument_group(
        "fp dataset provenance",
        "Rows excluded from the fp dataset by default, and how to put them back (#154)",
    )
    fp_group.add_argument(
        "--include-classifier-labels",
        action="store_true",
        help="Keep false_positive rows decided by the FP classifier alone",
    )
    fp_group.add_argument(
        "--include-unjudged-skips",
        action="store_true",
        help="Keep skipped rows from paths that never judged sportswear status",
    )
    fp_group.add_argument(
        "--include-rule-labels",
        action="store_true",
        help="Keep false_positive rows written by cleanup_false_positives.py's keyword sweep",
    )
    fp_group.add_argument(
        "--attested-before",
        type=str,
        metavar="YYYY-MM-DD",
        help=(
            "Owner-attested tier: also keep classifier-only negatives skipped before this date "
            "(UTC). Weaker than per-article human review; their label_source stays fp_classifier"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return validate_args(parser, parser.parse_args(argv))


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Reject flag combinations that would not do what they say."""
    fp_flags = [
        name for name in (
            "include_classifier_labels",
            "include_unjudged_skips",
            "include_rule_labels",
            "attested_before",
        )
        if getattr(args, name)
    ]
    if fp_flags and args.dataset != "fp":
        parser.error(f"--{fp_flags[0].replace('_', '-')} applies only to --dataset fp")
    if args.attested_before and args.include_classifier_labels:
        parser.error(
            "--attested-before has no effect with --include-classifier-labels, "
            "which already keeps every classifier-only negative"
        )
    if args.attested_before:
        try:
            args.attested_before = datetime.strptime(args.attested_before, "%Y-%m-%d").date()
        except ValueError:
            parser.error("--attested-before must be YYYY-MM-DD")
    return args


def write_and_report(
    records: list[dict],
    dataset: str,
    output_path: Path,
    report: FPExportReport | None = None,
) -> None:
    """Print statistics, write the file, and print the closing line.

    The closing ``Exported N records to ...`` line must be the last stdout line containing
    "records" or "exported": the model_training workflow reads the record count from the last
    such line (``src/agent/workflows/model_training.py``, ``export_training_data``).
    """
    print_stats(records, dataset, report)
    write_jsonl(records, output_path)
    print(f"\nExported {len(records)} records to {output_path}")


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
    report = None
    with db.get_session() as session:
        if args.dataset == "fp":
            records, report = export_false_positive_data(
                session,
                since,
                include_classifier_labels=args.include_classifier_labels,
                include_unjudged_skips=args.include_unjudged_skips,
                include_rule_labels=args.include_rule_labels,
                attested_before=args.attested_before,
            )
        elif args.dataset == "esg-prefilter":
            records = export_esg_prefilter_data(session, since)
        elif args.dataset == "esg-labels":
            records = export_esg_labels_data(session, since)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            return 1

    if not records:
        if report is not None:
            print_fp_report(report)
        logger.warning("No records found to export")
        return 0

    # Determine output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path("data")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / f"{args.dataset}_{timestamp}.jsonl"

    write_and_report(records, args.dataset, output_path, report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
