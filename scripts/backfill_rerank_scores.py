#!/usr/bin/env python3
"""Backfill rerank scores for existing label evidence.

This script processes existing label_evidence records that don't have
rerank_score values, loading the associated chunk text and running
cross-encoder reranking to compute and store the scores.

Usage:
    # Dry run to see what would be updated
    uv run python scripts/backfill_rerank_scores.py --dry-run

    # Process all evidence without rerank scores
    uv run python scripts/backfill_rerank_scores.py

    # Process with custom batch size
    uv run python scripts/backfill_rerank_scores.py --batch-size 100

    # Limit to specific number of records
    uv run python scripts/backfill_rerank_scores.py --limit 500
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func

from src.data_collection.database import db
from src.data_collection.models import ArticleChunk, LabelEvidence
from src.labeling.reranker import CrossEncoderReranker, RerankCandidate


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_evidence_without_rerank_scores(session, limit: int | None = None) -> list[LabelEvidence]:
    """Get evidence records that need rerank scores.

    Args:
        session: Database session
        limit: Maximum records to return

    Returns:
        List of LabelEvidence records without rerank_score
    """
    query = (
        session.query(LabelEvidence)
        .filter(LabelEvidence.rerank_score.is_(None))
        .filter(LabelEvidence.chunk_id.isnot(None))  # Must have associated chunk
    )

    if limit:
        query = query.limit(limit)

    return query.all()


def backfill_rerank_scores(
    session,
    evidence_records: list[LabelEvidence],
    reranker: CrossEncoderReranker,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Backfill rerank scores for evidence records.

    Args:
        session: Database session
        evidence_records: List of evidence records to process
        reranker: Cross-encoder reranker instance
        dry_run: If True, don't commit changes

    Returns:
        Tuple of (processed, updated, failed)
    """
    logger = logging.getLogger(__name__)
    processed = 0
    updated = 0
    failed = 0

    # Group evidence by chunk_id for efficiency
    chunk_ids = {ev.chunk_id for ev in evidence_records if ev.chunk_id}
    chunks_by_id = {}

    if chunk_ids:
        chunks = session.query(ArticleChunk).filter(ArticleChunk.id.in_(chunk_ids)).all()
        chunks_by_id = {c.id: c for c in chunks}

    for evidence in evidence_records:
        processed += 1

        if not evidence.chunk_id or evidence.chunk_id not in chunks_by_id:
            logger.warning(f"Evidence {evidence.id}: No chunk found")
            failed += 1
            continue

        chunk = chunks_by_id[evidence.chunk_id]

        try:
            # Create candidate with initial score
            initial_score = evidence.relevance_score or 0.5
            candidate = RerankCandidate(
                index=0,
                text=chunk.chunk_text,
                initial_score=initial_score,
            )

            # Run reranker
            results = reranker.rerank(evidence.excerpt, [candidate], top_k=1)

            if results:
                rerank_score = results[0].rerank_score

                if not dry_run:
                    evidence.rerank_score = rerank_score
                    # Also set match_method if not already set
                    if not evidence.match_method:
                        # Infer from context - if we have a chunk, it was matched
                        evidence.match_method = "combined"

                logger.debug(
                    f"Evidence {evidence.id}: rerank_score={rerank_score:.3f} "
                    f"(initial={initial_score:.3f})"
                )
                updated += 1
            else:
                logger.warning(f"Evidence {evidence.id}: Reranker returned no results")
                failed += 1

        except Exception as e:
            logger.error(f"Evidence {evidence.id}: Error during reranking: {e}")
            failed += 1

        # Log progress
        if processed % 100 == 0:
            logger.info(f"Processed {processed}/{len(evidence_records)} records")

    return processed, updated, failed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill rerank scores for existing evidence records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Commit after this many updates (default: 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without committing",
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

    # Initialize database
    db.init_db()

    # Initialize reranker (this will lazy-load the model on first use)
    logger.info("Initializing cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    # Get count of records to process
    with db.get_session() as session:
        total_without_rerank = (
            session.query(func.count(LabelEvidence.id))
            .filter(LabelEvidence.rerank_score.is_(None))
            .filter(LabelEvidence.chunk_id.isnot(None))
            .scalar()
        )

    logger.info(f"Found {total_without_rerank} evidence records without rerank_score")

    if total_without_rerank == 0:
        logger.info("Nothing to backfill")
        return 0

    if args.dry_run:
        logger.info("[DRY RUN] No changes will be committed")

    # Process in batches
    total_processed = 0
    total_updated = 0
    total_failed = 0
    batch_num = 0

    limit_remaining = args.limit

    while True:
        batch_num += 1
        batch_limit = args.batch_size
        if limit_remaining is not None:
            batch_limit = min(batch_limit, limit_remaining)

        if batch_limit <= 0:
            break

        with db.get_session() as session:
            # Get batch of records
            evidence_records = get_evidence_without_rerank_scores(session, limit=batch_limit)

            if not evidence_records:
                break

            logger.info(f"Batch {batch_num}: Processing {len(evidence_records)} records")

            # Process batch
            processed, updated, failed = backfill_rerank_scores(
                session,
                evidence_records,
                reranker,
                dry_run=args.dry_run,
            )

            total_processed += processed
            total_updated += updated
            total_failed += failed

            if not args.dry_run:
                session.commit()
                logger.info(f"Batch {batch_num}: Committed {updated} updates")

            if limit_remaining is not None:
                limit_remaining -= processed

    # Print summary
    print(f"\n{'=' * 60}")
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {total_processed}")
    print(f"Successfully updated:    {total_updated}")
    print(f"Failed:                  {total_failed}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were committed")

    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
