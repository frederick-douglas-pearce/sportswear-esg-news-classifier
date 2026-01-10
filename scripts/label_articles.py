#!/usr/bin/env python3
"""CLI script for LLM-based article labeling.

Usage:
    python scripts/label_articles.py [OPTIONS]

Examples:
    # Dry run to test without saving
    python scripts/label_articles.py --dry-run --batch-size 5

    # Label a batch of articles
    python scripts/label_articles.py --batch-size 10

    # Label a specific article
    python scripts/label_articles.py --article-id 12345678-1234-1234-1234-123456789abc

    # Check current labeling statistics
    python scripts/label_articles.py --stats

    # List available prompt versions
    python scripts/label_articles.py --list-prompts

    # Use a specific prompt version
    python scripts/label_articles.py --prompt-version v1.1.0 --dry-run --batch-size 5
"""

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling.config import labeling_settings
from src.labeling.database import labeling_db
from src.labeling.pipeline import LabelingPipeline
from src.labeling.prompt_manager import prompt_manager


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Label articles with ESG categories using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=labeling_settings.labeling_batch_size,
        help=f"Number of articles to process (default: {labeling_settings.labeling_batch_size})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without saving to database",
    )
    parser.add_argument(
        "--article-id",
        type=str,
        help="Label a specific article by UUID",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking for articles that already have chunks",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding generation (faster but no semantic evidence matching)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show labeling statistics and exit",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        help="Prompt version to use (default: production version from registry)",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt versions and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    return parser.parse_args()


def show_stats() -> None:
    """Display current labeling statistics."""
    from src.data_collection.database import db

    db.init_db()

    with labeling_db.db.get_session() as session:
        stats = labeling_db.get_labeling_stats(session)

    print("\n=== Labeling Statistics ===")
    print(f"Total articles:        {stats['total_articles']}")
    print(f"Pending labeling:      {stats['pending_labeling']}")
    print(f"Labeled:               {stats['labeled']}")
    print(f"Skipped:               {stats['skipped']}")
    print(f"False positives:       {stats['false_positive']}")
    print(f"Unlabelable:           {stats['unlabelable']}")
    print(f"Total brand labels:    {stats['total_brand_labels']}")
    print(f"Total chunks:          {stats['total_chunks']}")
    print()


def list_prompts() -> None:
    """List available prompt versions."""
    try:
        versions = prompt_manager.list_versions()
        production = prompt_manager.get_production_version()

        print("\n=== Available Prompt Versions ===")
        print()
        for version in versions:
            info = prompt_manager.get_version_info(version)
            prod_marker = " (production)" if version == production else ""
            print(f"  {version}{prod_marker}")
            if info.get("commit_message"):
                print(f"    {info['commit_message']}")
            if info.get("created_at"):
                print(f"    Created: {info['created_at'][:10]}")
            print()

    except FileNotFoundError:
        print("\nNo prompt registry found. Using hardcoded prompts.")
        print()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # List prompts and exit
    if args.list_prompts:
        list_prompts()
        return 0

    # Show stats and exit
    if args.stats:
        show_stats()
        return 0

    # Check for required API keys
    if not labeling_settings.anthropic_api_key:
        logger.error(
            "ANTHROPIC_API_KEY not set. Please set it in .env or environment."
        )
        return 1

    if not args.skip_embedding and not labeling_settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY not set. Embedding will be skipped. "
            "Set it in .env or use --skip-embedding flag."
        )
        args.skip_embedding = True

    # Parse article ID if provided
    article_ids = None
    if args.article_id:
        try:
            article_ids = [UUID(args.article_id)]
        except ValueError:
            logger.error(f"Invalid article ID format: {args.article_id}")
            return 1

    logger.info("Starting ESG Article Labeling")
    if args.dry_run:
        logger.info("[DRY RUN] No changes will be saved to database")

    # Run the pipeline
    pipeline = LabelingPipeline(prompt_version=args.prompt_version)

    try:
        stats = pipeline.label_articles(
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            article_ids=article_ids,
            skip_chunking=args.skip_chunking,
            skip_embedding=args.skip_embedding,
        )

        # Print results
        print("\n=== Labeling Results ===")

        # Show prompt version info
        pipeline_stats = pipeline.get_stats()
        if "labeler" in pipeline_stats:
            labeler_stats = pipeline_stats["labeler"]
            pv = labeler_stats.get("prompt_version", "unknown")
            ps_hash = labeler_stats.get("prompt_system_hash", "")[:8]
            pu_hash = labeler_stats.get("prompt_user_hash", "")[:8]
            print(f"Prompt version:         {pv} (system: {ps_hash}, user: {pu_hash})")

        print(f"Articles processed:     {stats.articles_processed}")
        print(f"Articles labeled:       {stats.articles_labeled}")
        print(f"Articles skipped:       {stats.articles_skipped}")
        print(f"False positives:        {stats.articles_false_positive}")
        print(f"Articles failed:        {stats.articles_failed}")
        print(f"Brand labels created:   {stats.brands_labeled}")
        if stats.false_positive_brands > 0:
            print(f"Non-sportswear brands:  {stats.false_positive_brands}")
        print(f"Chunks created:         {stats.chunks_created}")
        print(f"Embeddings generated:   {stats.embeddings_generated}")
        print(f"LLM API calls:          {stats.llm_calls}")
        print(f"Input tokens:           {stats.input_tokens}")
        print(f"Output tokens:          {stats.output_tokens}")

        # FP Classifier stats
        if stats.fp_classifier_calls > 0:
            print("\n=== FP Classifier Pre-filter ===")
            print(f"FP classifier calls:    {stats.fp_classifier_calls}")
            print(f"Skipped LLM:            {stats.fp_classifier_skipped}")
            print(f"Continued to LLM:       {stats.fp_classifier_continued}")
            if stats.fp_classifier_errors > 0:
                print(f"Classifier errors:      {stats.fp_classifier_errors}")
            # Calculate cost savings estimate
            if stats.fp_classifier_skipped > 0:
                # Estimate ~1500 input tokens and ~500 output tokens per skipped article
                saved_input = stats.fp_classifier_skipped * 1500
                saved_output = stats.fp_classifier_skipped * 500
                saved_input_cost = (saved_input / 1_000_000) * 3.00
                saved_output_cost = (saved_output / 1_000_000) * 15.00
                saved_cost = saved_input_cost + saved_output_cost
                print(f"Est. LLM cost saved:    ${saved_cost:.4f}")

        # Cost estimate
        if stats.input_tokens > 0 or stats.output_tokens > 0:
            input_cost = (stats.input_tokens / 1_000_000) * 3.00
            output_cost = (stats.output_tokens / 1_000_000) * 15.00
            embedding_cost = (stats.embeddings_generated * 500 / 1000) * 0.00002
            total_cost = input_cost + output_cost + embedding_cost
            print(f"Estimated cost:         ${total_cost:.4f}")

        if stats.errors:
            print(f"\nErrors ({len(stats.errors)}):")
            for error in stats.errors[:5]:
                print(f"  - {error}")
            if len(stats.errors) > 5:
                print(f"  ... and {len(stats.errors) - 5} more")

        return 0 if not stats.errors else 1

    except Exception as e:
        logger.error(f"Labeling failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
