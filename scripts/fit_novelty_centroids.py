#!/usr/bin/env python
"""Fit novelty centroids from article text using sentence-transformers.

This script queries articles from the database, computes sentence-transformer
embeddings (all-MiniLM-L6-v2, 384-dim), and fits K-Means clustering to create
reference centroids for novelty detection.

Using sentence-transformers instead of OpenAI embeddings provides:
- Lower dimensionality (384 vs 1536) - better clustering due to curse of dimensionality
- Free, local computation - no API costs
- Can score ALL articles before FP classification (not just labeled ones)

The centroids are saved to the models/ directory and used by the labeling
pipeline to detect novel/out-of-distribution articles.

Usage:
    # Fit centroids from all labeled articles only
    python scripts/fit_novelty_centroids.py

    # Fit with custom number of clusters
    python scripts/fit_novelty_centroids.py --n-clusters 30

    # Fit from all processed articles (labeled + false_positive + skipped)
    # RECOMMENDED: This gives full coverage of topics the classifiers see
    python scripts/fit_novelty_centroids.py --include-all-processed

    # Dry run (show stats without saving)
    python scripts/fit_novelty_centroids.py --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.novelty import NoveltyScorer
from src.labeling.config import labeling_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_article_texts(
    database_url: str,
    days: int | None = None,
    include_all_processed: bool = False,
    limit: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load article texts from database.

    Args:
        database_url: PostgreSQL connection string
        days: Only include articles from the last N days (None = all)
        include_all_processed: Include all processed articles (labeled, false_positive, skipped)
        limit: Maximum number of articles to load

    Returns:
        Tuple of (texts, article_ids)
    """
    engine = create_engine(database_url)

    # Build filters
    filters = ["full_content IS NOT NULL"]
    params = {}

    if not include_all_processed:
        # Only labeled articles (true positives)
        filters.append("labeling_status = 'labeled'")
    else:
        # Include all processed articles: labeled, false_positive, and skipped
        # These are all articles that classifiers have seen/will see
        filters.append("labeling_status IN ('labeled', 'false_positive', 'skipped')")

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filters.append("created_at >= :cutoff")
        params["cutoff"] = cutoff

    where_clause = " AND ".join(filters)
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = text(f"""
        SELECT
            id::text AS article_id,
            title,
            full_content
        FROM articles
        WHERE {where_clause}
        ORDER BY created_at DESC
        {limit_clause}
    """)

    logger.info("Querying articles from database...")

    with engine.connect() as conn:
        result = conn.execute(query, params)
        rows = result.fetchall()

    if not rows:
        raise ValueError("No articles found in database matching criteria")

    # Combine title + content for each article
    texts = []
    article_ids = []
    for row in rows:
        article_ids.append(row.article_id)
        title = row.title or ""
        content = row.full_content or ""
        text_combined = f"{title}\n\n{content}" if title else content
        texts.append(text_combined)

    logger.info(f"Loaded {len(texts)} articles from database")
    return texts, article_ids


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Fit novelty centroids from article text"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=25,
        help="Number of K-Means clusters (default: 25)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only use articles from the last N days (default: all)",
    )
    parser.add_argument(
        "--include-all-processed",
        action="store_true",
        help="Include all processed articles (labeled, false_positive, skipped) for full coverage",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to use (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for centroids file (default: from settings)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence-transformer model (default: from settings)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without saving centroids",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load article texts from database
    try:
        texts, article_ids = load_article_texts(
            database_url=labeling_settings.database_url,
            days=args.days,
            include_all_processed=args.include_all_processed,
            limit=args.limit,
        )
    except ValueError as e:
        logger.error(f"Failed to load articles: {e}")
        sys.exit(1)

    # Initialize novelty scorer with embedding model
    embedding_model = args.embedding_model or labeling_settings.novelty_embedding_model
    scorer = NoveltyScorer(
        n_clusters=args.n_clusters,
        embedding_model=embedding_model,
    )

    # Fit from texts (computes sentence-transformer embeddings internally)
    logger.info(f"Fitting novelty scorer using {embedding_model}...")
    scorer.fit_from_texts(texts, n_clusters=args.n_clusters, show_progress=True)

    # Get metadata and stats
    metadata = scorer.get_metadata()
    logger.info(f"Fitted {metadata['n_clusters']} clusters")
    logger.info(f"Embedding model: {metadata['embedding_model']}")
    logger.info(f"Embedding dimension: {metadata['embedding_dim']}")
    logger.info(f"Silhouette score: {metadata['silhouette_score']:.3f}")
    logger.info(f"Cluster sizes: {metadata['cluster_sizes']}")

    # Compute training novelty statistics (sanity check)
    # Re-compute embeddings for stats (we don't store them from fit_from_texts)
    logger.info("Computing training novelty statistics...")
    embeddings = scorer.embed_texts(texts, show_progress=True)
    training_stats = scorer.get_training_novelty_stats(embeddings)
    logger.info(
        f"Training novelty stats: mean={training_stats['mean']:.3f}, "
        f"std={training_stats['std']:.3f}, "
        f"p90={training_stats['p90']:.3f}, "
        f"p99={training_stats['p99']:.3f}"
    )

    if args.dry_run:
        logger.info("[DRY RUN] Would save centroids, but skipping")
        return

    # Save centroids
    output_path = (
        Path(args.output) if args.output else Path(labeling_settings.novelty_centroids_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scorer.save(output_path)
    logger.info(f"Saved novelty centroids to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("NOVELTY CENTROIDS FITTED SUCCESSFULLY")
    print("=" * 60)
    print(f"Articles used:     {len(article_ids)}")
    print(f"Embedding model:   {metadata['embedding_model']}")
    print(f"Embedding dim:     {metadata['embedding_dim']}")
    print(f"Clusters:          {metadata['n_clusters']}")
    print(f"Silhouette score:  {metadata['silhouette_score']:.3f}")
    print(f"Training novelty:  mean={training_stats['mean']:.3f}, p99={training_stats['p99']:.3f}")
    print(f"Output path:       {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
