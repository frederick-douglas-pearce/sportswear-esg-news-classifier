#!/usr/bin/env python3
"""Re-chunk labeled articles with new smaller chunk sizes.

This migration script re-processes all labeled articles with the new
chunk size parameters (200 target tokens instead of 500), generating
new embeddings and re-matching existing evidence to the new chunks.

This does NOT require LLM relabeling - it only:
1. Deletes existing article_chunks
2. Re-chunks article text with new smaller sizes
3. Generates new embeddings (~$0.50-$2.00 for 2000 articles)
4. Re-matches existing label_evidence to new chunks
5. Updates relevance_score based on new matching

Usage:
    # Preview what will be processed
    uv run python scripts/rechunk_articles.py --dry-run

    # Process all labeled articles
    uv run python scripts/rechunk_articles.py

    # Process specific article
    uv run python scripts/rechunk_articles.py --article-id UUID

    # Process in batches with progress
    uv run python scripts/rechunk_articles.py --batch-size 50 --verbose
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import delete, select
from sqlalchemy.orm import joinedload

from src.data_collection.database import db
from src.data_collection.models import Article, ArticleChunk, BrandLabel, LabelEvidence
from src.labeling.chunker import ArticleChunker
from src.labeling.embedder import OpenAIEmbedder
from src.labeling.evidence_matcher import EvidenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_labeled_articles(session, article_id: UUID | None = None) -> list[Article]:
    """Query labeled articles with brand labels and evidence loaded.

    Args:
        session: Database session
        article_id: Optional specific article ID

    Returns:
        List of Article objects
    """
    query = (
        session.query(Article)
        .filter(Article.labeling_status == "labeled")
        .options(
            joinedload(Article.brand_labels)
            .joinedload(BrandLabel.evidence)
        )
    )

    if article_id:
        query = query.filter(Article.id == article_id)

    return query.all()


def delete_article_chunks(session, article_id: UUID) -> int:
    """Delete existing chunks for an article.

    Args:
        session: Database session
        article_id: Article UUID

    Returns:
        Number of chunks deleted
    """
    result = session.execute(
        delete(ArticleChunk).where(ArticleChunk.article_id == article_id)
    )
    return result.rowcount


def rechunk_article(
    session,
    article: Article,
    chunker: ArticleChunker,
    embedder: OpenAIEmbedder,
    matcher: EvidenceMatcher,
    dry_run: bool = False,
) -> dict:
    """Re-chunk a single article and update evidence matching.

    Args:
        session: Database session
        article: Article to process
        chunker: Chunker with new settings
        embedder: OpenAI embedder for generating embeddings
        matcher: Evidence matcher for re-matching
        dry_run: If True, don't persist changes

    Returns:
        Dict with processing statistics
    """
    stats = {
        "article_id": str(article.id),
        "old_chunks": 0,
        "new_chunks": 0,
        "evidence_updated": 0,
        "embedding_tokens": 0,
    }

    # Get article content
    content = article.full_content or article.description or ""
    if not content:
        logger.warning(f"Article {article.id} has no content to chunk")
        return stats

    # Count and delete old chunks
    old_chunks = session.query(ArticleChunk).filter(
        ArticleChunk.article_id == article.id
    ).all()
    stats["old_chunks"] = len(old_chunks)

    if not dry_run:
        delete_article_chunks(session, article.id)

    # Create new chunks with smaller sizes
    chunks = chunker.chunk_article(content)
    stats["new_chunks"] = len(chunks)

    if not chunks:
        logger.warning(f"No chunks created for article {article.id}")
        return stats

    # Generate embeddings for new chunks
    chunk_texts = [c.text for c in chunks]
    if not dry_run:
        result = embedder.embed_batch(chunk_texts)
        embeddings = result.embeddings
        stats["embedding_tokens"] = result.total_tokens
    else:
        embeddings = [[0.0] * 1536] * len(chunks)  # Placeholder for dry run
        stats["embedding_tokens"] = sum(chunker.count_tokens(t) for t in chunk_texts)

    # Create new ArticleChunk records
    new_chunk_ids = []
    for chunk, embedding in zip(chunks, embeddings):
        if not dry_run:
            chunk_record = ArticleChunk(
                article_id=article.id,
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_count=chunk.token_count,
                embedding=embedding,
            )
            session.add(chunk_record)
            session.flush()  # Get the ID
            new_chunk_ids.append(chunk_record.id)
        else:
            new_chunk_ids.append(None)

    # Re-match existing evidence to new chunks
    for brand_label in article.brand_labels:
        for evidence in brand_label.evidence:
            # Find best matching chunk for this evidence
            matches = matcher.match_evidence_to_chunks(
                [evidence.excerpt],
                chunks,
                new_chunk_ids if not dry_run else None,
                embeddings,
            )

            if matches and matches[0].match_method != "none":
                match = matches[0]
                if not dry_run:
                    evidence.chunk_id = match.chunk_id
                    evidence.relevance_score = match.similarity_score
                stats["evidence_updated"] += 1

    return stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-chunk labeled articles with new smaller chunk sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--article-id",
        type=str,
        help="Process specific article by UUID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Articles per batch (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize components
    logger.info("Initializing chunker, embedder, and matcher...")
    chunker = ArticleChunker()  # Uses new defaults from config
    embedder = OpenAIEmbedder() if not args.dry_run else None
    matcher = EvidenceMatcher(embedder=embedder)

    logger.info(f"Chunk settings: target={chunker.target_tokens}, "
                f"max={chunker.max_tokens}, min={chunker.min_tokens}, "
                f"short_article_threshold={chunker.short_article_threshold}")

    # Initialize database
    db.init_db()

    # Track totals
    total_articles = 0
    total_old_chunks = 0
    total_new_chunks = 0
    total_evidence = 0
    total_tokens = 0

    with db.get_session() as session:
        # Query articles
        article_id = UUID(args.article_id) if args.article_id else None
        articles = get_labeled_articles(session, article_id)

        if not articles:
            logger.warning("No labeled articles found")
            return 0

        logger.info(f"Found {len(articles)} labeled articles to process")

        if args.dry_run:
            logger.info("[DRY RUN] No changes will be persisted")

        # Process in batches
        for i, article in enumerate(articles):
            try:
                stats = rechunk_article(
                    session,
                    article,
                    chunker,
                    embedder,
                    matcher,
                    dry_run=args.dry_run,
                )

                total_articles += 1
                total_old_chunks += stats["old_chunks"]
                total_new_chunks += stats["new_chunks"]
                total_evidence += stats["evidence_updated"]
                total_tokens += stats["embedding_tokens"]

                if args.verbose or (i + 1) % 10 == 0:
                    logger.info(
                        f"[{i + 1}/{len(articles)}] Article {str(article.id)[:8]}... "
                        f"chunks: {stats['old_chunks']} -> {stats['new_chunks']}, "
                        f"evidence: {stats['evidence_updated']}"
                    )

                # Commit in batches
                if not args.dry_run and (i + 1) % args.batch_size == 0:
                    session.commit()
                    logger.info(f"Committed batch of {args.batch_size} articles")

            except Exception as e:
                logger.error(f"Error processing article {article.id}: {e}")
                if not args.dry_run:
                    session.rollback()
                continue

        # Final commit
        if not args.dry_run:
            session.commit()

    # Print summary
    print("\n" + "=" * 60)
    print("RECHUNKING SUMMARY")
    print("=" * 60)
    print(f"\nArticles processed: {total_articles}")
    print(f"Old chunks deleted: {total_old_chunks}")
    print(f"New chunks created: {total_new_chunks}")
    print(f"Chunk ratio: {total_new_chunks / max(total_old_chunks, 1):.2f}x")
    print(f"Evidence re-matched: {total_evidence}")
    print(f"Embedding tokens: {total_tokens:,}")

    # Estimate cost (OpenAI text-embedding-3-small: $0.00002 per 1K tokens)
    estimated_cost = (total_tokens / 1000) * 0.00002
    print(f"Estimated embedding cost: ${estimated_cost:.4f}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
