"""Pipeline orchestration for LLM-based article labeling."""

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from src.data_collection.database import db
from src.data_collection.models import Article

from .chunker import ArticleChunker, Chunk
from .config import labeling_settings
from .database import LabelingDatabase, labeling_db
from .embedder import OpenAIEmbedder
from .evidence_matcher import EvidenceMatcher, match_all_evidence
from .labeler import ArticleLabeler
from .models import LabelingResponse

logger = logging.getLogger(__name__)


@dataclass
class LabelingStats:
    """Statistics from a labeling run."""

    articles_processed: int = 0
    articles_labeled: int = 0
    articles_skipped: int = 0
    articles_failed: int = 0
    brands_labeled: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


class LabelingPipeline:
    """Orchestrates the full article labeling workflow.

    Pipeline steps:
    1. Fetch pending articles from database
    2. Chunk articles into semantic units
    3. Generate embeddings for chunks
    4. Send to Claude for ESG classification
    5. Match evidence to chunks
    6. Save labels and evidence to database
    """

    def __init__(
        self,
        database: LabelingDatabase | None = None,
        chunker: ArticleChunker | None = None,
        embedder: OpenAIEmbedder | None = None,
        labeler: ArticleLabeler | None = None,
    ):
        """Initialize the pipeline.

        Args:
            database: Labeling database operations
            chunker: Article chunker
            embedder: OpenAI embedder
            labeler: Claude article labeler
        """
        self.database = database or labeling_db
        self.chunker = chunker or ArticleChunker()
        self.embedder = embedder
        self.labeler = labeler

        # Lazy initialization of API clients
        self._embedder_initialized = embedder is not None
        self._labeler_initialized = labeler is not None

    def _ensure_embedder(self) -> OpenAIEmbedder:
        """Ensure embedder is initialized."""
        if not self._embedder_initialized:
            self.embedder = OpenAIEmbedder()
            self._embedder_initialized = True
        return self.embedder

    def _ensure_labeler(self) -> ArticleLabeler:
        """Ensure labeler is initialized."""
        if not self._labeler_initialized:
            self.labeler = ArticleLabeler()
            self._labeler_initialized = True
        return self.labeler

    def label_articles(
        self,
        batch_size: int | None = None,
        dry_run: bool = False,
        article_ids: list[UUID] | None = None,
        skip_chunking: bool = False,
        skip_embedding: bool = False,
    ) -> LabelingStats:
        """Run the full labeling pipeline.

        Args:
            batch_size: Number of articles to process (default: from settings)
            dry_run: If True, don't save to database
            article_ids: Specific articles to label (overrides batch_size)
            skip_chunking: Skip chunking for already-chunked articles
            skip_embedding: Skip embedding generation

        Returns:
            LabelingStats with results
        """
        batch_size = batch_size or labeling_settings.labeling_batch_size
        stats = LabelingStats()

        db.init_db()

        # Create labeling run record
        run = None
        if not dry_run:
            with self.database.db.get_session() as session:
                run = self.database.create_labeling_run(
                    session,
                    config={
                        "batch_size": batch_size,
                        "skip_chunking": skip_chunking,
                        "skip_embedding": skip_embedding,
                        "article_ids": [str(id) for id in article_ids]
                        if article_ids
                        else None,
                    },
                )
                run_id = run.id

        try:
            # Get articles to process
            with self.database.db.get_session() as session:
                if article_ids:
                    articles = [
                        self.database.get_article_by_id(session, aid)
                        for aid in article_ids
                    ]
                    articles = [a for a in articles if a is not None]
                else:
                    articles = self.database.get_articles_pending_labeling(
                        session, limit=batch_size
                    )

                # Detach articles from session for processing
                article_data = [
                    {
                        "id": a.id,
                        "title": a.title,
                        "full_content": a.full_content,
                        "description": a.description,
                        "brands_mentioned": a.brands_mentioned or [],
                        "published_at": a.published_at,
                        "source_name": a.source_name,
                    }
                    for a in articles
                ]

            logger.info(f"Processing {len(article_data)} articles")

            for article in article_data:
                try:
                    result = self._process_article(
                        article,
                        dry_run=dry_run,
                        skip_chunking=skip_chunking,
                        skip_embedding=skip_embedding,
                    )

                    stats.articles_processed += 1

                    if result["labeled"]:
                        stats.articles_labeled += 1
                        stats.brands_labeled += result["brands_count"]
                    elif result["skipped"]:
                        stats.articles_skipped += 1
                    else:
                        stats.articles_failed += 1
                        if result.get("error"):
                            stats.errors.append(
                                f"Article {article['id']}: {result['error']}"
                            )

                    stats.chunks_created += result.get("chunks_count", 0)
                    stats.embeddings_generated += result.get("embeddings_count", 0)
                    stats.llm_calls += result.get("llm_calls", 0)
                    stats.input_tokens += result.get("input_tokens", 0)
                    stats.output_tokens += result.get("output_tokens", 0)

                except Exception as e:
                    logger.error(f"Failed to process article {article['id']}: {e}")
                    stats.articles_failed += 1
                    stats.errors.append(f"Article {article['id']}: {str(e)}")

            # Complete labeling run
            if not dry_run and run:
                with self.database.db.get_session() as session:
                    run = session.query(type(run)).get(run_id)
                    if run:
                        self.database.complete_labeling_run(
                            session,
                            run,
                            stats.articles_processed,
                            stats.brands_labeled,
                            stats.chunks_created,
                            stats.embeddings_generated,
                            stats.llm_calls,
                            stats.input_tokens,
                            stats.output_tokens,
                            status="success" if not stats.errors else "partial",
                        )

            logger.info(
                f"Labeling complete: {stats.articles_labeled} labeled, "
                f"{stats.articles_skipped} skipped, {stats.articles_failed} failed"
            )

        except Exception as e:
            logger.error(f"Labeling pipeline failed: {e}")
            if not dry_run and run:
                with self.database.db.get_session() as session:
                    run = session.query(type(run)).get(run_id)
                    if run:
                        self.database.complete_labeling_run(
                            session,
                            run,
                            stats.articles_processed,
                            stats.brands_labeled,
                            stats.chunks_created,
                            stats.embeddings_generated,
                            stats.llm_calls,
                            stats.input_tokens,
                            stats.output_tokens,
                            status="failed",
                            error_message=str(e),
                        )
            raise

        return stats

    def _process_article(
        self,
        article: dict[str, Any],
        dry_run: bool = False,
        skip_chunking: bool = False,
        skip_embedding: bool = False,
    ) -> dict[str, Any]:
        """Process a single article through the labeling pipeline.

        Returns:
            Dict with processing results
        """
        article_id = article["id"]
        content = article["full_content"] or article["description"] or ""
        brands = article["brands_mentioned"]

        result = {
            "labeled": False,
            "skipped": False,
            "error": None,
            "brands_count": 0,
            "chunks_count": 0,
            "embeddings_count": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Check for content
        if not content or len(content.strip()) < 100:
            logger.warning(f"Article {article_id} has insufficient content")
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "skipped", "Insufficient content"
                    )
            result["skipped"] = True
            return result

        # Check for brands
        if not brands:
            logger.warning(f"Article {article_id} has no brands mentioned")
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "skipped", "No brands mentioned"
                    )
            result["skipped"] = True
            return result

        # Step 1: Chunk the article
        chunks: list[Chunk] = []
        chunk_ids: list[UUID] = []
        chunk_embeddings: list[list[float]] = []

        if not skip_chunking:
            chunks = self.chunker.chunk_article(content)
            result["chunks_count"] = len(chunks)
            logger.debug(f"Created {len(chunks)} chunks for article {article_id}")

            # Step 2: Generate embeddings
            if not skip_embedding and chunks:
                embedder = self._ensure_embedder()
                texts = [chunk.text for chunk in chunks]
                emb_result = embedder.embed_batch(texts)
                chunk_embeddings = emb_result.embeddings
                result["embeddings_count"] = len(chunk_embeddings)

            # Save chunks to database
            if not dry_run and chunks:
                with self.database.db.get_session() as session:
                    db_chunks = self.database.save_chunks(
                        session, article_id, chunks, chunk_embeddings or None
                    )
                    chunk_ids = [c.id for c in db_chunks]

                    # Update status
                    status = "embedded" if chunk_embeddings else "chunked"
                    self.database.update_article_labeling_status(
                        session, article_id, status
                    )

        # Step 3: Label with Claude
        labeler = self._ensure_labeler()
        label_result = labeler.label_article(
            title=article["title"],
            content=content,
            brands=brands,
            published_at=article["published_at"],
            source_name=article["source_name"],
        )

        result["llm_calls"] = 1
        result["input_tokens"] = label_result.input_tokens
        result["output_tokens"] = label_result.output_tokens

        if not label_result.success:
            logger.warning(f"Labeling failed for article {article_id}: {label_result.error}")
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "failed", label_result.error
                    )
            result["error"] = label_result.error
            return result

        response = label_result.response
        if not response or not response.brand_analyses:
            logger.warning(f"No brand analyses returned for article {article_id}")
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "skipped", "No ESG content found"
                    )
            result["skipped"] = True
            return result

        # Step 4: Match evidence to chunks (if we have chunks)
        evidence_matches = {}
        if chunks:
            evidence_matches = match_all_evidence(
                response.brand_analyses,
                chunks,
                chunk_ids if chunk_ids else None,
                chunk_embeddings if chunk_embeddings else None,
                self.embedder,
            )

        # Step 5: Save labels and evidence
        if dry_run:
            logger.info(
                f"[DRY RUN] Would save {len(response.brand_analyses)} brand labels "
                f"for article {article_id}"
            )
            for analysis in response.brand_analyses:
                cats = analysis.get_applicable_categories()
                logger.info(
                    f"  {analysis.brand}: {', '.join(cats) if cats else 'no categories'}"
                )
        else:
            with self.database.db.get_session() as session:
                # Save brand labels
                db_labels = self.database.save_brand_labels(
                    session,
                    article_id,
                    response.brand_analyses,
                    model_version=label_result.model,
                )

                # Save evidence for each label
                for db_label in db_labels:
                    brand_evidence = evidence_matches.get(db_label.brand, {})
                    for category, matches in brand_evidence.items():
                        self.database.save_evidence(
                            session, db_label.id, category, matches
                        )

                # Update article status
                self.database.update_article_labeling_status(
                    session, article_id, "labeled"
                )

        result["labeled"] = True
        result["brands_count"] = len(response.brand_analyses)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get current pipeline statistics including API usage."""
        stats = {}

        if self._labeler_initialized and self.labeler:
            stats["labeler"] = self.labeler.get_stats()

        if self._embedder_initialized and self.embedder:
            stats["embedder"] = self.embedder.get_stats()

        with self.database.db.get_session() as session:
            stats["database"] = self.database.get_labeling_stats(session)

        return stats
