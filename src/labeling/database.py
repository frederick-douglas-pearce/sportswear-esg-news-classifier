"""Database operations for the labeling pipeline."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.data_collection.database import Database, db
from src.data_collection.models import (
    Article,
    ArticleChunk,
    BrandLabel,
    LabelEvidence,
    LabelingRun,
)

from .chunker import Chunk
from .evidence_matcher import EvidenceMatch
from .models import BrandAnalysis

logger = logging.getLogger(__name__)


class LabelingDatabase:
    """Database operations for the labeling pipeline."""

    def __init__(self, database: Database | None = None):
        """Initialize with a database connection."""
        self.db = database or db

    def get_articles_pending_labeling(
        self,
        session: Session,
        limit: int = 100,
        require_content: bool = True,
    ) -> list[Article]:
        """Get articles that need to be labeled.

        Args:
            session: Database session
            limit: Maximum articles to return
            require_content: Only return articles with full_content

        Returns:
            List of Article objects pending labeling
        """
        query = session.query(Article).filter(Article.labeling_status == "pending")

        if require_content:
            query = query.filter(Article.full_content.isnot(None))
            query = query.filter(Article.scrape_status == "success")

        return query.order_by(Article.created_at).limit(limit).all()

    def get_article_by_id(self, session: Session, article_id: UUID) -> Article | None:
        """Get a specific article by ID."""
        return session.query(Article).filter(Article.id == article_id).first()

    def save_chunks(
        self,
        session: Session,
        article_id: UUID,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> list[ArticleChunk]:
        """Save article chunks to the database.

        Args:
            session: Database session
            article_id: UUID of the parent article
            chunks: List of Chunk objects from the chunker
            embeddings: Optional list of embeddings for each chunk

        Returns:
            List of created ArticleChunk objects
        """
        # Delete any existing chunks for this article
        session.query(ArticleChunk).filter(
            ArticleChunk.article_id == article_id
        ).delete()

        db_chunks = []
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None

            db_chunk = ArticleChunk(
                article_id=article_id,
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_count=chunk.token_count,
                embedding=embedding,
            )
            session.add(db_chunk)
            db_chunks.append(db_chunk)

        session.flush()
        return db_chunks

    def get_chunks_for_article(
        self, session: Session, article_id: UUID
    ) -> list[ArticleChunk]:
        """Get all chunks for an article, ordered by index."""
        return (
            session.query(ArticleChunk)
            .filter(ArticleChunk.article_id == article_id)
            .order_by(ArticleChunk.chunk_index)
            .all()
        )

    def save_brand_labels(
        self,
        session: Session,
        article_id: UUID,
        brand_analyses: list[BrandAnalysis],
        model_version: str,
        labeled_by: str = "claude-sonnet",
        prompt_version: str | None = None,
    ) -> list[BrandLabel]:
        """Save brand labels from LLM analysis.

        Only saves labels for brands confirmed to be sportswear companies.
        Non-sportswear brands (animals, regions, cars, etc.) are filtered out.

        Args:
            session: Database session
            article_id: UUID of the article
            brand_analyses: List of BrandAnalysis from LLM
            model_version: Model version string
            labeled_by: Labeling source identifier
            prompt_version: Version of prompt template used for labeling

        Returns:
            List of created BrandLabel objects (only sportswear brands)
        """
        # Delete any existing labels for this article
        session.query(BrandLabel).filter(BrandLabel.article_id == article_id).delete()

        db_labels = []
        for analysis in brand_analyses:
            # Skip non-sportswear brands
            if not analysis.is_sportswear_brand:
                logger.info(
                    f"Skipping non-sportswear brand '{analysis.brand}': "
                    f"{analysis.not_sportswear_reason}"
                )
                continue

            # Skip if no categories (shouldn't happen for sportswear brands, but just in case)
            if not analysis.categories:
                logger.warning(
                    f"Sportswear brand '{analysis.brand}' has no categories, skipping"
                )
                continue

            categories = analysis.categories

            label = BrandLabel(
                article_id=article_id,
                brand=analysis.brand,
                environmental=categories["environmental"].applies,
                social=categories["social"].applies,
                governance=categories["governance"].applies,
                digital_transformation=categories["digital_transformation"].applies,
                environmental_sentiment=categories["environmental"].sentiment,
                social_sentiment=categories["social"].sentiment,
                governance_sentiment=categories["governance"].sentiment,
                digital_sentiment=categories["digital_transformation"].sentiment,
                confidence_score=analysis.confidence,
                reasoning=analysis.reasoning,
                labeled_by=labeled_by,
                model_version=model_version,
                prompt_version=prompt_version,
            )
            session.add(label)
            db_labels.append(label)

        session.flush()
        return db_labels

    def save_evidence(
        self,
        session: Session,
        brand_label_id: UUID,
        category: str,
        evidence_matches: list[EvidenceMatch],
    ) -> list[LabelEvidence]:
        """Save evidence excerpts for a brand label.

        Args:
            session: Database session
            brand_label_id: UUID of the parent BrandLabel
            category: ESG category name
            evidence_matches: List of matched evidence

        Returns:
            List of created LabelEvidence objects
        """
        db_evidence = []
        for match in evidence_matches:
            evidence = LabelEvidence(
                brand_label_id=brand_label_id,
                chunk_id=match.chunk_id,
                category=category,
                excerpt=match.excerpt,
                relevance_score=match.similarity_score,
            )
            session.add(evidence)
            db_evidence.append(evidence)

        session.flush()
        return db_evidence

    def update_article_labeling_status(
        self,
        session: Session,
        article_id: UUID,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update the labeling status of an article.

        Args:
            session: Database session
            article_id: UUID of the article
            status: New labeling status
            error: Optional error message
        """
        article = session.query(Article).filter(Article.id == article_id).first()
        if article:
            article.labeling_status = status
            article.labeling_error = error
            if status == "labeled":
                article.labeled_at = datetime.now(timezone.utc)
            elif status == "skipped":
                article.skipped_at = datetime.now(timezone.utc)

    def create_labeling_run(
        self,
        session: Session,
        config: dict[str, Any] | None = None,
        prompt_version: str | None = None,
        prompt_system_hash: str | None = None,
        prompt_user_hash: str | None = None,
    ) -> LabelingRun:
        """Create a new labeling run record.

        Args:
            session: Database session
            config: Run configuration dictionary
            prompt_version: Version of prompt templates used
            prompt_system_hash: SHA256 hash prefix of system prompt
            prompt_user_hash: SHA256 hash prefix of user prompt

        Returns:
            Created LabelingRun object
        """
        run = LabelingRun(
            config=config,
            prompt_version=prompt_version,
            prompt_system_hash=prompt_system_hash,
            prompt_user_hash=prompt_user_hash,
        )
        session.add(run)
        session.flush()
        return run

    def complete_labeling_run(
        self,
        session: Session,
        run: LabelingRun,
        articles_processed: int,
        brands_labeled: int,
        chunks_created: int,
        embeddings_generated: int,
        llm_calls: int,
        input_tokens: int,
        output_tokens: int,
        status: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Update a labeling run with final statistics."""
        run.completed_at = datetime.now(timezone.utc)
        run.articles_processed = articles_processed
        run.brands_labeled = brands_labeled
        run.chunks_created = chunks_created
        run.embeddings_generated = embeddings_generated
        run.llm_calls_made = llm_calls
        run.total_input_tokens = input_tokens
        run.total_output_tokens = output_tokens
        run.status = status
        run.error_message = error_message

        # Calculate estimated cost
        # Claude Sonnet: $3/1M input, $15/1M output
        # OpenAI embeddings: $0.00002/1K tokens
        llm_cost = (input_tokens / 1_000_000) * 3 + (output_tokens / 1_000_000) * 15
        # Rough estimate for embedding tokens (assume similar to chunk count)
        embedding_cost = (embeddings_generated * 500 / 1000) * 0.00002
        run.estimated_cost_usd = llm_cost + embedding_cost

    def get_labeling_stats(self, session: Session) -> dict[str, int]:
        """Get overall labeling statistics."""
        total = session.query(Article).count()
        pending = (
            session.query(Article).filter(Article.labeling_status == "pending").count()
        )
        labeled = (
            session.query(Article).filter(Article.labeling_status == "labeled").count()
        )
        skipped = (
            session.query(Article).filter(Article.labeling_status == "skipped").count()
        )
        false_positive = (
            session.query(Article)
            .filter(Article.labeling_status == "false_positive")
            .count()
        )
        unlabelable = (
            session.query(Article)
            .filter(Article.labeling_status == "unlabelable")
            .count()
        )
        total_labels = session.query(BrandLabel).count()
        total_chunks = session.query(ArticleChunk).count()

        return {
            "total_articles": total,
            "pending_labeling": pending,
            "labeled": labeled,
            "skipped": skipped,
            "false_positive": false_positive,
            "unlabelable": unlabelable,
            "total_brand_labels": total_labels,
            "total_chunks": total_chunks,
        }


# Default instance
labeling_db = LabelingDatabase()
