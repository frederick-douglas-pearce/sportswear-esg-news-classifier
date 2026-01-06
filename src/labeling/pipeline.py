"""Pipeline orchestration for LLM-based article labeling."""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID

from src.data_collection.database import db
from src.data_collection.models import Article, ClassifierPrediction

from .chunker import ArticleChunker, Chunk
from .classifier_client import ClassifierClient, ClassifierPredictionRecord, FPPredictionResult
from .config import labeling_settings
from .database import LabelingDatabase, labeling_db
from .embedder import OpenAIEmbedder
from .evidence_matcher import EvidenceMatcher, match_all_evidence
from .labeler import ArticleLabeler
from .models import LabelingResponse

logger = logging.getLogger(__name__)

# Title similarity threshold for deduplication (0.0-1.0)
# 0.9 means titles must be 90% similar to be considered duplicates
TITLE_SIMILARITY_THRESHOLD = 0.90


@dataclass
class LabelingStats:
    """Statistics from a labeling run."""

    articles_processed: int = 0
    articles_labeled: int = 0
    articles_skipped: int = 0
    articles_false_positive: int = 0
    articles_failed: int = 0
    articles_deduplicated: int = 0  # Articles skipped as title duplicates
    brands_labeled: int = 0
    false_positive_brands: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    # FP classifier stats
    fp_classifier_calls: int = 0
    fp_classifier_skipped: int = 0
    fp_classifier_continued: int = 0
    fp_classifier_errors: int = 0


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
        fp_client: ClassifierClient | None = None,
    ):
        """Initialize the pipeline.

        Args:
            database: Labeling database operations
            chunker: Article chunker
            embedder: OpenAI embedder
            labeler: Claude article labeler
            fp_client: FP classifier HTTP client
        """
        self.database = database or labeling_db
        self.chunker = chunker or ArticleChunker()
        self.embedder = embedder
        self.labeler = labeler
        self.fp_client = fp_client

        # Lazy initialization of API clients
        self._embedder_initialized = embedder is not None
        self._labeler_initialized = labeler is not None
        self._fp_client_initialized = fp_client is not None

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

    def _ensure_fp_client(self) -> ClassifierClient | None:
        """Ensure FP classifier client is initialized if enabled.

        Returns:
            ClassifierClient if FP classifier is enabled, None otherwise.
        """
        if not labeling_settings.fp_classifier_enabled:
            return None

        if not self._fp_client_initialized:
            self.fp_client = ClassifierClient(
                base_url=labeling_settings.fp_classifier_url,
                timeout=labeling_settings.fp_classifier_timeout,
            )
            self._fp_client_initialized = True
        return self.fp_client

    def _deduplicate_by_title(
        self,
        articles: list[dict[str, Any]],
        threshold: float = TITLE_SIMILARITY_THRESHOLD,
        dry_run: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deduplicate articles by title similarity using difflib.

        Uses SequenceMatcher to find articles with similar titles (likely the same
        story from different sources). Keeps the first occurrence and marks
        duplicates as skipped.

        Args:
            articles: List of article dicts with id, title, etc.
            threshold: Similarity ratio (0.0-1.0) above which titles are duplicates
            dry_run: If True, don't update database

        Returns:
            Tuple of (unique_articles, duplicate_articles)
        """
        if not articles or len(articles) <= 1:
            return articles, []

        unique_articles: list[dict[str, Any]] = []
        duplicate_articles: list[dict[str, Any]] = []
        seen_titles: list[tuple[str, UUID]] = []  # (normalized_title, article_id)

        def normalize_title(title: str) -> str:
            """Normalize title for comparison."""
            if not title:
                return ""
            # Lowercase and strip whitespace
            return " ".join(title.lower().split())

        for article in articles:
            article_id = article["id"]
            title = article.get("title", "")
            normalized = normalize_title(title)

            if not normalized:
                # No title - keep the article
                unique_articles.append(article)
                continue

            # Check against seen titles
            is_duplicate = False
            duplicate_of = None

            for seen_title, seen_id in seen_titles:
                ratio = SequenceMatcher(None, normalized, seen_title).ratio()
                if ratio >= threshold:
                    is_duplicate = True
                    duplicate_of = seen_id
                    logger.info(
                        f"Title duplicate detected: '{title[:50]}...' "
                        f"(similarity: {ratio:.2f}) - duplicate of article {seen_id}"
                    )
                    break

            if is_duplicate:
                duplicate_articles.append(article)

                # Mark as deduplicated in database (distinct from 'skipped' to exclude from exports)
                if not dry_run:
                    with self.database.db.get_session() as session:
                        skip_reason = f"Duplicate title (similar to article {duplicate_of})"
                        self.database.update_article_labeling_status(
                            session, article_id, "deduplicated", skip_reason
                        )
            else:
                unique_articles.append(article)
                seen_titles.append((normalized, article_id))

        if duplicate_articles:
            logger.info(
                f"Title deduplication: {len(unique_articles)} unique, "
                f"{len(duplicate_articles)} duplicates skipped"
            )

        return unique_articles, duplicate_articles

    def _run_fp_prefilter_batch(
        self,
        articles: list[dict[str, Any]],
        dry_run: bool = False,
    ) -> dict[UUID, tuple[bool, ClassifierPredictionRecord | None]]:
        """Run FP classifier pre-filter on a batch of articles.

        Args:
            articles: List of article dicts with id, title, full_content, etc.
            dry_run: If True, don't save predictions to database

        Returns:
            Dict mapping article_id to (should_continue, prediction_record)
        """
        fp_client = self._ensure_fp_client()
        if fp_client is None:
            # FP classifier not enabled - all articles continue
            return {article["id"]: (True, None) for article in articles}

        if not articles:
            return {}

        # Prepare batch request
        batch_articles = []
        article_ids = []
        for article in articles:
            article_id = article["id"]
            article_ids.append(article_id)

            content = article.get("full_content") or article.get("description") or ""
            category = article.get("category") or []
            if isinstance(category, str):
                category = [category]

            batch_articles.append({
                "title": article.get("title", ""),
                "content": content,
                "brands": article.get("brands_mentioned") or [],
                "source_name": article.get("source_name"),
                "category": category,
            })

        results: dict[UUID, tuple[bool, ClassifierPredictionRecord | None]] = {}

        try:
            # Single batch API call
            fp_results = fp_client.predict_fp_batch(batch_articles)

            # Get model info once for all predictions
            model_info = fp_client.get_model_info()
            model_version = model_info.get("version", "unknown")
            threshold = labeling_settings.fp_skip_llm_threshold

            # Process each result
            for article_id, result in zip(article_ids, fp_results):
                should_continue = result.probability >= threshold

                if should_continue:
                    action = "continued_to_llm"
                    skip_reason = None
                    logger.debug(
                        f"Article {article_id}: FP probability {result.probability:.3f} >= "
                        f"threshold {threshold}, continuing to LLM"
                    )
                else:
                    action = "skipped_llm"
                    skip_reason = (
                        f"High-confidence false positive: probability {result.probability:.3f} "
                        f"< threshold {threshold}"
                    )
                    logger.info(f"Article {article_id}: Skipping LLM - {skip_reason}")

                prediction = ClassifierPredictionRecord(
                    classifier_type="fp",
                    model_version=model_version,
                    probability=result.probability,
                    prediction=result.is_sportswear,
                    threshold_used=threshold,
                    action_taken=action,
                    confidence_level=result.confidence_level,
                    skip_reason=skip_reason,
                )

                if not dry_run:
                    self._save_classifier_prediction(article_id, prediction)

                results[article_id] = (should_continue, prediction)

            logger.info(
                f"FP classifier batch: {len(articles)} articles processed in single API call"
            )

        except Exception as e:
            logger.warning(
                f"FP classifier batch failed: {e}. "
                f"Continuing all articles to LLM (graceful degradation)."
            )

            # Graceful degradation: all articles continue to LLM
            for article_id in article_ids:
                prediction = ClassifierPredictionRecord(
                    classifier_type="fp",
                    model_version="unknown",
                    probability=0.0,
                    prediction=False,
                    threshold_used=labeling_settings.fp_skip_llm_threshold,
                    action_taken="failed",
                    error_message=str(e),
                )

                if not dry_run:
                    self._save_classifier_prediction(article_id, prediction)

                results[article_id] = (True, prediction)

        return results

    def _run_fp_prefilter(
        self,
        article: dict[str, Any],
        dry_run: bool = False,
    ) -> tuple[bool, ClassifierPredictionRecord | None]:
        """Run FP classifier pre-filter on a single article.

        This is a convenience wrapper around _run_fp_prefilter_batch for
        backward compatibility and single-article processing.

        Args:
            article: Article dict with id, title, full_content, etc.
            dry_run: If True, don't save predictions to database

        Returns:
            Tuple of (should_continue, prediction_record)
        """
        results = self._run_fp_prefilter_batch([article], dry_run=dry_run)
        article_id = article["id"]
        return results.get(article_id, (True, None))

    def _save_classifier_prediction(
        self,
        article_id: UUID,
        prediction: ClassifierPredictionRecord,
    ) -> None:
        """Save classifier prediction to database.

        Args:
            article_id: UUID of the article
            prediction: ClassifierPredictionRecord to save
        """
        with self.database.db.get_session() as session:
            db_prediction = ClassifierPrediction(
                article_id=article_id,
                classifier_type=prediction.classifier_type,
                model_version=prediction.model_version,
                probability=prediction.probability,
                prediction=prediction.prediction,
                threshold_used=prediction.threshold_used,
                confidence_level=prediction.confidence_level,
                esg_categories=prediction.esg_categories,
                action_taken=prediction.action_taken,
                skip_reason=prediction.skip_reason,
                error_message=prediction.error_message,
            )
            session.add(db_prediction)
            session.commit()

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
                        "category": a.category or [],
                    }
                    for a in articles
                ]

            logger.info(f"Fetched {len(article_data)} articles for processing")

            # Deduplicate by title similarity (reduces redundant LLM calls)
            article_data, duplicates = self._deduplicate_by_title(
                article_data, dry_run=dry_run
            )
            stats.articles_deduplicated = len(duplicates)

            logger.info(f"Processing {len(article_data)} articles after deduplication")

            # Run FP classifier batch pre-filter (single API call for all articles)
            fp_prefilter_results = self._run_fp_prefilter_batch(article_data, dry_run)

            for article in article_data:
                try:
                    # Get precomputed FP result for this article
                    fp_result = fp_prefilter_results.get(article["id"])

                    result = self._process_article(
                        article,
                        dry_run=dry_run,
                        skip_chunking=skip_chunking,
                        skip_embedding=skip_embedding,
                        fp_prefilter_result=fp_result,
                    )

                    stats.articles_processed += 1

                    if result["labeled"]:
                        stats.articles_labeled += 1
                        stats.brands_labeled += result["brands_count"]
                    elif result.get("false_positive"):
                        stats.articles_false_positive += 1
                        stats.false_positive_brands += result.get(
                            "false_positive_brands", 0
                        )
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

                    # FP classifier stats
                    if result.get("fp_classifier_called"):
                        stats.fp_classifier_calls += 1
                        if result.get("fp_classifier_skipped"):
                            stats.fp_classifier_skipped += 1
                        elif result.get("fp_classifier_continued"):
                            stats.fp_classifier_continued += 1
                        elif result.get("fp_classifier_error"):
                            stats.fp_classifier_errors += 1

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
                f"{stats.articles_skipped} skipped, {stats.articles_false_positive} false positives, "
                f"{stats.articles_deduplicated} deduplicated, {stats.articles_failed} failed"
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
        fp_prefilter_result: tuple[bool, ClassifierPredictionRecord | None] | None = None,
    ) -> dict[str, Any]:
        """Process a single article through the labeling pipeline.

        Args:
            article: Article dict with id, title, full_content, etc.
            dry_run: If True, don't save to database
            skip_chunking: Skip chunking for already-chunked articles
            skip_embedding: Skip embedding generation
            fp_prefilter_result: Precomputed FP classifier result from batch call

        Returns:
            Dict with processing results
        """
        article_id = article["id"]
        content = article["full_content"] or article["description"] or ""
        brands = article["brands_mentioned"]

        result = {
            "labeled": False,
            "skipped": False,
            "false_positive": False,
            "false_positive_brands": 0,
            "error": None,
            "brands_count": 0,
            "chunks_count": 0,
            "embeddings_count": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            # FP classifier tracking
            "fp_classifier_called": False,
            "fp_classifier_skipped": False,
            "fp_classifier_continued": False,
            "fp_classifier_error": False,
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

        # FP Classifier Pre-filter (use precomputed batch result)
        if fp_prefilter_result is not None:
            should_continue, fp_prediction = fp_prefilter_result
        else:
            # Fallback: no precomputed result (shouldn't happen in normal flow)
            should_continue, fp_prediction = True, None

        if fp_prediction is not None:
            result["fp_classifier_called"] = True
            if fp_prediction.action_taken == "skipped_llm":
                result["fp_classifier_skipped"] = True
            elif fp_prediction.action_taken == "continued_to_llm":
                result["fp_classifier_continued"] = True
            elif fp_prediction.action_taken == "failed":
                result["fp_classifier_error"] = True

        if not should_continue:
            # High-confidence false positive - skip LLM labeling
            logger.info(
                f"Article {article_id} marked as false_positive by FP classifier"
            )
            if not dry_run:
                with self.database.db.get_session() as session:
                    skip_reason = (
                        fp_prediction.skip_reason
                        if fp_prediction
                        else "FP classifier pre-filter"
                    )
                    self.database.update_article_labeling_status(
                        session, article_id, "false_positive", skip_reason
                    )
            result["false_positive"] = True
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

        # Check for false positive brands (non-sportswear matches)
        sportswear_brands = [
            a for a in response.brand_analyses if a.is_sportswear_brand
        ]
        non_sportswear_brands = [
            a for a in response.brand_analyses if not a.is_sportswear_brand
        ]

        result["false_positive_brands"] = len(non_sportswear_brands)

        # Log any false positives detected
        for analysis in non_sportswear_brands:
            logger.info(
                f"False positive detected: '{analysis.brand}' is not sportswear - "
                f"{analysis.not_sportswear_reason}"
            )

        # If ALL brands are false positives, mark article as false_positive
        if non_sportswear_brands and not sportswear_brands:
            reasons = [
                f"{a.brand}: {a.not_sportswear_reason}" for a in non_sportswear_brands
            ]
            reason_str = "; ".join(reasons)
            logger.info(
                f"Article {article_id} marked as false positive - "
                f"all brands are non-sportswear: {reason_str}"
            )
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "false_positive", reason_str[:500]
                    )
            result["false_positive"] = True
            return result

        # If no sportswear brands have ESG content, skip
        if not sportswear_brands:
            logger.warning(f"No sportswear brand ESG content for article {article_id}")
            if not dry_run:
                with self.database.db.get_session() as session:
                    self.database.update_article_labeling_status(
                        session, article_id, "skipped", "No sportswear brand ESG content"
                    )
            result["skipped"] = True
            return result

        # Step 4: Match evidence to chunks (if we have chunks)
        # Only match evidence for sportswear brands
        evidence_matches = {}
        if chunks:
            evidence_matches = match_all_evidence(
                sportswear_brands,  # Only sportswear brands
                chunks,
                chunk_ids if chunk_ids else None,
                chunk_embeddings if chunk_embeddings else None,
                self.embedder,
            )

        # Step 5: Save labels and evidence
        if dry_run:
            logger.info(
                f"[DRY RUN] Would save {len(sportswear_brands)} brand labels "
                f"for article {article_id}"
            )
            for analysis in sportswear_brands:
                cats = analysis.get_applicable_categories()
                logger.info(
                    f"  {analysis.brand}: {', '.join(cats) if cats else 'no categories'}"
                )
            if non_sportswear_brands:
                logger.info(
                    f"  (Filtered out {len(non_sportswear_brands)} non-sportswear brands)"
                )
        else:
            with self.database.db.get_session() as session:
                # Save brand labels (only sportswear brands are saved)
                db_labels = self.database.save_brand_labels(
                    session,
                    article_id,
                    sportswear_brands,  # Only sportswear brands
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
        result["brands_count"] = len(sportswear_brands)
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
