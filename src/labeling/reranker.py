"""Cross-encoder reranking for evidence quality improvement.

Cross-encoders jointly encode (query, document) pairs, enabling more accurate
relevance scoring than bi-encoders that encode them independently. This module
provides reranking capabilities to improve evidence matching quality.
"""

import logging
from dataclasses import dataclass

from .config import labeling_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """A candidate for reranking."""

    index: int
    text: str
    initial_score: float


@dataclass
class RerankResult:
    """Result of reranking a candidate."""

    index: int
    rerank_score: float
    combined_score: float


class CrossEncoderReranker:
    """Reranks evidence candidates using a cross-encoder model.

    The cross-encoder jointly encodes (query, document) pairs to produce
    more accurate relevance scores than bi-encoder embeddings, which encode
    query and document independently.

    The model is lazy-loaded on first use to avoid overhead when reranking
    is disabled.

    Attributes:
        model_name: Name of the cross-encoder model to use
        rerank_weight: Weight for combining rerank score with initial score
    """

    def __init__(
        self,
        model_name: str | None = None,
        rerank_weight: float | None = None,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder.
                       Defaults to settings.rerank_model.
            rerank_weight: Weight for rerank score in combined scoring (0-1).
                          Defaults to settings.rerank_weight.
        """
        self.model_name = model_name or labeling_settings.rerank_model
        self.rerank_weight = (
            rerank_weight if rerank_weight is not None else labeling_settings.rerank_weight
        )
        self._model = None  # Lazy load

    @property
    def model(self):
        """Get the cross-encoder model, loading lazily on first use.

        Returns:
            CrossEncoder model instance
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank candidates using the cross-encoder and return top-k results.

        Args:
            query: The evidence excerpt (query text)
            candidates: List of RerankCandidate objects containing:
                       - index: Original index in the candidate list
                       - text: Candidate text (chunk text)
                       - initial_score: Score from initial matching
            top_k: Maximum number of results to return. If None, returns all.
                  Defaults to labeling_settings.rerank_top_k if not specified.

        Returns:
            List of RerankResult objects sorted by combined_score descending,
            containing:
            - index: Original candidate index
            - rerank_score: Raw cross-encoder score (normalized to 0-1)
            - combined_score: Weighted combination of initial and rerank scores
        """
        if not candidates:
            return []

        if top_k is None:
            top_k = labeling_settings.rerank_top_k

        # Create query-document pairs for cross-encoder
        pairs = [(query, candidate.text) for candidate in candidates]

        # Get cross-encoder scores
        try:
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Cross-encoder prediction failed: {e}")
            # Fall back to initial scores only
            return [
                RerankResult(
                    index=c.index,
                    rerank_score=0.0,
                    combined_score=c.initial_score,
                )
                for c in candidates
            ]

        # Normalize scores to 0-1 range using sigmoid
        # ms-marco models output logits that can be negative
        import math

        def sigmoid(x: float) -> float:
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0

        normalized_scores = [sigmoid(float(score)) for score in raw_scores]

        # Compute combined scores and build results
        results = []
        for i, (candidate, rerank_score) in enumerate(zip(candidates, normalized_scores)):
            # Combined score: (1-weight)*initial + weight*rerank
            combined = (
                (1 - self.rerank_weight) * candidate.initial_score
                + self.rerank_weight * rerank_score
            )
            results.append(
                RerankResult(
                    index=candidate.index,
                    rerank_score=rerank_score,
                    combined_score=combined,
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Return top-k results
        if top_k and top_k < len(results):
            return results[:top_k]
        return results

    def rerank_single(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> RerankResult | None:
        """Rerank candidates and return only the best match.

        Convenience method when you only need the single best result.

        Args:
            query: The evidence excerpt (query text)
            candidates: List of candidates to rerank

        Returns:
            The best RerankResult, or None if no candidates
        """
        results = self.rerank(query, candidates, top_k=1)
        return results[0] if results else None
