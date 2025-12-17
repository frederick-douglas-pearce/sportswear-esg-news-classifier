"""Evidence matching to link LLM-extracted excerpts to article chunks."""

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from uuid import UUID

from .chunker import Chunk
from .embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


@dataclass
class EvidenceMatch:
    """Result of matching an evidence excerpt to a chunk."""

    excerpt: str
    chunk_id: UUID | None
    chunk_index: int | None
    similarity_score: float
    match_method: str  # "exact", "fuzzy", "embedding", "none"


class EvidenceMatcher:
    """Matches evidence excerpts from LLM responses to article chunks.

    Uses multiple matching strategies:
    1. Exact substring match
    2. Fuzzy text matching (for minor variations)
    3. Embedding similarity (for paraphrased excerpts)
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder | None = None,
        fuzzy_threshold: float = 0.8,
        embedding_threshold: float = 0.85,
    ):
        """Initialize the evidence matcher.

        Args:
            embedder: OpenAI embedder for semantic matching (optional)
            fuzzy_threshold: Minimum similarity for fuzzy matching (0.0-1.0)
            embedding_threshold: Minimum cosine similarity for embedding matching
        """
        self.embedder = embedder
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold

    def match_evidence_to_chunks(
        self,
        excerpts: list[str],
        chunks: list[Chunk],
        chunk_ids: list[UUID] | None = None,
        chunk_embeddings: list[list[float]] | None = None,
    ) -> list[EvidenceMatch]:
        """Match a list of evidence excerpts to article chunks.

        Args:
            excerpts: List of evidence excerpts from LLM
            chunks: List of article chunks
            chunk_ids: Optional list of database UUIDs for chunks
            chunk_embeddings: Optional pre-computed chunk embeddings

        Returns:
            List of EvidenceMatch results
        """
        if not excerpts:
            return []

        if not chunks:
            return [
                EvidenceMatch(
                    excerpt=excerpt,
                    chunk_id=None,
                    chunk_index=None,
                    similarity_score=0.0,
                    match_method="none",
                )
                for excerpt in excerpts
            ]

        results = []
        for excerpt in excerpts:
            match = self._match_single_excerpt(
                excerpt, chunks, chunk_ids, chunk_embeddings
            )
            results.append(match)

        return results

    def _match_single_excerpt(
        self,
        excerpt: str,
        chunks: list[Chunk],
        chunk_ids: list[UUID] | None,
        chunk_embeddings: list[list[float]] | None,
    ) -> EvidenceMatch:
        """Match a single excerpt to the best matching chunk."""
        excerpt_clean = excerpt.strip()

        # Strategy 1: Exact substring match
        for i, chunk in enumerate(chunks):
            if excerpt_clean in chunk.text:
                return EvidenceMatch(
                    excerpt=excerpt,
                    chunk_id=chunk_ids[i] if chunk_ids else None,
                    chunk_index=chunk.index,
                    similarity_score=1.0,
                    match_method="exact",
                )

        # Strategy 2: Fuzzy text matching
        best_fuzzy_score = 0.0
        best_fuzzy_index = -1

        for i, chunk in enumerate(chunks):
            # Check if excerpt is a fuzzy match to any part of the chunk
            score = self._fuzzy_match_score(excerpt_clean, chunk.text)
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_index = i

        if best_fuzzy_score >= self.fuzzy_threshold:
            return EvidenceMatch(
                excerpt=excerpt,
                chunk_id=chunk_ids[best_fuzzy_index] if chunk_ids else None,
                chunk_index=chunks[best_fuzzy_index].index,
                similarity_score=best_fuzzy_score,
                match_method="fuzzy",
            )

        # Strategy 3: Embedding similarity (if embedder available)
        if self.embedder and chunk_embeddings:
            try:
                excerpt_embedding = self.embedder.embed_text(excerpt_clean)
                best_sim_score = 0.0
                best_sim_index = -1

                for i, chunk_emb in enumerate(chunk_embeddings):
                    sim = self._cosine_similarity(excerpt_embedding, chunk_emb)
                    if sim > best_sim_score:
                        best_sim_score = sim
                        best_sim_index = i

                if best_sim_score >= self.embedding_threshold:
                    return EvidenceMatch(
                        excerpt=excerpt,
                        chunk_id=chunk_ids[best_sim_index] if chunk_ids else None,
                        chunk_index=chunks[best_sim_index].index,
                        similarity_score=best_sim_score,
                        match_method="embedding",
                    )
            except Exception as e:
                logger.warning(f"Embedding match failed: {e}")

        # No good match found - return best fuzzy match anyway with low confidence
        if best_fuzzy_index >= 0:
            return EvidenceMatch(
                excerpt=excerpt,
                chunk_id=chunk_ids[best_fuzzy_index] if chunk_ids else None,
                chunk_index=chunks[best_fuzzy_index].index,
                similarity_score=best_fuzzy_score,
                match_method="fuzzy",
            )

        return EvidenceMatch(
            excerpt=excerpt,
            chunk_id=None,
            chunk_index=None,
            similarity_score=0.0,
            match_method="none",
        )

    def _fuzzy_match_score(self, excerpt: str, chunk_text: str) -> float:
        """Calculate fuzzy match score between excerpt and chunk.

        Uses a sliding window approach to find the best matching substring.
        """
        excerpt_lower = excerpt.lower()
        chunk_lower = chunk_text.lower()

        # If excerpt is much longer than chunk, no match
        if len(excerpt) > len(chunk_text) * 1.5:
            return 0.0

        # For short excerpts, just compare directly
        if len(excerpt) <= 100:
            return SequenceMatcher(None, excerpt_lower, chunk_lower).ratio()

        # For longer excerpts, use sliding window
        best_ratio = 0.0
        window_size = min(len(excerpt) * 2, len(chunk_text))
        step = max(1, len(excerpt) // 4)

        for i in range(0, max(1, len(chunk_text) - window_size + 1), step):
            window = chunk_lower[i : i + window_size]
            ratio = SequenceMatcher(None, excerpt_lower, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio

        return best_ratio

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


def match_all_evidence(
    brand_analyses: list,  # List of BrandAnalysis
    chunks: list[Chunk],
    chunk_ids: list[UUID] | None = None,
    chunk_embeddings: list[list[float]] | None = None,
    embedder: OpenAIEmbedder | None = None,
) -> dict[str, dict[str, list[EvidenceMatch]]]:
    """Match all evidence from brand analyses to chunks.

    Args:
        brand_analyses: List of BrandAnalysis objects from LLM
        chunks: List of article chunks
        chunk_ids: Optional database UUIDs for chunks
        chunk_embeddings: Optional pre-computed chunk embeddings
        embedder: Optional embedder for semantic matching

    Returns:
        Nested dict: {brand_name: {category: [EvidenceMatch, ...]}}
    """
    matcher = EvidenceMatcher(embedder=embedder)
    results: dict[str, dict[str, list[EvidenceMatch]]] = {}

    for analysis in brand_analyses:
        brand_results: dict[str, list[EvidenceMatch]] = {}

        for category, label in analysis.categories.items():
            if label.applies and label.evidence:
                matches = matcher.match_evidence_to_chunks(
                    label.evidence,
                    chunks,
                    chunk_ids,
                    chunk_embeddings,
                )
                brand_results[category] = matches

        if brand_results:
            results[analysis.brand] = brand_results

    return results
