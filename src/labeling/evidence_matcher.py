"""Evidence matching to link LLM-extracted excerpts to article chunks."""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from uuid import UUID

from .chunker import Chunk
from .config import labeling_settings
from .embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)

# Confidence thresholds for categorizing match quality
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MEDIUM_THRESHOLD = 0.70
CONFIDENCE_LOW_THRESHOLD = 0.50


def _get_confidence_label(score: float) -> str:
    """Get confidence label based on similarity score."""
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    elif score >= CONFIDENCE_LOW_THRESHOLD:
        return "low"
    else:
        return "unmatched"


@dataclass
class EvidenceMatch:
    """Result of matching an evidence excerpt to a chunk."""

    excerpt: str
    chunk_id: UUID | None
    chunk_index: int | None
    similarity_score: float
    match_method: str  # "exact", "fuzzy", "embedding", "combined", "none"
    confidence: str = field(default="unmatched")  # "high", "medium", "low", "unmatched"
    context_snippet: str | None = field(default=None)  # Condensed snippet around match


class EvidenceMatcher:
    """Matches evidence excerpts from LLM responses to article chunks.

    Uses multiple matching strategies:
    1. Exact substring match
    2. Fuzzy text matching (for minor variations)
    3. Embedding similarity (for paraphrased excerpts)
    4. Combined scoring (fuzzy + embedding reranking)
    """

    def __init__(
        self,
        embedder: OpenAIEmbedder | None = None,
        fuzzy_threshold: float = 0.8,
        embedding_threshold: float = 0.85,
        min_confidence_threshold: float | None = None,
        use_embedding_rerank: bool | None = None,
        snippet_context_chars: int = 100,
    ):
        """Initialize the evidence matcher.

        Args:
            embedder: OpenAI embedder for semantic matching (optional)
            fuzzy_threshold: Minimum similarity for fuzzy matching (0.0-1.0)
            embedding_threshold: Minimum cosine similarity for embedding matching
            min_confidence_threshold: Minimum score to return a match (default: from settings)
                                      Below this, returns match_method="none"
            use_embedding_rerank: Use embedding to rerank fuzzy matches (default: from settings)
            snippet_context_chars: Characters of context around matched excerpt for snippets
        """
        self.embedder = embedder
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.min_confidence_threshold = (
            min_confidence_threshold
            if min_confidence_threshold is not None
            else labeling_settings.evidence_min_confidence
        )
        self.use_embedding_rerank = (
            use_embedding_rerank
            if use_embedding_rerank is not None
            else labeling_settings.evidence_use_embedding_rerank
        )
        self.snippet_context_chars = snippet_context_chars

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
        """Match a single excerpt to the best matching chunk.

        Matching strategy:
        1. Exact substring match (returns immediately with score 1.0)
        2. Fuzzy text matching with optional embedding reranking
        3. Apply min confidence threshold - low scores return match_method="none"
        """
        excerpt_clean = excerpt.strip()

        # Strategy 1: Exact substring match
        for i, chunk in enumerate(chunks):
            if excerpt_clean in chunk.text:
                snippet = self._extract_snippet(chunk.text, excerpt_clean)
                return EvidenceMatch(
                    excerpt=excerpt,
                    chunk_id=chunk_ids[i] if chunk_ids else None,
                    chunk_index=chunk.index,
                    similarity_score=1.0,
                    match_method="exact",
                    confidence="high",
                    context_snippet=snippet,
                )

        # Strategy 2: Calculate fuzzy scores for all chunks
        fuzzy_scores = []
        for i, chunk in enumerate(chunks):
            score = self._fuzzy_match_score(excerpt_clean, chunk.text)
            fuzzy_scores.append((i, score))

        # Strategy 3: If embedding reranking is enabled, compute combined scores
        embedding_scores: list[tuple[int, float]] = []
        excerpt_embedding: list[float] | None = None

        if self.use_embedding_rerank and self.embedder and chunk_embeddings:
            try:
                excerpt_embedding = self.embedder.embed_text(excerpt_clean)
                for i, chunk_emb in enumerate(chunk_embeddings):
                    sim = self._cosine_similarity(excerpt_embedding, chunk_emb)
                    embedding_scores.append((i, sim))
            except Exception as e:
                logger.warning(f"Embedding reranking failed: {e}")

        # Compute final scores - combine fuzzy and embedding if both available
        if embedding_scores and fuzzy_scores:
            # Combined scoring: 30% fuzzy, 70% embedding for semantic priority
            combined_scores = []
            fuzzy_dict = dict(fuzzy_scores)
            embedding_dict = dict(embedding_scores)

            for i in range(len(chunks)):
                fuzzy = fuzzy_dict.get(i, 0.0)
                emb = embedding_dict.get(i, 0.0)
                combined = 0.3 * fuzzy + 0.7 * emb
                combined_scores.append((i, combined, fuzzy, emb))

            # Sort by combined score descending
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score, best_fuzzy, best_emb = combined_scores[0]
            match_method = "combined"

            # Determine which component contributed more for logging
            logger.debug(
                f"Combined match: idx={best_idx}, "
                f"combined={best_score:.3f}, fuzzy={best_fuzzy:.3f}, emb={best_emb:.3f}"
            )
        else:
            # Fallback to fuzzy-only or embedding-only
            if embedding_scores:
                best_idx, best_score = max(embedding_scores, key=lambda x: x[1])
                match_method = "embedding"
            else:
                best_idx, best_score = max(fuzzy_scores, key=lambda x: x[1])
                match_method = "fuzzy"

        # Apply minimum confidence threshold
        if best_score < self.min_confidence_threshold:
            logger.debug(
                f"Match rejected: score {best_score:.3f} below threshold "
                f"{self.min_confidence_threshold:.3f}"
            )
            return EvidenceMatch(
                excerpt=excerpt,
                chunk_id=None,
                chunk_index=None,
                similarity_score=best_score,
                match_method="none",
                confidence="unmatched",
                context_snippet=None,
            )

        # Build successful match
        best_chunk = chunks[best_idx]
        snippet = self._extract_snippet(best_chunk.text, excerpt_clean)
        confidence = _get_confidence_label(best_score)

        return EvidenceMatch(
            excerpt=excerpt,
            chunk_id=chunk_ids[best_idx] if chunk_ids else None,
            chunk_index=best_chunk.index,
            similarity_score=best_score,
            match_method=match_method,
            confidence=confidence,
            context_snippet=snippet,
        )

    def _extract_snippet(self, chunk_text: str, excerpt: str) -> str:
        """Extract a condensed snippet around the matched excerpt.

        Args:
            chunk_text: Full text of the matched chunk
            excerpt: The evidence excerpt to find

        Returns:
            Snippet with context around the excerpt, or the excerpt itself if not found
        """
        excerpt_lower = excerpt.lower()
        chunk_lower = chunk_text.lower()

        # Try to find exact position first
        pos = chunk_lower.find(excerpt_lower)
        if pos == -1:
            # Try to find the first few words of the excerpt
            first_words = " ".join(excerpt.split()[:5]).lower()
            pos = chunk_lower.find(first_words)

        if pos == -1:
            # Can't find it - return truncated chunk
            if len(chunk_text) <= self.snippet_context_chars * 2:
                return chunk_text
            return chunk_text[: self.snippet_context_chars * 2] + "..."

        # Extract context around the match
        context_start = max(0, pos - self.snippet_context_chars)
        context_end = min(len(chunk_text), pos + len(excerpt) + self.snippet_context_chars)

        # Adjust to word boundaries
        if context_start > 0:
            # Find next space after context_start
            space_pos = chunk_text.find(" ", context_start)
            if space_pos != -1 and space_pos < pos:
                context_start = space_pos + 1

        if context_end < len(chunk_text):
            # Find last space before context_end
            space_pos = chunk_text.rfind(" ", pos + len(excerpt), context_end)
            if space_pos != -1:
                context_end = space_pos

        snippet = chunk_text[context_start:context_end].strip()

        # Add ellipsis indicators
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(chunk_text) else ""

        return f"{prefix}{snippet}{suffix}"

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
