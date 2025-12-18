"""Tests for evidence matching to link excerpts to article chunks."""

from uuid import uuid4

import pytest

from src.labeling.chunker import Chunk
from src.labeling.evidence_matcher import EvidenceMatch, EvidenceMatcher, match_all_evidence
from src.labeling.models import BrandAnalysis, CategoryLabel


class TestEvidenceMatchDataclass:
    """Tests for EvidenceMatch dataclass."""

    def test_evidence_match_creation(self):
        """Should create evidence match with all fields."""
        chunk_id = uuid4()
        match = EvidenceMatch(
            excerpt="Test excerpt",
            chunk_id=chunk_id,
            chunk_index=0,
            similarity_score=0.95,
            match_method="exact",
        )
        assert match.excerpt == "Test excerpt"
        assert match.chunk_id == chunk_id
        assert match.chunk_index == 0
        assert match.similarity_score == 0.95
        assert match.match_method == "exact"

    def test_evidence_match_no_chunk(self):
        """Should create evidence match with no chunk."""
        match = EvidenceMatch(
            excerpt="Unmatched excerpt",
            chunk_id=None,
            chunk_index=None,
            similarity_score=0.0,
            match_method="none",
        )
        assert match.chunk_id is None
        assert match.chunk_index is None
        assert match.match_method == "none"


class TestEvidenceMatcherExactMatch:
    """Tests for exact substring matching."""

    def test_exact_match_found(self):
        """Should find exact substring match."""
        matcher = EvidenceMatcher()
        chunks = [
            Chunk(index=0, text="Nike announced carbon neutrality goals for 2030.",
                  char_start=0, char_end=50, token_count=10),
            Chunk(index=1, text="The company plans to reduce emissions by 50%.",
                  char_start=50, char_end=100, token_count=10),
        ]
        chunk_ids = [uuid4(), uuid4()]

        excerpts = ["carbon neutrality goals"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks, chunk_ids)

        assert len(results) == 1
        assert results[0].match_method == "exact"
        assert results[0].similarity_score == 1.0
        assert results[0].chunk_index == 0
        assert results[0].chunk_id == chunk_ids[0]

    def test_exact_match_in_second_chunk(self):
        """Should find exact match in second chunk."""
        matcher = EvidenceMatcher()
        chunks = [
            Chunk(index=0, text="Article introduction here.",
                  char_start=0, char_end=30, token_count=5),
            Chunk(index=1, text="Nike committed to 100% renewable energy by 2025.",
                  char_start=30, char_end=80, token_count=10),
        ]
        chunk_ids = [uuid4(), uuid4()]

        excerpts = ["100% renewable energy"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks, chunk_ids)

        assert results[0].match_method == "exact"
        assert results[0].chunk_index == 1
        assert results[0].chunk_id == chunk_ids[1]


class TestEvidenceMatcherFuzzyMatch:
    """Tests for fuzzy text matching."""

    def test_fuzzy_match_minor_variation(self):
        """Should match with minor text variations."""
        matcher = EvidenceMatcher(fuzzy_threshold=0.8)
        chunks = [
            Chunk(index=0, text="Nike announced their carbon neutrality goals.",
                  char_start=0, char_end=50, token_count=8),
        ]

        # Slightly different text
        excerpts = ["Nike announced carbon neutrality goals"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks)

        assert results[0].match_method == "fuzzy"
        assert results[0].similarity_score >= 0.8
        assert results[0].chunk_index == 0

    def test_fuzzy_match_below_threshold(self):
        """Should not match when below threshold."""
        matcher = EvidenceMatcher(fuzzy_threshold=0.95)
        chunks = [
            Chunk(index=0, text="Nike announced something completely different.",
                  char_start=0, char_end=50, token_count=6),
        ]

        excerpts = ["Adidas committed to sustainability"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks)

        # Should still return a result, but with low score and fuzzy method
        # (falls back to best fuzzy match)
        assert results[0].similarity_score < 0.95


class TestEvidenceMatcherEmptyInputs:
    """Tests for edge cases with empty inputs."""

    def test_empty_excerpts(self):
        """Should return empty list for empty excerpts."""
        matcher = EvidenceMatcher()
        chunks = [
            Chunk(index=0, text="Some text.", char_start=0, char_end=10, token_count=2),
        ]

        results = matcher.match_evidence_to_chunks([], chunks)
        assert results == []

    def test_empty_chunks(self):
        """Should return unmatched evidence for empty chunks."""
        matcher = EvidenceMatcher()

        excerpts = ["Some evidence"]
        results = matcher.match_evidence_to_chunks(excerpts, [])

        assert len(results) == 1
        assert results[0].chunk_id is None
        assert results[0].chunk_index is None
        assert results[0].match_method == "none"
        assert results[0].similarity_score == 0.0

    def test_multiple_excerpts_empty_chunks(self):
        """Should return unmatched for all excerpts when no chunks."""
        matcher = EvidenceMatcher()

        excerpts = ["Evidence 1", "Evidence 2", "Evidence 3"]
        results = matcher.match_evidence_to_chunks(excerpts, [])

        assert len(results) == 3
        for result in results:
            assert result.match_method == "none"


class TestEvidenceMatcherMultipleExcerpts:
    """Tests for matching multiple excerpts."""

    def test_multiple_excerpts_different_chunks(self):
        """Should match excerpts to different chunks."""
        matcher = EvidenceMatcher()
        chunks = [
            Chunk(index=0, text="Nike announced carbon neutrality goals.",
                  char_start=0, char_end=40, token_count=6),
            Chunk(index=1, text="Adidas committed to sustainable materials.",
                  char_start=40, char_end=85, token_count=6),
        ]
        chunk_ids = [uuid4(), uuid4()]

        excerpts = ["carbon neutrality goals", "sustainable materials"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks, chunk_ids)

        assert len(results) == 2
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 1

    def test_multiple_excerpts_same_chunk(self):
        """Should match multiple excerpts to same chunk."""
        matcher = EvidenceMatcher()
        chunks = [
            Chunk(index=0, text="Nike announced carbon neutrality goals and renewable energy targets.",
                  char_start=0, char_end=70, token_count=10),
        ]
        chunk_ids = [uuid4()]

        excerpts = ["carbon neutrality", "renewable energy"]
        results = matcher.match_evidence_to_chunks(excerpts, chunks, chunk_ids)

        assert len(results) == 2
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 0


class TestFuzzyMatchScore:
    """Tests for fuzzy match scoring."""

    def test_fuzzy_match_identical_text(self):
        """Identical text should have high score."""
        matcher = EvidenceMatcher()
        score = matcher._fuzzy_match_score("test text", "test text")
        assert score == 1.0

    def test_fuzzy_match_similar_text(self):
        """Similar text should have reasonably high score."""
        matcher = EvidenceMatcher()
        # Excerpt should be shorter than chunk for matching
        score = matcher._fuzzy_match_score("test text", "test text here with more words")
        assert score > 0.4  # SequenceMatcher ratio for short strings

    def test_fuzzy_match_different_text(self):
        """Very different text should have low score."""
        matcher = EvidenceMatcher()
        score = matcher._fuzzy_match_score("apple banana", "xyz 123 abc")
        assert score < 0.5

    def test_fuzzy_match_excerpt_much_longer(self):
        """Excerpt much longer than chunk should return 0."""
        matcher = EvidenceMatcher()
        long_excerpt = "a" * 200
        short_chunk = "a" * 50
        score = matcher._fuzzy_match_score(long_excerpt, short_chunk)
        assert score == 0.0


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0."""
        matcher = EvidenceMatcher()
        vec = [1.0, 2.0, 3.0]
        sim = matcher._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0."""
        matcher = EvidenceMatcher()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = matcher._cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1.0."""
        matcher = EvidenceMatcher()
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        sim = matcher._cosine_similarity(vec1, vec2)
        assert abs(sim + 1.0) < 0.0001

    def test_cosine_similarity_different_lengths(self):
        """Different length vectors should return 0.0."""
        matcher = EvidenceMatcher()
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = matcher._cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0."""
        matcher = EvidenceMatcher()
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = matcher._cosine_similarity(vec1, vec2)
        assert sim == 0.0


class TestMatchAllEvidence:
    """Tests for the match_all_evidence helper function."""

    def test_match_all_evidence_single_brand(self):
        """Should match evidence for single brand."""
        chunks = [
            Chunk(index=0, text="Nike committed to carbon neutrality.",
                  char_start=0, char_end=40, token_count=6),
        ]
        chunk_ids = [uuid4()]

        brand_analyses = [
            BrandAnalysis(
                brand="Nike",
                categories={
                    "environmental": CategoryLabel(
                        applies=True, sentiment=1, evidence=["carbon neutrality"]
                    ),
                    "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "digital_transformation": CategoryLabel(
                        applies=False, sentiment=None, evidence=[]
                    ),
                },
                confidence=0.9,
                reasoning="Test",
            )
        ]

        results = match_all_evidence(brand_analyses, chunks, chunk_ids)

        assert "Nike" in results
        assert "environmental" in results["Nike"]
        assert len(results["Nike"]["environmental"]) == 1
        assert results["Nike"]["environmental"][0].match_method == "exact"

    def test_match_all_evidence_multiple_brands(self):
        """Should match evidence for multiple brands."""
        chunks = [
            Chunk(index=0, text="Nike committed to carbon neutrality.",
                  char_start=0, char_end=40, token_count=6),
            Chunk(index=1, text="Adidas announced worker safety programs.",
                  char_start=40, char_end=80, token_count=6),
        ]
        chunk_ids = [uuid4(), uuid4()]

        brand_analyses = [
            BrandAnalysis(
                brand="Nike",
                categories={
                    "environmental": CategoryLabel(
                        applies=True, sentiment=1, evidence=["carbon neutrality"]
                    ),
                    "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "digital_transformation": CategoryLabel(
                        applies=False, sentiment=None, evidence=[]
                    ),
                },
                confidence=0.9,
                reasoning="Test",
            ),
            BrandAnalysis(
                brand="Adidas",
                categories={
                    "environmental": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "social": CategoryLabel(
                        applies=True, sentiment=1, evidence=["worker safety"]
                    ),
                    "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "digital_transformation": CategoryLabel(
                        applies=False, sentiment=None, evidence=[]
                    ),
                },
                confidence=0.85,
                reasoning="Test",
            ),
        ]

        results = match_all_evidence(brand_analyses, chunks, chunk_ids)

        assert "Nike" in results
        assert "Adidas" in results
        assert "environmental" in results["Nike"]
        assert "social" in results["Adidas"]

    def test_match_all_evidence_no_applicable_categories(self):
        """Should return empty for brand with no applicable categories."""
        chunks = [
            Chunk(index=0, text="Some article text.",
                  char_start=0, char_end=20, token_count=4),
        ]

        brand_analyses = [
            BrandAnalysis(
                brand="Nike",
                categories={
                    "environmental": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "digital_transformation": CategoryLabel(
                        applies=False, sentiment=None, evidence=[]
                    ),
                },
                confidence=0.5,
                reasoning="No ESG content",
            )
        ]

        results = match_all_evidence(brand_analyses, chunks)

        # Brand with no applicable categories should not be in results
        assert "Nike" not in results

    def test_match_all_evidence_multiple_categories(self):
        """Should match evidence across multiple categories for same brand."""
        chunks = [
            Chunk(index=0, text="Nike committed to carbon neutrality and worker safety.",
                  char_start=0, char_end=55, token_count=9),
        ]
        chunk_ids = [uuid4()]

        brand_analyses = [
            BrandAnalysis(
                brand="Nike",
                categories={
                    "environmental": CategoryLabel(
                        applies=True, sentiment=1, evidence=["carbon neutrality"]
                    ),
                    "social": CategoryLabel(
                        applies=True, sentiment=1, evidence=["worker safety"]
                    ),
                    "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                    "digital_transformation": CategoryLabel(
                        applies=False, sentiment=None, evidence=[]
                    ),
                },
                confidence=0.9,
                reasoning="Test",
            )
        ]

        results = match_all_evidence(brand_analyses, chunks, chunk_ids)

        assert "Nike" in results
        assert "environmental" in results["Nike"]
        assert "social" in results["Nike"]
        assert len(results["Nike"]["environmental"]) == 1
        assert len(results["Nike"]["social"]) == 1
