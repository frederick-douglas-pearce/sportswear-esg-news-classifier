"""Tests for cross-encoder reranking module."""

from unittest.mock import MagicMock, patch

import pytest

from src.labeling.reranker import CrossEncoderReranker, RerankCandidate, RerankResult


class TestRerankCandidate:
    """Tests for RerankCandidate dataclass."""

    def test_candidate_creation(self):
        """Should create candidate with all fields."""
        candidate = RerankCandidate(
            index=0,
            text="Sample chunk text",
            initial_score=0.85,
        )
        assert candidate.index == 0
        assert candidate.text == "Sample chunk text"
        assert candidate.initial_score == 0.85


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_result_creation(self):
        """Should create result with all fields."""
        result = RerankResult(
            index=0,
            rerank_score=0.92,
            combined_score=0.88,
        )
        assert result.index == 0
        assert result.rerank_score == 0.92
        assert result.combined_score == 0.88


class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    def test_default_initialization(self):
        """Should initialize with default settings."""
        reranker = CrossEncoderReranker()
        # Model should be lazy-loaded
        assert reranker._model is None
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.rerank_weight == 0.6

    def test_custom_initialization(self):
        """Should initialize with custom settings."""
        reranker = CrossEncoderReranker(
            model_name="custom-model",
            rerank_weight=0.7,
        )
        assert reranker.model_name == "custom-model"
        assert reranker.rerank_weight == 0.7


class TestCrossEncoderRerankerRerank:
    """Tests for CrossEncoderReranker.rerank method."""

    def test_rerank_empty_candidates(self):
        """Should return empty list for empty candidates."""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", [])
        assert results == []

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_single_candidate(self, mock_cross_encoder_class):
        """Should rerank single candidate."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [2.5]  # Raw logit score
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_weight=0.6)
        candidates = [
            RerankCandidate(index=0, text="Relevant text", initial_score=0.8)
        ]

        results = reranker.rerank("query text", candidates)

        assert len(results) == 1
        assert results[0].index == 0
        # Sigmoid(2.5) ≈ 0.924
        assert results[0].rerank_score > 0.9
        # Combined = (1-0.6)*0.8 + 0.6*0.924 ≈ 0.874
        assert 0.8 < results[0].combined_score < 0.95

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_multiple_candidates(self, mock_cross_encoder_class):
        """Should rerank multiple candidates and sort by combined score."""
        # Setup mock
        mock_model = MagicMock()
        # Candidate 1: low score, Candidate 2: high score
        mock_model.predict.return_value = [-1.0, 3.0]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_weight=0.6)
        candidates = [
            RerankCandidate(index=0, text="Less relevant", initial_score=0.9),
            RerankCandidate(index=1, text="More relevant", initial_score=0.5),
        ]

        results = reranker.rerank("query text", candidates)

        assert len(results) == 2
        # Candidate 1 should be reranked higher despite lower initial score
        assert results[0].index == 1
        assert results[1].index == 0
        # Higher rerank score should dominate
        assert results[0].rerank_score > results[1].rerank_score

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_with_top_k(self, mock_cross_encoder_class):
        """Should limit results to top_k."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0, 2.0, 3.0]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        candidates = [
            RerankCandidate(index=0, text="Text 1", initial_score=0.8),
            RerankCandidate(index=1, text="Text 2", initial_score=0.7),
            RerankCandidate(index=2, text="Text 3", initial_score=0.6),
        ]

        results = reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2
        # Top 2 should be candidates with highest combined scores
        assert results[0].index == 2  # Highest rerank score
        assert results[1].index == 1  # Second highest

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_prediction_failure_fallback(self, mock_cross_encoder_class):
        """Should fall back to initial scores on prediction failure."""
        # Setup mock to raise exception
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        candidates = [
            RerankCandidate(index=0, text="Text 1", initial_score=0.8),
            RerankCandidate(index=1, text="Text 2", initial_score=0.7),
        ]

        results = reranker.rerank("query", candidates)

        assert len(results) == 2
        # Should use initial scores only
        for result, candidate in zip(results, candidates):
            assert result.rerank_score == 0.0
            assert result.combined_score == candidate.initial_score

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_weight_application(self, mock_cross_encoder_class):
        """Should correctly apply rerank_weight in combined score."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.0]  # Sigmoid(0) = 0.5
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker(rerank_weight=0.6)
        candidates = [
            RerankCandidate(index=0, text="Text", initial_score=0.8)
        ]

        results = reranker.rerank("query", candidates)

        # Sigmoid(0) = 0.5
        # Combined = (1-0.6)*0.8 + 0.6*0.5 = 0.32 + 0.3 = 0.62
        assert abs(results[0].rerank_score - 0.5) < 0.01
        expected_combined = (1 - 0.6) * 0.8 + 0.6 * 0.5
        assert abs(results[0].combined_score - expected_combined) < 0.01


class TestCrossEncoderRerankerRerankSingle:
    """Tests for CrossEncoderReranker.rerank_single method."""

    @patch("sentence_transformers.CrossEncoder")
    def test_rerank_single_returns_best(self, mock_cross_encoder_class):
        """Should return only the best result."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.0, 3.0]
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        candidates = [
            RerankCandidate(index=0, text="Text 1", initial_score=0.8),
            RerankCandidate(index=1, text="Text 2", initial_score=0.7),
        ]

        result = reranker.rerank_single("query", candidates)

        assert result is not None
        assert result.index == 1  # Higher rerank score

    def test_rerank_single_empty_candidates(self):
        """Should return None for empty candidates."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank_single("query", [])
        assert result is None


class TestCrossEncoderRerankerSigmoid:
    """Tests for sigmoid normalization edge cases."""

    @patch("sentence_transformers.CrossEncoder")
    def test_very_negative_score(self, mock_cross_encoder_class):
        """Should handle very negative raw scores."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [-100.0]  # Very negative
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        candidates = [
            RerankCandidate(index=0, text="Text", initial_score=0.5)
        ]

        results = reranker.rerank("query", candidates)

        # Sigmoid of very negative should be near 0
        assert results[0].rerank_score < 0.01

    @patch("sentence_transformers.CrossEncoder")
    def test_very_positive_score(self, mock_cross_encoder_class):
        """Should handle very positive raw scores."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [100.0]  # Very positive
        mock_cross_encoder_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        candidates = [
            RerankCandidate(index=0, text="Text", initial_score=0.5)
        ]

        results = reranker.rerank("query", candidates)

        # Sigmoid of very positive should be near 1
        assert results[0].rerank_score > 0.99
