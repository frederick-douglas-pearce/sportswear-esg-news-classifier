"""Tests for FP classifier pre-filter integration in labeling pipeline."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.labeling.classifier_client import (
    ClassifierClient,
    ClassifierPredictionRecord,
    FPPredictionResult,
)
from src.labeling.pipeline import LabelingPipeline, LabelingStats


class TestClassifierClient:
    """Tests for ClassifierClient HTTP client."""

    def test_init(self):
        """Should initialize with base URL and timeout."""
        client = ClassifierClient("http://localhost:8000", timeout=10.0)
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 10.0
        assert client._client is None
        assert client._model_info is None

    def test_init_strips_trailing_slash(self):
        """Should strip trailing slash from base URL."""
        client = ClassifierClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_get_client_lazy_init(self):
        """Should lazily initialize HTTP client."""
        client = ClassifierClient("http://localhost:8000")
        assert client._client is None

        with patch("src.labeling.classifier_client.httpx.Client") as mock_client_class:
            mock_http_client = MagicMock()
            mock_client_class.return_value = mock_http_client

            http_client = client._get_client()

            assert http_client == mock_http_client
            mock_client_class.assert_called_once_with(
                base_url="http://localhost:8000",
                timeout=30.0,
            )

    def test_close_cleans_up(self):
        """Should close HTTP client and clear model info."""
        client = ClassifierClient("http://localhost:8000")
        mock_http_client = MagicMock()
        client._client = mock_http_client
        client._model_info = {"version": "1.0"}

        client.close()

        mock_http_client.close.assert_called_once()
        assert client._client is None
        assert client._model_info is None

    def test_health_check_success(self):
        """Should return True when service is healthy."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "healthy", "model_loaded": True}
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = client.health_check()

            assert result is True
            mock_http_client.get.assert_called_once_with("/health")

    def test_health_check_unhealthy(self):
        """Should return False when service is unhealthy."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "unhealthy", "model_loaded": False}
            mock_http_client.get.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = client.health_check()

            assert result is False

    def test_health_check_error(self):
        """Should return False on connection error."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_http_client.get.side_effect = Exception("Connection refused")
            mock_get_client.return_value = mock_http_client

            result = client.health_check()

            assert result is False

    def test_predict_fp_success(self):
        """Should parse FP prediction response."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "is_sportswear": True,
                "probability": 0.85,
                "risk_level": "low",
                "threshold": 0.3,
            }
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = client.predict_fp(
                title="Nike announces new shoe",
                content="Nike releases running shoe...",
                brands=["Nike"],
                source_name="ESPN",
                category=["sports"],
            )

            assert isinstance(result, FPPredictionResult)
            assert result.is_sportswear is True
            assert result.probability == 0.85
            assert result.risk_level == "low"
            assert result.threshold == 0.3

    def test_predict_fp_batch_success(self):
        """Should parse batch FP prediction response."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "predictions": [
                    {"is_sportswear": True, "probability": 0.9, "risk_level": "low", "threshold": 0.3},
                    {"is_sportswear": False, "probability": 0.1, "risk_level": "high", "threshold": 0.3},
                ]
            }
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            articles = [
                {"title": "Nike shoe", "content": "Content 1"},
                {"title": "Puma cat", "content": "Content 2"},
            ]
            results = client.predict_fp_batch(articles)

            assert len(results) == 2
            assert results[0].is_sportswear is True
            assert results[1].is_sportswear is False


class TestClassifierPredictionRecord:
    """Tests for ClassifierPredictionRecord dataclass."""

    def test_create_fp_prediction(self):
        """Should create FP prediction record."""
        prediction = ClassifierPredictionRecord(
            classifier_type="fp",
            model_version="RF_tuned_v1",
            probability=0.85,
            prediction=True,
            threshold_used=0.3,
            action_taken="continued_to_llm",
            risk_level="low",
        )

        assert prediction.classifier_type == "fp"
        assert prediction.probability == 0.85
        assert prediction.prediction is True
        assert prediction.action_taken == "continued_to_llm"
        assert prediction.risk_level == "low"
        assert prediction.skip_reason is None
        assert prediction.error_message is None

    def test_create_skipped_prediction(self):
        """Should create prediction with skip reason."""
        prediction = ClassifierPredictionRecord(
            classifier_type="fp",
            model_version="RF_tuned_v1",
            probability=0.15,
            prediction=False,
            threshold_used=0.3,
            action_taken="skipped_llm",
            risk_level="high",
            skip_reason="High-confidence false positive: probability 0.15 < threshold 0.3",
        )

        assert prediction.action_taken == "skipped_llm"
        assert prediction.skip_reason is not None
        assert "0.15" in prediction.skip_reason

    def test_create_failed_prediction(self):
        """Should create prediction with error message."""
        prediction = ClassifierPredictionRecord(
            classifier_type="fp",
            model_version="unknown",
            probability=0.0,
            prediction=False,
            threshold_used=0.3,
            action_taken="failed",
            error_message="Connection refused",
        )

        assert prediction.action_taken == "failed"
        assert prediction.error_message == "Connection refused"


class TestLabelingStatsWithFP:
    """Tests for LabelingStats FP classifier fields."""

    def test_fp_stats_defaults(self):
        """Should have zero FP stats defaults."""
        stats = LabelingStats()
        assert stats.fp_classifier_calls == 0
        assert stats.fp_classifier_skipped == 0
        assert stats.fp_classifier_continued == 0
        assert stats.fp_classifier_errors == 0


class TestLabelingPipelineFPClient:
    """Tests for FP client in LabelingPipeline."""

    def test_init_with_fp_client(self):
        """Should accept FP client in constructor."""
        mock_fp_client = MagicMock()

        with patch("src.labeling.pipeline.db"):
            pipeline = LabelingPipeline(fp_client=mock_fp_client)

            assert pipeline.fp_client == mock_fp_client
            assert pipeline._fp_client_initialized is True

    def test_ensure_fp_client_disabled(self):
        """Should return None when FP classifier is disabled."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = False

                pipeline = LabelingPipeline()
                result = pipeline._ensure_fp_client()

                assert result is None

    def test_ensure_fp_client_lazy_init(self):
        """Should lazily initialize FP client when enabled."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                with patch("src.labeling.pipeline.ClassifierClient") as mock_client_class:
                    mock_settings.fp_classifier_enabled = True
                    mock_settings.fp_classifier_url = "http://localhost:8000"
                    mock_settings.fp_classifier_timeout = 30.0

                    mock_fp_client = MagicMock()
                    mock_client_class.return_value = mock_fp_client

                    pipeline = LabelingPipeline()
                    result = pipeline._ensure_fp_client()

                    assert result == mock_fp_client
                    assert pipeline._fp_client_initialized is True
                    mock_client_class.assert_called_once_with(
                        base_url="http://localhost:8000",
                        timeout=30.0,
                    )


class TestFPPrefilter:
    """Tests for FP pre-filter integration."""

    @pytest.fixture
    def mock_article(self):
        """Create mock article data."""
        return {
            "id": uuid4(),
            "title": "Puma animal spotted in park",
            "full_content": "A puma was spotted in the national park today...",
            "description": "Wildlife sighting",
            "brands_mentioned": ["Puma"],
            "published_at": datetime.now(timezone.utc),
            "source_name": "Wildlife News",
            "category": ["nature", "wildlife"],
        }

    def test_fp_prefilter_skips_llm_on_low_probability(self, mock_article):
        """Should skip LLM for high-confidence false positives."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=False,
                        probability=0.05,  # Very low - definitely not sportswear
                        risk_level="high",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)

                should_continue, prediction = pipeline._run_fp_prefilter(
                    mock_article, dry_run=True
                )

                assert should_continue is False
                assert prediction.action_taken == "skipped_llm"
                assert prediction.probability == 0.05

    def test_fp_prefilter_continues_on_high_probability(self, mock_article):
        """Should continue to LLM for likely sportswear articles."""
        mock_article["title"] = "Nike releases new running shoe"
        mock_article["full_content"] = "Nike announced a new performance running shoe..."
        mock_article["brands_mentioned"] = ["Nike"]

        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=True,
                        probability=0.95,  # High - definitely sportswear
                        risk_level="low",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)

                should_continue, prediction = pipeline._run_fp_prefilter(
                    mock_article, dry_run=True
                )

                assert should_continue is True
                assert prediction.action_taken == "continued_to_llm"
                assert prediction.probability == 0.95

    def test_fp_prefilter_disabled_returns_continue(self, mock_article):
        """Should return continue when FP classifier is disabled."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = False

                pipeline = LabelingPipeline()

                should_continue, prediction = pipeline._run_fp_prefilter(
                    mock_article, dry_run=True
                )

                assert should_continue is True
                assert prediction is None

    def test_fp_prefilter_graceful_degradation_on_error(self, mock_article):
        """Should continue to LLM on classifier error."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.side_effect = Exception("Connection refused")

                pipeline = LabelingPipeline(fp_client=mock_fp_client)

                should_continue, prediction = pipeline._run_fp_prefilter(
                    mock_article, dry_run=True
                )

                # Should continue despite error (graceful degradation)
                assert should_continue is True
                assert prediction.action_taken == "failed"
                assert prediction.error_message == "Connection refused"

    def test_fp_prefilter_passes_all_required_fields(self, mock_article):
        """Should pass all required fields to classifier API."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=True,
                        probability=0.8,
                        risk_level="low",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)
                pipeline._run_fp_prefilter(mock_article, dry_run=True)

                # Verify batch API is called with article data
                mock_fp_client.predict_fp_batch.assert_called_once()
                call_args = mock_fp_client.predict_fp_batch.call_args[0][0]
                assert len(call_args) == 1
                article_data = call_args[0]
                assert article_data["title"] == mock_article["title"]
                assert article_data["content"] == mock_article["full_content"]
                assert article_data["brands"] == mock_article["brands_mentioned"]
                assert article_data["source_name"] == mock_article["source_name"]
                assert article_data["category"] == mock_article["category"]

    def test_fp_prefilter_handles_string_category(self):
        """Should handle string category as well as list."""
        article = {
            "id": uuid4(),
            "title": "Test article",
            "full_content": "Test content",
            "brands_mentioned": ["Nike"],
            "source_name": "Test Source",
            "category": "sports",  # String instead of list
        }

        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=True,
                        probability=0.8,
                        risk_level="low",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)
                pipeline._run_fp_prefilter(article, dry_run=True)

                # Should convert string to list
                call_args = mock_fp_client.predict_fp_batch.call_args[0][0]
                assert call_args[0]["category"] == ["sports"]

    def test_fp_prefilter_uses_description_when_no_full_content(self):
        """Should fall back to description when full_content is None."""
        article = {
            "id": uuid4(),
            "title": "Test article",
            "full_content": None,
            "description": "This is the description",
            "brands_mentioned": ["Nike"],
            "source_name": "Test Source",
            "category": ["sports"],
        }

        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
                mock_settings.fp_classifier_enabled = True
                mock_settings.fp_classifier_url = "http://localhost:8000"
                mock_settings.fp_classifier_timeout = 30.0
                mock_settings.fp_skip_llm_threshold = 0.3

                mock_fp_client = MagicMock()
                mock_fp_client.predict_fp_batch.return_value = [
                    FPPredictionResult(
                        is_sportswear=True,
                        probability=0.8,
                        risk_level="low",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)
                pipeline._run_fp_prefilter(article, dry_run=True)

                # Should use description as content
                call_args = mock_fp_client.predict_fp_batch.call_args[0][0]
                assert call_args[0]["content"] == "This is the description"
