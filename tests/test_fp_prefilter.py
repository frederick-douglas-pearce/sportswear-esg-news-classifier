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
                "confidence_level": "low",
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
            assert result.confidence_level == "low"
            assert result.threshold == 0.3

    def test_predict_fp_batch_success(self):
        """Should parse batch FP prediction response."""
        client = ClassifierClient("http://localhost:8000")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "predictions": [
                    {"is_sportswear": True, "probability": 0.9, "confidence_level": "low", "threshold": 0.3},
                    {"is_sportswear": False, "probability": 0.1, "confidence_level": "high", "threshold": 0.3},
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
            confidence_level="low",
        )

        assert prediction.classifier_type == "fp"
        assert prediction.probability == 0.85
        assert prediction.prediction is True
        assert prediction.action_taken == "continued_to_llm"
        assert prediction.confidence_level == "low"
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
            confidence_level="low",
            skip_reason="Likely false positive (low risk): probability 0.150 < threshold 0.3",
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
        """Should skip LLM when the probability is below the threshold."""
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
                        confidence_level="high",
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
                        confidence_level="low",
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
                        confidence_level="low",
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
                        confidence_level="low",
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
                        confidence_level="low",
                        threshold=0.3,
                    )
                ]
                mock_fp_client.get_model_info.return_value = {"version": "1.0"}

                pipeline = LabelingPipeline(fp_client=mock_fp_client)
                pipeline._run_fp_prefilter(article, dry_run=True)

                # Should use description as content
                call_args = mock_fp_client.predict_fp_batch.call_args[0][0]
                assert call_args[0]["content"] == "This is the description"


def _prefilter(probability, confidence_level, threshold):
    """Run the FP pre-filter on one article with a mocked classifier response."""
    article = {
        "id": uuid4(),
        "title": "Watchdog group names corporate campaigns",
        "full_content": "Article body " * 20,
        "description": "Description",
        "brands_mentioned": ["Nike"],
        "published_at": datetime.now(timezone.utc),
        "source_name": "Test Source",
        "category": [],
    }
    with patch("src.labeling.pipeline.db"):
        with patch("src.labeling.pipeline.labeling_settings") as mock_settings:
            mock_settings.fp_classifier_enabled = True
            mock_settings.fp_skip_llm_threshold = threshold
            mock_fp_client = MagicMock()
            mock_fp_client.predict_fp_batch.return_value = [
                FPPredictionResult(
                    is_sportswear=probability >= threshold,
                    probability=probability,
                    confidence_level=confidence_level,
                    threshold=threshold,
                )
            ]
            mock_fp_client.get_model_info.return_value = {"version": "1.0"}
            pipeline = LabelingPipeline(fp_client=mock_fp_client)
            return pipeline._run_fp_prefilter(article, dry_run=True)


class TestFPSkipReasonWording:
    """A skip reason states the stored risk band and the numbers, nothing else (#116, D024)."""

    def test_issue_case_medium_band_is_not_called_high_confidence(self):
        """The near-threshold miss from the issue: p=0.467, threshold 0.53, medium."""
        should_continue, prediction = _prefilter(0.467, "medium", 0.53)

        assert should_continue is False
        assert prediction.skip_reason == (
            "Uncertain skip (medium risk): probability 0.467 < threshold 0.53"
        )
        assert prediction.confidence_level == "medium"

    def test_low_band(self):
        _, prediction = _prefilter(0.02, "low", 0.53)
        assert prediction.skip_reason == (
            "Likely false positive (low risk): probability 0.020 < threshold 0.53"
        )

    def test_high_band_skip_above_a_high_threshold(self):
        """Only reachable when the threshold exceeds the API's 0.6 band edge."""
        _, prediction = _prefilter(0.62, "high", 0.65)
        assert prediction.skip_reason == (
            "Below FP threshold (high risk): probability 0.620 < threshold 0.65"
        )

    @pytest.mark.parametrize("band", [None, "LOW", "very-long-unexpected-band-" * 20])
    def test_missing_or_unrecognised_band_asserts_no_band(self, band):
        """An unrecognised band string is never copied into the reason."""
        _, prediction = _prefilter(0.2, band, 0.53)
        assert prediction.skip_reason == (
            "Below FP threshold (risk band unknown): probability 0.200 < threshold 0.53"
        )

    @pytest.mark.parametrize("band", ["low", "medium", "high", None])
    def test_no_reason_claims_high_confidence(self, band):
        _, prediction = _prefilter(0.1, band, 0.7)
        assert "confidence" not in prediction.skip_reason.lower()

    def test_every_variant_fits_the_skip_reason_column(self):
        """classifier_predictions.skip_reason is String(255)."""
        from src.data_collection.models import ClassifierPrediction
        from src.labeling.pipeline import _FP_SKIP_LEAD_UNKNOWN, _FP_SKIP_LEADS, _fp_skip_reason

        limit = ClassifierPrediction.__table__.c.skip_reason.type.length
        for band in [*_FP_SKIP_LEADS, None]:
            reason = _fp_skip_reason(0.123456789, 0.123456789012345, band)
            assert len(reason) <= limit
        assert len(_FP_SKIP_LEAD_UNKNOWN) < limit


class TestFPSkipDecisionUnchanged:
    """#116 changes the wording only; routing stays #142's (AC3)."""

    def test_probability_equal_to_threshold_continues(self):
        should_continue, prediction = _prefilter(0.53, "medium", 0.53)
        assert should_continue is True
        assert prediction.action_taken == "continued_to_llm"
        assert prediction.skip_reason is None

    def test_just_below_threshold_skips_whatever_the_band(self):
        for band in ["low", "medium", "high", None]:
            should_continue, prediction = _prefilter(0.529, band, 0.53)
            assert should_continue is False
            assert prediction.action_taken == "skipped_llm"


class TestNotLowRiskSkipCount:
    """Skips outside the `low` band are counted, as a subset of all skips (#116)."""

    def _process_skip(self, band):
        mock_database = MagicMock()
        pipeline = LabelingPipeline(database=mock_database)
        article = {
            "id": uuid4(),
            "title": "t",
            "full_content": "Article body " * 20,
            "description": "d",
            "brands_mentioned": ["Nike"],
        }
        prediction = ClassifierPredictionRecord(
            classifier_type="fp",
            model_version="1.0",
            probability=0.4,
            prediction=False,
            threshold_used=0.53,
            action_taken="skipped_llm",
            confidence_level=band,
        )
        return pipeline._process_article(
            article, dry_run=True, fp_prefilter_result=(False, prediction)
        )

    @pytest.mark.parametrize(
        "band,expected", [("low", False), ("medium", True), ("high", True), (None, True)]
    )
    def test_skip_flag_by_band(self, band, expected):
        result = self._process_skip(band)
        assert result["fp_classifier_skipped"] is True
        assert result["fp_classifier_skipped_not_low"] is expected

    def test_label_articles_counts_not_low_skips_only_among_skips(self):
        """medium and None skips count; a low skip and a medium continue do not."""
        results = [
            {"fp_classifier_called": True, "fp_classifier_skipped": True,
             "fp_classifier_skipped_not_low": True, "false_positive": True},
            {"fp_classifier_called": True, "fp_classifier_skipped": True,
             "fp_classifier_skipped_not_low": True, "false_positive": True},
            {"fp_classifier_called": True, "fp_classifier_skipped": True,
             "fp_classifier_skipped_not_low": False, "false_positive": True},
            # A stray flag on a non-skip must not be counted.
            {"fp_classifier_called": True, "fp_classifier_continued": True,
             "fp_classifier_skipped_not_low": True},
        ]
        for r in results:
            r.setdefault("labeled", False)
            r.setdefault("skipped", False)

        articles = [{"id": uuid4()} for _ in results]
        mock_database = MagicMock()
        mock_database.get_articles_pending_labeling.return_value = []
        pipeline = LabelingPipeline(database=mock_database)
        with patch("src.labeling.pipeline.db"), \
             patch.object(pipeline, "_ensure_labeler"), \
             patch.object(pipeline, "_deduplicate_by_title", return_value=(articles, [])), \
             patch.object(pipeline, "_compute_novelty_batch", return_value={}), \
             patch.object(pipeline, "_run_fp_prefilter_batch", return_value={}), \
             patch.object(pipeline, "_process_article", side_effect=results):
            stats = pipeline.label_articles(dry_run=True)

        assert stats.fp_classifier_skipped == 3
        assert stats.fp_classifier_skipped_not_low == 2
        assert stats.fp_classifier_continued == 1

    def test_stats_default(self):
        assert LabelingStats().fp_classifier_skipped_not_low == 0
