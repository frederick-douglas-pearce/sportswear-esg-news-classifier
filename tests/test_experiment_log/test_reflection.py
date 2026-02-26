"""Tests for ExperimentReflector."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.experiment_log.models import (
    ExperimentAction,
    ExperimentEntry,
    ExperimentObservation,
    ExperimentReward,
    ExperimentState,
)
from src.experiment_log.reflection import ExperimentReflector


@pytest.fixture
def experiment():
    """Create a completed experiment for reflection."""
    return ExperimentEntry(
        experiment_id="fp_20260226_143000",
        classifier="fp",
        created_at=datetime(2026, 2, 26, 14, 30, tzinfo=timezone.utc),
        status="completed",
        state=ExperimentState(
            production_version="v2.1.0",
            production_metrics={"test_f2": 0.974},
            training_data={"total_records": 2318},
        ),
        action=ExperimentAction(
            type="automated_retraining",
            summary="Automated retraining of fp classifier with 2318 records",
        ),
        observation=ExperimentObservation(
            metrics={"test_f2": 0.976, "test_recall": 0.990},
            delta={"test_f2_delta": 0.002},
        ),
        reward=ExperimentReward(
            promoted=True,
            promoted_as="v2.2.0",
            target_met=True,
        ),
    )


def _make_mock_response(text):
    """Create a mock Anthropic response."""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = text
    mock_response.content = [mock_content]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    return mock_response


class TestReflectSuccess:
    @patch("src.experiment_log.reflection.Anthropic")
    def test_valid_json(self, mock_anthropic_cls, experiment):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        response_data = {
            "summary": "Minor improvement in F2 score",
            "hypothesis_result": "confirmed",
            "hypothesis_evidence": "F2 improved by 0.002",
            "surprises": ["Recall improved more than expected"],
            "next_steps": ["Monitor drift for 2 weeks"],
            "confidence": "high",
            "confidence_notes": "Clear improvement across metrics",
        }
        mock_client.messages.create.return_value = _make_mock_response(
            json.dumps(response_data)
        )

        reflector = ExperimentReflector(api_key="test-key")
        result = reflector.reflect(experiment)

        assert result.summary == "Minor improvement in F2 score"
        assert result.hypothesis_result == "confirmed"
        assert result.confidence == "high"
        assert len(result.surprises) == 1
        assert len(result.next_steps) == 1

    @patch("src.experiment_log.reflection.Anthropic")
    def test_json_in_code_block(self, mock_anthropic_cls, experiment):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        response_text = '```json\n{"summary": "Good result", "hypothesis_result": "confirmed"}\n```'
        mock_client.messages.create.return_value = _make_mock_response(response_text)

        reflector = ExperimentReflector(api_key="test-key")
        result = reflector.reflect(experiment)

        assert result.summary == "Good result"
        assert result.hypothesis_result == "confirmed"


class TestReflectRetry:
    @patch("src.experiment_log.reflection.time")
    @patch("src.experiment_log.reflection.Anthropic")
    def test_rate_limit_retry(self, mock_anthropic_cls, mock_time, experiment):
        from anthropic import RateLimitError

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # Create a proper RateLimitError
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        rate_error = RateLimitError(
            response=mock_resp,
            body={"error": {"message": "rate limited", "type": "rate_limit_error"}},
            message="rate limited",
        )

        response_data = {"summary": "Success after retry", "hypothesis_result": "confirmed"}
        success_response = _make_mock_response(json.dumps(response_data))

        mock_client.messages.create.side_effect = [
            rate_error,
            rate_error,
            success_response,
        ]

        reflector = ExperimentReflector(api_key="test-key", max_retries=3)
        result = reflector.reflect(experiment)

        assert result.summary == "Success after retry"
        assert mock_time.sleep.call_count == 2


class TestReflectFailure:
    @patch("src.experiment_log.reflection.Anthropic")
    def test_api_failure(self, mock_anthropic_cls, experiment):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API down")

        reflector = ExperimentReflector(api_key="test-key")
        result = reflector.reflect(experiment)

        assert "failed" in result.summary.lower()
        assert result.confidence == "low"

    @patch("src.experiment_log.reflection.Anthropic")
    def test_invalid_json(self, mock_anthropic_cls, experiment):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            "This is not JSON at all"
        )

        reflector = ExperimentReflector(api_key="test-key")
        result = reflector.reflect(experiment)

        # Should return empty reflection (no JSON parsed)
        assert result.summary == ""
        assert result.hypothesis_result == "inconclusive"
