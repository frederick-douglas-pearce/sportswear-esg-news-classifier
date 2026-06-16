"""Tests for experiment tracking integration in model_training workflow."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflows.model_training import (
    check_data_quality,
    compare_models,
    finalize_experiments,
)


@pytest.fixture
def models_dir(tmp_path):
    """Create models directory with test configs."""
    models = tmp_path / "models"
    models.mkdir()

    registry = {
        "fp": {
            "production": "v2.1.0",
            "versions": {
                "v2.1.0": {"metrics": {"test_f2": 0.974}}
            },
        }
    }
    with open(models / "registry.json", "w") as f:
        json.dump(registry, f)

    with open(models / "fp_feature_config.json", "w") as f:
        json.dump({"features": ["tfidf"]}, f)

    with open(models / "fp_classifier_config.json", "w") as f:
        json.dump({
            "model_name": "RandomForest",
            "test_f2": 0.976,
            "test_recall": 0.990,
            "test_precision": 0.925,
            "cv_f2": 0.968,
            "threshold": 0.45,
        }, f)

    return models


@pytest.fixture
def data_dir(tmp_path):
    """Create data directory with training data."""
    data = tmp_path / "data"
    data.mkdir()

    # Write training data file
    with open(data / "fp_training_data.jsonl", "w") as f:
        for i in range(100):
            record = {"title": f"article {i}", "is_sportswear": i % 3 != 0}
            f.write(json.dumps(record) + "\n")

    return data


@pytest.fixture
def mock_workflow():
    """Create a mock workflow object."""
    workflow = MagicMock()
    workflow.name = "model_training"
    workflow.dry_run = False
    workflow._workflow_state = MagicMock()
    return workflow


class TestCheckDataQualityCreatesExperiments:
    @patch("src.agent.workflows.model_training._get_tracker_class")
    def test_creates_experiments(self, mock_get_tracker, mock_workflow, data_dir):
        """Verify create_experiment is called for successful exports."""
        mock_tracker = MagicMock()
        mock_tracker.create_experiment.return_value = "fp_20260226_143000"
        mock_get_tracker.return_value = MagicMock(return_value=mock_tracker)

        context = {
            "export_success": True,
            "datasets": {
                "fp": {"success": True, "record_count": 2318},
            },
        }

        # Patch Path to use our data dir
        with patch(
            "src.agent.workflows.model_training.Path",
            side_effect=lambda x: data_dir if x == "data" else Path(x),
        ):
            result = check_data_quality(mock_workflow, context)

        assert "experiment_ids" in result
        assert result["experiment_ids"]["fp"] == "fp_20260226_143000"
        mock_tracker.create_experiment.assert_called_once()

    @patch("src.agent.workflows.model_training._get_tracker_class")
    def test_tracker_failure_doesnt_break_quality_check(
        self, mock_get_tracker, mock_workflow, data_dir
    ):
        """Verify quality check still returns normally if tracker fails."""
        mock_get_tracker.side_effect = Exception("tracker import failed")

        context = {
            "export_success": True,
            "datasets": {
                "fp": {"success": True, "record_count": 2318},
            },
        }

        with patch(
            "src.agent.workflows.model_training.Path",
            side_effect=lambda x: data_dir if x == "data" else Path(x),
        ):
            result = check_data_quality(mock_workflow, context)

        # Quality check still succeeded
        assert "datasets" in result
        # No experiment_ids since tracker failed
        assert "experiment_ids" not in result


class TestCompareModelsRecordsObservation:
    @patch("src.agent.workflows.model_training._get_tracker_class")
    def test_records_observation(self, mock_get_tracker, mock_workflow, models_dir):
        """Verify record_observation is called with comparison data."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = MagicMock(return_value=mock_tracker)

        context = {
            "classifiers": ["fp"],
            "experiment_ids": {"fp": "fp_20260226_143000"},
        }

        # Patch Path so compare_models reads from our test models_dir
        original_path = Path

        def patched_path(x):
            if x == "models":
                return models_dir
            return original_path(x)

        with patch(
            "src.agent.workflows.model_training.Path",
            side_effect=patched_path,
        ):
            result = compare_models(mock_workflow, context)

        mock_tracker.resume.assert_called_once_with("fp_20260226_143000")
        mock_tracker.record_observation.assert_called_once_with(result)


class TestFinalizeExperiments:
    @patch("src.agent.workflows.model_training.agent_settings")
    @patch("src.agent.workflows.model_training._get_reflector_class")
    @patch("src.agent.workflows.model_training._get_tracker_class")
    def test_records_reward_and_completes(
        self, mock_get_tracker, mock_get_reflector, mock_settings, mock_workflow
    ):
        """Verify reward recording, reflection, and completion."""
        mock_tracker = MagicMock()
        mock_tracker.experiment = MagicMock()
        mock_get_tracker.return_value = MagicMock(return_value=mock_tracker)

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = MagicMock(summary="Good result")
        mock_get_reflector.return_value = MagicMock(return_value=mock_reflector)

        mock_settings.llm_analysis_enabled = True
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.llm_analysis_model = "claude-sonnet-4-6"

        context = {
            "experiment_ids": {"fp": "fp_20260226_143000"},
            "promoted": ["fp"],
            "classifiers": {},
        }

        result = finalize_experiments(mock_workflow, context)

        assert "fp_20260226_143000" in result["experiments_finalized"]
        mock_tracker.resume.assert_called_once_with("fp_20260226_143000")
        mock_tracker.record_reward.assert_called_once()
        mock_tracker.record_reflection.assert_called_once()
        mock_tracker.complete.assert_called_once()

    @patch("src.agent.workflows.model_training.agent_settings")
    @patch("src.agent.workflows.model_training._get_tracker_class")
    def test_tracker_failure_doesnt_break_finalization(
        self, mock_get_tracker, mock_settings, mock_workflow
    ):
        """Verify step returns normally even if tracker fails."""
        mock_get_tracker.return_value = MagicMock(
            side_effect=Exception("tracker broken")
        )
        mock_settings.llm_analysis_enabled = False

        context = {
            "experiment_ids": {"fp": "fp_20260226_143000"},
            "promoted": [],
        }

        result = finalize_experiments(mock_workflow, context)

        # Should return empty finalized list, not raise
        assert result["experiments_finalized"] == []
