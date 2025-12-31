"""Tests for the MLOps tracking module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.mlops.tracking import ExperimentTracker


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_mlops_settings():
    """Create mock MLOps settings with MLflow disabled."""
    with patch('src.mlops.tracking.mlops_settings') as mock_settings:
        mock_settings.mlflow_enabled = False
        mock_settings.mlflow_tracking_uri = "file:./mlruns"
        mock_settings.get_experiment_name = MagicMock(return_value="test-experiment")
        yield mock_settings


@pytest.fixture
def mock_mlops_settings_enabled():
    """Create mock MLOps settings with MLflow enabled."""
    with patch('src.mlops.tracking.mlops_settings') as mock_settings:
        mock_settings.mlflow_enabled = True
        mock_settings.mlflow_tracking_uri = "file:./mlruns"
        mock_settings.get_experiment_name = MagicMock(return_value="test-experiment")
        yield mock_settings


@pytest.fixture
def disabled_tracker(mock_mlops_settings):
    """Create a tracker with MLflow disabled."""
    return ExperimentTracker("fp")


# ============================================================================
# Initialization Tests
# ============================================================================

class TestExperimentTrackerInit:
    """Tests for ExperimentTracker initialization."""

    def test_init_with_disabled_mlflow(self, mock_mlops_settings):
        """Test initialization when MLflow is disabled."""
        tracker = ExperimentTracker("fp")

        assert tracker.classifier_type == "fp"
        assert tracker.experiment_name == "test-experiment"
        assert tracker.enabled is False
        assert tracker._run is None

    def test_init_with_enabled_mlflow(self, mock_mlops_settings_enabled):
        """Test initialization when MLflow is enabled but not installed."""
        with patch.dict('sys.modules', {'mlflow': None}):
            tracker = ExperimentTracker("ep")

            assert tracker.classifier_type == "ep"
            # Should gracefully degrade when mlflow import fails
            assert tracker.enabled is False

    def test_init_creates_experiment_name(self, mock_mlops_settings):
        """Test that experiment name is created from classifier type."""
        tracker = ExperimentTracker("esg")

        mock_mlops_settings.get_experiment_name.assert_called_once_with("esg")


# ============================================================================
# Graceful Degradation Tests
# ============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation when MLflow is disabled."""

    def test_start_run_noop_when_disabled(self, disabled_tracker):
        """Test start_run is a no-op when disabled."""
        with disabled_tracker.start_run(run_name="test-run") as tracker:
            assert tracker is disabled_tracker
            assert tracker._run is None

    def test_log_params_noop_when_disabled(self, disabled_tracker):
        """Test log_params is a no-op when disabled."""
        # Should not raise any errors
        disabled_tracker.log_params({"n_estimators": 100, "max_depth": 5})

    def test_log_metrics_noop_when_disabled(self, disabled_tracker):
        """Test log_metrics is a no-op when disabled."""
        disabled_tracker.log_metrics({"f2_score": 0.95, "recall": 0.98})

    def test_log_artifact_noop_when_disabled(self, disabled_tracker):
        """Test log_artifact is a no-op when disabled."""
        disabled_tracker.log_artifact("/some/path/model.joblib")

    def test_log_model_config_noop_when_disabled(self, disabled_tracker):
        """Test log_model_config is a no-op when disabled."""
        disabled_tracker.log_model_config({"threshold": 0.5})

    def test_log_sklearn_model_noop_when_disabled(self, disabled_tracker):
        """Test log_sklearn_model is a no-op when disabled."""
        mock_model = MagicMock()
        disabled_tracker.log_sklearn_model(mock_model)

    def test_set_tag_noop_when_disabled(self, disabled_tracker):
        """Test set_tag is a no-op when disabled."""
        disabled_tracker.set_tag("key", "value")

    def test_get_run_id_returns_none_when_disabled(self, disabled_tracker):
        """Test get_run_id returns None when disabled."""
        assert disabled_tracker.get_run_id() is None


# ============================================================================
# Flatten Dict Utility Tests
# ============================================================================

class TestFlattenDict:
    """Tests for the _flatten_dict utility."""

    def test_flatten_simple_dict(self):
        """Test flattening a simple flat dictionary."""
        d = {"a": 1, "b": "hello", "c": 3.14}
        result = ExperimentTracker._flatten_dict(d)

        assert result == {"a": "1", "b": "hello", "c": "3.14"}

    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        d = {
            "model": {
                "n_estimators": 100,
                "max_depth": 5,
            },
            "threshold": 0.5,
        }
        result = ExperimentTracker._flatten_dict(d)

        assert result == {
            "model.n_estimators": "100",
            "model.max_depth": "5",
            "threshold": "0.5",
        }

    def test_flatten_deeply_nested_dict(self):
        """Test flattening a deeply nested dictionary."""
        d = {
            "level1": {
                "level2": {
                    "level3": "value",
                },
            },
        }
        result = ExperimentTracker._flatten_dict(d)

        assert result == {"level1.level2.level3": "value"}

    def test_flatten_empty_dict(self):
        """Test flattening an empty dictionary."""
        result = ExperimentTracker._flatten_dict({})
        assert result == {}

    def test_flatten_with_custom_separator(self):
        """Test flattening with a custom separator."""
        d = {"outer": {"inner": "value"}}
        result = ExperimentTracker._flatten_dict(d, sep="_")

        assert result == {"outer_inner": "value"}


# ============================================================================
# MLflow Integration Tests (Mocked)
# ============================================================================

class TestMlflowIntegration:
    """Tests for MLflow integration with mocked MLflow."""

    def test_start_run_calls_mlflow(self, mock_mlops_settings_enabled):
        """Test start_run calls MLflow methods."""
        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name = MagicMock(return_value=None)
        mock_mlflow.create_experiment = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run = MagicMock(return_value=mock_run)
        mock_mlflow.end_run = MagicMock()

        with patch.dict('sys.modules', {'mlflow': mock_mlflow}):
            tracker = ExperimentTracker("fp")
            tracker.enabled = True
            tracker._mlflow = mock_mlflow

            with tracker.start_run(run_name="test-run"):
                assert tracker._run == mock_run

            mock_mlflow.start_run.assert_called_once()
            mock_mlflow.end_run.assert_called_once()

    def test_log_params_calls_mlflow(self, mock_mlops_settings_enabled):
        """Test log_params calls MLflow log_params."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        params = {"n_estimators": 100, "max_depth": 5}
        tracker.log_params(params)

        mock_mlflow.log_params.assert_called_once()
        call_args = mock_mlflow.log_params.call_args[0][0]
        assert call_args["n_estimators"] == "100"
        assert call_args["max_depth"] == "5"

    def test_log_metrics_calls_mlflow(self, mock_mlops_settings_enabled):
        """Test log_metrics calls MLflow log_metrics."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        metrics = {"f2_score": 0.95, "recall": 0.98}
        tracker.log_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=None)

    def test_log_metrics_with_step(self, mock_mlops_settings_enabled):
        """Test log_metrics with step parameter."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        metrics = {"loss": 0.1}
        tracker.log_metrics(metrics, step=5)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=5)

    def test_log_artifact_calls_mlflow(self, mock_mlops_settings_enabled):
        """Test log_artifact calls MLflow log_artifact."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        tracker.log_artifact("/path/to/model.joblib")

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/model.joblib", None)

    def test_log_artifact_with_path(self, mock_mlops_settings_enabled):
        """Test log_artifact with artifact_path parameter."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        tracker.log_artifact("/path/to/model.joblib", "models")

        mock_mlflow.log_artifact.assert_called_once_with("/path/to/model.joblib", "models")

    def test_set_tag_calls_mlflow(self, mock_mlops_settings_enabled):
        """Test set_tag calls MLflow set_tag."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        tracker.set_tag("version", "1.0.0")

        mock_mlflow.set_tag.assert_called_once_with("version", "1.0.0")

    def test_get_run_id_returns_id(self, mock_mlops_settings_enabled):
        """Test get_run_id returns the run ID."""
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._run = mock_run

        assert tracker.get_run_id() == "abc123"


# ============================================================================
# Context Manager Tests
# ============================================================================

class TestContextManager:
    """Tests for context manager behavior."""

    def test_context_manager_yields_tracker(self, disabled_tracker):
        """Test context manager yields the tracker instance."""
        with disabled_tracker.start_run() as tracker:
            assert tracker is disabled_tracker

    def test_context_manager_handles_exception(self, mock_mlops_settings_enabled):
        """Test context manager properly cleans up on exception."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_mlflow.start_run = MagicMock(return_value=mock_run)
        mock_mlflow.end_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = None
        tracker.classifier_type = "fp"

        with pytest.raises(ValueError):
            with tracker.start_run():
                raise ValueError("Test error")

        # end_run should still be called
        mock_mlflow.end_run.assert_called_once()


# ============================================================================
# Log Model Config Tests
# ============================================================================

class TestLogModelConfig:
    """Tests for log_model_config method."""

    def test_log_model_config_creates_temp_file(self, mock_mlops_settings_enabled):
        """Test log_model_config creates and logs a temp file."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker.enabled = True
        tracker._mlflow = mock_mlflow
        tracker._run = mock_run

        config = {"threshold": 0.5, "model": "RandomForest"}
        tracker.log_model_config(config)

        # Should have called log_artifact with a temp file path
        mock_mlflow.log_artifact.assert_called_once()
        call_args = mock_mlflow.log_artifact.call_args
        assert call_args[1] == {} or call_args[0][1] == "config"
