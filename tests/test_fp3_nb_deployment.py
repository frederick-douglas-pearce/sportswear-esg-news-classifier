"""Tests for fp3_nb deployment utilities."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fp3_nb.deployment import (
    _make_json_serializable,
    create_deployment_pipeline,
    load_deployment_artifacts,
    save_deployment_artifacts,
    validate_pipeline,
)


@pytest.fixture
def simple_transformer():
    """Create a real sklearn transformer (StandardScaler) that can be pickled."""
    scaler = StandardScaler()
    # Fit it on some data
    X = np.random.randn(100, 10)
    scaler.fit(X)
    return scaler


@pytest.fixture
def fitted_classifier():
    """Create fitted classifier."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def sample_config():
    """Create sample configuration dict."""
    return {
        "threshold": 0.6,
        "target_recall": 0.98,
        "model_name": "TestModel",
        "cv_f2": 0.95,
        "test_f2": 0.94,
    }


class TestMakeJsonSerializable:
    """Tests for _make_json_serializable function."""

    def test_numpy_int_conversion(self):
        """Test numpy int conversion."""
        data = {"count": np.int64(42)}
        result = _make_json_serializable(data)

        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_numpy_float_conversion(self):
        """Test numpy float conversion."""
        data = {"score": np.float64(0.95)}
        result = _make_json_serializable(data)

        assert result["score"] == 0.95
        assert isinstance(result["score"], float)

    def test_numpy_array_conversion(self):
        """Test numpy array conversion."""
        data = {"array": np.array([1, 2, 3])}
        result = _make_json_serializable(data)

        assert result["array"] == [1, 2, 3]
        assert isinstance(result["array"], list)

    def test_numpy_bool_conversion(self):
        """Test numpy bool conversion."""
        data = {"flag": np.bool_(True)}
        result = _make_json_serializable(data)

        assert result["flag"] is True
        assert isinstance(result["flag"], bool)

    def test_nested_dict(self):
        """Test nested dictionary conversion."""
        data = {
            "outer": {
                "inner": np.int64(10),
                "nested_array": np.array([1.0, 2.0])
            }
        }
        result = _make_json_serializable(data)

        assert result["outer"]["inner"] == 10
        assert result["outer"]["nested_array"] == [1.0, 2.0]

    def test_list_with_numpy(self):
        """Test list containing numpy types."""
        data = {"items": [np.int64(1), np.float64(2.5), "string"]}
        result = _make_json_serializable(data)

        assert result["items"] == [1, 2.5, "string"]

    def test_regular_python_types_unchanged(self):
        """Test that regular Python types are unchanged."""
        data = {"int": 42, "float": 3.14, "str": "hello", "list": [1, 2]}
        result = _make_json_serializable(data)

        assert result == data


class TestCreateDeploymentPipeline:
    """Tests for create_deployment_pipeline function."""

    def test_creates_pipeline(self, simple_transformer, fitted_classifier):
        """Test that function creates sklearn Pipeline."""
        pipeline = create_deployment_pipeline(
            simple_transformer,
            fitted_classifier
        )

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "features"
        assert pipeline.steps[1][0] == "classifier"

    def test_custom_pipeline_name(self, simple_transformer, fitted_classifier):
        """Test custom pipeline name is stored."""
        pipeline = create_deployment_pipeline(
            simple_transformer,
            fitted_classifier,
            pipeline_name="custom_name"
        )

        assert pipeline._pipeline_name == "custom_name"


class TestSaveDeploymentArtifacts:
    """Tests for save_deployment_artifacts function."""

    def test_saves_pipeline_and_config(
        self, tmp_path, simple_transformer, fitted_classifier, sample_config, capsys
    ):
        """Test saving pipeline and config files."""
        pipeline = create_deployment_pipeline(simple_transformer, fitted_classifier)

        paths = save_deployment_artifacts(
            pipeline,
            sample_config,
            tmp_path,
            pipeline_name="test_classifier"
        )

        # Check files exist
        assert paths["pipeline"].exists()
        assert paths["config"].exists()

        # Check file names
        assert paths["pipeline"].name == "test_classifier_pipeline.joblib"
        assert paths["config"].name == "test_classifier_config.json"

        # Check output
        captured = capsys.readouterr()
        assert "Pipeline saved to:" in captured.out
        assert "Configuration saved to:" in captured.out

    def test_config_is_valid_json(
        self, tmp_path, simple_transformer, fitted_classifier, sample_config
    ):
        """Test that saved config is valid JSON."""
        pipeline = create_deployment_pipeline(simple_transformer, fitted_classifier)

        # Add numpy types to config
        config_with_numpy = {
            **sample_config,
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(0.95),
        }

        paths = save_deployment_artifacts(
            pipeline,
            config_with_numpy,
            tmp_path
        )

        # Should be valid JSON
        with open(paths["config"]) as f:
            loaded_config = json.load(f)

        assert loaded_config["numpy_int"] == 42
        assert loaded_config["numpy_float"] == 0.95

    def test_creates_directory_if_needed(
        self, tmp_path, simple_transformer, fitted_classifier, sample_config
    ):
        """Test that directory is created if it doesn't exist."""
        pipeline = create_deployment_pipeline(simple_transformer, fitted_classifier)
        nested_path = tmp_path / "nested" / "models"

        paths = save_deployment_artifacts(
            pipeline,
            sample_config,
            nested_path
        )

        assert nested_path.exists()
        assert paths["pipeline"].exists()


class TestValidatePipeline:
    """Tests for validate_pipeline function."""

    def test_successful_validation(self, capsys):
        """Test successful pipeline validation."""
        # Create a real pipeline with sklearn components
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200))
        ])

        # Fit the pipeline
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        pipeline.fit(X, y)

        # Validate with test data
        test_data = np.random.randn(3, 10)

        results = validate_pipeline(
            pipeline,
            test_data,
            threshold=0.5,
            verbose=True
        )

        assert results["success"] is True
        assert results["n_samples"] == 3
        assert len(results["predictions"]) == 3
        assert len(results["probabilities"]) == 3

        captured = capsys.readouterr()
        assert "PIPELINE VALIDATION" in captured.out
        assert "Validation PASSED" in captured.out

    def test_applies_threshold(self):
        """Test that threshold is applied correctly."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200))
        ])

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        pipeline.fit(X, y)

        test_data = np.random.randn(5, 10)

        # Test with different thresholds
        results_low = validate_pipeline(
            pipeline, test_data, threshold=0.1, verbose=False
        )
        results_high = validate_pipeline(
            pipeline, test_data, threshold=0.9, verbose=False
        )

        # Lower threshold should give more positive predictions
        sum_low = sum(results_low["predictions_at_threshold"])
        sum_high = sum(results_high["predictions_at_threshold"])
        assert sum_low >= sum_high

    def test_handles_error(self, capsys):
        """Test handling of prediction errors."""
        # Create mock pipeline that raises error
        mock_pipeline = MagicMock()
        mock_pipeline.predict.side_effect = ValueError("Test error")

        results = validate_pipeline(
            mock_pipeline,
            ["test text"],
            verbose=False
        )

        assert results["success"] is False
        assert "error" in results
        assert "Test error" in results["error"]

    def test_silent_mode(self, capsys):
        """Test silent validation."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200))
        ])

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        pipeline.fit(X, y)

        validate_pipeline(
            pipeline,
            np.random.randn(2, 10),
            verbose=False
        )

        captured = capsys.readouterr()
        assert "PIPELINE VALIDATION" not in captured.out


class TestLoadDeploymentArtifacts:
    """Tests for load_deployment_artifacts function."""

    def test_loads_saved_artifacts(
        self, tmp_path, simple_transformer, fitted_classifier, sample_config, capsys
    ):
        """Test loading saved artifacts."""
        pipeline = create_deployment_pipeline(simple_transformer, fitted_classifier)

        save_deployment_artifacts(
            pipeline,
            sample_config,
            tmp_path,
            pipeline_name="test_classifier"
        )

        # Clear output from save
        capsys.readouterr()

        # Load artifacts
        loaded = load_deployment_artifacts(
            tmp_path,
            pipeline_name="test_classifier"
        )

        assert "pipeline" in loaded
        assert "config" in loaded
        assert isinstance(loaded["pipeline"], Pipeline)
        assert loaded["config"]["threshold"] == 0.6

        captured = capsys.readouterr()
        assert "Loaded pipeline from:" in captured.out
        assert "Loaded config from:" in captured.out

    def test_missing_pipeline_raises_error(self, tmp_path):
        """Test error when pipeline file is missing."""
        with pytest.raises(FileNotFoundError, match="Pipeline not found"):
            load_deployment_artifacts(tmp_path, pipeline_name="nonexistent")

    def test_missing_config_raises_error(self, tmp_path, simple_transformer, fitted_classifier):
        """Test error when config file is missing."""
        import joblib

        # Save only pipeline, not config
        pipeline = create_deployment_pipeline(simple_transformer, fitted_classifier)
        pipeline_path = tmp_path / "test_classifier_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        with pytest.raises(FileNotFoundError, match="Config not found"):
            load_deployment_artifacts(tmp_path, pipeline_name="test_classifier")
