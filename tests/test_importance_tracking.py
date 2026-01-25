"""Tests for the importance tracking module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.mlops.importance_tracking import (
    ImportanceHistory,
    ImportanceRecord,
    StabilityAnalysis,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_shap_importance():
    """Sample SHAP importance dict."""
    return {
        "group_importance": {"lsa_features": 0.5, "ner_features": 0.3, "brand_features": 0.2},
        "group_contribution_pct": {"lsa_features": 50.0, "ner_features": 30.0, "brand_features": 20.0},
        "top_groups": [("lsa_features", 0.5), ("ner_features", 0.3), ("brand_features", 0.2)],
    }


@pytest.fixture
def sample_perm_importance():
    """Sample permutation importance dict."""
    return {
        "baseline_score": 0.95,
        "group_importance": {"lsa_features": 0.1, "ner_features": 0.05, "brand_features": 0.02},
        "group_std": {"lsa_features": 0.02, "ner_features": 0.01, "brand_features": 0.005},
        "group_contribution_pct": {"lsa_features": 58.8, "ner_features": 29.4, "brand_features": 11.8},
        "top_groups": [
            ("lsa_features", 0.1, 0.02),
            ("ner_features", 0.05, 0.01),
            ("brand_features", 0.02, 0.005),
        ],
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics dict."""
    return {"f2": 0.985, "recall": 0.992, "precision": 0.960}


@pytest.fixture
def temp_history_dir():
    """Create a temporary directory for history files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# ImportanceRecord Tests
# ============================================================================


class TestImportanceRecord:
    """Tests for ImportanceRecord dataclass."""

    def test_create_record(self, sample_shap_importance, sample_perm_importance, sample_metrics):
        """Test creating an ImportanceRecord."""
        record = ImportanceRecord(
            timestamp="2026-01-24T10:00:00",
            classifier_type="fp",
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
        )

        assert record.classifier_type == "fp"
        assert record.metrics["f2"] == 0.985
        assert len(record.shap_importance["top_groups"]) == 3

    def test_record_to_dict(self, sample_shap_importance, sample_perm_importance, sample_metrics):
        """Test converting record to dictionary."""
        record = ImportanceRecord(
            timestamp="2026-01-24T10:00:00",
            classifier_type="fp",
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
            notes="Test run",
        )

        data = record.to_dict()

        assert data["classifier_type"] == "fp"
        assert data["version"] == "v1.0.0"
        assert data["notes"] == "Test run"

    def test_record_from_dict(self, sample_shap_importance, sample_perm_importance, sample_metrics):
        """Test creating record from dictionary."""
        data = {
            "timestamp": "2026-01-24T10:00:00",
            "classifier_type": "fp",
            "shap_importance": sample_shap_importance,
            "perm_importance": sample_perm_importance,
            "metrics": sample_metrics,
            "version": "v1.0.0",
            "model_params": {"n_estimators": 100},
            "data_info": {"n_train": 1000},
            "feature_method": "tfidf_lsa_ner_proximity_brands",
            "notes": "",
        }

        record = ImportanceRecord.from_dict(data)

        assert record.classifier_type == "fp"
        assert record.model_params["n_estimators"] == 100
        assert record.feature_method == "tfidf_lsa_ner_proximity_brands"

    def test_record_roundtrip(self, sample_shap_importance, sample_perm_importance, sample_metrics):
        """Test record roundtrip (to_dict -> from_dict)."""
        original = ImportanceRecord(
            timestamp="2026-01-24T10:00:00",
            classifier_type="fp",
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
            model_params={"n_estimators": 100},
            data_info={"n_train": 1000},
            feature_method="tfidf_lsa_ner_proximity_brands",
            notes="Test",
        )

        data = original.to_dict()
        restored = ImportanceRecord.from_dict(data)

        assert restored.classifier_type == original.classifier_type
        assert restored.version == original.version
        assert restored.metrics == original.metrics


# ============================================================================
# ImportanceHistory Tests
# ============================================================================


class TestImportanceHistory:
    """Tests for ImportanceHistory class."""

    def test_create_history(self, temp_history_dir):
        """Test creating a new history."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        assert history.classifier_type == "fp"
        assert len(history) == 0

    def test_add_record(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test adding a record to history."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        record = history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            model_params={"n_estimators": 100},
            version="v1.0.0",
        )

        assert len(history) == 1
        assert record.classifier_type == "fp"
        assert record.metrics["f2"] == 0.985

    def test_save_and_load(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test saving and loading history."""
        # Create and save
        history1 = ImportanceHistory("fp", history_dir=temp_history_dir)
        history1.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
        )
        history1.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics={"f2": 0.990, "recall": 0.995, "precision": 0.970},
            version="v1.1.0",
        )
        saved_path = history1.save()

        assert saved_path.exists()

        # Load
        history2 = ImportanceHistory.load("fp", history_dir=temp_history_dir)

        assert len(history2) == 2
        assert history2.records[0].version == "v1.0.0"
        assert history2.records[1].version == "v1.1.0"

    def test_get_latest_record(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test getting the latest record."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
        )
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics={"f2": 0.990, "recall": 0.995, "precision": 0.970},
            version="v1.1.0",
        )

        latest = history.get_latest_record()

        assert latest is not None
        assert latest.version == "v1.1.0"
        assert latest.metrics["f2"] == 0.990

    def test_get_latest_record_empty_history(self, temp_history_dir):
        """Test getting latest record from empty history."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        assert history.get_latest_record() is None

    def test_get_records_by_version(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test filtering records by version."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
        )
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.1.0",
        )
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",  # Another v1.0.0
        )

        v1_records = history.get_records_by_version("v1.0.0")

        assert len(v1_records) == 2

    def test_summary(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test history summary."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
            version="v1.0.0",
        )

        summary = history.summary()

        assert "fp" in summary
        assert "Records: 1" in summary
        assert "F2:" in summary

    def test_summary_empty_history(self, temp_history_dir):
        """Test summary with empty history."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        summary = history.summary()

        assert "No importance history" in summary

    def test_history_file_is_json_serializable(
        self, temp_history_dir, sample_shap_importance, sample_perm_importance, sample_metrics
    ):
        """Test that saved history is valid JSON."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)
        history.add_record(
            shap_importance=sample_shap_importance,
            perm_importance=sample_perm_importance,
            metrics=sample_metrics,
        )
        saved_path = history.save()

        # Should be valid JSON
        with open(saved_path) as f:
            data = json.load(f)

        assert "records" in data
        assert len(data["records"]) == 1


# ============================================================================
# StabilityAnalysis Tests
# ============================================================================


class TestStabilityAnalysis:
    """Tests for stability analysis."""

    def test_analyze_stability_insufficient_runs(self, temp_history_dir):
        """Test stability analysis with insufficient runs."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        # Add only 2 records (less than min_runs=3)
        for i in range(2):
            history.add_record(
                shap_importance={
                    "group_contribution_pct": {"a": 50.0, "b": 30.0, "c": 20.0}
                },
                perm_importance={
                    "baseline_score": 0.95,
                    "group_contribution_pct": {"a": 50.0, "b": 30.0, "c": 20.0},
                },
                metrics={"f2": 0.98},
            )

        result = history.analyze_stability(min_runs=3)

        assert result is None

    def test_analyze_stability_basic(self, temp_history_dir):
        """Test basic stability analysis."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        # Add varied importance values
        test_data = [
            {"a": 50.0, "b": 30.0, "c": 20.0},
            {"a": 55.0, "b": 25.0, "c": 20.0},
            {"a": 45.0, "b": 35.0, "c": 20.0},
            {"a": 52.0, "b": 28.0, "c": 20.0},
        ]

        for data in test_data:
            history.add_record(
                shap_importance={"group_contribution_pct": data},
                perm_importance={
                    "baseline_score": 0.95,
                    "group_contribution_pct": data,
                },
                metrics={"f2": 0.98},
            )

        result = history.analyze_stability()

        assert result is not None
        assert result.n_runs == 4
        assert "a" in result.shap_mean
        assert "a" in result.perm_mean
        assert "a" in result.perm_cv

    def test_analyze_stability_identifies_stable_features(self, temp_history_dir):
        """Test that stability analysis identifies most/least stable features."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        # Create data where 'stable' has low variance and 'variable' has high variance
        for i in range(5):
            history.add_record(
                shap_importance={
                    "group_contribution_pct": {
                        "stable": 50.0,  # Always 50
                        "variable": 30.0 + i * 10,  # Varies from 30 to 70
                        "medium": 20.0 + i * 2,  # Moderate variance
                    }
                },
                perm_importance={
                    "baseline_score": 0.95,
                    "group_contribution_pct": {
                        "stable": 50.0,
                        "variable": 30.0 + i * 10,
                        "medium": 20.0 + i * 2,
                    },
                },
                metrics={"f2": 0.98},
            )

        result = history.analyze_stability()

        # 'stable' should have CV = 0 (no variance)
        assert result.perm_cv["stable"] == 0.0

        # 'stable' should be in most stable groups
        most_stable_names = [g for g, _ in result.most_stable_groups]
        assert "stable" in most_stable_names

        # 'variable' should be in least stable groups
        least_stable_names = [g for g, _ in result.least_stable_groups]
        assert "variable" in least_stable_names

    def test_analyze_stability_correlation_matrix(self, temp_history_dir):
        """Test that correlation matrix is computed."""
        history = ImportanceHistory("fp", history_dir=temp_history_dir)

        # Create perfectly correlated data
        for i in range(5):
            val = 50.0 + i * 5
            history.add_record(
                shap_importance={
                    "group_contribution_pct": {"a": val, "b": 100 - val}
                },
                perm_importance={
                    "baseline_score": 0.95,
                    "group_contribution_pct": {"a": val, "b": 100 - val},
                },
                metrics={"f2": 0.98},
            )

        result = history.analyze_stability()

        assert result.correlation_matrix is not None
        assert "a" in result.correlation_matrix
        assert "b" in result.correlation_matrix["a"]

        # a and b should be perfectly negatively correlated
        assert abs(result.correlation_matrix["a"]["b"] + 1.0) < 0.01

    def test_stability_analysis_to_dict(self):
        """Test StabilityAnalysis to_dict method."""
        analysis = StabilityAnalysis(
            n_runs=5,
            shap_mean={"a": 50.0, "b": 30.0},
            shap_std={"a": 5.0, "b": 3.0},
            shap_cv={"a": 0.1, "b": 0.1},
            perm_mean={"a": 50.0, "b": 30.0},
            perm_std={"a": 5.0, "b": 3.0},
            perm_cv={"a": 0.1, "b": 0.1},
            correlation_matrix={"a": {"a": 1.0, "b": -0.5}, "b": {"a": -0.5, "b": 1.0}},
            most_stable_groups=[("a", 0.1), ("b", 0.1)],
            least_stable_groups=[("b", 0.1), ("a", 0.1)],
            performance_correlation={"a": 0.5, "b": -0.3},
        )

        data = analysis.to_dict()

        assert data["n_runs"] == 5
        assert "shap_mean" in data
        assert "perm_cv" in data
        assert "correlation_matrix" in data
