"""Tests for fp1_nb modeling utilities."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.fp1_nb.modeling import (
    _calculate_total_combinations,
    compare_models,
    compare_val_test_performance,
    create_search_object,
    evaluate_model,
    extract_cv_metrics,
    f2_scorer,
    get_best_model,
    get_best_params_summary,
)


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def fitted_search(sample_data):
    """Create fitted GridSearchCV object."""
    X, y = sample_data
    param_grid = {"C": [0.1, 1.0]}
    search = create_search_object(
        search_type="grid",
        estimator=LogisticRegression(max_iter=200, random_state=42),
        param_grid=param_grid,
        cv=3,
        verbose=0
    )
    search.fit(X, y)
    return search


class TestF2Scorer:
    """Tests for f2_scorer."""

    def test_f2_scorer_exists(self):
        """Test that f2_scorer is defined."""
        assert f2_scorer is not None

    def test_f2_scorer_callable(self, sample_data):
        """Test that f2_scorer works with sklearn."""
        X, y = sample_data
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        # f2_scorer should be usable
        score = f2_scorer(model, X, y)
        assert 0 <= score <= 1


class TestCreateSearchObject:
    """Tests for create_search_object function."""

    def test_create_grid_search(self, sample_data):
        """Test creating GridSearchCV object."""
        X, y = sample_data
        param_grid = {"C": [0.1, 1.0, 10.0]}

        search = create_search_object(
            search_type="grid",
            estimator=LogisticRegression(max_iter=200),
            param_grid=param_grid,
            cv=3,
            verbose=0
        )

        assert isinstance(search, GridSearchCV)
        assert search.param_grid == param_grid
        assert search.cv == 3
        assert search.return_train_score is True

    def test_create_random_search(self, sample_data):
        """Test creating RandomizedSearchCV object."""
        from sklearn.model_selection import RandomizedSearchCV

        param_grid = {"C": [0.1, 1.0, 10.0]}

        search = create_search_object(
            search_type="random",
            estimator=LogisticRegression(max_iter=200),
            param_grid=param_grid,
            n_iter=5,
            cv=3,
            random_state=42,
            verbose=0
        )

        assert isinstance(search, RandomizedSearchCV)
        assert search.n_iter == 5
        # RandomizedSearchCV stores param_distributions, not param_grid
        assert search.param_distributions == param_grid

    def test_invalid_search_type(self):
        """Test that invalid search type raises error."""
        with pytest.raises(ValueError, match="must be 'grid' or 'random'"):
            create_search_object(
                search_type="invalid",
                estimator=LogisticRegression(),
                param_grid={"C": [1.0]}
            )

    def test_default_scoring(self):
        """Test default scoring metrics are set."""
        search = create_search_object(
            search_type="grid",
            estimator=LogisticRegression(max_iter=200),
            param_grid={"C": [1.0]},
            verbose=0
        )

        assert isinstance(search.scoring, dict)
        assert "f2" in search.scoring
        assert "recall" in search.scoring
        assert "precision" in search.scoring


class TestCalculateTotalCombinations:
    """Tests for _calculate_total_combinations function."""

    def test_single_param(self):
        """Test with single parameter."""
        param_grid = {"C": [0.1, 1.0, 10.0]}
        assert _calculate_total_combinations(param_grid) == 3

    def test_multiple_params(self):
        """Test with multiple parameters."""
        param_grid = {
            "C": [0.1, 1.0],
            "penalty": ["l1", "l2"]
        }
        assert _calculate_total_combinations(param_grid) == 4

    def test_three_params(self):
        """Test with three parameters."""
        param_grid = {
            "C": [0.1, 1.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
        }
        assert _calculate_total_combinations(param_grid) == 8


class TestExtractCvMetrics:
    """Tests for extract_cv_metrics function."""

    def test_extract_metrics(self, fitted_search):
        """Test extracting CV metrics from fitted search."""
        metrics = extract_cv_metrics(fitted_search)

        # Check that expected metrics are present
        assert "cv_f2" in metrics
        assert "cv_recall" in metrics
        assert "cv_precision" in metrics

        # Check that std metrics are also present
        assert "cv_f2_std" in metrics

        # Check that values are reasonable
        for key, value in metrics.items():
            assert isinstance(value, (int, float, np.floating))


class TestGetBestParamsSummary:
    """Tests for get_best_params_summary function."""

    def test_get_summary(self, fitted_search, capsys):
        """Test getting best params summary."""
        summary = get_best_params_summary(
            fitted_search,
            model_name="TestLR",
            verbose=True
        )

        assert "model_name" in summary
        assert summary["model_name"] == "TestLR"
        assert "best_params" in summary
        assert "best_score" in summary
        assert "cv_metrics" in summary

        captured = capsys.readouterr()
        assert "BEST PARAMETERS: TestLR" in captured.out

    def test_silent_mode(self, fitted_search, capsys):
        """Test silent mode."""
        get_best_params_summary(fitted_search, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestCompareModels:
    """Tests for compare_models function."""

    def test_compare_multiple_models(self, capsys):
        """Test comparing multiple models."""
        metrics = [
            {"model_name": "Model1", "accuracy": 0.9, "f1": 0.85},
            {"model_name": "Model2", "accuracy": 0.85, "f1": 0.9},
        ]

        with patch("matplotlib.pyplot.show"):
            df = compare_models(
                metrics,
                metrics_to_display=["accuracy", "f1"],
                verbose=True
            )

        assert len(df) == 2
        assert "accuracy" in df.columns
        assert "f1" in df.columns

    def test_silent_mode(self, capsys):
        """Test silent mode."""
        metrics = [{"model_name": "Model1", "accuracy": 0.9}]

        df = compare_models(metrics, verbose=False)

        captured = capsys.readouterr()
        assert "Model Comparison" not in captured.out


class TestGetBestModel:
    """Tests for get_best_model function."""

    def test_get_best_by_metric(self, capsys):
        """Test getting best model by primary metric."""
        comparison_df = pd.DataFrame({
            "accuracy": [0.9, 0.85, 0.95],
            "f1": [0.85, 0.9, 0.8],
        }, index=["Model1", "Model2", "Model3"])

        best_name, best_metrics = get_best_model(
            comparison_df,
            primary_metric="accuracy"
        )

        assert best_name == "Model3"
        assert best_metrics["accuracy"] == 0.95

        captured = capsys.readouterr()
        assert "Best model by accuracy: Model3" in captured.out

    def test_invalid_metric(self):
        """Test with invalid metric."""
        comparison_df = pd.DataFrame({
            "accuracy": [0.9, 0.85],
        }, index=["Model1", "Model2"])

        with pytest.raises(ValueError, match="not found in comparison"):
            get_best_model(comparison_df, primary_metric="invalid_metric")


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_basic(self, sample_data, capsys):
        """Test basic model evaluation."""
        X, y = sample_data
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(
            model, X, y,
            model_name="TestLR",
            dataset_name="Test",
            verbose=True,
            plot=False
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "true_positives" in metrics
        assert "false_positives" in metrics

        captured = capsys.readouterr()
        assert "MODEL EVALUATION: TestLR on Test" in captured.out

    def test_silent_mode(self, sample_data, capsys):
        """Test silent mode."""
        X, y = sample_data
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        evaluate_model(model, X, y, verbose=False, plot=False)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestCompareValTestPerformance:
    """Tests for compare_val_test_performance function."""

    def test_compare_performance(self, capsys):
        """Test comparing validation vs test performance."""
        val_metrics = {
            "accuracy": 0.9,
            "f1": 0.85,
            "recall": 0.8,
        }
        test_metrics = {
            "accuracy": 0.88,
            "f1": 0.83,
            "recall": 0.79,
        }

        df = compare_val_test_performance(
            val_metrics,
            test_metrics,
            metrics_to_compare=["accuracy", "f1", "recall"],
            verbose=True
        )

        assert len(df) == 3
        assert "validation" in df.columns
        assert "test" in df.columns
        assert "difference" in df.columns

        captured = capsys.readouterr()
        assert "VALIDATION vs TEST PERFORMANCE" in captured.out

    def test_overfitting_warning(self, capsys):
        """Test overfitting warning."""
        val_metrics = {"accuracy": 0.95}
        test_metrics = {"accuracy": 0.80}  # Large drop

        compare_val_test_performance(
            val_metrics,
            test_metrics,
            metrics_to_compare=["accuracy"],
            verbose=True
        )

        captured = capsys.readouterr()
        assert "[WARNING]" in captured.out

    def test_good_generalization(self, capsys):
        """Test good generalization message."""
        val_metrics = {"accuracy": 0.90}
        test_metrics = {"accuracy": 0.89}  # Small drop

        compare_val_test_performance(
            val_metrics,
            test_metrics,
            metrics_to_compare=["accuracy"],
            verbose=True
        )

        captured = capsys.readouterr()
        assert "[OK] Model generalizes well" in captured.out
