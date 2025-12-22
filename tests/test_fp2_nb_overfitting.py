"""Tests for fp2_nb overfitting analysis utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.fp2_nb.overfitting_analysis import (
    _generate_gap_recommendation,
    analyze_cv_train_val_gap,
    analyze_iteration_performance,
    get_gap_summary,
)


@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def fitted_grid_search(sample_data):
    """Create fitted GridSearchCV with train scores."""
    X, y = sample_data
    param_grid = {"C": [0.1, 1.0, 10.0]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(
        LogisticRegression(max_iter=200, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring={'f2': 'f1'},  # Using f1 as proxy for f2 in tests
        refit='f2',
        return_train_score=True,
        verbose=0
    )
    search.fit(X, y)
    return search


@pytest.fixture
def fitted_rf_search(sample_data):
    """Create fitted GridSearchCV for Random Forest."""
    X, y = sample_data
    param_grid = {"max_depth": [3, 5, 10]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(
        RandomForestClassifier(n_estimators=20, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring={'f2': 'f1'},
        refit='f2',
        return_train_score=True,
        verbose=0
    )
    search.fit(X, y)
    return search


@pytest.fixture
def fitted_iteration_search(sample_data):
    """Create fitted GridSearchCV varying only n_estimators."""
    X, y = sample_data
    param_grid = {"n_estimators": [10, 20, 30, 40, 50]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(
        RandomForestClassifier(max_depth=5, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring={'f2': 'f1'},
        refit='f2',
        return_train_score=True,
        verbose=0
    )
    search.fit(X, y)
    return search


class TestAnalyzeCvTrainValGap:
    """Tests for analyze_cv_train_val_gap function."""

    def test_returns_expected_keys(self, fitted_grid_search):
        """Test that function returns expected dictionary keys."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                model_name="Test LR",
                verbose=True
            )

        expected_keys = [
            'best_train_score', 'best_val_score', 'gap', 'gap_pct',
            'diagnosis', 'overfitting_detected', 'recommendation',
            'model_name', 'metric'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_gap_calculation(self, fitted_grid_search):
        """Test that gap is calculated correctly."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                verbose=False
            )

        expected_gap = result['best_train_score'] - result['best_val_score']
        assert abs(result['gap'] - expected_gap) < 1e-6

        expected_pct = expected_gap / result['best_train_score']
        assert abs(result['gap_pct'] - expected_pct) < 1e-6

    def test_good_fit_diagnosis(self, fitted_grid_search):
        """Test diagnosis for good fit (small gap)."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                gap_threshold_warning=0.5,  # Very high threshold
                gap_threshold_severe=0.6,
                verbose=False
            )

        assert result['diagnosis'] == 'Good fit'
        assert result['overfitting_detected'] == False

    def test_moderate_overfitting_diagnosis(self, fitted_grid_search):
        """Test diagnosis for moderate overfitting."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                gap_threshold_warning=0.001,  # Very low threshold
                gap_threshold_severe=0.5,
                verbose=False
            )

        # With very low warning threshold, should detect overfitting
        if result['gap_pct'] >= 0.001:
            assert result['diagnosis'] == 'MODERATE OVERFITTING'
            assert result['overfitting_detected'] == True

    def test_severe_overfitting_diagnosis(self, fitted_grid_search):
        """Test diagnosis for severe overfitting."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                gap_threshold_warning=0.0001,
                gap_threshold_severe=0.001,  # Very low severe threshold
                verbose=False
            )

        # With very low thresholds, should detect severe overfitting
        if result['gap_pct'] >= 0.001:
            assert result['diagnosis'] == 'SEVERE OVERFITTING'
            assert result['overfitting_detected'] == True

    def test_missing_train_scores_raises_error(self, sample_data):
        """Test error when train scores not available."""
        X, y = sample_data
        search = GridSearchCV(
            LogisticRegression(max_iter=200),
            param_grid={"C": [1.0]},
            cv=3,
            return_train_score=False  # No train scores
        )
        search.fit(X, y)

        with pytest.raises(ValueError, match="Training scores not found"):
            analyze_cv_train_val_gap(search, metric="score")

    def test_prints_analysis(self, fitted_grid_search, capsys):
        """Test that analysis is printed when verbose=True."""
        with patch("matplotlib.pyplot.show"):
            analyze_cv_train_val_gap(
                fitted_grid_search,
                metric="f2",
                model_name="Test Model",
                verbose=True
            )

        captured = capsys.readouterr()
        assert "Train-Validation Gap Analysis" in captured.out or "OVERFITTING WARNING" in captured.out
        assert "Training score:" in captured.out
        assert "Validation score:" in captured.out
        assert "Gap:" in captured.out
        assert "Diagnosis:" in captured.out


class TestGenerateGapRecommendation:
    """Tests for _generate_gap_recommendation function."""

    def test_good_fit_recommendation(self):
        """Test recommendation for good fit."""
        rec = _generate_gap_recommendation(
            "Test Model", 0.02, "Good fit", 0.05, 0.10
        )
        assert "healthy generalization" in rec

    def test_rf_severe_recommendation(self):
        """Test recommendation for RF with severe overfitting."""
        rec = _generate_gap_recommendation(
            "Random Forest", 0.15, "SEVERE OVERFITTING", 0.05, 0.10
        )
        assert "severe overfitting" in rec
        assert "max_depth" in rec
        assert "min_samples_leaf" in rec

    def test_rf_moderate_recommendation(self):
        """Test recommendation for RF with moderate overfitting."""
        rec = _generate_gap_recommendation(
            "RF_tuned", 0.07, "MODERATE OVERFITTING", 0.05, 0.10
        )
        assert "moderate overfitting" in rec
        assert "max_depth" in rec.lower()

    def test_lr_recommendation(self):
        """Test recommendation for Logistic Regression."""
        rec = _generate_gap_recommendation(
            "Logistic Regression", 0.12, "SEVERE OVERFITTING", 0.05, 0.10
        )
        assert "severe overfitting" in rec
        assert "C" in rec

    def test_hgb_recommendation(self):
        """Test recommendation for HistGradientBoosting."""
        rec = _generate_gap_recommendation(
            "HGB_tuned", 0.08, "MODERATE OVERFITTING", 0.05, 0.10
        )
        assert "moderate overfitting" in rec
        assert "l2_regularization" in rec or "early_stopping" in rec

    def test_generic_recommendation(self):
        """Test recommendation for unknown model type."""
        rec = _generate_gap_recommendation(
            "Unknown Model", 0.12, "SEVERE OVERFITTING", 0.05, 0.10
        )
        assert "reducing model complexity" in rec or "regularization" in rec


class TestAnalyzeIterationPerformance:
    """Tests for analyze_iteration_performance function."""

    def test_returns_expected_keys(self, fitted_iteration_search):
        """Test that function returns expected dictionary keys."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                verbose=True
            )

        expected_keys = [
            'optimal_value', 'optimal_train_score', 'optimal_val_score',
            'optimal_gap', 'optimal_gap_pct', 'tracking_df', 'tuned_value',
            'param_name'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_tracking_df_structure(self, fitted_iteration_search):
        """Test structure of tracking DataFrame."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                verbose=False
            )

        df = result['tracking_df']
        expected_cols = ['n_estimators', 'train_score_mean', 'train_score_std',
                         'val_score_mean', 'val_score_std', 'gap', 'gap_pct']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Should have one row per parameter value tested
        assert len(df) == 5  # [10, 20, 30, 40, 50]

    def test_optimal_value_is_best_validation(self, fitted_iteration_search):
        """Test that optimal_value corresponds to best validation score."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                verbose=False
            )

        df = result['tracking_df']
        best_idx = df['val_score_mean'].idxmax()
        expected_optimal = df.loc[best_idx, 'n_estimators']

        # Handle numpy types
        if hasattr(expected_optimal, 'item'):
            expected_optimal = expected_optimal.item()

        assert result['optimal_value'] == expected_optimal

    def test_missing_param_raises_error(self, fitted_iteration_search):
        """Test error when parameter not found."""
        with pytest.raises(ValueError, match="not found"):
            with patch("matplotlib.pyplot.show"):
                analyze_iteration_performance(
                    fitted_iteration_search,
                    param_name="nonexistent_param",
                    metric="f2"
                )

    def test_tuned_value_in_output(self, fitted_iteration_search):
        """Test that tuned_value is included in output."""
        with patch("matplotlib.pyplot.show"):
            result = analyze_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                tuned_value=30,
                verbose=False
            )

        assert result['tuned_value'] == 30

    def test_prints_analysis(self, fitted_iteration_search, capsys):
        """Test that analysis is printed when verbose=True."""
        with patch("matplotlib.pyplot.show"):
            analyze_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                model_name="Test RF",
                verbose=True
            )

        captured = capsys.readouterr()
        assert "Performance Analysis" in captured.out
        assert "Optimal n_estimators" in captured.out


class TestGetGapSummary:
    """Tests for get_gap_summary function."""

    def test_single_model(self, fitted_grid_search, capsys):
        """Test summary for single model."""
        tuned_models = {"LR": fitted_grid_search}

        df = get_gap_summary(tuned_models, metric="f2", verbose=True)

        assert len(df) == 1
        assert df['model'].iloc[0] == "LR"
        assert 'train_score' in df.columns
        assert 'val_score' in df.columns
        assert 'gap' in df.columns
        assert 'gap_pct' in df.columns
        assert 'diagnosis' in df.columns

        captured = capsys.readouterr()
        assert "TRAIN-VALIDATION GAP SUMMARY" in captured.out

    def test_multiple_models(self, fitted_grid_search, fitted_rf_search):
        """Test summary for multiple models."""
        tuned_models = {
            "LR": fitted_grid_search,
            "RF": fitted_rf_search
        }

        df = get_gap_summary(tuned_models, metric="f2", verbose=False)

        assert len(df) == 2
        assert set(df['model']) == {"LR", "RF"}

    def test_diagnosis_values(self, fitted_grid_search):
        """Test that diagnosis values are valid."""
        tuned_models = {"LR": fitted_grid_search}

        df = get_gap_summary(tuned_models, metric="f2", verbose=False)

        valid_diagnoses = {'Good', 'MODERATE', 'SEVERE'}
        assert df['diagnosis'].iloc[0] in valid_diagnoses

    def test_silent_mode(self, fitted_grid_search, capsys):
        """Test silent mode."""
        tuned_models = {"LR": fitted_grid_search}

        get_gap_summary(tuned_models, metric="f2", verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""
