"""Tests for fp2_nb overfitting analysis utilities."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.fp2_nb.overfitting_analysis import (
    analyze_overfitting,
    plot_iteration_performance,
    plot_train_val_gap,
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
        scoring={"f2": "f1"},  # Using f1 as proxy for f2 in tests
        refit="f2",
        return_train_score=True,
        verbose=0
    )
    search.fit(X, y)
    return search


@pytest.fixture
def fitted_iteration_search(sample_data):
    """Create fitted GridSearchCV varying only n_estimators."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = sample_data
    param_grid = {"n_estimators": [10, 20, 30, 40, 50]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(
        RandomForestClassifier(max_depth=5, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring={"f2": "f1"},
        refit="f2",
        return_train_score=True,
        verbose=0
    )
    search.fit(X, y)
    return search


class TestPlotTrainValGap:
    """Tests for plot_train_val_gap function."""

    def test_returns_figure_and_dataframe(self, fitted_grid_search):
        """Test that function returns figure and dataframe."""
        with patch("matplotlib.pyplot.show"):
            fig, gap_df = plot_train_val_gap(
                fitted_grid_search,
                metric="f2",
            )

        assert fig is not None
        assert isinstance(gap_df, pd.DataFrame)
        assert "train_score" in gap_df.columns
        assert "val_score" in gap_df.columns
        assert "gap" in gap_df.columns
        assert "overfitting" in gap_df.columns

    def test_gap_calculation(self, fitted_grid_search):
        """Test that gap is calculated correctly."""
        with patch("matplotlib.pyplot.show"):
            _, gap_df = plot_train_val_gap(
                fitted_grid_search,
                metric="f2"
            )

        # Gap should be train - val
        expected_gaps = gap_df["train_score"] - gap_df["val_score"]
        np.testing.assert_array_almost_equal(gap_df["gap"], expected_gaps)

    def test_overfitting_threshold(self, fitted_grid_search):
        """Test overfitting detection threshold."""
        with patch("matplotlib.pyplot.show"):
            _, gap_df = plot_train_val_gap(
                fitted_grid_search,
                metric="f2",
                gap_threshold=0.5  # High threshold
            )

        # With high threshold, nothing should be flagged as overfitting
        assert not gap_df["overfitting"].any()

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

        with pytest.raises(ValueError, match="Train scores not found"):
            with patch("matplotlib.pyplot.show"):
                plot_train_val_gap(search, metric="score")

    def test_prints_summary(self, fitted_grid_search, capsys):
        """Test that summary is printed."""
        with patch("matplotlib.pyplot.show"):
            plot_train_val_gap(fitted_grid_search, metric="f2")

        captured = capsys.readouterr()
        assert "Overfitting Analysis Summary" in captured.out
        assert "Total combinations:" in captured.out


class TestPlotIterationPerformance:
    """Tests for plot_iteration_performance function."""

    def test_returns_figure(self, fitted_iteration_search):
        """Test that function returns figure."""
        with patch("matplotlib.pyplot.show"):
            fig = plot_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2"
            )

        assert fig is not None

    def test_missing_param_raises_error(self, fitted_iteration_search):
        """Test error when parameter not found."""
        with pytest.raises(ValueError, match="not found in cv_results"):
            with patch("matplotlib.pyplot.show"):
                plot_iteration_performance(
                    fitted_iteration_search,
                    param_name="nonexistent_param",
                    metric="f2"
                )

    def test_custom_title(self, fitted_iteration_search):
        """Test custom title."""
        with patch("matplotlib.pyplot.show"):
            fig = plot_iteration_performance(
                fitted_iteration_search,
                param_name="n_estimators",
                metric="f2",
                title="Custom Title"
            )

        # Check that the axes title matches
        assert fig.axes[0].get_title() == "Custom Title"


class TestAnalyzeOverfitting:
    """Tests for analyze_overfitting function."""

    def test_single_model(self, fitted_grid_search, capsys):
        """Test analyzing single model."""
        tuned_models = {"LR": fitted_grid_search}

        df = analyze_overfitting(
            tuned_models,
            metric="f2",
            gap_threshold=0.05,
            verbose=True
        )

        assert len(df) == 1
        assert df["model"].iloc[0] == "LR"
        assert "train_score" in df.columns
        assert "val_score" in df.columns
        assert "gap" in df.columns
        assert "overfitting" in df.columns

        captured = capsys.readouterr()
        assert "OVERFITTING ANALYSIS" in captured.out

    def test_multiple_models(self, sample_data, capsys):
        """Test analyzing multiple models."""
        X, y = sample_data

        # Create two fitted search objects
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search1 = GridSearchCV(
            LogisticRegression(max_iter=200),
            param_grid={"C": [0.1, 1.0]},
            cv=cv,
            scoring={"f2": "f1"},
            refit="f2",
            return_train_score=True,
            verbose=0
        )
        search1.fit(X, y)

        search2 = GridSearchCV(
            LogisticRegression(max_iter=200, penalty="l1", solver="saga"),
            param_grid={"C": [0.1, 1.0]},
            cv=cv,
            scoring={"f2": "f1"},
            refit="f2",
            return_train_score=True,
            verbose=0
        )
        search2.fit(X, y)

        tuned_models = {"LR_L2": search1, "LR_L1": search2}

        df = analyze_overfitting(tuned_models, metric="f2", verbose=True)

        assert len(df) == 2
        assert set(df["model"]) == {"LR_L2", "LR_L1"}

        captured = capsys.readouterr()
        assert "LR_L2" in captured.out
        assert "LR_L1" in captured.out

    def test_silent_mode(self, fitted_grid_search, capsys):
        """Test silent mode."""
        tuned_models = {"LR": fitted_grid_search}

        analyze_overfitting(tuned_models, metric="f2", verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_overfitting_flag(self, sample_data):
        """Test overfitting flag is set correctly."""
        X, y = sample_data

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        search = GridSearchCV(
            LogisticRegression(max_iter=200),
            param_grid={"C": [0.001]},  # May cause underfitting/overfitting
            cv=cv,
            scoring={"f2": "f1"},
            refit="f2",
            return_train_score=True,
            verbose=0
        )
        search.fit(X, y)

        # With very low threshold, should flag as overfitting
        df_low = analyze_overfitting(
            {"Model": search},
            metric="f2",
            gap_threshold=0.0001,
            verbose=False
        )

        # With very high threshold, should not flag as overfitting
        df_high = analyze_overfitting(
            {"Model": search},
            metric="f2",
            gap_threshold=1.0,
            verbose=False
        )

        # High threshold should never show overfitting
        assert not df_high["overfitting"].iloc[0]

    def test_n_combinations_reported(self, fitted_grid_search):
        """Test that number of combinations is reported."""
        tuned_models = {"LR": fitted_grid_search}

        df = analyze_overfitting(tuned_models, metric="f2", verbose=False)

        assert df["n_combinations"].iloc[0] == 3  # C has 3 values
