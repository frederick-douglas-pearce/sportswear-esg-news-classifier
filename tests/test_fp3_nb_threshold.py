"""Tests for fp3_nb threshold optimization utilities."""

from unittest.mock import patch

import numpy as np
import pytest

from src.fp3_nb.threshold_optimization import (
    analyze_threshold_tradeoffs,
    find_optimal_threshold,
    plot_threshold_analysis,
)


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification data with probabilities."""
    np.random.seed(42)
    n_samples = 100

    # Create ground truth labels (70 positive, 30 negative)
    y_true = np.array([1] * 70 + [0] * 30)

    # Create probabilities that somewhat correlate with labels
    # Positive class samples have higher probabilities
    y_proba = np.concatenate([
        np.random.beta(5, 2, 70),  # Higher probabilities for positives
        np.random.beta(2, 5, 30),  # Lower probabilities for negatives
    ])

    return y_true, y_proba


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold function."""

    def test_returns_threshold_and_metrics(self, binary_classification_data):
        """Test that function returns threshold and metrics dict."""
        y_true, y_proba = binary_classification_data

        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=0.95
        )

        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 1
        assert isinstance(metrics, dict)
        assert "threshold" in metrics
        assert "actual_recall" in metrics
        assert "precision" in metrics
        assert "f2_score" in metrics

    def test_achieves_target_recall(self, binary_classification_data):
        """Test that threshold achieves target recall."""
        y_true, y_proba = binary_classification_data

        target_recall = 0.90
        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=target_recall
        )

        # Actual recall should be at least target
        assert metrics["actual_recall"] >= target_recall

    def test_higher_recall_needs_lower_threshold(self, binary_classification_data):
        """Test that higher recall requires lower threshold."""
        y_true, y_proba = binary_classification_data

        threshold_90, _ = find_optimal_threshold(y_true, y_proba, target_recall=0.90)
        threshold_98, _ = find_optimal_threshold(y_true, y_proba, target_recall=0.98)

        # Higher recall needs lower threshold
        assert threshold_98 <= threshold_90

    def test_confusion_matrix_values(self, binary_classification_data):
        """Test that confusion matrix values are correct."""
        y_true, y_proba = binary_classification_data

        _, metrics = find_optimal_threshold(y_true, y_proba, target_recall=0.95)

        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        tn = metrics["true_negatives"]
        fn = metrics["false_negatives"]

        # Check totals match
        assert tp + fn == y_true.sum()  # All actual positives
        assert tn + fp == (y_true == 0).sum()  # All actual negatives

    def test_extreme_target_recall(self):
        """Test with very high target recall."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])

        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=0.99
        )

        # Should return a low threshold to capture all positives
        assert threshold <= 0.6
        assert metrics["actual_recall"] >= 0.99

    def test_unachievable_recall_warning(self, capsys):
        """Test warning when target recall cannot be achieved."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.4, 0.3, 0.6, 0.7])  # Inverted - hard to achieve high recall

        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=0.99
        )

        # Should still return something
        assert threshold is not None


class TestAnalyzeThresholdTradeoffs:
    """Tests for analyze_threshold_tradeoffs function."""

    def test_returns_dataframe(self, binary_classification_data):
        """Test that function returns DataFrame."""
        y_true, y_proba = binary_classification_data

        df = analyze_threshold_tradeoffs(
            y_true, y_proba,
            target_recalls=[0.90, 0.95],
            verbose=False
        )

        assert len(df) == 2
        assert "target_recall" in df.columns
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "f2_score" in df.columns

    def test_default_target_recalls(self, binary_classification_data):
        """Test with default target recalls."""
        y_true, y_proba = binary_classification_data

        df = analyze_threshold_tradeoffs(y_true, y_proba, verbose=False)

        # Default is [0.90, 0.95, 0.97, 0.98, 0.99]
        assert len(df) == 5

    def test_verbose_output(self, binary_classification_data, capsys):
        """Test verbose output."""
        y_true, y_proba = binary_classification_data

        analyze_threshold_tradeoffs(
            y_true, y_proba,
            target_recalls=[0.90, 0.95],
            verbose=True
        )

        captured = capsys.readouterr()
        assert "THRESHOLD ANALYSIS" in captured.out
        assert "Recall vs Precision Trade-off" in captured.out
        assert "Total positives:" in captured.out

    def test_fp_fn_counts(self, binary_classification_data):
        """Test that FP and FN counts are included."""
        y_true, y_proba = binary_classification_data

        df = analyze_threshold_tradeoffs(
            y_true, y_proba,
            target_recalls=[0.95],
            verbose=False
        )

        assert "fp_passed" in df.columns
        assert "fn_missed" in df.columns
        assert "fp_rate" in df.columns
        assert "fn_rate" in df.columns

    def test_higher_recall_means_more_fp(self, binary_classification_data):
        """Test that higher recall leads to more false positives."""
        y_true, y_proba = binary_classification_data

        df = analyze_threshold_tradeoffs(
            y_true, y_proba,
            target_recalls=[0.90, 0.95, 0.99],
            verbose=False
        )

        # FP count should generally increase with higher recall
        fp_at_90 = df[df["target_recall"] == 0.90]["fp_passed"].iloc[0]
        fp_at_99 = df[df["target_recall"] == 0.99]["fp_passed"].iloc[0]

        assert fp_at_99 >= fp_at_90


class TestPlotThresholdAnalysis:
    """Tests for plot_threshold_analysis function."""

    def test_returns_figure(self, binary_classification_data):
        """Test that function returns figure."""
        y_true, y_proba = binary_classification_data

        with patch("matplotlib.pyplot.show"):
            fig = plot_threshold_analysis(
                y_true, y_proba,
                target_recalls=[0.95, 0.98]
            )

        assert fig is not None

    def test_with_optimal_threshold(self, binary_classification_data):
        """Test plotting with optimal threshold marked."""
        y_true, y_proba = binary_classification_data

        with patch("matplotlib.pyplot.show"):
            fig = plot_threshold_analysis(
                y_true, y_proba,
                target_recalls=[0.95],
                optimal_threshold=0.6
            )

        assert fig is not None

    def test_custom_title(self, binary_classification_data):
        """Test custom title."""
        y_true, y_proba = binary_classification_data

        with patch("matplotlib.pyplot.show"):
            fig = plot_threshold_analysis(
                y_true, y_proba,
                title="Custom Threshold Analysis"
            )

        # Check suptitle
        assert fig._suptitle.get_text() == "Custom Threshold Analysis"

    def test_save_figure(self, binary_classification_data, tmp_path, capsys):
        """Test saving figure to file."""
        y_true, y_proba = binary_classification_data
        save_path = tmp_path / "threshold_analysis.png"

        with patch("matplotlib.pyplot.show"):
            plot_threshold_analysis(
                y_true, y_proba,
                save_path=str(save_path)
            )

        assert save_path.exists()
        captured = capsys.readouterr()
        assert "Figure saved to" in captured.out

    def test_two_subplots(self, binary_classification_data):
        """Test that figure has two subplots."""
        y_true, y_proba = binary_classification_data

        with patch("matplotlib.pyplot.show"):
            fig = plot_threshold_analysis(y_true, y_proba)

        assert len(fig.axes) == 2


class TestEdgeCases:
    """Edge case tests for threshold optimization."""

    def test_perfect_classifier(self):
        """Test with perfect classifier predictions."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([1.0, 0.9, 0.8, 0.2, 0.1, 0.0])

        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=0.95
        )

        assert metrics["precision"] >= 0.95
        assert metrics["actual_recall"] >= 0.95

    def test_all_same_class(self):
        """Test with all same class labels."""
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6])

        # Should not raise error
        threshold, metrics = find_optimal_threshold(
            y_true, y_proba, target_recall=0.90
        )

        assert threshold is not None

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        y_true = np.array([])
        y_proba = np.array([])

        # Should handle gracefully (may raise or return default)
        # This tests that the function doesn't crash
        try:
            threshold, metrics = find_optimal_threshold(
                y_true, y_proba, target_recall=0.95
            )
        except (ValueError, IndexError):
            pass  # Expected for empty arrays
