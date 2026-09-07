"""Tests for the MLOps monitoring module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.mlops.monitoring import DriftMonitor, DriftReport, run_drift_analysis


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_mlops_settings_disabled():
    """Create mock MLOps settings with Evidently disabled."""
    with patch('src.mlops.monitoring.mlops_settings') as mock_settings:
        mock_settings.evidently_enabled = False
        mock_settings.drift_threshold = 0.1
        mock_settings.get_reports_dir = MagicMock(return_value=Path("/tmp/reports"))
        yield mock_settings


@pytest.fixture
def mock_mlops_settings_enabled():
    """Create mock MLOps settings with Evidently enabled."""
    with patch('src.mlops.monitoring.mlops_settings') as mock_settings:
        mock_settings.evidently_enabled = True
        mock_settings.drift_threshold = 0.1
        mock_settings.get_reports_dir = MagicMock(return_value=Path("/tmp/reports"))
        yield mock_settings


@pytest.fixture
def reference_data():
    """Create reference dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "probability": np.random.uniform(0.3, 0.7, 100),
        "prediction": np.random.choice([0, 1], 100, p=[0.3, 0.7]),
        "text_length": np.random.randint(100, 500, 100),
    })


@pytest.fixture
def current_data_no_drift(reference_data):
    """Create current data with no drift."""
    np.random.seed(43)
    return pd.DataFrame({
        "probability": np.random.uniform(0.3, 0.7, 50),
        "prediction": np.random.choice([0, 1], 50, p=[0.3, 0.7]),
        "text_length": np.random.randint(100, 500, 50),
    })


@pytest.fixture
def current_data_with_drift():
    """Create current data with significant drift."""
    np.random.seed(44)
    return pd.DataFrame({
        "probability": np.random.uniform(0.7, 0.95, 50),  # Shifted distribution
        "prediction": np.random.choice([0, 1], 50, p=[0.1, 0.9]),  # Higher positive rate
        "text_length": np.random.randint(100, 500, 50),
    })


@pytest.fixture
def disabled_monitor(mock_mlops_settings_disabled):
    """Create a drift monitor with Evidently disabled."""
    return DriftMonitor("fp")


# ============================================================================
# DriftReport Dataclass Tests
# ============================================================================

class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_drift_report_creation(self):
        """Test basic DriftReport creation."""
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=True,
            drift_score=0.15,
            threshold=0.1,
            details={"test": "value"},
        )

        assert report.classifier_type == "fp"
        assert report.drift_detected is True
        assert report.drift_score == 0.15
        assert report.threshold == 0.1
        assert report.details == {"test": "value"}
        assert report.report_path is None

    def test_drift_report_with_path(self):
        """Test DriftReport with report path."""
        report = DriftReport(
            classifier_type="ep",
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.05,
            threshold=0.1,
            details={},
            report_path=Path("/tmp/report.html"),
        )

        assert report.report_path == Path("/tmp/report.html")

    def test_drift_report_no_drift(self):
        """Test DriftReport when no drift detected."""
        report = DriftReport(
            classifier_type="esg",
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.02,
            threshold=0.1,
            details={"reference_size": 100, "current_size": 50},
        )

        assert report.drift_detected is False
        assert report.drift_score < report.threshold


# ============================================================================
# DriftMonitor Initialization Tests
# ============================================================================

class TestDriftMonitorInit:
    """Tests for DriftMonitor initialization."""

    def test_init_with_disabled_evidently(self, mock_mlops_settings_disabled):
        """Test initialization when Evidently is disabled."""
        monitor = DriftMonitor("fp")

        assert monitor.classifier_type == "fp"
        assert monitor.enabled is False
        assert monitor.threshold == 0.1
        assert monitor._evidently is None

    def test_init_with_enabled_evidently_not_installed(self, mock_mlops_settings_enabled):
        """Test initialization when Evidently is enabled but not installed."""
        with patch.dict('sys.modules', {'evidently': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                monitor = DriftMonitor("ep")
                # Should gracefully degrade
                assert monitor.enabled is False

    def test_init_creates_correct_classifier_type(self, mock_mlops_settings_disabled):
        """Test that classifier type is set correctly."""
        for clf_type in ["fp", "ep", "esg"]:
            monitor = DriftMonitor(clf_type)
            assert monitor.classifier_type == clf_type


# ============================================================================
# Legacy Drift Check Tests
# ============================================================================

class TestLegacyDriftCheck:
    """Tests for legacy KS-based drift detection."""

    def test_legacy_drift_check_no_drift(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """Test legacy drift check with similar distributions."""
        report = disabled_monitor._legacy_drift_check(
            current_data_no_drift, reference_data
        )

        assert report.classifier_type == "fp"
        assert report.drift_score < 0.5  # Should be relatively low
        assert "probability_ks_statistic" in report.details
        assert "probability_p_value" in report.details
        assert "reference_size" in report.details
        assert "current_size" in report.details

    def test_legacy_drift_check_with_drift(
        self, disabled_monitor, reference_data, current_data_with_drift
    ):
        """Test legacy drift check with different distributions."""
        report = disabled_monitor._legacy_drift_check(
            current_data_with_drift, reference_data
        )

        # Drift score should be higher due to shifted distribution
        assert report.drift_score > 0.3

    def test_legacy_drift_check_includes_prediction_rate(
        self, disabled_monitor, reference_data, current_data_with_drift
    ):
        """Test that legacy check includes prediction rate comparison."""
        report = disabled_monitor._legacy_drift_check(
            current_data_with_drift, reference_data
        )

        assert "reference_prediction_rate" in report.details
        assert "current_prediction_rate" in report.details
        assert "prediction_rate_diff" in report.details

    def test_legacy_drift_check_missing_columns(self, disabled_monitor):
        """Test legacy drift check when columns are missing."""
        # Data without expected columns
        current = pd.DataFrame({"other_col": [1, 2, 3]})
        reference = pd.DataFrame({"other_col": [4, 5, 6]})

        report = disabled_monitor._legacy_drift_check(current, reference)

        # Should still produce a report with basic info
        assert report.drift_score == 0.0
        assert report.details["reference_size"] == 3
        assert report.details["current_size"] == 3


# ============================================================================
# Check Drift Method Tests
# ============================================================================

class TestCheckDrift:
    """Tests for the main check_drift method."""

    def test_check_drift_with_empty_data(self, disabled_monitor):
        """Test check_drift with empty dataframes."""
        empty_df = pd.DataFrame()

        report = disabled_monitor.check_drift(
            current_data=empty_df,
            reference_data=empty_df,
        )

        assert report.drift_detected is False
        assert report.drift_score == 0.0
        assert "error" in report.details
        assert "Insufficient data" in report.details["error"]

    def test_check_drift_with_empty_current(self, disabled_monitor, reference_data):
        """Test check_drift with empty current data."""
        empty_df = pd.DataFrame()

        report = disabled_monitor.check_drift(
            current_data=empty_df,
            reference_data=reference_data,
        )

        assert report.drift_detected is False
        assert "error" in report.details

    def test_check_drift_with_empty_reference(self, disabled_monitor, current_data_no_drift):
        """Test check_drift with empty reference data."""
        empty_df = pd.DataFrame()

        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=empty_df,
        )

        assert report.drift_detected is False
        assert "error" in report.details

    def test_check_drift_uses_legacy_when_disabled(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """Test that check_drift uses legacy method when Evidently disabled."""
        with patch.object(
            disabled_monitor, '_legacy_drift_check', wraps=disabled_monitor._legacy_drift_check
        ) as mock_legacy:
            report = disabled_monitor.check_drift(
                current_data=current_data_no_drift,
                reference_data=reference_data,
            )

            mock_legacy.assert_called_once()
            assert "probability_ks_statistic" in report.details

    def test_check_drift_loads_data_when_not_provided(self, mock_mlops_settings_disabled):
        """Test that check_drift loads data when not provided."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref:

            mock_load_logs.return_value = pd.DataFrame({
                "probability": [0.5, 0.6, 0.7, 0.8],
                "prediction": [0, 1, 1, 1],
            })
            mock_load_ref.return_value = pd.DataFrame({
                "probability": [0.4, 0.5, 0.6, 0.7],
                "prediction": [0, 0, 1, 1],
            })

            monitor = DriftMonitor("fp")
            report = monitor.check_drift(days=7)

            mock_load_logs.assert_called_once_with("fp", days=7, from_database=False)
            mock_load_ref.assert_called_once_with("fp")

    def test_check_drift_loads_from_database_when_specified(self, mock_mlops_settings_disabled):
        """Test that check_drift loads from database when from_database=True."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref:

            mock_load_logs.return_value = pd.DataFrame({
                "probability": [0.5, 0.6, 0.7, 0.8],
                "prediction": [0, 1, 1, 1],
            })
            mock_load_ref.return_value = pd.DataFrame({
                "probability": [0.4, 0.5, 0.6, 0.7],
                "prediction": [0, 0, 1, 1],
            })

            monitor = DriftMonitor("fp")
            report = monitor.check_drift(days=7, from_database=True)

            mock_load_logs.assert_called_once_with("fp", days=7, from_database=True)
            mock_load_ref.assert_called_once_with("fp")

    def test_check_drift_splits_data_when_no_reference(self, mock_mlops_settings_disabled):
        """Test that check_drift splits current data when no reference exists."""
        current_data = pd.DataFrame({
            "probability": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "prediction": [0, 0, 1, 1, 1, 1],
        })

        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref:

            mock_load_logs.return_value = current_data
            mock_load_ref.side_effect = FileNotFoundError("No reference data")

            monitor = DriftMonitor("fp")
            report = monitor.check_drift()

            # Should still produce a valid report using split data
            assert report.classifier_type == "fp"


# ============================================================================
# Evidently Drift Check Tests (Mocked)
# ============================================================================

class TestEvidentlyDriftCheck:
    """Tests for Evidently-based drift detection (mocked)."""

    def test_evidently_drift_check_no_numeric_columns(self, mock_mlops_settings_enabled):
        """Test Evidently drift check when no numeric columns available."""
        # Data without probability, prediction, novelty_score, or brand_ columns
        current = pd.DataFrame({"other": ["a", "b", "c"]})
        reference = pd.DataFrame({"other": ["x", "y", "z"]})

        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        monitor._evidently = {
            "Report": MagicMock(),
            "ValueDrift": MagicMock(),
        }

        report = monitor._evidently_drift_check(current, reference, save_report=False)

        assert report.drift_detected is False
        assert "error" in report.details
        assert "No columns available for drift detection" in report.details["error"]

    def test_evidently_drift_check_with_valid_data(self, mock_mlops_settings_enabled):
        """Test Evidently drift check with valid data (mocked)."""
        current = pd.DataFrame({
            "probability": [0.5, 0.6, 0.7],
            "prediction": [0, 1, 1],
        })
        reference = pd.DataFrame({
            "probability": [0.4, 0.5, 0.6],
            "prediction": [0, 0, 1],
        })

        # Mock Evidently report snapshot
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": "probability", "threshold": 0.05},
                    "value": 0.5,  # p-value > threshold means no drift
                },
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": "prediction", "threshold": 0.05},
                    "value": 0.3,  # p-value > threshold means no drift
                },
            ]
        }

        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot

        mock_Report = MagicMock(return_value=mock_report)

        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        monitor._evidently = {
            "Report": mock_Report,
            "ValueDrift": MagicMock(),
        }

        report = monitor._evidently_drift_check(current, reference, save_report=False)

        assert report.classifier_type == "fp"
        assert "reference_size" in report.details
        assert "current_size" in report.details
        mock_report.run.assert_called_once()


# ============================================================================
# run_drift_analysis Function Tests
# ============================================================================

class TestRunDriftAnalysis:
    """Tests for run_drift_analysis convenience function."""

    def test_run_drift_analysis_creates_monitor(self, mock_mlops_settings_disabled):
        """Test that run_drift_analysis creates a DriftMonitor."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref:

            mock_load_logs.return_value = pd.DataFrame({
                "probability": [0.5, 0.6],
                "prediction": [0, 1],
            })
            mock_load_ref.return_value = pd.DataFrame({
                "probability": [0.4, 0.5],
                "prediction": [0, 1],
            })

            report = run_drift_analysis("fp", days=7, save_report=False, send_alert=False)

            assert report.classifier_type == "fp"

    def test_run_drift_analysis_sends_alert_on_drift(self, mock_mlops_settings_disabled):
        """Test that run_drift_analysis sends alert when drift detected."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref, \
             patch('src.mlops.monitoring.DriftMonitor.check_drift') as mock_check:

            # Return a report with drift detected
            mock_check.return_value = DriftReport(
                classifier_type="fp",
                timestamp=datetime.now(),
                drift_detected=True,
                drift_score=0.25,
                threshold=0.1,
                details={"test": "data"},
            )

            with patch('src.mlops.alerts.send_drift_alert') as mock_alert:
                report = run_drift_analysis("fp", days=7, save_report=False, send_alert=True)

                mock_alert.assert_called_once_with(
                    classifier_type="fp",
                    drift_score=0.25,
                    threshold=0.1,
                    details={"test": "data"},
                )

    def test_run_drift_analysis_no_alert_when_no_drift(self, mock_mlops_settings_disabled):
        """Test that run_drift_analysis doesn't send alert when no drift."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref, \
             patch('src.mlops.monitoring.DriftMonitor.check_drift') as mock_check:

            # Return a report without drift
            mock_check.return_value = DriftReport(
                classifier_type="fp",
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.05,
                threshold=0.1,
                details={},
            )

            with patch('src.mlops.alerts.send_drift_alert') as mock_alert:
                report = run_drift_analysis("fp", days=7, save_report=False, send_alert=True)

                mock_alert.assert_not_called()

    def test_run_drift_analysis_no_alert_when_disabled(self, mock_mlops_settings_disabled):
        """Test that run_drift_analysis doesn't send alert when send_alert=False."""
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref, \
             patch('src.mlops.monitoring.DriftMonitor.check_drift') as mock_check:

            # Return a report with drift detected
            mock_check.return_value = DriftReport(
                classifier_type="fp",
                timestamp=datetime.now(),
                drift_detected=True,
                drift_score=0.25,
                threshold=0.1,
                details={},
            )

            with patch('src.mlops.alerts.send_drift_alert') as mock_alert:
                report = run_drift_analysis("fp", days=7, save_report=False, send_alert=False)

                # Alert should not be called even with drift
                mock_alert.assert_not_called()

    def test_run_drift_analysis_passes_from_database(self, mock_mlops_settings_disabled):
        """Test that run_drift_analysis passes from_database to check_drift."""
        with patch('src.mlops.monitoring.DriftMonitor.check_drift') as mock_check:
            mock_check.return_value = DriftReport(
                classifier_type="fp",
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.05,
                threshold=0.1,
                details={},
            )

            report = run_drift_analysis(
                "fp", days=7, save_report=False, send_alert=False, from_database=True
            )

            mock_check.assert_called_once_with(
                days=7, save_report=False, from_database=True
            )


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestMonitoringEdgeCases:
    """Edge case tests for monitoring module."""

    def test_drift_threshold_boundary(self, disabled_monitor, reference_data):
        """Test drift detection at threshold boundary."""
        # Create data that produces drift score near threshold
        current_data = pd.DataFrame({
            "probability": np.linspace(0.35, 0.75, 50),  # Slightly shifted
            "prediction": [1] * 35 + [0] * 15,  # Similar rate
        })

        report = disabled_monitor._legacy_drift_check(current_data, reference_data)

        # Just verify report is valid - exact threshold behavior depends on data
        # Note: numpy booleans are np.bool_ type, not Python bool
        assert report.drift_detected in (True, False)
        assert 0 <= float(report.drift_score) <= 1

    def test_monitor_handles_nan_values(self, disabled_monitor):
        """Test that monitor handles NaN values gracefully."""
        current = pd.DataFrame({
            "probability": [0.5, np.nan, 0.7, 0.8],
            "prediction": [0, 1, np.nan, 1],
        })
        reference = pd.DataFrame({
            "probability": [0.4, 0.5, 0.6, 0.7],
            "prediction": [0, 0, 1, 1],
        })

        # Should not raise an error
        report = disabled_monitor._legacy_drift_check(current, reference)
        assert report is not None

    def test_monitor_with_single_sample(self, disabled_monitor):
        """Test monitor with minimal data (single sample)."""
        current = pd.DataFrame({
            "probability": [0.5],
            "prediction": [1],
        })
        reference = pd.DataFrame({
            "probability": [0.5],
            "prediction": [1],
        })

        report = disabled_monitor._legacy_drift_check(current, reference)
        assert report is not None

    def test_drift_report_timestamp_is_recent(self, disabled_monitor, reference_data, current_data_no_drift):
        """Test that drift report has a recent timestamp."""
        before = datetime.now()
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference_data,
        )
        after = datetime.now()

        assert before <= report.timestamp <= after


# ============================================================================
# Indeterminate verdict + the asymmetric novelty_score guard (issue #71)
# ============================================================================

class TestIndeterminateVerdict:
    """A report with nothing measured must say so, not report no-drift.

    Before #71 both branches below returned drift_detected=False with no way to
    tell them from a real clean result, so `monitor_drift.py` exited 0 and the
    workflow reported "all classifiers healthy".
    """

    def test_default_report_is_not_indeterminate(self):
        """Control: the flag defaults off, so a real result is unaffected."""
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.05,
            threshold=0.1,
            details={},
        )

        assert report.indeterminate is False

    def test_empty_data_is_indeterminate(self, disabled_monitor):
        """The EP path: no predictions ever recorded, so nothing to compare."""
        empty = pd.DataFrame()

        report = disabled_monitor.check_drift(
            current_data=empty, reference_data=empty, save_report=False
        )

        assert report.indeterminate is True
        assert report.drift_detected is False  # because nothing was measured
        assert report.details["reference_size"] == 0
        assert report.details["current_size"] == 0

    def test_empty_current_only_is_indeterminate(
        self, disabled_monitor, reference_data
    ):
        report = disabled_monitor.check_drift(
            current_data=pd.DataFrame(),
            reference_data=reference_data,
            save_report=False,
        )

        assert report.indeterminate is True

    def test_real_comparison_is_not_indeterminate(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """Control: a genuine comparison stays determinate."""
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference_data,
            save_report=False,
        )

        assert report.indeterminate is False

    def test_no_common_columns_is_indeterminate(self, mock_mlops_settings_enabled):
        """A reference written against an older schema measures nothing."""
        current = pd.DataFrame({"other": ["a", "b", "c"]})
        reference = pd.DataFrame({"different": ["x", "y", "z"]})

        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        monitor._evidently = {"Report": MagicMock(), "ValueDrift": MagicMock()}

        report = monitor._evidently_drift_check(current, reference, save_report=False)

        assert report.indeterminate is True
        assert "No columns available" in report.details["error"]


class TestNoveltyScoreGuardSymmetry:
    """The live raise site of issue #71.

    `novelty_score` was added to the database after the reference parquet was
    written, so current had the column and reference did not. The stats block
    guarded only `current_data` and then read `reference_data["novelty_score"]`,
    raising KeyError on every run from 2026-01-17 to 2026-09-06.
    """

    def _monitor(self):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": "probability", "threshold": 0.05},
                    "value": 0.5,
                }
            ]
        }
        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=mock_report),
            "ValueDrift": MagicMock(),
        }
        return monitor

    def test_reference_without_novelty_does_not_raise(
        self, mock_mlops_settings_enabled
    ):
        """The exact production shape: current has the column, reference does not."""
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )
        reference = pd.DataFrame(
            {"probability": [0.4, 0.5, 0.6], "prediction": [0, 0, 1]}
        )

        report = self._monitor()._evidently_drift_check(
            current, reference, save_report=False
        )

        # Current-side stats are still reported; reference-side are simply absent.
        assert "current_novelty_mean" in report.details
        assert "reference_novelty_mean" not in report.details
        assert report.indeterminate is False

    def test_current_without_novelty_does_not_raise(self, mock_mlops_settings_enabled):
        """The inverse, which the original guard happened to survive."""
        current = pd.DataFrame(
            {"probability": [0.5, 0.6, 0.7], "prediction": [0, 1, 1]}
        )
        reference = pd.DataFrame(
            {
                "probability": [0.4, 0.5, 0.6],
                "prediction": [0, 0, 1],
                "novelty_score": [0.1, 0.3, 0.5],
            }
        )

        report = self._monitor()._evidently_drift_check(
            current, reference, save_report=False
        )

        assert "reference_novelty_mean" in report.details
        assert "current_novelty_mean" not in report.details

    def test_both_sides_present_still_reports_both(self, mock_mlops_settings_enabled):
        """Control: the guard split did not drop the stats it should still emit."""
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )
        reference = pd.DataFrame(
            {
                "probability": [0.4, 0.5, 0.6],
                "prediction": [0, 0, 1],
                "novelty_score": [0.1, 0.3, 0.5],
            }
        )

        report = self._monitor()._evidently_drift_check(
            current, reference, save_report=False
        )

        assert "current_novelty_mean" in report.details
        assert "reference_novelty_mean" in report.details


class TestPartialComparisonIsVisible:
    """A reference missing columns must not yield a verdict that looks complete.

    Fixing the novelty_score KeyError removed a crash that had been
    accidentally surfacing this. Without the record below, a stale reference
    now compares whatever it shares and reports a whole-looking result -- the
    epic's defect class re-entering through the fix for it.
    """

    def _monitor_with_metrics(self, columns):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": c, "threshold": 0.05},
                    "value": 0.5,
                }
                for c in columns
            ]
        }
        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=mock_report),
            "ValueDrift": MagicMock(),
        }
        return monitor

    def test_missing_reference_column_is_recorded(self, mock_mlops_settings_enabled):
        """The production shape: reference predates novelty_score."""
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )
        reference = pd.DataFrame(
            {"probability": [0.4, 0.5, 0.6], "prediction": [0, 0, 1]}
        )

        report = self._monitor_with_metrics(["probability", "prediction"])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert report.details["columns_missing_from_reference"] == ["novelty_score"]
        assert "novelty_score" not in report.details["columns_checked"]

    def test_missing_reference_column_is_logged(self, mock_mlops_settings_enabled, caplog):
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )
        reference = pd.DataFrame(
            {"probability": [0.4, 0.5, 0.6], "prediction": [0, 0, 1]}
        )

        with caplog.at_level("WARNING"):
            self._monitor_with_metrics(["probability", "prediction"])._evidently_drift_check(
                current, reference, save_report=False
            )

        assert "novelty_score" in caplog.text
        assert "NOT assessed" in caplog.text

    def test_complete_reference_records_nothing_missing(self, mock_mlops_settings_enabled):
        """Control: a matching schema reports an empty missing-list, not a false alarm."""
        frame = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )

        report = self._monitor_with_metrics(
            ["probability", "prediction", "novelty_score"]
        )._evidently_drift_check(frame, frame, save_report=False)

        assert report.details["columns_missing_from_reference"] == []


class TestLegacyPathIsAlsoGuarded:
    """The legacy path is the DEFAULT one, and it had none of the guards.

    `EVIDENTLY_ENABLED` defaults to false, and `_setup_evidently` also falls
    back here on ImportError. Before this, `max(drift_scores) if drift_scores
    else 0.0` turned "nothing was comparable" into drift score 0.0 -> no drift
    -> exit 0 -> healthy: issue #71's exact shape, on the path taken by default.
    """

    def test_no_shared_columns_is_indeterminate(self, disabled_monitor):
        current = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        reference = pd.DataFrame({"unrelated": [1, 2], "other": [3, 4]})

        report = disabled_monitor.check_drift(
            current_data=current, reference_data=reference, save_report=False
        )

        assert report.indeterminate is True
        assert report.drift_detected is False
        assert "No columns available" in report.details["error"]

    def test_reference_with_only_brand_columns_is_indeterminate(self, disabled_monitor):
        """A brand-only reference shares no CORE column, so nothing is measurable."""
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [1, 0]}
        )
        reference = pd.DataFrame({"brand_nike": [1, 1], "brand_puma": [0, 1]})

        report = disabled_monitor.check_drift(
            current_data=current, reference_data=reference, save_report=False
        )

        assert report.indeterminate is True

    def test_real_comparison_is_not_indeterminate(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """Control: the guard has not made every legacy run indeterminate."""
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference_data,
            save_report=False,
        )

        assert report.indeterminate is False
        assert report.details["columns_missing_from_reference"] == []

    def test_partial_reference_records_the_gap(self, disabled_monitor):
        """Shares `probability` but not `novelty_score` — measurable, but partial."""
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6, 0.7],
                "prediction": [0, 1, 1],
                "novelty_score": [0.2, 0.4, 0.9],
            }
        )
        reference = pd.DataFrame(
            {"probability": [0.4, 0.5, 0.6], "prediction": [0, 0, 1]}
        )

        report = disabled_monitor.check_drift(
            current_data=current, reference_data=reference, save_report=False
        )

        assert report.indeterminate is False
        assert report.details["columns_missing_from_reference"] == ["novelty_score"]


class TestAllThreeAsymmetricReadsAreGuarded:
    """`novelty_score` was the one that fired; three sites had the shape.

    A reference sharing only `brand_*` columns would have died on
    `reference_prob_mean` first. An earlier revision of the #71 fix corrected
    novelty_score alone and asserted the others were already symmetric.
    """

    def _monitor(self, columns):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": c, "threshold": 0.05},
                    "value": 0.5,
                }
                for c in columns
            ]
        }
        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=mock_report),
            "ValueDrift": MagicMock(),
        }
        return monitor

    def test_brand_only_reference_does_not_raise(self, mock_mlops_settings_enabled):
        """The case that would have died on `reference_prob_mean`."""
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [1, 0]}
        )
        reference = pd.DataFrame({"brand_nike": [1, 1]})

        report = self._monitor(["brand_nike"])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert "current_prob_mean" in report.details
        assert "reference_prob_mean" not in report.details
        assert "current_prediction_rate" in report.details
        assert "reference_prediction_rate" not in report.details

    def test_both_sides_present_reports_both(self, mock_mlops_settings_enabled):
        """Control: splitting the guards did not drop stats that should appear."""
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})

        report = self._monitor(["probability", "prediction"])._evidently_drift_check(
            frame, frame, save_report=False
        )

        for key in (
            "current_prob_mean",
            "reference_prob_mean",
            "current_prediction_rate",
            "reference_prediction_rate",
        ):
            assert key in report.details


class TestUnreadableEvidentlyMetrics:
    """A snapshot we cannot read a metric out of is not evidence of health."""

    def _monitor(self, metrics):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {"metrics": metrics}
        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=mock_report),
            "ValueDrift": MagicMock(),
        }
        return monitor

    def test_no_metrics_at_all_is_indeterminate(self, mock_mlops_settings_enabled):
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})

        report = self._monitor([])._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        assert "No drift metrics" in report.details["error"]

    def test_unrecognised_metric_name_is_indeterminate(self, mock_mlops_settings_enabled):
        """The Evidently API has already changed once (this targets 'v0.7+')."""
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [{"metric_name": "SomeRenamedMetric", "config": {}, "value": 0.5}]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True

    def test_readable_metrics_are_not_indeterminate(self, mock_mlops_settings_enabled):
        """Control."""
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            }
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is False


class TestCheckedInReferenceDataset:
    """Guard the artifact whose staleness caused #71."""

    def test_fp_reference_has_the_core_columns(self):
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "data/reference/fp_reference.parquet"
        if not path.exists():
            pytest.skip("no checked-in FP reference dataset")

        df = pd.read_parquet(path)

        # The 2026-01-17 file lacked novelty_score, which is what raised.
        for column in ("probability", "prediction", "novelty_score"):
            assert column in df.columns, f"reference is missing {column}"
        assert df["novelty_score"].notna().any()
        assert len(df) > 0
