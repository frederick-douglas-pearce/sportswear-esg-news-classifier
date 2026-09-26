"""Tests for the MLOps monitoring module."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
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
        # 1 reproduces the pre-#94 guard exactly (it tested `.empty`), so every
        # test written before the floor existed keeps its semantics. The tests
        # for the floor itself raise this deliberately.
        mock_settings.drift_min_sample_size = 1
        mock_settings.get_reports_dir = MagicMock(return_value=Path("/tmp/reports"))
        yield mock_settings


@pytest.fixture
def mock_mlops_settings_enabled():
    """Create mock MLOps settings with Evidently enabled."""
    with patch('src.mlops.monitoring.mlops_settings') as mock_settings:
        mock_settings.evidently_enabled = True
        mock_settings.drift_threshold = 0.1
        mock_settings.drift_min_sample_size = 1
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

    def test_no_reference_is_indeterminate_not_healthy(self, mock_mlops_settings_disabled):
        """A missing reference must not be answered by self-comparison.

        This used to split `current_data` in half and compare the halves. Two
        halves of one window share a distribution by construction, so the KS
        statistic came back ~0 and the report said `drift_detected=False`,
        `indeterminate=False` -- a healthy verdict manufactured out of the
        absence of a baseline, which is issue #71's class. Reachable today for
        `esg`, for EP, and on any fresh checkout.
        """
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

            assert report.classifier_type == "fp"
            assert report.indeterminate is True, (
                "no reference means nothing was measured"
            )
            assert report.drift_detected is False
            assert "reference" in report.details["error"].lower()
            # The absence itself has to be recorded, not just logged: the run
            # archive is what #75/#76 will read.
            assert "reference_path" in report.details
            assert report.details["current_size"] == 6

    def test_no_reference_does_not_consult_the_checkers(
        self, mock_mlops_settings_disabled
    ):
        """The refusal happens before any statistic is computed.

        Guards the mechanism rather than the return value: if the fallback were
        restored, these would run on the split halves and the assertions above
        could still be satisfied by a coincidence of the data.
        """
        with patch('src.mlops.monitoring.load_prediction_logs') as mock_load_logs, \
             patch('src.mlops.monitoring.load_reference_dataset') as mock_load_ref:

            mock_load_logs.return_value = pd.DataFrame({
                "probability": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "prediction": [0, 0, 1, 1, 1, 1],
            })
            mock_load_ref.side_effect = FileNotFoundError("No reference data")

            monitor = DriftMonitor("fp")
            with patch.object(monitor, "_legacy_drift_check") as mock_legacy, \
                 patch.object(monitor, "_evidently_drift_check") as mock_evidently:
                report = monitor.check_drift()

            mock_legacy.assert_not_called()
            mock_evidently.assert_not_called()
            assert report.indeterminate is True


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
    raising KeyError on every run from 2026-01-25 to 2026-09-06.
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

    def test_partial_reference_guards_each_frame_separately(
        self, mock_mlops_settings_enabled
    ):
        """Reference has `probability` but not `prediction` or `novelty_score`.

        One core column is comparable, so the pass proceeds to the stats block
        — which is the code under test here. `reference_prediction_rate` and
        `reference_novelty_mean` must be absent without raising, while their
        current-side counterparts are present.
        """
        current = pd.DataFrame(
            {
                "probability": [0.5, 0.6],
                "prediction": [0, 1],
                "novelty_score": [0.2, 0.8],
            }
        )
        reference = pd.DataFrame({"probability": [0.4, 0.5]})

        report = self._monitor(["probability"])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert report.indeterminate is False
        assert "reference_prob_mean" in report.details
        assert "current_prediction_rate" in report.details
        assert "reference_prediction_rate" not in report.details
        assert "current_novelty_mean" in report.details
        assert "reference_novelty_mean" not in report.details

    def test_reference_missing_probability_guards_that_read(
        self, mock_mlops_settings_enabled
    ):
        """Covers `reference_prob_mean` specifically.

        The brand-only case no longer reaches the stats block — widening the
        indeterminate guard to `total_core == 0` made it early-return above it —
        so without this test, reverting `reference_prob_mean`'s guard back to
        `current_data` would raise KeyError in production and no test would
        fail. The reference here keeps `prediction`, so one core column is
        comparable and the pass proceeds.
        """
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "novelty_score": [0.2, 0.8]}
        )
        reference = pd.DataFrame({"prediction": [0, 1], "novelty_score": [0.1, 0.3]})

        report = self._monitor(["prediction", "novelty_score"])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert report.indeterminate is False
        assert "current_prob_mean" in report.details
        assert "reference_prob_mean" not in report.details
        assert report.details["columns_missing_from_reference"] == ["probability"]

    def test_brand_only_reference_is_indeterminate(self, mock_mlops_settings_enabled):
        """No core metric assessed, so any drift score would be fabricated.

        An earlier version of this test asserted only that it did not raise,
        which blessed a HEALTHY verdict with a 0.0 score — the epic's defect
        class inside the test written to guard against it.
        """
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [1, 0]}
        )
        reference = pd.DataFrame({"brand_nike": [1, 1]})

        report = self._monitor(["brand_nike"])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert report.indeterminate is True
        assert report.drift_detected is False

    def test_both_paths_agree_on_a_brand_only_reference(
        self, mock_mlops_settings_enabled, disabled_monitor
    ):
        """The Evidently and legacy paths must reach the same verdict.

        They disagreed: legacy called this indeterminate while Evidently
        reported healthy, and `docs/MLOPS.md` documented the legacy answer as
        if it were universal.
        """
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [1, 0]}
        )
        reference = pd.DataFrame({"brand_nike": [1, 1]})

        evidently = self._monitor(["brand_nike"])._evidently_drift_check(
            current, reference, save_report=False
        )
        legacy = disabled_monitor.check_drift(
            current_data=current, reference_data=reference, save_report=False
        )

        assert evidently.indeterminate == legacy.indeterminate is True

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

    def _monitor(self, metrics, threshold=0.1):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = threshold
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
        assert "No core drift metrics" in report.details["error"]

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

    def test_an_unreadable_value_is_not_counted_as_no_drift(
        self, mock_mlops_settings_enabled
    ):
        """p=1.0 coercion made an unreadable metric *prop up* the guard.

        `p_value = value if isinstance(value, (int, float)) else 1.0` turned an
        unreadable metric into `col_drift=False` AND incremented `total_core`,
        so `total_core == 0` could not fire and the run reported a
        measured-looking 0.0 built from a metric nobody could read.
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": None,
            }
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        assert report.details["metrics_unreadable"] == ["probability"]
        # Pins the subset relation from the OTHER side: an unreadable metric is
        # in BOTH records. Without this, deleting the `columns_skipped` write
        # from the readability guard leaves the whole suite green while the
        # comments and docs go on telling readers `columns_skipped` is the
        # wider record.
        assert "probability" in report.details["columns_skipped"]
        # It must not have been silently scored as "no drift".
        assert "probability_p_value" not in report.details

    def test_one_unreadable_core_metric_does_not_hide_behind_a_readable_one(
        self, mock_mlops_settings_enabled
    ):
        """The partial case: one core metric readable, one not."""
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "prediction", "threshold": 0.05},
                "value": "not-a-number",
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        # Still assessed -- one core metric was readable -- but the gap is
        # recorded rather than counted as evidence of health.
        assert report.indeterminate is False
        assert report.details["metrics_unreadable"] == ["prediction"]
        assert "prediction_p_value" not in report.details

    def test_a_nan_value_is_not_counted_as_no_drift(
        self, mock_mlops_settings_enabled
    ):
        """The value Evidently actually returns, which `None` does not stand in for.

        `float('nan')` IS an instance of `float`, so it passes the readability
        guard; `nan < threshold` is then False and the column is scored as "did
        not drift" and counted toward `total_core`. Evidently's chi-square
        returns this for a column constant at the same value in both frames
        (#103) -- an input `_categorical_p_value` refuses on the legacy path,
        though under its own reason (`"one category in both frames"`, written at
        the category check, not at its finite check).
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": float("nan"),
            }
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        assert "probability_p_value" not in report.details
        assert "probability" not in report.details["columns_assessed"]
        # Recorded, not silently dropped. The string describes what this path
        # observed; it is not the legacy path's reason for this input.
        assert report.details["columns_skipped"]["probability"] == (
            "p-value is not finite"
        )

    def test_a_nan_value_is_not_recorded_as_unreadable(
        self, mock_mlops_settings_enabled
    ):
        """`metrics_unreadable` is the narrower set, and stays narrow.

        A NaN value WAS read -- the statistic is undefined, not the metric
        broken. Pinned so `metrics_unreadable` remains a proper subset of
        `columns_skipped` rather than the two coinciding by accident.
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": float("nan"),
            }
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert "probability" not in report.details.get("metrics_unreadable", [])
        assert "probability" in report.details["columns_skipped"]

    def test_a_nan_brand_metric_does_not_inflate_the_denominator(
        self, mock_mlops_settings_enabled
    ):
        """The denominator half of the defect, written so the VERDICT flips.

        `threshold=0.5`, not this class's default 0.1: one drifting brand and
        one NaN brand give `brand_drift_score` 0.5 before the fix and 1.0 after,
        and `0.5 > 0.1` is already True -- so at the default threshold a
        `drift_detected` assertion stays green on revert and proves nothing.
        """
        frame = pd.DataFrame(
            {
                "probability": [0.5, 0.6],
                "prediction": [0, 1],
                "brand_nike": [0, 1],
                "brand_puma": [0, 0],
            }
        )
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_nike", "threshold": 0.05},
                "value": 0.001,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_puma", "threshold": 0.05},
                "value": float("nan"),
            },
        ]

        report = self._monitor(metrics, threshold=0.5)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.details["brand_assessed_count"] == 1
        assert report.details["brand_drift_score"] == 1.0
        assert "brand_puma" not in report.details["columns_assessed"]
        # 1-of-1 brand columns drifted, not 1-of-2.
        assert report.drift_detected is True

    def test_one_nan_core_metric_does_not_hide_behind_a_readable_one(
        self, mock_mlops_settings_enabled
    ):
        """The partial-core shape the issue's end-to-end reproduction shows.

        An all-True `prediction` column alongside a healthy `probability`: still
        a verdict, because one core metric was readable, but the gap is recorded
        rather than counted as evidence of health.
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [1, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "prediction", "threshold": 0.05},
                "value": float("nan"),
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is False
        assert "prediction_p_value" not in report.details
        assert "prediction" not in report.details["columns_assessed"]
        assert report.details["columns_skipped"]["prediction"] == (
            "p-value is not finite"
        )

    def test_all_brand_metrics_nan_records_a_zero_denominator(
        self, mock_mlops_settings_enabled
    ):
        """`brand_drift_score` 0.0 out of nothing -- recorded, not fixed here.

        With every brand metric skipped, `total_brand` reaches 0 while brand
        columns WERE offered, so `brand_drift_score` is a fabricated 0.0 of
        exactly #105's class. Pinned so the new route to it is visible; making
        it indeterminate is #105's call, not this fix's.
        """
        frame = pd.DataFrame(
            {
                "probability": [0.5, 0.6],
                "prediction": [0, 1],
                "brand_nike": [0, 0],
                "brand_puma": [0, 0],
            }
        )
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_nike", "threshold": 0.05},
                "value": float("nan"),
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_puma", "threshold": 0.05},
                "value": float("nan"),
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.details["brand_assessed_count"] == 0
        assert report.details["brand_drift_score"] == 0.0
        assert sorted(report.details["columns_skipped"]) == [
            "brand_nike",
            "brand_puma",
        ]
        # `brand_drift_score == 0.0` alone does not discriminate: it reads 0.0
        # pre-fix too, as 0-of-2. The assessed record is what separates a
        # denominator of nothing from an honest zero.
        assert report.details["columns_assessed"] == ["probability"]

    def test_all_core_metrics_nan_does_not_report_an_unreadable_snapshot(
        self, mock_mlops_settings_enabled
    ):
        """The operator-facing reason must not name a cause that did not occur.

        `details["error"]` is the only report-derived prose that escapes to the
        operator email, the run archive and the CI summary -- `columns_skipped`
        reaches none of them until #104. Saying the metrics could not be *read*
        points the reader at a renamed metric or a changed snapshot shape, which
        is what this guard was originally added for. On this route every metric
        WAS read and came back non-finite, and skipping them is what makes the
        `total_core == 0` branch reachable at all.
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [1, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": float("nan"),
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "prediction", "threshold": 0.05},
                "value": float("nan"),
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        assert "could be read" not in report.details["error"]
        assert "usable" in report.details["error"]
        assert report.details["columns_assessed"] == []
        assert sorted(report.details["columns_skipped"]) == [
            "prediction",
            "probability",
        ]

    def test_an_infinite_value_is_not_counted_as_no_drift(
        self, mock_mlops_settings_enabled
    ):
        """The predicate is *finite*, not *not-NaN*.

        `inf < threshold` is False just as `nan < threshold` is, so `value !=
        value` would leave this one scored as "did not drift".
        """
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": float("inf"),
            }
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        assert "probability_p_value" not in report.details
        assert "probability" in report.details["columns_skipped"]


class TestDriftScoreMatchesTheDetection:
    """`drift_detected` reads both scores; the report must not carry one."""

    def _monitor(self, metrics, threshold=0.1):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = threshold
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {"metrics": metrics}
        mock_report = MagicMock()
        mock_report.run.return_value = mock_snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=mock_report),
            "ValueDrift": MagicMock(),
        }
        return monitor

    def _frame(self):
        return pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [0, 1]}
        )

    def test_brand_only_drift_does_not_report_a_zero_score(
        self, mock_mlops_settings_enabled
    ):
        """The live "score 0.0 exceeds 0.15" alert.

        `drift_detected = core_drifted > 0 or brand_drift_score > threshold`,
        but the report carried `core_drift_score` alone -- so brand-only drift
        emitted `drift_detected=True` with `drift_score=0.0`, an alert
        contradicted by its own number.
        """
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.9,  # not drifting
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "prediction", "threshold": 0.05},
                "value": 0.9,  # not drifting
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_nike", "threshold": 0.05},
                "value": 0.001,  # drifting
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            self._frame(), self._frame(), save_report=False
        )

        assert report.drift_detected is True
        assert report.drift_score > 0.0, (
            "a detection reported with a 0.0 score contradicts itself"
        )
        assert report.drift_score == report.details["brand_drift_score"]
        # Both components stay available.
        assert report.details["core_drift_score"] == 0.0

    def test_core_drift_still_reports_the_core_score(
        self, mock_mlops_settings_enabled
    ):
        """Control: core drift must not be masked by a quiet brand score."""
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.001,  # drifting
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "brand_nike", "threshold": 0.05},
                "value": 0.9,  # not drifting
            },
        ]

        report = self._monitor(metrics)._evidently_drift_check(
            self._frame(), self._frame(), save_report=False
        )

        assert report.drift_detected is True
        assert report.drift_score == 1.0
        assert report.details["brand_drift_score"] == 0.0


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


class TestReportIsSerializable:
    """A report must survive `json.dumps`, whatever produced its numbers.

    `drift_detected = overall_drift > self.threshold` yields a `numpy.bool_`
    when `overall_drift` came from scipy/pandas. `numpy.bool_` does NOT
    subclass `bool` (`numpy.float64` DOES subclass `float`, which is why the
    score slipped through unnoticed), so `json.dumps` refused it — crashing
    `print_summary_json` on every successful run of the legacy path, which is
    the path taken by default.
    """

    def test_numpy_scalars_are_coerced(self):
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=np.bool_(False),
            drift_score=np.float64(0.05),
            threshold=np.float64(0.1),
            details={},
            indeterminate=np.bool_(False),
        )

        assert type(report.drift_detected) is bool
        assert type(report.indeterminate) is bool
        assert type(report.drift_score) is float
        json.dumps(
            {
                "drift_detected": report.drift_detected,
                "drift_score": report.drift_score,
                "indeterminate": report.indeterminate,
                "threshold": report.threshold,
            }
        )

    def test_a_real_legacy_report_is_json_serializable(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """End to end through the real scipy path, which is where it came from."""
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference_data,
            save_report=False,
        )

        json.dumps(
            {
                "drift_detected": report.drift_detected,
                "drift_score": report.drift_score,
                "indeterminate": report.indeterminate,
            }
        )

    def test_values_are_preserved_not_just_coerced(self):
        """Control: coercion must not flatten a real result to False/0.0."""
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=np.bool_(True),
            drift_score=np.float64(0.42),
            threshold=np.float64(0.1),
            details={},
        )

        assert report.drift_detected is True
        assert report.drift_score == pytest.approx(0.42)


class TestDetailsAreSerializable:
    """`details` reaches the `--output` JSON that CI reads with `jq`.

    Evidently returns `numpy.float64` for a metric value, so
    `col_drift = p_value < p_value_threshold` is a `numpy.bool_`. Coercing only
    the four scalar fields left `--output` raising `TypeError` mid-write on the
    live path, exiting 1, and leaving a truncated file behind.
    """

    def test_numpy_inside_details_is_coerced(self):
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.0,
            threshold=0.1,
            details={
                "probability_drift": np.bool_(True),
                "probability_p_value": np.float64(0.0035),
                "reference_size": np.int64(934),
                "core_metrics_drifted": ["probability"],
                "nested": {"brand_nike_drift": np.bool_(False)},
            },
        )

        json.dumps(report.details)
        assert type(report.details["probability_drift"]) is bool
        assert type(report.details["reference_size"]) is int
        assert type(report.details["nested"]["brand_nike_drift"]) is bool

    def test_values_inside_details_are_preserved(self):
        """Control: coercion must not flatten what it converts."""
        report = DriftReport(
            classifier_type="fp",
            timestamp=datetime.now(),
            drift_detected=False,
            drift_score=0.0,
            threshold=0.1,
            details={"drifted": np.bool_(True), "p": np.float64(0.0035)},
        )

        assert report.details["drifted"] is True
        assert report.details["p"] == pytest.approx(0.0035)
        assert report.details == {"drifted": True, "p": pytest.approx(0.0035)}

    def test_the_full_output_report_is_serializable(self, mock_mlops_settings_enabled):
        """The exact dict `scripts/monitor_drift.py --output` writes."""
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        snapshot = MagicMock()
        # numpy value, as Evidently really returns.
        snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift",
                    "config": {"column": "probability", "threshold": 0.05},
                    "value": np.float64(0.0035),
                }
            ]
        }
        run = MagicMock()
        run.run.return_value = snapshot
        monitor._evidently = {"Report": MagicMock(return_value=run), "ValueDrift": MagicMock()}
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})

        report = monitor._evidently_drift_check(frame, frame, save_report=False)

        json.dumps(
            {
                "classifier_type": report.classifier_type,
                "drift_detected": report.drift_detected,
                "drift_score": report.drift_score,
                "threshold": report.threshold,
                "indeterminate": report.indeterminate,
                "details": report.details,
            }
        )


# ============================================================================
# Issue #102 -- the legacy path assessed 2 of 4 signal groups
# ============================================================================

class TestLegacyPathAssessesEverySignalGroup:
    """`_legacy_drift_check` is the default path, and the ImportError fallback.

    It compared `probability` and `prediction` and nothing else: a total
    distributional shift in `novelty_score`, or in any `brand_*` column, was
    reported as healthy while `columns_missing_from_reference` stayed empty
    because those columns ARE in the reference. They were simply never looked
    at, so a partial check was indistinguishable from a whole one (issue #102).
    """

    @staticmethod
    def _frames(novelty_ref, novelty_curr, n=60):
        """Two frames identical but for `novelty_score`."""
        rng = np.random.default_rng(7)
        return (
            pd.DataFrame(
                {
                    "probability": rng.uniform(0.4, 0.6, n),
                    "prediction": np.tile([0, 1], n // 2),
                    "novelty_score": novelty_ref,
                }
            ),
            pd.DataFrame(
                {
                    "probability": rng.uniform(0.4, 0.6, n),
                    "prediction": np.tile([0, 1], n // 2),
                    "novelty_score": novelty_curr,
                }
            ),
        )

    def test_a_total_novelty_shift_does_not_read_as_healthy(self, disabled_monitor):
        """The issue's own demonstration, as a test.

        Injected drift: 0.0-0.1 in the reference, 0.9-1.0 in every current row.
        Before #102 this returned `drift_detected=False` with a drift_score of
        0.063 built from `probability` and `prediction` alone.
        """
        rng = np.random.default_rng(11)
        reference, current = self._frames(
            rng.uniform(0.0, 0.1, 60), rng.uniform(0.9, 1.0, 60)
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.drift_detected is True
        assert report.indeterminate is False
        assert "novelty_score" in report.details["columns_assessed"]
        assert report.details["novelty_score_ks_statistic"] > 0.9

    def test_brand_drift_reaches_the_verdict(self, disabled_monitor):
        """A brand column that flips wholesale is assessed and it bites.

        `brand_*` columns were never read on this path at all, so this frame
        used to report healthy.
        """
        n = 80
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_nike": [1] * (n // 2) + [0] * (n // 2),
                "brand_puma": [1, 0] * (n // 2),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                # brand_nike flips from 50% of rows to ~5%
                "brand_nike": [1] * 4 + [0] * (n - 4),
                "brand_puma": [1, 0] * (n // 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "brand_nike" in report.details["columns_assessed"]
        assert "brand_nike" in report.details["brand_metrics_drifted"]
        assert report.details["brand_drift_score"] > 0
        assert report.drift_detected is True

    def test_core_score_keeps_its_effect_size_meaning(self, disabled_monitor):
        """AC-5: brand must not redefine the core score (D017).

        `DRIFT_THRESHOLD` is tuned against an effect-size instrument and the run
        archive holds history computed that way, so `probability`/`prediction`/
        `novelty_score` stay magnitudes rather than a fraction of significant
        tests. Brand contributes only via its own component.
        """
        n = 60
        frame = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(frame.copy(), frame.copy())

        # Identical frames: KS statistic is 0 and the rate difference is 0, so
        # the core score is the max of those magnitudes -- not 0-of-2-drifted.
        assert report.details["core_drift_score"] == pytest.approx(0.0, abs=1e-9)
        assert report.drift_score == pytest.approx(0.0, abs=1e-9)
        assert report.details["brand_drift_score"] == 0.0

    def test_a_degenerate_brand_column_is_not_evidence_of_no_drift(
        self, disabled_monitor
    ):
        """AC-6: #102's fix must not plant #103's shape.

        A brand mentioned in no article is an all-zero column. Its chi-square
        p-value is undefined; written naively `NaN < 0.01` is False, so the
        column would count as "not drifted" WHILE still incrementing the
        denominator -- a NaN p-value treated as evidence of no drift, diluting
        `brand_drift_score`. It must be skipped and recorded instead.
        """
        n = 40
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_absent": [0] * n,
                "brand_everywhere": [1] * n,
                "brand_real": [1] * (n // 2) + [0] * (n // 2),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_absent": [0] * n,
                "brand_everywhere": [1] * n,
                "brand_real": [1] * 2 + [0] * (n - 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        skipped = report.details["columns_skipped"]
        assert "brand_absent" in skipped
        assert "brand_everywhere" in skipped
        assert "brand_absent" not in report.details["columns_assessed"]
        # The denominator counts only what was actually assessed: one column.
        assert report.details["brand_assessed_count"] == 1
        assert report.details["brand_drift_score"] == pytest.approx(1.0)

    def test_a_degenerate_novelty_column_is_skipped_not_scored(
        self, disabled_monitor
    ):
        """Same guard on the novelty branch: constant in both frames."""
        reference, current = self._frames(np.full(60, 0.5), np.full(60, 0.5))

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "novelty_score" in report.details["columns_skipped"]
        assert "novelty_score" not in report.details["columns_assessed"]
        assert "novelty_score_ks_statistic" not in report.details

    def test_an_all_nan_novelty_column_is_skipped_not_scored(
        self, disabled_monitor
    ):
        """NaN-drop leaves nothing to compare, which is not 'no drift'."""
        reference, current = self._frames(np.full(60, np.nan), np.full(60, np.nan))

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "novelty_score" in report.details["columns_skipped"]
        assert "novelty_score_ks_statistic" not in report.details

    def test_columns_assessed_lists_what_was_actually_read(self, disabled_monitor):
        """The record #104 will carry and #105 will cross-check.

        It is what was ASSESSED, not what was offered -- that is the whole
        distinction, since `columns_missing_from_reference` already covers
        offered-but-absent and reported `[]` on exactly the frames #102 is about.
        """
        n = 40
        reference, current = self._frames(
            np.linspace(0.1, 0.9, n), np.linspace(0.1, 0.9, n), n=n
        )
        reference["brand_nike"] = [1, 0] * (n // 2)
        current["brand_nike"] = [1, 0] * (n // 2)

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert set(report.details["columns_assessed"]) == {
            "probability",
            "prediction",
            "novelty_score",
            "brand_nike",
        }
        assert report.details["columns_missing_from_reference"] == []

    def test_both_paths_report_the_same_assessed_record(
        self, mock_mlops_settings_enabled, disabled_monitor
    ):
        """AC-2: same `details` keys, NOT the same number (D017, D008).

        The two remain different instruments -- legacy core is an effect size,
        Evidently core is a fraction of significant tests -- so this asserts the
        shape they share, which is what #104 lifts into the summary.
        """
        shared = ("columns_assessed", "columns_skipped", "brand_drift_score")
        frame = pd.DataFrame(
            {"probability": [0.5, 0.6, 0.4, 0.7], "prediction": [0, 1, 0, 1]}
        )

        monitor = DriftMonitor("fp")
        snapshot = MagicMock()
        snapshot.dict.return_value = {
            "metrics": [
                {
                    "metric_name": "ValueDrift(column=probability)",
                    "config": {"column": "probability", "threshold": 0.05},
                    "value": 0.5,
                }
            ]
        }
        run = MagicMock()
        run.run.return_value = snapshot
        monitor._evidently = {
            "Report": MagicMock(return_value=run),
            "ValueDrift": MagicMock(),
        }

        evidently = monitor._evidently_drift_check(frame, frame, save_report=False)
        legacy = disabled_monitor._legacy_drift_check(frame, frame)

        for key in shared:
            assert key in evidently.details, f"Evidently path missing {key!r}"
            assert key in legacy.details, f"legacy path missing {key!r}"

    def test_a_brand_only_reference_is_still_indeterminate(self, disabled_monitor):
        """Control: assessing brand did NOT make brand sufficient on its own.

        The indeterminacy guard stays keyed on the CORE columns, matching the
        Evidently path's `total_core == 0`. Reading a brand assessment as a
        verdict would re-open exactly what that guard closed.
        """
        current = pd.DataFrame(
            {"probability": [0.5, 0.6], "prediction": [0, 1], "brand_nike": [1, 0]}
        )
        reference = pd.DataFrame({"brand_nike": [1, 1], "brand_puma": [0, 1]})

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.indeterminate is True
        assert report.drift_detected is False


# ============================================================================
# Round-2 review findings on #102, and #94 pulled forward
# ============================================================================

class TestNoUnmeasuredColumnReachesTheVerdictAsHealth:
    """Round 1 of #102's review found the fix reproducing the defect it fixed.

    `_comparable_series` was written to stop an unmeasurable column being
    counted as evidence of no drift, and was applied to `novelty_score` alone.
    Its two sibling core branches went unguarded, so a single NaN in
    `probability` poisoned `max(drift_scores)` -- NaN comparisons are False, so
    `max` keeps whichever operand it started with and `nan > threshold` is False
    -- and the run reported healthy over a total novelty shift, with
    `columns_assessed` naming `probability` as measured.
    """

    def test_a_nan_probability_does_not_mask_a_total_novelty_shift(
        self, disabled_monitor
    ):
        """#102's own demonstration, with one NaN column added."""
        n = 60
        reference = pd.DataFrame(
            {
                "probability": np.full(n, np.nan),
                "prediction": np.tile([0, 1], n // 2),
                "novelty_score": np.linspace(0.0, 0.1, n),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.full(n, np.nan),
                "prediction": np.tile([0, 1], n // 2),
                "novelty_score": np.linspace(0.9, 1.0, n),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.drift_detected is True
        assert not np.isnan(report.drift_score)
        assert "probability" not in report.details["columns_assessed"]
        assert "probability" in report.details["columns_skipped"]

    def test_a_nan_prediction_does_not_mask_a_total_novelty_shift(
        self, disabled_monitor
    ):
        """The middle branch of the three the guard was widened to cover.

        `probability` and `novelty_score` each have a test; `prediction` had
        none, and it is the one branch that does not use `ks_2samp` -- it takes
        `.mean()` of the surviving rows, so an unguarded all-NaN column yields
        `nan` from a different expression than its two siblings.
        """
        n = 60
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.full(n, np.nan),
                "novelty_score": np.linspace(0.0, 0.1, n),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.full(n, np.nan),
                "novelty_score": np.linspace(0.9, 1.0, n),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.drift_detected is True
        assert not np.isnan(report.drift_score)
        assert "prediction" not in report.details["columns_assessed"]
        assert "prediction" in report.details["columns_skipped"]
        assert "prediction_rate_diff" not in report.details

    def test_the_reported_score_is_never_nan(self, disabled_monitor):
        """A NaN score serializes as a bare `NaN`, which is not valid JSON.

        `scripts/monitor_drift.py` writes `report.details` and the score into
        the `--output` file that `.github/workflows/monitoring.yml` reads.
        """
        n = 40
        frame = pd.DataFrame(
            {
                "probability": np.full(n, np.nan),
                "prediction": np.tile([0, 1], n // 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(frame.copy(), frame.copy())

        assert not np.isnan(report.drift_score)
        json.dumps({"drift_score": report.drift_score})

    def test_a_brand_column_absent_from_the_reference_is_recorded(
        self, disabled_monitor
    ):
        """It used to be `continue`d past, leaving no trace in any field.

        `_missing_from_reference` covers `CORE_DRIFT_COLUMNS` only by design, so
        nothing else would have caught it. Reachable: `_add_brand_columns` runs
        only in `load_predictions_from_database`, so a reference built from
        files carries no brand column at all and every one of them vanished.
        """
        n = 40
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_nike": [1, 0] * (n // 2),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_nike": [1, 0] * (n // 2),
                "brand_adidas": [1, 0] * (n // 2),
                "brand_puma": [1, 0] * (n // 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        skipped = report.details["columns_skipped"]
        assert "brand_adidas" in skipped
        assert "brand_puma" in skipped
        assert "brand_nike" in report.details["columns_assessed"]
        assert report.details["brand_assessed_count"] == 1


class TestCategoricalPValueIsLabelSafe:
    """A dtype mismatch made `value_counts().get(...)` index positionally.

    `value_counts()` sorts by count descending, so a positional read takes the
    most-frequent category rather than the labelled one. A total flip then built
    a symmetric table and returned p=1.0 -- a confident "no drift" on the
    loudest possible signal, recorded as assessed.
    """

    def test_a_total_flip_is_detected_across_an_int_bool_mismatch(self):
        from src.mlops.monitoring import _categorical_p_value

        reference = pd.Series([1] * 10 + [0] * 90)
        current_bool = pd.Series([True] * 90 + [False] * 10)
        current_int = pd.Series([1] * 90 + [0] * 10)

        p_mixed, _ = _categorical_p_value(reference, current_bool, 1)
        p_same, _ = _categorical_p_value(reference, current_int, 1)

        assert p_mixed is not None
        assert p_mixed == pytest.approx(p_same)
        assert p_mixed < 0.01

    def test_the_reverse_mismatch_is_assessed_not_skipped(self):
        from src.mlops.monitoring import _categorical_p_value

        reference_bool = pd.Series([True] * 10 + [False] * 90)
        current_int = pd.Series([1] * 90 + [0] * 10)

        assert _categorical_p_value(reference_bool, current_int, 1)[0] is not None

    def test_an_uncomparable_dtype_is_skipped_not_raised(self):
        """`sorted()` over mixed str/int raised, aborting the whole check."""
        from src.mlops.monitoring import _categorical_p_value

        assert (
            _categorical_p_value(
                pd.Series(["0", "1"] * 50), pd.Series([0, 1] * 50), 1
            )[0]
            is None
        )


class TestMinimumSampleSize:
    """#94, pulled forward: too few rows to mean anything is not health.

    A statistic computed from a handful of rows reports as confidently as one
    computed from plenty. The floor applies to the reference and the current
    window independently -- a 900-row reference against a 3-row window is as
    untrustworthy as the reverse -- and per column after the NaN drop, which is
    where a nullable column like `novelty_score` lands.
    """

    @pytest.fixture(autouse=True)
    def _floor(self, mock_mlops_settings_disabled):
        """Raise the floor off the fixture default, which exists for OTHER tests."""
        mock_mlops_settings_disabled.drift_min_sample_size = 30
        return mock_mlops_settings_disabled

    def test_the_shipped_default_is_the_floor_these_tests_exercise(self):
        """Pin the production default, since the fixture above overrides it.

        Without this the floor could be changed in `config.py` and every test
        here would keep passing against the value it sets for itself.
        """
        import os
        from unittest.mock import patch as _patch

        from src.mlops.config import MLOpsSettings

        with _patch.dict(os.environ, {}, clear=True):
            assert MLOpsSettings().drift_min_sample_size == 30

    def test_a_tiny_current_window_is_indeterminate_not_healthy(
        self, disabled_monitor, reference_data
    ):
        current = pd.DataFrame(
            {"probability": [0.5, 0.6, 0.4], "prediction": [0, 1, 0]}
        )

        report = disabled_monitor.check_drift(
            current_data=current, reference_data=reference_data, save_report=False
        )

        assert report.indeterminate is True
        assert report.drift_detected is False

    def test_a_tiny_reference_is_indeterminate_not_healthy(
        self, disabled_monitor, current_data_no_drift
    ):
        reference = pd.DataFrame(
            {"probability": [0.5, 0.6, 0.4], "prediction": [0, 1, 0]}
        )

        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference,
            save_report=False,
        )

        assert report.indeterminate is True

    def test_a_single_surviving_row_does_not_drive_the_verdict(
        self, disabled_monitor
    ):
        """`novelty_score` is the one nullable core column.

        One non-null row yielded a KS statistic of 1.0 -- the loudest value the
        instrument can produce -- and the legacy path appends the statistic and
        never reads the p-value that says it means nothing.
        """
        n = 60
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "novelty_score": np.full(n, 0.1),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "novelty_score": [0.9] + [np.nan] * (n - 1),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "novelty_score" in report.details["columns_skipped"]
        assert "novelty_score" not in report.details["columns_assessed"]
        assert "novelty_score_ks_statistic" not in report.details
        assert report.drift_detected is False

    def test_the_control_still_passes_at_a_real_sample_size(
        self, disabled_monitor, reference_data, current_data_no_drift
    ):
        """The floor has not made every run indeterminate."""
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift,
            reference_data=reference_data,
            save_report=False,
        )

        assert report.indeterminate is False


class TestReportedScoreSaysWhichComponentProducedIt:
    """`drift_score = max(core, brand)` mixes two instruments in one number.

    Keeping the max is the ruled design -- it mirrors the Evidently path and is
    what stops brand-only drift alerting as "score 0.0 exceeds 0.15". What was
    missing is any record of which component won, so a reader of the archive
    cannot tell an effect size from a fraction of significant tests.
    """

    def test_brand_driven_score_is_labelled(self, disabled_monitor):
        n = 80
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_nike": [1] * (n // 2) + [0] * (n // 2),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.4, 0.6, n),
                "prediction": np.tile([0, 1], n // 2),
                "brand_nike": [1] * 4 + [0] * (n - 4),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.details["drift_score_source"] == "brand"
        assert report.drift_score == report.details["brand_drift_score"]

    def test_core_driven_score_is_labelled(self, disabled_monitor):
        n = 60
        reference = pd.DataFrame(
            {
                "probability": np.linspace(0.0, 0.1, n),
                "prediction": np.tile([0, 1], n // 2),
            }
        )
        current = pd.DataFrame(
            {
                "probability": np.linspace(0.9, 1.0, n),
                "prediction": np.tile([0, 1], n // 2),
            }
        )

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert report.details["drift_score_source"] == "core"
        assert report.drift_score == report.details["core_drift_score"]


def _drift_frame(n_rows: int, **brands: int) -> pd.DataFrame:
    """A frame the legacy path can score, with brand columns at stated rates.

    `probability` and `prediction` are present and steady so the report is a
    real verdict rather than the indeterminate return -- these tests are about
    what the brand component counts, not about whether anything was measured.
    """
    data: dict[str, Any] = {
        "probability": np.linspace(0.4, 0.6, n_rows),
        "prediction": np.array([0, 1] * (n_rows // 2) + [0] * (n_rows % 2)),
    }
    for name, n_positive in brands.items():
        data[name] = [1] * n_positive + [0] * (n_rows - n_positive)
    return pd.DataFrame(data)


class TestTheSampleFloorReachesBrandColumnsToo:
    """#94's floor was threaded through the core guard only (round-2 finding B1).

    `_comparable_series` carries `min_size`, and only `probability`,
    `prediction` and `novelty_score` call it. The brand loop called
    `_categorical_p_value` with no sample-size argument at all, while five
    places said the floor applied per column after the NaN drop.
    """

    @pytest.fixture(autouse=True)
    def _floor(self, mock_mlops_settings_disabled):
        """Production floor, not the fixture default that exists for other tests."""
        mock_mlops_settings_disabled.drift_min_sample_size = 30
        return mock_mlops_settings_disabled

    def test_the_row_floor_applies_to_brand_columns_after_the_nan_drop(
        self, disabled_monitor
    ):
        reference = _drift_frame(934, brand_nike=336)
        current = _drift_frame(73, brand_nike=19)
        current["brand_nike"] = current["brand_nike"].astype(float)
        current.loc[25:, "brand_nike"] = np.nan  # 25 non-null, floor is 30

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "brand_nike" in report.details["columns_skipped"]
        assert "brand_nike" not in report.details["columns_assessed"]
        assert report.details["brand_assessed_count"] == 0

    def test_a_column_above_the_floor_is_still_assessed(self, disabled_monitor):
        """The floor has not emptied the brand component."""
        reference = _drift_frame(934, brand_nike=336)
        current = _drift_frame(73, brand_nike=19)

        report = disabled_monitor._legacy_drift_check(current, reference)

        assert "brand_nike" in report.details["columns_assessed"]
        assert "brand_nike" not in report.details["columns_skipped"]
        assert report.details["brand_assessed_count"] == 1

    def test_a_rare_brand_still_detects_a_spike(self, disabled_monitor):
        """Why there is no minimum-expected-cell floor here.

        The obvious reading of `brand_li-ning` -- 3 positives in the shipped
        934-row reference, p=1.0 against a quiet week, smallest expected cell
        0.22 -- is that it cannot detect anything and only dilutes the
        denominator. It is power-ASYMMETRIC, not powerless: it cannot evidence a
        decrease and it detects an increase sharply. A rare brand suddenly
        appearing is the drift this project most wants to hear about, so a floor
        that discards the dead reading discards this detection with it.
        """
        reference = _drift_frame(934, brand_li_ning=3)
        quiet = _drift_frame(73, brand_li_ning=0)
        spike = _drift_frame(73, brand_li_ning=3)

        quiet_report = disabled_monitor._legacy_drift_check(quiet, reference)
        spike_report = disabled_monitor._legacy_drift_check(spike, reference)

        assert quiet_report.details["brand_drifted_count"] == 0
        assert spike_report.details["brand_metrics_drifted"] == ["brand_li_ning"]
        assert spike_report.details["brand_li_ning_p_value"] < 0.01

    def test_each_cause_of_a_skip_is_named_distinctly(self, disabled_monitor):
        """Four different failures used to arrive as one string, "not comparable".

        The reason is the only record of why a column produced no reading, so
        collapsing the causes makes the record unable to answer the question it
        exists for.
        """
        reference = _drift_frame(934, brand_thin=336, brand_constant=0)
        current = _drift_frame(73, brand_thin=19, brand_constant=0)
        current["brand_thin"] = current["brand_thin"].astype(float)
        current.loc[25:, "brand_thin"] = np.nan
        current["brand_absent"] = [1] * 19 + [0] * 54

        report = disabled_monitor._legacy_drift_check(current, reference)

        skipped = report.details["columns_skipped"]
        reasons = {
            skipped["brand_thin"],
            skipped["brand_constant"],
            skipped["brand_absent"],
        }
        assert len(reasons) == 3, f"causes collapsed: {reasons}"


class TestCategoricalPValueReportsWhyItDeclined:
    """The helper returns `(p_value, reason)`; exactly one of the two is set."""

    def test_a_readable_comparison_returns_a_p_value_and_no_reason(self):
        from src.mlops.monitoring import _categorical_p_value

        p_value, reason = _categorical_p_value(
            pd.Series([1] * 300 + [0] * 634), pd.Series([1] * 19 + [0] * 54), 1
        )

        assert reason is None
        assert p_value is not None

    def test_a_declined_comparison_returns_a_reason_and_no_p_value(self):
        from src.mlops.monitoring import _categorical_p_value

        p_value, reason = _categorical_p_value(
            pd.Series([0] * 934), pd.Series([0] * 73), 1
        )

        assert p_value is None
        assert reason

    def test_the_row_floor_is_honoured_when_passed(self):
        from src.mlops.monitoring import _categorical_p_value

        reference = pd.Series([1] * 300 + [0] * 634)
        current = pd.Series([1] * 10 + [0] * 15)  # 25 rows

        assert _categorical_p_value(reference, current, 30)[0] is None
        assert _categorical_p_value(reference, current, 1)[0] is not None


class TestReferenceProvenanceInReport:
    """Every report after the reference is loaded names its baseline (#97 AC-3)."""

    @staticmethod
    def _frames():
        now = pd.Timestamp("2026-09-25T12:00:00Z")
        reference = pd.DataFrame({
            "timestamp": pd.date_range(now - pd.Timedelta(days=40), periods=100, freq="h"),
            "probability": np.random.default_rng(1).uniform(0.3, 0.7, 100),
            "prediction": np.random.default_rng(2).choice([0, 1], 100),
        })
        reference.attrs["reference_window"] = {"requested_end": "2026-09-18T12:00:00+00:00"}
        current = pd.DataFrame({
            "timestamp": pd.date_range(now - pd.Timedelta(days=5), periods=50, freq="h"),
            "probability": np.random.default_rng(3).uniform(0.3, 0.7, 50),
            "prediction": np.random.default_rng(4).choice([0, 1], 50),
        })
        return reference, current

    def test_legacy_path_report_carries_it(self, disabled_monitor):
        reference, current = self._frames()

        report = disabled_monitor.check_drift(current_data=current, reference_data=reference)

        assert report.details["reference_window"] == reference.attrs["reference_window"]
        assert report.details["reference_overlaps_current"] is False
        assert report.details["reference_observed"]["end"]

    def test_evidently_path_report_carries_it(self, disabled_monitor):
        """Attached in check_drift, after whichever checker ran."""
        reference, current = self._frames()
        disabled_monitor.enabled = True
        stub = DriftReport(
            classifier_type="fp", timestamp=datetime.now(), drift_detected=False,
            drift_score=0.0, threshold=0.1, details={"columns_checked": ["probability"]},
        )
        with patch.object(disabled_monitor, "_evidently_drift_check", return_value=stub):
            report = disabled_monitor.check_drift(current_data=current, reference_data=reference)

        assert report.details["reference_window"] == reference.attrs["reference_window"]
        assert report.details["reference_overlaps_current"] is False

    def test_insufficient_data_report_carries_it(self, disabled_monitor, mock_mlops_settings_disabled):
        reference, current = self._frames()
        mock_mlops_settings_disabled.drift_min_sample_size = 1000

        report = disabled_monitor.check_drift(current_data=current, reference_data=reference)

        assert report.indeterminate
        assert report.details["reference_window"] == reference.attrs["reference_window"]
        assert report.details["reference_overlaps_current"] is False


class TestEveryReturnCarriesCoverageKeys:
    """Every report `check_drift` can return names its coverage (#104, D022).

    `None` means not recorded or not applicable on that path; an empty value
    means measured and none found. The summary emitter guarantees the keys
    reach the archive whatever a path does, so these tests pin the VALUES --
    the part a direct reader of `details` (the `--verbose` dump) depends on.
    """

    @staticmethod
    def _evidently_monitor(metrics):
        monitor = DriftMonitor.__new__(DriftMonitor)
        monitor.classifier_type = "fp"
        monitor.enabled = True
        monitor.threshold = 0.1
        snapshot = MagicMock()
        snapshot.dict.return_value = {"metrics": metrics}
        run = MagicMock()
        run.run.return_value = snapshot
        monitor._evidently = {"Report": MagicMock(return_value=run), "ValueDrift": MagicMock()}
        return monitor

    @staticmethod
    def _coverage(report):
        from src.mlops.monitoring import COVERAGE_KEYS, OFFERED_KEY

        keys = (*COVERAGE_KEYS, OFFERED_KEY)
        missing = [k for k in keys if k not in report.details]
        assert not missing, f"coverage keys absent: {missing}"
        return {k: report.details[k] for k in keys}

    def test_the_key_set_is_named_once(self):
        from src.mlops.monitoring import COVERAGE_KEYS, OFFERED_KEY

        assert COVERAGE_KEYS == (
            "columns_assessed",
            "columns_skipped",
            "columns_missing_from_reference",
            "metrics_unreadable",
        )
        assert OFFERED_KEY == "columns_checked"

    def test_no_reference(self, disabled_monitor, current_data_no_drift):
        """With no reference, what it lacks is unknowable -- None, not []."""
        with patch(
            "src.mlops.monitoring.load_reference_dataset",
            side_effect=FileNotFoundError("absent"),
        ):
            report = disabled_monitor.check_drift(current_data=current_data_no_drift)

        assert report.indeterminate is True
        assert self._coverage(report) == {
            "columns_assessed": [],
            "columns_skipped": {},
            "columns_missing_from_reference": None,
            "metrics_unreadable": None,
            "columns_checked": None,
        }

    def test_below_the_sample_floor(
        self, disabled_monitor, mock_mlops_settings_disabled, reference_data
    ):
        """Both frames exist, so what the reference lacks IS known."""
        mock_mlops_settings_disabled.drift_min_sample_size = 1000
        current = reference_data.assign(novelty_score=0.5)

        report = disabled_monitor.check_drift(
            current_data=current, reference_data=reference_data
        )

        assert report.indeterminate is True
        assert self._coverage(report) == {
            "columns_assessed": [],
            "columns_skipped": {},
            "columns_missing_from_reference": ["novelty_score"],
            "metrics_unreadable": None,
            "columns_checked": None,
        }

    def test_evidently_no_common_columns(self, mock_mlops_settings_enabled):
        """No Report was run, so readability was never checked: None."""
        current = pd.DataFrame({"probability": [0.5, 0.6]})
        reference = pd.DataFrame({"other": [1, 2]})

        report = self._evidently_monitor([])._evidently_drift_check(
            current, reference, save_report=False
        )

        assert report.indeterminate is True
        assert self._coverage(report) == {
            "columns_assessed": [],
            "columns_skipped": {},
            "columns_missing_from_reference": ["probability"],
            "metrics_unreadable": None,
            "columns_checked": [],
        }

    def test_evidently_no_usable_core_metric(self, mock_mlops_settings_enabled):
        """The Report ran and nothing in it was unreadable: [] is a measurement."""
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})

        report = self._evidently_monitor([])._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is True
        coverage = self._coverage(report)
        assert coverage["metrics_unreadable"] == []
        assert coverage["columns_checked"] == ["probability", "prediction"]
        assert coverage["columns_assessed"] == []

    def test_evidently_verdict(self, mock_mlops_settings_enabled):
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            }
        ]

        report = self._evidently_monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert report.indeterminate is False
        coverage = self._coverage(report)
        assert coverage["metrics_unreadable"] == []
        assert coverage["columns_assessed"] == ["probability"]
        assert coverage["columns_checked"] == ["probability", "prediction"]

    def test_evidently_unreadable_metric_is_listed(self, mock_mlops_settings_enabled):
        frame = pd.DataFrame({"probability": [0.5, 0.6], "prediction": [0, 1]})
        metrics = [
            {
                "metric_name": "ValueDrift",
                "config": {"column": "probability", "threshold": 0.05},
                "value": 0.5,
            },
            {
                "metric_name": "ValueDrift",
                "config": {"column": "prediction", "threshold": 0.05},
                "value": "not a number",
            },
        ]

        report = self._evidently_monitor(metrics)._evidently_drift_check(
            frame, frame, save_report=False
        )

        assert self._coverage(report)["metrics_unreadable"] == ["prediction"]

    def test_legacy_verdict(self, disabled_monitor, reference_data, current_data_no_drift):
        """The legacy path has no metric snapshot and no offered set: None."""
        report = disabled_monitor.check_drift(
            current_data=current_data_no_drift, reference_data=reference_data
        )

        assert report.indeterminate is False
        coverage = self._coverage(report)
        assert coverage["metrics_unreadable"] is None
        assert coverage["columns_checked"] is None
        assert "probability" in coverage["columns_assessed"]

    def test_legacy_no_comparable_column(self, disabled_monitor):
        current = pd.DataFrame({"other": [1, 2, 3]})

        report = disabled_monitor.check_drift(current_data=current, reference_data=current)

        assert report.indeterminate is True
        coverage = self._coverage(report)
        assert coverage["metrics_unreadable"] is None
        assert coverage["columns_checked"] is None
