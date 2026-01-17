"""Tests for drift monitoring workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.state import StateManager, WorkflowStatus
from src.agent.workflows.drift_monitoring import (
    DriftMonitoringWorkflow,
    _parse_drift_output,
    check_ep_drift,
    check_fp_drift,
    evaluate_drift_results,
    generate_drift_report,
    send_drift_alerts,
)


@pytest.fixture
def state_manager(tmp_path):
    """Create a fresh StateManager instance."""
    state_file = tmp_path / "state.yaml"
    return StateManager(state_file=state_file)


@pytest.fixture
def mock_workflow(state_manager):
    """Create a mock workflow for testing."""
    return DriftMonitoringWorkflow(state_manager=state_manager, dry_run=True)


class TestParseDriftOutput:
    """Tests for drift output parsing."""

    def test_parse_drift_detected_yes(self):
        """Test parsing when drift is detected."""
        output = """
DRIFT MONITORING REPORT - FP
============================================================

Timestamp: 2024-01-16 10:00:00
Drift Detected: YES
Drift Score: 0.1500 (threshold: 0.1000)

============================================================
⚠️  ACTION REQUIRED: Drift detected - consider retraining
"""
        result = _parse_drift_output(output, "fp")

        assert result.get("fp_drift_detected") is True
        assert result.get("fp_drift_score") == 0.15
        assert result.get("fp_action_required") is True

    def test_parse_drift_detected_no(self):
        """Test parsing when no drift detected."""
        output = """
DRIFT MONITORING REPORT - FP
============================================================

Timestamp: 2024-01-16 10:00:00
Drift Detected: NO
Drift Score: 0.0500 (threshold: 0.1000)

============================================================
✅ Status: Healthy - no significant drift detected
"""
        result = _parse_drift_output(output, "fp")

        assert result.get("fp_drift_detected") is False
        assert result.get("fp_drift_score") == 0.05
        assert result.get("fp_healthy") is True

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        result = _parse_drift_output("", "fp")
        assert result == {}


class TestCheckFpDrift:
    """Tests for FP drift check step."""

    def test_check_fp_drift_success(self, mock_workflow):
        """Test successful FP drift check."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = MagicMock(
                success=True,
                exit_code=0,
                duration_seconds=5.0,
                stdout="Drift Detected: NO\nDrift Score: 0.05 (threshold: 0.1)\nHealthy",
                stderr="",
            )

            result = check_fp_drift(mock_workflow, {"drift_days": 7})

            assert result["fp_drift_check_success"] is True
            assert result["fp_drift_exit_code"] == 0
            mock_run.assert_called_once_with(
                classifier="fp",
                days=7,
                from_db=True,
                html_report=False,
                alert=False,
            )

    def test_check_fp_drift_failure(self, mock_workflow):
        """Test FP drift check failure."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = MagicMock(
                success=False,
                exit_code=1,
                duration_seconds=2.0,
                stdout="",
                stderr="Database connection failed",
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_drift_check_success"] is False
            assert "fp_drift_error" in result


class TestCheckEpDrift:
    """Tests for EP drift check step."""

    def test_check_ep_drift_success(self, mock_workflow):
        """Test successful EP drift check."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = MagicMock(
                success=True,
                exit_code=0,
                duration_seconds=4.0,
                stdout="Drift Detected: YES\nDrift Score: 0.15\nACTION REQUIRED",
                stderr="",
            )

            result = check_ep_drift(mock_workflow, {"drift_days": 14})

            assert result["ep_drift_check_success"] is True
            mock_run.assert_called_once_with(
                classifier="ep",
                days=14,
                from_db=True,
                html_report=False,
                alert=False,
            )


class TestEvaluateDriftResults:
    """Tests for drift evaluation step."""

    def test_no_drift_detected(self, mock_workflow):
        """Test evaluation when no drift detected."""
        context = {
            "fp_drift_detected": False,
            "ep_drift_detected": False,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["any_drift_detected"] is False
        assert result["classifiers_with_drift"] == []
        assert "No action needed" in result["recommendation"]

    def test_fp_drift_detected(self, mock_workflow):
        """Test evaluation when FP drift detected."""
        context = {
            "fp_drift_detected": True,
            "ep_drift_detected": False,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["any_drift_detected"] is True
        assert "fp" in result["classifiers_with_drift"]
        assert "ep" not in result["classifiers_with_drift"]

    def test_both_drift_detected(self, mock_workflow):
        """Test evaluation when both classifiers have drift."""
        context = {
            "fp_drift_detected": True,
            "ep_drift_detected": True,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["any_drift_detected"] is True
        assert "fp" in result["classifiers_with_drift"]
        assert "ep" in result["classifiers_with_drift"]
        assert "retrain" in result["recommendation"].lower()


class TestSendDriftAlerts:
    """Tests for drift alert step."""

    def test_skipped_on_dry_run(self, mock_workflow):
        """Test alerts skipped in dry run mode."""
        context = {"dry_run": True, "any_drift_detected": True}

        result = send_drift_alerts(mock_workflow, context)

        assert result["alerts_skipped"] is True
        assert result["reason"] == "dry_run"

    def test_no_alerts_when_no_drift(self, mock_workflow):
        """Test no alerts sent when no drift detected."""
        context = {"dry_run": False, "any_drift_detected": False}

        result = send_drift_alerts(mock_workflow, context)

        assert result["alerts_sent"] is False
        assert result["reason"] == "no_drift"

    def test_alerts_sent_on_drift(self, mock_workflow):
        """Test alerts sent when drift detected."""
        with patch(
            "src.agent.workflows.drift_monitoring.send_drift_notification"
        ) as mock_notify:
            mock_notify.return_value = {"console": True}

            context = {
                "dry_run": False,
                "any_drift_detected": True,
                "fp_drift_detected": True,
                "fp_drift_score": 0.15,
                "fp_threshold": 0.1,
                "ep_drift_detected": False,
            }

            result = send_drift_alerts(mock_workflow, context)

            assert result["alerts_sent"] is True
            assert result["alert_count"] == 1
            mock_notify.assert_called_once()


class TestGenerateDriftReport:
    """Tests for drift report generation."""

    def test_report_generation(self, mock_workflow, capsys):
        """Test report is generated correctly."""
        context = {
            "drift_days": 7,
            "fp_drift_check_success": True,
            "fp_drift_detected": False,
            "fp_drift_score": 0.05,
            "fp_threshold": 0.1,
            "fp_healthy": True,
            "ep_drift_check_success": True,
            "ep_drift_detected": True,
            "ep_drift_score": 0.15,
            "ep_threshold": 0.1,
            "any_drift_detected": True,
            "classifiers_with_drift": ["ep"],
            "recommendation": "Retrain EP classifier",
        }

        result = generate_drift_report(mock_workflow, context)

        assert "report" in result
        report = result["report"]
        assert report["drift_days"] == 7
        assert report["fp_classifier"]["healthy"] is True
        assert report["ep_classifier"]["drift_detected"] is True
        assert report["overall"]["any_drift_detected"] is True

        # Check console output
        captured = capsys.readouterr()
        assert "DRIFT MONITORING SUMMARY" in captured.out
        assert "Healthy" in captured.out


class TestDriftMonitoringWorkflow:
    """Tests for DriftMonitoringWorkflow class."""

    def test_workflow_registered(self):
        """Test workflow is registered."""
        from src.agent.workflows import WorkflowRegistry

        assert "drift_monitoring" in WorkflowRegistry.list()

    def test_workflow_has_expected_steps(self):
        """Test workflow has expected steps."""
        step_names = [s.name for s in DriftMonitoringWorkflow.steps]

        assert "check_fp_drift" in step_names
        assert "check_ep_drift" in step_names
        assert "evaluate_drift_results" in step_names
        assert "send_drift_alerts" in step_names
        assert "generate_drift_report" in step_names

    def test_alert_step_skipped_on_dry_run(self):
        """Test alert step is skipped in dry-run mode."""
        alert_step = next(
            s for s in DriftMonitoringWorkflow.steps if s.name == "send_drift_alerts"
        )
        assert alert_step.skip_on_dry_run is True

    def test_dry_run_workflow(self, state_manager):
        """Test workflow runs in dry-run mode."""
        with patch(
            "src.agent.workflows.drift_monitoring.run_monitor_drift"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                success=True,
                exit_code=0,
                duration_seconds=5.0,
                stdout="Drift Detected: NO\nHealthy",
                stderr="",
            )

            workflow = DriftMonitoringWorkflow(
                state_manager=state_manager, dry_run=True
            )
            result = workflow.run()

            assert result.status == WorkflowStatus.COMPLETED
            # Alert step should be skipped
            assert result.steps["send_drift_alerts"].result.get("skipped") is True
