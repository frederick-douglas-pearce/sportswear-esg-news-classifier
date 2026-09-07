"""Tests for drift monitoring workflow.

The load-bearing tests here are the ones asserting that a check which produced
no verdict does NOT read as healthy (issue #71). They assert the mechanism
rather than the rendered text: a test checking only that the summary lacks the
word "healthy" also passes for a workflow that crashed before printing at all.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.health import HealthVerdict
from src.agent.state import StateManager, WorkflowStatus
from src.agent.workflows.drift_monitoring import (
    DriftMonitoringWorkflow,
    check_ep_drift,
    check_fp_drift,
    evaluate_drift_results,
    fail_on_unknown_verdict,
    generate_drift_report,
    send_drift_alerts,
)
from src.mlops.exit_codes import (
    EXIT_DRIFT_DETECTED,
    EXIT_INDETERMINATE,
    EXIT_NO_DRIFT,
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


def script_result(
    classifier="fp",
    exit_code=EXIT_NO_DRIFT,
    drift_detected=False,
    drift_score=0.05,
    threshold=0.1,
    indeterminate=False,
    error=None,
    stderr="",
    summary="auto",
):
    """Build a ScriptResult double matching what monitor_drift.py emits.

    `summary="auto"` builds a self-consistent summary; pass None for "the
    script produced no parseable summary", or a dict to inject a broken one.
    """
    if summary == "auto":
        summary = {
            "classifier": classifier,
            "exit_code": exit_code,
            "indeterminate": indeterminate,
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "threshold": threshold,
            "error": error,
        }
    return MagicMock(
        success=exit_code == 0,
        exit_code=exit_code,
        duration_seconds=5.0,
        stdout="",
        stderr=stderr,
        parsed_output=summary,
    )


class TestCheckFpDrift:
    """Tests for FP drift check step — exit code to verdict."""

    def test_healthy(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT)

            result = check_fp_drift(mock_workflow, {"drift_days": 7})

            assert result["fp_verdict"] == HealthVerdict.HEALTHY.value
            assert result["fp_drift_score"] == 0.05
            mock_run.assert_called_once_with(
                classifier="fp", days=7, from_db=True, html_report=False, alert=False
            )

    def test_drift_detected_is_degraded_not_a_failure(self, mock_workflow):
        """Exit 1 is a result. Before #71 it was indistinguishable from a crash."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_DRIFT_DETECTED, drift_detected=True, drift_score=0.15
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.DEGRADED.value
            assert result["fp_drift_score"] == 0.15

    def test_indeterminate_is_unknown_never_healthy(self, mock_workflow):
        """AC5: a failed check must not read as healthy."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE,
                indeterminate=True,
                error="Insufficient data for drift analysis",
                stderr="KeyError: 'novelty_score'",
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value
            assert result["fp_verdict"] != HealthVerdict.HEALTHY.value
            assert result["fp_error"] == "Insufficient data for drift analysis"

    def test_exit_zero_without_summary_is_unknown(self, mock_workflow):
        """A verdict with no evidence behind it is not a verdict.

        The exit code says healthy; nothing corroborates it. Trusting the code
        alone would ship a healthy claim with a null score.
        """
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT, summary=None)

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value
            assert "no valid summary" in result["fp_error"]

    def test_summary_missing_required_field_is_unknown(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT,
                summary={"classifier": "fp", "exit_code": 0, "drift_detected": False},
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_summary_for_wrong_classifier_is_unknown(self, mock_workflow):
        """A summary describing another run is not evidence about this one."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT, classifier="ep"
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_summary_disagreeing_with_exit_code_is_unknown(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            result_double = script_result(exit_code=EXIT_NO_DRIFT)
            result_double.parsed_output["exit_code"] = EXIT_DRIFT_DETECTED

            mock_run.return_value = result_double
            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_unrecognised_exit_code_is_unknown(self, mock_workflow):
        """The runner reports -1 for a timeout or an execution error.

        The summary is deliberately PRESENT and self-consistent, so only the
        unrecognised-exit-code guard can produce the verdict. With
        `summary=None` this test also passes when that guard is deleted, via
        the no-summary branch below it.
        """
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=-1, stderr="killed")

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value
            assert "unrecognised exit code -1" in result["fp_error"]


class TestCheckEpDrift:
    """Tests for the EP gate (AC4)."""

    def test_skipped_by_default_without_running_the_check(self, mock_workflow):
        """The mechanism is that the script is never invoked, not just labelled."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            result = check_ep_drift(mock_workflow, {})

            mock_run.assert_not_called()

        assert result["ep_verdict"] == HealthVerdict.SKIPPED.value
        assert result["ep_skip_reason"]
        assert "on hold" in result["ep_skip_reason"].lower()

    def test_runs_when_enabled(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run, patch(
            "src.agent.workflows.drift_monitoring.agent_settings"
        ) as mock_settings:
            mock_settings.ep_drift_enabled = True
            mock_run.return_value = script_result(classifier="ep")

            result = check_ep_drift(mock_workflow, {"drift_days": 14})

            assert result["ep_verdict"] == HealthVerdict.HEALTHY.value
            mock_run.assert_called_once_with(
                classifier="ep", days=14, from_db=True, html_report=False, alert=False
            )


class TestEvaluateDriftResults:
    """Tests for drift evaluation step."""

    def test_healthy_when_checked_and_clean(self, mock_workflow):
        context = {
            "fp_verdict": HealthVerdict.HEALTHY.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["all_checked_healthy"] is True
        assert result["classifiers_skipped"] == ["ep"]
        assert "No action needed" in result["recommendation"]

    def test_unknown_verdict_is_not_healthy(self, mock_workflow):
        """AC5, at the evaluation step: the 217-times bug."""
        context = {
            "fp_verdict": HealthVerdict.UNKNOWN.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["all_checked_healthy"] is False
        assert result["classifiers_unknown"] == ["fp"]
        assert "UNKNOWN" in result["recommendation"]
        assert "healthy" not in result["recommendation"].lower()

    def test_absent_verdict_is_not_healthy(self, mock_workflow):
        """A context with no verdict keys at all — the pre-#71 shape exactly."""
        result = evaluate_drift_results(mock_workflow, {})

        assert result["all_checked_healthy"] is False
        assert sorted(result["classifiers_unknown"]) == ["ep", "fp"]

    def test_drift_detected(self, mock_workflow):
        context = {
            "fp_verdict": HealthVerdict.DEGRADED.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["any_drift_detected"] is True
        assert result["classifiers_with_drift"] == ["fp"]
        assert result["all_checked_healthy"] is False

    def test_all_skipped_is_not_healthy(self, mock_workflow):
        """'Every non-skipped check passed' is vacuously true here (issue #95)."""
        context = {
            "fp_verdict": HealthVerdict.SKIPPED.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        result = evaluate_drift_results(mock_workflow, context)

        assert result["all_checked_healthy"] is False


class TestSendDriftAlerts:
    """Tests for the alert step."""

    def test_skipped_on_dry_run(self, mock_workflow):
        result = send_drift_alerts(mock_workflow, {"dry_run": True})

        assert result["alerts_skipped"] is True
        assert result["reason"] == "dry_run"

    def test_no_alerts_when_healthy(self, mock_workflow):
        context = {
            "dry_run": False,
            "fp_verdict": HealthVerdict.HEALTHY.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        result = send_drift_alerts(mock_workflow, context)

        assert result["alerts_sent"] is False

    def test_alert_sent_on_drift(self, mock_workflow):
        with patch(
            "src.agent.workflows.drift_monitoring.send_drift_notification"
        ) as mock_notify:
            mock_notify.return_value = {"console": True}
            context = {
                "dry_run": False,
                "fp_verdict": HealthVerdict.DEGRADED.value,
                "fp_drift_score": 0.15,
                "fp_threshold": 0.1,
                "ep_verdict": HealthVerdict.SKIPPED.value,
            }

            result = send_drift_alerts(mock_workflow, context)

            assert result["alert_count"] == 1
            mock_notify.assert_called_once()

    def test_alert_sent_on_failed_check(self, mock_workflow):
        """AC2: an induced check failure produces an alert.

        This is the notification that never fired across 231 days.
        """
        with patch(
            "src.agent.workflows.drift_monitoring.send_check_failure_notification"
        ) as mock_notify:
            mock_notify.return_value = {"console": True}
            context = {
                "dry_run": False,
                "fp_verdict": HealthVerdict.UNKNOWN.value,
                "fp_error": "Insufficient data for drift analysis",
                "fp_drift_exit_code": EXIT_INDETERMINATE,
                "ep_verdict": HealthVerdict.SKIPPED.value,
            }

            result = send_drift_alerts(mock_workflow, context)

            assert result["alert_count"] == 1
            assert result["alert_details"][0]["kind"] == "check_failed"
            kwargs = mock_notify.call_args.kwargs
            assert kwargs["check_name"] == "FP drift"
            assert kwargs["reason"] == "Insufficient data for drift analysis"


class TestGenerateDriftReport:
    """Tests for drift report generation."""

    def test_unknown_verdict_in_report_structure(self, mock_workflow):
        """Asserted on the returned structure, not on captured stdout."""
        context = {
            "drift_days": 7,
            "fp_verdict": HealthVerdict.UNKNOWN.value,
            "fp_error": "KeyError: 'novelty_score'",
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": "EP classifier is on hold",
            "classifiers_unknown": ["fp"],
            "classifiers_skipped": ["ep"],
            "all_checked_healthy": False,
            "recommendation": "Drift status UNKNOWN for FP",
        }

        report = generate_drift_report(mock_workflow, context)["report"]

        assert report["fp_classifier"]["verdict"] == HealthVerdict.UNKNOWN.value
        assert report["ep_classifier"]["verdict"] == HealthVerdict.SKIPPED.value
        assert report["overall"]["all_checked_healthy"] is False

    def test_summary_says_not_monitored_on_unknown(self, mock_workflow, capsys):
        context = {
            "drift_days": 7,
            "fp_verdict": HealthVerdict.UNKNOWN.value,
            "fp_error": "KeyError: 'novelty_score'",
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": "EP classifier is on hold",
            "classifiers_unknown": ["fp"],
            "classifiers_skipped": ["ep"],
            "all_checked_healthy": False,
            "recommendation": "Drift status UNKNOWN for FP",
        }

        generate_drift_report(mock_workflow, context)

        out = capsys.readouterr().out
        assert "UNKNOWN" in out
        assert "NOT being monitored" in out
        # The real string production prints on the healthy path. Asserting the
        # pre-#71 wording ("All classifiers healthy") would be inert: it exists
        # nowhere in the codebase, so it can never appear.
        assert "All checked classifiers healthy" not in out

    def test_healthy_report(self, mock_workflow, capsys):
        context = {
            "drift_days": 7,
            "fp_verdict": HealthVerdict.HEALTHY.value,
            "fp_drift_detected": False,
            "fp_drift_score": 0.05,
            "fp_threshold": 0.1,
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": "EP classifier is on hold",
            "all_checked_healthy": True,
            "classifiers_skipped": ["ep"],
            "recommendation": "No action needed - FP healthy",
        }

        report = generate_drift_report(mock_workflow, context)["report"]

        assert report["fp_classifier"]["verdict"] == HealthVerdict.HEALTHY.value
        out = capsys.readouterr().out
        assert "All checked classifiers healthy" in out


class TestFailOnUnknownVerdict:
    """Tests for the terminal step (AC2's non-zero workflow status)."""

    def test_passes_when_all_verdicts_explicit(self, mock_workflow):
        context = {
            "fp_verdict": HealthVerdict.HEALTHY.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        assert fail_on_unknown_verdict(mock_workflow, context)["verdicts_confirmed"]

    def test_passes_on_degraded(self, mock_workflow):
        """Drift detected is a reported result; the alert is its channel."""
        context = {
            "fp_verdict": HealthVerdict.DEGRADED.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        assert fail_on_unknown_verdict(mock_workflow, context)["verdicts_confirmed"]

    def test_raises_on_unknown(self, mock_workflow):
        context = {
            "fp_verdict": HealthVerdict.UNKNOWN.value,
            "fp_error": "KeyError: 'novelty_score'",
            "ep_verdict": HealthVerdict.SKIPPED.value,
        }

        with pytest.raises(RuntimeError, match="no verdict"):
            fail_on_unknown_verdict(mock_workflow, context)

    def test_raises_on_absent_verdict(self, mock_workflow):
        """A handler that returned no verdict key must not slip past the gate.

        Testing `== "unknown"` alone would pass this context, since the value
        is None — re-creating the bug at the one step placed to catch it.
        """
        with pytest.raises(RuntimeError, match="no verdict"):
            fail_on_unknown_verdict(mock_workflow, {})

    def test_raises_on_unrecognised_verdict(self, mock_workflow):
        context = {"fp_verdict": "probably_fine", "ep_verdict": HealthVerdict.SKIPPED.value}

        with pytest.raises(RuntimeError, match="no verdict"):
            fail_on_unknown_verdict(mock_workflow, context)


class TestDriftMonitoringWorkflow:
    """End-to-end tests over the real workflow."""

    def test_workflow_registered(self):
        from src.agent.workflows import WorkflowRegistry

        assert "drift_monitoring" in WorkflowRegistry.list()

    def test_workflow_has_expected_steps_in_order(self):
        """The terminal step must run AFTER the alert and report steps."""
        step_names = [s.name for s in DriftMonitoringWorkflow.steps]

        assert step_names == [
            "check_fp_drift",
            "check_ep_drift",
            "evaluate_drift_results",
            "send_drift_alerts",
            "generate_drift_report",
            "fail_on_unknown_verdict",
        ]

    def test_alert_step_skipped_on_dry_run(self):
        alert_step = next(
            s for s in DriftMonitoringWorkflow.steps if s.name == "send_drift_alerts"
        )
        assert alert_step.skip_on_dry_run is True

    def test_terminal_step_not_skipped_on_dry_run(self):
        """A dry run should still surface that a check could not tell us anything."""
        step = next(
            s for s in DriftMonitoringWorkflow.steps if s.name == "fail_on_unknown_verdict"
        )
        assert step.skip_on_dry_run is False

    def test_failed_check_fails_the_workflow_after_alerting(self, state_manager):
        """AC2, end to end: non-healthy summary AND an alert AND non-zero status.

        The report-step assertion is what proves the terminal step is positioned
        after the reporting steps rather than merely raising early.
        """
        with patch(
            "src.agent.workflows.drift_monitoring.run_monitor_drift"
        ) as mock_run, patch(
            "src.agent.workflows.drift_monitoring.send_check_failure_notification"
        ) as mock_notify:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE,
                indeterminate=True,
                error="Insufficient data for drift analysis",
            )
            mock_notify.return_value = {"console": True}

            workflow = DriftMonitoringWorkflow(state_manager=state_manager, dry_run=False)
            result = workflow.run()

            assert result.status == WorkflowStatus.FAILED

            mock_notify.assert_called_once()

            assert (
                result.steps["generate_drift_report"].status == WorkflowStatus.COMPLETED
            )
            assert (
                result.steps["send_drift_alerts"].status == WorkflowStatus.COMPLETED
            )

            report = result.steps["generate_drift_report"].result["report"]
            assert report["fp_classifier"]["verdict"] == HealthVerdict.UNKNOWN.value
            assert report["overall"]["all_checked_healthy"] is False

    def test_healthy_check_completes_and_sends_no_alert(self, state_manager):
        """Control: a green run proves the pipeline can go red rather than always being red."""
        with patch(
            "src.agent.workflows.drift_monitoring.run_monitor_drift"
        ) as mock_run, patch(
            "src.agent.workflows.drift_monitoring.send_check_failure_notification"
        ) as mock_notify, patch(
            "src.agent.workflows.drift_monitoring.send_drift_notification"
        ) as mock_drift_notify:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT)

            workflow = DriftMonitoringWorkflow(state_manager=state_manager, dry_run=False)
            result = workflow.run()

            assert result.status == WorkflowStatus.COMPLETED
            mock_notify.assert_not_called()
            mock_drift_notify.assert_not_called()

            report = result.steps["generate_drift_report"].result["report"]
            assert report["overall"]["all_checked_healthy"] is True

    def test_dry_run_workflow(self, state_manager):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT)

            workflow = DriftMonitoringWorkflow(state_manager=state_manager, dry_run=True)
            result = workflow.run()

            assert result.status == WorkflowStatus.COMPLETED
            assert result.steps["send_drift_alerts"].result.get("skipped") is True


class TestDiagnosisReachesTheContext:
    """The traceback must land in the context, not only in the log.

    The run archive and the alert body are built from the context. On the
    dominant failure path — the script raised, so it returned before printing a
    summary — a bare fallback records the literal string "no verdict produced",
    which is an alert that names no cause.
    """

    def test_stderr_becomes_the_error_when_no_summary_exists(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE,
                summary=None,
                stderr="Traceback (most recent call last):\n  KeyError: 'novelty_score'",
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value
            assert "novelty_score" in result["fp_error"]
            assert result["fp_error"] != "no verdict produced"

    def test_structured_error_is_preferred_over_stderr(self, mock_workflow):
        """When the script DID produce a summary, its own reason is better."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE,
                indeterminate=True,
                error="Insufficient data for drift analysis",
                stderr="some noisy warning",
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_error"] == "Insufficient data for drift analysis"

    def test_silent_failure_says_so_rather_than_claiming_a_reason(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE, summary=None, stderr=""
            )

            result = check_fp_drift(mock_workflow, {})

            assert "wrote nothing to stderr" in result["fp_error"]

    def test_tail_not_head_of_stderr(self, mock_workflow):
        """A traceback's diagnosis is at the END (issue #81, same runner).

        Swapping `tail(...)` back to `result.stderr[:200]` fails this test.
        """
        stderr = ("noise line\n" * 500) + "KeyError: 'the_real_cause'"
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_INDETERMINATE, summary=None, stderr=stderr
            )

            result = check_fp_drift(mock_workflow, {})

            assert "the_real_cause" in result["fp_error"]


class TestSummaryMustCarryEvidence:
    """`_validate_summary` checks values, not just that keys are present."""

    def test_summary_marked_indeterminate_is_not_healthy(self, mock_workflow):
        """A summary that says "I could not tell" outranks its exit code."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT, indeterminate=True
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_null_drift_score_is_not_healthy(self, mock_workflow):
        """A health claim with no measurement under it is not a health claim."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT, drift_score=None
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_non_numeric_drift_score_is_not_healthy(self, mock_workflow):
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT, drift_score="not-a-number"
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_summary_contradicting_the_exit_code_is_not_healthy(self, mock_workflow):
        """exit 0 but drift_detected=true — two statements of one fact, disagreeing."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_NO_DRIFT, drift_detected=True
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.UNKNOWN.value

    def test_wellformed_healthy_summary_still_passes(self, mock_workflow):
        """Control: the new checks have not made every summary unreadable."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT)

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.HEALTHY.value

    def test_wellformed_degraded_summary_still_passes(self, mock_workflow):
        """Control for the drift_detected/exit-code agreement check."""
        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(
                exit_code=EXIT_DRIFT_DETECTED, drift_detected=True, drift_score=0.4
            )

            result = check_fp_drift(mock_workflow, {})

            assert result["fp_verdict"] == HealthVerdict.DEGRADED.value


class TestSummaryReadsTheVerdictNotTheMetric:
    """A degraded classifier must never print as Healthy."""

    def test_degraded_without_metrics_does_not_print_healthy(self, mock_workflow, capsys):
        """`_classifier_report` builds `drift_detected` with a bare `.get()`.

        Deriving the printed status from that metric rather than from the
        verdict printed "Status: Healthy" for a DEGRADED classifier.
        """
        context = {
            "drift_days": 7,
            "fp_verdict": HealthVerdict.DEGRADED.value,
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": "on hold",
        }

        generate_drift_report(mock_workflow, context)

        out = capsys.readouterr().out
        assert "DRIFT DETECTED" in out
        assert "Status: Healthy" not in out


class TestBothVerdictsAlert:
    def test_drift_and_failed_check_both_alert(self, mock_workflow):
        with patch(
            "src.agent.workflows.drift_monitoring.send_drift_notification"
        ) as mock_drift, patch(
            "src.agent.workflows.drift_monitoring.send_check_failure_notification"
        ) as mock_fail:
            mock_drift.return_value = {"console": True}
            mock_fail.return_value = {"console": True}
            context = {
                "dry_run": False,
                "fp_verdict": HealthVerdict.DEGRADED.value,
                "fp_drift_score": 0.4,
                "fp_threshold": 0.15,
                "ep_verdict": HealthVerdict.UNKNOWN.value,
                "ep_error": "Insufficient data",
            }

            result = send_drift_alerts(mock_workflow, context)

            assert result["alert_count"] == 2
            mock_drift.assert_called_once()
            mock_fail.assert_called_once()


class TestDegradedRunEndToEnd:
    """Drift detected is a RESULT: the workflow completes and alerts."""

    def test_degraded_completes_with_a_drift_alert(self, state_manager):
        with patch(
            "src.agent.workflows.drift_monitoring.run_monitor_drift"
        ) as mock_run, patch(
            "src.agent.workflows.drift_monitoring.send_drift_notification"
        ) as mock_drift, patch(
            "src.agent.workflows.drift_monitoring.send_check_failure_notification"
        ) as mock_fail:
            mock_run.return_value = script_result(
                exit_code=EXIT_DRIFT_DETECTED, drift_detected=True, drift_score=0.4
            )
            mock_drift.return_value = {"console": True}

            workflow = DriftMonitoringWorkflow(state_manager=state_manager, dry_run=False)
            result = workflow.run()

            assert result.status == WorkflowStatus.COMPLETED
            mock_drift.assert_called_once()
            mock_fail.assert_not_called()
            report = result.steps["generate_drift_report"].result["report"]
            assert report["fp_classifier"]["verdict"] == HealthVerdict.DEGRADED.value


class TestVerdictsSerialiseSafely:
    """Context is dumped to the run archive RAW — only `.value` is YAML-safe.

    `HealthVerdict` inherits from `str`, but `yaml.dump` still renders an enum
    MEMBER as a `!!python/object/apply:` tag, and `yaml.safe_dump` refuses it.
    `WorkflowStatus` survives only because `to_dict()` calls `.value`
    explicitly; `context` and `StepState.result` get no such treatment. So a
    member stored in context would reach the archive as a Python tag, and the
    next `safe_load` would raise into the bare except in `StateManager._load`
    that resets all workflow state.
    """

    def test_a_member_is_not_yaml_safe(self):
        """Pins the reason the discipline below is required."""
        import yaml

        with pytest.raises(yaml.YAMLError):
            yaml.safe_dump({"verdict": HealthVerdict.HEALTHY})

    def test_check_steps_store_plain_strings(self, mock_workflow):
        import yaml

        with patch("src.agent.workflows.drift_monitoring.run_monitor_drift") as mock_run:
            mock_run.return_value = script_result(exit_code=EXIT_NO_DRIFT)
            fp = check_fp_drift(mock_workflow, {})
        ep = check_ep_drift(mock_workflow, {})

        for produced in (fp, ep):
            for value in produced.values():
                assert not isinstance(value, HealthVerdict), (
                    "store verdict.value, not the enum member — see HealthVerdict"
                )
            # The real check: it survives the archive writer's safe round trip.
            assert yaml.safe_load(yaml.safe_dump(produced)) == produced

    def test_the_report_structure_is_yaml_safe(self, mock_workflow):
        import yaml

        context = {
            "drift_days": 7,
            "fp_verdict": HealthVerdict.HEALTHY.value,
            "fp_drift_detected": False,
            "fp_drift_score": 0.05,
            "fp_threshold": 0.1,
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": "on hold",
            "all_checked_healthy": True,
        }

        result = generate_drift_report(mock_workflow, context)

        assert yaml.safe_load(yaml.safe_dump(result)) == result
