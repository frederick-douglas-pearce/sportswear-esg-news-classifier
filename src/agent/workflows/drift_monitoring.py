"""Drift monitoring workflow."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..config import agent_settings
from ..notifications import send_drift_notification
from ..runner import run_monitor_drift
from .base import StepDefinition, Workflow, WorkflowRegistry

logger = logging.getLogger(__name__)


def check_fp_drift(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run drift detection for FP classifier."""
    logger.info("Running FP classifier drift detection")

    result = run_monitor_drift(
        classifier="fp",
        days=context.get("drift_days", 7),
        from_db=True,
        html_report=context.get("generate_html", False),
        alert=False,  # We handle alerts in the notification step
    )

    drift_result = {
        "fp_drift_check_success": result.success,
        "fp_drift_exit_code": result.exit_code,
        "fp_drift_duration": result.duration_seconds,
    }

    # Parse drift detection output
    if result.stdout:
        drift_result.update(_parse_drift_output(result.stdout, "fp"))

    if not result.success:
        drift_result["fp_drift_error"] = result.stderr[:500]
        logger.error(f"FP drift check failed: {result.stderr[:200]}")

    return drift_result


def check_ep_drift(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run drift detection for EP classifier."""
    logger.info("Running EP classifier drift detection")

    result = run_monitor_drift(
        classifier="ep",
        days=context.get("drift_days", 7),
        from_db=True,
        html_report=context.get("generate_html", False),
        alert=False,
    )

    drift_result = {
        "ep_drift_check_success": result.success,
        "ep_drift_exit_code": result.exit_code,
        "ep_drift_duration": result.duration_seconds,
    }

    if result.stdout:
        drift_result.update(_parse_drift_output(result.stdout, "ep"))

    if not result.success:
        drift_result["ep_drift_error"] = result.stderr[:500]
        logger.error(f"EP drift check failed: {result.stderr[:200]}")

    return drift_result


def _parse_drift_output(output: str, prefix: str) -> dict[str, Any]:
    """Parse drift monitoring output for key metrics."""
    result = {}
    lines = output.strip().split("\n")

    for line in lines:
        line_lower = line.lower()

        if "drift detected:" in line_lower:
            result[f"{prefix}_drift_detected"] = "yes" in line_lower
        elif "drift score:" in line_lower:
            try:
                # Parse "Drift Score: 0.1234 (threshold: 0.1)"
                parts = line.split(":")
                if len(parts) >= 2:
                    score_part = parts[1].strip().split()[0]
                    result[f"{prefix}_drift_score"] = float(score_part)
            except (ValueError, IndexError):
                pass
        elif "threshold:" in line_lower:
            try:
                # Parse threshold from "0.1234 (threshold: 0.1)"
                if "(" in line:
                    threshold_str = line.split("threshold:")[1].strip().rstrip(")")
                    result[f"{prefix}_threshold"] = float(threshold_str)
            except (ValueError, IndexError):
                pass
        elif "action required" in line_lower:
            result[f"{prefix}_action_required"] = True
        elif "healthy" in line_lower:
            result[f"{prefix}_healthy"] = True

    return result


def evaluate_drift_results(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate drift detection results and determine actions needed."""
    fp_drift = context.get("fp_drift_detected", False)
    ep_drift = context.get("ep_drift_detected", False)

    evaluation = {
        "any_drift_detected": fp_drift or ep_drift,
        "classifiers_with_drift": [],
    }

    if fp_drift:
        evaluation["classifiers_with_drift"].append("fp")
        logger.warning("FP classifier drift detected")

    if ep_drift:
        evaluation["classifiers_with_drift"].append("ep")
        logger.warning("EP classifier drift detected")

    if evaluation["any_drift_detected"]:
        evaluation["recommendation"] = "Consider retraining affected classifiers"
    else:
        evaluation["recommendation"] = "No action needed - all classifiers healthy"

    logger.info(f"Drift evaluation: {evaluation['recommendation']}")

    return evaluation


def send_drift_alerts(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send notifications for any detected drift."""
    if context.get("dry_run"):
        logger.info("Dry run - skipping drift alerts")
        return {"alerts_skipped": True, "reason": "dry_run"}

    if not context.get("any_drift_detected"):
        logger.info("No drift detected - no alerts needed")
        return {"alerts_sent": False, "reason": "no_drift"}

    alerts_sent = []

    # Send FP drift alert
    if context.get("fp_drift_detected"):
        result = send_drift_notification(
            classifier_type="fp",
            drift_score=context.get("fp_drift_score", 0),
            threshold=context.get("fp_threshold", 0.1),
            details={
                "action_required": context.get("fp_action_required", False),
                "recommendation": "Retrain FP classifier with recent data",
            },
        )
        alerts_sent.append({"classifier": "fp", "result": result})

    # Send EP drift alert
    if context.get("ep_drift_detected"):
        result = send_drift_notification(
            classifier_type="ep",
            drift_score=context.get("ep_drift_score", 0),
            threshold=context.get("ep_threshold", 0.1),
            details={
                "action_required": context.get("ep_action_required", False),
                "recommendation": "Retrain EP classifier with recent data",
            },
        )
        alerts_sent.append({"classifier": "ep", "result": result})

    return {
        "alerts_sent": True,
        "alert_count": len(alerts_sent),
        "alert_details": alerts_sent,
    }


def generate_drift_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Generate drift monitoring summary report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow.name,
        "drift_days": context.get("drift_days", 7),
    }

    # FP classifier status
    report["fp_classifier"] = {
        "check_success": context.get("fp_drift_check_success", False),
        "drift_detected": context.get("fp_drift_detected", False),
        "drift_score": context.get("fp_drift_score"),
        "threshold": context.get("fp_threshold"),
        "healthy": context.get("fp_healthy", not context.get("fp_drift_detected", False)),
    }

    # EP classifier status
    report["ep_classifier"] = {
        "check_success": context.get("ep_drift_check_success", False),
        "drift_detected": context.get("ep_drift_detected", False),
        "drift_score": context.get("ep_drift_score"),
        "threshold": context.get("ep_threshold"),
        "healthy": context.get("ep_healthy", not context.get("ep_drift_detected", False)),
    }

    # Overall status
    report["overall"] = {
        "any_drift_detected": context.get("any_drift_detected", False),
        "classifiers_with_drift": context.get("classifiers_with_drift", []),
        "recommendation": context.get("recommendation", ""),
    }

    # Log summary
    _log_drift_summary(report)

    return {"report": report}


def _log_drift_summary(report: dict[str, Any]) -> None:
    """Log a human-readable drift summary."""
    print("\n" + "=" * 60)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Analysis Period: Last {report['drift_days']} days")

    print(f"\nFP Classifier:")
    fp = report["fp_classifier"]
    if fp["check_success"]:
        status = "DRIFT DETECTED" if fp["drift_detected"] else "Healthy"
        print(f"  Status: {status}")
        if fp["drift_score"] is not None:
            print(f"  Drift Score: {fp['drift_score']:.4f}")
        if fp["threshold"] is not None:
            print(f"  Threshold: {fp['threshold']:.4f}")
    else:
        print("  Status: Check failed")

    print(f"\nEP Classifier:")
    ep = report["ep_classifier"]
    if ep["check_success"]:
        status = "DRIFT DETECTED" if ep["drift_detected"] else "Healthy"
        print(f"  Status: {status}")
        if ep["drift_score"] is not None:
            print(f"  Drift Score: {ep['drift_score']:.4f}")
        if ep["threshold"] is not None:
            print(f"  Threshold: {ep['threshold']:.4f}")
    else:
        print("  Status: Check failed")

    print(f"\nOverall:")
    overall = report["overall"]
    if overall["any_drift_detected"]:
        print(f"  WARNING: Drift detected in {', '.join(overall['classifiers_with_drift'])}")
    else:
        print("  All classifiers healthy")
    print(f"  Recommendation: {overall['recommendation']}")

    print("=" * 60 + "\n")


@WorkflowRegistry.register
class DriftMonitoringWorkflow(Workflow):
    """Drift monitoring workflow.

    Steps:
    1. Check FP classifier drift
    2. Check EP classifier drift
    3. Evaluate drift results
    4. Send alerts if drift detected
    5. Generate summary report
    """

    name = "drift_monitoring"
    description = "Monitor FP and EP classifiers for data drift"

    steps = [
        StepDefinition(
            name="check_fp_drift",
            description="Run drift detection for FP classifier",
            handler=check_fp_drift,
        ),
        StepDefinition(
            name="check_ep_drift",
            description="Run drift detection for EP classifier",
            handler=check_ep_drift,
        ),
        StepDefinition(
            name="evaluate_drift_results",
            description="Evaluate drift detection results",
            handler=evaluate_drift_results,
        ),
        StepDefinition(
            name="send_drift_alerts",
            description="Send notifications for detected drift",
            handler=send_drift_alerts,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="generate_drift_report",
            description="Generate drift monitoring summary report",
            handler=generate_drift_report,
        ),
    ]
