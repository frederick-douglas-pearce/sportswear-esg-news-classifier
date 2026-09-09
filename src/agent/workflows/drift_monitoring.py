"""Drift monitoring workflow.

Every read in this module obeys one rule, because breaking it is what made
issue #71 possible: **a missing value never resolves to healthy.** Before the
fix, `evaluate_drift_results` read `context.get("fp_drift_detected", False)`
and `generate_drift_report` read
`context.get("fp_healthy", not context.get("fp_drift_detected", False))`. A
check that never ran set neither key, so the second expression evaluated
`not False` -> True -> "all classifiers healthy". That ran 219 times.

So the check steps produce an explicit `HealthVerdict`, the reporting steps
read it rather than inferring one, and the terminal gate refuses to let the
workflow finish green when any verdict is missing or `unknown`. That gate is no
longer written here: #74 moved it into `base.fail_on_unresolved_verdicts`, and
the aggregation into `health.summarize`, so every workflow adopting the
vocabulary shares one implementation instead of re-deriving it (D010).
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ..config import agent_settings
from ..health import HealthVerdict, summarize, verdict_of
from ..notifications import send_check_failure_notification, send_drift_notification
from ..runner import ScriptResult, run_monitor_drift, tail
from .base import (
    StepDefinition,
    Workflow,
    WorkflowRegistry,
    fail_on_unresolved_verdicts,
)
from src.mlops.exit_codes import (
    EXIT_DRIFT_DETECTED,
    EXIT_INDETERMINATE,
    EXIT_NO_DRIFT,
)

logger = logging.getLogger(__name__)

# Fields the machine-readable summary from scripts/monitor_drift.py must carry.
# A summary missing any of them is not trusted, because the alternative is
# reading a health verdict off a partial object -- which is how absence became
# health in the first place.
REQUIRED_SUMMARY_FIELDS = (
    "classifier",
    "exit_code",
    "indeterminate",
    "drift_detected",
    "drift_score",
    "threshold",
)

# The scripts/monitor_drift.py exit-code contract, mapped to verdicts. Imported
# rather than written as literals so the two halves cannot drift apart: this
# module already pulls src.mlops in transitively (via ..runner), so there is no
# import cost to avoid by hardcoding them.
_VERDICT_BY_EXIT_CODE = {
    EXIT_NO_DRIFT: HealthVerdict.HEALTHY,
    EXIT_DRIFT_DETECTED: HealthVerdict.DEGRADED,
    EXIT_INDETERMINATE: HealthVerdict.UNKNOWN,
}


def _validate_summary(result: ScriptResult, classifier: str) -> dict[str, Any] | None:
    """Return the script's summary if it is present and self-consistent.

    Returns None when the summary is absent, malformed, incomplete, or
    describes a different run than the one we just made. The caller treats
    None as `unknown` rather than falling back to the exit code alone: an exit
    code with no accompanying evidence is a verdict nobody can check.
    """
    summary = result.parsed_output
    if not isinstance(summary, dict):
        logger.error(
            f"{classifier} drift check produced no parseable summary "
            f"(exit code {result.exit_code})"
        )
        return None

    missing = [f for f in REQUIRED_SUMMARY_FIELDS if f not in summary]
    if missing:
        logger.error(
            f"{classifier} drift summary is missing required fields: "
            f"{', '.join(missing)}"
        )
        return None

    if summary["classifier"] != classifier:
        logger.error(
            f"drift summary is for '{summary['classifier']}' but this check ran "
            f"'{classifier}' -- refusing to read it"
        )
        return None

    if summary["exit_code"] != result.exit_code:
        logger.error(
            f"{classifier} drift summary reports exit code "
            f"{summary['exit_code']} but the process exited {result.exit_code} "
            f"-- refusing to read it"
        )
        return None

    # `indeterminate` must AGREE with the exit code. Checking only that the key
    # is present would let a summary saying "I could not tell" be read as
    # healthy whenever it arrived with exit 0 -- the field named after the
    # invariant collected and then discarded. Rejecting it outright would be
    # wrong in the other direction: on exit 2 the flag is *supposed* to be
    # true, and that summary carries the reason the alert needs.
    expected_indeterminate = result.exit_code == EXIT_INDETERMINATE
    if summary["indeterminate"] is not expected_indeterminate:
        logger.error(
            f"{classifier} drift summary says indeterminate="
            f"{summary['indeterminate']!r} but exit code {result.exit_code} says "
            f"{expected_indeterminate} -- refusing to read it"
        )
        return None

    # A verdict needs a number behind it. Presence is not enough: a summary
    # whose drift_score is null, or a string, is a health claim with no
    # measurement under it.
    for field in ("drift_score", "threshold"):
        if not isinstance(summary[field], (int, float)) or isinstance(
            summary[field], bool
        ):
            logger.error(
                f"{classifier} drift summary has a non-numeric {field} "
                f"({summary[field]!r}) -- refusing to read it"
            )
            return None

    if not isinstance(summary["drift_detected"], bool):
        logger.error(
            f"{classifier} drift summary has a non-boolean drift_detected "
            f"({summary['drift_detected']!r}) -- refusing to read it"
        )
        return None

    # The two statements of the same fact must agree.
    expected_detected = result.exit_code == EXIT_DRIFT_DETECTED
    if summary["drift_detected"] != expected_detected:
        logger.error(
            f"{classifier} drift summary says drift_detected="
            f"{summary['drift_detected']} but exit code {result.exit_code} says "
            f"{expected_detected} -- refusing to read it"
        )
        return None

    return summary


def _run_drift_check(classifier: str, context: dict[str, Any]) -> dict[str, Any]:
    """Run one classifier's drift check and turn the outcome into a verdict.

    The verdict comes from the script's exit code (src/mlops/exit_codes.py),
    never from scraping its human-readable report. `_parse_drift_output`, which
    did the scraping, keyed "healthy" off the presence of that word in stdout
    and so reported healthy for output it had never seen.
    """
    logger.info(f"Running {classifier.upper()} classifier drift detection")

    result = run_monitor_drift(
        classifier=classifier,
        days=context.get("drift_days", 7),
        from_db=True,
        html_report=context.get("generate_html", False),
        alert=False,  # We handle alerts in the notification step
    )

    out: dict[str, Any] = {
        f"{classifier}_drift_exit_code": result.exit_code,
        f"{classifier}_drift_duration": result.duration_seconds,
    }

    verdict = _VERDICT_BY_EXIT_CODE.get(result.exit_code)
    if verdict is None:
        # An exit code outside the contract -- a crash, a signal, a timeout
        # (the runner reports -1 for those). Unknown, never healthy.
        logger.error(
            f"{classifier.upper()} drift check exited {result.exit_code}, which "
            f"is outside the exit-code contract"
        )
        out[f"{classifier}_verdict"] = HealthVerdict.UNKNOWN.value
        out[f"{classifier}_error"] = (
            f"unrecognised exit code {result.exit_code}: {tail(result.stderr)}"
        )
        return out

    summary = _validate_summary(result, classifier)
    if summary is None and verdict is not HealthVerdict.UNKNOWN:
        # The process claimed a verdict but produced no evidence for it. Do not
        # take its word: a healthy claim with no score behind it is exactly the
        # shape this workflow exists to stop reporting.
        out[f"{classifier}_verdict"] = HealthVerdict.UNKNOWN.value
        out[f"{classifier}_error"] = (
            f"exited {result.exit_code} but produced no valid summary; "
            f"stderr: {tail(result.stderr)}"
        )
        return out

    out[f"{classifier}_verdict"] = verdict.value

    if verdict is HealthVerdict.UNKNOWN:
        # Keep the diagnosis IN THE CONTEXT, not only in the log. This is the
        # dominant failure path -- the script raised and returned before
        # printing a summary -- so `(summary or {}).get("error")` is None and a
        # bare fallback would record the literal string "no verdict produced".
        # That string is what would land in the run archive #75/#76 read and in
        # the alert body, i.e. an alert that names no cause, which is most of
        # what made #71 invisible across 223 failed runs.
        #
        # tail, not head: a traceback's diagnosis is at the END of the stream
        # (issue #81, same runner).
        reason = (summary or {}).get("error") or tail(result.stderr).strip() or (
            "no verdict produced, and the command wrote nothing to stderr"
        )
        out[f"{classifier}_error"] = reason
        logger.error(
            f"{classifier.upper()} drift check produced no verdict: {reason}"
        )
        return out

    # Healthy or degraded: the metrics are required, not optional. Reading them
    # with .get(default) is how a failed check's absent score became 0.0 and
    # then "no drift".
    out[f"{classifier}_drift_detected"] = summary["drift_detected"]
    out[f"{classifier}_drift_score"] = summary["drift_score"]
    out[f"{classifier}_threshold"] = summary["threshold"]

    if verdict is HealthVerdict.DEGRADED:
        # States the score; does NOT claim it exceeds the threshold. Core drift
        # is a COUNT test (`core_drifted > 0`), not a threshold test, so a
        # detection can be correct while the score sits below the threshold --
        # and this line used to read "score 0.0 exceeds 0.15" on brand-only
        # drift, an alert contradicted by its own number.
        logger.warning(
            f"{classifier.upper()} classifier drift detected: "
            f"score {summary['drift_score']} (threshold {summary['threshold']})"
        )
    else:
        logger.info(
            f"{classifier.upper()} classifier healthy: "
            f"drift score {summary['drift_score']}"
        )

    return out


def check_fp_drift(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run drift detection for FP classifier."""
    return _run_drift_check("fp", context)


def check_ep_drift(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run drift detection for EP classifier, unless it is gated off.

    EP is on hold and `classifier_predictions` has never held an `ep` row, so
    the check found no data to compare and reported Healthy on every run. It is
    now skipped explicitly, with the reason recorded -- `skipped` is a verdict,
    not a silent absence, and it never counts toward "all classifiers healthy".
    """
    if not agent_settings.ep_drift_enabled:
        reason = agent_settings.ep_drift_skip_reason
        logger.info(f"EP drift check skipped: {reason}")
        return {
            "ep_verdict": HealthVerdict.SKIPPED.value,
            "ep_skip_reason": reason,
        }

    return _run_drift_check("ep", context)


def _drift_verdict(context: dict[str, Any], classifier: str) -> HealthVerdict:
    """Read one classifier's verdict off the drift key convention.

    A thin binding of `health.verdict_of` to this workflow's `<c>_verdict` keys.
    The coerce-absent-to-UNKNOWN rule it relies on lives in `health`, shared with
    every other workflow, rather than being re-derived here (D010.1).
    """
    return verdict_of(context, f"{classifier}_verdict")


def evaluate_drift_results(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate drift detection results and determine actions needed."""
    verdicts = {c: _drift_verdict(context, c) for c in ("fp", "ep")}

    summary = summarize(verdicts)
    drifted = summary["degraded"]
    unknown = summary["unknown"]
    checked = summary["checked"]

    evaluation: dict[str, Any] = {
        "any_drift_detected": bool(drifted),
        "classifiers_with_drift": drifted,
        "classifiers_unknown": unknown,
        "classifiers_skipped": summary["skipped"],
        # "At least one check ran and every check that ran passed" -- not
        # "no failures found", which is vacuously true when nothing ran. That
        # rule now lives in `health.summarize`, shared, so a workflow adopting
        # the vocabulary cannot re-introduce the vacuous form (D010.7, #95).
        "all_checked_healthy": summary["all_checked_healthy"],
    }

    for classifier in drifted:
        logger.warning(f"{classifier.upper()} classifier drift detected")
    for classifier in unknown:
        logger.error(
            f"{classifier.upper()} drift check produced no verdict - "
            f"that classifier is currently unmonitored"
        )

    if unknown:
        evaluation["recommendation"] = (
            f"Drift status UNKNOWN for {', '.join(c.upper() for c in unknown)} - "
            f"the check did not complete, so these classifiers are unmonitored"
        )
    elif drifted:
        evaluation["recommendation"] = "Consider retraining affected classifiers"
    elif checked:
        evaluation["recommendation"] = (
            f"No action needed - {', '.join(c.upper() for c in checked)} healthy"
        )
    else:
        evaluation["recommendation"] = (
            "No classifiers were checked - nothing is being monitored"
        )

    logger.info(f"Drift evaluation: {evaluation['recommendation']}")

    return evaluation


def _delivered(result: dict[str, bool]) -> bool:
    """Did this alert reach a real channel?

    `NotificationManager.send()` returns `{"console": True}` when NO channel is
    enabled -- which is the documented default (`AGENT_EMAIL_ENABLED=false` and
    no `ALERT_WEBHOOK_URL`). Counting that as delivery made `alerts_sent: True`
    for an alert that reached nobody but the agent log, which is precisely
    where #71 says nobody was looking. "console" is a fallback, not a channel.
    """
    return any(value for channel, value in result.items() if channel != "console")


def send_drift_alerts(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send notifications for detected drift and for checks that produced none."""
    if context.get("dry_run"):
        logger.info("Dry run - skipping drift alerts")
        return {"alerts_skipped": True, "reason": "dry_run"}

    alerts_sent = []

    for classifier in ("fp", "ep"):
        verdict = _drift_verdict(context, classifier)

        if verdict is HealthVerdict.DEGRADED:
            result = send_drift_notification(
                classifier_type=classifier,
                drift_score=context[f"{classifier}_drift_score"],
                threshold=context[f"{classifier}_threshold"],
                details={
                    "recommendation": (
                        f"Retrain {classifier.upper()} classifier with recent data"
                    ),
                },
            )
            alerts_sent.append(
                {
                    "classifier": classifier,
                    "kind": "drift",
                    "result": result,
                    "delivered": _delivered(result),
                }
            )

        elif verdict is HealthVerdict.UNKNOWN:
            # The alert that never fired across 223 failed runs.
            result = send_check_failure_notification(
                check_name=f"{classifier.upper()} drift",
                reason=context.get(f"{classifier}_error") or "no verdict produced",
                details={
                    "exit_code": context.get(f"{classifier}_drift_exit_code"),
                    "days": context.get("drift_days", 7),
                },
            )
            alerts_sent.append(
                {
                    "classifier": classifier,
                    "kind": "check_failed",
                    "result": result,
                    "delivered": _delivered(result),
                }
            )

    if not alerts_sent:
        logger.info("No drift and no failed checks - no alerts needed")
        return {"alerts_sent": False, "reason": "nothing_to_report"}

    # `alerts_sent` means DELIVERED, not attempted.
    #
    # Both notification helpers return `dict[str, bool]` -- one entry per
    # channel -- and no channel need have accepted it: an HTTP error, Resend
    # rejecting the key, or (the default configuration) no channel enabled at
    # all, which returns `{"console": True}`. Any of those left this step
    # reporting `alerts_sent: True`. That is #72's class in the alerting path itself: the
    # step whose whole job is to raise the alarm recording success for an alarm
    # nobody received. Not escalated to a workflow failure -- an undelivered
    # alert about drift should not erase the drift finding -- but it is stated,
    # and the archive #75/#76 will read now carries the difference.
    undelivered = [a for a in alerts_sent if not a["delivered"]]
    for alert in undelivered:
        logger.error(
            f"{alert['classifier'].upper()} {alert['kind']} alert reached NO "
            f"channel (results: {alert['result']})"
        )

    return {
        "alerts_sent": not undelivered,
        "alerts_attempted": len(alerts_sent),
        "alerts_delivered": len(alerts_sent) - len(undelivered),
        "alerts_undelivered": [
            f"{a['classifier']}:{a['kind']}" for a in undelivered
        ],
        "alert_count": len(alerts_sent),
        "alert_details": alerts_sent,
    }


def _classifier_report(context: dict[str, Any], classifier: str) -> dict[str, Any]:
    """Build one classifier's section of the report."""
    verdict = _drift_verdict(context, classifier)
    return {
        "verdict": verdict.value,
        "drift_detected": context.get(f"{classifier}_drift_detected"),
        "drift_score": context.get(f"{classifier}_drift_score"),
        "threshold": context.get(f"{classifier}_threshold"),
        "error": context.get(f"{classifier}_error"),
        "skip_reason": context.get(f"{classifier}_skip_reason"),
    }


def generate_drift_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Generate drift monitoring summary report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow.name,
        "drift_days": context.get("drift_days", 7),
        "fp_classifier": _classifier_report(context, "fp"),
        "ep_classifier": _classifier_report(context, "ep"),
        "overall": {
            "all_checked_healthy": context.get("all_checked_healthy", False),
            "any_drift_detected": context.get("any_drift_detected", False),
            "classifiers_with_drift": context.get("classifiers_with_drift", []),
            "classifiers_unknown": context.get("classifiers_unknown", []),
            "classifiers_skipped": context.get("classifiers_skipped", []),
            "recommendation": context.get("recommendation", ""),
        },
    }

    _log_drift_summary(report)

    return {"report": report}


def _log_classifier_line(label: str, section: dict[str, Any]) -> None:
    """Print one classifier's status block."""
    verdict = section["verdict"]
    print(f"\n{label}:")

    if verdict == HealthVerdict.SKIPPED.value:
        print(f"  Status: SKIPPED - {section['skip_reason'] or 'no reason recorded'}")
        return

    if verdict == HealthVerdict.UNKNOWN.value:
        print(
            f"  Status: UNKNOWN - check did not complete "
            f"({section['error'] or 'no reason recorded'})"
        )
        print("  This classifier is NOT being monitored.")
        return

    # Read the VERDICT, not `drift_detected`. Deriving the healthy/degraded
    # split from the metric is a second source for a fact the verdict already
    # states, and `_classifier_report` builds that metric with a bare `.get()`
    # -- so a degraded classifier whose metric keys were absent printed
    # "Status: Healthy". That is this module's own rule broken inside it.
    if verdict == HealthVerdict.DEGRADED.value:
        print("  Status: DRIFT DETECTED")
    elif verdict == HealthVerdict.HEALTHY.value:
        print("  Status: Healthy")
    else:
        print(f"  Status: {verdict}")
    if section["drift_score"] is not None:
        print(f"  Drift Score: {section['drift_score']:.4f}")
    if section["threshold"] is not None:
        print(f"  Threshold: {section['threshold']:.4f}")


def _log_drift_summary(report: dict[str, Any]) -> None:
    """Log a human-readable drift summary."""
    print("\n" + "=" * 60)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Analysis Period: Last {report['drift_days']} days")

    _log_classifier_line("FP Classifier", report["fp_classifier"])
    _log_classifier_line("EP Classifier", report["ep_classifier"])

    overall = report["overall"]
    print("\nOverall:")
    if overall["classifiers_unknown"]:
        names = ", ".join(c.upper() for c in overall["classifiers_unknown"])
        print(f"  UNKNOWN: {names} could not be checked - not monitored")
    if overall["any_drift_detected"]:
        names = ", ".join(c.upper() for c in overall["classifiers_with_drift"])
        print(f"  WARNING: Drift detected in {names}")
    if overall["all_checked_healthy"]:
        print("  All checked classifiers healthy")
    if overall["classifiers_skipped"]:
        names = ", ".join(c.upper() for c in overall["classifiers_skipped"])
        print(f"  Skipped (not checked): {names}")
    print(f"  Recommendation: {overall['recommendation']}")

    print("=" * 60 + "\n")


def _unmonitored_message(classifiers: list[str], reasons: dict[str, str]) -> str:
    """Drift's wording for the shared terminal gate.

    Kept verbatim from the hand-rolled gate this replaced, so the text archived
    on a failed run -- which #75/#76 will read -- does not move (D010.3).
    """
    names = ", ".join(c.upper() for c in classifiers)
    detail = "; ".join(f"{c}: {reasons[c]}" for c in classifiers)
    return (
        f"Drift check produced no verdict for {names} - these classifiers are "
        f"unmonitored and this run must not be recorded as successful "
        f"({detail})"
    )


# The terminal gate. Built from the shared factory rather than hand-rolled here:
# #74 moved the escalation path into `base.fail_on_unresolved_verdicts`, which is
# exactly the retirement D008 asked for when it called the local version a bridge.
#
# The module-level NAME and the registered step name below are both deliberately
# unchanged. Renaming the step is the one edit that would move an archive key
# #75/#76 may match on, and it must not be coupled to this refactor (D010.3).
fail_on_unknown_verdict = fail_on_unresolved_verdicts(
    ("fp", "ep"),
    describe=_unmonitored_message,
)


@WorkflowRegistry.register
class DriftMonitoringWorkflow(Workflow):
    """Drift monitoring workflow.

    Steps:
    1. Check FP classifier drift
    2. Check EP classifier drift
    3. Evaluate drift results
    4. Send alerts for drift and for failed checks
    5. Generate summary report
    6. Fail the workflow if any check produced no verdict
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
            description="Send notifications for drift and failed checks",
            handler=send_drift_alerts,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="generate_drift_report",
            description="Generate drift monitoring summary report",
            handler=generate_drift_report,
        ),
        StepDefinition(
            name="fail_on_unknown_verdict",
            description="Fail the workflow if any check produced no verdict",
            handler=fail_on_unknown_verdict,
            # Deliberately NOT skip_on_dry_run: a dry run should still surface
            # that a check could not tell us anything.
        ),
    ]
