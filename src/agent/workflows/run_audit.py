"""Run-archive audit: detect a scheduled workflow that has stopped running.

This is the Class-B half of #76, and the only liveness net in the system.
#73 makes a failed step fail its run, #74 makes an unresolved check fail its
run, and #75 counts consecutive failures -- but all three read a run that
*happened*. A job that never runs writes no archive for any of them to read, so
its silence looks exactly like success. This workflow is the external observer
that compares expected against actual.

What it does not cover, stated because "liveness check" invites the wrong
assumption
------------------------------------------------------------------------------
This auditor is itself a scheduled job on the same host, driven by the same
cron, as the workflows it audits. Two shapes follow, and only the first is
covered while it is happening:

* **One workflow stops while its siblings keep running.** The auditor runs on
  its own schedule, sees that workflow's newest archive is older than it should
  be, and alerts. This is the case the audit exists for.
* **The host is off, asleep, or cron itself is broken.** Nothing runs, including
  this auditor, so no alert is raised at the time. When the host returns, the
  auditor runs and reports the gap -- so the failure is detected on recovery
  rather than during.

Nothing running on the host can close that second gap: a process cannot observe
its own absence. Closing it needs an off-host dead-man's-switch, which is a
second piece of infrastructure and out of scope here. **The same reasoning
applies to this workflow auditing itself** -- it is in its own skip set for the
liveness check, because a stalled auditor cannot report that it has stalled.

Why no test-archive filter
--------------------------
The archive contains runs written by the test suite under production workflow
names, from before the fixtures were isolated. This module deliberately builds
no heuristic to tell them apart: a misclassifying filter is worse than none, and
the liveness question is asked of the *newest* run per workflow, where a
surplus of old records cannot change the answer. The residual risk is a test run
on the same host writing a fresh production-named archive while a real job is
dead, which would mask that job. The structural fix is session-scoped archive
isolation in the test harness, which is filed separately -- it changes every
agent test module and does not belong to this issue.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ..archive import latest_run_per_workflow
from ..config import agent_settings
from ..health import HealthVerdict, summarize
from ..notifications import (
    delivered,
    send_check_failure_notification,
    send_stale_workflow_notification,
)
from .base import (
    StepDefinition,
    StepFailure,
    Workflow,
    WorkflowRegistry,
    fail_on_unresolved_verdicts,
)

logger = logging.getLogger(__name__)

#: Context key holding every subject this run audited, so that the terminal
#: gate can build itself over the same set the check step actually used.
AUDITED_KEY = "audited_workflows"


def _hours_since(moment: datetime, now: datetime) -> float:
    return (now - moment).total_seconds() / 3600.0


def check_liveness(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Compare each expected workflow's newest run against its cadence.

    The verdict mapping is the load-bearing decision here (D012), and it splits
    two absences that look alike:

    * **an expected workflow has no run within its interval, including one that
      has never run at all** -> ``degraded``. The check ran and found a real,
      actionable problem, which is what ``DEGRADED`` means. Mapping this to
      ``unknown`` would trip the terminal gate and fail the auditor's own run at
      the exact moment it *succeeded* at detecting a dead job -- making a correct
      detection indistinguishable from the auditor malfunctioning, and leaving
      the operator with a failed status instead of an alert naming the workflow.
    * **the archive could not be read at all** -> ``unknown`` for every subject.
      Here the auditor genuinely cannot tell, and the gate should fail the run.
    """
    settings = agent_settings
    intervals = dict(settings.audit_expected_interval_hours)
    skipped = dict(settings.audit_skipped_workflows)
    now = datetime.now(timezone.utc)

    subjects = sorted(set(intervals) | set(skipped))
    result: dict[str, Any] = {
        AUDITED_KEY: subjects,
        "audit_checked_at": now.isoformat(),
        "stale_workflows": [],
    }

    try:
        latest = latest_run_per_workflow(
            settings.history_dir, workflows=set(intervals)
        )
    except OSError as exc:
        # The auditor cannot tell anything. Every expected subject is unknown,
        # and the terminal gate fails the run -- which is correct: an auditor
        # that cannot read the archive must not report the jobs healthy.
        logger.error(f"could not read the run archive: {exc}")
        result["archive_readable"] = False
        result["audit_error"] = str(exc)
        for name in intervals:
            result[f"{name}_verdict"] = HealthVerdict.UNKNOWN.value
            result[f"{name}_error"] = f"could not read the run archive: {exc}"
        for name, reason in skipped.items():
            result[f"{name}_verdict"] = HealthVerdict.SKIPPED.value
            result[f"{name}_skip_reason"] = reason
        return result

    result["archive_readable"] = True

    stale: list[str] = []
    for name, interval_hours in sorted(intervals.items()):
        threshold = interval_hours + settings.audit_grace_hours
        run = latest.get(name)

        if run is None:
            result[f"{name}_verdict"] = HealthVerdict.DEGRADED.value
            result[f"{name}_error"] = (
                "no archived run at all; this workflow is expected every "
                f"{interval_hours:g}h and has never reported"
            )
            result[f"{name}_last_run"] = None
            stale.append(name)
            continue

        age = _hours_since(run.run_at, now)
        result[f"{name}_last_run"] = run.run_at.isoformat()
        result[f"{name}_age_hours"] = round(age, 2)

        if age > threshold:
            result[f"{name}_verdict"] = HealthVerdict.DEGRADED.value
            result[f"{name}_error"] = (
                f"last run was {age:.1f}h ago; expected every "
                f"{interval_hours:g}h (stale past {threshold:g}h)"
            )
            stale.append(name)
        else:
            result[f"{name}_verdict"] = HealthVerdict.HEALTHY.value

    for name, reason in sorted(skipped.items()):
        result[f"{name}_verdict"] = HealthVerdict.SKIPPED.value
        result[f"{name}_skip_reason"] = reason

    result["stale_workflows"] = stale
    return result


def send_stale_alerts(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Alert on every stalled workflow, and on an unreadable archive.

    Routed through ``notifications`` rather than ``mlops.alerts``: this is an
    operator-facing alert and ``notifications`` is what reaches the operator's
    configured channel. ``delivered()`` is what keeps ``alerts_sent`` honest --
    the notifier falls back to the console when no channel is enabled, and
    counting that as delivery is how an alert reaches nobody while the archive
    records that it was sent.
    """
    sent: list[dict[str, Any]] = []

    if not context.get("archive_readable", True):
        result = send_check_failure_notification(
            check_name="run archive audit",
            reason=context.get("audit_error") or "the run archive could not be read",
        )
        sent.append(
            {
                "subject": "run archive audit",
                "channels": result,
                "delivered": delivered(result),
            }
        )
        return {"alerts_sent": sent, "alerts_delivered": any(a["delivered"] for a in sent)}

    for name in context.get("stale_workflows", []):
        result = send_stale_workflow_notification(
            workflow_name=name,
            reason=context.get(f"{name}_error") or "no recent run",
            details={"last_run": context.get(f"{name}_last_run")},
        )
        sent.append(
            {
                "subject": name,
                "channels": result,
                "delivered": delivered(result),
            }
        )

    return {
        "alerts_sent": sent,
        "alerts_delivered": any(alert["delivered"] for alert in sent),
        "reason": "nothing_to_report" if not sent else "stale_workflows_reported",
    }


def generate_audit_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Summarize the audit, reading verdicts rather than inferring health."""
    subjects = context.get(AUDITED_KEY, [])
    summary = summarize({name: context.get(f"{name}_verdict") for name in subjects})

    if summary["all_checked_healthy"]:
        recommendation = "No action needed - every audited workflow is running."
    elif summary["degraded"]:
        recommendation = (
            "Stalled: " + ", ".join(summary["degraded"]) + ". Check cron and the "
            "job's own logs; its work is not being done."
        )
    else:
        recommendation = (
            "The audit could not reach a verdict for: "
            + ", ".join(summary["unknown"])
            + ". These workflows are currently unmonitored."
        )

    return {"audit_summary": dict(summary), "recommendation": recommendation}


def fail_on_unknown_verdict(
    workflow: Workflow, context: dict[str, Any]
) -> dict[str, Any] | StepFailure:
    """Fail the run if any audited subject produced no verdict.

    Built at call time rather than at import, because the subjects are the
    configured workflows and a test that rebinds the cadence config must audit
    the set it configured. The shared gate from ``base`` does the deciding --
    this only supplies it the subject list the check step actually used (D010).
    """
    subjects = context.get(AUDITED_KEY) or []
    if not subjects:
        # A cadence config naming nothing is an auditor that checks nothing and
        # reports success: this epic's defect, in the instrument built for it.
        return StepFailure(
            error=(
                "the run audit was configured with no workflows to audit, so it "
                "checked nothing; an empty audit is not a healthy one"
            ),
            context={AUDITED_KEY: []},
        )
    gate = fail_on_unresolved_verdicts(sorted(subjects))
    return gate(workflow, context)


@WorkflowRegistry.register
class RunAuditWorkflow(Workflow):
    """Audit the run archive for workflows that have stopped running.

    Steps:
    1. Compare each expected workflow's newest run against its cadence
    2. Alert on anything stalled, or on an unreadable archive
    3. Generate a summary report
    4. Fail the workflow if any audited subject produced no verdict
    """

    name = "run_audit"
    description = "Detect scheduled workflows that have stopped producing runs"

    steps = [
        StepDefinition(
            name="check_liveness",
            description="Compare each workflow's newest run against its cadence",
            handler=check_liveness,
        ),
        StepDefinition(
            name="send_stale_alerts",
            description="Notify about stalled workflows and unreadable archives",
            handler=send_stale_alerts,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="generate_audit_report",
            description="Generate the run audit summary report",
            handler=generate_audit_report,
        ),
        StepDefinition(
            name="fail_on_unknown_verdict",
            description="Fail the workflow if any audited subject has no verdict",
            handler=fail_on_unknown_verdict,
            # Deliberately NOT skip_on_dry_run, matching drift_monitoring: a dry
            # run should still surface that the audit could not tell anything.
        ),
    ]
