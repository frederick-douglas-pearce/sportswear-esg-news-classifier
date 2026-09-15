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

How late an alert can be
------------------------
A stall is reported no earlier than ``interval + audit_grace_hours`` after the
workflow's last run, and no later than one audit period after that: the auditor
can only answer the question at the moments cron runs it. The bound is
``interval + grace + audit period``, re-derivable from ``AgentSettings`` and
``scripts/setup_cron.sh`` rather than restated as a figure here.

Which half of that to tune is not obvious and is worth stating. Grace covers
start-time jitter only, so it wants to be small; it is the **audit period** that
sets the resolution, and shrinking grace under a once-daily audit buys nothing
because detection still lands on the next tick. That is why the auditor is
scheduled several times a day rather than alongside the jobs it watches.

Why no test-archive filter
--------------------------
The archive can hold runs written by the test suite, because
``AgentSettings.history_dir`` resolves to the real archive unless a test rebinds
it and only some agent test modules do. This module builds no heuristic to tell
records from real ones -- a misclassifying filter is worse than none. Exclusion
is by explicit allowlist instead: ``latest_run_per_workflow`` is called with the
configured cadence set, so a record under a name that is not a real workflow is
never read. That is a fact about the name and cannot misfire.

What it leaves open is a test writing a *production*-named archive while that
job is actually dead, which would mask it. Nothing here prevents that; what
prevents it today is that the test modules which write into the real archive
use synthetic names, which is a convention rather than a mechanism. Making the
isolation structural -- one session-scoped fixture in ``tests/conftest.py`` --
is #124.
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


def _cadence_config_error(
    intervals: dict[str, float], skipped: dict[str, str]
) -> str | None:
    """Reject a cadence configuration that cannot produce an honest audit.

    Both shapes rejected here are this epic's defect relocated into the
    auditor's own configuration, and both are silent without this check -- the
    run archives ``completed`` having established nothing about anything.
    """
    overlap = sorted(set(intervals) & set(skipped))
    if overlap:
        # Paged as expected and reported as skipped in the same run: the audit
        # alerts on the workflow while its recorded verdict says it was never
        # looked at. Which of the two dicts wins is an artifact of loop order,
        # so there is no correct precedence to pick -- the config is the bug.
        return (
            "these workflows are configured as both audited and skipped, so the "
            "audit's own verdict for them is ambiguous: " + ", ".join(overlap)
        )
    if not intervals:
        return (
            "no workflow has an expected interval, so this audit would check "
            "nothing and report success; an empty audit is not a healthy one"
        )
    return None


def check_liveness(
    workflow: Workflow, context: dict[str, Any]
) -> dict[str, Any] | StepFailure:
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

    config_error = _cadence_config_error(intervals, skipped)
    if config_error is not None:
        logger.error(f"run audit cadence config is unusable: {config_error}")
        return StepFailure(
            error=config_error,
            context={AUDITED_KEY: [], "audit_config_error": config_error},
        )

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

        if age < 0:
            # A newest run dated in the future -- a clock that moved, or a
            # record that did not come from a run. Left alone it is a negative
            # age, which passes every freshness test there is: this workflow
            # would be reported healthy forever, and the auditor would go
            # permanently quiet about it. That is the exact silence this epic
            # exists to remove, so the case gets its own branch rather than
            # falling through to the comparison below.
            #
            # DEGRADED and not UNKNOWN, per D012: the archive was read fine, so
            # this is not the can't-tell case. It is a real, actionable problem
            # attached to a named subject, which means it belongs in an alert
            # naming that subject rather than in a failed auditor run.
            result[f"{name}_verdict"] = HealthVerdict.DEGRADED.value
            result[f"{name}_error"] = (
                "the newest archived run is dated in the future "
                f"({run.run_at.isoformat()}); its freshness cannot be judged, "
                "and left unreported it would never be called stale again"
            )
            stale.append(name)
            continue

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

    # Reached because `StepFailure` does not halt the step loop (see `base`), so
    # this step still runs after `check_liveness` rejected the config. A
    # misconfigured auditor is worth paging about: nothing is being watched, and
    # the FAILED status alone is the kind of signal #71 found nobody reads.
    config_error = context.get("audit_config_error")
    if config_error:
        result = send_check_failure_notification(
            check_name="run archive audit",
            reason=str(config_error),
        )
        sent.append(
            {
                "subject": "run archive audit",
                "channels": result,
                "delivered": delivered(result),
            }
        )
        return {
            "alerts_sent": sent,
            "alerts_delivered": any(alert["delivered"] for alert in sent),
            "reason": "audit_misconfigured",
        }

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

    if not subjects:
        # No subjects means `check_liveness` rejected the cadence config; the
        # summary is already non-healthy (`summarize` refuses vacuous truth),
        # but the recommendation has to say what to do rather than render an
        # empty list of workflows that could not be reached.
        recommendation = (
            "The audit established nothing: it ran with no workflow to check. "
            "Fix the cadence configuration in src/agent/config.py - until then "
            "no workflow has a liveness detector."
        )
    elif summary["all_checked_healthy"]:
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
        # Defence in depth. `check_liveness` already refuses to run against a
        # cadence config that names nothing, so reaching here means the check
        # step did not write the key at all -- but an auditor that checks
        # nothing and reports success is this epic's defect inside the
        # instrument built for it, and it gets two chances to be caught.
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
    2. Alert on anything stalled, an unreadable archive, or an unusable config
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
            description="Notify about stalls, unreadable archives, bad config",
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
