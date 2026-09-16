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
workflow's last run, and no later than one audit period after that, because the
auditor can only answer at the moments cron runs it. The bound is
``interval + grace + audit period``: the first two are in ``AgentSettings``,
the third in ``scripts/setup_cron.sh``. All three terms matter, so tuning one
alone will not give the latency you asked for.

Why no test-archive filter
--------------------------
This module builds no heuristic to tell a test-written record from a real one:
a misclassifying filter is worse than none. Exclusion is by explicit allowlist
instead -- ``latest_run_per_workflow`` is called with the configured cadence
set, so a record under a name that is not a real workflow is never read. That is
a fact about the name and cannot misfire.

What an allowlist cannot exclude is a record written under a *real* workflow
name by something other than that workflow. Test-harness isolation is #124.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..archive import consecutive_failures, latest_run_per_workflow
from ..config import agent_settings
from ..health import HealthVerdict, summarize
from ..notifications import (
    delivered,
    send_check_failure_notification,
    send_consecutive_failure_notification,
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

#: This workflow's own name, bound once. The escalation dedupe below reads this
#: workflow's *previous* archived run, so the name is used as data and not only
#: as an identity -- `RunAuditWorkflow.name` is set from this constant so the
#: two cannot drift.
AUDIT_WORKFLOW_NAME = "run_audit"

#: Suffix of the per-workflow high-water mark the escalator carries forward. One
#: spelling, read by `_prior_escalation_marks` and written by
#: `send_failure_escalations`.
ESCALATED_SUFFIX = "_escalated_run_id"

#: How far into the future a run's start time may sit before the auditor treats
#: it as a broken record rather than as clock noise. A time daemon stepping the
#: clock backwards at boot can date a run seconds ahead of now, and paging an
#: operator about a healthy workflow is the false alarm this epic is the wrong
#: place to introduce. Deliberately NOT `audit_grace_hours`: grace is measured
#: in hours and a run genuinely hours into the future is the case the branch
#: exists to catch, so reusing it would swallow exactly what it should report.
CLOCK_SKEW_TOLERANCE_HOURS = 300 / 3600  # five minutes


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

        if age < -CLOCK_SKEW_TOLERANCE_HOURS:
            # A newest run dated well into the future -- a clock that moved, or
            # a record that did not come from a run. Left alone it is a negative
            # age, which passes every freshness test there is: this workflow
            # would be reported healthy forever, and the auditor would go
            # permanently quiet about it. That is the exact silence this epic
            # exists to remove, so the case gets its own branch rather than
            # falling through to the comparison below. A few seconds of skew is
            # tolerated above rather than paged about.
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


def _prior_escalation_marks(history_dir: Path | None = None) -> dict[str, str]:
    """The high-water marks this auditor's previous run carried.

    Reads *this* workflow's own newest archived record. The current run is not in
    the archive yet -- `_archive_workflow` is written by
    `complete_workflow`/`fail_workflow` at finalize (D009) -- so from inside a
    step this cleanly yields the previous run.

    Reading its own prior context is a different operation from the liveness
    self-audit skip, which stands: the auditor still does not check its own
    cadence, because a process cannot observe its own absence.

    **Absence yields NO marks, and the caller escalates on no mark.** A missing,
    unreadable or first-ever record all land here. Returning marks the caller
    would read as "already escalated" is the one failure this must not have:
    suppressing because you cannot tell whether you already alerted is silent
    success reproduced inside the escalator (D014).
    """
    try:
        previous = latest_run_per_workflow(
            history_dir, workflows={AUDIT_WORKFLOW_NAME}
        )
    except OSError as exc:
        logger.warning(
            f"could not read this auditor's own prior run ({exc}); "
            "escalating without dedupe rather than suppressing"
        )
        return {}

    run = previous.get(AUDIT_WORKFLOW_NAME)
    if run is None:
        return {}

    marks: dict[str, str] = {}
    for key, value in run.context.items():
        if key.endswith(ESCALATED_SUFFIX) and isinstance(value, str):
            marks[key[: -len(ESCALATED_SUFFIX)]] = value
    return marks


def check_failure_streaks(
    workflow: Workflow, context: dict[str, Any]
) -> dict[str, Any] | StepFailure:
    """Count each watched workflow's trailing failures and decide what is due.

    The compute half of #75. Split from the alerting half for the reason
    `check_liveness`/`send_stale_alerts` are split: the alert step is
    `skip_on_dry_run`, and a dry run should still compute and report what it
    found.

    **It writes no `{name}_verdict` key** (D014). `check_liveness` writes one per
    subject and step results merge into a flat context, so a second writer would
    clobber the liveness verdict by loop order. Its own namespace is
    `{name}_consecutive_failures` / `{name}_newest_run_id` / `{name}` +
    `ESCALATED_SUFFIX`.

    **A detection is never `unknown`.** "Failed N times" is real and actionable,
    the same shape D012 ruled `degraded` for a stalled job, so it alerts and lets
    this run complete. The only genuine can't-tell case is an unreadable archive,
    and `check_liveness` already converts that into `unknown` over the same
    subject set and already fails the run at the terminal gate -- so this step
    guards on its flags and delegates rather than contributing a second opinion.
    """
    settings = agent_settings

    # A config `check_liveness` already rejected does not get a second,
    # differently-shaped opinion here, exactly as `send_stale_alerts` short-
    # circuits on it.
    config_error = context.get("audit_config_error")
    if config_error:
        return {
            "failure_streaks_checked": False,
            "failure_streak_skip_reason": "audit_misconfigured",
        }

    if not context.get("archive_readable", True):
        return {
            "failure_streaks_checked": False,
            "failure_streak_skip_reason": "archive_unreadable",
        }

    # Exactly the liveness allowlist, and the same variable. Not
    # `intervals | skipped` -- `model_training` has no cadence and `run_audit`
    # is this workflow -- and never every name on disk, which would read the
    # synthetic records a test harness can leave behind (#124).
    watched = set(settings.audit_expected_interval_hours)
    threshold = settings.consecutive_failure_threshold

    try:
        streaks = consecutive_failures(settings.history_dir, workflows=watched)
        latest = latest_run_per_workflow(settings.history_dir, workflows=watched)
    except OSError as exc:
        # Not caught into an empty result: "no consecutive failures anywhere" is
        # exactly the silent all-clear this issue exists to remove. Reaching
        # here means the archive became unreadable between `check_liveness` and
        # now, so the run fails rather than reporting nothing to escalate.
        return StepFailure(
            error=f"could not read the run archive for failure streaks: {exc}",
            context={
                "failure_streaks_checked": False,
                "failure_streak_skip_reason": "archive_unreadable",
            },
        )

    prior = _prior_escalation_marks(settings.history_dir)

    due: list[str] = []
    result: dict[str, Any] = {
        "failure_streaks_checked": True,
        "consecutive_failure_threshold": threshold,
    }

    for name in sorted(watched):
        # A workflow with no archived run is ABSENT from `streaks`, and 0 is the
        # right reading here: never-ran is a liveness finding (`degraded`), not a
        # failure streak, and `check_liveness` already owns its alert. What must
        # not happen is the absence being re-read as healthy -- it is not, it is
        # reported by the other step.
        streak = streaks.get(name, 0)
        run = latest.get(name)
        newest = run.run_id if run is not None else None

        result[f"{name}_consecutive_failures"] = streak
        if newest is not None:
            result[f"{name}_newest_run_id"] = newest

        mark = prior.get(name)
        if mark is not None:
            # CARRY THE MARK FORWARD UNCONDITIONALLY. Writing it only when this
            # pass escalates makes the ledger forget the moment a pass
            # suppresses: pass 1 escalates and records, pass 2 suppresses and
            # records nothing, pass 3 reads an empty ledger and re-escalates.
            # That bug is invisible to a two-pass test (D014).
            result[f"{name}{ESCALATED_SUFFIX}"] = mark

        if streak >= threshold and newest is not None and newest != mark:
            due.append(name)

    result["failure_escalations_due"] = due
    return result


def send_failure_escalations(
    workflow: Workflow, context: dict[str, Any]
) -> dict[str, Any]:
    """Escalate every workflow whose failure streak has crossed the threshold.

    Routed through `notifications` for the reason `send_stale_alerts` is, and
    `delivered()` is what keeps the record honest: the notifier falls back to the
    console when no channel is enabled, and counting that as delivery is how an
    alert reaches nobody while the archive records that it was sent.

    Advances the high-water mark only for what it actually alerted on. Marks for
    everything else were already carried forward by the compute step, so a
    suppressing pass does not erase them.
    """
    due = context.get("failure_escalations_due") or []
    threshold = context.get("consecutive_failure_threshold")

    sent: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    for name in due:
        streak = context.get(f"{name}_consecutive_failures")
        channels = send_consecutive_failure_notification(
            workflow_name=name,
            consecutive_failures=streak,
            threshold=threshold,
            details={"last_run": context.get(f"{name}_last_run")},
        )
        sent.append(
            {
                "subject": name,
                "consecutive_failures": streak,
                "channels": channels,
                "delivered": delivered(channels),
            }
        )
        newest = context.get(f"{name}_newest_run_id")
        if newest:
            result[f"{name}{ESCALATED_SUFFIX}"] = newest

    result["failure_escalations_sent"] = sent
    result["failure_escalations_delivered"] = any(
        alert["delivered"] for alert in sent
    )
    return result


def _repeatedly_failing(context: dict[str, Any], subjects: list[str]) -> list[str]:
    """Audited subjects whose trailing failure streak has reached the threshold.

    Read back off the compute step's own namespace rather than recomputed, so the
    report cannot disagree with what was escalated.
    """
    threshold = context.get("consecutive_failure_threshold")
    if not isinstance(threshold, int):
        return []
    failing = []
    for name in subjects:
        streak = context.get(f"{name}_consecutive_failures")
        if isinstance(streak, int) and streak >= threshold:
            failing.append(name)
    return sorted(failing)


def generate_audit_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Summarize the audit, reading verdicts rather than inferring health.

    **The liveness verdict alone is not the whole answer, and saying so is the
    point.** A workflow can be perfectly live -- running on schedule, archiving
    every run -- and still fail every one of those runs, which is precisely what
    the streak check detects. Rendering "no action needed - every audited
    workflow is running" beside an escalation that just fired would be this
    epic's own defect committed by the report, so the streak result is carried
    into every branch rather than only the healthy one (#75).
    """
    subjects = context.get(AUDITED_KEY, [])
    summary = summarize({name: context.get(f"{name}_verdict") for name in subjects})
    failing = _repeatedly_failing(context, subjects)

    if context.get("failure_streaks_checked") is False:
        streak_clause = (
            " The failure-streak check did not run ("
            + str(context.get("failure_streak_skip_reason") or "reason not recorded")
            + "), so nothing is watching for a workflow that runs and fails."
        )
    elif failing:
        streak_clause = (
            " Running but failing every run: "
            + ", ".join(failing)
            + " - live, so the liveness check above is clean, and their work is "
            "still not being done."
        )
    else:
        streak_clause = ""

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
        recommendation = (
            "No action needed - every audited workflow is running."
            if not (failing or streak_clause)
            else "Every audited workflow is running on schedule."
        )
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

    return {
        "audit_summary": dict(summary),
        "recommendation": recommendation + streak_clause,
        "workflows_failing_repeatedly": failing,
    }


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
    3. Count each watched workflow's trailing failures (#75)
    4. Escalate any workflow whose streak has crossed the threshold (#75)
    5. Generate a summary report
    6. Fail the workflow if any audited subject produced no verdict

    Liveness and the failure streak are two checks over one data source serving
    one purpose -- surfacing scheduled work that is silently not being done. One
    asks *did it run*, the other *did it keep succeeding* (D014).
    """

    name = AUDIT_WORKFLOW_NAME
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
            name="check_failure_streaks",
            description="Count each watched workflow's trailing failed runs",
            handler=check_failure_streaks,
            # Deliberately NOT skip_on_dry_run: a dry run should still compute
            # and report what it found, exactly as check_liveness does.
        ),
        StepDefinition(
            name="send_failure_escalations",
            description="Escalate workflows failing N runs in a row",
            handler=send_failure_escalations,
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
