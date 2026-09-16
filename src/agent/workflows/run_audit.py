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
from ..config import agent_settings, parse_failure_threshold
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

#: Suffix of the per-workflow high-water mark the escalator carries forward.
#: One spelling, three users: read by `_prior_escalation_marks`, **carried
#: forward by `check_failure_streaks`** and advanced by
#: `send_failure_escalations`. The carry-forward is the one easiest to overlook
#: and is the whole of D014.2 -- a pass that suppresses must still write the
#: mark, or the ledger forgets and the next pass re-escalates.
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
    """The high-water marks carried by this auditor's newest READABLE run.

    Reads *this* workflow's own archive. The current run is not in it yet --
    `_archive_workflow` is written by `complete_workflow`/`fail_workflow` at
    finalize (D009) -- so from inside a step this yields a previous run.

    Reading its own prior context is a different operation from the liveness
    self-audit skip, which stands: the auditor still does not check its own
    cadence, because a process cannot observe its own absence.

    **Newest readable, not newest.** `iter_runs` logs and skips a record that
    does not parse, so a corrupt newest record falls back to the one before it.
    That is the safe direction and is deliberate: those marks are at most one
    pass stale, and staleness here resolves toward a duplicate alert for the one
    workflow the corrupt pass had just advanced -- never toward suppression.
    Reading only the single newest file would turn one corrupt record into an
    empty ledger and re-escalate every current streak at once.

    **An empty archive yields no marks, and the caller escalates on no mark.**
    Suppressing because you cannot tell whether you already alerted is silent
    success reproduced inside the escalator.
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
        # `key` is guarded as well as `value`: a record can parse as YAML and
        # still carry a non-string key, and `key.endswith` would raise
        # `AttributeError` -- which is not an `OSError`, so no guard downstream
        # catches it and the whole audit aborts.
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.endswith(ESCALATED_SUFFIX):
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

    **It writes no `{name}_verdict` key** (D014.4). `check_liveness` writes one
    per subject and step results merge into a flat context, so a second writer
    would clobber the liveness verdict by loop order. Its own namespace is
    `{name}_consecutive_failures` / `{name}_newest_run_id` / `{name}` +
    `ESCALATED_SUFFIX`.

    **A detection is never `unknown`.** "Failed N times" is real and actionable,
    the same shape D012 ruled `degraded` for a stalled job, so it alerts and lets
    this run complete. The only can't-tell case is an unreadable archive, and
    `check_liveness` already converts that into `unknown` over the same subject
    set and already fails the run at the terminal gate -- so this step guards on
    its flags and delegates rather than contributing a second opinion.

    **Every exit carries the mark ledger forward.** A pass that returns without
    the marks archives a context without them, and the next pass reads an empty
    ledger and re-escalates every current streak. That is true of the ordinary
    path and equally of each early return below -- including the `StepFailure`,
    whose `context` REPLACES the result rather than merging with one.
    """
    settings = agent_settings

    # Read the ledger BEFORE any exit can skip it, and carry EVERY prior mark --
    # not only the marks of currently-watched workflows. A workflow briefly
    # dropped from the cadence config would otherwise lose its mark on an
    # otherwise-healthy pass and re-escalate when it came back.
    prior = _prior_escalation_marks(settings.history_dir)
    carried: dict[str, Any] = {
        f"{name}{ESCALATED_SUFFIX}": run_id for name, run_id in prior.items()
    }

    # A config `check_liveness` already rejected does not get a second,
    # differently-shaped opinion here, exactly as `send_stale_alerts` short-
    # circuits on it.
    if context.get("audit_config_error"):
        return {
            **carried,
            "failure_streaks_checked": False,
            "failure_streak_skip_reason": "audit_misconfigured",
        }

    if not context.get("archive_readable", True):
        return {
            **carried,
            "failure_streaks_checked": False,
            "failure_streak_skip_reason": "archive_unreadable",
        }

    # Parsed here rather than at construction: a bad value fails this one step
    # loudly, the way `_cadence_config_error` fails `check_liveness`, instead of
    # raising from module scope and stopping every agent workflow (D014.7 was
    # reversed on this at #75's scope ruling).
    threshold, threshold_error = parse_failure_threshold(
        settings.consecutive_failure_threshold_raw
    )
    if threshold_error is not None:
        logger.error(f"consecutive-failure threshold is unusable: {threshold_error}")
        return StepFailure(
            error=threshold_error,
            context={
                **carried,
                "failure_streaks_checked": False,
                "failure_streak_skip_reason": "threshold_unusable",
            },
        )

    # Exactly the liveness allowlist, and the same variable. NOT because the
    # excluded workflows lack a cadence -- a failure streak needs no cadence,
    # only a sequence of archived runs, so that reasoning (given at the plan
    # gate) was a category error, corrected at the scope ruling. The real bars:
    # `model_training` PAUSES for notebooks and `run_succeeded` is false for any
    # non-`completed` status, so watching it would fire on every ordinary
    # train-then-pause cycle; and an auditor escalating about its own failing
    # runs, from inside a possibly-failing run, is a feedback loop that needs its
    # own decision rather than a one-line allowlist change. Never every name on
    # disk, which would read the synthetic records a test harness leaves (#124).
    watched = set(settings.audit_expected_interval_hours)

    try:
        streaks = consecutive_failures(settings.history_dir, workflows=watched)
        latest = latest_run_per_workflow(settings.history_dir, workflows=watched)
    except OSError as exc:
        # Not caught into an empty result: "no consecutive failures anywhere" is
        # exactly the silent all-clear this issue exists to remove. This guard
        # covers an archive that cannot be LISTED. It does not cover a DELETED
        # one: `history_dir` calls `mkdir(exist_ok=True)` on read (D013), so a
        # deleted archive is recreated and reads as empty -- that is #125, and
        # until it is fixed liveness's stale alerts are the compensating signal.
        return StepFailure(
            error=f"could not read the run archive for failure streaks: {exc}",
            context={
                **carried,
                "failure_streaks_checked": False,
                "failure_streak_skip_reason": "archive_unreadable",
            },
        )

    # A workflow that has stopped running is already being paged by
    # `send_stale_alerts`. Escalating it again here pages the operator twice for
    # one workflow, and the streak alert's "still running on schedule" would be
    # false. Its streak is still recorded; only the second page is suppressed.
    stale = set(context.get("stale_workflows") or [])

    due: list[str] = []
    suppressed_stale: list[str] = []
    result: dict[str, Any] = {
        **carried,
        "failure_streaks_checked": True,
        "consecutive_failure_threshold": threshold,
    }

    for name in sorted(watched):
        # A workflow with no archived run is ABSENT from `streaks`, and 0 is the
        # right reading: never-ran is a liveness finding (`degraded`), which
        # `check_liveness` owns. The absence must not be re-read as healthy --
        # it is not, it is reported by the other step.
        streak = streaks.get(name, 0)
        run = latest.get(name)
        newest = run.run_id if run is not None else None

        result[f"{name}_consecutive_failures"] = streak
        if newest is not None:
            result[f"{name}_newest_run_id"] = newest

        if streak < threshold or newest is None or newest == prior.get(name):
            continue
        if name in stale:
            suppressed_stale.append(name)
            continue
        due.append(name)

    result["failure_escalations_due"] = due
    result["failure_escalations_suppressed_stale"] = suppressed_stale
    return result


def send_failure_escalations(
    workflow: Workflow, context: dict[str, Any]
) -> dict[str, Any]:
    """Escalate every workflow whose failure streak has crossed the threshold.

    Routed through `notifications` for the reason `send_stale_alerts` is.

    **The mark advances only when the alert went somewhere, and the two
    no-delivery cases are different.** `NotificationManager.send` returns
    `{"console": True}` when no channel is configured at all, and `delivered()`
    reports that as false. Those are not the same event:

    * **console-only** -- there is nothing to deliver to, so delivery is not
      applicable and the mark advances. Re-escalating to a console nobody reads,
      every pass forever, is a busy-loop rather than a signal.
    * **a configured channel was attempted and failed** -- a dead SMTP host, a
      rejected key, an HTTP 500. Every notifier swallows its exception and
      returns False, so this is silent. The mark does NOT advance, and the next
      pass retries. Burning it here would lose the alert permanently, which is
      this epic's own defect inside the escalator.

    Marks for workflows this pass did not alert on were already carried forward
    by `check_failure_streaks`, so a suppressing pass does not erase them.
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
        reached = delivered(channels)
        console_only = set(channels) == {"console"}
        sent.append(
            {
                "subject": name,
                "consecutive_failures": streak,
                "channels": channels,
                "delivered": reached,
            }
        )

        newest = context.get(f"{name}_newest_run_id")
        if newest and (reached or console_only):
            result[f"{name}{ESCALATED_SUFFIX}"] = newest
        elif not reached:
            logger.error(
                f"escalation for {name} reached no configured channel; leaving "
                "the mark unadvanced so the next audit retries"
            )

    result["failure_escalations_sent"] = sent
    result["failure_escalations_delivered"] = any(
        alert["delivered"] for alert in sent
    )
    return result


def _repeatedly_failing(context: dict[str, Any], subjects: list[str]) -> list[str]:
    """Audited subjects whose trailing failure streak has reached the threshold.

    Read off the compute step's own namespace rather than recomputed from the
    archive, so the streak the report describes is the one the escalation was
    decided on. It deliberately does NOT read `failure_escalations_sent`: the
    report describes the workflow's state, and a pass that suppressed a repeat
    alert has not stopped the workflow from failing.
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

    **The liveness verdict alone is not the whole answer, and neither is the
    prose.** A workflow can be perfectly live -- running on schedule, archiving
    every run -- and fail every one of those runs. Two surfaces have to say so:
    the recommendation a human reads, and `audit_summary`, which is the typed
    shape #77/#78/#79 bind to and the one that lands in the archive. Fixing only
    the sentence would leave the machine-readable half asserting the opposite,
    which is this epic's defect committed by the report.
    """
    subjects = context.get(AUDITED_KEY, [])
    summary = dict(summarize({name: context.get(f"{name}_verdict") for name in subjects}))
    failing = _repeatedly_failing(context, subjects)
    stale = set(context.get("stale_workflows") or [])

    # `is not True` and not `is False`: an ABSENT key means the step did not run
    # and left no record, which must not render as a clean streak check.
    if context.get("failure_streaks_checked") is not True:
        streak_clause = (
            " The failure-streak check did not run ("
            + str(context.get("failure_streak_skip_reason") or "reason not recorded")
            + "), so nothing is watching for a workflow that runs and fails."
        )
    elif failing:
        # Split by liveness: a workflow that is failing AND stale is already
        # named as stalled above, and calling it "live" there would be false.
        live = [name for name in failing if name not in stale]
        parts = []
        if live:
            parts.append(
                "Running but failing every run: "
                + ", ".join(live)
                + " - on schedule, so the liveness check above is clean, and "
                "their work is still not being done."
            )
        both = [name for name in failing if name in stale]
        if both:
            parts.append(
                "Stalled AND failing when they did run: "
                + ", ".join(both)
                + " - the stall is reported above; the failures came first."
            )
        streak_clause = " " + " ".join(parts)
    else:
        streak_clause = ""

    # The typed surface has to agree with the prose. A streak at or past the
    # threshold is a check that ran and found a real problem, so a summary
    # claiming every check passed is false whatever the liveness verdicts say.
    if failing:
        summary["all_checked_healthy"] = False
    summary["failing_repeatedly"] = failing

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
    elif summary["unknown"]:
        recommendation = (
            "The audit could not reach a verdict for: "
            + ", ".join(summary["unknown"])
            + ". These workflows are currently unmonitored."
        )
    else:
        # Every liveness check passed and the streak check is what made the
        # summary non-healthy. Without this branch the `unknown` join above
        # rendered an empty list.
        recommendation = "Every audited workflow is running on schedule."

    return {
        "audit_summary": summary,
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
