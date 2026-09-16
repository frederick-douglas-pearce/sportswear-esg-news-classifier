"""Tests for the consecutive-failure escalator (#75).

Split from `test_agent_archive.py` only for size; it shares that file's
discipline. Every test writes into a `tmp_path` archive bound through the
`history` fixture, never the real one — this epic exists to reason about a
directory that historical test runs polluted under production workflow names
(#124), so a test that wrote into it would manufacture the artifact under
discussion.

**Timestamps here are relative to real `now`, not to a fixed constant.** The
escalator's behaviour depends on the liveness verdict: a workflow that failed
twice *and then stopped running* is deliberately not escalated a second time,
because `send_stale_alerts` has already paged for it. A fixed past date makes
every archive stale and silently exercises only that suppression path.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agent.config import AgentSettings, agent_settings, parse_failure_threshold
from src.agent.health import HealthVerdict
from src.agent.notifications import send_consecutive_failure_notification
from src.agent import workflows as _agent_workflows  # noqa: F401
from src.agent.archive import consecutive_failures, latest_run_per_workflow
from src.agent.state import StateManager
from src.agent.workflows import run_audit as run_audit_module
from src.agent.workflows.base import StepFailure
from src.agent.workflows.run_audit import (
    ESCALATED_SUFFIX,
    RunAuditWorkflow,
    check_failure_streaks,
    check_liveness,
    fail_on_unknown_verdict,
    generate_audit_report,
)
from tests.test_agent_archive import history, write_archive  # noqa: F401

WATCHED = "daily_labeling"


@pytest.fixture
def audited(history):  # noqa: F811
    """Audit exactly one expected workflow and one skipped one."""
    with (
        patch.object(
            agent_settings, "audit_expected_interval_hours", {WATCHED: 24.0}
        ),
        patch.object(agent_settings, "audit_grace_hours", 6.0),
        patch.object(
            agent_settings, "audit_skipped_workflows", {"model_training": "not on cron"}
        ),
    ):
        yield history


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _failed(directory, name, run_at):
    """An archived run that failed under the #73/#74 contract."""
    return write_archive(
        directory, name, run_at, status="failed", error="the step reported failure"
    )


def _live_failing_twice(directory, name=WATCHED):
    """Two failed runs recent enough that liveness still reads the workflow live.

    The scenario the escalator is for: on schedule, archiving every run, and
    failing every one of them.
    """
    _failed(directory, name, _now() - timedelta(hours=20))
    _failed(directory, name, _now() - timedelta(minutes=30))


# --------------------------------------------------------------------------
# `_audit_pass` drives the REGISTERED step list.
#
# It does not hard-code the order it believes in: the dedupe's correctness
# depends on `check_failure_streaks` running after `check_liveness` (so
# `archive_readable`, `audit_config_error` and `stale_workflows` exist) and
# before the terminal gate, and both of that step's guards default OPEN. A
# helper that called the handlers in its own order would keep passing through a
# reorder that breaks production.
# --------------------------------------------------------------------------

def _audit_pass(directory, run_at=None, channels=None, dry_run=False) -> dict:
    """Run one audit pass over the REGISTERED steps and archive it.

    Returns the merged context, so a test reads `failure_escalations_sent` off
    it exactly as the report step does.
    """
    context: dict = {}
    failures: list[tuple[str, str]] = []
    notifier = MagicMock(
        return_value=channels if channels is not None else {"email": True}
    )
    with patch(
        "src.agent.workflows.run_audit.send_consecutive_failure_notification",
        notifier,
    ), patch(
        "src.agent.workflows.run_audit.send_stale_workflow_notification",
        # A concrete return, not a bare MagicMock: `send_stale_alerts` puts the
        # notifier's result into the context, and the context is archived as
        # YAML, which cannot represent a mock.
        return_value={"email": True},
    ), patch(
        "src.agent.workflows.run_audit.send_check_failure_notification",
        return_value={"email": True},
    ):
        for step in RunAuditWorkflow.steps:
            if dry_run and step.skip_on_dry_run:
                continue
            outcome = step.handler(None, context)
            if isinstance(outcome, StepFailure):
                # The loop CONTINUES, exactly as `base._execute_step` does: a
                # returned failure marks the step failed and does not raise, so
                # later steps still run and still see the context. Breaking here
                # would make the helper disagree with the runner on the one path
                # where a step reports its own failure.
                context.update(outcome.context or {})
                failures.append((step.name, outcome.error))
                continue
            if outcome:
                context.update(outcome)

    write_archive(
        directory,
        "run_audit",
        run_at if run_at is not None else _now(),
        context=dict(context),
    )
    context["_notifier"] = notifier
    context["_step_failures"] = failures
    return context


def _escalated(context: dict) -> list[str]:
    return [alert["subject"] for alert in context.get("failure_escalations_sent", [])]


# --------------------------------------------------------------------------
# The registered shape itself
# --------------------------------------------------------------------------


def test_the_registered_step_order_is_the_one_the_dedupe_depends_on():
    """Both guards in `check_failure_streaks` default open, so order is load-bearing.

    Move it above `check_liveness` and it would run against a config the auditor
    rejected and return `StepFailure` instead of delegating on an unreadable
    archive — with every behavioural test still green, because they would follow
    the reorder. This asserts the contract directly.
    """
    names = [step.name for step in RunAuditWorkflow.steps]

    assert names == [
        "check_liveness",
        "send_stale_alerts",
        "check_failure_streaks",
        "send_failure_escalations",
        "generate_audit_report",
        "fail_on_unknown_verdict",
    ]


def test_only_the_alerting_steps_are_skipped_on_a_dry_run():
    """A dry run must still compute and report; it must not send.

    Asserted rather than left to the comments on the `StepDefinition`s.
    """
    flags = {step.name: step.skip_on_dry_run for step in RunAuditWorkflow.steps}

    assert flags["check_failure_streaks"] is False
    assert flags["send_failure_escalations"] is True
    assert flags["generate_audit_report"] is False
    assert flags["fail_on_unknown_verdict"] is False


def test_the_workflow_runs_end_to_end_over_its_registered_steps(audited, tmp_path):
    """The six steps compose through the real runner, not only as loose handlers."""
    _live_failing_twice(audited)
    manager = StateManager(state_file=tmp_path / "state.yaml")

    with patch(
        "src.agent.workflows.run_audit.send_consecutive_failure_notification",
        return_value={"email": True},
    ), patch("src.agent.workflows.run_audit.send_stale_workflow_notification"):
        result = RunAuditWorkflow(state_manager=manager).run()

    assert result.context["failure_escalations_due"] == [WATCHED]
    assert result.context[f"{WATCHED}{ESCALATED_SUFFIX}"]


# --------------------------------------------------------------------------
# The escalation decision
# --------------------------------------------------------------------------


def test_one_failure_does_not_escalate(audited):
    """AC-1's lower half: the first failure warns nobody."""
    _failed(audited, WATCHED, _now() - timedelta(minutes=30))

    context = _audit_pass(audited)

    assert context[f"{WATCHED}_consecutive_failures"] == 1
    assert _escalated(context) == []


def test_two_consecutive_failures_escalate(audited):
    """AC-1's upper half, and the notifier's ARGUMENTS are observed.

    `_escalated()` reads the loop variable, not the call, so without asserting
    `call_args` the alert could ship reading "failed its last None runs in a
    row" with every test green.
    """
    _live_failing_twice(audited)

    context = _audit_pass(audited)

    assert context[f"{WATCHED}_consecutive_failures"] == 2
    assert _escalated(context) == [WATCHED]
    assert context["failure_escalations_delivered"] is True

    kwargs = context["_notifier"].call_args.kwargs
    assert kwargs["workflow_name"] == WATCHED
    assert kwargs["consecutive_failures"] == 2
    assert kwargs["threshold"] == 2


def test_an_isolated_failure_between_successes_does_not_escalate(audited):
    """AC-2, end to end through the steps rather than only the counter."""
    write_archive(audited, WATCHED, _now() - timedelta(hours=40))
    _failed(audited, WATCHED, _now() - timedelta(hours=20))
    write_archive(audited, WATCHED, _now() - timedelta(minutes=30))

    context = _audit_pass(audited)

    assert context[f"{WATCHED}_consecutive_failures"] == 0
    assert _escalated(context) == []


def test_an_escalation_is_not_repeated_across_three_audit_passes(audited):
    """AC-1's "exactly once", and it takes THREE passes to test.

    The high-water mark must be carried forward by a pass that suppresses. An
    implementation that records only what it escalated *this* pass passes a
    two-pass test and fails here: pass 1 escalates and records, pass 2
    suppresses and records nothing, pass 3 reads an empty ledger and
    re-escalates. Two passes cannot tell the two implementations apart.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited)
    second = _audit_pass(audited)
    third = _audit_pass(audited)

    assert _escalated(first) == [WATCHED]
    assert _escalated(second) == [], "the mark was written but not read"
    assert _escalated(third) == [], "the mark was read but not carried forward"


def test_a_further_failed_run_escalates_again(audited):
    """The semantics ratified at the plan gate: one alert per failed run.

    A job failing daily nags daily rather than going quiet after the first
    alert. The alternative reading of AC-1 — once per streak episode — is what
    this test would fail, which is the point of pinning it.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited)
    quiet = _audit_pass(audited)

    _failed(audited, WATCHED, _now() - timedelta(minutes=5))
    after = _audit_pass(audited)

    assert _escalated(first) == [WATCHED]
    assert _escalated(quiet) == []
    assert _escalated(after) == [WATCHED]
    assert after[f"{WATCHED}_consecutive_failures"] == 3


# --------------------------------------------------------------------------
# The ledger survives every exit
# --------------------------------------------------------------------------


def test_a_pass_that_finds_the_config_unusable_does_not_erase_the_ledger(audited):
    """One transient bad pass must not re-arm every streak in the system.

    `check_failure_streaks` returns early on a config the auditor already
    rejected. If that return carries no marks, the pass archives a context
    without them and the NEXT pass reads an empty ledger.
    """
    _live_failing_twice(audited)
    first = _audit_pass(audited)
    assert _escalated(first) == [WATCHED]

    with patch.object(agent_settings, "audit_expected_interval_hours", {}):
        broken = _audit_pass(audited)

    assert broken.get(f"{WATCHED}{ESCALATED_SUFFIX}"), "the mark was dropped"

    third = _audit_pass(audited)
    assert _escalated(third) == []


def test_a_workflow_briefly_dropped_from_the_cadence_config_keeps_its_mark(audited):
    """The carry-forward covers EVERY prior mark, not only the watched subset.

    `audit_expected_interval_hours` is hand-edited, so a workflow can leave and
    return. Carrying only watched names would discard its mark on an otherwise
    healthy pass and re-escalate on its return.
    """
    _live_failing_twice(audited)
    first = _audit_pass(audited)
    assert _escalated(first) == [WATCHED]

    with patch.object(
        agent_settings, "audit_expected_interval_hours", {"website_export": 24.0}
    ):
        elsewhere = _audit_pass(audited)

    assert elsewhere.get(f"{WATCHED}{ESCALATED_SUFFIX}"), "the mark was dropped"

    third = _audit_pass(audited)
    assert _escalated(third) == []


def test_an_archive_that_becomes_unreadable_mid_pass_fails_the_step(audited):
    """The `OSError` branch, which no flag-guard test reaches.

    `check_liveness` succeeds, the directory then disappears, and the streak
    read raises. Catching that into an empty result would restore the silent
    all-clear: "no consecutive failures anywhere".
    """
    _live_failing_twice(audited)
    context = check_liveness(None, {})
    assert context["archive_readable"] is True

    with patch(
        "src.agent.workflows.run_audit.consecutive_failures",
        side_effect=OSError("archive vanished"),
    ):
        outcome = check_failure_streaks(None, context)

    assert isinstance(outcome, StepFailure)
    assert outcome.context["failure_streak_skip_reason"] == "archive_unreadable"
    assert "archive vanished" in outcome.error


def test_a_step_failure_still_carries_the_ledger_forward(audited):
    """`StepFailure.context` REPLACES the result, so it must carry the marks itself."""
    _live_failing_twice(audited)
    first = _audit_pass(audited)
    assert _escalated(first) == [WATCHED]

    context = check_liveness(None, {})
    with patch(
        "src.agent.workflows.run_audit.consecutive_failures",
        side_effect=OSError("archive vanished"),
    ):
        outcome = check_failure_streaks(None, context)

    assert outcome.context.get(f"{WATCHED}{ESCALATED_SUFFIX}")


def test_a_corrupt_newest_prior_record_falls_back_to_the_one_before_it(audited):
    """Two prior records, newest corrupt — the case one record cannot exercise.

    `iter_runs` skips an unparseable file, so the marks come from the newest
    READABLE record. Those are at most one pass stale and resolve toward a
    duplicate alert, never toward suppression. Reading only the single newest
    file would turn one corrupt record into an empty ledger and re-escalate
    every current streak at once.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited, run_at=_now() - timedelta(minutes=20))
    second = _audit_pass(audited, run_at=_now() - timedelta(minutes=10))
    assert _escalated(first) == [WATCHED]
    assert _escalated(second) == []

    stamp = (_now() - timedelta(minutes=10)).strftime("%Y%m%d_%H%M%S")
    (audited / f"run_audit_{stamp}.yaml").write_text("{ not: valid: yaml")

    third = _audit_pass(audited)

    assert _escalated(third) == [], "the fallback record's marks were not read"


def test_an_empty_archive_of_its_own_runs_escalates_rather_than_suppressing(audited):
    """With no readable prior record at all, the escalator alerts.

    Suppressing because you cannot tell whether you already alerted is silent
    success reproduced inside the escalator.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited, run_at=_now() - timedelta(minutes=10))
    assert _escalated(first) == [WATCHED]

    for path in audited.glob("run_audit_*.yaml"):
        path.unlink()

    second = _audit_pass(audited)
    assert _escalated(second) == [WATCHED]


def test_a_non_string_context_key_does_not_abort_the_audit(audited):
    """A record can parse as YAML and still carry a non-string key.

    `key.endswith` would raise `AttributeError`, which is not an `OSError`, so
    nothing downstream catches it and the whole unattended audit aborts.
    """
    _live_failing_twice(audited)
    write_archive(
        audited, "run_audit", _now() - timedelta(minutes=10), context={1: "an int key"}
    )

    context = _audit_pass(audited)

    assert _escalated(context) == [WATCHED]


# --------------------------------------------------------------------------
# Delivery gates the mark
# --------------------------------------------------------------------------


def test_a_failed_send_to_a_configured_channel_does_not_burn_the_mark(audited):
    """Every notifier swallows its exception and returns False.

    A dead SMTP host, a rejected key or an HTTP 500 is therefore silent. If the
    mark advanced anyway the alert would be lost permanently and never retried.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited, channels={"email": False})

    assert _escalated(first) == [WATCHED]
    assert first["failure_escalations_delivered"] is False
    assert f"{WATCHED}{ESCALATED_SUFFIX}" not in first

    second = _audit_pass(audited, channels={"email": True})
    assert _escalated(second) == [WATCHED], "the retry did not happen"


def test_a_console_only_result_advances_the_mark(audited):
    """`{"console": True}` is the no-channel-configured default, not a failure.

    There is nothing to deliver to, so re-escalating to a console nobody reads
    on every pass forever is a busy-loop rather than a signal.
    """
    _live_failing_twice(audited)

    first = _audit_pass(audited, channels={"console": True})

    assert _escalated(first) == [WATCHED]
    assert first["failure_escalations_delivered"] is False
    assert first[f"{WATCHED}{ESCALATED_SUFFIX}"]

    second = _audit_pass(audited, channels={"console": True})
    assert _escalated(second) == []


# --------------------------------------------------------------------------
# The threshold knob
# --------------------------------------------------------------------------


def test_a_bad_threshold_does_not_stop_the_module_from_importing():
    """The knob one step reads must not be able to stop every agent workflow.

    Validating in `__post_init__` raised from module scope, where
    `agent_settings` is constructed, so a mistyped value took down every entry
    point including the auditor meant to notice.
    """
    with patch.dict("os.environ", {"AGENT_CONSECUTIVE_FAILURE_THRESHOLD": "0"}):
        settings = AgentSettings()

    assert settings.consecutive_failure_threshold_raw == "0"


@pytest.mark.parametrize(
    "raw, expected_fragment",
    [("0", "at least 1"), ("-3", "at least 1"), ("abc", "not a number")],
)
def test_an_unusable_threshold_is_refused_with_a_message_naming_the_variable(
    raw, expected_fragment
):
    value, error = parse_failure_threshold(raw)

    assert value is None
    assert "AGENT_CONSECUTIVE_FAILURE_THRESHOLD" in error
    assert expected_fragment in error


def test_the_default_threshold_is_two():
    """The UP direction is unguarded, so the default is pinned instead."""
    value, error = parse_failure_threshold(
        AgentSettings().consecutive_failure_threshold_raw
    )

    assert (value, error) == (2, None)


def test_an_unusable_threshold_fails_the_step_and_keeps_the_ledger(audited):
    """Loud, and scoped to the check it disables — the `_cadence_config_error` shape."""
    _live_failing_twice(audited)
    first = _audit_pass(audited)
    assert _escalated(first) == [WATCHED]

    context = check_liveness(None, {})
    with patch.object(agent_settings, "consecutive_failure_threshold_raw", "0"):
        outcome = check_failure_streaks(None, context)

    assert isinstance(outcome, StepFailure)
    assert outcome.context["failure_streak_skip_reason"] == "threshold_unusable"
    assert outcome.context.get(f"{WATCHED}{ESCALATED_SUFFIX}")


# --------------------------------------------------------------------------
# The allowlist, and what the report says
# --------------------------------------------------------------------------


def test_the_escalator_passes_the_allowlist_to_both_archive_reads(audited):
    """Asserted on the CALLS, because the output cannot see this.

    `check_failure_streaks` iterates `watched` and looks each name up, so extra
    entries in the returned dicts are never consulted and removing `workflows=`
    changes no observable context. The round-2 re-check found the previous
    version of this test — which asserted only on the context — passed under the
    exact mutation its docstring said must fail. The allowlist is defence in
    depth against a record written under an invented name (#124), and this is
    what actually pins it.
    """
    _live_failing_twice(audited)
    # OUTSIDE the patch block. `check_liveness` calls `latest_run_per_workflow`
    # with the same allowlist, so recording it here would satisfy the streak
    # assertion below no matter what `check_failure_streaks` passed -- which is
    # how the previous version of this test passed under the mutation it names.
    liveness = check_liveness(None, {})

    with patch.object(
        run_audit_module, "consecutive_failures", wraps=consecutive_failures
    ) as streaks, patch.object(
        run_audit_module, "latest_run_per_workflow", wraps=latest_run_per_workflow
    ) as latest:
        check_failure_streaks(None, liveness)

    assert streaks.call_args.kwargs["workflows"] == {WATCHED}
    seen = [call.kwargs.get("workflows") for call in latest.call_args_list]
    assert seen.count({WATCHED}) == 1, "the streak read dropped its allowlist"
    assert {"run_audit"} in seen, "the ledger read dropped its allowlist"


def test_records_under_names_outside_the_cadence_config_are_not_escalated(audited):
    """The behavioural half: a synthetic name and an unwatched workflow."""
    _live_failing_twice(audited)
    _failed(audited, "failing", _now() - timedelta(hours=20))
    _failed(audited, "failing", _now() - timedelta(minutes=30))
    _failed(audited, "model_training", _now() - timedelta(hours=20))
    _failed(audited, "model_training", _now() - timedelta(minutes=30))

    context = _audit_pass(audited)

    assert _escalated(context) == [WATCHED]
    assert "failing_consecutive_failures" not in context
    assert "model_training_consecutive_failures" not in context


def test_a_stalled_and_failing_workflow_is_not_paged_twice(audited):
    """`send_stale_alerts` has already reported it; one workflow, one page."""
    _failed(audited, WATCHED, _now() - timedelta(days=6))
    _failed(audited, WATCHED, _now() - timedelta(days=5))

    context = _audit_pass(audited)

    assert context[f"{WATCHED}_verdict"] == HealthVerdict.DEGRADED.value
    assert context[f"{WATCHED}_consecutive_failures"] == 2
    assert _escalated(context) == []
    assert context["failure_escalations_suppressed_stale"] == [WATCHED]


def test_the_report_does_not_say_no_action_needed_while_a_workflow_keeps_failing(
    audited,
):
    """The epic's own defect, committed by the report."""
    _live_failing_twice(audited)

    context = _audit_pass(audited)
    report = generate_audit_report(None, context)

    assert context[f"{WATCHED}_verdict"] == HealthVerdict.HEALTHY.value
    assert "No action needed" not in report["recommendation"]
    assert WATCHED in report["recommendation"]
    assert report["workflows_failing_repeatedly"] == [WATCHED]


def test_the_archived_summary_does_not_claim_health_beside_an_escalation(audited):
    """`audit_summary` is the typed surface #77/#78/#79 bind to, and it is archived.

    Fixing only the recommendation would leave the machine-readable half
    asserting the opposite of the sentence beside it.
    """
    _live_failing_twice(audited)

    context = _audit_pass(audited)
    report = generate_audit_report(None, context)

    assert report["audit_summary"]["all_checked_healthy"] is False
    assert report["audit_summary"]["failing_repeatedly"] == [WATCHED]


def test_a_clean_audit_still_reports_no_action_needed(audited):
    """The honest all-clear must survive: a healthy, succeeding workflow."""
    write_archive(audited, WATCHED, _now() - timedelta(minutes=30))

    context = _audit_pass(audited)
    report = generate_audit_report(None, context)

    assert report["audit_summary"]["all_checked_healthy"] is True
    assert report["recommendation"] == (
        "No action needed - every audited workflow is running."
    )


def test_an_absent_streak_check_does_not_render_as_a_clean_one(audited):
    """An absent key means the step left no record, not that it found nothing."""
    context = check_liveness(None, {})

    report = generate_audit_report(None, context)

    assert "did not run" in report["recommendation"]
    assert "reason not recorded" in report["recommendation"]


def test_a_detected_streak_still_lets_the_audit_run_complete(audited):
    """D012's rule, applied to this detector: a detection is never `unknown`.

    Failing the auditor's own run at the moment it correctly detected something
    would make a working detector indistinguishable from a broken one.
    """
    _live_failing_twice(audited)

    context = _audit_pass(audited)
    gate = fail_on_unknown_verdict(None, context)

    assert not isinstance(gate, StepFailure)


# --------------------------------------------------------------------------
# The notifier itself
#
# It is patched out everywhere it is used, so without these its type, severity
# and message could all change with no test failing.
# --------------------------------------------------------------------------


@patch("src.agent.notifications.NotificationManager")
def test_the_escalation_notification_carries_its_type_and_severity(manager_cls):
    manager_cls.return_value.send.return_value = {"email": True}

    send_consecutive_failure_notification(
        workflow_name="daily_labeling", consecutive_failures=3, threshold=2
    )

    sent = manager_cls.return_value.send.call_args.args[0]
    assert sent.notification_type.value == "check_failed"
    assert sent.severity == "error"
    assert "daily_labeling" in sent.subject
    assert "3" in sent.message and "2" in sent.message
    assert sent.details["consecutive_failures"] == 3
    assert sent.details["threshold"] == 2


def test_a_disabled_streak_check_is_never_reported_as_no_action_needed(audited):
    """Both surfaces, on the path an earlier fix opened and then mis-reported.

    A healthy, succeeding workflow with an unusable threshold: liveness is clean,
    so the summary would read healthy and the recommendation would read "No
    action needed" — beside a clause saying the streak check did not run. An
    earlier fix deleted the guard on this branch as redundant and shipped
    exactly that self-contradiction; round 2 of review caught it.

    Note the assertion the previous test lacked: that the all-clear is ABSENT,
    not merely that the warning is present.
    """
    write_archive(audited, WATCHED, _now() - timedelta(minutes=30))

    with patch.object(agent_settings, "consecutive_failure_threshold_raw", "0"):
        context = _audit_pass(audited)
        report = generate_audit_report(None, context)

    assert context["failure_streaks_checked"] is False
    assert context[f"{WATCHED}_verdict"] == HealthVerdict.HEALTHY.value
    assert "No action needed" not in report["recommendation"]
    assert "did not run" in report["recommendation"]
    assert report["audit_summary"]["all_checked_healthy"] is False


def test_a_malformed_key_reads_as_a_failure_and_never_as_a_success(audited):
    """Both halves of the round-3 finding, in one record.

    Keys are partitioned rather than filtered, so a record this reader cannot
    fully parse is evidence something is wrong and never evidence that nothing
    is. Dropping the unreadable part was tried and was wrong: a run archived
    `completed` whose failed step sat under a non-string key read as a SUCCESS
    and reset the streak.

    The record carries a non-string key **beside** a string one in each mapping,
    so `sorted()` actually has two keys to compare — the `TypeError` path a
    single-key record never reaches.
    """
    write_archive(
        audited,
        WATCHED,
        _now() - timedelta(minutes=30),
        status="completed",
        steps={
            "label": {"status": "completed"},
            7: {"status": "failed", "error": "the real failure"},
        },
        context={"ok": True, 9: "a key this reader cannot interpret"},
    )

    assert consecutive_failures(audited, workflows={WATCHED}) == {WATCHED: 1}

    context = _audit_pass(audited)
    assert context["failure_streaks_checked"] is True
    assert context[f"{WATCHED}_consecutive_failures"] == 1


def test_a_malformed_record_does_not_abort_the_audit_or_erase_the_ledger(audited):
    """`TypeError`/`AttributeError` are not `OSError`, so no guard catches them.

    An abort would mean the step never returns, so the carried marks never reach
    the context and the next pass re-alerts everything — the defect the ledger
    exists to prevent.
    """
    _live_failing_twice(audited)
    first = _audit_pass(audited)
    assert _escalated(first) == [WATCHED]

    write_archive(
        audited,
        "run_audit",
        _now() - timedelta(minutes=5),
        context={"ok": True, 1: "an int key"},
    )

    second = _audit_pass(audited)

    assert second["failure_streaks_checked"] is True
    assert second.get(f"{WATCHED}{ESCALATED_SUFFIX}"), "the ledger was erased"
