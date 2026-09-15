"""Tests for the run-archive reader and the liveness audit (#76).

Every test here writes into a tmp_path archive, never the real one. That is not
generic hygiene: this change exists to reason about a directory that historical
test runs polluted under production workflow names, so a test that wrote into it
would be manufacturing the very artifact under discussion.
"""

import importlib.util
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.agent.archive import (
    SIGNAL_KINDS,
    ArchivedRun,
    failure_signals,
    iter_runs,
    latest_run_per_workflow,
    run_succeeded,
    vacuous_success_signals,
)
from src.agent.config import AgentSettings, agent_settings
from src.agent.health import HealthVerdict
from src.agent.notifications import delivered
from src.agent.state import StepState, WorkflowState, WorkflowStatus
from src.agent.workflows.base import StepFailure
from src.agent.workflows.run_audit import (
    check_liveness,
    fail_on_unknown_verdict,
    generate_audit_report,
    send_stale_alerts,
)

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


def _load_sweep():
    """Load `scripts/audit_archive.py`, which is a script and not a package."""
    path = Path(__file__).parent.parent / "scripts" / "audit_archive.py"
    spec = importlib.util.spec_from_file_location("audit_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load_sweep()


def run_sweep(*argv: str) -> int:
    """Invoke the Class-A sweep's real entry point and return its exit code."""
    with patch.object(sys, "argv", ["audit_archive.py", *argv]):
        return sweep.main()


@pytest.fixture
def history(tmp_path):
    """An isolated archive directory, bound as the agent's history dir."""
    directory = tmp_path / "history"
    directory.mkdir()
    with patch.object(AgentSettings, "history_dir", directory):
        yield directory


def write_archive(directory: Path, name: str, run_at: datetime, **fields) -> Path:
    """Write one archive file the way `_archive_workflow` names it."""
    run_id = run_at.strftime("%Y%m%d_%H%M%S")
    record = {
        "name": name,
        "status": "completed",
        "started_at": run_at.isoformat(),
        "completed_at": run_at.isoformat(),
        "current_step": None,
        "steps": {},
        "context": {},
        "error": None,
        "run_id": run_id,
    }
    record.update(fields)
    path = directory / f"{name}_{run_id}.yaml"
    path.write_text(yaml.dump(record, default_flow_style=False))
    return path


def make_run(name: str = "drift_monitoring", **fields) -> ArchivedRun:
    """An ArchivedRun with no file behind it, for signal/policy tests."""
    data = {"name": name, "status": "completed", "steps": {}, "context": {}}
    data.update(fields)
    return ArchivedRun(
        workflow_name=name,
        run_id="20260914_120000",
        run_at=NOW,
        path=Path(f"{name}_20260914_120000.yaml"),
        data=data,
    )


# --------------------------------------------------------------------------
# The reader
# --------------------------------------------------------------------------


def test_iter_runs_parses_names_containing_underscores(history):
    """Every real workflow name has an underscore; split('_') misparses them."""
    for name in ("daily_labeling", "website_export", "drift_monitoring"):
        write_archive(history, name, NOW)

    found = {run.workflow_name for run in iter_runs(history)}

    assert found == {"daily_labeling", "website_export", "drift_monitoring"}


def test_iter_runs_yields_oldest_first(history):
    """Ordering is contract, not incidental: #75 needs a time-ordered sequence."""
    for days in (3, 1, 2):
        write_archive(history, "daily_labeling", NOW - timedelta(days=days))

    order = [run.run_at for run in iter_runs(history)]

    assert order == sorted(order)


def test_iter_runs_skips_unparseable_archive_without_losing_the_rest(history):
    """One bad record must not blind the reader to every other one."""
    write_archive(history, "daily_labeling", NOW)
    (history / "daily_labeling_20260101_000000.yaml").write_text("{[not: valid")

    runs = list(iter_runs(history))

    assert len(runs) == 1
    assert runs[0].run_id == NOW.strftime("%Y%m%d_%H%M%S")


def test_iter_runs_allowlist_excludes_synthetic_test_names(history):
    """Synthetic archives are excluded by name, never by a content heuristic."""
    write_archive(history, "drift_monitoring", NOW)
    write_archive(history, "resume_test", NOW)
    write_archive(history, "dryrun_test", NOW)

    runs = list(iter_runs(history, workflows={"drift_monitoring"}))

    assert [run.workflow_name for run in runs] == ["drift_monitoring"]


def test_iter_runs_raises_when_the_directory_is_absent(tmp_path):
    """'No archive directory' must not read the same as 'an empty archive'."""
    with pytest.raises(OSError):
        list(iter_runs(tmp_path / "nope"))


def test_latest_run_per_workflow_returns_the_newest(history):
    write_archive(history, "daily_labeling", NOW - timedelta(days=5))
    write_archive(history, "daily_labeling", NOW - timedelta(days=1))

    latest = latest_run_per_workflow(history, workflows={"daily_labeling"})

    assert latest["daily_labeling"].run_at == NOW - timedelta(days=1)


# --------------------------------------------------------------------------
# Class A: vacuous success
# --------------------------------------------------------------------------


def test_vacuous_success_flags_a_completed_run_carrying_a_failure(history):
    """AC-1: status completed, embedded failure signal -> flagged."""
    run = make_run(
        status="completed",
        context={"fp_drift_check_success": False},
    )

    signals = vacuous_success_signals(run)

    assert [s.kind for s in signals] == ["success_flag_false"]
    assert "context.fp_drift_check_success" in str(signals[0])


def test_vacuous_success_is_empty_for_a_run_that_reported_failure():
    """A run that reported failure is not lying about it."""
    run = make_run(status="failed", context={"fp_drift_check_success": False})

    assert vacuous_success_signals(run) == []


def test_signal_kinds_matches_what_the_extractor_emits():
    """`SIGNAL_KINDS` is a listing, not a derivation, so pin it to the emitter.

    `--kind` validates against `SIGNAL_KINDS`. A kind emitted but not listed
    makes `--kind <it>` an argparse error; a kind listed but never emitted makes
    it a filter that matches nothing and exits 0. Neither shows up anywhere
    else: every other test in this file picks its kinds from the same constant,
    so it agrees with itself no matter what either side says.
    """
    everything = make_run(
        status="failed",
        error="the run failed",
        steps={"label": {"status": "failed", "error": "labeling died"}},
        context={
            "fp_verdict": "unknown",
            "fp_drift_check_success": False,
            "errors": ["export failed"],
        },
    )

    assert {s.kind for s in failure_signals(everything)} == SIGNAL_KINDS


def test_failure_signals_names_its_evidence():
    """A finding names where it was found, so it can be checked not believed."""
    run = make_run(
        steps={"check_fp": {"status": "failed", "error": "boom"}},
        context={"errors": ["export failed"], "fp_verdict": "unknown"},
    )

    kinds = {s.kind for s in failure_signals(run)}

    assert kinds == {"step_failed", "step_error", "context_errors", "verdict_unknown"}


# --------------------------------------------------------------------------
# run_succeeded: the policy #75 imports
# --------------------------------------------------------------------------


def test_run_succeeded_ignores_ordinary_non_failure_context():
    """Over-broad matching here becomes #75's alert noise."""
    run = make_run(
        context={
            "alerts_sent": False,
            "alerts_skipped": True,
            "reason": "nothing_to_report",
        }
    )

    assert run_succeeded(run) is True


def test_run_succeeded_is_false_on_an_unresolved_verdict():
    assert run_succeeded(make_run(context={"fp_verdict": "unknown"})) is False


def test_run_succeeded_is_false_on_a_failed_step():
    run = make_run(steps={"check_fp": {"status": "failed", "error": None}})

    assert run_succeeded(run) is False


def test_run_succeeded_is_false_when_the_record_has_no_status():
    """Absence of a failure signal is not evidence of success."""
    run = make_run()
    del run.data["status"]

    assert run_succeeded(run) is False


def test_run_succeeded_ignores_the_historical_success_flag():
    """The pre-#73/#74 form is archaeology, not a live failure signal.

    Reading it as one would make #75 re-alert on history. The same record is
    still a Class A finding -- that is the whole point of separating extraction
    from policy.
    """
    run = make_run(context={"fp_drift_check_success": False})

    assert run_succeeded(run) is True
    assert [s.kind for s in vacuous_success_signals(run)] == ["success_flag_false"]


# --------------------------------------------------------------------------
# The reader is coupled to the writer's shape, through CI rather than hope
# --------------------------------------------------------------------------


def _archive_state(directory: Path, state: WorkflowState) -> None:
    path = directory / f"{state.name}_{state.run_id}.yaml"
    path.write_text(yaml.dump(state.to_dict(), default_flow_style=False))


def test_round_trip_clean_run_reads_as_succeeded(history):
    """Built from real WorkflowState, so a field rename in state.py breaks CI."""
    state = WorkflowState(
        name="daily_labeling",
        status=WorkflowStatus.COMPLETED,
        run_id=NOW.strftime("%Y%m%d_%H%M%S"),
        steps={"label": StepState(name="label", status=WorkflowStatus.COMPLETED)},
    )
    _archive_state(history, state)

    (run,) = iter_runs(history, workflows={"daily_labeling"})

    assert run_succeeded(run) is True
    assert failure_signals(run) == []


def test_round_trip_completed_run_with_a_failed_step_reads_as_failed(history):
    """The vacuous-success shape, constructed through the real serializer."""
    state = WorkflowState(
        name="daily_labeling",
        status=WorkflowStatus.COMPLETED,
        run_id=NOW.strftime("%Y%m%d_%H%M%S"),
        steps={
            "label": StepState(
                name="label", status=WorkflowStatus.FAILED, error="labeling died"
            )
        },
    )
    _archive_state(history, state)

    (run,) = iter_runs(history, workflows={"daily_labeling"})

    assert run_succeeded(run) is False
    assert {s.kind for s in vacuous_success_signals(run)} == {
        "step_failed",
        "step_error",
    }


def test_round_trip_couples_the_context_the_verdict_signals_live_in(history):
    """`context` is the field #73/#74 write their signals into, so pin it too.

    The other round-trip tests couple `status`, `steps` and `error`. Without
    this one, `context` could be renamed in `state.py` and the reader would
    quietly stop seeing every unresolved verdict and every collected error --
    degrading to "sees no signal" with a green suite, which is the exact defect
    the round-trip was added to make impossible.
    """
    state = WorkflowState(
        name="drift_monitoring",
        status=WorkflowStatus.COMPLETED,
        run_id=NOW.strftime("%Y%m%d_%H%M%S"),
        context={"fp_verdict": "unknown", "errors": ["reference data missing"]},
    )
    _archive_state(history, state)

    (run,) = iter_runs(history, workflows={"drift_monitoring"})

    assert run.context == state.context
    assert {s.kind for s in failure_signals(run)} == {
        "verdict_unknown",
        "context_errors",
    }
    assert run_succeeded(run) is False


def test_round_trip_failed_run_reads_as_failed(history):
    state = WorkflowState(
        name="daily_labeling",
        status=WorkflowStatus.FAILED,
        run_id=NOW.strftime("%Y%m%d_%H%M%S"),
        error="the run failed",
    )
    _archive_state(history, state)

    (run,) = iter_runs(history, workflows={"daily_labeling"})

    assert run_succeeded(run) is False


# --------------------------------------------------------------------------
# Class B: liveness
# --------------------------------------------------------------------------


@pytest.fixture
def audited(history):
    """Audit exactly one expected workflow and one skipped one."""
    with (
        patch.object(
            agent_settings, "audit_expected_interval_hours", {"daily_labeling": 24.0}
        ),
        patch.object(agent_settings, "audit_grace_hours", 6.0),
        patch.object(
            agent_settings, "audit_skipped_workflows", {"model_training": "not on cron"}
        ),
    ):
        yield history


def test_fresh_workflow_is_healthy(audited):
    write_archive(audited, "daily_labeling", datetime.now(timezone.utc))

    result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.HEALTHY.value
    assert result["stale_workflows"] == []


def test_stale_workflow_is_degraded(audited):
    """AC-2: no run within the expected interval -> flagged stalled."""
    write_archive(
        audited, "daily_labeling", datetime.now(timezone.utc) - timedelta(hours=40)
    )

    result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.DEGRADED.value
    assert result["stale_workflows"] == ["daily_labeling"]


class _FrozenDatetime(datetime):
    """`datetime` whose `now()` is NOW, so a boundary can be hit exactly."""

    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def frozen_now():
    with patch("src.agent.workflows.run_audit.datetime", _FrozenDatetime):
        yield NOW


def test_a_run_exactly_at_the_grace_boundary_is_still_healthy(audited, frozen_now):
    """The comparison is `age > threshold`, so the boundary itself passes.

    Pinned because the off-by-one goes both ways and neither direction is
    visible from a test that writes "40 hours ago": `>=` here would alert on a
    workflow that ran precisely on time, and a wrong threshold expression would
    not be caught by any other test in this file.
    """
    write_archive(audited, "daily_labeling", NOW - timedelta(hours=30))

    result = check_liveness(None, {})

    assert result["daily_labeling_age_hours"] == 30.0
    assert result["daily_labeling_verdict"] == HealthVerdict.HEALTHY.value


def test_a_run_one_second_past_the_grace_boundary_is_degraded(audited, frozen_now):
    """The other side of the same boundary: interval 24h + grace 6h."""
    write_archive(audited, "daily_labeling", NOW - timedelta(hours=30, seconds=1))

    result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.DEGRADED.value
    assert result["stale_workflows"] == ["daily_labeling"]


def test_a_few_seconds_of_backwards_clock_skew_is_not_a_page(audited, frozen_now):
    """A time daemon stepping the clock at boot must not alert on a live job.

    The future-dated branch below exists for a record that cannot have come
    from a run. Without a tolerance it also fires on a run archived seconds
    before a backwards step -- paging an operator about a healthy workflow,
    with text saying its run is dated in the future. A false page is the one
    thing an epic about honest alerting should not ship.
    """
    write_archive(audited, "daily_labeling", NOW + timedelta(seconds=30))

    result = check_liveness(None, {})

    assert result["daily_labeling_age_hours"] < 0
    assert result["daily_labeling_verdict"] == HealthVerdict.HEALTHY.value
    assert result["stale_workflows"] == []


def test_the_skew_tolerance_is_not_wide_enough_to_swallow_a_real_anomaly(
    audited, frozen_now
):
    """The tolerance covers clock noise, not a future-dated record.

    Pinned separately because the obvious wrong fix is to reuse
    `audit_grace_hours` (hours) as the tolerance. A run dated two hours ahead
    is the case the branch exists to catch, so an hours-wide tolerance would
    report it healthy -- the silence the branch was added to remove, restored
    by the guard meant to soften it.
    """
    write_archive(audited, "daily_labeling", NOW + timedelta(hours=2))

    result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.DEGRADED.value
    assert result["stale_workflows"] == ["daily_labeling"]


def test_a_future_dated_run_is_degraded_rather_than_permanently_fresh(
    audited, frozen_now
):
    """A negative age passes every freshness test, silencing the workflow.

    This is the epic's own failure mode inside the detector: a clock that moved
    or a planted record makes the newest archive future-dated, the age goes
    negative, `age > threshold` is false forever, and the auditor never reports
    that workflow again. It has to be its own branch -- clamping the age to
    zero would produce the same permanent silence.
    """
    write_archive(audited, "daily_labeling", NOW + timedelta(days=1))

    result = check_liveness(None, {})

    assert result["daily_labeling_age_hours"] < 0
    assert result["daily_labeling_verdict"] == HealthVerdict.DEGRADED.value
    assert result["stale_workflows"] == ["daily_labeling"]
    assert "future" in result["daily_labeling_error"]


def test_a_workflow_that_never_ran_is_degraded_not_unknown(audited):
    """D012, and the single most important assertion in this file.

    Mapping "never ran" to `unknown` would trip the terminal gate, failing the
    auditor's own run at the exact moment it correctly detected a dead job --
    a correct detection made indistinguishable from the auditor malfunctioning.
    """
    result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.DEGRADED.value
    assert result["daily_labeling_last_run"] is None
    assert result["stale_workflows"] == ["daily_labeling"]


def test_an_unreadable_archive_is_unknown_for_every_subject(audited):
    """The genuine can't-tell case, and the only one that is `unknown`."""
    with patch(
        "src.agent.workflows.run_audit.latest_run_per_workflow",
        side_effect=OSError("permission denied"),
    ):
        result = check_liveness(None, {})

    assert result["daily_labeling_verdict"] == HealthVerdict.UNKNOWN.value
    assert result["archive_readable"] is False


def test_a_skipped_workflow_records_its_reason(audited):
    result = check_liveness(None, {})

    assert result["model_training_verdict"] == HealthVerdict.SKIPPED.value
    assert result["model_training_skip_reason"] == "not on cron"


def test_the_gate_lets_a_detected_stall_complete_the_run(audited):
    """A stall is a verdict, so the auditor reports it rather than failing."""
    context = check_liveness(None, {})

    outcome = fail_on_unknown_verdict(None, context)

    assert not isinstance(outcome, StepFailure)


def test_the_gate_fails_the_run_when_the_archive_is_unreadable(audited):
    with patch(
        "src.agent.workflows.run_audit.latest_run_per_workflow",
        side_effect=OSError("permission denied"),
    ):
        context = check_liveness(None, {})

    outcome = fail_on_unknown_verdict(None, context)

    assert isinstance(outcome, StepFailure)


def test_the_gate_fails_an_audit_configured_to_check_nothing():
    """An auditor that checks nothing and reports success is this epic's bug."""
    outcome = fail_on_unknown_verdict(None, {"audited_workflows": []})

    assert isinstance(outcome, StepFailure)


def test_an_all_skipped_cadence_config_fails_instead_of_auditing_nothing(audited):
    """An auditor that checks nothing and archives `completed` is the epic's bug.

    Caught at the check step rather than only at the terminal gate, so the
    recorded error names the cause instead of reporting an absent verdict.
    """
    with patch.object(agent_settings, "audit_expected_interval_hours", {}):
        outcome = check_liveness(None, {})

    assert isinstance(outcome, StepFailure)
    assert "check nothing" in outcome.error


def test_a_workflow_in_both_cadence_dicts_fails_instead_of_being_guessed_at(audited):
    """Audited and skipped at once: alerted on while recorded as never looked at.

    Which dict wins is an artifact of loop order, so there is no precedence to
    pick. The configuration is the defect and the run says so.
    """
    with patch.object(
        agent_settings, "audit_skipped_workflows", {"daily_labeling": "on hold"}
    ):
        outcome = check_liveness(None, {})

    assert isinstance(outcome, StepFailure)
    assert "daily_labeling" in outcome.error


def test_the_report_does_not_call_an_audit_of_nothing_healthy(audited):
    """`summarize` refuses vacuous truth; the report must not reintroduce it.

    The step loop keeps running after a `StepFailure`, so the report step still
    executes against the rejected config's context and must not render either
    "no action needed" or an empty list of unreachable workflows.
    """
    with patch.object(agent_settings, "audit_expected_interval_hours", {}):
        failure = check_liveness(None, {})

    report = generate_audit_report(None, dict(failure.context))

    assert report["audit_summary"]["all_checked_healthy"] is False
    assert "established nothing" in report["recommendation"]


def test_a_misconfigured_audit_alerts_rather_than_only_failing_its_own_run(audited):
    """A FAILED status nobody reads is what #71 found; page the operator."""
    with patch.object(agent_settings, "audit_expected_interval_hours", {}):
        failure = check_liveness(None, {})

    with patch(
        "src.agent.workflows.run_audit.send_check_failure_notification",
        return_value={"email": True},
    ) as notify:
        result = send_stale_alerts(None, dict(failure.context))

    assert notify.called
    assert result["reason"] == "audit_misconfigured"
    assert result["alerts_delivered"] is True


def test_alerts_are_not_counted_as_delivered_when_they_reach_only_the_console(
    audited,
):
    """`{"console": True}` is the no-channel fallback, not a delivery."""
    context = check_liveness(None, {})

    with patch(
        "src.agent.workflows.run_audit.send_stale_workflow_notification",
        return_value={"console": True},
    ):
        result = send_stale_alerts(None, context)

    assert result["alerts_sent"][0]["subject"] == "daily_labeling"
    assert result["alerts_delivered"] is False


def test_an_alert_that_reached_a_real_channel_is_counted_as_delivered(audited):
    """The positive half of `delivered()`, which nothing else asserts.

    Without it `def delivered(result): return False` passes the whole suite --
    every alert would be recorded as having reached nobody, and #75's future
    escalation logic would read that. A predicate needs both of its answers
    pinned or only one of them is tested.
    """
    context = check_liveness(None, {})

    with patch(
        "src.agent.workflows.run_audit.send_stale_workflow_notification",
        return_value={"console": True, "email": True},
    ):
        result = send_stale_alerts(None, context)

    assert result["alerts_delivered"] is True
    assert delivered({"email": True}) is True
    assert delivered({"webhook": False, "console": True}) is False


# --------------------------------------------------------------------------
# The Class-A sweep's exit-code contract
#
# The script is read by a human once, so its exit code is the only part a
# caller can act on mechanically -- and 0 is the answer an operator will
# believe. Each of the three codes is pinned through the real entry point.
# --------------------------------------------------------------------------


def test_sweep_exits_0_when_no_run_reported_success_over_a_failure(history, capsys):
    write_archive(history, "daily_labeling", NOW)

    assert run_sweep("--history-dir", str(history)) == 0
    assert "No archived run reported success" in capsys.readouterr().out


def test_sweep_exits_1_and_names_the_run_when_one_did(history, capsys):
    write_archive(
        history,
        "daily_labeling",
        NOW,
        status="completed",
        steps={"label": {"status": "failed", "error": "labeling died"}},
    )

    assert run_sweep("--history-dir", str(history)) == 1
    assert "daily_labeling 20260914_120000" in capsys.readouterr().out


def test_sweep_does_not_exit_0_when_the_archive_cannot_be_read(tmp_path, capsys):
    """"Cannot look" must never share an answer with "looked, found nothing".

    The code is 2, which argparse also uses for a usage error; the cause is on
    stderr. What must hold is that it is not 0.
    """
    assert run_sweep("--history-dir", str(tmp_path / "gone")) != 0
    assert "could not read the run archive" in capsys.readouterr().err


def test_sweep_rejects_an_unrecognised_kind_instead_of_matching_nothing(history):
    """A typo must not filter everything out and then report a clean archive.

    Without `choices`, `--kind sucess_flag_false` matches no signal, prints "no
    archived run reported success over a failure signal" and exits 0: a clean
    bill of health manufactured by a filter that could never match.

    The assertion is that this is NOT 0, not that 2 identifies the cause. 2 is
    argparse's usage-error code and this script does not reclaim it, so 2 also
    means an unreadable archive; what the contract guarantees is that 0 always
    means checked-and-clean. See the script docstring.
    """
    write_archive(
        history,
        "daily_labeling",
        NOW,
        context={"fp_drift_check_success": False},
    )

    with pytest.raises(SystemExit) as exit_info:
        run_sweep("--history-dir", str(history), "--kind", "sucess_flag_false")

    assert exit_info.value.code == 2


def test_sweep_kind_filter_reports_only_the_requested_kind(history, capsys):
    """The filter itself, against a record carrying more than one kind.

    Accepting every valid `--kind` proves only that argparse let it through.
    Without this, a filter that dropped nothing -- or everything -- passes:
    one over-reports findings the operator asked to exclude, the other exits 0
    on an archive that has them.
    """
    write_archive(
        history,
        "daily_labeling",
        NOW,
        status="completed",
        steps={"label": {"status": "failed", "error": "labeling died"}},
        context={"labeling_success": False},
    )

    assert run_sweep("--history-dir", str(history), "--kind", "success_flag_false") == 1

    output = capsys.readouterr().out
    assert "[success_flag_false]" in output
    assert "[step_failed]" not in output
    assert "[step_error]" not in output


def test_sweep_kind_filter_accepts_every_listed_kind(history):
    """Each `SIGNAL_KINDS` entry is a valid `--kind`, so `choices` stays usable."""
    for kind in SIGNAL_KINDS:
        assert run_sweep("--history-dir", str(history), "--kind", kind) == 0


def test_sweep_all_signals_shows_honest_failures_without_changing_the_verdict(
    history, capsys
):
    """The control group. A run that reported its failure is not a finding."""
    write_archive(
        history,
        "daily_labeling",
        NOW,
        status="failed",
        error="the run failed",
    )

    assert run_sweep("--history-dir", str(history), "--all-signals") == 0

    output = capsys.readouterr().out
    assert "reported failure - not a finding" in output
    assert "No archived run reported success" in output


def test_sweep_without_all_signals_stays_silent_about_honest_failures(history, capsys):
    write_archive(
        history, "daily_labeling", NOW, status="failed", error="the run failed"
    )

    assert run_sweep("--history-dir", str(history)) == 0
    assert "daily_labeling" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# The auditor's own config cannot silently stop covering a scheduled job
# --------------------------------------------------------------------------


def test_every_cron_scheduled_workflow_is_audited_or_explicitly_skipped():
    """The drift direction that is silent: cron gains a job, config does not.

    A scheduled job absent from both dicts has no liveness detector, which is
    this epic's defect reproduced inside the auditor's own configuration. The
    opposite directions -- an entry removed or lengthened -- surface as a noisy
    stale alert, which is the safe way to be wrong.
    """
    setup_cron = Path(__file__).parent.parent / "scripts" / "setup_cron.sh"
    scheduled = set(
        re.findall(
            r'^AGENT_\w+_ENTRY="\$AGENT_\w+_SCHEDULE \$AGENT_SCRIPT (\S+)"',
            setup_cron.read_text(),
            flags=re.MULTILINE,
        )
    )

    assert scheduled, "no agent cron entries found - has setup_cron.sh changed shape?"

    covered = set(agent_settings.audit_expected_interval_hours) | set(
        agent_settings.audit_skipped_workflows
    )

    assert scheduled <= covered, (
        f"scheduled but not audited or skipped: {sorted(scheduled - covered)}"
    )
