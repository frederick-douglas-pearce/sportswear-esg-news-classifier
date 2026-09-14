"""Tests for the run-archive reader and the liveness audit (#76).

Every test here writes into a tmp_path archive, never the real one. That is not
generic hygiene: this change exists to reason about a directory that historical
test runs polluted under production workflow names, so a test that wrote into it
would be manufacturing the very artifact under discussion.
"""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.agent.archive import (
    ArchivedRun,
    failure_signals,
    iter_runs,
    latest_run_per_workflow,
    run_succeeded,
    vacuous_success_signals,
)
from src.agent.config import AgentSettings, agent_settings
from src.agent.health import HealthVerdict
from src.agent.state import StepState, WorkflowState, WorkflowStatus
from src.agent.workflows.base import StepFailure
from src.agent.workflows.run_audit import (
    check_liveness,
    fail_on_unknown_verdict,
    generate_audit_report,
    send_stale_alerts,
)

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


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


def test_the_report_does_not_call_an_all_skipped_audit_healthy(audited):
    """`summarize` refuses vacuous truth; the report must not reintroduce it."""
    with patch.object(agent_settings, "audit_expected_interval_hours", {}):
        context = check_liveness(None, {})

    report = generate_audit_report(None, context)

    assert report["audit_summary"]["all_checked_healthy"] is False


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
