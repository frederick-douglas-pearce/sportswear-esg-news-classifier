"""Tests for the shared health-verdict contract (#74).

The vocabulary landed in #71 but was wired into `drift_monitoring` alone. #74
makes it a shared contract: the operations move into `src/agent/health.py`, the
escalation gate into `workflows/base.py`, and both get the coverage the
drift-local version never needed.

Three of these classes exist because of a specific way this can go wrong:

- `TestSummarizeIsNotVacuous` — `all(...)` over an empty sequence is `True`, so
  "no failures found" reports healthy having checked nothing (#95).
- `TestSummarySurvivesTheRunArchive` — the state file is written with
  `yaml.dump` and read with `yaml.safe_load`, so a non-primitive in the context
  loads into a bare `except` that resets all workflow state.
- `TestImportDirection` — `health` must stay an import leaf; the reverse edge is a
  cycle today, via the adopter that already exists (D010.2).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.agent.health import (
    HealthSummary,
    HealthVerdict,
    summarize,
    unresolved,
    verdict_of,
)
from src.agent.state import StateManager, WorkflowState, WorkflowStatus
from src.agent.workflows.base import (
    StepDefinition,
    StepFailure,
    Workflow,
    WorkflowRegistry,
    fail_on_unresolved_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def state_manager(tmp_path):
    """A StateManager backed by a throwaway state file."""
    return StateManager(state_file=tmp_path / "state.yaml")


@pytest.fixture
def cleanup_registry():
    """Restore the workflow registry after a test registers into it."""
    original = dict(WorkflowRegistry._workflows)
    yield
    WorkflowRegistry._workflows = original


@pytest.fixture(autouse=True)
def isolated_history(tmp_path):
    """Keep these tests out of the real ~/.esg-agent/history archive.

    The workflows below reach `complete_workflow`/`fail_workflow`, which archive
    into `agent_settings.history_dir`. Their names are local (`unresolved_run`
    and friends), so this is not the "archived under a production name" case —
    it is the plainer one: a test suite should not deposit runs in the directory
    #76 will audit.
    """
    from unittest.mock import patch

    from src.agent.config import AgentSettings

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    with patch.object(AgentSettings, "history_dir", history_dir):
        yield history_dir


class TestVerdictOf:
    """AC5, read side: absence must resolve to unknown, never to healthy."""

    @pytest.mark.parametrize("verdict", list(HealthVerdict))
    def test_an_explicit_verdict_round_trips(self, verdict):
        assert verdict_of({"x_verdict": verdict.value}, "x_verdict") is verdict

    def test_an_absent_key_is_unknown(self):
        """The pre-#71 shape: a check that failed set no key at all."""
        assert verdict_of({}, "x_verdict") is HealthVerdict.UNKNOWN

    def test_an_explicit_none_is_unknown(self):
        assert verdict_of({"x_verdict": None}, "x_verdict") is HealthVerdict.UNKNOWN

    def test_an_unrecognised_value_is_unknown(self):
        assert verdict_of({"x_verdict": "probably_fine"}, "x_verdict") is HealthVerdict.UNKNOWN

class TestSummarizeIsNotVacuous:
    """`all_checked_healthy` must mean "something ran and it passed" (#95)."""

    def test_no_checks_at_all_is_not_healthy(self):
        assert summarize({})["all_checked_healthy"] is False

    def test_every_check_skipped_is_not_healthy(self):
        """The vacuous-truth case, and the reason this helper is shared.

        `all(v is HEALTHY for v in non_skipped)` returns True here, so a run
        that checked nothing would report healthy — the epic's own thesis
        reproduced inside the mechanism written to enforce it.
        """
        verdicts = {"fp": HealthVerdict.SKIPPED, "ep": HealthVerdict.SKIPPED}

        assert summarize(verdicts)["all_checked_healthy"] is False

    def test_one_real_check_alongside_a_skip_is_healthy(self):
        verdicts = {"fp": HealthVerdict.HEALTHY, "ep": HealthVerdict.SKIPPED}

        summary = summarize(verdicts)

        assert summary["all_checked_healthy"] is True
        assert summary["checked"] == ["fp"]
        assert summary["skipped"] == ["ep"]

    def test_unknown_blocks_healthy(self):
        verdicts = {"fp": HealthVerdict.HEALTHY, "ep": HealthVerdict.UNKNOWN}

        summary = summarize(verdicts)

        assert summary["all_checked_healthy"] is False
        assert summary["unknown"] == ["ep"]

    def test_degraded_blocks_healthy(self):
        verdicts = {"fp": HealthVerdict.DEGRADED}

        summary = summarize(verdicts)

        assert summary["all_checked_healthy"] is False
        assert summary["degraded"] == ["fp"]

    def test_a_skipped_check_is_not_counted_as_checked(self):
        summary = summarize({"ep": HealthVerdict.SKIPPED})

        assert summary["checked"] == []
        assert summary["skipped"] == ["ep"]


class TestUnresolved:
    """The gate's predicate: only an explicit good verdict passes."""

    def test_explicit_verdicts_are_resolved(self):
        verdicts = {
            "a": HealthVerdict.HEALTHY,
            "b": HealthVerdict.DEGRADED,
            "c": HealthVerdict.SKIPPED,
        }

        assert unresolved(verdicts) == []

    def test_unknown_is_unresolved(self):
        assert unresolved({"a": HealthVerdict.UNKNOWN}) == ["a"]


class TestSummarySurvivesTheRunArchive:
    """A summary in the context must round-trip the real save/load path."""

    def test_summary_is_a_plain_dict_at_runtime(self):
        """`HealthSummary` is a TypedDict, so this is a `dict`, not an object."""
        assert type(summarize({"a": HealthVerdict.HEALTHY})) is dict

    def test_state_manager_round_trips_a_summary_in_context(self, state_manager):
        """The load path is `yaml.safe_load`, so test that, not `safe_dump`.

        Production writes with `yaml.dump`, which happily serializes a dataclass
        or an enum member — so a `yaml.safe_dump` assertion would PASS on
        precisely the object that breaks, and the breakage only appears on the
        next run, as `StateManager._load` falling into its bare `except` and
        resetting every workflow's state to `{}`.
        """
        summary = summarize({"fp": HealthVerdict.HEALTHY, "ep": HealthVerdict.SKIPPED})
        state_manager.create_workflow(name="round_trip", steps=["one"], context={})
        state_manager.update_context("round_trip", {"health": summary})

        reloaded = StateManager(state_file=state_manager.state_file)

        assert reloaded.get_workflow("round_trip") is not None, (
            "state failed to load: a non-primitive reached the archive"
        )
        assert reloaded.get_workflow("round_trip").context["health"] == summary

    def test_the_serialization_premise_this_design_rests_on_still_holds(self):
        """Pin PyYAML's behaviour, not this repo's — the premise, not the rule.

        Everything above assumes `yaml.dump` writes a `str, Enum` member as a
        Python object tag that `safe_load` then refuses. That is a property of
        PyYAML, so if a future version changed it, the reasoning in
        `HealthVerdict`'s docstring would silently become wrong while every
        other test kept passing. This test fails in that case and nowhere else;
        it does NOT guard `summarize` (that is the round-trip test above).
        """
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(yaml.dump({"verdict": HealthVerdict.HEALTHY}))

        assert yaml.safe_load(yaml.dump({"verdict": HealthVerdict.HEALTHY.value})) == {
            "verdict": "healthy"
        }


class TestArchivedRunsPredatingTheVerdict:
    """AC4: an archive written before this vocabulary existed still loads."""

    # A drift_monitoring archive in the pre-#71 shape: no verdict keys anywhere,
    # and `fp_drift_detected: false` recorded by a check that never ran — the
    # absence that used to render as "all classifiers healthy".
    LEGACY_ARCHIVE = """
name: drift_monitoring
status: completed
started_at: '2026-01-15T03:00:01'
completed_at: '2026-01-15T03:00:45'
current_step: null
run_id: '20260115_030001'
error: null
steps:
  check_fp_drift:
    name: check_fp_drift
    status: completed
    started_at: '2026-01-15T03:00:01'
    completed_at: '2026-01-15T03:00:30'
    error: null
    result:
      fp_drift_checked: true
context:
  dry_run: false
  fp_drift_detected: false
  any_drift_detected: false
  recommendation: No action needed - all classifiers healthy
"""

    def test_a_pre_verdict_archive_loads_without_error(self):
        data = yaml.safe_load(self.LEGACY_ARCHIVE)

        state = WorkflowState.from_dict(data)

        assert state.name == "drift_monitoring"
        assert state.status is WorkflowStatus.COMPLETED
        assert state.steps["check_fp_drift"].status is WorkflowStatus.COMPLETED

    def test_a_pre_verdict_archive_reports_no_health(self):
        """Back-compat must not manufacture a verdict for a run that had none.

        Loading is only half of AC4. The other half is that an old archive does
        not acquire a *healthy* reading it never earned — the file says
        "all classifiers healthy" in a free-text field, and the vocabulary must
        still resolve it to unknown.
        """
        state = WorkflowState.from_dict(yaml.safe_load(self.LEGACY_ARCHIVE))

        assert verdict_of(state.context, "fp_verdict") is HealthVerdict.UNKNOWN
        assert summarize(
            {"fp": verdict_of(state.context, "fp_verdict")}
        )["all_checked_healthy"] is False

    def test_a_pre_verdict_archive_survives_a_state_file_load(self, tmp_path):
        """The same file through `StateManager`, not just `from_dict`."""
        state_file = tmp_path / "state.yaml"
        state_file.write_text(
            yaml.dump({"workflows": {"drift_monitoring": yaml.safe_load(self.LEGACY_ARCHIVE)}})
        )

        manager = StateManager(state_file=state_file)

        assert manager.get_workflow("drift_monitoring") is not None, (
            "a pre-verdict archive was dropped by _load's bare except"
        )


class TestUnresolvedVerdictFailsTheRun:
    """AC2 end-to-end: unknown must not leave a COMPLETED-looking status."""

    @staticmethod
    def _build(name, steps, state_manager):
        class _Workflow(Workflow):
            pass

        _Workflow.name = name
        _Workflow.description = f"{name} test workflow"
        _Workflow.steps = [
            StepDefinition(name=n, description=n, handler=h) for n, h in steps
        ]
        WorkflowRegistry._workflows[name] = _Workflow
        return _Workflow(state_manager=state_manager)

    def test_a_check_that_produced_no_verdict_fails_the_workflow(
        self, state_manager, cleanup_registry
    ):
        """The whole point, end to end: no verdict, no green run."""
        workflow = self._build(
            "unresolved_run",
            [
                ("check", lambda wf, ctx: {"other_key": True}),
                ("gate", fail_on_unresolved_verdicts(("fp",))),
            ],
            state_manager,
        )

        state = workflow.run()

        assert state.status is WorkflowStatus.FAILED
        assert state.status is not WorkflowStatus.COMPLETED
        assert "No verdict was produced" in state.error

    def test_a_raising_check_never_reaches_the_gate(
        self, state_manager, cleanup_registry
    ):
        """The gate is NOT what catches a raising check — the base runner is.

        Worth a test because the opposite is easy to assume, and an earlier
        version of this file asserted it: `_execute_step` re-raises, so `run()`'s
        `except` finalizes the workflow and every later step, gate included, is
        left PENDING. AC5's "a check that raised yields unknown" is therefore a
        statement about `verdict_of` (a raise writes no verdict key), not about
        this gate — and a workflow that *swallows* its error is the case the
        gate exists for, covered by the test above.
        """

        def explode(workflow, context):
            raise RuntimeError("evidently blew up")

        workflow = self._build(
            "raised_run",
            [("check", explode), ("gate", fail_on_unresolved_verdicts(("fp",)))],
            state_manager,
        )

        state = workflow.run()

        assert state.status is WorkflowStatus.FAILED
        assert state.steps["check"].status is WorkflowStatus.FAILED
        assert state.steps["gate"].status is WorkflowStatus.PENDING
        assert "evidently blew up" in state.error
        # And the verdict a raise leaves behind is unknown, which is AC5's half.
        assert verdict_of(state.context, "fp_verdict") is HealthVerdict.UNKNOWN

    def test_explicit_verdicts_let_the_workflow_complete(
        self, state_manager, cleanup_registry
    ):
        """The gate must not fail a run that did produce verdicts.

        A gate that failed everything would satisfy the tests above while being
        useless, so pin the passing direction too.
        """
        workflow = self._build(
            "resolved_run",
            [
                ("check", lambda wf, ctx: {"fp_verdict": HealthVerdict.HEALTHY.value}),
                ("gate", fail_on_unresolved_verdicts(("fp",))),
            ],
            state_manager,
        )

        state = workflow.run()

        assert state.status is WorkflowStatus.COMPLETED
        assert state.context["verdicts_confirmed"] is True

    def test_a_degraded_verdict_is_a_result_not_an_unresolved_check(
        self, state_manager, cleanup_registry
    ):
        """Degraded means the check worked and found a problem.

        It is reported through the workflow's own alerting, not by failing the
        run at this gate — the gate is about checks that produced nothing.
        """
        workflow = self._build(
            "degraded_run",
            [
                ("check", lambda wf, ctx: {"fp_verdict": HealthVerdict.DEGRADED.value}),
                ("gate", fail_on_unresolved_verdicts(("fp",))),
            ],
            state_manager,
        )

        assert workflow.run().status is WorkflowStatus.COMPLETED

    def test_the_failed_gate_records_its_payload_on_the_step(
        self, state_manager, cleanup_registry
    ):
        """`StepFailure.context` reaches `StepState.result` (#73's contract).

        This is the per-step attribution #75/#76 read: the workflow context is a
        flat namespace several steps write, so "which step said this" would
        otherwise be lost to the archive.
        """
        workflow = self._build(
            "payload_run",
            [("gate", fail_on_unresolved_verdicts(("fp",)))],
            state_manager,
        )

        state = workflow.run()

        assert state.steps["gate"].result["unresolved_verdicts"] == ["fp"]
        assert state.steps["gate"].error is not None


class TestGateReasons:
    """The gate reports why a check is unresolved, not only which (D010.5)."""

    def test_a_recorded_reason_reaches_the_message_and_the_payload(self):
        handler = fail_on_unresolved_verdicts(("fp",))

        result = handler(None, {"fp_error": "KeyError: 'novelty_score'"})

        assert isinstance(result, StepFailure)
        assert "KeyError: 'novelty_score'" in result.error
        assert result.context["unresolved_reasons"]["fp"] == "KeyError: 'novelty_score'"

    def test_a_missing_reason_says_so_rather_than_rendering_empty(self):
        """`{c}_error` absent must not produce "fp: " with nothing after it.

        That exact shape — a failure line naming no cause — is what made the
        original incident unreadable in the logs.
        """
        result = fail_on_unresolved_verdicts(("fp",))(None, {})

        assert "no reason recorded" in result.error

    def test_reason_key_is_configurable_per_workflow(self):
        """Wave-3 workflows do not have to adopt drift's key convention."""
        handler = fail_on_unresolved_verdicts(
            ("feed",), verdict_key="{subject}_health", reason_key="{subject}_why"
        )

        result = handler(None, {"feed_why": "empty file"})

        assert "empty file" in result.error

    def test_verdict_key_is_honoured_and_can_pass_the_gate(self):
        """The passing direction, which is what actually pins `verdict_key`.

        Asserting only that a custom-key context FAILS proves nothing: the gate
        fails an absent verdict either way, so ignoring `verdict_key` entirely
        would pass such a test. This one fails if the template is ignored.
        """
        handler = fail_on_unresolved_verdicts(("feed",), verdict_key="{subject}_health")

        assert handler(None, {"feed_health": HealthVerdict.HEALTHY.value}) == {
            "verdicts_confirmed": True
        }

    def test_a_non_string_reason_is_coerced_before_it_reaches_the_archive(self):
        """A reason is whatever the workflow stored, so the gate must coerce it.

        An exception object or a `Path` under `<subject>_error` serializes
        through `yaml.dump` and only fails on the NEXT run's `safe_load`, taking
        all workflow state with it. Drift's reasons are always f-strings, so this
        guards the factory for #77/#78/#79 rather than a live path.
        """
        result = fail_on_unresolved_verdicts(("fp",))(
            None, {"fp_error": RuntimeError("KeyError: 'novelty_score'")}
        )

        assert result.context["unresolved_reasons"]["fp"] == "KeyError: 'novelty_score'"
        assert yaml.safe_load(yaml.dump(result.context)) == result.context


class TestGateConstructionRefusesVacuousConfigurations:
    """The factory rejects two shapes that would make the gate meaningless."""

    def test_an_empty_subject_list_is_refused(self):
        """A gate over no subjects would confirm nothing and report success.

        The same vacuous truth `summarize` refuses — caught at construction
        rather than at runtime, because `subjects` is fixed when the workflow
        class is defined.
        """
        with pytest.raises(ValueError, match="at least one subject"):
            fail_on_unresolved_verdicts(())

    def test_a_bare_string_subject_is_refused(self):
        """`str` satisfies `Sequence[str]` and iterates one subject per char.

        A single-subject workflow (#78's feed, #79's stage) is exactly where
        this would be written, and it would fail every run with subjects named
        `f`, `e`, `e`, `d`.
        """
        with pytest.raises(TypeError, match="bare string"):
            fail_on_unresolved_verdicts("feed")


class TestImportDirection:
    """`health` stays an import leaf so `base` can import it (D010.2).

    The cycle is live **today**, not waiting on a second adopter:
    `workflows/__init__` eagerly imports every workflow module and
    `drift_monitoring` already imports `health`, so adding the reverse edge
    breaks `import src.agent.health` immediately.

    Note what each test below is worth. The subprocess pair is a smoke check
    only — under a real cycle this *file* would fail to collect (it imports
    `workflows.base` at module scope), so those two would never get to run. The
    test with actual bite is the `ast` one: it is the only thing here that
    catches a lazy `from .workflows.base import ...` placed inside a function,
    which leaves module import clean and reintroduces the cycle at call time.
    """

    @pytest.mark.parametrize(
        "order",
        [
            "import src.agent.health, src.agent.workflows",
            "import src.agent.workflows, src.agent.health",
        ],
    )
    def test_both_import_orders_work(self, order):
        result = subprocess.run(
            [sys.executable, "-c", order],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"import order {order!r} failed — health is no longer a leaf:\n{result.stderr}"
        )

    def test_health_imports_nothing_from_the_agent_package(self):
        """Parse the imports rather than grep the text.

        Two reasons this is `ast` and not a substring scan. A scan matches the
        module's own prose — this file's docstrings discuss `import
        src.agent.health` at length — and, more importantly, `ast.walk` reaches
        imports nested inside functions, which a runtime import check would miss
        entirely. A lazy `from ..workflows.base import ...` inside a function
        body reintroduces the cycle at call time while leaving module import
        clean.
        """
        tree = ast.parse((REPO_ROOT / "src" / "agent" / "health.py").read_text())

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))

        offenders = {
            name
            for name in imported
            if name.startswith(".") or name.split(".")[0:2] == ["src", "agent"]
        }
        assert offenders == set(), (
            f"health.py must stay an import leaf, but imports {sorted(offenders)}"
        )


def test_health_summary_is_a_typed_dict_not_a_dataclass():
    """Guard the choice itself, in one line that names the requirement.

    A dataclass would in fact break several tests here — `summarize(...)[...]`
    is a subscript, so the partition tests would raise `TypeError`, and the
    round-trip test would fail on load. This is not the only guard; it is the
    one that says *why* in its name, so the next reader does not have to infer
    the constraint from a `TypeError` in an unrelated test.
    """
    assert issubclass(HealthSummary, dict)
