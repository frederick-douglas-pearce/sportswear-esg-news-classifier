"""Tests for agent workflows."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.agent.state import StateManager, WorkflowStatus
from src.agent.workflows.base import (
    StepDefinition,
    StepFailure,
    Workflow,
    WorkflowRegistry,
)


@pytest.fixture
def state_manager(tmp_path):
    """Create a fresh StateManager instance."""
    state_file = tmp_path / "state.yaml"
    return StateManager(state_file=state_file)


@pytest.fixture
def cleanup_registry():
    """Clean up registry after tests."""
    # Store original workflows
    original = dict(WorkflowRegistry._workflows)
    yield
    # Restore original workflows
    WorkflowRegistry._workflows = original


class TestStepDefinition:
    """Tests for StepDefinition dataclass."""

    def test_basic_step(self):
        """Test basic step definition."""
        handler = lambda w, c: {"result": True}
        step = StepDefinition(
            name="test_step",
            description="A test step",
            handler=handler,
        )

        assert step.name == "test_step"
        assert step.description == "A test step"
        assert step.handler is handler
        assert step.skip_on_dry_run is False
        assert step.requires_approval is False

    def test_step_with_options(self):
        """Test step with all options."""
        handler = lambda w, c: None
        step = StepDefinition(
            name="approval_step",
            description="Needs approval",
            handler=handler,
            skip_on_dry_run=True,
            requires_approval=True,
        )

        assert step.skip_on_dry_run is True
        assert step.requires_approval is True


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry."""

    def test_register_workflow(self, cleanup_registry):
        """Test registering a workflow."""

        @WorkflowRegistry.register
        class TestWorkflow(Workflow):
            name = "test_workflow"
            description = "A test workflow"
            steps = []

        assert "test_workflow" in WorkflowRegistry.list()
        assert WorkflowRegistry.get("test_workflow") is TestWorkflow

    def test_list_workflows(self, cleanup_registry):
        """Test listing registered workflows."""
        # Register multiple workflows
        @WorkflowRegistry.register
        class Workflow1(Workflow):
            name = "workflow1"
            description = "First"
            steps = []

        @WorkflowRegistry.register
        class Workflow2(Workflow):
            name = "workflow2"
            description = "Second"
            steps = []

        workflows = WorkflowRegistry.list()
        assert "workflow1" in workflows
        assert "workflow2" in workflows

    def test_get_unknown_workflow(self, cleanup_registry):
        """Test getting unknown workflow returns None."""
        result = WorkflowRegistry.get("nonexistent")
        assert result is None

    def test_create_workflow(self, cleanup_registry, state_manager):
        """Test creating workflow instance."""

        @WorkflowRegistry.register
        class TestWorkflow(Workflow):
            name = "test_create"
            description = "Test"
            steps = []

        workflow = WorkflowRegistry.create(
            name="test_create",
            state_manager=state_manager,
            dry_run=True,
        )

        assert isinstance(workflow, TestWorkflow)
        assert workflow.dry_run is True

    def test_create_unknown_workflow_raises(self, cleanup_registry, state_manager):
        """Test creating unknown workflow raises ValueError."""
        with pytest.raises(ValueError, match="Unknown workflow"):
            WorkflowRegistry.create("nonexistent", state_manager=state_manager)


class TestWorkflow:
    """Tests for Workflow base class."""

    @pytest.fixture
    def simple_workflow(self, state_manager, cleanup_registry):
        """Create a simple test workflow."""
        step_results = {}

        def step1_handler(workflow, context):
            step_results["step1"] = True
            return {"step1_result": "done"}

        def step2_handler(workflow, context):
            step_results["step2"] = context.get("step1_result")
            return {"step2_result": "complete"}

        @WorkflowRegistry.register
        class SimpleWorkflow(Workflow):
            name = "simple"
            description = "Simple test workflow"
            steps = [
                StepDefinition(
                    name="step1",
                    description="First step",
                    handler=step1_handler,
                ),
                StepDefinition(
                    name="step2",
                    description="Second step",
                    handler=step2_handler,
                ),
            ]

        return SimpleWorkflow(state_manager=state_manager), step_results

    def test_workflow_run_success(self, simple_workflow):
        """Test successful workflow execution."""
        workflow, step_results = simple_workflow

        result = workflow.run()

        assert result.status == WorkflowStatus.COMPLETED
        assert step_results["step1"] is True
        assert step_results["step2"] == "done"
        assert result.context.get("step2_result") == "complete"

    def test_workflow_dry_run(self, state_manager, cleanup_registry):
        """Test dry run mode skips configured steps."""
        executed_steps = []

        def regular_step(workflow, context):
            executed_steps.append("regular")
            return {}

        def skip_step(workflow, context):
            executed_steps.append("skipped")
            return {}

        @WorkflowRegistry.register
        class DryRunWorkflow(Workflow):
            name = "dryrun_test"
            description = "Test dry run"
            steps = [
                StepDefinition(
                    name="regular",
                    description="Regular step",
                    handler=regular_step,
                ),
                StepDefinition(
                    name="skipped",
                    description="Skipped in dry run",
                    handler=skip_step,
                    skip_on_dry_run=True,
                ),
            ]

        workflow = DryRunWorkflow(state_manager=state_manager, dry_run=True)
        result = workflow.run()

        assert result.status == WorkflowStatus.COMPLETED
        assert "regular" in executed_steps
        assert "skipped" not in executed_steps

    def test_workflow_step_failure(self, state_manager, cleanup_registry):
        """Test workflow handles step failure."""

        def failing_step(workflow, context):
            raise RuntimeError("Step failed!")

        @WorkflowRegistry.register
        class FailingWorkflow(Workflow):
            name = "failing"
            description = "Failing workflow"
            steps = [
                StepDefinition(
                    name="fail",
                    description="This fails",
                    handler=failing_step,
                ),
            ]

        workflow = FailingWorkflow(state_manager=state_manager)
        result = workflow.run()

        assert result.status == WorkflowStatus.FAILED
        assert "Step failed!" in result.error

    def test_workflow_pause_for_approval(self, state_manager, cleanup_registry):
        """Test workflow pauses for approval-required steps."""

        def step1(workflow, context):
            return {"done": True}

        def approval_step(workflow, context):
            return {"approved": True}

        @WorkflowRegistry.register
        class ApprovalWorkflow(Workflow):
            name = "approval"
            description = "Needs approval"
            steps = [
                StepDefinition(
                    name="step1",
                    description="First step",
                    handler=step1,
                ),
                StepDefinition(
                    name="approval",
                    description="Requires approval",
                    handler=approval_step,
                    requires_approval=True,
                ),
            ]

        workflow = ApprovalWorkflow(state_manager=state_manager)
        result = workflow.run()

        assert result.status == WorkflowStatus.PAUSED
        assert "approval" in result.context.get("pause_reason", "")

    def test_workflow_resume(self, state_manager, cleanup_registry):
        """Test resuming a paused workflow.

        Note: requires_approval steps are placeholders for manual work.
        The step handler is NOT executed - the user does the work manually,
        then resumes to continue with remaining steps.
        """
        executed_steps = []

        def step1(workflow, context):
            executed_steps.append("step1")
            return {}

        def step2(workflow, context):
            # This won't be called - requires_approval steps are manual
            executed_steps.append("step2")
            return {}

        def step3(workflow, context):
            executed_steps.append("step3")
            return {}

        @WorkflowRegistry.register
        class ResumeWorkflow(Workflow):
            name = "resume_test"
            description = "Resume test"
            steps = [
                StepDefinition(name="step1", description="1", handler=step1),
                StepDefinition(
                    name="step2",
                    description="2 - manual work",
                    handler=step2,
                    requires_approval=True,
                ),
                StepDefinition(name="step3", description="3", handler=step3),
            ]

        # Run until pause
        workflow = ResumeWorkflow(state_manager=state_manager)
        result = workflow.run()

        assert result.status == WorkflowStatus.PAUSED
        assert executed_steps == ["step1"]

        # Resume - step2 is skipped (manual work already done by user)
        resumed_result = workflow.resume()

        assert resumed_result.status == WorkflowStatus.COMPLETED
        # step2 handler is NOT called - it's a placeholder for manual work
        assert "step2" not in executed_steps
        assert "step3" in executed_steps
        assert executed_steps == ["step1", "step3"]

    def test_workflow_context_propagation(self, state_manager, cleanup_registry):
        """Test context is propagated between steps."""
        contexts = []

        def step1(workflow, context):
            contexts.append(dict(context))
            return {"from_step1": "value1"}

        def step2(workflow, context):
            contexts.append(dict(context))
            return {"from_step2": "value2"}

        @WorkflowRegistry.register
        class ContextWorkflow(Workflow):
            name = "context_test"
            description = "Context test"
            steps = [
                StepDefinition(name="step1", description="1", handler=step1),
                StepDefinition(name="step2", description="2", handler=step2),
            ]

        workflow = ContextWorkflow(state_manager=state_manager)
        result = workflow.run(context={"initial": "data"})

        # First step sees initial context
        assert contexts[0].get("initial") == "data"

        # Second step sees step1's output
        assert contexts[1].get("from_step1") == "value1"

        # Final context has all updates
        assert result.context.get("from_step1") == "value1"
        assert result.context.get("from_step2") == "value2"

    def test_step_names_property(self, state_manager, cleanup_registry):
        """Test step_names property."""

        @WorkflowRegistry.register
        class NamesWorkflow(Workflow):
            name = "names_test"
            description = "Names test"
            steps = [
                StepDefinition(name="a", description="A", handler=lambda w, c: None),
                StepDefinition(name="b", description="B", handler=lambda w, c: None),
                StepDefinition(name="c", description="C", handler=lambda w, c: None),
            ]

        workflow = NamesWorkflow(state_manager=state_manager)
        assert workflow.step_names == ["a", "b", "c"]


class TestDailyLabelingWorkflow:
    """Tests for DailyLabelingWorkflow."""

    def test_workflow_registered(self):
        """Test daily_labeling workflow is registered."""
        from src.agent.workflows import daily_labeling  # noqa: F401

        assert "daily_labeling" in WorkflowRegistry.list()

    def test_workflow_has_expected_steps(self):
        """Test workflow has expected steps."""
        from src.agent.workflows.daily_labeling import DailyLabelingWorkflow

        step_names = [s.name for s in DailyLabelingWorkflow.steps]
        assert "check_collection_status" in step_names
        assert "run_labeling" in step_names
        assert "check_labeling_quality" in step_names
        assert "generate_report" in step_names

    @patch("src.agent.workflows.daily_labeling.db")
    def test_check_collection_status(self, mock_db, state_manager, cleanup_registry):
        """Test check_collection_status step."""
        from src.agent.workflows.daily_labeling import check_collection_status

        # Mock database response
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        # Mock collection runs query
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(
            runs=5, fetched=100, scraped=95, failed=5
        )
        mock_session.execute.return_value = mock_result

        # Create minimal workflow for testing
        @WorkflowRegistry.register
        class TestWorkflow(Workflow):
            name = "test_collection"
            description = "Test"
            steps = []

        workflow = TestWorkflow(state_manager=state_manager)

        # Run step
        result = check_collection_status(workflow, {})

        assert result["collection_runs_24h"] == 5
        assert result["articles_fetched_24h"] == 100


class TestParseLabelingOutput:
    """Tests for _parse_labeling_output helper function."""

    def test_parse_basic_stats(self):
        """Test parsing basic labeling stats."""
        from src.agent.workflows.daily_labeling import _parse_labeling_output

        output = """
=== Labeling Results ===
Articles processed:     10
Articles labeled:       7
Articles skipped:       1
False positives:        2
Articles failed:        0
LLM API calls:          8
Estimated cost:         $0.1234
"""
        stats = _parse_labeling_output(output)

        assert stats["articles_processed"] == 10
        assert stats["articles_labeled"] == 7
        assert stats["articles_skipped"] == 1
        assert stats["false_positives"] == 2
        assert stats["articles_failed"] == 0
        assert stats["llm_calls"] == 8
        assert stats["estimated_cost_usd"] == 0.1234

    def test_parse_fp_classifier_stats(self):
        """Test parsing FP classifier pre-filter stats including cost savings."""
        from src.agent.workflows.daily_labeling import _parse_labeling_output

        output = """
=== Labeling Results ===
Articles processed:     20
Articles labeled:       10
Articles failed:        0
LLM API calls:          12
Estimated cost:         $0.5000

=== FP Classifier Pre-filter ===
FP classifier calls:    20
Skipped LLM:            8
Continued to LLM:       12
Est. LLM cost saved:    $0.0960
"""
        stats = _parse_labeling_output(output)

        assert stats["articles_processed"] == 20
        assert stats["fp_classifier_calls"] == 20
        assert stats["fp_skipped_llm"] == 8
        assert stats["fp_continued_llm"] == 12
        assert stats["fp_cost_saved_usd"] == 0.0960

    def test_parse_error_type_breakdown(self):
        """Test parsing error type breakdown section."""
        from src.agent.workflows.daily_labeling import _parse_labeling_output

        output = """
=== Labeling Results ===
Articles processed:     15
Articles failed:        5

Errors (5):

  Error Type Breakdown:
    connection: 3
    timeout: 2

  Individual Errors:
    - Article 1: connection error
"""
        stats = _parse_labeling_output(output)

        assert stats["articles_failed"] == 5
        assert "error_types" in stats
        assert stats["error_types"]["connection"] == 3
        assert stats["error_types"]["timeout"] == 2

    def test_report_surfaces_truncated_analysis(self):
        """A truncated analysis should be visible in the report, not look like 'not run' (#42)."""
        from src.agent.workflows.daily_labeling import generate_report

        workflow = MagicMock()
        workflow.name = "daily_labeling"
        context = {
            "llm_analysis_completed": False,
            "llm_analysis_error": "Analysis response truncated at max_tokens (8000); JSON incomplete",
            "llm_analysis_truncated": True,
        }

        result = generate_report(workflow, context)
        llm = result["report"]["llm_analysis"]

        assert llm["completed"] is False
        assert llm["truncated"] is True
        assert "truncated" in llm["error"].lower()

    def test_report_omits_error_when_analysis_not_run(self):
        """No error key when analysis was skipped rather than failed (#42)."""
        from src.agent.workflows.daily_labeling import generate_report

        workflow = MagicMock()
        workflow.name = "daily_labeling"
        context = {"llm_analysis_skipped": True}

        result = generate_report(workflow, context)
        llm = result["report"]["llm_analysis"]

        assert llm["completed"] is False
        assert "error" not in llm


class TestCheckLabelingQuality:
    """Tests for DB-sourced quality metrics (issue #81).

    The 2026-09-05 incident: labeling processed 12 articles and one failed, the
    script exited non-zero, the runner retried, the retry found nothing pending
    and printed zeros, and those zeros became the report — driving error_rate to
    0 so high_error_rate could never fire.
    """

    OUTCOMES_20260905 = {
        "articles_processed": 12,
        "articles_labeled": 5,
        "articles_skipped": 6,
        "articles_false_positive": 0,
        "articles_failed": 1,
        "articles_deduplicated": 1,
        "estimated_cost_usd": 0.0348,
        "run_count": 2,
        "incomplete_runs": 1,
        "any_run_incomplete": True,
        "counts_unavailable": False,
    }

    def test_uses_database_over_stdout_zeros(self):
        """The no-op retry's zeros must not win over the real run's counts."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "labeling_started_at": "2026-09-05T13:30:00+00:00",
            # What the retry printed.
            "labeling_output": {
                "articles_processed": 0,
                "articles_labeled": 0,
                "articles_failed": 0,
            },
        }

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes",
            return_value=self.OUTCOMES_20260905,
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["metrics_source"] == "database"
        assert result["articles_processed"] == 12
        assert result["articles_labeled"] == 5
        assert result["articles_failed"] == 1
        assert result["error_rate"] == pytest.approx(1 / 12)

    def test_flags_partial_failure(self):
        """A run that did work and also errored is flagged even at a low error rate."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {"labeling_started_at": "2026-09-05T13:30:00+00:00"}

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes",
            return_value=self.OUTCOMES_20260905,
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["partial_failure"] is True
        # 1/12 is under the 10% threshold, so this is the only signal available.
        assert result["high_error_rate"] is False

    def test_partial_failure_from_exit_code_alone(self):
        """Exit code 2 flags a partial failure even without run rows (dry run)."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "dry_run": True,
            "labeling_partial_failure": True,
            "labeling_output": {"articles_processed": 3, "articles_failed": 1},
        }

        result = check_labeling_quality(MagicMock(), context)

        assert result["partial_failure"] is True
        assert result["metrics_source"] == "stdout"

    def test_falls_back_to_stdout_when_no_run_rows(self):
        """A dry run writes no run row, so stdout remains the only source."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "dry_run": True,
            "labeling_started_at": "2026-09-05T13:30:00+00:00",
            "labeling_output": {
                "articles_processed": 10,
                "articles_labeled": 4,
                "articles_failed": 2,
            },
        }

        result = check_labeling_quality(MagicMock(), context)

        assert result["metrics_source"] == "stdout"
        assert result["articles_processed"] == 10
        assert result["error_rate"] == pytest.approx(0.2)

    def test_database_error_falls_back_instead_of_raising(self):
        """A metrics query failure must not take the workflow down."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "labeling_started_at": "2026-09-05T13:30:00+00:00",
            "labeling_output": {"articles_processed": 8, "articles_failed": 1},
        }

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes",
            side_effect=RuntimeError("connection refused"),
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["metrics_source"] == "stdout"
        assert result["articles_processed"] == 8

    def test_high_error_rate_still_fires(self):
        """The existing threshold behaviour is preserved on DB-sourced counts."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        outcomes = dict(self.OUTCOMES_20260905, articles_processed=10, articles_failed=5)
        context = {"labeling_started_at": "2026-09-05T13:30:00+00:00"}

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes", return_value=outcomes
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["high_error_rate"] is True

    def test_skipped_labeling_short_circuits(self):
        from src.agent.workflows.daily_labeling import check_labeling_quality

        result = check_labeling_quality(MagicMock(), {"labeling_skipped": True})

        assert result == {"quality_check_skipped": True}

    def test_fp_rate_uses_database_numerator(self):
        """FP rate must not divide a stdout numerator by a database denominator.

        Mixing them reproduces the #81 blind spot for high_fp_rate: on a retried
        run the numerator is the no-op's 0 while the denominator is the real 12.
        """
        from src.agent.workflows.daily_labeling import check_labeling_quality

        outcomes = dict(
            self.OUTCOMES_20260905, articles_processed=10, articles_false_positive=8
        )
        context = {
            "labeling_started_at": "2026-09-05T13:30:00+00:00",
            "labeling_output": {"false_positives": 0},  # the retry's stdout
        }

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes", return_value=outcomes
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["false_positives"] == 8
        assert result["fp_rate"] == pytest.approx(0.8)
        assert result["high_fp_rate"] is True

    def test_degraded_metrics_flagged_when_no_run_row(self):
        """A real run always writes a run row; falling back to stdout is a defect.

        This is how a deploy that precedes migration 007 would otherwise
        reproduce #81 — the run raises after doing the work, the retry finds
        nothing, and stdout zeros get published as if authoritative.
        """
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "dry_run": False,
            "labeling_started_at": "2026-09-05T13:30:00+00:00",
            "labeling_output": {"articles_processed": 0, "articles_failed": 0},
        }

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes", return_value=None
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["metrics_degraded"] is True
        assert result["partial_failure"] is True

    def test_dry_run_stdout_is_not_degraded(self):
        """A dry run legitimately writes no run row."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        context = {
            "dry_run": True,
            "labeling_output": {"articles_processed": 5, "articles_failed": 0},
        }

        result = check_labeling_quality(MagicMock(), context)

        assert result["metrics_degraded"] is False
        assert result["metrics_source"] == "stdout"

    def test_unfinished_run_is_not_read_as_zero(self):
        """A killed run's zeros are the absence of a result, not a result."""
        from src.agent.workflows.daily_labeling import check_labeling_quality

        timed_out = {
            "articles_processed": 0,
            "articles_labeled": 0,
            "articles_skipped": 0,
            "articles_false_positive": 0,
            "articles_failed": 0,
            "articles_deduplicated": 0,
            "estimated_cost_usd": 0.0,
            "run_count": 0,
            "incomplete_runs": 1,
            "any_run_incomplete": True,
            "counts_unavailable": True,
        }
        context = {"labeling_started_at": "2026-09-05T13:30:00+00:00"}

        with patch(
            "src.agent.workflows.daily_labeling._read_run_outcomes", return_value=timed_out
        ):
            result = check_labeling_quality(MagicMock(), context)

        assert result["partial_failure"] is True
        assert result["metrics_degraded"] is True


class TestReportUsesAuthoritativeCounts:
    """generate_report must publish the DB counts, not stdout's (issue #81)."""

    def test_report_reads_counts_from_context(self):
        from src.agent.workflows.daily_labeling import generate_report

        workflow = MagicMock()
        workflow.name = "daily_labeling"
        context = {
            # Written by check_labeling_quality from the database.
            "articles_processed": 12,
            "articles_labeled": 5,
            "articles_skipped": 6,
            "articles_failed": 1,
            "articles_deduplicated": 1,
            "estimated_cost_usd": 0.0348,
            "metrics_source": "database",
            "partial_failure": True,
            # The retry's stdout, which must not be used for article counts.
            "labeling_output": {
                "articles_processed": 0,
                "articles_labeled": 0,
                "fp_classifier_calls": 13,
            },
        }

        report = generate_report(workflow, context)["report"]

        assert report["labeling"]["articles_processed"] == 12
        assert report["labeling"]["articles_labeled"] == 5
        assert report["labeling"]["articles_failed"] == 1
        assert report["labeling"]["metrics_source"] == "database"
        assert report["quality"]["partial_failure"] is True
        # FP classifier counters are not persisted, so stdout is still correct.
        assert report["labeling"]["fp_classifier_calls"] == 13

    def test_skipped_and_false_positives_do_not_double_count(self):
        """skipped + FP + labeled + failed must reconcile against processed.

        `articles_skipped` used to be persisted as skipped + false positives
        while the report also showed FPs separately, so a 12-article run printed
        counts summing to 16.
        """
        from src.agent.workflows.daily_labeling import generate_report

        workflow = MagicMock()
        workflow.name = "daily_labeling"
        context = {
            "articles_processed": 12,
            "articles_labeled": 5,
            "articles_skipped": 2,
            "false_positives": 4,
            "articles_failed": 1,
            "articles_deduplicated": 0,
            "metrics_source": "database",
        }

        labeling = generate_report(workflow, context)["report"]["labeling"]

        assert (
            labeling["articles_labeled"]
            + labeling["articles_skipped"]
            + labeling["false_positives"]
            + labeling["articles_failed"]
        ) == labeling["articles_processed"]


class TestLlmAnalysisStatsSource:
    """The LLM narrative must not contradict the counts printed above it (#81)."""

    def test_analysis_receives_database_counts(self):
        from src.agent.workflows import daily_labeling

        context = {
            "articles_processed": 12,
            "articles_labeled": 5,
            "articles_skipped": 6,
            "false_positives": 0,
            "articles_failed": 1,
            "error_rate": 1 / 12,
            "fp_rate": 0.0,
            "metrics_source": "database",
            # The retry's stdout, which must not reach the model.
            "labeling_output": {
                "articles_processed": 0,
                "articles_labeled": 0,
                "articles_failed": 0,
            },
        }

        captured = {}

        class _Analyzer:
            def __init__(self, *a, **k):
                pass

            def analyze_labeling_results(self, stats=None, **kwargs):
                captured["stats"] = stats
                return MagicMock(success=False, error="stopped after capture")

        with patch.object(daily_labeling.agent_settings, "llm_analysis_enabled", True):
            with patch.object(daily_labeling.agent_settings, "anthropic_api_key", "k"):
                with patch.object(daily_labeling.agent_settings, "llm_error_threshold", 0.0):
                    with patch(
                        "src.agent.llm.LabelingAnalyzer", _Analyzer
                    ), patch(
                        "src.agent.llm.get_recent_labeling_samples",
                        return_value=([{"id": "a"}], [], []),
                    ):
                        daily_labeling.run_llm_analysis(MagicMock(), context)

        assert captured["stats"]["articles_processed"] == 12
        assert captured["stats"]["articles_labeled"] == 5
        assert captured["stats"]["articles_failed"] == 1


class TestDailyLabelingSendNotification:
    """Tests for the daily_labeling send_notification step (#51)."""

    def _context(self):
        return {
            "report": {
                "labeling": {"articles_processed": 3, "articles_labeled": 2},
                "collection": {},
                "quality": {},
            }
        }

    def test_raises_when_all_configured_channels_fail(self):
        """DNS down: email/webhook attempted and all fail -> raise so the run is FAILED."""
        from src.agent.workflows.daily_labeling import send_notification

        with patch(
            "src.agent.notifications.send_labeling_summary",
            return_value={"email": False, "webhook": False},
        ):
            with pytest.raises(RuntimeError, match="Notification delivery failed"):
                send_notification(MagicMock(), self._context())

    def test_does_not_raise_when_no_channels_configured(self):
        """Console fallback (nothing configured) is not a delivery failure."""
        from src.agent.workflows.daily_labeling import send_notification

        with patch(
            "src.agent.notifications.send_labeling_summary",
            return_value={"console": True},
        ):
            result = send_notification(MagicMock(), self._context())

        # Console fallback counts as delivered; the key point is it must not raise.
        assert result["channels"] == ["console"]

    def test_succeeds_when_one_channel_delivers(self):
        """At least one real channel delivering is success."""
        from src.agent.workflows.daily_labeling import send_notification

        with patch(
            "src.agent.notifications.send_labeling_summary",
            return_value={"email": True, "webhook": False},
        ):
            result = send_notification(MagicMock(), self._context())

        assert result["notification_sent"] is True
        assert result["channels"] == ["email"]


class TestWebsiteExportWorkflow:
    """Tests for WebsiteExportWorkflow."""

    def test_workflow_registered(self):
        """Test website_export workflow is registered."""
        from src.agent.workflows import website_export  # noqa: F401

        assert "website_export" in WorkflowRegistry.list()

    def test_workflow_has_expected_steps(self):
        """Test workflow has expected steps."""
        from src.agent.workflows.website_export import WebsiteExportWorkflow

        step_names = [s.name for s in WebsiteExportWorkflow.steps]
        assert "export_feeds" in step_names
        assert "validate_export" in step_names
        assert "commit_and_push" in step_names

    def test_commit_step_skipped_on_dry_run(self):
        """Test commit step is skipped in dry-run mode."""
        from src.agent.workflows.website_export import WebsiteExportWorkflow

        commit_step = next(
            s for s in WebsiteExportWorkflow.steps if s.name == "commit_and_push"
        )
        assert commit_step.skip_on_dry_run is True


def _script_result(
    command: list[str],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Any:
    """Build a ScriptResult-shaped object for patching run_script."""
    from src.agent.runner import ScriptResult

    return ScriptResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.0,
        started_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_website_settings(tmp_path):
    """Patch agent_settings with explicit attributes — no MagicMock auto-attrs."""
    with patch("src.agent.workflows.website_export.agent_settings") as m:
        m.website_repo_path = tmp_path
        m.website_expected_branch = "main"
        yield m


class TestPrepareWorktree:
    """Tests for the prepare_worktree step (branch assertion + FF pull)."""

    @patch("src.agent.workflows.website_export.run_script")
    def test_skips_when_website_repo_not_configured(self, mock_run_script):
        from src.agent.workflows.website_export import prepare_worktree

        with patch("src.agent.workflows.website_export.agent_settings") as m:
            m.website_repo_path = None
            result = prepare_worktree(MagicMock(), {})

        assert result == {"prepare_skipped": True, "reason": "website_repo_not_configured"}
        assert mock_run_script.call_count == 0

    @patch("src.agent.workflows.website_export.run_script")
    def test_skips_on_dry_run(self, mock_run_script, mock_website_settings):
        from src.agent.workflows.website_export import prepare_worktree

        result = prepare_worktree(MagicMock(), {"dry_run": True})

        assert result == {"prepare_skipped": True, "reason": "dry_run"}
        assert mock_run_script.call_count == 0

    @patch("src.agent.workflows.website_export.run_script")
    def test_fails_on_wrong_branch(self, mock_run_script, mock_website_settings):
        from src.agent.workflows.website_export import prepare_worktree

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"],
                stdout="feature/agentfluent-baseline-post\n",
            ),
            AssertionError("should not pull after wrong-branch abort"),
        ]

        result = prepare_worktree(MagicMock(), {})

        assert result["prepare_ready"] is False
        assert result["error"] == "wrong_branch"
        assert result["current_branch"] == "feature/agentfluent-baseline-post"
        assert result["expected_branch"] == "main"
        assert mock_run_script.call_count == 1

    @patch("src.agent.workflows.website_export.run_script")
    def test_fails_when_branch_check_command_fails(
        self, mock_run_script, mock_website_settings
    ):
        from src.agent.workflows.website_export import prepare_worktree

        mock_run_script.return_value = _script_result(
            ["git", "symbolic-ref", "--short", "HEAD"],
            exit_code=1,
            stderr="fatal: ref HEAD is not a symbolic ref\n",
        )

        result = prepare_worktree(MagicMock(), {})

        assert result["prepare_ready"] is False
        assert result["error"] == "git_branch_check_failed"
        assert "symbolic ref" in result["git_stderr"]

    @patch("src.agent.workflows.website_export.run_script")
    def test_fails_when_ff_pull_fails(self, mock_run_script, mock_website_settings):
        from src.agent.workflows.website_export import prepare_worktree

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(
                ["git", "pull", "--ff-only", "origin", "main"],
                exit_code=1,
                stderr="fatal: Not possible to fast-forward, aborting.\n",
            ),
        ]

        result = prepare_worktree(MagicMock(), {})

        assert result["prepare_ready"] is False
        assert result["error"] == "git_pull_ff_failed"
        assert "fast-forward" in result["git_stderr"]
        assert result["current_branch"] == "main"

    @patch("src.agent.workflows.website_export.run_script")
    def test_runs_BEFORE_export_writes_anything(
        self, mock_run_script, mock_website_settings
    ):
        """prepare_worktree must be cwd-pure: branch check + pull, no fs writes."""
        from src.agent.workflows.website_export import prepare_worktree

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(["git", "pull", "--ff-only", "origin", "main"]),
        ]

        result = prepare_worktree(MagicMock(), {})

        assert result["prepare_ready"] is True
        # All commands run via run_script, no direct file writes
        commands = [call.args[0] for call in mock_run_script.call_args_list]
        for cmd in commands:
            assert cmd[0] == "git"
        # FF pull was the second call
        assert commands[1] == ["git", "pull", "--ff-only", "origin", "main"]


class TestCommitAndPushGuards:
    """Tests for commit_and_push() after the prepare_worktree refactor."""

    @pytest.fixture
    def passing_context(self):
        return {
            "export_skipped": False,
            "validation_skipped": False,
            "validation_passed": True,
            "prepare_ready": True,
            "dry_run": False,
            "json_output": "/tmp/esg_news.json",
            "atom_output": "/tmp/esg_news.atom",
        }

    def test_skips_when_prepare_failed(self, passing_context):
        from src.agent.workflows.website_export import commit_and_push

        ctx = dict(passing_context, prepare_ready=False)
        result = commit_and_push(MagicMock(), ctx)

        assert result == {"git_skipped": True, "reason": "prepare_failed"}

    @patch("src.agent.workflows.website_export.run_script")
    def test_defense_in_depth_branch_check(
        self, mock_run_script, mock_website_settings, passing_context
    ):
        """If branch flipped between prepare_worktree and commit_and_push, abort."""
        from src.agent.workflows.website_export import commit_and_push

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"],
                stdout="feature/sneak-in\n",
            ),
            AssertionError("must not stage/commit after wrong-branch abort"),
        ]

        result = commit_and_push(MagicMock(), passing_context)

        assert result["git_success"] is False
        assert result["error"] == "wrong_branch"
        assert result["current_branch"] == "feature/sneak-in"

    @patch("src.agent.workflows.website_export.run_script")
    def test_fails_when_status_command_fails(
        self, mock_run_script, mock_website_settings, passing_context
    ):
        """git status failure must not be silently treated as no_changes."""
        from src.agent.workflows.website_export import commit_and_push

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(
                ["git", "status", "--porcelain", "--", "_data/esg_news.json",
                 "assets/feeds/esg_news.atom"],
                exit_code=128,
                stderr="fatal: Unable to create '.git/index.lock': File exists.\n",
            ),
            AssertionError("must not proceed after status failure"),
        ]

        result = commit_and_push(MagicMock(), passing_context)

        assert result["git_success"] is False
        assert result["error"] == "git_status_failed"
        assert "index.lock" in result["git_stderr"]

    @patch("src.agent.workflows.website_export.run_script")
    def test_skips_when_no_feed_changes(
        self, mock_run_script, mock_website_settings, passing_context
    ):
        """Empty status output scoped to feed files = no commit needed."""
        from src.agent.workflows.website_export import commit_and_push

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(
                ["git", "status", "--porcelain", "--", "_data/esg_news.json",
                 "assets/feeds/esg_news.atom"],
                stdout="",
            ),
        ]

        result = commit_and_push(MagicMock(), passing_context)

        assert result["git_skipped"] is True
        assert result["reason"] == "no_changes"

    @patch("src.agent.workflows.website_export.run_script")
    def test_pushes_with_explicit_refspec_on_happy_path(
        self, mock_run_script, mock_website_settings, passing_context
    ):
        from src.agent.workflows.website_export import commit_and_push

        mock_run_script.side_effect = [
            _script_result(  # defense-in-depth branch check
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(  # scoped status — changes present
                ["git", "status", "--porcelain", "--", "_data/esg_news.json",
                 "assets/feeds/esg_news.atom"],
                stdout=" M _data/esg_news.json\n",
            ),
            _script_result(["git", "add", "_data/esg_news.json",
                            "assets/feeds/esg_news.atom"]),
            _script_result(["git", "commit"]),
            _script_result(["git", "push"]),
        ]

        result = commit_and_push(MagicMock(), passing_context)

        assert result["git_success"] is True
        assert result["branch"] == "main"
        assert result["current_branch"] == "main"
        assert result["expected_branch"] == "main"

        push_call = mock_run_script.call_args_list[-1]
        assert push_call.args[0] == ["git", "push", "origin", "HEAD:main"]

    @patch("src.agent.workflows.website_export.run_script")
    def test_all_git_calls_pin_dry_run_false(
        self, mock_run_script, mock_website_settings, passing_context
    ):
        """Every run_script call inside commit_and_push must pass dry_run=False
        to prevent AGENT_DRY_RUN=true env from injecting --dry-run into git."""
        from src.agent.workflows.website_export import commit_and_push

        mock_run_script.side_effect = [
            _script_result(
                ["git", "symbolic-ref", "--short", "HEAD"], stdout="main\n"
            ),
            _script_result(
                ["git", "status", "--porcelain", "--", "_data/esg_news.json",
                 "assets/feeds/esg_news.atom"],
                stdout=" M _data/esg_news.json\n",
            ),
            _script_result(["git", "add"]),
            _script_result(["git", "commit"]),
            _script_result(["git", "push"]),
        ]

        commit_and_push(MagicMock(), passing_context)

        for call in mock_run_script.call_args_list:
            assert call.kwargs.get("dry_run") is False, (
                f"run_script call {call.args[0]!r} omitted dry_run=False"
            )


class TestSendErrorNotification:
    """Tests for send_error_notification: aggregation, formatting, and raise."""

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_silent_success_when_no_errors(self, mock_manager_cls):
        from src.agent.workflows.website_export import send_error_notification

        result = send_error_notification(
            MagicMock(), {"export_success": True, "validation_passed": True}
        )

        assert result == {"notification_sent": False, "reason": "no_errors"}
        mock_manager_cls.assert_not_called()

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_raises_workflow_error_on_failure(self, mock_manager_cls):
        """Failures must raise so Workflow.run marks status FAILED, not COMPLETED."""
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "git_success": False,
            "error": "wrong_branch",
            "current_branch": "feature/foo",
            "expected_branch": "main",
            # Wrong-branch aborts before validation runs, so the real flow marks
            # validation as skipped (not failed).
            "validation_skipped": True,
        }

        with pytest.raises(WorkflowError, match="failed with 1 error"):
            send_error_notification(MagicMock(), context)

        # Notification was still dispatched before the raise.
        mock_manager_cls.return_value.send.assert_called_once()

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_wrong_branch_message_includes_branch_and_worktree_hint(
        self, mock_manager_cls
    ):
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "git_success": False,
            "error": "wrong_branch",
            "current_branch": "feature/foo",
            "expected_branch": "main",
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "feature/foo" in sent.message
        assert "worktree" in sent.message.lower()

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_wrong_branch_notification_omits_feed_paths(self, mock_manager_cls):
        """When the only failure is wrong_branch, feed paths shouldn't appear
        in details — files were written locally but never pushed."""
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "git_success": False,
            "error": "wrong_branch",
            "current_branch": "feature/foo",
            "expected_branch": "main",
            "json_output": "/tmp/esg_news.json",
            "atom_output": "/tmp/esg_news.atom",
            # Wrong-branch aborts before validation runs (validation is skipped).
            "validation_skipped": True,
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "json_output" not in sent.details
        assert "atom_output" not in sent.details

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_pull_failure_message_does_NOT_assert_divergence(self, mock_manager_cls):
        """The pull-failed message should enumerate likely causes (network,
        auth, divergence, dirty tree) rather than asserting divergence."""
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "prepare_ready": False,
            "error": "git_pull_ff_failed",
            "current_branch": "main",
            "expected_branch": "main",
            "git_stderr": "fatal: unable to access 'https://github.com/...': "
            "Could not resolve host\n",
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        msg_lower = sent.message.lower()
        assert "network failure" in msg_lower
        assert "could not resolve host" in msg_lower

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_branch_check_failed_has_dedicated_message(self, mock_manager_cls):
        """git_branch_check_failed must not fall through to generic else."""
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "prepare_ready": False,
            "error": "git_branch_check_failed",
            "expected_branch": "main",
            "git_stderr": "fatal: ref HEAD is not a symbolic ref\n",
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "detached HEAD" in sent.message
        assert "symbolic ref" in sent.message

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_scorecard_failure_triggers_notification(self, mock_manager_cls):
        """Scorecard save failures must surface even when git push succeeded."""
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "export_success": True,
            "validation_passed": True,
            "scorecard_saved": False,
            "error": "DB connection refused",
            "git_success": True,
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "Scorecard snapshot save failed" in sent.message
        assert "DB connection refused" in sent.message

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_status_failed_includes_stderr_hint(self, mock_manager_cls):
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}

        context = {
            "git_success": False,
            "error": "git_status_failed",
            "current_branch": "main",
            "expected_branch": "main",
            "git_stderr": "fatal: Unable to create '.git/index.lock'\n",
        }

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "index.lock" in sent.message


class TestValidateExport:
    """Tests for the validate_export step (JSON syntax + Jekyll YAML check)."""

    def _write(self, tmp_path, raw: str) -> dict:
        path = tmp_path / "esg_news.json"
        path.write_text(raw, encoding="utf-8")
        return {"json_output": str(path), "atom_output": None}

    def test_counts_articles_in_dict_feed(self, tmp_path):
        """Feed top level is a dict; count its articles list (not len(dict))."""
        from src.agent.workflows.website_export import validate_export

        ctx = self._write(tmp_path, json.dumps({"articles": [{"id": "1"}, {"id": "2"}]}))
        result = validate_export(MagicMock(), ctx)

        assert result["validation_passed"] is True
        assert result["json_article_count"] == 2
        assert result["yaml_valid"] is True

    def test_fails_on_c1_control_chars(self, tmp_path):
        """Valid JSON containing C1 mojibake must fail (Jekyll/Psych rejects it)."""
        from src.agent.workflows.website_export import validate_export

        ctx = self._write(tmp_path, '{"articles": [{"t": "McDonald\x92s"}]}')
        result = validate_export(MagicMock(), ctx)

        assert result["validation_passed"] is False
        assert result["yaml_valid"] is False
        assert any("U+0092" in e for e in result["errors"])

    def test_fails_on_malformed_json_without_duplicate_yaml_error(self, tmp_path):
        from src.agent.workflows.website_export import validate_export

        ctx = self._write(tmp_path, "{not valid json")
        result = validate_export(MagicMock(), ctx)

        assert result["validation_passed"] is False
        assert any(e.startswith("JSON error") for e in result["errors"])
        assert not any(e.startswith("YAML error") for e in result["errors"])

    def test_skips_when_export_skipped(self):
        from src.agent.workflows.website_export import validate_export

        result = validate_export(MagicMock(), {"export_skipped": True})
        assert result == {"validation_skipped": True}


class TestSendErrorNotificationValidationDefault:
    """A missing validation_passed flag must be treated as failure, not success."""

    @patch("src.agent.workflows.website_export.NotificationManager")
    def test_missing_validation_flag_triggers_alert(self, mock_manager_cls):
        from src.agent.workflows.website_export import (
            WorkflowError,
            send_error_notification,
        )

        mock_manager_cls.return_value.send.return_value = {"email": True}
        # No validation_passed key, and not skipped -> should be flagged as failure.
        context = {"prepare_ready": True, "export_success": True}

        with pytest.raises(WorkflowError):
            send_error_notification(MagicMock(), context)

        sent = mock_manager_cls.return_value.send.call_args.args[0]
        assert "Validation failed" in sent.message


class TestStepFailureContract:
    """The StepFailure return contract (#73).

    A handler that catches its own error and returns a dict is recorded
    COMPLETED, so a run whose work failed archives as `status: completed,
    error: null`. These tests pin the mechanism that replaces that: the
    handler returns `StepFailure`, the base runner marks the step FAILED, and
    one `_finalize()` turns step outcomes into the workflow verdict.
    """

    @pytest.fixture(autouse=True)
    def isolated_history(self, tmp_path):
        """Keep these tests out of the real ~/.esg-agent/history archive."""
        from src.agent.config import AgentSettings

        history_dir = tmp_path / "history"
        history_dir.mkdir()
        with patch.object(AgentSettings, "history_dir", history_dir):
            self.history_dir = history_dir
            yield history_dir

    @staticmethod
    def _build(name, steps, state_manager):
        """Register and instantiate a workflow from (step_name, handler) pairs."""

        class _Workflow(Workflow):
            pass

        _Workflow.name = name
        _Workflow.description = f"{name} test workflow"
        _Workflow.steps = [
            StepDefinition(name=n, description=n, handler=h, **kw)
            for n, h, kw in ((s[0], s[1], s[2] if len(s) > 2 else {}) for s in steps)
        ]
        WorkflowRegistry._workflows[name] = _Workflow
        return _Workflow(state_manager=state_manager)

    # --- AC1: a returned failure marks the workflow FAILED --------------------

    def test_returned_failure_fails_the_workflow(self, state_manager, cleanup_registry):
        """AC1: no raise, and the workflow is FAILED with a non-empty error."""
        workflow = self._build(
            "sf_ac1",
            [
                ("first", lambda w, c: {"ok": True}),
                ("middle", lambda w, c: StepFailure(error="labeling produced nothing")),
                ("last", lambda w, c: {"last_ran": True}),
            ],
            state_manager,
        )

        result = workflow.run()

        assert result.status == WorkflowStatus.FAILED
        assert result.error
        assert "labeling produced nothing" in result.error

    def test_failure_detail_is_recorded_on_the_step(
        self, state_manager, cleanup_registry
    ):
        """The contract clause five downstream stories read (D009 point 2).

        `error` is always on the step, and `context` is recorded as that step's
        result as well as merged into the workflow context. Per-step attribution
        is the point: the workflow context is a flat namespace several steps
        write the same keys into, so #76's archive audit cannot tell from the
        context alone which step contributed which key.
        """
        workflow = self._build(
            "sf_detail",
            [("only", lambda w, c: StepFailure(error="boom", context={"k": "v"}))],
            state_manager,
        )

        result = workflow.run()

        step = result.steps["only"]
        assert step.status == WorkflowStatus.FAILED
        assert step.error == "boom"
        assert step.result == {"k": "v"}, "context must survive as per-step detail"
        assert result.context.get("k") == "v", "and must still reach the context"

    def test_failure_with_no_context_records_no_result(
        self, state_manager, cleanup_registry
    ):
        """An empty context is recorded as None, not as an empty dict.

        Distinguishes "this step carried no payload" from "{}", and matches a
        raising step, which has no payload to record.
        """
        workflow = self._build(
            "sf_nocontext",
            [("only", lambda w, c: StepFailure(error="boom"))],
            state_manager,
        )

        step = workflow.run().steps["only"]
        assert step.status == WorkflowStatus.FAILED
        assert step.result is None

    def test_raising_step_records_no_result(self, state_manager, cleanup_registry):
        """A step that fails by raising has no payload -- result stays None."""

        def boom(workflow, context):
            raise RuntimeError("exploded")

        workflow = self._build("sf_raise_result", [("only", boom)], state_manager)

        step = workflow.run().steps["only"]
        assert step.status == WorkflowStatus.FAILED
        assert step.error == "exploded"
        assert step.result is None

    def test_failure_context_reaches_downstream_steps(
        self, state_manager, cleanup_registry
    ):
        """StepFailure.context fully replaces the dict return for context."""
        seen = {}

        def failing(workflow, context):
            return StepFailure(error="no", context={"rows": 0, "reason": "dns"})

        def later(workflow, context):
            seen.update(context)
            return {}

        workflow = self._build(
            "sf_ctx", [("failing", failing), ("later", later)], state_manager
        )
        result = workflow.run()

        assert seen.get("rows") == 0
        assert seen.get("reason") == "dns"
        assert result.context.get("reason") == "dns"

    def test_remaining_steps_still_run_after_a_returned_failure(
        self, state_manager, cleanup_registry
    ):
        """Continue, not halt -- what aggregate-then-notify steps depend on."""
        ran = []

        workflow = self._build(
            "sf_continue",
            [
                ("a", lambda w, c: ran.append("a")),
                ("b", lambda w, c: StepFailure(error="b failed")),
                ("c", lambda w, c: ran.append("c")),
            ],
            state_manager,
        )
        result = workflow.run()

        assert ran == ["a", "c"], "a returned failure must not halt the loop"
        assert result.steps["c"].status == WorkflowStatus.COMPLETED
        assert result.status == WorkflowStatus.FAILED

    # --- AC2: the existing raise path is unchanged ----------------------------

    def test_raising_handler_still_fails_step_and_workflow(
        self, state_manager, cleanup_registry
    ):
        """AC2: regression guard on the pre-existing channel."""

        def boom(workflow, context):
            raise RuntimeError("exploded")

        workflow = self._build(
            "sf_raise", [("boom", boom), ("never", lambda w, c: {})], state_manager
        )
        result = workflow.run()

        assert result.status == WorkflowStatus.FAILED
        assert result.steps["boom"].status == WorkflowStatus.FAILED
        assert result.steps["boom"].error == "exploded"
        assert "exploded" in result.error
        # A raise still aborts the loop, unlike a returned failure.
        assert result.steps["never"].status == WorkflowStatus.PENDING

    def test_exception_after_a_returned_failure_reports_both(
        self, state_manager, cleanup_registry
    ):
        """Both channels in one run: the summary must not lose the first.

        Before the shared _finalize(), the except branch set the workflow error
        to str(e) alone, so an earlier StepFailure vanished from the record.
        """

        def boom(workflow, context):
            raise RuntimeError("second failure")

        workflow = self._build(
            "sf_both",
            [
                ("returned", lambda w, c: StepFailure(error="first failure")),
                ("raised", boom),
            ],
            state_manager,
        )
        result = workflow.run()

        assert result.status == WorkflowStatus.FAILED
        assert "first failure" in result.error
        assert "second failure" in result.error

    def test_exception_outside_any_step_still_reaches_the_workflow_error(
        self, state_manager, cleanup_registry
    ):
        """An exception matching no failed step is added, not dropped."""
        workflow = self._build("sf_outside", [("only", lambda w, c: {})], state_manager)
        workflow.run()

        # Re-finalize with an exception no step recorded -- the shape of a
        # failure raised by the runner itself rather than by a handler.
        workflow._finalize(exception=RuntimeError("runner blew up"))
        result = state_manager.get_workflow("sf_outside")

        assert result.status == WorkflowStatus.FAILED
        assert "runner blew up" in result.error

    # --- AC3: the resume path honors the same rule ----------------------------

    def test_returned_failure_on_resume_fails_the_workflow(
        self, state_manager, cleanup_registry
    ):
        """AC3: resume() had no NON-EXCEPTION failure branch before this change.

        It completed the run when every step had completed and did nothing
        otherwise, so a resumed run carrying a failed step was left RUNNING and,
        because the archive is written only by complete_workflow/fail_workflow,
        never archived. A resumed step that *raised* was always caught by
        resume()'s except and failed there, so this gap was latent until
        StepFailure made a non-raising failure possible -- which is why this
        test has to use StepFailure to reach it at all.
        """
        workflow = self._build(
            "sf_resume",
            [
                ("first", lambda w, c: {}),
                ("approval", lambda w, c: {}, {"requires_approval": True}),
                ("after", lambda w, c: StepFailure(error="post-resume failure")),
            ],
            state_manager,
        )

        paused = workflow.run()
        assert paused.status == WorkflowStatus.PAUSED

        resumed = workflow.resume()

        assert resumed.status == WorkflowStatus.FAILED
        assert "post-resume failure" in resumed.error
        assert list(self.history_dir.glob("sf_resume_*.yaml")), (
            "a failed resume must be archived"
        )

    def test_resume_reaches_the_same_verdict_as_run(
        self, state_manager, cleanup_registry
    ):
        """Drift guard on the shared _finalize() (D009 point 4).

        run() and resume() must not carry two copies of this logic. If they are
        ever forked -- e.g. resume() regains a bare 'Not all steps completed' --
        the error text diverges and this fails.
        """

        def fail(workflow, context):
            return StepFailure(error="identical failure")

        via_run = self._build("sf_same_run", [("only", fail)], state_manager).run()

        resumed_wf = self._build(
            "sf_same_resume",
            [
                ("approval", lambda w, c: {}, {"requires_approval": True}),
                ("only", fail),
            ],
            state_manager,
        )
        resumed_wf.run()
        via_resume = resumed_wf.resume()

        assert via_run.status == via_resume.status == WorkflowStatus.FAILED
        assert via_run.error == via_resume.error == (
            "Step 'only' failed: identical failure"
        )

    def test_approval_pause_is_not_a_failure(self, state_manager, cleanup_registry):
        """The PAUSED guard: waiting on a human is not a failed run."""
        workflow = self._build(
            "sf_pause",
            [
                ("first", lambda w, c: {}),
                ("approval", lambda w, c: {}, {"requires_approval": True}),
                ("after", lambda w, c: {}),
            ],
            state_manager,
        )
        result = workflow.run()

        assert result.status == WorkflowStatus.PAUSED
        assert result.error is None
        assert not list(self.history_dir.glob("sf_pause_*.yaml"))

    # --- AC4: the two hand-rolled workarounds, re-expressed --------------------

    def test_website_export_shape_aggregates_and_fails(
        self, state_manager, cleanup_registry
    ):
        """AC4: the aggregate-then-notify terminal step, via StepFailure.

        This is the shape of website_export.send_error_notification: earlier
        steps record failures in context and keep going, and a terminal step
        aggregates them.

        What it does and does not establish. The base runner is real, and the
        aggregation genuinely reads context written by a *failed* earlier step,
        which is only possible because a returned failure does not halt the
        loop. But `notify` below is a hand-written stand-in, not
        `website_export.send_error_notification` -- that function is not
        migrated by #73 and is not imported here. So this demonstrates that the
        mechanism supports the shape; it does not observe the real function, and
        nothing here would fail if that function turned out to be unmigratable.
        """
        notified = {}

        def export(workflow, context):
            return StepFailure(
                error="export failed", context={"export_success": False}
            )

        def commit(workflow, context):
            return {"git_success": True}

        def notify(workflow, context):
            errors = []
            if context.get("export_success") is False:
                errors.append("Export failed")
            if context.get("git_success") is False:
                errors.append("Git operation failed")
            notified["errors"] = errors
            if errors:
                return StepFailure(
                    error=f"Website export failed with {len(errors)} error(s): "
                    + ", ".join(errors),
                    context={"notification_sent": True},
                )
            return {"notification_sent": False}

        workflow = self._build(
            "sf_export_shape",
            [("export", export), ("commit", commit), ("notify", notify)],
            state_manager,
        )
        result = workflow.run()

        # The terminal step saw the failed step's context, which is only true
        # because a returned failure does not halt the loop.
        assert notified["errors"] == ["Export failed"]
        assert result.steps["commit"].status == WorkflowStatus.COMPLETED
        assert result.status == WorkflowStatus.FAILED
        assert "export failed" in result.error
        assert "Website export failed with 1 error(s)" in result.error
        assert result.context.get("notification_sent") is True

    def test_daily_labeling_shape_fails_when_labeling_failed(
        self, state_manager, cleanup_registry
    ):
        """AC4: the gap the current daily_labeling workaround leaves open.

        daily_labeling.send_notification raises only when every notification
        channel fails; it says nothing about whether labeling worked. So a
        failed labeling run whose email was delivered archives as completed --
        three such runs exist in the real history (a fourth failed run predates
        that guard). Expressed via StepFailure the run fails, and the
        notification is still sent.

        As with the website_export-shape test above, the handlers here are
        stand-ins: the real send_notification is not migrated by #73.
        """
        sent = []

        def run_labeling(workflow, context):
            return StepFailure(
                error="labeling failed: [Errno 2] No such file or directory: 'uv'",
                context={"labeling_success": False, "articles_labeled": 0},
            )

        def send_notification(workflow, context):
            sent.append(context.get("labeling_success"))
            return {"notification_sent": True, "channels": ["email"]}

        workflow = self._build(
            "sf_labeling_shape",
            [("run_labeling", run_labeling), ("send_notification", send_notification)],
            state_manager,
        )
        result = workflow.run()

        assert sent == [False], "the notification step must still run"
        assert result.context.get("notification_sent") is True
        assert result.status == WorkflowStatus.FAILED
        assert "No such file or directory" in result.error

    # --- AC5: the archive records the failure ---------------------------------

    def test_failed_middle_step_archives_as_failed(
        self, state_manager, cleanup_registry
    ):
        """AC5: the regression the 230 archived false-success runs describe."""
        workflow = self._build(
            "sf_archive",
            [
                ("first", lambda w, c: {"ok": True}),
                ("middle", lambda w, c: StepFailure(error="middle step failed")),
                ("last", lambda w, c: {"ok": True}),
            ],
            state_manager,
        )
        workflow.run()

        archives = list(self.history_dir.glob("sf_archive_*.yaml"))
        assert len(archives) == 1
        archived = yaml.safe_load(archives[0].read_text())

        assert archived["status"] == "failed"
        assert archived["error"]
        assert "middle step failed" in archived["error"]
        assert archived["steps"]["middle"]["status"] == "failed"

    # --- the error summary ----------------------------------------------------

    def test_summary_names_every_failed_step(self, state_manager, cleanup_registry):
        """Continue semantics means several steps can fail in one run."""
        workflow = self._build(
            "sf_multi",
            [
                ("one", lambda w, c: StepFailure(error="first problem")),
                ("two", lambda w, c: {}),
                ("three", lambda w, c: StepFailure(error="third problem")),
            ],
            state_manager,
        )
        result = workflow.run()

        assert "Step 'one' failed: first problem" in result.error
        assert "Step 'three' failed: third problem" in result.error
        assert "two" not in result.error

    def test_summary_is_bounded_but_step_error_is_not(
        self, state_manager, cleanup_registry
    ):
        """A step error can be a captured stderr tail; the digest is capped."""
        long_error = "E" * 4000
        workflow = self._build(
            "sf_bounded",
            [("only", lambda w, c: StepFailure(error=long_error))],
            state_manager,
        )
        result = workflow.run()

        assert len(result.error) < len(long_error)
        assert "truncated" in result.error
        # The full text is not lost -- it stays on the step.
        assert result.steps["only"].error == long_error

    def test_fallback_error_when_no_step_recorded_one(
        self, state_manager, cleanup_registry
    ):
        """workflow.error is never empty, even with nothing to echo."""
        workflow = self._build("sf_fallback", [("only", lambda w, c: {})], state_manager)
        workflow.run()

        # Force the no-failed-step, not-all-completed shape directly.
        state = state_manager.get_workflow("sf_fallback")
        state.steps["only"].status = WorkflowStatus.PENDING
        workflow._finalize()

        assert state_manager.get_workflow("sf_fallback").error == (
            "Not all steps completed"
        )

    # --- regressions the code review caught -----------------------------------

    def test_a_failed_step_is_not_masked_by_a_later_pause(
        self, state_manager, cleanup_registry
    ):
        """A pause must not swallow a failure.

        `_finalize` checked `all_completed`, then returned early on PAUSED
        without asking whether any step had FAILED. So a step returning
        StepFailure followed by any pausing step archived nowhere, reported
        `paused` with error None, and exited 0 -- a failed run that reads as
        waiting for a human, which is this epic's own defect class.

        Reachable the moment a workflow with an approval step adopts the
        contract: `model_training.notify_and_pause` pauses unconditionally,
        immediately after the step #79 migrates.
        """
        workflow = self._build(
            "sf_fail_then_pause",
            [
                ("export", lambda w, c: StepFailure(error="export failed")),
                ("approval", lambda w, c: {}, {"requires_approval": True}),
                ("later", lambda w, c: {}),
            ],
            state_manager,
        )

        result = workflow.run()

        assert result.status == WorkflowStatus.FAILED, (
            "a FAILED step must beat a pause"
        )
        assert "export failed" in result.error
        assert list(self.history_dir.glob("sf_fail_then_pause_*.yaml")), (
            "and the run must be archived, not left unrecorded"
        )

    # The other half of that guard -- an ordinary approval pause must still
    # pause -- is `test_approval_pause_is_not_a_failure` above, which asserts
    # the same thing plus the absence of an archive. Note what neither test
    # can do: dropping the `not any_failed` qualifier changes behaviour ONLY
    # when a step has failed, so no clean-pause test can detect it. The
    # fail-then-pause test above is the guard for that qualifier.

    def test_a_missing_step_record_fails_the_run_instead_of_escaping(
        self, state_manager, cleanup_registry
    ):
        """The finalizer must never be the thing that leaves a run RUNNING.

        `resume()` loads state persisted by an earlier process, so a step added
        to the workflow class since that run started has no record. Indexing
        `steps[name]` raised KeyError from inside `_finalize` -- including from
        the `except` handler that calls it -- so the exception escaped `resume()`
        entirely, leaving the run RUNNING and unarchived and every later
        `agent continue` dead with "Workflow is not paused".

        On main this input ended FAILED and archived, so the unified finalizer
        must not be worse.
        """
        workflow = self._build(
            "sf_missing",
            [
                ("first", lambda w, c: {}),
                ("approval", lambda w, c: {}, {"requires_approval": True}),
            ],
            state_manager,
        )
        assert workflow.run().status == WorkflowStatus.PAUSED

        # The workflow definition gains a step while the run is paused.
        workflow.steps = list(workflow.steps) + [
            StepDefinition(name="added_later", description="new", handler=lambda w, c: {})
        ]

        result = workflow.resume()

        assert result.status == WorkflowStatus.FAILED
        assert "added_later" in result.error
        assert list(self.history_dir.glob("sf_missing_*.yaml")), (
            "the run must still be archived"
        )

    def test_a_missing_step_record_is_named_on_the_non_exception_path(
        self, state_manager, cleanup_registry
    ):
        """The other route to a missing record, which reaches _finalize cleanly.

        In the test above the added step is *after* `current_step`, so
        `_execute_step` runs it and `start_step` raises before `_finalize` is
        reached -- that exercises the `except` route, and its assertion on the
        error text is satisfied by the ValueError alone. Here the added step
        sits *before* `current_step`, so it is never in `remaining_steps`, never
        executed, and nothing raises: `_finalize()` is entered with no exception
        and a step that has no record. This is the case that pins
        `_failure_summary`'s "has no recorded state" branch -- deleting that
        branch leaves the test above green but makes this one fail with the
        bare fallback.
        """
        workflow = self._build(
            "sf_missing_clean",
            [
                ("first", lambda w, c: {}),
                ("approval", lambda w, c: {}, {"requires_approval": True}),
            ],
            state_manager,
        )
        assert workflow.run().status == WorkflowStatus.PAUSED

        # A step is inserted BEFORE the pause point while the run is paused.
        workflow.steps = [
            workflow.steps[0],
            StepDefinition(name="inserted", description="new", handler=lambda w, c: {}),
            workflow.steps[1],
        ]

        result = workflow.resume()

        assert result.status == WorkflowStatus.FAILED
        assert "Step 'inserted' has no recorded state" in result.error
        assert "the workflow definition changed" in result.error
        assert list(self.history_dir.glob("sf_missing_clean_*.yaml"))

    def test_raising_on_the_resume_path_fails_and_archives(
        self, state_manager, cleanup_registry
    ):
        """`resume()`'s except branch, which nothing else covers.

        Reverting `resume()`'s `self._finalize(exception=e)` to a bare
        `fail_workflow(self.name, str(e))` passes every other test in this
        class -- exactly the run/resume fork the shared finalizer exists to
        prevent, on the path AC3 names.
        """

        def boom(workflow, context):
            raise RuntimeError("resumed step exploded")

        workflow = self._build(
            "sf_resume_raise",
            [
                ("approval", lambda w, c: {}, {"requires_approval": True}),
                ("after", boom),
            ],
            state_manager,
        )
        assert workflow.run().status == WorkflowStatus.PAUSED

        result = workflow.resume()

        assert result.status == WorkflowStatus.FAILED
        assert result.steps["after"].status == WorkflowStatus.FAILED
        # The shared summary shape, not the bare str(e) the old branch produced.
        assert result.error == "Step 'after' failed: resumed step exploded"
        assert list(self.history_dir.glob("sf_resume_raise_*.yaml"))

    def test_a_raised_exception_is_not_reported_twice(
        self, state_manager, cleanup_registry
    ):
        """Pins the dedup guard: a step's own raise appears once, not twice.

        Deleting the `any(step.error == raw ...)` check leaves every other test
        passing, because they use substring membership.
        """

        def boom(workflow, context):
            raise RuntimeError("single failure")

        workflow = self._build("sf_dedup", [("only", boom)], state_manager)

        result = workflow.run()

        assert result.error == "Step 'only' failed: single failure"
        assert "Workflow error:" not in result.error

    def test_an_exception_with_no_message_still_names_its_type(
        self, state_manager, cleanup_registry
    ):
        """`raise RuntimeError()` must not produce an error that names nothing."""

        def boom(workflow, context):
            raise RuntimeError()

        workflow = self._build("sf_empty_exc", [("only", boom)], state_manager)

        result = workflow.run()

        assert result.error
        assert "RuntimeError" in result.error

    def test_the_summary_cap_is_per_step_not_per_summary(
        self, state_manager, cleanup_registry
    ):
        """Every failed step keeps its own budget.

        A cap applied to the joined summary instead would silently drop the
        later steps' names, contradicting
        `test_summary_names_every_failed_step`.
        """
        long_a, long_b = "A" * 4000, "B" * 4000
        workflow = self._build(
            "sf_percap",
            [
                ("one", lambda w, c: StepFailure(error=long_a)),
                ("two", lambda w, c: StepFailure(error=long_b)),
            ],
            state_manager,
        )

        result = workflow.run()

        assert "Step 'one' failed: " + "A" * 500 in result.error
        assert "Step 'two' failed: " + "B" * 500 in result.error
        assert result.error.count("truncated") == 2
