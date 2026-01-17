"""Tests for agent workflows."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent.state import StateManager, WorkflowStatus
from src.agent.workflows.base import StepDefinition, Workflow, WorkflowRegistry


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
