"""Tests for agent workflows."""

import json
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
