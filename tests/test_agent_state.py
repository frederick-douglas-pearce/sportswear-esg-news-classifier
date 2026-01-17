"""Tests for agent state management."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.agent.state import StateManager, StepState, WorkflowState, WorkflowStatus


@pytest.fixture
def state_file(tmp_path):
    """Create a temporary state file path."""
    return tmp_path / "state.yaml"


@pytest.fixture
def state_manager(state_file):
    """Create a fresh StateManager instance."""
    return StateManager(state_file=state_file)


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self):
        """Test all status enum values."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"


class TestStepState:
    """Tests for StepState dataclass."""

    def test_default_values(self):
        """Test StepState default values."""
        step = StepState(name="test_step")

        assert step.name == "test_step"
        assert step.status == WorkflowStatus.PENDING
        assert step.started_at is None
        assert step.completed_at is None
        assert step.error is None
        assert step.result is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        step = StepState(
            name="test_step",
            status=WorkflowStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            result={"key": "value"},
        )

        data = step.to_dict()

        assert data["name"] == "test_step"
        assert data["status"] == "completed"
        assert data["started_at"] == now.isoformat()
        assert data["completed_at"] == now.isoformat()
        assert data["result"] == {"key": "value"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "name": "test_step",
            "status": "running",
            "started_at": now.isoformat(),
            "completed_at": None,
            "error": None,
            "result": None,
        }

        step = StepState.from_dict(data)

        assert step.name == "test_step"
        assert step.status == WorkflowStatus.RUNNING
        assert step.started_at.isoformat() == now.isoformat()

    def test_from_dict_with_missing_fields(self):
        """Test from_dict with minimal data."""
        data = {"name": "minimal_step"}

        step = StepState.from_dict(data)

        assert step.name == "minimal_step"
        assert step.status == WorkflowStatus.PENDING


class TestWorkflowState:
    """Tests for WorkflowState dataclass."""

    def test_default_values(self):
        """Test WorkflowState default values."""
        workflow = WorkflowState(name="test_workflow")

        assert workflow.name == "test_workflow"
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.started_at is None
        assert workflow.current_step is None
        assert workflow.steps == {}
        assert workflow.context == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        workflow = WorkflowState(
            name="test_workflow",
            status=WorkflowStatus.RUNNING,
            started_at=now,
            current_step="step1",
            steps={"step1": StepState(name="step1")},
            context={"key": "value"},
            run_id="20260115_120000",
        )

        data = workflow.to_dict()

        assert data["name"] == "test_workflow"
        assert data["status"] == "running"
        assert data["started_at"] == now.isoformat()
        assert data["current_step"] == "step1"
        assert "step1" in data["steps"]
        assert data["context"] == {"key": "value"}
        assert data["run_id"] == "20260115_120000"

    def test_from_dict(self):
        """Test creation from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "name": "test_workflow",
            "status": "completed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "current_step": None,
            "steps": {
                "step1": {
                    "name": "step1",
                    "status": "completed",
                    "started_at": now.isoformat(),
                    "completed_at": now.isoformat(),
                }
            },
            "context": {"result": True},
            "run_id": "20260115_120000",
        }

        workflow = WorkflowState.from_dict(data)

        assert workflow.name == "test_workflow"
        assert workflow.status == WorkflowStatus.COMPLETED
        assert "step1" in workflow.steps
        assert workflow.steps["step1"].status == WorkflowStatus.COMPLETED


class TestStateManager:
    """Tests for StateManager class."""

    def test_initialization_creates_empty_state(self, state_manager, state_file):
        """Test that StateManager initializes with empty state."""
        assert state_manager.list_workflows() == []
        assert not state_file.exists()  # Not created until first save

    def test_create_workflow(self, state_manager):
        """Test creating a new workflow."""
        workflow = state_manager.create_workflow(
            name="test_workflow",
            steps=["step1", "step2", "step3"],
            context={"key": "value"},
        )

        assert workflow.name == "test_workflow"
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.started_at is not None
        assert workflow.run_id is not None
        assert len(workflow.steps) == 3
        assert workflow.context == {"key": "value"}

    def test_create_workflow_persists_to_file(self, state_manager, state_file):
        """Test that creating workflow saves to file."""
        state_manager.create_workflow(name="test", steps=["step1"])

        assert state_file.exists()

        with open(state_file) as f:
            data = yaml.safe_load(f)

        assert "workflows" in data
        assert "test" in data["workflows"]

    def test_start_workflow(self, state_manager):
        """Test starting a workflow."""
        state_manager.create_workflow(name="test", steps=["step1"])
        workflow = state_manager.start_workflow("test")

        assert workflow.status == WorkflowStatus.RUNNING
        assert workflow.started_at is not None

    def test_start_workflow_not_found(self, state_manager):
        """Test starting non-existent workflow raises error."""
        with pytest.raises(ValueError, match="not found"):
            state_manager.start_workflow("nonexistent")

    def test_start_step(self, state_manager):
        """Test starting a step."""
        state_manager.create_workflow(name="test", steps=["step1", "step2"])
        step = state_manager.start_step("test", "step1")

        assert step.status == WorkflowStatus.RUNNING
        assert step.started_at is not None

        workflow = state_manager.get_workflow("test")
        assert workflow.current_step == "step1"

    def test_start_step_not_found(self, state_manager):
        """Test starting non-existent step raises error."""
        state_manager.create_workflow(name="test", steps=["step1"])

        with pytest.raises(ValueError, match="Step.*not found"):
            state_manager.start_step("test", "nonexistent")

    def test_complete_step(self, state_manager):
        """Test completing a step."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_step("test", "step1")
        step = state_manager.complete_step("test", "step1", result={"count": 5})

        assert step.status == WorkflowStatus.COMPLETED
        assert step.completed_at is not None
        assert step.result == {"count": 5}

    def test_fail_step(self, state_manager):
        """Test failing a step."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_step("test", "step1")
        step = state_manager.fail_step("test", "step1", error="Something broke")

        assert step.status == WorkflowStatus.FAILED
        assert step.completed_at is not None
        assert step.error == "Something broke"

    def test_pause_workflow(self, state_manager):
        """Test pausing a workflow."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_workflow("test")
        workflow = state_manager.pause_workflow("test", reason="Waiting for user")

        assert workflow.status == WorkflowStatus.PAUSED
        assert workflow.context.get("pause_reason") == "Waiting for user"

    def test_resume_workflow(self, state_manager):
        """Test resuming a paused workflow."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_workflow("test")
        state_manager.pause_workflow("test", reason="Test pause")
        workflow = state_manager.resume_workflow("test")

        assert workflow.status == WorkflowStatus.RUNNING
        assert "pause_reason" not in workflow.context

    def test_resume_workflow_not_paused(self, state_manager):
        """Test resuming non-paused workflow raises error."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_workflow("test")

        with pytest.raises(ValueError, match="not paused"):
            state_manager.resume_workflow("test")

    def test_complete_workflow(self, state_manager):
        """Test completing a workflow."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_workflow("test")
        workflow = state_manager.complete_workflow("test", context_update={"final": True})

        assert workflow.status == WorkflowStatus.COMPLETED
        assert workflow.completed_at is not None
        assert workflow.context.get("final") is True

    def test_fail_workflow(self, state_manager):
        """Test failing a workflow."""
        state_manager.create_workflow(name="test", steps=["step1"])
        state_manager.start_workflow("test")
        workflow = state_manager.fail_workflow("test", error="Fatal error")

        assert workflow.status == WorkflowStatus.FAILED
        assert workflow.completed_at is not None
        assert workflow.error == "Fatal error"

    def test_update_context(self, state_manager):
        """Test updating workflow context."""
        state_manager.create_workflow(
            name="test", steps=["step1"], context={"key1": "value1"}
        )
        workflow = state_manager.update_context(
            "test", updates={"key2": "value2", "key3": "value3"}
        )

        assert workflow.context["key1"] == "value1"
        assert workflow.context["key2"] == "value2"
        assert workflow.context["key3"] == "value3"

    def test_get_workflow(self, state_manager):
        """Test getting workflow by name."""
        state_manager.create_workflow(name="test", steps=["step1"])

        workflow = state_manager.get_workflow("test")
        assert workflow is not None
        assert workflow.name == "test"

        missing = state_manager.get_workflow("nonexistent")
        assert missing is None

    def test_list_workflows(self, state_manager):
        """Test listing all workflows."""
        state_manager.create_workflow(name="workflow1", steps=["step1"])
        state_manager.create_workflow(name="workflow2", steps=["step1"])

        workflows = state_manager.list_workflows()

        assert len(workflows) == 2
        names = [w.name for w in workflows]
        assert "workflow1" in names
        assert "workflow2" in names

    def test_get_status(self, state_manager):
        """Test getting status summary."""
        state_manager.create_workflow(name="running", steps=["step1"])
        state_manager.start_workflow("running")
        state_manager.start_step("running", "step1")

        state_manager.create_workflow(name="paused", steps=["step1"])
        state_manager.start_workflow("paused")
        state_manager.pause_workflow("paused")

        status = state_manager.get_status()

        assert len(status) == 2
        assert status["running"]["status"] == "running"
        assert status["running"]["current_step"] == "step1"
        assert status["paused"]["status"] == "paused"

    def test_load_persisted_state(self, state_file):
        """Test loading state from existing file."""
        # Create state with first manager
        manager1 = StateManager(state_file=state_file)
        manager1.create_workflow(name="test", steps=["step1", "step2"])
        manager1.start_workflow("test")
        manager1.start_step("test", "step1")
        manager1.complete_step("test", "step1", result={"done": True})

        # Create new manager that loads from file
        manager2 = StateManager(state_file=state_file)

        workflow = manager2.get_workflow("test")
        assert workflow is not None
        assert workflow.status == WorkflowStatus.RUNNING
        assert workflow.steps["step1"].status == WorkflowStatus.COMPLETED
        assert workflow.steps["step1"].result == {"done": True}

    def test_archive_workflow(self, state_manager, tmp_path):
        """Test workflow archiving on completion."""
        # Configure history directory
        from unittest.mock import patch

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        from src.agent.config import AgentSettings

        with patch.object(AgentSettings, "history_dir", history_dir):
            state_manager.create_workflow(name="test", steps=["step1"])
            state_manager.start_workflow("test")
            state_manager.start_step("test", "step1")
            state_manager.complete_step("test", "step1")

            workflow = state_manager.get_workflow("test")
            state_manager.complete_workflow("test")

            # Check archive file exists
            archive_files = list(history_dir.glob("test_*.yaml"))
            assert len(archive_files) == 1
