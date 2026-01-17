"""YAML-based state management for workflow orchestration."""

import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .config import agent_settings

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Waiting for user action (e.g., run notebooks)
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepState:
    """State of a single workflow step."""

    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepState":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=WorkflowStatus(data.get("status", "pending")),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            error=data.get("error"),
            result=data.get("result"),
        )


@dataclass
class WorkflowState:
    """State of a workflow execution."""

    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_step: str | None = None
    steps: dict[str, StepState] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    run_id: str | None = None  # Unique ID for this run

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": self.current_step,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "context": self.context,
            "error": self.error,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        """Create from dictionary."""
        steps = {}
        for step_name, step_data in data.get("steps", {}).items():
            steps[step_name] = StepState.from_dict(step_data)

        return cls(
            name=data["name"],
            status=WorkflowStatus(data.get("status", "pending")),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            current_step=data.get("current_step"),
            steps=steps,
            context=data.get("context", {}),
            error=data.get("error"),
            run_id=data.get("run_id"),
        )


class StateManager:
    """Manages workflow state with YAML persistence."""

    def __init__(self, state_file: Path | None = None):
        """Initialize state manager.

        Args:
            state_file: Path to state file. Defaults to agent_settings.state_file.
        """
        self.state_file = state_file or agent_settings.state_file
        self._state: dict[str, WorkflowState] = {}
        self._load()

    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = yaml.safe_load(f) or {}
                for workflow_name, workflow_data in data.get("workflows", {}).items():
                    self._state[workflow_name] = WorkflowState.from_dict(workflow_data)
                logger.debug(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state from {self.state_file}: {e}")
                self._state = {}

    def _save(self) -> None:
        """Save state to file atomically."""
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = self.state_file.with_suffix(".yaml.tmp")
        data = {"workflows": {k: v.to_dict() for k, v in self._state.items()}}

        with open(temp_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        temp_file.rename(self.state_file)
        logger.debug(f"Saved state to {self.state_file}")

    def get_workflow(self, name: str) -> WorkflowState | None:
        """Get workflow state by name."""
        return self._state.get(name)

    def create_workflow(
        self,
        name: str,
        steps: list[str],
        context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Create a new workflow execution.

        Args:
            name: Workflow name (e.g., 'daily_labeling')
            steps: List of step names in order
            context: Initial context data

        Returns:
            Created WorkflowState
        """
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%d_%H%M%S")

        workflow = WorkflowState(
            name=name,
            status=WorkflowStatus.PENDING,
            started_at=now,
            run_id=run_id,
            steps={step: StepState(name=step) for step in steps},
            context=context or {},
        )
        self._state[name] = workflow
        self._save()
        return workflow

    def start_workflow(self, name: str) -> WorkflowState:
        """Mark workflow as running."""
        workflow = self._state.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not found")

        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc)
        self._save()
        return workflow

    def start_step(self, workflow_name: str, step_name: str) -> StepState:
        """Mark a step as running."""
        workflow = self._state.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        step = workflow.steps.get(step_name)
        if not step:
            raise ValueError(f"Step '{step_name}' not found in workflow '{workflow_name}'")

        step.status = WorkflowStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        workflow.current_step = step_name
        self._save()
        return step

    def complete_step(
        self,
        workflow_name: str,
        step_name: str,
        result: dict[str, Any] | None = None,
    ) -> StepState:
        """Mark a step as completed."""
        workflow = self._state.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        step = workflow.steps.get(step_name)
        if not step:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = WorkflowStatus.COMPLETED
        step.completed_at = datetime.now(timezone.utc)
        step.result = result
        self._save()
        return step

    def fail_step(
        self,
        workflow_name: str,
        step_name: str,
        error: str,
    ) -> StepState:
        """Mark a step as failed."""
        workflow = self._state.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        step = workflow.steps.get(step_name)
        if not step:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = WorkflowStatus.FAILED
        step.completed_at = datetime.now(timezone.utc)
        step.error = error
        self._save()
        return step

    def pause_workflow(self, name: str, reason: str | None = None) -> WorkflowState:
        """Pause workflow (waiting for user action)."""
        workflow = self._state.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not found")

        workflow.status = WorkflowStatus.PAUSED
        if reason:
            workflow.context["pause_reason"] = reason
        self._save()
        return workflow

    def resume_workflow(self, name: str) -> WorkflowState:
        """Resume a paused workflow."""
        workflow = self._state.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not found")

        if workflow.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow '{name}' is not paused (status: {workflow.status})")

        workflow.status = WorkflowStatus.RUNNING
        workflow.context.pop("pause_reason", None)
        self._save()
        return workflow

    def complete_workflow(
        self,
        name: str,
        context_update: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Mark workflow as completed."""
        workflow = self._state.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not found")

        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now(timezone.utc)
        if context_update:
            workflow.context.update(context_update)
        self._save()
        self._archive_workflow(name)
        return workflow

    def fail_workflow(self, name: str, error: str) -> WorkflowState:
        """Mark workflow as failed."""
        workflow = self._state.get(name)
        if not workflow:
            raise ValueError(f"Workflow '{name}' not found")

        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = datetime.now(timezone.utc)
        workflow.error = error
        self._save()
        self._archive_workflow(name)
        return workflow

    def update_context(
        self,
        workflow_name: str,
        updates: dict[str, Any],
    ) -> WorkflowState:
        """Update workflow context."""
        workflow = self._state.get(workflow_name)
        if not workflow:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow.context.update(updates)
        self._save()
        return workflow

    def _archive_workflow(self, name: str) -> None:
        """Archive completed/failed workflow to history."""
        workflow = self._state.get(name)
        if not workflow:
            return

        history_dir = agent_settings.history_dir
        archive_file = history_dir / f"{name}_{workflow.run_id}.yaml"

        with open(archive_file, "w") as f:
            yaml.dump(workflow.to_dict(), f, default_flow_style=False)

        logger.info(f"Archived workflow to {archive_file}")

    def list_workflows(self) -> list[WorkflowState]:
        """List all workflows."""
        return list(self._state.values())

    def get_status(self) -> dict[str, Any]:
        """Get summary of all workflow states."""
        return {
            name: {
                "status": wf.status.value,
                "current_step": wf.current_step,
                "started_at": wf.started_at.isoformat() if wf.started_at else None,
            }
            for name, wf in self._state.items()
        }


# Global state manager instance
state_manager = StateManager()
