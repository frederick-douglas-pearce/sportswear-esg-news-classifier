"""Base workflow class and registry."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import agent_settings
from ..state import StateManager, WorkflowState, WorkflowStatus, state_manager

logger = logging.getLogger(__name__)


@dataclass
class StepDefinition:
    """Definition of a workflow step."""

    name: str
    description: str
    handler: Callable[["Workflow", dict[str, Any]], dict[str, Any] | None]
    skip_on_dry_run: bool = False
    requires_approval: bool = False


class Workflow(ABC):
    """Base class for workflow definitions.

    Subclasses define steps and their execution logic. The base class
    handles state management, error handling, and step transitions.
    """

    # Subclasses must define these
    name: str = ""
    description: str = ""
    steps: list[StepDefinition] = []

    def __init__(
        self,
        state_manager: StateManager | None = None,
        dry_run: bool | None = None,
    ):
        """Initialize workflow.

        Args:
            state_manager: State manager instance (default: global state_manager)
            dry_run: Override dry_run setting
        """
        self.state = state_manager or globals()["state_manager"]
        self.dry_run = dry_run if dry_run is not None else agent_settings.dry_run
        self._workflow_state: WorkflowState | None = None

    @property
    def step_names(self) -> list[str]:
        """Get list of step names."""
        return [step.name for step in self.steps]

    def run(self, context: dict[str, Any] | None = None) -> WorkflowState:
        """Execute the workflow.

        Args:
            context: Initial context data

        Returns:
            Final workflow state
        """
        context = context or {}
        context["dry_run"] = self.dry_run

        # Create workflow state
        self._workflow_state = self.state.create_workflow(
            name=self.name,
            steps=self.step_names,
            context=context,
        )
        self.state.start_workflow(self.name)

        logger.info(f"Starting workflow: {self.name} (dry_run={self.dry_run})")

        try:
            for step in self.steps:
                if self._workflow_state.status == WorkflowStatus.PAUSED:
                    logger.info(f"Workflow paused at step: {step.name}")
                    break

                # Skip steps in dry-run mode if configured
                if self.dry_run and step.skip_on_dry_run:
                    logger.info(f"Skipping step (dry-run): {step.name}")
                    self.state.complete_step(
                        self.name,
                        step.name,
                        result={"skipped": True, "reason": "dry_run"},
                    )
                    continue

                # Handle approval-required steps
                if step.requires_approval:
                    # Set current_step so resume knows where we paused
                    self._workflow_state.current_step = step.name
                    self.state.pause_workflow(
                        self.name,
                        reason=f"Approval required for: {step.name}",
                    )
                    logger.info(f"Workflow paused - approval required for: {step.name}")
                    break

                # Execute step
                self._execute_step(step)

            # Check if all steps completed
            all_completed = all(
                self._workflow_state.steps[s].status == WorkflowStatus.COMPLETED
                for s in self.step_names
            )

            if all_completed:
                self.state.complete_workflow(self.name)
                logger.info(f"Workflow completed: {self.name}")
            elif self._workflow_state.status != WorkflowStatus.PAUSED:
                # Some steps didn't complete but workflow isn't paused
                self.state.fail_workflow(self.name, "Not all steps completed")

        except Exception as e:
            logger.exception(f"Workflow failed: {e}")
            self.state.fail_workflow(self.name, str(e))

        return self.state.get_workflow(self.name)  # type: ignore

    def _execute_step(self, step: StepDefinition) -> None:
        """Execute a single step with error handling."""
        logger.info(f"Executing step: {step.name} - {step.description}")
        self.state.start_step(self.name, step.name)

        try:
            result = step.handler(self, self._workflow_state.context)  # type: ignore
            self.state.complete_step(self.name, step.name, result=result)
            if result:
                self.state.update_context(self.name, result)
            logger.info(f"Step completed: {step.name}")

        except Exception as e:
            logger.error(f"Step failed: {step.name} - {e}")
            self.state.fail_step(self.name, step.name, str(e))
            raise

    def resume(self) -> WorkflowState:
        """Resume a paused workflow.

        Returns:
            Final workflow state
        """
        self._workflow_state = self.state.get_workflow(self.name)
        if not self._workflow_state:
            raise ValueError(f"No workflow state found for: {self.name}")

        if self._workflow_state.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow is not paused: {self._workflow_state.status}")

        self.state.resume_workflow(self.name)
        logger.info(f"Resuming workflow: {self.name}")

        # Find the next pending step after current
        current_step = self._workflow_state.current_step
        found_current = False
        remaining_steps = []

        for step in self.steps:
            if step.name == current_step:
                found_current = True
                # Mark the approval step as completed (user resumed = manual work done)
                if step.requires_approval:
                    self.state.complete_step(
                        self.name,
                        step.name,
                        result={"approved": True, "manual_completion": True},
                    )
                continue
            if found_current:
                remaining_steps.append(step)

        # Execute remaining steps
        try:
            for step in remaining_steps:
                if self._workflow_state.status == WorkflowStatus.PAUSED:
                    break

                step_state = self._workflow_state.steps.get(step.name)
                if step_state and step_state.status == WorkflowStatus.COMPLETED:
                    continue

                if step.requires_approval:
                    # Set current_step so next resume knows where we paused
                    self._workflow_state.current_step = step.name
                    self.state.pause_workflow(
                        self.name,
                        reason=f"Approval required for: {step.name}",
                    )
                    break

                self._execute_step(step)

            all_completed = all(
                self._workflow_state.steps[s].status == WorkflowStatus.COMPLETED
                for s in self.step_names
            )

            if all_completed:
                self.state.complete_workflow(self.name)
                logger.info(f"Workflow completed: {self.name}")

        except Exception as e:
            logger.exception(f"Workflow failed on resume: {e}")
            self.state.fail_workflow(self.name, str(e))

        return self.state.get_workflow(self.name)  # type: ignore


class WorkflowRegistry:
    """Registry of available workflows."""

    _workflows: dict[str, type[Workflow]] = {}

    @classmethod
    def register(cls, workflow_class: type[Workflow]) -> type[Workflow]:
        """Register a workflow class.

        Can be used as a decorator:
            @WorkflowRegistry.register
            class MyWorkflow(Workflow):
                ...
        """
        cls._workflows[workflow_class.name] = workflow_class
        return workflow_class

    @classmethod
    def get(cls, name: str) -> type[Workflow] | None:
        """Get a workflow class by name."""
        return cls._workflows.get(name)

    @classmethod
    def list(cls) -> list[str]:
        """List registered workflow names."""
        return list(cls._workflows.keys())

    @classmethod
    def create(
        cls,
        name: str,
        state_manager: StateManager | None = None,
        dry_run: bool | None = None,
    ) -> Workflow:
        """Create a workflow instance by name.

        Args:
            name: Workflow name
            state_manager: State manager instance
            dry_run: Override dry_run setting

        Returns:
            Workflow instance

        Raises:
            ValueError: If workflow not found
        """
        workflow_class = cls.get(name)
        if not workflow_class:
            available = ", ".join(cls.list())
            raise ValueError(f"Unknown workflow: {name}. Available: {available}")

        return workflow_class(state_manager=state_manager, dry_run=dry_run)
