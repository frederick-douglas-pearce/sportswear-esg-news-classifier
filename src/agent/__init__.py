"""Agent module for automating project maintenance workflows.

This module provides a lightweight orchestration system for running
recurring workflows like daily labeling, model training, and monitoring.

Key components:
- config: Agent configuration from environment variables
- state: YAML-based state management with checkpointing
- runner: Script execution wrapper with retry logic
- workflows: Workflow definitions (daily_labeling, model_training, etc.)
"""

from .config import agent_settings
from .runner import ScriptResult, run_script
from .state import WorkflowState, WorkflowStatus, state_manager

__all__ = [
    "agent_settings",
    "run_script",
    "ScriptResult",
    "state_manager",
    "WorkflowState",
    "WorkflowStatus",
]
