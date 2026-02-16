"""Workflow learning module for recording and analyzing user workflows.

This module allows users to record their screen + voice while demonstrating
a workflow, then have Claude analyze the recording and generate a replayable
Agent Skill.

Key components:
- screenpipe_client: Wrapper for Screenpipe REST API
- session_manager: Recording session state management
- analyzer: Claude-based recording analysis
- skill_generator: Generate SKILL.md from analysis

Usage:
    # Start a recording session
    uv run python -m src.workflow_learning start "model-training-fp"

    # Stop the session
    uv run python -m src.workflow_learning stop

    # Analyze and generate skill
    uv run python -m src.workflow_learning analyze <session-id>
"""

from .analyzer import RecordingAnalyzer
from .config import workflow_learning_settings
from .models import (
    AnalysisResult,
    AudioTranscript,
    RecordingSession,
    ScreenContent,
    SessionStatus,
    WorkflowStep,
)
from .screenpipe_client import ScreenpipeClient, ScreenpipeError
from .session_manager import SessionManager
from .skill_generator import SkillGenerator, generate_skill_from_session

__all__ = [
    "AnalysisResult",
    "AudioTranscript",
    "RecordingAnalyzer",
    "RecordingSession",
    "ScreenContent",
    "ScreenpipeClient",
    "ScreenpipeError",
    "SessionManager",
    "SessionStatus",
    "SkillGenerator",
    "WorkflowStep",
    "generate_skill_from_session",
    "workflow_learning_settings",
]
