"""Data models for workflow learning."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionStatus(str, Enum):
    """Status of a recording session."""

    RECORDING = "recording"
    STOPPED = "stopped"
    ANALYZED = "analyzed"
    SKILL_GENERATED = "skill_generated"
    FAILED = "failed"


class ScreenContent(BaseModel):
    """Screen content captured from Screenpipe."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    timestamp: datetime
    app_name: str = ""
    window_title: str = ""
    ocr_text: str = ""
    file_path: str | None = None  # Screenshot file if available


class AudioTranscript(BaseModel):
    """Audio transcription from Screenpipe."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    timestamp: datetime
    text: str
    duration_seconds: float = 0.0
    device_name: str = ""  # Microphone name
    is_input: bool = True  # True for mic, False for system audio


class WorkflowStep(BaseModel):
    """A discrete step extracted from the workflow recording."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    step_number: int
    action: str  # What the user did (e.g., "Run training script")
    target: str = ""  # What they acted on (e.g., "fp classifier")
    purpose: str = ""  # Why (e.g., "Train the false positive classifier")
    command: str | None = None  # Executable command if applicable
    evidence: list[str] = Field(default_factory=list)  # Supporting OCR/transcript snippets
    notes: str = ""  # Additional context from voice narration


class RecordingSession(BaseModel):
    """A workflow recording session."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    session_id: str
    workflow_name: str
    description: str = ""
    status: SessionStatus = SessionStatus.RECORDING
    started_at: datetime
    stopped_at: datetime | None = None
    analyzed_at: datetime | None = None
    skill_generated_at: datetime | None = None

    # Content retrieved from Screenpipe
    screen_content: list[ScreenContent] = Field(default_factory=list)
    audio_transcripts: list[AudioTranscript] = Field(default_factory=list)

    # Analysis results
    extracted_steps: list[WorkflowStep] = Field(default_factory=list)
    analysis_summary: str = ""
    skill_path: str | None = None  # Path to generated SKILL.md

    # Error tracking
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "session_id": self.session_id,
            "workflow_name": self.workflow_name,
            "description": self.description,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "skill_generated_at": (
                self.skill_generated_at.isoformat() if self.skill_generated_at else None
            ),
            "screen_content": [sc.model_dump() for sc in self.screen_content],
            "audio_transcripts": [at.model_dump() for at in self.audio_transcripts],
            "extracted_steps": [es.model_dump() for es in self.extracted_steps],
            "analysis_summary": self.analysis_summary,
            "skill_path": self.skill_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordingSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            workflow_name=data["workflow_name"],
            description=data.get("description", ""),
            status=SessionStatus(data.get("status", "recording")),
            started_at=datetime.fromisoformat(data["started_at"]),
            stopped_at=(
                datetime.fromisoformat(data["stopped_at"])
                if data.get("stopped_at")
                else None
            ),
            analyzed_at=(
                datetime.fromisoformat(data["analyzed_at"])
                if data.get("analyzed_at")
                else None
            ),
            skill_generated_at=(
                datetime.fromisoformat(data["skill_generated_at"])
                if data.get("skill_generated_at")
                else None
            ),
            screen_content=[
                ScreenContent(**sc) for sc in data.get("screen_content", [])
            ],
            audio_transcripts=[
                AudioTranscript(**at) for at in data.get("audio_transcripts", [])
            ],
            extracted_steps=[
                WorkflowStep(**es) for es in data.get("extracted_steps", [])
            ],
            analysis_summary=data.get("analysis_summary", ""),
            skill_path=data.get("skill_path"),
            error=data.get("error"),
        )


class AnalysisResult(BaseModel):
    """Result of Claude analysis of a recording."""

    success: bool
    steps: list[WorkflowStep] = Field(default_factory=list)
    summary: str = ""
    skill_name: str = ""
    skill_description: str = ""
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
