"""Workflow learning configuration settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkflowLearningSettings:
    """Configuration for the workflow learning module."""

    # Screenpipe settings
    screenpipe_api_url: str = field(
        default_factory=lambda: os.getenv(
            "SCREENPIPE_API_URL", "http://localhost:3030"
        )
    )

    # Recording settings
    recording_output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WORKFLOW_RECORDING_DIR",
                str(
                    Path(__file__).parent.parent.parent
                    / "data"
                    / "workflow_recordings"
                ),
            )
        )
    )

    # Skills output settings
    skills_output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WORKFLOW_SKILLS_DIR",
                str(Path(__file__).parent.parent.parent / ".claude" / "skills" / "learned"),
            )
        )
    )

    # Analysis settings
    analysis_model: str = field(
        default_factory=lambda: os.getenv(
            "WORKFLOW_ANALYSIS_MODEL", "claude-haiku-4-5-20251001"
        )
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MAX_RETRIES", "3"))
    )
    retry_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("WORKFLOW_RETRY_DELAY", "1.0"))
    )

    # Content retrieval settings
    max_screen_frames: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MAX_SCREEN_FRAMES", "1000"))
    )
    max_audio_chunks: int = field(
        default_factory=lambda: int(os.getenv("WORKFLOW_MAX_AUDIO_CHUNKS", "500"))
    )

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.recording_output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_dir(self) -> Path:
        """Path to session state files directory."""
        return self.recording_output_dir / "sessions"


# Global settings instance
workflow_learning_settings = WorkflowLearningSettings()
