"""Agent configuration settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentSettings:
    """Configuration for the agent orchestrator."""

    # State management
    state_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("AGENT_STATE_DIR", str(Path.home() / ".esg-agent"))
        )
    )

    # Workflow settings
    dry_run: bool = field(
        default_factory=lambda: os.getenv("AGENT_DRY_RUN", "false").lower() == "true"
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_RETRIES", "3"))
    )
    retry_delay_seconds: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RETRY_DELAY", "5"))
    )

    # Script execution
    default_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("AGENT_DEFAULT_TIMEOUT", "600"))
    )

    # LLM analysis settings
    llm_analysis_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_LLM_ANALYSIS", "true").lower()
        == "true"
    )
    llm_error_threshold: float = field(
        default_factory=lambda: float(os.getenv("AGENT_LLM_ERROR_THRESHOLD", "0.0"))
    )  # 0.0 = always run LLM, >0 = only if error_rate exceeds threshold

    # Email notifications
    email_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_EMAIL_ENABLED", "false").lower()
        == "true"
    )
    email_recipient: str | None = field(
        default_factory=lambda: os.getenv("AGENT_EMAIL_RECIPIENT") or None
    )
    email_sender: str | None = field(
        default_factory=lambda: os.getenv("AGENT_EMAIL_SENDER") or None
    )

    # Resend API (recommended for email)
    resend_api_key: str | None = field(
        default_factory=lambda: os.getenv("RESEND_API_KEY") or None
    )

    # SMTP settings (legacy fallback)
    smtp_host: str = field(
        default_factory=lambda: os.getenv("AGENT_SMTP_HOST", "localhost")
    )
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("AGENT_SMTP_PORT", "587"))
    )
    smtp_password: str | None = field(
        default_factory=lambda: os.getenv("AGENT_SMTP_PASSWORD") or None
    )

    # Project paths
    project_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "AGENT_PROJECT_ROOT",
                str(Path(__file__).parent.parent.parent),
            )
        )
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_LOGS_DIR", "logs/agent"))
    )

    # Website export paths (for feed export workflow)
    website_repo_path: Path | None = field(
        default_factory=lambda: Path(p)
        if (p := os.getenv("AGENT_WEBSITE_REPO_PATH"))
        else None
    )

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.project_root / self.logs_dir).mkdir(parents=True, exist_ok=True)

    @property
    def state_file(self) -> Path:
        """Path to the main state file."""
        return self.state_dir / "state.yaml"

    @property
    def history_dir(self) -> Path:
        """Path to workflow history directory."""
        history = self.state_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        return history

    def get_workflow_log_path(self, workflow_name: str) -> Path:
        """Get path to workflow-specific log file."""
        log_dir = self.project_root / self.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{workflow_name}.log"


agent_settings = AgentSettings()
