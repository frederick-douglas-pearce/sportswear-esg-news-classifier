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
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )
    llm_analysis_model: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_LLM_MODEL", "claude-haiku-4-5-20251001"
        )
    )

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

    # Drift monitoring
    #
    # The EP classifier is on hold (see CLAUDE.md). Run against no EP data, its
    # drift check found nothing to compare and reported "Healthy" (#71). Gating
    # it on config rather than on a row count is deliberate: EP-on-hold is a
    # governance decision. A row-count gate would re-enable monitoring the
    # moment one stray `ep` row landed (issues #71, #96).
    ep_drift_enabled: bool = field(
        default_factory=lambda: os.getenv("AGENT_EP_DRIFT_ENABLED", "false").lower()
        == "true"
    )
    ep_drift_skip_reason: str = field(
        default_factory=lambda: os.getenv(
            "AGENT_EP_DRIFT_SKIP_REASON",
            "EP classifier is on hold (CLAUDE.md). Set "
            "AGENT_EP_DRIFT_ENABLED=true when it resumes.",
        )
    )

    # Run-archive audit (#76)
    #
    # What the auditor expects to see in the run archive. Deliberately NOT
    # env-overridable and deliberately not derived from crontab: this is a
    # governance decision about which scheduled jobs are watched, and parsing
    # the live crontab would couple the audit to the host it happens to run on.
    #
    # The direction that matters is ADDING a workflow: a job that gains a cron
    # entry and no entry here is a scheduled job with no detector, which is this
    # epic's own defect reproduced in the auditor's configuration. A test in
    # tests/test_agent_archive.py asserts every workflow scheduled through
    # cron_agent.sh appears in one of the two dicts below, so that drift breaks
    # CI instead of going quiet. The other directions -- a cron entry removed or
    # lengthened -- surface as a noisy stale alert, which is the safe way to be
    # wrong and needs no guard.
    audit_expected_interval_hours: dict[str, float] = field(
        default_factory=lambda: {
            "daily_labeling": 24.0,
            "website_export": 24.0,
            "drift_monitoring": 24.0,
        }
    )
    # Added to a workflow's interval before it is called stale, so that ordinary
    # jitter in start time is not an alert. A run is stale when its age exceeds
    # interval + grace.
    #
    # Sized for start-time jitter, not for detection latency: `run_id` is
    # stamped when a run starts (state.py), so a long run does not age its own
    # archive. Latency also depends on how often the audit itself runs -- the
    # bound is interval + grace + the audit's period, set in
    # scripts/setup_cron.sh.
    audit_grace_hours: float = 3.0
    # Workflows deliberately not audited, each with the reason stated. A skip is
    # a decision someone made; HealthVerdict.SKIPPED requires it be recorded.
    audit_skipped_workflows: dict[str, str] = field(
        default_factory=lambda: {
            "model_training": (
                "Not scheduled -- run by hand and pauses for notebooks, so it has "
                "no cadence to be late against."
            ),
            "run_audit": (
                "The auditor cannot audit itself: a process cannot observe its "
                "own absence. Detecting a stalled auditor needs an off-host "
                "dead-man's-switch, which is out of scope."
            ),
        }
    )

    # Consecutive-failure escalation (#75)
    #
    # How many runs of a watched workflow must fail in a row before the auditor
    # escalates. Env-overridable, unlike the cadence dicts above: which jobs are
    # watched is a governance decision, while N is an operator's sensitivity
    # knob. The inconsistency with `audit_grace_hours` -- also a sensitivity
    # knob, also code-only -- is deliberate and was ratified at #75's plan gate.
    #
    # Held as the RAW string and parsed by `parse_failure_threshold` at the point
    # of use, so a bad value fails the one step that reads it rather than the
    # import. Validating in `__post_init__` raised from module scope, where
    # `agent_settings` is constructed: one mistyped knob then stopped every agent
    # workflow, including the auditor that exists to notice things going dark.
    # That is this epic's own defect class recursing through its configuration,
    # and D014.7 (which asked for the `__post_init__` raise) was reversed at
    # #75's code-review scope ruling. `_cadence_config_error` is the precedent --
    # loud, and scoped to the check it disables.
    #
    # Nothing guards the UP direction: a large N is a detector that never fires
    # and says nothing about it.
    consecutive_failure_threshold_raw: str = field(
        default_factory=lambda: os.getenv("AGENT_CONSECUTIVE_FAILURE_THRESHOLD", "2")
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
    website_expected_branch: str = field(
        default_factory=lambda: os.getenv("AGENT_WEBSITE_EXPECTED_BRANCH", "main")
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


def parse_failure_threshold(raw: str) -> tuple[int | None, str | None]:
    """Parse the consecutive-failure threshold, returning (value, error).

    Returns exactly one of the two: a usable threshold, or a message naming the
    variable and quoting what was found. Never raises, and never clamps -- a
    clamp would run the escalator at a threshold nobody configured and report
    nothing wrong, which is the failure mode #75 exists to remove.

    N = 0 is refused because it would escalate a workflow that has not failed at
    all. That is the DOWN direction; nothing here guards the UP direction, where
    a large N is a detector that never fires.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, (
            f"AGENT_CONSECUTIVE_FAILURE_THRESHOLD is not a number: {raw!r}"
        )
    if value < 1:
        return None, (
            f"AGENT_CONSECUTIVE_FAILURE_THRESHOLD must be at least 1; got {value}"
        )
    return value, None


agent_settings = AgentSettings()
