"""Script execution wrapper with retry logic and output capture."""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import agent_settings

logger = logging.getLogger(__name__)


@dataclass
class ScriptResult:
    """Result of a script execution."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    started_at: datetime
    success: bool = field(init=False)
    parsed_output: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        self.success = self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat(),
            "stdout_length": len(self.stdout),
            "stderr_length": len(self.stderr),
            "parsed_output": self.parsed_output,
        }


def run_script(
    command: list[str],
    timeout: int | None = None,
    retries: int | None = None,
    retry_delay: int | None = None,
    dry_run: bool | None = None,
    cwd: Path | None = None,
    parse_json_output: bool = False,
) -> ScriptResult:
    """Execute a script with retry logic and output capture.

    Args:
        command: Command and arguments to execute
        timeout: Timeout in seconds (default: agent_settings.default_timeout_seconds)
        retries: Number of retry attempts (default: agent_settings.max_retries)
        retry_delay: Delay between retries in seconds (default: agent_settings.retry_delay_seconds)
        dry_run: If True, add --dry-run flag (default: agent_settings.dry_run)
        cwd: Working directory (default: agent_settings.project_root)
        parse_json_output: If True, attempt to parse last JSON object from stdout

    Returns:
        ScriptResult with execution details
    """
    timeout = timeout or agent_settings.default_timeout_seconds
    retries = retries if retries is not None else agent_settings.max_retries
    retry_delay = retry_delay or agent_settings.retry_delay_seconds
    dry_run = dry_run if dry_run is not None else agent_settings.dry_run
    cwd = cwd or agent_settings.project_root

    # Add --dry-run flag if enabled and not already present
    if dry_run and "--dry-run" not in command:
        command = command + ["--dry-run"]

    last_result: ScriptResult | None = None

    for attempt in range(retries + 1):
        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        try:
            logger.info(
                f"Running command (attempt {attempt + 1}/{retries + 1}): {' '.join(command)}"
            )

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            duration = time.time() - start_time

            script_result = ScriptResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                started_at=started_at,
            )

            # Parse JSON output if requested
            if parse_json_output and result.stdout:
                script_result.parsed_output = _parse_json_from_output(result.stdout)

            if script_result.success:
                logger.info(
                    f"Command succeeded in {duration:.2f}s (exit code: {result.returncode})"
                )
                return script_result

            # Command failed
            logger.warning(
                f"Command failed (exit code: {result.returncode}): {result.stderr[:500]}"
            )
            last_result = script_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.warning(f"Command timed out after {timeout}s")
            last_result = ScriptResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {timeout} seconds",
                duration_seconds=duration,
                started_at=started_at,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Command execution error: {e}")
            last_result = ScriptResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                started_at=started_at,
            )

        # Retry with exponential backoff
        if attempt < retries:
            wait_time = retry_delay * (2**attempt)
            logger.info(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    return last_result  # type: ignore


def _parse_json_from_output(output: str) -> dict[str, Any] | None:
    """Parse JSON from script output.

    Looks for JSON objects in the output, preferring the last one found.
    This handles scripts that may output logs before JSON.
    """
    # Try to parse the entire output as JSON first
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Look for JSON objects in the output (find last { ... } block)
    lines = output.strip().split("\n")
    json_lines = []
    in_json = False
    brace_count = 0

    for line in reversed(lines):
        if "}" in line:
            in_json = True
        if in_json:
            json_lines.insert(0, line)
            brace_count += line.count("{") - line.count("}")
            if brace_count == 0 and "{" in line:
                break

    if json_lines:
        try:
            return json.loads("\n".join(json_lines))
        except json.JSONDecodeError:
            pass

    return None


def run_uv_script(
    script_path: str,
    args: list[str] | None = None,
    **kwargs: Any,
) -> ScriptResult:
    """Run a Python script using uv run.

    Args:
        script_path: Path to the script relative to project root
        args: Additional arguments for the script
        **kwargs: Additional arguments passed to run_script

    Returns:
        ScriptResult with execution details
    """
    command = ["uv", "run", "python", script_path]
    if args:
        command.extend(args)
    return run_script(command, **kwargs)


def run_label_articles(
    batch_size: int | None = None,
    dry_run: bool | None = None,
    stats_only: bool = False,
) -> ScriptResult:
    """Run the label_articles.py script.

    Args:
        batch_size: Number of articles to process (None = all pending)
        dry_run: If True, don't save to database
        stats_only: If True, only show stats without labeling

    Returns:
        ScriptResult with execution details
    """
    args = []
    if stats_only:
        args.append("--stats")
    elif batch_size is not None:
        args.extend(["--batch-size", str(batch_size)])

    return run_uv_script(
        "scripts/label_articles.py",
        args=args,
        dry_run=dry_run if not stats_only else False,
        timeout=1800,  # 30 minutes for labeling
    )


def run_export_training_data(
    dataset: str,
    output_path: str | None = None,
) -> ScriptResult:
    """Run export_training_data.py script.

    Args:
        dataset: Dataset type (fp, esg-prefilter, esg-labels)
        output_path: Output file path (optional)

    Returns:
        ScriptResult with execution details
    """
    args = ["--dataset", dataset]
    if output_path:
        args.extend(["-o", output_path])

    return run_uv_script(
        "scripts/export_training_data.py",
        args=args,
        dry_run=False,  # Export doesn't have dry-run
    )


def run_monitor_drift(
    classifier: str,
    days: int = 7,
    from_db: bool = True,
    html_report: bool = False,
    alert: bool = False,
) -> ScriptResult:
    """Run monitor_drift.py script.

    Args:
        classifier: Classifier type (fp, ep, esg)
        days: Number of days to analyze
        from_db: Load predictions from database
        html_report: Generate HTML report
        alert: Send webhook alert if drift detected

    Returns:
        ScriptResult with execution details
    """
    args = ["--classifier", classifier, "--days", str(days)]
    if from_db:
        args.append("--from-db")
    if html_report:
        args.append("--html-report")
    if alert:
        args.append("--alert")

    return run_uv_script(
        "scripts/monitor_drift.py",
        args=args,
        dry_run=False,
    )


def run_export_website_feed(
    json_output: str | None = None,
    atom_output: str | None = None,
    format: str = "both",
) -> ScriptResult:
    """Run export_website_feed.py script.

    Args:
        json_output: Path for JSON output
        atom_output: Path for Atom output
        format: Output format (json, atom, both)

    Returns:
        ScriptResult with execution details
    """
    args = ["--format", format]
    if json_output:
        args.extend(["--json-output", json_output])
    if atom_output:
        args.extend(["--atom-output", atom_output])

    return run_uv_script(
        "scripts/export_website_feed.py",
        args=args,
        dry_run=False,
    )


def run_backup_status() -> ScriptResult:
    """Run backup_db.sh status.

    Returns:
        ScriptResult with execution details
    """
    return run_script(
        ["./scripts/backup_db.sh", "status"],
        retries=0,  # No retries for status check
    )


def get_labeling_stats() -> dict[str, Any]:
    """Get current labeling statistics.

    Returns:
        Dictionary with labeling stats or empty dict on error
    """
    result = run_label_articles(stats_only=True)
    if result.success and result.stdout:
        # Parse stats from output
        stats = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                try:
                    stats[key] = int(value)
                except ValueError:
                    try:
                        stats[key] = float(value)
                    except ValueError:
                        stats[key] = value
        return stats
    return {}
