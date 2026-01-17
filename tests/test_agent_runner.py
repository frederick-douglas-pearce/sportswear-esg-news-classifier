"""Tests for agent script runner."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.runner import (
    ScriptResult,
    _parse_json_from_output,
    run_export_training_data,
    run_export_website_feed,
    run_label_articles,
    run_monitor_drift,
    run_script,
    run_uv_script,
)


@pytest.fixture
def mock_agent_settings(tmp_path):
    """Mock agent settings for testing."""
    with patch("src.agent.runner.agent_settings") as mock:
        mock.project_root = tmp_path
        mock.default_timeout_seconds = 60
        mock.max_retries = 2
        mock.retry_delay_seconds = 1
        mock.dry_run = False
        yield mock


class TestScriptResult:
    """Tests for ScriptResult dataclass."""

    def test_success_on_zero_exit_code(self):
        """Test success is True for exit code 0."""
        result = ScriptResult(
            command=["test"],
            exit_code=0,
            stdout="output",
            stderr="",
            duration_seconds=1.5,
            started_at=datetime.now(timezone.utc),
        )

        assert result.success is True

    def test_failure_on_nonzero_exit_code(self):
        """Test success is False for non-zero exit code."""
        result = ScriptResult(
            command=["test"],
            exit_code=1,
            stdout="",
            stderr="error",
            duration_seconds=1.0,
            started_at=datetime.now(timezone.utc),
        )

        assert result.success is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        result = ScriptResult(
            command=["echo", "hello"],
            exit_code=0,
            stdout="hello\n",
            stderr="",
            duration_seconds=0.5,
            started_at=now,
            parsed_output={"key": "value"},
        )

        data = result.to_dict()

        assert data["command"] == ["echo", "hello"]
        assert data["exit_code"] == 0
        assert data["success"] is True
        assert data["duration_seconds"] == 0.5
        assert data["started_at"] == now.isoformat()
        assert data["stdout_length"] == 6
        assert data["stderr_length"] == 0
        assert data["parsed_output"] == {"key": "value"}


class TestParseJsonFromOutput:
    """Tests for JSON parsing from script output."""

    def test_parse_clean_json(self):
        """Test parsing clean JSON output."""
        output = '{"key": "value", "count": 42}'
        result = _parse_json_from_output(output)

        assert result == {"key": "value", "count": 42}

    def test_parse_json_with_prefix_logs(self):
        """Test parsing JSON with log lines before it."""
        output = """
2024-01-15 10:00:00 - INFO - Starting
2024-01-15 10:00:01 - INFO - Processing
{"result": "success", "count": 10}
"""
        result = _parse_json_from_output(output)

        assert result == {"result": "success", "count": 10}

    def test_parse_multiline_json(self):
        """Test parsing multi-line JSON."""
        output = """
Some log output
{
    "key1": "value1",
    "key2": 123
}
"""
        result = _parse_json_from_output(output)

        assert result == {"key1": "value1", "key2": 123}

    def test_parse_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        output = "This is not JSON at all"
        result = _parse_json_from_output(output)

        assert result is None

    def test_parse_empty_output_returns_none(self):
        """Test that empty output returns None."""
        result = _parse_json_from_output("")

        assert result is None


class TestRunScript:
    """Tests for run_script function."""

    def test_successful_command(self, mock_agent_settings, tmp_path):
        """Test running a successful command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="success", stderr=""
            )

            result = run_script(["echo", "hello"])

            assert result.success is True
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_failed_command_with_retries(self, mock_agent_settings):
        """Test command failure triggers retries."""
        call_count = 0

        def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(returncode=1, stdout="", stderr="error")

        with patch("subprocess.run", side_effect=mock_subprocess):
            with patch("time.sleep"):  # Skip actual sleep
                result = run_script(["failing_command"])

        # Should be called 3 times (1 initial + 2 retries)
        assert call_count == 3
        assert result.success is False
        assert result.exit_code == 1

    def test_successful_after_retry(self, mock_agent_settings):
        """Test command succeeds after retry."""
        call_count = 0

        def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return MagicMock(returncode=1, stdout="", stderr="error")
            return MagicMock(returncode=0, stdout="success", stderr="")

        with patch("subprocess.run", side_effect=mock_subprocess):
            with patch("time.sleep"):
                result = run_script(["flaky_command"])

        assert call_count == 2
        assert result.success is True

    def test_timeout_handling(self, mock_agent_settings):
        """Test timeout is handled correctly."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=60)

            with patch("time.sleep"):
                result = run_script(["slow_command"], retries=0)

        assert result.success is False
        assert result.exit_code == -1
        assert "Timeout" in result.stderr

    def test_dry_run_flag_added(self, mock_agent_settings):
        """Test --dry-run flag is added when enabled."""
        mock_agent_settings.dry_run = True

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            run_script(["some_script.py"])

            called_command = mock_run.call_args[0][0]
            assert "--dry-run" in called_command

    def test_dry_run_flag_not_duplicated(self, mock_agent_settings):
        """Test --dry-run flag not duplicated if already present."""
        mock_agent_settings.dry_run = True

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            run_script(["some_script.py", "--dry-run"])

            called_command = mock_run.call_args[0][0]
            assert called_command.count("--dry-run") == 1

    def test_custom_working_directory(self, mock_agent_settings, tmp_path):
        """Test custom working directory is used."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            run_script(["test"], cwd=custom_dir)

            _, kwargs = mock_run.call_args
            assert kwargs["cwd"] == custom_dir

    def test_json_parsing_enabled(self, mock_agent_settings):
        """Test JSON output parsing when enabled."""
        json_output = '{"result": "success"}'

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json_output, stderr=""
            )

            result = run_script(["test"], parse_json_output=True)

            assert result.parsed_output == {"result": "success"}


class TestRunUvScript:
    """Tests for run_uv_script function."""

    def test_constructs_uv_command(self, mock_agent_settings):
        """Test that uv run command is constructed correctly."""
        with patch("src.agent.runner.run_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_uv_script("scripts/test.py", args=["--flag", "value"])

            called_command = mock_run.call_args[0][0]
            assert called_command == [
                "uv",
                "run",
                "python",
                "scripts/test.py",
                "--flag",
                "value",
            ]


class TestRunLabelArticles:
    """Tests for run_label_articles helper."""

    def test_stats_only_mode(self, mock_agent_settings):
        """Test stats-only mode."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_label_articles(stats_only=True)

            call_args = mock_run.call_args
            assert call_args[0][0] == "scripts/label_articles.py"
            assert "--stats" in call_args[1]["args"]

    def test_batch_size_option(self, mock_agent_settings):
        """Test batch size option."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_label_articles(batch_size=50)

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "--batch-size" in args
            assert "50" in args

    def test_extended_timeout(self, mock_agent_settings):
        """Test labeling uses extended timeout."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_label_articles()

            call_args = mock_run.call_args
            assert call_args[1]["timeout"] == 1800  # 30 minutes


class TestRunExportTrainingData:
    """Tests for run_export_training_data helper."""

    def test_dataset_argument(self, mock_agent_settings):
        """Test dataset argument is passed."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_export_training_data(dataset="fp")

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "--dataset" in args
            assert "fp" in args

    def test_output_path_argument(self, mock_agent_settings):
        """Test output path argument."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_export_training_data(dataset="fp", output_path="/tmp/output.jsonl")

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "-o" in args
            assert "/tmp/output.jsonl" in args


class TestRunMonitorDrift:
    """Tests for run_monitor_drift helper."""

    def test_basic_drift_check(self, mock_agent_settings):
        """Test basic drift monitoring."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_monitor_drift(classifier="fp", days=7)

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "--classifier" in args
            assert "fp" in args
            assert "--days" in args
            assert "7" in args
            assert "--from-db" in args

    def test_html_report_option(self, mock_agent_settings):
        """Test HTML report option."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_monitor_drift(classifier="fp", html_report=True)

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "--html-report" in args


class TestRunExportWebsiteFeed:
    """Tests for run_export_website_feed helper."""

    def test_both_outputs(self, mock_agent_settings):
        """Test both JSON and Atom outputs."""
        with patch("src.agent.runner.run_uv_script") as mock_run:
            mock_run.return_value = MagicMock(success=True)

            run_export_website_feed(
                json_output="/path/to/feed.json",
                atom_output="/path/to/feed.atom",
                format="both",
            )

            call_args = mock_run.call_args
            args = call_args[1]["args"]
            assert "--format" in args
            assert "both" in args
            assert "--json-output" in args
            assert "/path/to/feed.json" in args
            assert "--atom-output" in args
            assert "/path/to/feed.atom" in args
