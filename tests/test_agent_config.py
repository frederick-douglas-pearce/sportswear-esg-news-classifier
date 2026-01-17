"""Tests for agent configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestAgentSettings:
    """Tests for AgentSettings configuration."""

    def test_default_values(self, tmp_path):
        """Test default configuration values.

        Note: Some defaults may be overridden by .env if loaded.
        This test verifies core defaults that don't depend on .env.
        """
        with patch.dict(os.environ, {"AGENT_STATE_DIR": str(tmp_path)}, clear=False):
            from src.agent.config import AgentSettings

            settings = AgentSettings()

            # These defaults should always be consistent
            assert settings.dry_run is False
            assert settings.max_retries == 3
            assert settings.retry_delay_seconds == 5
            assert settings.default_timeout_seconds == 600
            assert settings.llm_analysis_enabled is True
            assert settings.llm_error_threshold == 0.0
            # Email settings may come from .env, just verify they're the right type
            assert isinstance(settings.email_enabled, bool)
            assert isinstance(settings.smtp_port, int)

    def test_environment_variable_override(self, tmp_path):
        """Test configuration from environment variables."""
        env_vars = {
            "AGENT_STATE_DIR": str(tmp_path),
            "AGENT_DRY_RUN": "true",
            "AGENT_MAX_RETRIES": "5",
            "AGENT_RETRY_DELAY": "10",
            "AGENT_DEFAULT_TIMEOUT": "1200",
            "AGENT_LLM_ANALYSIS": "false",
            "AGENT_LLM_ERROR_THRESHOLD": "0.15",
            "AGENT_EMAIL_ENABLED": "true",
            "AGENT_EMAIL_RECIPIENT": "test@example.com",
            "AGENT_SMTP_HOST": "smtp.example.com",
            "AGENT_SMTP_PORT": "587",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from src.agent.config import AgentSettings

            settings = AgentSettings()

            assert settings.dry_run is True
            assert settings.max_retries == 5
            assert settings.retry_delay_seconds == 10
            assert settings.default_timeout_seconds == 1200
            assert settings.llm_analysis_enabled is False
            assert settings.llm_error_threshold == 0.15
            assert settings.email_enabled is True
            assert settings.email_recipient == "test@example.com"
            assert settings.smtp_host == "smtp.example.com"
            assert settings.smtp_port == 587

    def test_state_file_property(self, tmp_path):
        """Test state_file property."""
        with patch.dict(os.environ, {"AGENT_STATE_DIR": str(tmp_path)}):
            from src.agent.config import AgentSettings

            settings = AgentSettings()
            assert settings.state_file == tmp_path / "state.yaml"

    def test_history_dir_property(self, tmp_path):
        """Test history_dir property creates directory."""
        with patch.dict(os.environ, {"AGENT_STATE_DIR": str(tmp_path)}):
            from src.agent.config import AgentSettings

            settings = AgentSettings()
            history_dir = settings.history_dir

            assert history_dir == tmp_path / "history"
            assert history_dir.exists()

    def test_directories_created_on_init(self, tmp_path):
        """Test that required directories are created on initialization."""
        state_dir = tmp_path / "agent_state"
        logs_dir = tmp_path / "project" / "logs" / "agent"

        with patch.dict(
            os.environ,
            {
                "AGENT_STATE_DIR": str(state_dir),
                "AGENT_PROJECT_ROOT": str(tmp_path / "project"),
                "AGENT_LOGS_DIR": "logs/agent",
            },
        ):
            from src.agent.config import AgentSettings

            settings = AgentSettings()

            assert state_dir.exists()
            assert logs_dir.exists()

    def test_website_repo_path_none_when_not_set(self, tmp_path):
        """Test website_repo_path is None when not configured."""
        with patch.dict(
            os.environ, {"AGENT_STATE_DIR": str(tmp_path)}, clear=False
        ):
            # Remove the env var if it exists
            os.environ.pop("AGENT_WEBSITE_REPO_PATH", None)

            from src.agent.config import AgentSettings

            settings = AgentSettings()
            assert settings.website_repo_path is None

    def test_website_repo_path_when_set(self, tmp_path):
        """Test website_repo_path when configured."""
        repo_path = tmp_path / "website"
        repo_path.mkdir()

        with patch.dict(
            os.environ,
            {
                "AGENT_STATE_DIR": str(tmp_path),
                "AGENT_WEBSITE_REPO_PATH": str(repo_path),
            },
        ):
            from src.agent.config import AgentSettings

            settings = AgentSettings()
            assert settings.website_repo_path == repo_path
