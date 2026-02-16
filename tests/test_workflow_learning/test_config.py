"""Tests for workflow learning configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestWorkflowLearningSettings:
    """Tests for WorkflowLearningSettings."""

    def test_default_screenpipe_url(self):
        """Test default Screenpipe API URL."""
        # Import fresh to get defaults
        with patch.dict(os.environ, {}, clear=True):
            from src.workflow_learning.config import WorkflowLearningSettings

            settings = WorkflowLearningSettings()
            assert settings.screenpipe_api_url == "http://localhost:3030"

    def test_custom_screenpipe_url_from_env(self):
        """Test Screenpipe URL from environment variable."""
        with patch.dict(os.environ, {"SCREENPIPE_API_URL": "http://custom:9999"}):
            from src.workflow_learning.config import WorkflowLearningSettings

            settings = WorkflowLearningSettings()
            assert settings.screenpipe_api_url == "http://custom:9999"

    def test_default_analysis_model(self):
        """Test default analysis model."""
        with patch.dict(os.environ, {}, clear=True):
            from src.workflow_learning.config import WorkflowLearningSettings

            settings = WorkflowLearningSettings()
            assert settings.analysis_model == "claude-sonnet-4-20250514"

    def test_sessions_dir_property(self, tmp_path):
        """Test sessions_dir property."""
        from src.workflow_learning.config import WorkflowLearningSettings

        settings = WorkflowLearningSettings()
        settings.recording_output_dir = tmp_path / "recordings"
        settings.recording_output_dir.mkdir(parents=True, exist_ok=True)

        assert settings.sessions_dir == tmp_path / "recordings" / "sessions"

    def test_max_screen_frames_default(self):
        """Test default max screen frames."""
        with patch.dict(os.environ, {}, clear=True):
            from src.workflow_learning.config import WorkflowLearningSettings

            settings = WorkflowLearningSettings()
            assert settings.max_screen_frames == 1000

    def test_max_audio_chunks_default(self):
        """Test default max audio chunks."""
        with patch.dict(os.environ, {}, clear=True):
            from src.workflow_learning.config import WorkflowLearningSettings

            settings = WorkflowLearningSettings()
            assert settings.max_audio_chunks == 500

    def test_directories_created_on_init(self, tmp_path):
        """Test that directories are created on initialization."""
        from src.workflow_learning.config import WorkflowLearningSettings

        recording_dir = tmp_path / "new_recordings"
        skills_dir = tmp_path / "new_skills"

        with patch.dict(
            os.environ,
            {
                "WORKFLOW_RECORDING_DIR": str(recording_dir),
                "WORKFLOW_SKILLS_DIR": str(skills_dir),
            },
        ):
            settings = WorkflowLearningSettings()

            assert recording_dir.exists()
            assert (recording_dir / "sessions").exists()
            assert skills_dir.exists()
