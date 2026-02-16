"""Tests for session manager."""

from datetime import datetime, timezone

import pytest

from src.workflow_learning.models import (
    RecordingSession,
    ScreenContent,
    SessionStatus,
    WorkflowStep,
)
from src.workflow_learning.session_manager import SessionManager


@pytest.fixture
def temp_sessions_dir(tmp_path):
    """Create a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def session_manager(temp_sessions_dir):
    """Create a SessionManager with temporary directory."""
    return SessionManager(sessions_dir=temp_sessions_dir)


class TestSessionManager:
    """Tests for SessionManager."""

    def test_start_session(self, session_manager):
        """Test starting a new recording session."""
        session = session_manager.start_session(
            workflow_name="test-workflow",
            description="Testing the workflow",
        )

        assert session.workflow_name == "test-workflow"
        assert session.description == "Testing the workflow"
        assert session.status == SessionStatus.RECORDING
        assert session.started_at is not None
        assert "test-workflow" in session.session_id

    def test_start_session_creates_file(self, session_manager, temp_sessions_dir):
        """Test that starting a session creates a YAML file."""
        session = session_manager.start_session(workflow_name="test")

        session_file = temp_sessions_dir / f"{session.session_id}.yaml"
        assert session_file.exists()

    def test_start_session_sets_active(self, session_manager, temp_sessions_dir):
        """Test that starting a session sets it as active."""
        session = session_manager.start_session(workflow_name="test")

        active_file = temp_sessions_dir / "_active.yaml"
        assert active_file.exists()

        active_session = session_manager.get_active_session()
        assert active_session.session_id == session.session_id

    def test_start_session_fails_if_active(self, session_manager):
        """Test that starting a session fails if one is already active."""
        session_manager.start_session(workflow_name="first")

        with pytest.raises(ValueError, match="already recording"):
            session_manager.start_session(workflow_name="second")

    def test_stop_session(self, session_manager):
        """Test stopping a recording session."""
        session = session_manager.start_session(workflow_name="test")
        session_id = session.session_id

        stopped = session_manager.stop_session()

        assert stopped.session_id == session_id
        assert stopped.status == SessionStatus.STOPPED
        assert stopped.stopped_at is not None

    def test_stop_session_clears_active(self, session_manager):
        """Test that stopping a session clears the active marker."""
        session_manager.start_session(workflow_name="test")
        session_manager.stop_session()

        assert session_manager.get_active_session() is None

    def test_stop_session_by_id(self, session_manager):
        """Test stopping a session by ID."""
        session = session_manager.start_session(workflow_name="test")
        session_manager._set_active_session(None)  # Clear active marker

        stopped = session_manager.stop_session(session_id=session.session_id)
        assert stopped.status == SessionStatus.STOPPED

    def test_stop_session_fails_no_active(self, session_manager):
        """Test that stopping fails when no active session."""
        with pytest.raises(ValueError, match="No active"):
            session_manager.stop_session()

    def test_stop_session_fails_not_recording(self, session_manager):
        """Test that stopping fails for non-recording session."""
        session = session_manager.start_session(workflow_name="test")
        session_manager.stop_session()

        with pytest.raises(ValueError, match="not recording"):
            session_manager.stop_session(session_id=session.session_id)

    def test_get_session(self, session_manager):
        """Test getting a session by ID."""
        session = session_manager.start_session(workflow_name="test")

        retrieved = session_manager.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_session_not_found(self, session_manager):
        """Test getting a non-existent session."""
        result = session_manager.get_session("nonexistent")
        assert result is None

    def test_update_session(self, session_manager):
        """Test updating a session."""
        session = session_manager.start_session(workflow_name="test")

        # Add some content
        session.screen_content.append(
            ScreenContent(
                timestamp=datetime.now(timezone.utc),
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ ls",
            )
        )
        session_manager.update_session(session)

        # Reload and verify
        reloaded = session_manager.get_session(session.session_id)
        assert len(reloaded.screen_content) == 1
        assert reloaded.screen_content[0].ocr_text == "$ ls"

    def test_list_sessions(self, session_manager):
        """Test listing all sessions."""
        # Create multiple sessions
        s1 = session_manager.start_session(workflow_name="first")
        session_manager.stop_session()

        s2 = session_manager.start_session(workflow_name="second")
        session_manager.stop_session()

        sessions = session_manager.list_sessions()

        assert len(sessions) == 2
        # Should be sorted newest first
        assert sessions[0].session_id == s2.session_id
        assert sessions[1].session_id == s1.session_id

    def test_list_sessions_empty(self, session_manager):
        """Test listing sessions when none exist."""
        sessions = session_manager.list_sessions()
        assert sessions == []

    def test_delete_session(self, session_manager, temp_sessions_dir):
        """Test deleting a session."""
        session = session_manager.start_session(workflow_name="test")
        session_manager.stop_session()
        session_id = session.session_id

        result = session_manager.delete_session(session_id)

        assert result is True
        assert not (temp_sessions_dir / f"{session_id}.yaml").exists()
        assert session_manager.get_session(session_id) is None

    def test_delete_session_clears_active(self, session_manager):
        """Test that deleting active session clears active marker."""
        session = session_manager.start_session(workflow_name="test")

        session_manager.delete_session(session.session_id)

        assert session_manager.get_active_session() is None

    def test_delete_session_not_found(self, session_manager):
        """Test deleting a non-existent session."""
        result = session_manager.delete_session("nonexistent")
        assert result is False

    def test_session_persistence(self, temp_sessions_dir):
        """Test that sessions persist across manager instances."""
        # Create session with first manager
        manager1 = SessionManager(sessions_dir=temp_sessions_dir)
        session = manager1.start_session(workflow_name="persist-test")
        session.extracted_steps.append(
            WorkflowStep(step_number=1, action="Test", command="test")
        )
        manager1.update_session(session)
        manager1.stop_session()
        session_id = session.session_id

        # Load with new manager
        manager2 = SessionManager(sessions_dir=temp_sessions_dir)
        reloaded = manager2.get_session(session_id)

        assert reloaded is not None
        assert reloaded.workflow_name == "persist-test"
        assert reloaded.status == SessionStatus.STOPPED
        assert len(reloaded.extracted_steps) == 1
