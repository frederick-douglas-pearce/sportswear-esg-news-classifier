"""Session management for workflow recording."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import workflow_learning_settings
from .models import RecordingSession, SessionStatus

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages recording session state with YAML persistence."""

    def __init__(self, sessions_dir: Path | None = None):
        """Initialize session manager.

        Args:
            sessions_dir: Directory for session files. Defaults to config setting.
        """
        self.sessions_dir = sessions_dir or workflow_learning_settings.sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_session_file = self.sessions_dir / "_active.yaml"

    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.sessions_dir / f"{session_id}.yaml"

    def _load_session(self, session_id: str) -> RecordingSession | None:
        """Load session from file."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return None

        try:
            with open(session_file) as f:
                data = yaml.safe_load(f)
            return RecordingSession.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def _save_session(self, session: RecordingSession) -> None:
        """Save session to file atomically."""
        session_file = self._get_session_file(session.session_id)
        temp_file = session_file.with_suffix(".yaml.tmp")

        with open(temp_file, "w") as f:
            yaml.dump(session.to_dict(), f, default_flow_style=False, sort_keys=False)

        temp_file.rename(session_file)
        logger.debug(f"Saved session to {session_file}")

    def _get_active_session_id(self) -> str | None:
        """Get currently active session ID."""
        if not self._active_session_file.exists():
            return None

        try:
            with open(self._active_session_file) as f:
                data = yaml.safe_load(f)
            return data.get("session_id")
        except Exception:
            return None

    def _set_active_session(self, session_id: str | None) -> None:
        """Set or clear the active session."""
        if session_id is None:
            if self._active_session_file.exists():
                self._active_session_file.unlink()
        else:
            with open(self._active_session_file, "w") as f:
                yaml.dump({"session_id": session_id}, f)

    def start_session(
        self,
        workflow_name: str,
        description: str = "",
    ) -> RecordingSession:
        """Start a new recording session.

        Args:
            workflow_name: Name of the workflow (e.g., 'model-training-fp')
            description: Optional description of what will be demonstrated

        Returns:
            Created RecordingSession

        Raises:
            ValueError: If there's already an active session
        """
        # Check for existing active session
        active_id = self._get_active_session_id()
        if active_id:
            active_session = self._load_session(active_id)
            if active_session and active_session.status == SessionStatus.RECORDING:
                raise ValueError(
                    f"Session '{active_id}' is already recording. "
                    "Stop it first with 'stop' command."
                )

        # Generate session ID from timestamp
        now = datetime.now(timezone.utc)
        session_id = f"{workflow_name}_{now.strftime('%Y%m%d_%H%M%S')}"

        session = RecordingSession(
            session_id=session_id,
            workflow_name=workflow_name,
            description=description,
            status=SessionStatus.RECORDING,
            started_at=now,
        )

        self._save_session(session)
        self._set_active_session(session_id)

        logger.info(f"Started recording session: {session_id}")
        return session

    def stop_session(self, session_id: str | None = None) -> RecordingSession:
        """Stop a recording session.

        Args:
            session_id: Session to stop. If None, stops the active session.

        Returns:
            Updated RecordingSession

        Raises:
            ValueError: If no session found or session not recording
        """
        if session_id is None:
            session_id = self._get_active_session_id()
            if not session_id:
                raise ValueError("No active recording session to stop.")

        session = self._load_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")

        if session.status != SessionStatus.RECORDING:
            raise ValueError(
                f"Session '{session_id}' is not recording (status: {session.status.value})."
            )

        session.status = SessionStatus.STOPPED
        session.stopped_at = datetime.now(timezone.utc)

        self._save_session(session)
        self._set_active_session(None)

        logger.info(f"Stopped recording session: {session_id}")
        return session

    def get_session(self, session_id: str) -> RecordingSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            RecordingSession or None if not found
        """
        return self._load_session(session_id)

    def get_active_session(self) -> RecordingSession | None:
        """Get the currently active (recording) session.

        Returns:
            RecordingSession or None if no active session
        """
        session_id = self._get_active_session_id()
        if not session_id:
            return None
        return self._load_session(session_id)

    def update_session(self, session: RecordingSession) -> None:
        """Update a session.

        Args:
            session: Session to update
        """
        self._save_session(session)
        logger.debug(f"Updated session: {session.session_id}")

    def list_sessions(self) -> list[RecordingSession]:
        """List all sessions.

        Returns:
            List of all sessions, sorted by start time (newest first)
        """
        sessions = []
        for session_file in self.sessions_dir.glob("*.yaml"):
            # Skip active marker file
            if session_file.name.startswith("_"):
                continue

            session_id = session_file.stem
            session = self._load_session(session_id)
            if session:
                sessions.append(session)

        # Sort by start time, newest first
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return False

        # Clear active marker if this is the active session
        if self._get_active_session_id() == session_id:
            self._set_active_session(None)

        session_file.unlink()
        logger.info(f"Deleted session: {session_id}")
        return True


# Global session manager instance
session_manager = SessionManager()
