"""Tests for workflow learning data models."""

from datetime import datetime, timezone

import pytest

from src.workflow_learning.models import (
    AnalysisResult,
    AudioTranscript,
    RecordingSession,
    ScreenContent,
    SessionStatus,
    WorkflowStep,
)


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert SessionStatus.RECORDING.value == "recording"
        assert SessionStatus.STOPPED.value == "stopped"
        assert SessionStatus.ANALYZED.value == "analyzed"
        assert SessionStatus.SKILL_GENERATED.value == "skill_generated"
        assert SessionStatus.FAILED.value == "failed"


class TestScreenContent:
    """Tests for ScreenContent model."""

    def test_create_screen_content(self):
        """Test creating ScreenContent."""
        now = datetime.now(timezone.utc)
        content = ScreenContent(
            timestamp=now,
            app_name="Terminal",
            window_title="bash",
            ocr_text="$ uv run python scripts/train.py",
        )

        assert content.timestamp == now
        assert content.app_name == "Terminal"
        assert content.window_title == "bash"
        assert content.ocr_text == "$ uv run python scripts/train.py"
        assert content.file_path is None

    def test_screen_content_with_file_path(self):
        """Test ScreenContent with screenshot file path."""
        content = ScreenContent(
            timestamp=datetime.now(timezone.utc),
            app_name="Browser",
            window_title="Example",
            ocr_text="Hello",
            file_path="/tmp/screenshot.png",
        )
        assert content.file_path == "/tmp/screenshot.png"


class TestAudioTranscript:
    """Tests for AudioTranscript model."""

    def test_create_audio_transcript(self):
        """Test creating AudioTranscript."""
        now = datetime.now(timezone.utc)
        transcript = AudioTranscript(
            timestamp=now,
            text="Now I'm going to train the classifier",
            duration_seconds=5.2,
            device_name="MacBook Pro Microphone",
            is_input=True,
        )

        assert transcript.timestamp == now
        assert transcript.text == "Now I'm going to train the classifier"
        assert transcript.duration_seconds == 5.2
        assert transcript.device_name == "MacBook Pro Microphone"
        assert transcript.is_input is True

    def test_audio_transcript_defaults(self):
        """Test AudioTranscript default values."""
        transcript = AudioTranscript(
            timestamp=datetime.now(timezone.utc),
            text="Hello",
        )
        assert transcript.duration_seconds == 0.0
        assert transcript.device_name == ""
        assert transcript.is_input is True


class TestWorkflowStep:
    """Tests for WorkflowStep model."""

    def test_create_workflow_step(self):
        """Test creating WorkflowStep."""
        step = WorkflowStep(
            step_number=1,
            action="Run training script",
            target="FP classifier",
            purpose="Train the false positive classifier model",
            command="uv run python scripts/train.py --classifier fp",
            evidence=["$ uv run python", "Training started..."],
            notes="This takes about 2 minutes",
        )

        assert step.step_number == 1
        assert step.action == "Run training script"
        assert step.target == "FP classifier"
        assert step.purpose == "Train the false positive classifier model"
        assert step.command == "uv run python scripts/train.py --classifier fp"
        assert len(step.evidence) == 2
        assert step.notes == "This takes about 2 minutes"

    def test_workflow_step_defaults(self):
        """Test WorkflowStep default values."""
        step = WorkflowStep(step_number=1, action="Do something")
        assert step.target == ""
        assert step.purpose == ""
        assert step.command is None
        assert step.evidence == []
        assert step.notes == ""
        assert step.tool_type == "bash"
        assert step.expected_output == ""
        assert step.success_criteria == ""
        assert step.on_failure == ""
        assert step.category == ""

    def test_workflow_step_notebook_fields(self):
        """Test WorkflowStep with notebook-specific fields."""
        step = WorkflowStep(
            step_number=1,
            action="Train model",
            command="model.fit(X_train, y_train)",
            tool_type="jupyter",
            expected_output="Training completes, shows feature importances",
            success_criteria="F2 score > 0.95",
            on_failure="Adjust hyperparameters and retrain",
            category="core",
        )
        assert step.tool_type == "jupyter"
        assert step.expected_output == "Training completes, shows feature importances"
        assert step.success_criteria == "F2 score > 0.95"
        assert step.on_failure == "Adjust hyperparameters and retrain"
        assert step.category == "core"

    def test_workflow_step_serialization_with_new_fields(self):
        """Test that new fields are included in serialization."""
        step = WorkflowStep(
            step_number=1,
            action="Review output",
            tool_type="review",
            expected_output="Confusion matrix looks good",
            success_criteria="Recall > 95%",
            on_failure="Check training data balance",
            category="verification",
        )
        data = step.model_dump()
        assert data["tool_type"] == "review"
        assert data["expected_output"] == "Confusion matrix looks good"
        assert data["success_criteria"] == "Recall > 95%"
        assert data["on_failure"] == "Check training data balance"
        assert data["category"] == "verification"

    def test_workflow_step_deserialization_without_new_fields(self):
        """Test backwards compatibility — old data without new fields still works."""
        data = {
            "step_number": 1,
            "action": "Run script",
            "command": "uv run python scripts/train.py",
        }
        step = WorkflowStep(**data)
        assert step.tool_type == "bash"
        assert step.expected_output == ""
        assert step.success_criteria == ""
        assert step.on_failure == ""
        assert step.category == ""


class TestRecordingSession:
    """Tests for RecordingSession model."""

    def test_create_recording_session(self):
        """Test creating RecordingSession."""
        now = datetime.now(timezone.utc)
        session = RecordingSession(
            session_id="test-session-20240101_120000",
            workflow_name="model-training",
            description="Training the FP classifier",
            status=SessionStatus.RECORDING,
            started_at=now,
        )

        assert session.session_id == "test-session-20240101_120000"
        assert session.workflow_name == "model-training"
        assert session.description == "Training the FP classifier"
        assert session.status == SessionStatus.RECORDING
        assert session.started_at == now
        assert session.stopped_at is None
        assert session.screen_content == []
        assert session.audio_transcripts == []

    def test_recording_session_to_dict(self):
        """Test RecordingSession serialization to dict."""
        now = datetime.now(timezone.utc)
        session = RecordingSession(
            session_id="test-session",
            workflow_name="test",
            started_at=now,
            screen_content=[
                ScreenContent(
                    timestamp=now,
                    app_name="Terminal",
                    window_title="bash",
                    ocr_text="$ ls",
                )
            ],
            extracted_steps=[
                WorkflowStep(step_number=1, action="List files", command="ls")
            ],
        )

        data = session.to_dict()

        assert data["session_id"] == "test-session"
        assert data["workflow_name"] == "test"
        assert data["status"] == "recording"
        assert data["started_at"] == now.isoformat()
        assert len(data["screen_content"]) == 1
        assert len(data["extracted_steps"]) == 1

    def test_recording_session_from_dict(self):
        """Test RecordingSession deserialization from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "session_id": "test-session",
            "workflow_name": "test",
            "description": "Test session",
            "status": "stopped",
            "started_at": now.isoformat(),
            "stopped_at": now.isoformat(),
            "analyzed_at": None,
            "skill_generated_at": None,
            "screen_content": [
                {
                    "timestamp": now.isoformat(),
                    "app_name": "Terminal",
                    "window_title": "bash",
                    "ocr_text": "$ ls",
                }
            ],
            "audio_transcripts": [],
            "extracted_steps": [
                {"step_number": 1, "action": "List files", "command": "ls"}
            ],
            "analysis_summary": "",
            "skill_path": None,
            "error": None,
        }

        session = RecordingSession.from_dict(data)

        assert session.session_id == "test-session"
        assert session.workflow_name == "test"
        assert session.status == SessionStatus.STOPPED
        assert len(session.screen_content) == 1
        assert session.screen_content[0].app_name == "Terminal"
        assert len(session.extracted_steps) == 1
        assert session.extracted_steps[0].action == "List files"

    def test_recording_session_roundtrip(self):
        """Test RecordingSession serialization roundtrip."""
        now = datetime.now(timezone.utc)
        original = RecordingSession(
            session_id="roundtrip-test",
            workflow_name="test-workflow",
            description="Testing roundtrip",
            status=SessionStatus.ANALYZED,
            started_at=now,
            stopped_at=now,
            analyzed_at=now,
            screen_content=[
                ScreenContent(
                    timestamp=now,
                    app_name="VSCode",
                    window_title="editor.py",
                    ocr_text="def main():",
                )
            ],
            audio_transcripts=[
                AudioTranscript(
                    timestamp=now,
                    text="Opening the main function",
                    duration_seconds=2.5,
                )
            ],
            extracted_steps=[
                WorkflowStep(
                    step_number=1,
                    action="Open file",
                    target="editor.py",
                    purpose="Edit the main function",
                )
            ],
            analysis_summary="Test analysis",
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = RecordingSession.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.workflow_name == original.workflow_name
        assert restored.status == original.status
        assert len(restored.screen_content) == 1
        assert len(restored.audio_transcripts) == 1
        assert len(restored.extracted_steps) == 1
        assert restored.analysis_summary == original.analysis_summary


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_create_successful_result(self):
        """Test creating successful AnalysisResult."""
        result = AnalysisResult(
            success=True,
            steps=[
                WorkflowStep(step_number=1, action="Test step"),
            ],
            summary="This workflow does testing",
            skill_name="test-workflow",
            skill_description="A test workflow",
            input_tokens=1000,
            output_tokens=500,
            model="claude-haiku-4-5-20251001",
        )

        assert result.success is True
        assert len(result.steps) == 1
        assert result.summary == "This workflow does testing"
        assert result.skill_name == "test-workflow"
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating failed AnalysisResult."""
        result = AnalysisResult(
            success=False,
            error="API rate limited",
            model="claude-haiku-4-5-20251001",
        )

        assert result.success is False
        assert result.error == "API rate limited"
        assert result.steps == []
