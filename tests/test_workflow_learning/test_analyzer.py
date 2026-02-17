"""Tests for workflow recording analyzer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.workflow_learning.analyzer import RecordingAnalyzer
from src.workflow_learning.models import (
    AudioTranscript,
    RecordingSession,
    ScreenContent,
    SessionStatus,
)


@pytest.fixture
def sample_session():
    """Create a sample recording session with content."""
    now = datetime.now(timezone.utc)
    return RecordingSession(
        session_id="test-session",
        workflow_name="test-workflow",
        description="Test workflow for unit tests",
        status=SessionStatus.STOPPED,
        started_at=now,
        stopped_at=now,
        screen_content=[
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ uv run python scripts/train.py --classifier fp\nTraining started...",
            ),
            ScreenContent(
                timestamp=now + timedelta(seconds=10),
                app_name="Terminal",
                window_title="bash",
                ocr_text="Model saved to models/fp_classifier.joblib",
            ),
        ],
        audio_transcripts=[
            AudioTranscript(
                timestamp=now + timedelta(seconds=1),
                text="Now I'm going to train the false positive classifier",
                duration_seconds=3.0,
            ),
            AudioTranscript(
                timestamp=now + timedelta(seconds=12),
                text="The training is complete and the model is saved",
                duration_seconds=2.5,
            ),
        ],
    )


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client."""
    with patch("src.workflow_learning.analyzer.Anthropic") as mock:
        yield mock


class TestRecordingAnalyzer:
    """Tests for RecordingAnalyzer."""

    def test_init_requires_api_key(self):
        """Test that analyzer requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "src.workflow_learning.config.workflow_learning_settings.anthropic_api_key",
                None,
            ):
                with pytest.raises(ValueError, match="API key required"):
                    RecordingAnalyzer(api_key=None)

    def test_init_with_api_key(self, mock_anthropic):
        """Test analyzer initialization with API key."""
        analyzer = RecordingAnalyzer(api_key="test-key")
        assert analyzer.api_key == "test-key"
        mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_analyze_empty_session(self, mock_anthropic):
        """Test analyzing session with no content."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        empty_session = RecordingSession(
            session_id="empty",
            workflow_name="empty",
            started_at=datetime.now(timezone.utc),
        )

        result = analyzer.analyze_session(empty_session)

        assert result.success is False
        assert "No screen content" in result.error

    def test_analyze_session_success(self, mock_anthropic, sample_session):
        """Test successful session analysis."""
        # Mock Claude response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""```json
{
    "skill_name": "train-fp-classifier",
    "skill_description": "Train the false positive classifier",
    "summary": "This workflow trains the FP classifier model.",
    "steps": [
        {
            "step_number": 1,
            "action": "Run training script",
            "target": "FP classifier",
            "purpose": "Train the model",
            "command": "uv run python scripts/train.py --classifier fp",
            "category": "core",
            "evidence": ["Training started..."],
            "notes": "Training the false positive classifier"
        }
    ]
}
```"""
            )
        ]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=500)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        analyzer = RecordingAnalyzer(api_key="test-key")
        result = analyzer.analyze_session(sample_session)

        assert result.success is True
        assert result.skill_name == "train-fp-classifier"
        assert len(result.steps) == 1
        assert result.steps[0].action == "Run training script"
        assert result.steps[0].command == "uv run python scripts/train.py --classifier fp"
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_analyze_session_uses_timeline_format(self, mock_anthropic, sample_session):
        """Test that analyze_session uses the timeline format in the prompt."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"skill_name": "test", "summary": "", "steps": []}')
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        analyzer = RecordingAnalyzer(api_key="test-key")
        analyzer.analyze_session(sample_session)

        # Check that the user prompt contains timeline markers
        call_args = mock_client.messages.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "## Timeline" in user_msg
        assert "SCREEN |" in user_msg
        assert "AUDIO" in user_msg
        # Should NOT contain old separate sections
        assert "## Screen Content (OCR)" not in user_msg
        assert "## Audio Transcription" not in user_msg

    def test_analyze_session_includes_project_context(self, mock_anthropic, sample_session):
        """Test that analyze_session includes project context in system prompt."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"skill_name": "test", "summary": "", "steps": []}')
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        analyzer = RecordingAnalyzer(api_key="test-key")
        analyzer.analyze_session(sample_session)

        call_args = mock_client.messages.create.call_args
        system_prompt = call_args[1]["system"]
        assert "Project Context" in system_prompt

    def test_analyze_session_rate_limit_retry(self, mock_anthropic, sample_session):
        """Test that analysis retries on rate limit."""
        from anthropic import RateLimitError

        # First call raises rate limit, second succeeds
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"skill_name": "test", "summary": "", "steps": []}'
            )
        ]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            RateLimitError("Rate limited", response=MagicMock(), body={}),
            mock_response,
        ]
        mock_anthropic.return_value = mock_client

        analyzer = RecordingAnalyzer(api_key="test-key", retry_delay=0.01)

        with patch("time.sleep"):  # Speed up test
            result = analyzer.analyze_session(sample_session)

        assert result.success is True
        assert mock_client.messages.create.call_count == 2

    def test_analyze_session_max_retries_exceeded(self, mock_anthropic, sample_session):
        """Test that analysis fails after max retries."""
        from anthropic import RateLimitError

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RateLimitError(
            "Rate limited", response=MagicMock(), body={}
        )
        mock_anthropic.return_value = mock_client

        analyzer = RecordingAnalyzer(api_key="test-key", max_retries=2, retry_delay=0.01)

        with patch("time.sleep"):
            result = analyzer.analyze_session(sample_session)

        assert result.success is False
        assert "Rate limit exceeded" in result.error


class TestDeduplicateFrames:
    """Tests for OCR frame deduplication."""

    def test_identical_frames_deduplicated(self, mock_anthropic):
        """Test that identical consecutive frames are removed."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        frames = [
            ScreenContent(
                timestamp=now + timedelta(seconds=i),
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ uv run python scripts/train.py\nTraining started...",
            )
            for i in range(5)
        ]

        result = analyzer._deduplicate_frames(frames)

        # First + last = 2 (last is added back since it was deduped)
        assert len(result) == 2
        assert result[0] is frames[0]
        assert result[-1] is frames[-1]

    def test_all_unique_frames_kept(self, mock_anthropic):
        """Test that all unique frames are kept."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        unique_texts = [
            "$ uv run python scripts/train.py --classifier fp\nTraining started...",
            "Opening notebook fp1_EDA_FE.ipynb in Jupyter\nCell 1: import pandas as pd",
            "SELECT COUNT(*) FROM articles WHERE labeling_status = 'labeled'\n  count: 1523",
            "docker compose up -d\nCreating network sportswear_default\nStarting postgres_1",
            "git status\nOn branch main\nChanges not staged for commit:\n  modified: src/agent/config.py",
        ]
        frames = [
            ScreenContent(
                timestamp=now + timedelta(seconds=i * 10),
                app_name="Terminal",
                window_title="bash",
                ocr_text=text,
            )
            for i, text in enumerate(unique_texts)
        ]

        result = analyzer._deduplicate_frames(frames)

        assert len(result) == 5

    def test_threshold_behavior(self, mock_anthropic):
        """Test that threshold controls dedup sensitivity."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        # Two frames that are similar but not identical
        frames = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ uv run python scripts/train.py\nTraining started... epoch 1",
            ),
            ScreenContent(
                timestamp=now + timedelta(seconds=1),
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ uv run python scripts/train.py\nTraining started... epoch 2",
            ),
        ]

        # With very strict threshold (0.99), both should be kept
        result_strict = analyzer._deduplicate_frames(frames, similarity_threshold=0.99)
        assert len(result_strict) == 2

        # With very loose threshold (0.5), second should be deduped
        result_loose = analyzer._deduplicate_frames(frames, similarity_threshold=0.5)
        # First + last (added back) = 2
        assert len(result_loose) == 2

    def test_empty_input(self, mock_anthropic):
        """Test empty frame list."""
        analyzer = RecordingAnalyzer(api_key="test-key")
        result = analyzer._deduplicate_frames([])
        assert result == []

    def test_single_frame(self, mock_anthropic):
        """Test single frame is returned as-is."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        frames = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="Some content",
            )
        ]

        result = analyzer._deduplicate_frames(frames)
        assert len(result) == 1
        assert result[0] is frames[0]

    def test_last_frame_always_kept(self, mock_anthropic):
        """Test that the last frame is always kept even if duplicate."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        frames = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="Same content here",
            ),
            ScreenContent(
                timestamp=now + timedelta(seconds=1),
                app_name="Terminal",
                window_title="bash",
                ocr_text="Different content entirely with new text",
            ),
            ScreenContent(
                timestamp=now + timedelta(seconds=2),
                app_name="Terminal",
                window_title="bash",
                ocr_text="Same content here",  # Same as first
            ),
        ]

        result = analyzer._deduplicate_frames(frames)

        assert frames[-1] in result


class TestFormatTimeline:
    """Tests for chronological timeline interleaving."""

    def test_interleaving_order(self, mock_anthropic):
        """Test that screen and audio entries are interleaved by timestamp."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        screen = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ uv run python scripts/train.py",
            ),
            ScreenContent(
                timestamp=now + timedelta(seconds=10),
                app_name="Terminal",
                window_title="bash",
                ocr_text="Training complete",
            ),
        ]
        audio = [
            AudioTranscript(
                timestamp=now + timedelta(seconds=5),
                text="Running the training script now",
                duration_seconds=2.0,
            ),
        ]

        result = analyzer._format_timeline(screen, audio)

        # Check order: screen(0s) -> audio(5s) -> screen(10s)
        screen1_pos = result.index("$ uv run python scripts/train.py")
        audio_pos = result.index("Running the training script now")
        screen2_pos = result.index("Training complete")

        assert screen1_pos < audio_pos < screen2_pos

    def test_screen_format(self, mock_anthropic):
        """Test screen entry format in timeline."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        screen = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ ls\nfile.py",
            ),
        ]

        result = analyzer._format_timeline(screen, [])

        assert "SCREEN | Terminal - bash" in result
        assert "$ ls\nfile.py" in result

    def test_audio_format(self, mock_anthropic):
        """Test audio entry format in timeline."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        audio = [
            AudioTranscript(
                timestamp=now,
                text="Now I'm training the model",
                duration_seconds=2.0,
            ),
        ]

        result = analyzer._format_timeline([], audio)

        assert "AUDIO" in result
        assert '"Now I\'m training the model"' in result

    def test_empty_audio_skipped(self, mock_anthropic):
        """Test that empty audio transcripts are skipped."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        audio = [
            AudioTranscript(timestamp=now, text="", duration_seconds=0.0),
            AudioTranscript(timestamp=now, text="   ", duration_seconds=0.0),
            AudioTranscript(
                timestamp=now + timedelta(seconds=1),
                text="Actual content",
                duration_seconds=1.0,
            ),
        ]

        result = analyzer._format_timeline([], audio)

        assert "Actual content" in result
        assert result.count("AUDIO") == 1

    def test_truncation(self, mock_anthropic):
        """Test that timeline is truncated when exceeding max_chars."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        screen = [
            ScreenContent(
                timestamp=now + timedelta(seconds=i),
                app_name="Terminal",
                window_title="bash",
                ocr_text="x" * 500,
            )
            for i in range(100)
        ]

        result = analyzer._format_timeline(screen, [], max_chars=2000)

        assert "truncated" in result.lower()
        assert len(result) < 5000

    def test_empty_inputs(self, mock_anthropic):
        """Test with no screen content or audio."""
        analyzer = RecordingAnalyzer(api_key="test-key")
        result = analyzer._format_timeline([], [])
        assert result == ""


class TestLoadProjectContext:
    """Tests for project context loading."""

    def test_loads_existing_file(self, mock_anthropic, tmp_path):
        """Test loading project context from existing file."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        context_file = tmp_path / "project_context.md"
        context_file.write_text("# My Project\n\nSome commands here.")

        result = analyzer._load_project_context(context_file)

        assert "## Project Context" in result
        assert "# My Project" in result
        assert "Some commands here." in result

    def test_missing_file_graceful_fallback(self, mock_anthropic, tmp_path):
        """Test graceful fallback when file doesn't exist."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        missing_file = tmp_path / "nonexistent.md"
        result = analyzer._load_project_context(missing_file)

        assert "## Project Context" in result
        assert "No project context available" in result

    def test_default_path(self, mock_anthropic):
        """Test that default path points to prompts directory."""
        from src.workflow_learning.analyzer import PROJECT_CONTEXT_PATH

        assert "prompts" in str(PROJECT_CONTEXT_PATH)
        assert "workflow_learning" in str(PROJECT_CONTEXT_PATH)
        assert str(PROJECT_CONTEXT_PATH).endswith("project_context.md")


class TestFormatScreenContent:
    """Tests for the legacy _format_screen_content method."""

    def test_format_screen_content(self, mock_anthropic):
        """Test formatting screen content for prompt."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        content = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="$ ls\nfile1.py\nfile2.py",
            ),
            ScreenContent(
                timestamp=now,
                app_name="VSCode",
                window_title="editor.py",
                ocr_text="def main():\n    pass",
            ),
        ]

        result = analyzer._format_screen_content(content)

        assert "Terminal" in result
        assert "bash" in result
        assert "$ ls" in result
        assert "VSCode" in result
        assert "def main()" in result

    def test_format_screen_content_truncates(self, mock_anthropic):
        """Test that screen content is truncated when too long."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        # Create content that exceeds max_chars
        long_text = "x" * 3000
        content = [
            ScreenContent(
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text=long_text,
            )
            for _ in range(30)  # Will exceed 50000 chars
        ]

        result = analyzer._format_screen_content(content, max_chars=5000)

        assert len(result) < 10000  # Should be truncated
        assert "truncated" in result.lower()


class TestFormatAudioTranscripts:
    """Tests for the legacy _format_audio_transcripts method."""

    def test_format_audio_transcripts(self, mock_anthropic):
        """Test formatting audio transcripts for prompt."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        transcripts = [
            AudioTranscript(
                timestamp=now,
                text="First I'll open the file",
                duration_seconds=2.0,
            ),
            AudioTranscript(
                timestamp=now,
                text="Now running the command",
                duration_seconds=1.5,
            ),
        ]

        result = analyzer._format_audio_transcripts(transcripts)

        assert "First I'll open the file" in result
        assert "Now running the command" in result

    def test_format_audio_transcripts_skips_empty(self, mock_anthropic):
        """Test that empty transcripts are skipped."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        now = datetime.now(timezone.utc)
        transcripts = [
            AudioTranscript(timestamp=now, text="", duration_seconds=0.0),
            AudioTranscript(timestamp=now, text="   ", duration_seconds=0.0),
            AudioTranscript(timestamp=now, text="Actual content", duration_seconds=1.0),
        ]

        result = analyzer._format_audio_transcripts(transcripts)

        assert "Actual content" in result
        # Should only have one line (plus possible empty lines)
        assert result.strip().count("\n") == 0


class TestParseJsonResponse:
    """Tests for JSON parsing."""

    def test_parse_json_response_code_block(self, mock_anthropic):
        """Test parsing JSON from markdown code block."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        response = """Here's the analysis:

```json
{
    "skill_name": "test",
    "summary": "A test workflow"
}
```

That's the result."""

        result = analyzer._parse_json_response(response)

        assert result["skill_name"] == "test"
        assert result["summary"] == "A test workflow"

    def test_parse_json_response_no_code_block(self, mock_anthropic):
        """Test parsing JSON without code block."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        response = '{"skill_name": "test", "summary": "A test"}'

        result = analyzer._parse_json_response(response)

        assert result["skill_name"] == "test"

    def test_parse_json_response_invalid(self, mock_anthropic):
        """Test handling invalid JSON response."""
        analyzer = RecordingAnalyzer(api_key="test-key")

        response = "This is not JSON at all, just text."

        result = analyzer._parse_json_response(response)

        assert "raw_response" in result
