"""Tests for workflow recording analyzer."""

from datetime import datetime, timezone
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
                timestamp=now,
                app_name="Terminal",
                window_title="bash",
                ocr_text="Model saved to models/fp_classifier.joblib",
            ),
        ],
        audio_transcripts=[
            AudioTranscript(
                timestamp=now,
                text="Now I'm going to train the false positive classifier",
                duration_seconds=3.0,
            ),
            AudioTranscript(
                timestamp=now,
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
