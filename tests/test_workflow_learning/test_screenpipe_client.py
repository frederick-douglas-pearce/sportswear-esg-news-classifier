"""Tests for Screenpipe client."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.workflow_learning.models import AudioTranscript, ScreenContent
from src.workflow_learning.screenpipe_client import ScreenpipeClient, ScreenpipeError


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    with patch("httpx.Client") as mock:
        yield mock.return_value


@pytest.fixture
def client(mock_httpx_client):
    """Create a ScreenpipeClient with mocked HTTP client."""
    client = ScreenpipeClient()
    client._client = mock_httpx_client
    return client


class TestScreenpipeClient:
    """Tests for ScreenpipeClient."""

    def test_health_check_success(self, client, mock_httpx_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx_client.get.return_value = mock_response

        result = client.health_check()

        assert result is True
        mock_httpx_client.get.assert_called_once_with("http://localhost:3030/health")

    def test_health_check_failure(self, client, mock_httpx_client):
        """Test failed health check."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_httpx_client.get.return_value = mock_response

        result = client.health_check()

        assert result is False

    def test_health_check_connection_error(self, client, mock_httpx_client):
        """Test health check with connection error."""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")

        result = client.health_check()

        assert result is False

    def test_search_content_basic(self, client, mock_httpx_client):
        """Test basic content search."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        result = client.search_content(start_time=start, end_time=end)

        assert result == {"data": []}
        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert call_args[0][0] == "http://localhost:3030/search"

    def test_search_content_with_filters(self, client, mock_httpx_client):
        """Test content search with filters."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        client.search_content(
            start_time=start,
            end_time=end,
            content_type="ocr",
            query="python",
            app_name="Terminal",
            limit=50,
        )

        call_args = mock_httpx_client.get.call_args
        params = call_args[1]["params"]
        assert params["content_type"] == "ocr"
        assert params["q"] == "python"
        assert params["app_name"] == "Terminal"
        assert params["limit"] == 50

    def test_search_content_http_error(self, client, mock_httpx_client):
        """Test content search with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_httpx_client.get.return_value = mock_response

        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ScreenpipeError, match="API error"):
            client.search_content(start_time=start, end_time=end)

    def test_search_content_connection_error(self, client, mock_httpx_client):
        """Test content search with connection error."""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")

        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ScreenpipeError, match="connection error"):
            client.search_content(start_time=start, end_time=end)

    def test_get_screen_content(self, client, mock_httpx_client):
        """Test retrieving screen content."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "content": {
                        "type": "OCR",
                        "timestamp": "2024-01-01T12:30:00Z",
                        "app_name": "Terminal",
                        "window_name": "bash",
                        "text": "$ ls -la",
                    }
                },
                {
                    "content": {
                        "type": "OCR",
                        "timestamp": "2024-01-01T12:31:00Z",
                        "app_name": "VSCode",
                        "window_name": "editor.py",
                        "text": "def main():",
                    }
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        results = client.get_screen_content(start_time=start, end_time=end)

        assert len(results) == 2
        assert isinstance(results[0], ScreenContent)
        assert results[0].app_name == "Terminal"
        assert results[0].ocr_text == "$ ls -la"
        assert results[1].app_name == "VSCode"

    def test_get_screen_content_respects_max_frames(self, client, mock_httpx_client):
        """Test that get_screen_content passes max_frames as limit to API."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # API returns exactly what was requested (respecting limit)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "content": {
                        "type": "OCR",
                        "timestamp": f"2024-01-01T12:{i:02d}:00Z",
                        "app_name": "Terminal",
                        "window_name": "bash",
                        "text": f"command {i}",
                    }
                }
                for i in range(5)  # API respects limit=5
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        results = client.get_screen_content(
            start_time=start, end_time=end, max_frames=5
        )

        # Verify the limit was passed to the API
        call_args = mock_httpx_client.get.call_args
        params = call_args[1]["params"]
        assert params["limit"] == 5

        assert len(results) == 5

    def test_get_audio_transcriptions(self, client, mock_httpx_client):
        """Test retrieving audio transcriptions."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "content": {
                        "type": "Audio",
                        "timestamp": "2024-01-01T12:30:00Z",
                        "transcription": "Now I'm going to run the training script",
                        "duration_secs": 3.5,
                        "device_name": "MacBook Pro Microphone",
                        "is_input_device": True,
                    }
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.get.return_value = mock_response

        results = client.get_audio_transcriptions(start_time=start, end_time=end)

        assert len(results) == 1
        assert isinstance(results[0], AudioTranscript)
        assert "training script" in results[0].text
        assert results[0].duration_seconds == 3.5
        assert results[0].is_input is True

    def test_parse_timestamp_with_z(self, client):
        """Test parsing timestamp ending with Z."""
        result = client._parse_timestamp("2024-01-01T12:30:00Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30

    def test_parse_timestamp_with_offset(self, client):
        """Test parsing timestamp with offset."""
        result = client._parse_timestamp("2024-01-01T12:30:00+00:00")
        assert result.year == 2024
        assert result.hour == 12

    def test_parse_timestamp_empty(self, client):
        """Test parsing empty timestamp."""
        result = client._parse_timestamp("")
        assert result is not None  # Should return current time

    def test_context_manager(self, mock_httpx_client):
        """Test client as context manager."""
        with ScreenpipeClient() as client:
            client._client = mock_httpx_client
            assert client is not None

        # close() should be called
        mock_httpx_client.close.assert_called_once()


class TestScreenpipeClientPagination:
    """Tests for Screenpipe client pagination."""

    def test_get_screen_content_pagination(self, client, mock_httpx_client):
        """Test that screen content retrieval handles pagination."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # First page - full 100 items
        page1_response = MagicMock()
        page1_response.json.return_value = {
            "data": [
                {
                    "content": {
                        "type": "OCR",
                        "timestamp": f"2024-01-01T12:{i:02d}:00Z",
                        "app_name": "Terminal",
                        "window_name": "bash",
                        "text": f"page1 command {i}",
                    }
                }
                for i in range(100)
            ]
        }
        page1_response.raise_for_status = MagicMock()

        # Second page - partial results (indicating end)
        page2_response = MagicMock()
        page2_response.json.return_value = {
            "data": [
                {
                    "content": {
                        "type": "OCR",
                        "timestamp": "2024-01-01T12:50:00Z",
                        "app_name": "Terminal",
                        "window_name": "bash",
                        "text": "page2 command",
                    }
                }
            ]
        }
        page2_response.raise_for_status = MagicMock()

        mock_httpx_client.get.side_effect = [page1_response, page2_response]

        results = client.get_screen_content(
            start_time=start, end_time=end, max_frames=150
        )

        assert len(results) == 101  # 100 from page1 + 1 from page2
        assert mock_httpx_client.get.call_count == 2
