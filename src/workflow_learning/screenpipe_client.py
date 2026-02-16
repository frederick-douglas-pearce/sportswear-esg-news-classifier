"""Screenpipe REST API client for retrieving screen and audio recordings."""

import logging
from datetime import datetime
from typing import Any

import httpx

from .config import workflow_learning_settings
from .models import AudioTranscript, ScreenContent

logger = logging.getLogger(__name__)


class ScreenpipeError(Exception):
    """Raised when Screenpipe API returns an error."""

    pass


class ScreenpipeClient:
    """Client for interacting with Screenpipe REST API.

    Screenpipe is a passive recording tool that captures screen content (OCR)
    and audio transcriptions. This client queries the local Screenpipe API
    to retrieve recordings for a given time range.

    API Documentation: https://docs.screenpi.pe/docs/api-reference
    """

    def __init__(
        self,
        api_url: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the Screenpipe client.

        Args:
            api_url: Screenpipe API URL. Defaults to config setting.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url or workflow_learning_settings.screenpipe_api_url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> bool:
        """Check if Screenpipe is running and accessible.

        Returns:
            True if Screenpipe is healthy, False otherwise
        """
        try:
            response = self._client.get(f"{self.api_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Screenpipe health check failed: {e}")
            return False

    def search_content(
        self,
        start_time: datetime,
        end_time: datetime,
        content_type: str = "all",
        query: str | None = None,
        app_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search Screenpipe content within a time range.

        Args:
            start_time: Start of time range (UTC)
            end_time: End of time range (UTC)
            content_type: Type of content - "ocr", "audio", or "all"
            query: Optional text query to filter results
            app_name: Optional app name to filter results
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            Raw API response dict with 'data' containing results

        Raises:
            ScreenpipeError: If API request fails
        """
        params: dict[str, Any] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "content_type": content_type,
            "offset": offset,
        }

        if query:
            params["q"] = query
        if app_name:
            params["app_name"] = app_name
        if limit:
            params["limit"] = limit

        try:
            response = self._client.get(f"{self.api_url}/search", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ScreenpipeError(f"Screenpipe API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise ScreenpipeError(f"Screenpipe connection error: {e}") from e

    def get_screen_content(
        self,
        start_time: datetime,
        end_time: datetime,
        app_name: str | None = None,
        max_frames: int | None = None,
    ) -> list[ScreenContent]:
        """Get screen OCR content for a time range.

        Args:
            start_time: Start of time range (UTC)
            end_time: End of time range (UTC)
            app_name: Optional app name filter
            max_frames: Maximum frames to return

        Returns:
            List of ScreenContent objects
        """
        max_frames = max_frames or workflow_learning_settings.max_screen_frames
        results: list[ScreenContent] = []
        offset = 0
        page_size = 100  # API page size

        while len(results) < max_frames:
            try:
                response = self.search_content(
                    start_time=start_time,
                    end_time=end_time,
                    content_type="ocr",
                    app_name=app_name,
                    limit=min(page_size, max_frames - len(results)),
                    offset=offset,
                )

                data = response.get("data", [])
                if not data:
                    break

                for item in data:
                    content = item.get("content", {})
                    if content.get("type") == "OCR":
                        results.append(
                            ScreenContent(
                                timestamp=self._parse_timestamp(content.get("timestamp", "")),
                                app_name=content.get("app_name", ""),
                                window_title=content.get("window_name", ""),
                                ocr_text=content.get("text", ""),
                                file_path=content.get("file_path"),
                            )
                        )

                if len(data) < page_size:
                    break
                offset += page_size

            except ScreenpipeError as e:
                logger.error(f"Error fetching screen content: {e}")
                break

        logger.info(f"Retrieved {len(results)} screen frames")
        return results

    def get_audio_transcriptions(
        self,
        start_time: datetime,
        end_time: datetime,
        max_chunks: int | None = None,
    ) -> list[AudioTranscript]:
        """Get audio transcriptions for a time range.

        Args:
            start_time: Start of time range (UTC)
            end_time: End of time range (UTC)
            max_chunks: Maximum chunks to return

        Returns:
            List of AudioTranscript objects
        """
        max_chunks = max_chunks or workflow_learning_settings.max_audio_chunks
        results: list[AudioTranscript] = []
        offset = 0
        page_size = 100

        while len(results) < max_chunks:
            try:
                response = self.search_content(
                    start_time=start_time,
                    end_time=end_time,
                    content_type="audio",
                    limit=min(page_size, max_chunks - len(results)),
                    offset=offset,
                )

                data = response.get("data", [])
                if not data:
                    break

                for item in data:
                    content = item.get("content", {})
                    if content.get("type") == "Audio":
                        results.append(
                            AudioTranscript(
                                timestamp=self._parse_timestamp(
                                    content.get("timestamp", "")
                                ),
                                text=content.get("transcription", ""),
                                duration_seconds=content.get("duration_secs", 0.0),
                                device_name=content.get("device_name", ""),
                                is_input=content.get("is_input_device", True),
                            )
                        )

                if len(data) < page_size:
                    break
                offset += page_size

            except ScreenpipeError as e:
                logger.error(f"Error fetching audio transcriptions: {e}")
                break

        logger.info(f"Retrieved {len(results)} audio chunks")
        return results

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from Screenpipe API.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Parsed datetime (UTC)
        """
        if not timestamp_str:
            return datetime.now()

        try:
            # Handle various ISO formats
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            logger.warning(f"Failed to parse timestamp: {timestamp_str}")
            return datetime.now()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "ScreenpipeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
