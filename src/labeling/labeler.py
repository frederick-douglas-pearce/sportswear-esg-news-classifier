"""Claude-based article labeling for ESG classification."""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from .config import (
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_PROMPT_TEMPLATE,
    TARGET_SPORTSWEAR_BRANDS,
    labeling_settings,
)
from .models import BrandAnalysis, CategoryLabel, LabelingResponse
from .prompt_manager import PromptManager, PromptVersion

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """Result of labeling an article."""

    success: bool
    response: LabelingResponse | None = None
    error: str | None = None
    error_type: str | None = None  # Categorized error type for diagnostics
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    prompt_version: str = ""
    prompt_system_hash: str = ""
    prompt_user_hash: str = ""


def classify_api_error(error: Exception) -> tuple[str, str]:
    """Classify an API error into a category and detailed message.

    Returns:
        Tuple of (error_type, detailed_message)

    Error types:
        - authentication: Invalid or missing API key
        - rate_limit: Too many requests
        - timeout: Request timed out
        - connection: Network/connection issues (internet outage, DNS, etc.)
        - server_error: Anthropic server issues (500, 502, 503, etc.)
        - api_error: Other API errors
        - unknown: Unrecognized errors
    """
    error_str = str(error)

    if isinstance(error, AuthenticationError):
        return "authentication", f"API authentication failed: {error_str}"

    if isinstance(error, RateLimitError):
        return "rate_limit", f"API rate limit exceeded: {error_str}"

    if isinstance(error, APITimeoutError):
        return "timeout", f"API request timed out: {error_str}"

    if isinstance(error, APIConnectionError):
        # Check for specific connection issues
        error_lower = error_str.lower()
        if "connection refused" in error_lower:
            return "connection", f"Connection refused (service unavailable): {error_str}"
        if "name or service not known" in error_lower or "dns" in error_lower:
            return "connection", f"DNS resolution failed (possible internet outage): {error_str}"
        if "network is unreachable" in error_lower:
            return "connection", f"Network unreachable (internet outage): {error_str}"
        if "connection reset" in error_lower:
            return "connection", f"Connection reset by server: {error_str}"
        if "ssl" in error_lower or "certificate" in error_lower:
            return "connection", f"SSL/TLS error: {error_str}"
        return "connection", f"API connection error: {error_str}"

    if isinstance(error, InternalServerError):
        return "server_error", f"Anthropic server error (500): {error_str}"

    if isinstance(error, APIStatusError):
        status_code = getattr(error, "status_code", None)
        if status_code:
            if status_code == 502:
                return "server_error", f"Anthropic bad gateway (502): {error_str}"
            if status_code == 503:
                return "server_error", f"Anthropic service unavailable (503): {error_str}"
            if status_code == 504:
                return "server_error", f"Anthropic gateway timeout (504): {error_str}"
            if status_code >= 500:
                return "server_error", f"Anthropic server error ({status_code}): {error_str}"
            if status_code == 400:
                return "api_error", f"Bad request (400): {error_str}"
            if status_code == 404:
                return "api_error", f"Not found (404) - check model name: {error_str}"
        return "api_error", f"API status error: {error_str}"

    # Check for network-related errors that might not be wrapped
    error_lower = error_str.lower()
    if any(
        keyword in error_lower
        for keyword in ["connection", "network", "socket", "timeout", "dns", "unreachable"]
    ):
        return "connection", f"Network error: {error_str}"

    return "unknown", f"Unexpected error: {type(error).__name__}: {error_str}"


class ArticleLabeler:
    """Labels articles with ESG categories using Claude.

    Features:
    - Structured JSON output parsing
    - Automatic retry with exponential backoff for rate limits
    - Token usage tracking for cost estimation
    - Fallback parsing for malformed responses
    - Versioned prompt support
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 2000,
        prompt_version: str | None = None,
    ):
        """Initialize the labeler.

        Args:
            api_key: Anthropic API key (default: from settings)
            model: Model to use for labeling (default: from settings)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial delay between retries in seconds
            max_tokens: Maximum output tokens
            prompt_version: Version of prompts to use (default: production version)
        """
        self.api_key = api_key or labeling_settings.anthropic_api_key
        self.model = model or labeling_settings.labeling_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)

        # Load versioned prompts
        self.prompt_manager = PromptManager()
        self._prompt_version = prompt_version or labeling_settings.prompt_version

        try:
            self.loaded_prompt = self.prompt_manager.load_version(self._prompt_version)
            logger.info(
                f"Loaded prompt version {self.loaded_prompt.version} "
                f"(system: {self.loaded_prompt.system_prompt_hash}, "
                f"user: {self.loaded_prompt.user_prompt_hash})"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to load versioned prompts: {e}. Using hardcoded fallback.")
            self.loaded_prompt = None

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0

    @property
    def prompt_version(self) -> str:
        """Get the current prompt version string."""
        if self.loaded_prompt:
            return self.loaded_prompt.version
        return "legacy"

    @property
    def prompt_system_hash(self) -> str:
        """Get the system prompt hash."""
        if self.loaded_prompt:
            return self.loaded_prompt.system_prompt_hash
        return ""

    @property
    def prompt_user_hash(self) -> str:
        """Get the user prompt hash."""
        if self.loaded_prompt:
            return self.loaded_prompt.user_prompt_hash
        return ""

    def label_article(
        self,
        title: str,
        content: str,
        brands: list[str],
        published_at: datetime | None = None,
        source_name: str | None = None,
    ) -> LabelingResult:
        """Label an article with ESG categories for each brand.

        Args:
            title: Article title
            content: Article content (full_content or description)
            brands: List of brand names to analyze
            published_at: Publication date
            source_name: Name of the publication source

        Returns:
            LabelingResult with parsed response or error
        """
        if not content or not content.strip():
            return LabelingResult(
                success=False,
                error="No content provided",
                model=self.model,
                prompt_version=self.prompt_version,
                prompt_system_hash=self.prompt_system_hash,
                prompt_user_hash=self.prompt_user_hash,
            )

        if not brands:
            return LabelingResult(
                success=False,
                error="No brands to analyze",
                model=self.model,
                prompt_version=self.prompt_version,
                prompt_system_hash=self.prompt_system_hash,
                prompt_user_hash=self.prompt_user_hash,
            )

        # Truncate content if too long
        max_content_tokens = labeling_settings.max_article_tokens
        content = self._truncate_content(content, max_content_tokens)

        # Build prompts using versioned templates or fallback to hardcoded
        if self.loaded_prompt:
            system_prompt = self.prompt_manager.get_formatted_system_prompt(
                version=self.loaded_prompt.version,
                brands=list(TARGET_SPORTSWEAR_BRANDS),
            )
            user_prompt = self.prompt_manager.get_formatted_user_prompt(
                version=self.loaded_prompt.version,
                title=title,
                published_at=published_at.strftime("%Y-%m-%d") if published_at else "Unknown",
                source_name=source_name or "Unknown",
                brands=", ".join(brands),
                content=content,
            )
        else:
            # Fallback to hardcoded prompts
            system_prompt = LABELING_SYSTEM_PROMPT
            user_prompt = LABELING_USER_PROMPT_TEMPLATE.format(
                title=title,
                published_at=published_at.strftime("%Y-%m-%d") if published_at else "Unknown",
                source_name=source_name or "Unknown",
                brands=", ".join(brands),
                content=content,
            )

        # Call Claude API with retry
        try:
            response_text, input_tokens, output_tokens = self._call_api_with_retry(
                user_prompt, system_prompt
            )
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_api_calls += 1

            # Parse the response
            parsed = self._parse_response(response_text)

            if parsed:
                return LabelingResult(
                    success=True,
                    response=parsed,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    prompt_version=self.prompt_version,
                    prompt_system_hash=self.prompt_system_hash,
                    prompt_user_hash=self.prompt_user_hash,
                )
            else:
                return LabelingResult(
                    success=False,
                    error="Failed to parse LLM response",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model,
                    prompt_version=self.prompt_version,
                    prompt_system_hash=self.prompt_system_hash,
                    prompt_user_hash=self.prompt_user_hash,
                )

        except Exception as e:
            error_type, error_msg = classify_api_error(e)
            logger.error(f"[{error_type}] Labeling failed: {error_msg}")
            return LabelingResult(
                success=False,
                error=error_msg,
                error_type=error_type,
                model=self.model,
                prompt_version=self.prompt_version,
                prompt_system_hash=self.prompt_system_hash,
                prompt_user_hash=self.prompt_user_hash,
            )

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to approximately max_tokens.

        Uses a rough estimate of 4 characters per token.
        """
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content

        # Truncate at sentence boundary if possible
        truncated = content[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:
            truncated = truncated[: last_period + 1]

        logger.debug(f"Truncated content from {len(content)} to {len(truncated)} chars")
        return truncated + "\n\n[Content truncated...]"

    def _call_api_with_retry(
        self, user_prompt: str, system_prompt: str
    ) -> tuple[str, int, int]:
        """Call Claude API with automatic retry for rate limits and transient errors.

        Args:
            user_prompt: The user message content
            system_prompt: The system prompt to use

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            Various Anthropic exceptions with detailed error classification
        """
        delay = self.retry_delay
        last_error: Exception | None = None
        last_error_type: str | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                response_text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                return response_text, input_tokens, output_tokens

            except RateLimitError as e:
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{error_type}] Rate limit hit, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"[{error_type}] Rate limit exceeded after {self.max_retries} attempts: {error_msg}"
                    )
                    raise

            except APITimeoutError as e:
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{error_type}] Request timed out, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"[{error_type}] Timeout after {self.max_retries} attempts: {error_msg}"
                    )
                    raise

            except APIConnectionError as e:
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{error_type}] Connection error, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"[{error_type}] Connection failed after {self.max_retries} attempts: {error_msg}"
                    )
                    raise

            except InternalServerError as e:
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{error_type}] Server error, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"[{error_type}] Server error after {self.max_retries} attempts: {error_msg}"
                    )
                    raise

            except AuthenticationError as e:
                # Don't retry authentication errors
                error_type, error_msg = classify_api_error(e)
                logger.error(f"[{error_type}] {error_msg}")
                raise

            except APIStatusError as e:
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                status_code = getattr(e, "status_code", None)

                # Retry on 5xx errors, don't retry on 4xx (client errors)
                if status_code and status_code >= 500 and attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{error_type}] Server error ({status_code}), retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"[{error_type}] {error_msg}")
                    raise

            except Exception as e:
                # Catch any other unexpected errors
                error_type, error_msg = classify_api_error(e)
                last_error = e
                last_error_type = error_type
                logger.error(
                    f"[{error_type}] Unexpected error in API call "
                    f"(attempt {attempt + 1}/{self.max_retries}): {error_msg}"
                )
                # Don't retry unknown errors
                raise

        raise RuntimeError("API call failed after all retries")

    def _parse_response(self, response_text: str) -> LabelingResponse | None:
        """Parse the LLM response into a structured LabelingResponse.

        Handles various response formats and attempts recovery from malformed JSON.
        """
        # Extract JSON from response (may be wrapped in markdown code blocks)
        json_str = self._extract_json(response_text)
        if not json_str:
            logger.warning("No JSON found in response")
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            # Try to fix common issues
            json_str = self._fix_json(json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON even after fixing")
                return None

        # Validate and convert to Pydantic model
        try:
            return LabelingResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"Pydantic validation error: {e}")
            # Try to fix the data structure
            fixed_data = self._fix_response_structure(data)
            if fixed_data:
                try:
                    return LabelingResponse.model_validate(fixed_data)
                except Exception as e2:
                    logger.error(f"Failed to validate fixed data: {e2}")
            return None

    def _extract_json(self, text: str) -> str | None:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code blocks first
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            for match in matches:
                if "{" in match:
                    return match.strip()

        # Try to find raw JSON object
        json_pattern = r"\{[\s\S]*\}"
        match = re.search(json_pattern, text)
        if match:
            return match.group()

        return None

    def _fix_json(self, json_str: str) -> str:
        """Attempt to fix common JSON issues."""
        # Remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # Fix unquoted keys (simple cases)
        json_str = re.sub(r"(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)

        return json_str

    def _fix_response_structure(self, data: dict) -> dict | None:
        """Attempt to fix the response data structure."""
        if "brand_analyses" not in data:
            # Check if the data itself is a brand analysis
            if "brand" in data and "categories" in data:
                data = {
                    "brand_analyses": [data],
                    "article_summary": data.get("reasoning", "No summary provided"),
                }
            else:
                return None

        # Ensure all brand analyses have complete category structures
        for analysis in data.get("brand_analyses", []):
            if "categories" in analysis:
                for cat_name in [
                    "environmental",
                    "social",
                    "governance",
                    "digital_transformation",
                ]:
                    if cat_name not in analysis["categories"]:
                        analysis["categories"][cat_name] = {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        }

        if "article_summary" not in data:
            data["article_summary"] = "No summary provided"

        return data

    def get_stats(self) -> dict[str, int | float | str]:
        """Get labeling statistics.

        Returns:
            Dictionary with usage statistics
        """
        # Cost estimates for Claude Sonnet
        # Input: $3.00 per 1M tokens, Output: $15.00 per 1M tokens
        input_cost = (self.total_input_tokens / 1_000_000) * 3.00
        output_cost = (self.total_output_tokens / 1_000_000) * 15.00

        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_api_calls": self.total_api_calls,
            "estimated_cost_usd": input_cost + output_cost,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_system_hash": self.prompt_system_hash,
            "prompt_user_hash": self.prompt_user_hash,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
