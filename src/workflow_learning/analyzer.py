"""Claude-based analysis of workflow recordings."""

import json
import logging
import re
import time
from typing import Any

from anthropic import Anthropic, RateLimitError

from .config import workflow_learning_settings
from .models import (
    AnalysisResult,
    AudioTranscript,
    RecordingSession,
    ScreenContent,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing workflow recordings to extract replayable step-by-step instructions.

Your task is to analyze screen content (OCR text) and audio transcriptions from a user demonstrating a workflow. You should:

1. Correlate screen content with voice narration to understand what the user is doing
2. Identify discrete actions performed (running commands, editing files, clicking UI elements)
3. Extract the sequential workflow steps in order
4. Infer the purpose of each step from narration and context
5. Generate executable commands where possible

Focus on:
- Terminal/command-line commands (look for shell prompts, command outputs)
- File paths mentioned or visible
- Key decisions explained in narration
- Sequence and dependencies between steps

Output should be practical and actionable - another user should be able to follow the steps.

Respond with a JSON object containing your analysis."""


ANALYSIS_USER_PROMPT = """Analyze this workflow recording and extract step-by-step instructions.

## Workflow Information
- Name: {workflow_name}
- Description: {description}
- Duration: {duration}

## Screen Content (OCR)
The following is a chronological list of screen content captured during the recording.
Each entry shows the timestamp, application, window title, and visible text.

{screen_content}

## Audio Transcription (Voice Narration)
The following is the user's voice narration during the recording.
Use this to understand the intent and purpose of each action.

{audio_transcript}

## Instructions

Analyze the above recording and respond with a JSON object in this format:

```json
{{
    "skill_name": "short-kebab-case-name",
    "skill_description": "One-sentence description of what this workflow does",
    "summary": "2-3 sentence overview of the workflow",
    "steps": [
        {{
            "step_number": 1,
            "action": "What the user did (verb phrase)",
            "target": "What they acted on",
            "purpose": "Why they did it",
            "command": "Exact command if applicable, or null",
            "evidence": ["Relevant OCR text snippets", "Relevant transcription snippets"],
            "notes": "Additional context from narration"
        }}
    ],
    "prerequisites": ["List of prerequisites or setup required"],
    "environment_variables": ["ENV_VAR_NAME=description"],
    "file_dependencies": ["path/to/required/file"]
}}
```

Important guidelines:
- Extract ALL distinct steps, not just major ones
- Include exact commands when visible in terminal output
- Preserve file paths exactly as shown
- Note any environment variables or configuration mentioned
- If a step depends on a previous step's output, note that
- Use evidence from BOTH screen content AND audio to justify each step
"""


class RecordingAnalyzer:
    """Analyzes workflow recordings using Claude."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        """Initialize the analyzer.

        Args:
            api_key: Anthropic API key (default: from config)
            model: Model to use (default: from config)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial delay between retries in seconds
        """
        self.api_key = api_key or workflow_learning_settings.anthropic_api_key
        self.model = model or workflow_learning_settings.analysis_model
        self.max_retries = max_retries or workflow_learning_settings.max_retries
        self.retry_delay = retry_delay or workflow_learning_settings.retry_delay_seconds

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)

    def analyze_session(self, session: RecordingSession) -> AnalysisResult:
        """Analyze a recording session and extract workflow steps.

        Args:
            session: RecordingSession with screen_content and audio_transcripts

        Returns:
            AnalysisResult with extracted steps
        """
        if not session.screen_content and not session.audio_transcripts:
            return AnalysisResult(
                success=False,
                error="No screen content or audio transcriptions to analyze",
                model=self.model,
            )

        # Format content for the prompt
        screen_text = self._format_screen_content(session.screen_content)
        audio_text = self._format_audio_transcripts(session.audio_transcripts)

        # Calculate duration
        duration = "Unknown"
        if session.started_at and session.stopped_at:
            delta = session.stopped_at - session.started_at
            minutes = int(delta.total_seconds() / 60)
            seconds = int(delta.total_seconds() % 60)
            duration = f"{minutes}m {seconds}s"

        user_prompt = ANALYSIS_USER_PROMPT.format(
            workflow_name=session.workflow_name,
            description=session.description or "No description provided",
            duration=duration,
            screen_content=screen_text or "No screen content captured",
            audio_transcript=audio_text or "No audio transcription captured",
        )

        return self._call_api(user_prompt)

    def _format_screen_content(
        self, screen_content: list[ScreenContent], max_chars: int = 50000
    ) -> str:
        """Format screen content for the prompt.

        Args:
            screen_content: List of screen captures
            max_chars: Maximum characters to include

        Returns:
            Formatted string
        """
        if not screen_content:
            return ""

        lines = []
        total_chars = 0

        for sc in screen_content:
            entry = f"\n### [{sc.timestamp.strftime('%H:%M:%S')}] {sc.app_name} - {sc.window_title}\n"
            entry += sc.ocr_text.strip()[:2000]  # Limit per entry

            if total_chars + len(entry) > max_chars:
                lines.append("\n... (truncated)")
                break

            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def _format_audio_transcripts(
        self, transcripts: list[AudioTranscript], max_chars: int = 20000
    ) -> str:
        """Format audio transcripts for the prompt.

        Args:
            transcripts: List of audio transcriptions
            max_chars: Maximum characters to include

        Returns:
            Formatted string
        """
        if not transcripts:
            return ""

        lines = []
        total_chars = 0

        for at in transcripts:
            if not at.text.strip():
                continue

            entry = f"[{at.timestamp.strftime('%H:%M:%S')}] {at.text.strip()}"

            if total_chars + len(entry) > max_chars:
                lines.append("... (truncated)")
                break

            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    def _call_api(self, user_prompt: str) -> AnalysisResult:
        """Call Claude API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Claude API for analysis (attempt {attempt + 1})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system=ANALYSIS_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                # Extract response text
                response_text = response.content[0].text

                # Parse JSON from response
                analysis = self._parse_json_response(response_text)

                # Extract steps
                steps = []
                for step_data in analysis.get("steps", []):
                    steps.append(
                        WorkflowStep(
                            step_number=step_data.get("step_number", 0),
                            action=step_data.get("action", ""),
                            target=step_data.get("target", ""),
                            purpose=step_data.get("purpose", ""),
                            command=step_data.get("command"),
                            evidence=step_data.get("evidence", []),
                            notes=step_data.get("notes", ""),
                        )
                    )

                return AnalysisResult(
                    success=True,
                    steps=steps,
                    summary=analysis.get("summary", ""),
                    skill_name=analysis.get("skill_name", ""),
                    skill_description=analysis.get("skill_description", ""),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=self.model,
                )

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    return AnalysisResult(
                        success=False,
                        error=f"Rate limit exceeded: {e}",
                        model=self.model,
                    )

            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                return AnalysisResult(
                    success=False,
                    error=str(e),
                    model=self.model,
                )

        return AnalysisResult(
            success=False,
            error="Max retries exceeded",
            model=self.model,
        )

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from response, handling markdown code blocks."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Return empty if no JSON found
                logger.warning("No JSON found in response")
                return {"raw_response": response_text}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"raw_response": response_text, "parse_error": str(e)}
