"""Claude-based analysis of workflow recordings."""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
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

# Path to project context file
PROJECT_CONTEXT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "workflow_learning" / "project_context.md"
)


ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing workflow recordings to extract replayable step-by-step instructions.

Your task is to analyze a chronological timeline of screen content (OCR text) and audio narration from a user demonstrating a workflow. You should:

1. Prioritize audio narration for understanding intent — screen content is supporting evidence
2. Correlate screen content with nearby narration to understand what the user is doing
3. Identify discrete workflow actions (running commands, editing files, reviewing output)
4. Skip navigation noise (cd, ls, directory browsing, switching windows) unless it's a meaningful part of the workflow
5. Generate executable commands that match project conventions (see Project Context below)

Focus on:
- Terminal/command-line commands (look for shell prompts, command outputs)
- File paths mentioned or visible
- Key decisions explained in narration
- Sequence and dependencies between steps

When identifying commands:
- Use project-specific tools when available (e.g., the Jupyter MCP server for notebook execution, the agent orchestrator for automated workflows)
- Prefer `uv run` for all Python execution
- Reference the Project Context below for available CLI commands and conventions

Output should be practical and actionable — another user should be able to follow the steps.

{project_context}

Respond with a JSON object containing your analysis."""


ANALYSIS_USER_PROMPT = """Analyze this workflow recording and extract step-by-step instructions.

## Workflow Information
- Name: {workflow_name}
- Description: {description}
- Duration: {duration}

## Timeline
The following is a chronological timeline interleaving screen content (OCR) and audio narration.
SCREEN entries show what was visible on screen. AUDIO entries show what the user said.
Use the timestamps to correlate what the user was doing with what they were saying.

{timeline}

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
            "category": "setup | core | verification",
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
- Commands should match project conventions (use `uv run`, project module paths, etc.)
- Skip setup/navigation steps (cd, ls, window switching) that aren't part of the core workflow
- Categorize each step: "setup" for preparation, "core" for main workflow actions, "verification" for checking results
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

        # Deduplicate screen frames
        deduped_frames = self._deduplicate_frames(session.screen_content)

        # Build interleaved timeline
        timeline = self._format_timeline(deduped_frames, session.audio_transcripts)

        # Load project context
        project_context = self._load_project_context()

        # Format system prompt with project context
        system_prompt = ANALYSIS_SYSTEM_PROMPT.format(
            project_context=project_context,
        )

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
            timeline=timeline or "No content captured",
        )

        return self._call_api(user_prompt, system_prompt)

    def _deduplicate_frames(
        self,
        frames: list[ScreenContent],
        similarity_threshold: float = 0.85,
    ) -> list[ScreenContent]:
        """Remove consecutive frames with nearly identical OCR text.

        Uses SequenceMatcher to compare adjacent frames. Keeps a frame only
        if its content differs meaningfully from the previous kept frame.

        Args:
            frames: List of screen captures
            similarity_threshold: Ratio above which frames are considered duplicates

        Returns:
            Deduplicated list of frames
        """
        if len(frames) <= 1:
            return list(frames)

        kept = [frames[0]]

        for frame in frames[1:]:
            prev_text = kept[-1].ocr_text.strip()
            curr_text = frame.ocr_text.strip()

            ratio = SequenceMatcher(None, prev_text, curr_text).ratio()
            if ratio < similarity_threshold:
                kept.append(frame)

        # Always keep the last frame if it was deduplicated away
        if frames[-1] not in kept:
            kept.append(frames[-1])

        removed = len(frames) - len(kept)
        if removed > 0:
            logger.info(
                f"Deduplicated {removed} of {len(frames)} screen frames "
                f"(threshold={similarity_threshold})"
            )

        return kept

    def _format_timeline(
        self,
        screen_content: list[ScreenContent],
        audio_transcripts: list[AudioTranscript],
        max_chars: int = 60000,
    ) -> str:
        """Build a chronologically interleaved timeline of screen and audio content.

        Args:
            screen_content: Deduplicated screen captures
            audio_transcripts: Audio transcriptions
            max_chars: Maximum characters budget for the timeline

        Returns:
            Formatted timeline string
        """
        # Build unified list of (timestamp, type, formatted_entry)
        entries: list[tuple[float, str]] = []

        for sc in screen_content:
            text = sc.ocr_text.strip()[:2000]
            entry = (
                f"### [{sc.timestamp.strftime('%H:%M:%S')}] "
                f"SCREEN | {sc.app_name} - {sc.window_title}\n{text}"
            )
            entries.append((sc.timestamp.timestamp(), entry))

        for at in audio_transcripts:
            if not at.text.strip():
                continue
            entry = (
                f"### [{at.timestamp.strftime('%H:%M:%S')}] AUDIO\n"
                f'"{at.text.strip()}"'
            )
            entries.append((at.timestamp.timestamp(), entry))

        if not entries:
            return ""

        # Sort by timestamp
        entries.sort(key=lambda x: x[0])

        # Build output within budget
        lines = []
        total_chars = 0

        for _, entry in entries:
            if total_chars + len(entry) > max_chars:
                lines.append("\n... (truncated)")
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n\n".join(lines)

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

    def _load_project_context(self, context_path: Path | None = None) -> str:
        """Load project context from file.

        Args:
            context_path: Path to context file (default: PROJECT_CONTEXT_PATH)

        Returns:
            Formatted project context string, or empty section if file not found
        """
        path = context_path or PROJECT_CONTEXT_PATH

        try:
            content = path.read_text().strip()
            return f"## Project Context\n\n{content}"
        except FileNotFoundError:
            logger.warning(f"Project context file not found: {path}")
            return "## Project Context\n\nNo project context available."

    def _call_api(self, user_prompt: str, system_prompt: str | None = None) -> AnalysisResult:
        """Call Claude API with retry logic."""
        system = system_prompt or ANALYSIS_SYSTEM_PROMPT

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Claude API for analysis (attempt {attempt + 1})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system=system,
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
