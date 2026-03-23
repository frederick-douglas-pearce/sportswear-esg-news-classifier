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
    ExtractedDecision,
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

When the recording shows notebook content (Jupyter cells, output, plots):
- Identify which cells are important and what their output means
- Extract metrics mentioned in narration with their expected values/thresholds
- Note figure descriptions from narration (what a good result looks like)
- Generate steps using tool_type "jupyter" with executeCode for cell execution
- Include success_criteria and on_failure for checkpoint steps
- When the user describes what a figure "should look like", capture that in expected_output

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
            "tool_type": "bash | jupyter | review | manual",
            "category": "setup | core | verification",
            "expected_output": "What the output should look like (for checkpoints)",
            "success_criteria": "How to verify this step succeeded",
            "on_failure": "What to do if criteria not met",
            "evidence": ["Relevant OCR text snippets", "Relevant transcription snippets"],
            "notes": "Additional context from narration"
        }}
    ],
    "prerequisites": ["List of prerequisites or setup required"],
    "environment_variables": ["ENV_VAR_NAME=description"],
    "file_dependencies": ["path/to/required/file"],
    "decisions": [
        {{
            "trigger": "What prompted the decision (e.g., metric drop, error observed)",
            "trigger_type": "metric_observation | error_pattern | hypothesis_test | user_preference",
            "options": [{{"id": "opt_a", "description": "First option", "expected_outcome": "What would happen"}}, {{"id": "opt_b", "description": "Second option", "expected_outcome": "What would happen"}}],
            "chosen": "opt_a",
            "reasoning": "Why this option was chosen over alternatives",
            "phase": "feature_engineering | model_selection | evaluation | hyperparameter_tuning | data_preprocessing",
            "evidence": ["Relevant narration quotes that explain this decision"],
            "outcome": "What happened after the decision, if visible in the recording"
        }}
    ]
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
- For notebook workflows: use tool_type "jupyter" for cell execution, "review" for output inspection
- Include expected_output and success_criteria for checkpoint steps where the user verifies results
- Include on_failure guidance when the user explains what to do if results aren't satisfactory
- Extract decisions from narration where the user explains WHY they made a choice
- A decision has a trigger (what prompted it), options (what was considered), chosen option, and reasoning
- Look for phrases like "I'm removing...", "I decided to...", "this isn't working so...", "let's try... instead"
- If the recording shows the outcome of a decision, include it
- Only include decisions array if decisions were found; omit or use empty array otherwise
"""


REFINEMENT_SYSTEM_PROMPT = """You are an expert at refining workflow skills by incorporating new recording data.

You have an existing skill (step-by-step workflow instructions) and a new recording session. Your task is to produce an improved, merged version of the workflow steps.

Rules for refinement:
1. Merge new information into the existing skill structure
2. Add detail to steps that the new recording elaborates on (e.g., specific metrics, expected outputs)
3. Add new steps if the recording reveals previously missed actions
4. Update success_criteria and expected_output with specifics from narration
5. Preserve steps from the existing skill that the new recording doesn't contradict
6. Do NOT discard existing steps just because the new recording doesn't cover them
7. If the new recording shows a different approach for a step, note both approaches in the notes

When the recording shows notebook content (Jupyter cells, output, plots):
- Identify which cells are important and what their output means
- Extract metrics mentioned in narration with their expected values/thresholds
- Note figure descriptions from narration (what a good result looks like)
- Generate steps using tool_type "jupyter" with executeCode for cell execution
- Include success_criteria and on_failure for checkpoint steps

{project_context}

Respond with a JSON object containing your refined analysis."""


REFINEMENT_USER_PROMPT = """Refine the existing skill with information from a new recording session.

## Existing Skill

{existing_skill}

## New Recording Session
- Name: {workflow_name}
- Description: {description}
- Duration: {duration}

## Timeline
{timeline}

## Instructions

Merge the new recording information into the existing skill and respond with a JSON object in this format:

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
            "tool_type": "bash | jupyter | review | manual",
            "category": "setup | core | verification",
            "expected_output": "What the output should look like (for checkpoints)",
            "success_criteria": "How to verify this step succeeded",
            "on_failure": "What to do if criteria not met",
            "evidence": ["Relevant OCR text snippets", "Relevant transcription snippets"],
            "notes": "Additional context from narration"
        }}
    ],
    "prerequisites": ["List of prerequisites or setup required"],
    "environment_variables": ["ENV_VAR_NAME=description"],
    "file_dependencies": ["path/to/required/file"],
    "decisions": [
        {{
            "trigger": "What prompted the decision",
            "trigger_type": "metric_observation | error_pattern | hypothesis_test | user_preference",
            "options": [{{"id": "opt_a", "description": "First option", "expected_outcome": "What would happen"}}],
            "chosen": "opt_a",
            "reasoning": "Why this option was chosen",
            "phase": "feature_engineering | model_selection | evaluation | hyperparameter_tuning | data_preprocessing",
            "evidence": ["Narration quotes"],
            "outcome": "What happened after, if visible"
        }}
    ]
}}
```

Important guidelines:
- Preserve existing steps that the new recording doesn't contradict
- Add detail from the new recording to existing steps where applicable
- Add new steps discovered in the recording at the appropriate position
- Update expected_output and success_criteria with specific values from narration
- For notebook workflows: use tool_type "jupyter" for cell execution, "review" for output inspection
- Extract decisions from narration where the user explains WHY they made a choice
- Look for phrases like "I'm removing...", "I decided to...", "this isn't working so...", "let's try... instead"
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

    def refine_skill(
        self,
        existing_skill: str,
        session: RecordingSession,
    ) -> AnalysisResult:
        """Refine an existing skill with new recording data.

        Takes an existing SKILL.md content and a new recording session,
        producing an improved analysis that merges both sources.

        Args:
            existing_skill: Content of the current SKILL.md file
            session: New recording session with screen_content and audio_transcripts

        Returns:
            AnalysisResult with refined steps
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
        system_prompt = REFINEMENT_SYSTEM_PROMPT.format(
            project_context=project_context,
        )

        # Calculate duration
        duration = "Unknown"
        if session.started_at and session.stopped_at:
            delta = session.stopped_at - session.started_at
            minutes = int(delta.total_seconds() / 60)
            seconds = int(delta.total_seconds() % 60)
            duration = f"{minutes}m {seconds}s"

        user_prompt = REFINEMENT_USER_PROMPT.format(
            existing_skill=existing_skill,
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
                    temperature=0.0,
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
                            tool_type=step_data.get("tool_type", "bash"),
                            expected_output=step_data.get("expected_output", ""),
                            success_criteria=step_data.get("success_criteria", ""),
                            on_failure=step_data.get("on_failure", ""),
                            category=step_data.get("category", ""),
                        )
                    )

                # Extract decisions
                decisions = []
                for dec_data in analysis.get("decisions", []):
                    decisions.append(
                        ExtractedDecision(
                            trigger=dec_data.get("trigger", ""),
                            trigger_type=dec_data.get("trigger_type", "metric_observation"),
                            options=dec_data.get("options", []),
                            chosen=dec_data.get("chosen", ""),
                            reasoning=dec_data.get("reasoning", ""),
                            phase=dec_data.get("phase", ""),
                            evidence=dec_data.get("evidence", []),
                            outcome=dec_data.get("outcome", ""),
                        )
                    )

                return AnalysisResult(
                    success=True,
                    steps=steps,
                    decisions=decisions,
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
