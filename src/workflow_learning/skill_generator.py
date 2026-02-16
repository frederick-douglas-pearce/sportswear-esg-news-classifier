"""Generate Agent Skills (SKILL.md) from analyzed workflow recordings."""

import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import workflow_learning_settings
from .models import AnalysisResult, RecordingSession, WorkflowStep

logger = logging.getLogger(__name__)


class SkillGenerator:
    """Generates SKILL.md files from analyzed workflow recordings."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize the skill generator.

        Args:
            output_dir: Directory for generated skills. Defaults to config setting.
        """
        self.output_dir = output_dir or workflow_learning_settings.skills_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_skill(
        self,
        session: RecordingSession,
        analysis: AnalysisResult,
        skill_name: str | None = None,
    ) -> Path:
        """Generate a SKILL.md file from a recording session and analysis.

        Args:
            session: The recording session
            analysis: The analysis result with extracted steps
            skill_name: Optional override for skill name (default: from analysis)

        Returns:
            Path to the generated SKILL.md file
        """
        # Determine skill name
        name = skill_name or analysis.skill_name or session.workflow_name
        # Ensure kebab-case
        name = name.lower().replace(" ", "-").replace("_", "-")

        # Create skill directory
        skill_dir = self.output_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        # Generate content
        content = self._generate_content(session, analysis, name)

        # Write file
        skill_file.write_text(content)
        logger.info(f"Generated skill: {skill_file}")

        return skill_file

    def _generate_content(
        self,
        session: RecordingSession,
        analysis: AnalysisResult,
        skill_name: str,
    ) -> str:
        """Generate the SKILL.md content.

        Args:
            session: The recording session
            analysis: The analysis result
            skill_name: The skill name

        Returns:
            Markdown content for SKILL.md
        """
        lines = []

        # Frontmatter
        description = analysis.skill_description or session.description or f"Execute the {session.workflow_name} workflow"
        lines.extend([
            "---",
            f"name: {skill_name}",
            f"description: {description}",
            "---",
            "",
        ])

        # Title and overview
        title = skill_name.replace("-", " ").title()
        lines.extend([
            f"# {title} Skill",
            "",
            f"This skill was learned from a workflow recording on {session.started_at.strftime('%Y-%m-%d')}.",
            "",
        ])

        # Summary
        if analysis.summary:
            lines.extend([
                "## Overview",
                "",
                analysis.summary,
                "",
            ])

        # Steps
        if analysis.steps:
            lines.extend([
                "## Steps",
                "",
            ])

            for step in analysis.steps:
                lines.extend(self._format_step(step))

        # Additional metadata
        lines.extend([
            "## Metadata",
            "",
            f"- **Recorded**: {session.started_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"- **Workflow**: {session.workflow_name}",
        ])

        if session.description:
            lines.append(f"- **Description**: {session.description}")

        if session.stopped_at and session.started_at:
            duration = session.stopped_at - session.started_at
            minutes = int(duration.total_seconds() / 60)
            lines.append(f"- **Recording Duration**: {minutes} minutes")

        lines.append("")

        return "\n".join(lines)

    def _format_step(self, step: WorkflowStep) -> list[str]:
        """Format a workflow step for the skill file.

        Args:
            step: The workflow step

        Returns:
            List of markdown lines
        """
        lines = [
            f"### Step {step.step_number}: {step.action}",
            "",
        ]

        # Purpose
        if step.purpose:
            lines.extend([
                step.purpose,
                "",
            ])

        # Target
        if step.target:
            lines.append(f"**Target**: {step.target}")
            lines.append("")

        # Command block
        if step.command:
            lines.extend([
                "```bash",
                step.command,
                "```",
                "",
            ])

        # Notes from narration
        if step.notes:
            lines.extend([
                f"**Notes**: {step.notes}",
                "",
            ])

        # Evidence (collapsed for readability)
        if step.evidence:
            lines.extend([
                "<details>",
                "<summary>Supporting evidence from recording</summary>",
                "",
            ])
            for evidence in step.evidence:
                lines.append(f"- {evidence}")
            lines.extend([
                "",
                "</details>",
                "",
            ])

        return lines


def generate_skill_from_session(
    session: RecordingSession,
    analysis: AnalysisResult,
    skill_name: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Convenience function to generate a skill from a session.

    Args:
        session: The recording session
        analysis: The analysis result
        skill_name: Optional skill name override
        output_dir: Optional output directory override

    Returns:
        Path to the generated SKILL.md file
    """
    generator = SkillGenerator(output_dir=output_dir)
    return generator.generate_skill(session, analysis, skill_name)
