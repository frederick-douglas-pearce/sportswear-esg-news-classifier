"""Tests for skill generator."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.workflow_learning.models import (
    AnalysisResult,
    RecordingSession,
    SessionStatus,
    WorkflowStep,
)
from src.workflow_learning.skill_generator import SkillGenerator, generate_skill_from_session


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture
def sample_session():
    """Create a sample recording session."""
    now = datetime.now(timezone.utc)
    return RecordingSession(
        session_id="test-session-20240101_120000",
        workflow_name="model-training",
        description="Train the FP classifier model",
        status=SessionStatus.ANALYZED,
        started_at=now,
        stopped_at=now,
    )


@pytest.fixture
def sample_analysis():
    """Create a sample analysis result."""
    return AnalysisResult(
        success=True,
        steps=[
            WorkflowStep(
                step_number=1,
                action="Export training data",
                target="FP classifier dataset",
                purpose="Generate JSONL file with labeled examples",
                command="uv run python scripts/export_training_data.py --dataset fp",
                evidence=["Exported 500 examples"],
                notes="This creates data/training/fp_dataset.jsonl",
            ),
            WorkflowStep(
                step_number=2,
                action="Train classifier",
                target="FP model",
                purpose="Train Random Forest model on the dataset",
                command="uv run python scripts/train.py --classifier fp",
                evidence=["Model trained", "F2 score: 0.974"],
                notes="Training takes about 2 minutes",
            ),
            WorkflowStep(
                step_number=3,
                action="Evaluate model",
                purpose="Check model performance metrics",
            ),
        ],
        summary="This workflow exports training data, trains the FP classifier, and evaluates performance.",
        skill_name="train-fp-classifier",
        skill_description="Train the false positive classifier model",
        input_tokens=2000,
        output_tokens=800,
        model="claude-sonnet-4-20250514",
    )


class TestSkillGenerator:
    """Tests for SkillGenerator."""

    def test_generate_skill_creates_file(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generate_skill creates a SKILL.md file."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        assert skill_path.exists()
        assert skill_path.name == "SKILL.md"
        assert skill_path.parent.name == "train-fp-classifier"

    def test_generate_skill_content_frontmatter(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generated skill has correct frontmatter."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        content = skill_path.read_text()

        assert content.startswith("---")
        assert "name: train-fp-classifier" in content
        assert "description: Train the false positive classifier model" in content

    def test_generate_skill_content_steps(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generated skill contains all steps."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        content = skill_path.read_text()

        assert "## Steps" in content
        assert "### Step 1: Export training data" in content
        assert "### Step 2: Train classifier" in content
        assert "### Step 3: Evaluate model" in content

    def test_generate_skill_content_commands(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generated skill contains bash commands."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        content = skill_path.read_text()

        assert "```bash" in content
        assert "uv run python scripts/export_training_data.py" in content
        assert "uv run python scripts/train.py --classifier fp" in content

    def test_generate_skill_content_evidence(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generated skill contains evidence sections."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        content = skill_path.read_text()

        assert "<details>" in content
        assert "Supporting evidence" in content
        assert "Exported 500 examples" in content
        assert "F2 score: 0.974" in content

    def test_generate_skill_content_metadata(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that generated skill contains metadata."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
        )

        content = skill_path.read_text()

        assert "## Metadata" in content
        assert "**Recorded**" in content
        assert "**Workflow**: model-training" in content
        assert "Train the FP classifier model" in content

    def test_generate_skill_custom_name(self, temp_skills_dir, sample_session, sample_analysis):
        """Test generating skill with custom name."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
            skill_name="custom-skill-name",
        )

        assert skill_path.parent.name == "custom-skill-name"

        content = skill_path.read_text()
        assert "name: custom-skill-name" in content

    def test_generate_skill_converts_to_kebab_case(self, temp_skills_dir, sample_session, sample_analysis):
        """Test that skill name is converted to kebab-case."""
        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=sample_analysis,
            skill_name="My Custom Skill_Name",
        )

        assert skill_path.parent.name == "my-custom-skill-name"

    def test_generate_skill_uses_workflow_name_fallback(self, temp_skills_dir, sample_session):
        """Test that workflow name is used when no skill name provided."""
        # Analysis without skill_name
        analysis = AnalysisResult(
            success=True,
            steps=[WorkflowStep(step_number=1, action="Test step")],
            summary="Test summary",
            skill_name="",  # Empty
            skill_description="Test description",
        )

        generator = SkillGenerator(output_dir=temp_skills_dir)

        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=analysis,
        )

        assert skill_path.parent.name == "model-training"  # Falls back to workflow_name

    def test_generate_skill_step_without_command(self, temp_skills_dir, sample_session):
        """Test generating skill with step that has no command."""
        analysis = AnalysisResult(
            success=True,
            steps=[
                WorkflowStep(
                    step_number=1,
                    action="Review results",
                    purpose="Check the output manually",
                    # No command
                )
            ],
            summary="Test",
            skill_name="test-skill",
            skill_description="Test",
        )

        generator = SkillGenerator(output_dir=temp_skills_dir)
        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=analysis,
        )

        content = skill_path.read_text()

        assert "### Step 1: Review results" in content
        # Should not have bash code block if no command
        assert content.count("```bash") == 0

    def test_generate_skill_step_with_notes(self, temp_skills_dir, sample_session):
        """Test generating skill with step notes."""
        analysis = AnalysisResult(
            success=True,
            steps=[
                WorkflowStep(
                    step_number=1,
                    action="Run script",
                    notes="Important: run this in the project root",
                )
            ],
            summary="Test",
            skill_name="test-skill",
            skill_description="Test",
        )

        generator = SkillGenerator(output_dir=temp_skills_dir)
        skill_path = generator.generate_skill(
            session=sample_session,
            analysis=analysis,
        )

        content = skill_path.read_text()

        assert "**Notes**: Important: run this in the project root" in content


class TestGenerateSkillFromSession:
    """Tests for the convenience function."""

    def test_generate_skill_from_session(self, tmp_path, sample_session, sample_analysis):
        """Test the convenience function."""
        output_dir = tmp_path / "skills"
        output_dir.mkdir()

        skill_path = generate_skill_from_session(
            session=sample_session,
            analysis=sample_analysis,
            output_dir=output_dir,
        )

        assert skill_path.exists()
        assert skill_path.name == "SKILL.md"
