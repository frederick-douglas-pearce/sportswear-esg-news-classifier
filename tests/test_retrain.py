"""Tests for the retrain script and deployment trigger functionality."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.retrain import (
    SemanticVersion,
    get_current_version,
    get_next_version,
    compare_versions,
    trigger_deploy_workflow,
)


# ============================================================================
# SemanticVersion Tests
# ============================================================================

class TestSemanticVersion:
    """Tests for SemanticVersion dataclass."""

    def test_str_representation(self):
        """Test string representation of version."""
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "v1.2.3"

    def test_parse_semantic_version(self):
        """Test parsing a semantic version string."""
        v = SemanticVersion.parse("v1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_semantic_without_prefix(self):
        """Test parsing version without v prefix."""
        v = SemanticVersion.parse("2.0.1")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 1

    def test_parse_legacy_single_number(self):
        """Test parsing legacy single-number version."""
        v = SemanticVersion.parse("v3")
        assert v.major == 3
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_legacy_without_prefix(self):
        """Test parsing legacy version without prefix."""
        v = SemanticVersion.parse("5")
        assert v.major == 5
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_empty_string(self):
        """Test parsing empty string returns 0.0.0."""
        v = SemanticVersion.parse("")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_invalid_format_raises(self):
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError):
            SemanticVersion.parse("invalid")

        with pytest.raises(ValueError):
            SemanticVersion.parse("v1.2")  # Only two numbers

    def test_bump_major(self):
        """Test major version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_major()
        assert str(new_v) == "v2.0.0"

    def test_bump_minor(self):
        """Test minor version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_minor()
        assert str(new_v) == "v1.3.0"

    def test_bump_patch(self):
        """Test patch version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_patch()
        assert str(new_v) == "v1.2.4"

    def test_comparison_gt(self):
        """Test greater than comparison."""
        assert SemanticVersion(2, 0, 0) > SemanticVersion(1, 9, 9)
        assert SemanticVersion(1, 2, 0) > SemanticVersion(1, 1, 9)
        assert SemanticVersion(1, 2, 3) > SemanticVersion(1, 2, 2)

    def test_comparison_ge(self):
        """Test greater than or equal comparison."""
        assert SemanticVersion(2, 0, 0) >= SemanticVersion(1, 9, 9)
        assert SemanticVersion(1, 2, 3) >= SemanticVersion(1, 2, 3)


# ============================================================================
# Version Registry Tests
# ============================================================================

class TestVersionRegistry:
    """Tests for version registry operations."""

    @pytest.fixture
    def registry_with_versions(self, tmp_path):
        """Create a registry file with existing versions."""
        registry = {
            "fp": {
                "production": "v1.2.0",
                "versions": {
                    "v1.0.0": {"metrics": {"test_f2": 0.90}},
                    "v1.1.0": {"metrics": {"test_f2": 0.92}},
                    "v1.2.0": {"metrics": {"test_f2": 0.95}},
                },
            },
            "ep": {
                "production": "v2.0.0",
                "versions": {
                    "v1.0.0": {"metrics": {"test_f2": 0.85}},
                    "v2.0.0": {"metrics": {"test_f2": 0.93}},
                },
            },
        }
        registry_path = tmp_path / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)
        return registry_path

    @pytest.fixture
    def empty_registry(self, tmp_path):
        """Create an empty registry file."""
        registry_path = tmp_path / "registry.json"
        with open(registry_path, "w") as f:
            json.dump({}, f)
        return registry_path

    def test_get_current_version_fp(self, registry_with_versions):
        """Test getting current FP version."""
        v = get_current_version("fp", registry_with_versions)
        assert v is not None
        assert str(v) == "v1.2.0"

    def test_get_current_version_ep(self, registry_with_versions):
        """Test getting current EP version."""
        v = get_current_version("ep", registry_with_versions)
        assert v is not None
        assert str(v) == "v2.0.0"

    def test_get_current_version_nonexistent_classifier(self, registry_with_versions):
        """Test getting version for nonexistent classifier."""
        v = get_current_version("esg", registry_with_versions)
        assert v is None

    def test_get_current_version_empty_registry(self, empty_registry):
        """Test getting version from empty registry."""
        v = get_current_version("fp", empty_registry)
        assert v is None

    def test_get_current_version_missing_file(self, tmp_path):
        """Test getting version when registry doesn't exist."""
        v = get_current_version("fp", tmp_path / "nonexistent.json")
        assert v is None

    def test_get_next_version_minor(self, registry_with_versions):
        """Test getting next minor version."""
        next_v = get_next_version("fp", registry_with_versions, "minor")
        assert next_v == "v1.3.0"

    def test_get_next_version_major(self, registry_with_versions):
        """Test getting next major version."""
        next_v = get_next_version("fp", registry_with_versions, "major")
        assert next_v == "v2.0.0"

    def test_get_next_version_patch(self, registry_with_versions):
        """Test getting next patch version."""
        next_v = get_next_version("fp", registry_with_versions, "patch")
        assert next_v == "v1.2.1"

    def test_get_next_version_first_version(self, empty_registry):
        """Test getting first version for new classifier."""
        next_v = get_next_version("fp", empty_registry, "minor")
        assert next_v == "v1.0.0"

    def test_get_next_version_nonexistent_registry(self, tmp_path):
        """Test getting version when registry doesn't exist."""
        next_v = get_next_version("fp", tmp_path / "nonexistent.json", "minor")
        assert next_v == "v1.0.0"


# ============================================================================
# Version Comparison Tests
# ============================================================================

class TestVersionComparison:
    """Tests for version comparison logic."""

    def test_compare_no_production(self):
        """Test comparison when no production model exists."""
        result = compare_versions(None, {"test_f2": 0.95})
        assert result["comparison"] == "no_production"
        assert result["improvement"] is True
        assert "first production model" in result["details"]

    def test_compare_improvement(self):
        """Test comparison with improved metrics."""
        prod_metrics = {"test_f2": 0.90, "test_recall": 0.85}
        new_metrics = {"test_f2": 0.95, "test_recall": 0.92}

        result = compare_versions(prod_metrics, new_metrics)
        assert result["comparison"] == "completed"
        assert result["improvement"] is True
        assert result["production_f2"] == 0.90
        assert result["new_f2"] == 0.95
        assert result["f2_difference"] == pytest.approx(0.05)

    def test_compare_regression(self):
        """Test comparison with worse metrics."""
        prod_metrics = {"test_f2": 0.95, "test_recall": 0.92}
        new_metrics = {"test_f2": 0.90, "test_recall": 0.88}

        result = compare_versions(prod_metrics, new_metrics)
        assert result["improvement"] is False
        assert result["f2_difference"] < 0

    def test_compare_equal_metrics(self):
        """Test comparison with equal metrics."""
        prod_metrics = {"test_f2": 0.95}
        new_metrics = {"test_f2": 0.95}

        result = compare_versions(prod_metrics, new_metrics)
        assert result["improvement"] is True  # >= is improvement
        assert result["f2_difference"] == 0

    def test_compare_significant_improvement(self):
        """Test significant improvement detection."""
        prod_metrics = {"test_f2": 0.90}
        new_metrics = {"test_f2": 0.92}  # >1% improvement

        result = compare_versions(prod_metrics, new_metrics)
        assert result["significant_improvement"] is True
        assert result["f2_pct_change"] > 1.0

    def test_compare_uses_cv_metrics_fallback(self):
        """Test that comparison falls back to CV metrics."""
        prod_metrics = {"cv_f2": 0.90, "cv_recall": 0.85}  # No test_ metrics
        new_metrics = {"test_f2": 0.95}

        result = compare_versions(prod_metrics, new_metrics)
        assert result["production_f2"] == 0.90


# ============================================================================
# Deployment Trigger Tests
# ============================================================================

class TestDeploymentTrigger:
    """Tests for GitHub Actions deployment trigger."""

    def test_trigger_skips_patch_versions(self):
        """Test that patch versions skip deployment."""
        result = trigger_deploy_workflow("fp", "v1.0.1", "patch")
        assert result is True  # Returns True (skip successful)

    def test_trigger_gh_not_installed(self):
        """Test graceful handling when gh CLI not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = trigger_deploy_workflow("fp", "v1.1.0", "minor")
            assert result is False

    def test_trigger_gh_not_available(self):
        """Test handling when gh CLI returns error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="gh: command not found")

            result = trigger_deploy_workflow("fp", "v1.1.0", "minor")
            assert result is False

    def test_trigger_workflow_not_found(self):
        """Test handling when workflow doesn't exist."""
        with patch("subprocess.run") as mock_run:
            # First call: gh --version succeeds
            # Second call: workflow run fails
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="could not find any workflows"),
            ]

            result = trigger_deploy_workflow("fp", "v1.1.0", "minor")
            assert result is False

    def test_trigger_workflow_success(self):
        """Test successful workflow trigger."""
        with patch("subprocess.run") as mock_run:
            # Both calls succeed
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = trigger_deploy_workflow("fp", "v1.1.0", "minor")
            assert result is True

    def test_trigger_major_version(self):
        """Test deployment trigger for major version."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = trigger_deploy_workflow("fp", "v2.0.0", "major")
            assert result is True

            # Verify the correct command was called
            calls = mock_run.call_args_list
            assert len(calls) == 2  # gh --version, gh workflow run

            workflow_call = calls[1]
            cmd = workflow_call[0][0]  # First positional arg is the command list
            assert "gh" in cmd
            assert "workflow" in cmd
            assert "run" in cmd
            assert "deploy.yml" in cmd
            assert "-f" in cmd
            assert "classifier=fp" in cmd
            assert "version=v2.0.0" in cmd
            assert "bump_type=major" in cmd

    def test_trigger_ep_classifier(self):
        """Test deployment trigger for EP classifier."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = trigger_deploy_workflow("ep", "v1.0.0", "minor")
            assert result is True

            # Verify classifier=ep in command
            workflow_call = mock_run.call_args_list[1]
            cmd = workflow_call[0][0]
            assert "classifier=ep" in cmd

    def test_trigger_handles_generic_error(self):
        """Test handling of generic workflow trigger error."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),  # gh --version
                MagicMock(returncode=1, stderr="some other error"),  # workflow run
            ]

            result = trigger_deploy_workflow("fp", "v1.1.0", "minor")
            assert result is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestRetrainIntegration:
    """Integration tests for retrain workflow."""

    @pytest.fixture
    def full_registry(self, tmp_path):
        """Create a full registry setup for integration testing."""
        registry = {
            "fp": {
                "production": "v1.0.0",
                "versions": {
                    "v1.0.0": {
                        "created_at": "2024-01-01T00:00:00",
                        "metrics": {"test_f2": 0.90, "test_recall": 0.88},
                    }
                },
            }
        }
        registry_path = tmp_path / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)
        return registry_path

    def test_full_version_progression(self, full_registry):
        """Test version progression through multiple bumps."""
        # Get current version
        current = get_current_version("fp", full_registry)
        assert str(current) == "v1.0.0"

        # Minor bump
        next_minor = get_next_version("fp", full_registry, "minor")
        assert next_minor == "v1.1.0"

        # Major bump
        next_major = get_next_version("fp", full_registry, "major")
        assert next_major == "v2.0.0"

        # Patch bump
        next_patch = get_next_version("fp", full_registry, "patch")
        assert next_patch == "v1.0.1"

    def test_comparison_leads_to_correct_decision(self, full_registry):
        """Test that comparison results lead to correct promotion decisions."""
        prod_metrics = {"test_f2": 0.90, "test_recall": 0.88}

        # Better model should promote
        better_metrics = {"test_f2": 0.95, "test_recall": 0.92}
        result = compare_versions(prod_metrics, better_metrics)
        assert result["improvement"] is True

        # Worse model should not promote
        worse_metrics = {"test_f2": 0.85, "test_recall": 0.80}
        result = compare_versions(prod_metrics, worse_metrics)
        assert result["improvement"] is False
