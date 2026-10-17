"""Tests for the labeling prompt versions in prompts/labeling/ (#192)."""

import json
from pathlib import Path

import pytest

from src.labeling.prompt_manager import PromptManager

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "labeling"


@pytest.fixture
def manager():
    return PromptManager(prompts_dir=PROMPTS_DIR)


class TestRegistry:
    def test_production_is_still_v1_10_0(self):
        """v1.11.0 is a candidate until #192 PR2's contrastive eval."""
        registry = json.loads((PROMPTS_DIR / "registry.json").read_text())
        assert registry["production"] == "v1.10.0"

    def test_v1_11_0_is_registered_as_candidate(self, manager):
        assert "v1.11.0" in manager.list_versions()
        assert manager.get_version_info("v1.11.0")["tags"]["status"] == "candidate"

    def test_config_version_matches_directory(self):
        config = json.loads((PROMPTS_DIR / "v1.11.0" / "config.json").read_text())
        assert config["version"] == "v1.11.0"


class TestV1_11_0:
    """Every brand is judged, and the product guidance is one rule."""

    def _prompts(self, manager):
        system = manager.get_formatted_system_prompt(version="v1.11.0", brands=["Nike", "Puma"])
        user = manager.get_formatted_user_prompt(
            version="v1.11.0", title="T", brands="Nike, Puma", content="C"
        )
        return system, user

    def test_formats_through_the_prompt_manager(self, manager):
        system, user = self._prompts(manager)
        assert "Nike, Puma" in system
        assert "**Brands to Analyze**: Nike, Puma" in user
        # doubled braces in the template become one pair after formatting
        assert '"categories": {}' in user

    def test_no_omission_instruction(self, manager):
        system, user = self._prompts(manager)
        for phrase in (
            "DO NOT include that brand",
            "should be omitted entirely",
            "Only include brands that have at least one category",
            "provide ESG analysis only for confirmed sportswear brands",
        ):
            assert phrase not in user
        assert "or omit the brand" not in system

    def test_every_brand_is_judged(self, manager):
        _, user = self._prompts(manager)
        assert "must appear in `brand_analyses` exactly once" in user
        assert "exactly as it is written" in user

    def test_one_product_article_rule(self, manager):
        system, _ = self._prompts(manager)
        assert system.count("**Product-article rule** (the one rule for every product article") == 1
        assert "Product announcements, reviews, sales" not in system
        assert "**Shopping/deals/product promotion articles** → `is_sportswear_brand: false`" not in system

    def test_v1_10_0_is_unchanged_by_this_version(self, manager):
        """Production still carries v1.8.0's omit instruction until PR2 flips the pointer."""
        user = manager.get_formatted_user_prompt(version="v1.10.0", brands="Nike")
        assert "DO NOT include that brand" in user
