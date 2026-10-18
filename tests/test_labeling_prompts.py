"""Tests for the labeling prompt versions in prompts/labeling/ (#192)."""

import json
import re
from pathlib import Path

import pytest

from src.labeling.disposition import LABELED, dispose_reply
from src.labeling.models import LabelingResponse
from src.labeling.prompt_manager import PromptManager

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "labeling"
# Both sides of the Product-article rule, worded the same in the prompt and docs/LABELING.md
RULE_TRUE_RELEASE = (
    'A release date counts on its own, however it is phrased: "releases November 14", '
    '"available from November 14" or "goes on sale November 14" for a product not sold before'
)
RULE_FALSE_RETAIL = (
    "only retail details: product names, prices, discounts, links, stock or where to buy, "
    "or promo codes"
)
# The whole false side, pinned: any added clause (a synonym for a release date) fails the test
RULE_FALSE_SIDE = (
    "for this brand, the article gives only retail details: product names, prices, discounts, "
    "links, stock or where to buy, or promo codes. A restock, the dates a discount or sale event "
    'runs ("20% off until Sunday"), or a one-line sales blurb ("perfect for cold months") is a '
    "retail detail, not a release or a description"
)


def _bullet(lines, marker):
    """The top-level bullet holding marker, with its continuation and indented sub-bullets."""
    start = next(i for i, line in enumerate(lines) if marker in line)
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip() or line.startswith("- "):
            break
        block.append(line)
    return "\n".join(block)


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
        """No line in either prompt tells the model to leave a brand out of the reply."""
        system, user = self._prompts(manager)
        omission = re.compile(
            r"\bomit|\bleave\b.*\bout\b|\bleft out\b|do not include|only include|\bexclude",
            re.IGNORECASE,
        )
        allowed = {"Never leave a brand out of the reply."}
        hits = []
        for line in (system + "\n" + user).splitlines():
            if omission.search(line):
                rest = line
                for phrase in allowed:
                    rest = rest.replace(phrase, "")
                if omission.search(rest):
                    hits.append(line.strip())
        assert hits == []

    def test_every_brand_is_judged(self, manager):
        _, user = self._prompts(manager)
        assert "must appear in `brand_analyses` exactly once" in user
        assert "exactly as it is written" in user

    def test_json_example_parses_and_is_disposed_as_labeled(self, manager):
        """The reply shape the prompt shows is one the parser and dispose_reply accept."""
        _, user = self._prompts(manager)
        example = json.loads(re.search(r"```json\n(.*?)\n```", user, re.DOTALL).group(1))
        response = LabelingResponse.model_validate(example)
        no_esg = [a for a in response.brand_analyses if a.is_sportswear_brand and not a.categories]
        assert no_esg, "the example shows a sportswear brand with empty categories"
        assert all("is_sportswear_brand" in a.model_fields_set for a in response.brand_analyses)
        sent = [a.brand for a in response.brand_analyses]
        assert dispose_reply(sent, response.brand_analyses).status == LABELED

    def test_one_product_article_rule(self, manager):
        system, _ = self._prompts(manager)
        assert system.count("**Product-article rule** (the one rule for every product article") == 1
        assert "Product announcements, reviews, sales" not in system
        assert "Shopping/deals/product promotion articles" not in system
        # judged per brand by content, not by genre
        assert "Judge each brand by what the article gives THAT brand" in system
        assert RULE_TRUE_RELEASE in system
        assert RULE_FALSE_RETAIL in system

    def test_a_bare_release_is_a_true_example(self, manager):
        """The bare-release example is listed as a release announcement, not a collaboration."""
        system, _ = self._prompts(manager)
        rule = system[system.index("**Product-article rule** (the one rule"):]
        rule = rule[:rule.index("**Key test for financial articles:**")]
        assert "Reebok Angel Reese 1 releases November 14" in rule
        launch = next(line for line in rule.splitlines() if "A collaboration launch" in line)
        assert "New Balance 550" not in launch

    def test_brand_notes_name_the_company_only(self, manager):
        """The Mizuno, ANTA, Vans and Decathlon notes no longer list the topics that count."""
        system, _ = self._prompts(manager)
        assert re.search(r"articles about [^\n]* are valid", system) is None
        assert system.count("is the brand we track.") == 4

    def test_held_out_eval_case_is_not_a_prompt_example(self, manager):
        """#192 PR2 evaluates the CLOT/adidas release roundup, so the prompt must not quote it."""
        system, user = self._prompts(manager)
        assert "CLOT" not in system + user

    def test_labeling_doc_states_the_same_rule(self):
        doc = (PROMPTS_DIR.parents[1] / "docs" / "LABELING.md").read_text()
        rule = doc[doc.index("**Product-article rule** (prompt v1.11.0"):]
        assert "Judge each brand by what the article gives that brand" in rule
        assert RULE_TRUE_RELEASE in rule
        assert RULE_FALSE_RETAIL in rule

    def test_labeling_doc_has_no_genre_rule_for_shopping(self, manager):
        """The Stock Article section defers to the Product-article rule instead of a genre rule."""
        doc = (PROMPTS_DIR.parents[1] / "docs" / "LABELING.md").read_text()
        system, _ = self._prompts(manager)
        assert re.search(r"(?i)\*\*shopping/deals", doc + system) is None
        assert "| Shopping/deals |" not in doc
        assert "product promotion, not business news" not in doc + system
        assert '| Release announcement | "Reebok Angel Reese 1 releases November 14" | Yes | `skipped` |' in doc

    def test_release_dates_are_on_the_true_side_in_prompt_and_doc(self, manager):
        """A first sale date is a release (true); the dates a discount runs are retail (false)."""
        doc = (PROMPTS_DIR.parents[1] / "docs" / "LABELING.md").read_text()
        system, _ = self._prompts(manager)
        assert "a sale date" not in doc + system
        doc = doc[doc.index("**Product-article rule** (prompt v1.11.0"):]
        system = system[system.index("**Product-article rule** (the one rule"):]
        false_bullets = (
            f"- **`is_sportswear_brand: false`** — {RULE_FALSE_SIDE}.",
            f"- `is_sportswear_brand: false` → {RULE_FALSE_SIDE}",
        )
        for text, false_bullet in zip((system, doc), false_bullets):
            lines = text.splitlines()
            true_side = next(line for line in lines if "`is_sportswear_brand: true`" in line)
            assert '"goes on sale November 14" for a product not sold before' in true_side
            assert _bullet(lines, "`is_sportswear_brand: false`") == false_bullet


class TestV1_10_0:
    def test_v1_10_0_still_has_the_omit_instruction(self, manager):
        """Production keeps v1.8.0's omit wording until #192 PR2 promotes v1.11.0."""
        user = manager.get_formatted_user_prompt(version="v1.10.0", brands="Nike")
        assert "DO NOT include that brand" in user
