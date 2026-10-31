"""Tests for candidate prompt v1.13.0 and its case file (#230, D040): the owner's policy rulings
from #192's backfill review, stated in the prompt and scored by owner_rulings_192.json."""

import json
from pathlib import Path

import pytest

from scripts import compare_prompts
from src.evaluation.comparator import Expectation, matching_prompt_example, prompt_examples
from src.labeling.prompt_manager import PromptManager

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts" / "labeling"
CASE_FILE = PROMPTS_DIR / "eval_cases" / "owner_rulings_192.json"
PRODUCT_CASE_FILE = PROMPTS_DIR / "eval_cases" / "product_and_stability.json"

# One phrase per ruling the v1.13.0 system prompt states (#230); a reworded ruling fails here
RULING_PHRASES = {
    1: "Does the article report job cuts, layoffs, or closures that cost jobs at the brand?",
    2: "a data breach or cybersecurity incident at the brand is Digital Transformation with negative sentiment (-1)",
    3: "an activist shareholder, a stake that concentrates ownership in one holder, or a fight over board seats or control",
    4: "a hosted event is a brand event, not a youth program",
    5: "A single passing sentence does not carry a category, whatever the category.",
    6: "a news summary appended to a template stock page",
    7: "IS about the sportswear brand, even when it also reports the AI pivot",
    8: "Could be the Can-Am Spyder (a three-wheeled motorcycle)",
}

# The owner's ruling per non-Allbirds case: brand, sportswear, categories, sentiments. One changes
# only with this table (#230's rulings, #192's 2026-10-28 review record).
EXPECTED = {
    "20dc82b4-23db-4514-a79c-b559f7a3cb40": ("Nike", True, ("social",), (("social", -1),)),
    "7de2362b-3ba8-4dcd-8bdd-694ac3b16d68": ("Converse", True, ("social",), (("social", -1),)),
    "26daa219-5160-430d-af73-c3f831c10393": (
        "Puma", True, ("governance", "social"), (("governance", 0), ("social", -1)),
    ),
    "a70e6272-10e9-4077-b5b5-7877440e3fae": ("Nike", True, ("social",), (("social", -1),)),
    "de4528d7-280c-4924-853d-95d341be2022": (
        "Under Armour", True, ("digital_transformation", "governance"),
        (("digital_transformation", -1), ("governance", -1)),
    ),
    "7445aca0-ec06-4d6d-9933-c935944e09be": (
        "Adidas", True, ("digital_transformation",), (("digital_transformation", 1),),
    ),
    "c7c4d22d-cb03-4b73-a5ff-05d06d402917": ("Lululemon", True, ("governance",), None),
    "cc938f72-aa6e-4ef1-aa88-48d226fc8e4e": ("Lululemon", True, ("governance",), None),
    "925b8c15-2b18-4a3b-a0b2-ab9977402bab": ("Lululemon", True, ("governance",), None),
    "6fbe3e58-45b2-4152-9d7a-3e45f4191612": ("Lululemon", True, ("governance",), None),
    "af3a3548-2e10-4786-a35e-1de14fd598d8": ("Brooks Running", True, (), None),
    "159cd128-f30a-412f-b94f-cead2fd15739": ("Adidas", True, (), None),
    "4ea31398-2774-49f9-a38d-a97404ece306": ("Adidas", True, (), None),
    "367e1182-7d43-412e-ba93-a99067732e4c": ("Adidas", True, (), None),
    "7aabb199-fae3-4a94-abbd-e009761b8e93": ("Athleta", True, ("social",), (("social", 1),)),
    "baf68bd8-87d4-440e-89ee-596f00c2a475": ("Lululemon", False, None, None),
    "95da51df-784e-4da4-99fa-5870bf260f13": ("Nike", False, None, None),
    "4eff3601-7090-4373-aa0e-45660875e428": ("Deckers", False, None, None),
    "e3cdcf6b-555c-4728-a09c-d7222bef76fe": ("Columbia Sportswear", False, None, None),
    "edfc9c0e-01c2-425c-8aef-f5fa4b06340e": ("Columbia Sportswear", False, None, None),
    "e781a7b3-4b00-4cef-981a-369817b48b58": ("Spyder", False, None, None),
}


@pytest.fixture(scope="module")
def cases():
    return json.loads(CASE_FILE.read_text())["cases"]


@pytest.fixture(scope="module")
def specs():
    return compare_prompts.parse_case_file(json.loads(CASE_FILE.read_text()))


@pytest.fixture(scope="module")
def v1_13_0():
    return PromptManager(prompts_dir=PROMPTS_DIR).load_version("v1.13.0")


class TestCandidate:
    def test_registered_as_a_candidate_with_v1_12_0s_call_settings(self):
        registry = json.loads((PROMPTS_DIR / "registry.json").read_text())
        entry = registry["versions"]["v1.13.0"]
        assert registry["production"] == "v1.12.0"
        assert entry["tags"]["status"] == "candidate"
        assert entry["model_recommendations"] == registry["versions"]["v1.12.0"]["model_recommendations"]

    def test_config_names_its_version_and_v1_12_0s_files(self):
        config = json.loads((PROMPTS_DIR / "v1.13.0" / "config.json").read_text())
        base = json.loads((PROMPTS_DIR / "v1.12.0" / "config.json").read_text())
        assert config["version"] == "v1.13.0"
        assert {k: v for k, v in config.items() if k != "version"} == {
            k: v for k, v in base.items() if k != "version"
        }

    @pytest.mark.parametrize("name", ["user_prompt.txt", "reply_schema.json"])
    def test_only_the_system_prompt_changes(self, name):
        """The schema hash carries over (D037 §6), so only the system prompt is evaluated anew."""
        assert (PROMPTS_DIR / "v1.13.0" / name).read_bytes() == (PROMPTS_DIR / "v1.12.0" / name).read_bytes()

    @pytest.mark.parametrize("ruling", sorted(RULING_PHRASES))
    def test_states_each_ruling(self, v1_13_0, ruling):
        assert RULING_PHRASES[ruling] in v1_13_0.system_prompt

    def test_ruling_6_is_stated_where_it_overrides_the_earlier_rules(self, v1_13_0):
        """The ESG-first paragraph and the financial key test decide before the template block, so
        the appended-summary carve-out is in all three (architect review, C1)."""
        text = v1_13_0.system_prompt
        esg_first = text[text.index("**ESG content decides first:**"):]
        esg_first = esg_first[: esg_first.index("\n")]
        assert "appended to a template stock page" in esg_first
        key_test = text[text.index("**Key test for financial articles:**"):]
        key_test = key_test[: key_test.index("\n")]
        assert "appended under template data are not human-written analysis" in key_test
        assert "**Template pages with an appended news summary** → `mention_only`" in text

    def test_job_cuts_are_stated_where_they_override_the_earlier_rules(self, v1_13_0):
        """Reported cuts decide the content type and survive the passing-sentence rule; cost-cutting
        language without cuts does not count (#234's re-ruling of fbefe459, architect review A1)."""
        text = v1_13_0.system_prompt
        esg_first = text[text.index("**ESG content decides first:**"):]
        esg_first = esg_first[: esg_first.index("\n")]
        assert "or job cuts it announces or carries out, its content type is `business`" in esg_first
        assert "Cost-cutting language with no cuts reported" in text
        assert "A reported workforce reduction is not a passing sentence" in text

    def test_allbirds_sale_in_the_subject_wins_over_the_pivot(self, v1_13_0):
        """#234's line: the sale in the subject decides; a passing mention of it does not."""
        line = next(x for x in v1_13_0.system_prompt.splitlines() if x.startswith("- **Allbirds**"))
        assert "unless the sale of the shoe business is part of its subject" in line
        assert "Only a story about the AI business that mentions the sale in passing" in line

    def test_v1_12_0_text_is_not_v1_13_0s(self):
        """v1.13.0's rulings were added to a copy, not to production v1.12.0. That v1.12.0 is
        unchanged byte for byte is pinned by its hashes in tests/test_content_type.py
        (V1_12_0_SYSTEM_HASH and its siblings)."""
        system = (PROMPTS_DIR / "v1.12.0" / "system_prompt.txt").read_text()
        assert "Does the article report job cuts" not in system
        assert '"Elliott takes $1B stake in Lululemon"' in system


class TestCaseFile:
    def test_owner_rulings_pinned(self, specs):
        got = {
            k: (s.expected.brand, s.expected.is_sportswear_brand, s.expected.categories, s.expected.sentiments)
            for k, s in specs.items()
            if s.expected and s.expected.brand != "Allbirds"
        }
        assert got == EXPECTED

    def test_allbirds_cases_follow_the_review(self, specs):
        """16 pivot rows and 10 sale rows (ruling 7; #234 moved 6 from pivot to sale), and 4 rows
        naming Allbirds as a reference."""
        allbirds = [s for s in specs.values() if s.expected and s.expected.brand == "Allbirds"]
        pivot = [s for s in allbirds if "ruling_7" in s.strata and not s.expected.is_sportswear_brand]
        sale = [s for s in allbirds if "ruling_7" in s.strata and s.expected.is_sportswear_brand]
        reference = [s for s in allbirds if "allbirds_reference" in s.strata]
        assert (len(pivot), len(sale), len(reference), len(allbirds)) == (16, 10, 4, 30)
        assert all(s.expected.categories == () and "no_category" in s.strata for s in sale)
        assert all(not s.expected.is_sportswear_brand for s in reference)

    def test_every_ruling_has_a_case_and_ruling_9_is_unscored(self, specs):
        for n in range(1, 10):
            assert any(f"ruling_{n}" in s.strata for s in specs.values()), n
        [gray] = [s for s in specs.values() if "ruling_9" in s.strata]
        assert gray.expected is None

    def test_no_category_cases_rule_no_category(self, specs):
        for s in specs.values():
            if "no_category" in s.strata:
                assert s.expected.categories == ()

    def test_no_case_is_shared_with_the_product_case_file(self, specs):
        product = compare_prompts.parse_case_file(json.loads(PRODUCT_CASE_FILE.read_text()))
        assert not set(specs) & set(product)
        merged = compare_prompts.read_case_files([PRODUCT_CASE_FILE, CASE_FILE])
        assert len(merged) == len(specs) + len(product)

    def test_no_ruling_case_is_a_prompt_example_of_the_candidate(self, cases, v1_13_0):
        """A quoted example matching a case title would pull the case out of accuracy (C4)."""
        quotes = prompt_examples(v1_13_0.system_prompt + v1_13_0.user_prompt)
        assert quotes
        hits = {c["article_id"]: matching_prompt_example(c["title"], quotes) for c in cases}
        assert {k: v for k, v in hits.items() if v} == {}

    def test_every_case_is_an_owner_ruling_on_the_same_side(self, specs):
        """Each case is a ruled article, and its expectation is the ruling (#234): not sportswear
        is false_positive, sportswear with no category a judged skip, and categories labeled."""
        from src.data_collection.provenance import HUMAN_RULINGS

        ruled = {k: s.expected for k, s in specs.items() if s.expected}
        assert len(ruled) == len(specs) - 1  # the gray ruling 9 case has no expectation
        for article_id, expected in ruled.items():
            if not expected.is_sportswear_brand:
                want = "false_positive"
            else:
                want = "labeled" if expected.categories else "skipped"
            assert HUMAN_RULINGS.get(article_id) == want, article_id

    def test_fbefe459_scores_ruling_1s_boundary_from_the_product_case_file(self):
        """Puma's job cuts inside an earnings summary (#234's re-ruling) count toward ruling 1."""
        product = compare_prompts.parse_case_file(json.loads(PRODUCT_CASE_FILE.read_text()))
        puma = product["fbefe459-5ddf-4e05-b092-1bbc31b823cb"]
        assert "ruling_1" in puma.strata
        assert puma.expected == Expectation("Puma", True, ("social",), (("social", -1),))

    def test_sentiments_parse_as_sorted_pairs(self, specs):
        puma = specs["26daa219-5160-430d-af73-c3f831c10393"].expected
        assert puma == Expectation("Puma", True, ("governance", "social"), (("governance", 0), ("social", -1)))
