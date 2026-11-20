"""Tests for candidate prompt v1.14.0 and its case file (#248, D046): the owner's rulings from
#232's spot-check and #248's template-stock finding, stated in the prompt and scored by
owner_rulings_232.json."""

import json
from pathlib import Path

import pytest

from scripts import compare_prompts
from src.data_collection.provenance import HUMAN_RULINGS
from src.evaluation.comparator import RULING_FAMILIES, matching_prompt_example, prompt_examples
from src.labeling.prompt_manager import PromptManager

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts" / "labeling"
CASES_DIR = PROMPTS_DIR / "eval_cases"
CASE_FILE = CASES_DIR / "owner_rulings_232.json"
PRODUCT_CASE_FILE = CASES_DIR / "product_and_stability.json"
ALL_CASE_FILES = sorted(CASES_DIR.glob("*.json"))

# One phrase per change v1.14.0 states (#248); a reworded ruling fails here
RULING_PHRASES = {
    "r1": "even when the article's genre is a deal listing: what it is made of, how it is built, how it fits or feels to wear",
    "r2": "No ESG category is not the same as no substantive content.",
    "r3": "It needs a judgment or analysis about the brand",
    "r3_price": "A price move, or results reported against consensus, is not `financial` on its own",
    "r3_teaser": "A short teaser of a human-written financial piece is `financial`",
    "template": "**Filing and data-feed notices** → `mention_only`",
    "r6": "a line of employee severance costs in reported results",
    "r1_one_sentence": "or how it fits or holds up in wear (\"runs half a size small\", \"still comfortable after a full shift\")",
    "r2_scope": "a story about the brand's own sponsorship or an athlete's deal with it (the deal itself: its signing, terms, renewal or end, by either party)",
    "r3_rating": "A rating or price-target change with no reason given (\"shares fell after a downgrade\") is data, not a judgment: `mention_only`.",
    "r3_heading": "whether the company is the article's subject or one of several it judges.**",
}

# The owner's ruling per case: brand, sportswear, categories, sentiments. One changes only with
# this table (#232's 2026-11-18 record, #248's 2026-11-03 decision, #249's ruling on 57acb8be).
EXPECTED = {
    "3bb19314-ab3a-4a52-977f-09eb2c5df027": ("Hoka", True, (), None),
    "6c12f566-f74d-45f3-b83d-254190b88b7a": ("Adidas", True, (), None),
    "4068c564-5de9-4174-9f5e-45c7b9ea691c": ("Skechers", True, (), None),
    "aa37cad0-25ac-49a1-9aa4-db23b0187e28": ("Nike", False, None, None),
    "c9c97e73-1c8b-4303-8640-ece1c5893d07": ("Adidas", True, (), None),
    "83c84d1f-1039-43b6-a12e-5bb904ae99dc": ("Adidas", True, (), None),
    "ca703893-209b-49bf-89f9-738ddee5ccdf": ("Nike", True, (), None),
    "45d3a491-db08-4d41-9826-c59c17759fd8": ("Nike", True, (), None),
    "17246bda-9850-453e-b9b1-d04a0a0050cc": ("Nike", True, (), None),
    "0d42d019-3335-45ac-8811-40ffbd48b7fb": ("Under Armour", False, None, None),
    "7fe795b6-07cb-40f8-b48d-1ce01524dc42": ("Nike", True, (), None),
    "16633ce9-af72-4d69-bb16-f940fc30bd6c": ("Nike", True, (), None),
    "457e531a-9468-4dca-886a-2ce97ddcb8ad": ("Nike", False, None, None),
    "27640210-8aa2-4072-aeb5-0d4b7cdc5d33": ("Nike", False, None, None),
    "113483ca-da72-483a-8702-80cdd2f2d2e6": ("Under Armour", False, None, None),
    "f0906277-41d1-48f2-bfe6-6c79d02dc70a": ("Nike", True, (), None),
    "7bf35563-a928-48ad-8c10-b5e8dd556eba": ("Under Armour", False, None, None),
    "711b5caa-e4c1-4128-b7f1-bfaa7442eee0": ("Lululemon", False, None, None),
    "9cd2ca15-a112-4fe1-8cfa-3ee8b51b0333": ("Nike", False, None, None),
    "81ab6c27-f216-431b-bda7-621bb3bac187": ("Nike", False, None, None),
    "02067ad1-d4dd-4c94-9af6-0d23d2d2122f": ("Nike", False, None, None),
    "201cc55a-2e10-404e-896b-8c9041f02203": ("Nike", False, None, None),
    "45e8c505-57ba-436c-ae91-b7ec261b695d": ("Columbia Sportswear", False, None, None),
    "57acb8be-eff4-456c-83f4-9883b3baf79a": (
        "Adidas", True, ("digital_transformation",), (("digital_transformation", 1),),
    ),
}

# D046's corrections: #232's (R1, R2) and #248's template rows (v1.13.0 stored them, the owner
# corrected them), and #238's two R3 wraps (v1.13.0 answered them wrong in D040's run 2)
CORRECTIONS = {
    "v114_correction_r1": {"3bb19314", "6c12f566", "4068c564"},
    "v114_correction_r2": {"c9c97e73", "83c84d1f", "ca703893", "45d3a491"},
    "v114_correction_r3": {"a7eeaa13", "b8891fbc"},
    "v114_correction_template": {"7bf35563", "711b5caa", "9cd2ca15", "81ab6c27", "02067ad1", "201cc55a", "45e8c505"},
}

# D046's guards: #232 confirmations v1.14.0 must not break
GUARDS = {
    "v114_guard_r1": {"aa37cad0"},
    "v114_guard_r3": {"17246bda", "0d42d019", "7fe795b6", "16633ce9"},
    "v114_guard_r4": {"457e531a", "27640210"},
    "v114_guard_r5": {"113483ca"},
    "v114_guard_r6": {"f0906277"},
}


@pytest.fixture(scope="module")
def cases():
    return json.loads(CASE_FILE.read_text())["cases"]


@pytest.fixture(scope="module")
def specs():
    return compare_prompts.parse_case_file(json.loads(CASE_FILE.read_text()))


@pytest.fixture(scope="module")
def v1_14_0():
    return PromptManager(prompts_dir=PROMPTS_DIR).load_version("v1.14.0")


def _paragraph(text: str, start: str) -> str:
    """The line of `text` that begins with `start`, read to its end."""
    rest = text[text.index(start):]
    return rest[: rest.index("\n")]


class TestCandidate:
    def test_registered_as_a_candidate_with_v1_13_0s_call_settings(self):
        """Production stays v1.13.0 until v1.14.0's eval meets D046's bar."""
        registry = json.loads((PROMPTS_DIR / "registry.json").read_text())
        entry = registry["versions"]["v1.14.0"]
        assert registry["production"] == "v1.13.0"
        assert entry["tags"]["status"] == "candidate"
        assert entry["model_recommendations"] == registry["versions"]["v1.13.0"]["model_recommendations"]

    def test_config_names_its_version_and_v1_13_0s_files(self):
        config = json.loads((PROMPTS_DIR / "v1.14.0" / "config.json").read_text())
        base = json.loads((PROMPTS_DIR / "v1.13.0" / "config.json").read_text())
        assert config["version"] == "v1.14.0"
        assert {k: v for k, v in config.items() if k != "version"} == {
            k: v for k, v in base.items() if k != "version"
        }

    @pytest.mark.parametrize("name", ["user_prompt.txt", "reply_schema.json"])
    def test_only_the_system_prompt_changes(self, name):
        """The schema hash carries over (D037 §6), so only the system prompt is evaluated anew."""
        assert (PROMPTS_DIR / "v1.14.0" / name).read_bytes() == (PROMPTS_DIR / "v1.13.0" / name).read_bytes()

    @pytest.mark.parametrize("ruling", sorted(RULING_PHRASES))
    def test_states_each_ruling(self, v1_14_0, ruling):
        assert RULING_PHRASES[ruling] in v1_14_0.system_prompt

    def test_r3_replaces_both_rules_it_overrides(self, v1_14_0):
        """v1.13.0 made a line in a market wrap, and a place in a list, mention_only however
        financial; R3 makes a judgment about the brand financial there too (architect B1)."""
        text = v1_14_0.system_prompt
        assert "even when that line gives a reason" not in text
        assert "however financial the line is" not in text
        wrap = _paragraph(text, "   - **A market wrap or news roundup")
        assert "is `financial`, however short" in wrap
        assert "results reported against consensus" in wrap
        definition = _paragraph(text, "**`financial`:")
        assert "A brief judgment counts, and so does one about the brand as one of several companies" in definition
        key_test = _paragraph(text, "**Key test for financial articles:**")
        assert "however brief the judgment" in key_test
        assert "A price move or results against consensus, with no judgment about the brand, is not analysis," in key_test

    def test_financial_heading_admits_a_judgment_in_a_list_or_wrap(self, v1_14_0):
        """The heading no longer makes the brand the article's subject, which beat R3 on market
        wraps (code review P1), and keeps human-written analysis as the test."""
        heading = _paragraph(v1_14_0.system_prompt, "**`financial`:")
        assert "the article's subject is the company's stock" not in heading
        assert heading.startswith("**`financial`: the article gives human-written analysis or commentary")
        assert RULING_PHRASES["r3_heading"] in heading

    def test_a_rating_change_with_no_reason_stays_mention_only_in_both_rules(self, v1_14_0):
        """The owner's edge (code review P6): in the market-wrap bullet and the key test."""
        text = v1_14_0.system_prompt
        assert RULING_PHRASES["r3_rating"] in _paragraph(text, "   - **A market wrap or news roundup")
        assert "neither is a rating or price-target change with no reason given" in _paragraph(
            text, "**Key test for financial articles:**"
        )

    def test_preview_and_notice_rules_replace_v1_13_0s_unqualified_preview(self, v1_14_0):
        """#248: a preview that only lists estimates, and data-feed notices, are mention_only."""
        text = v1_14_0.system_prompt
        assert "- An earnings preview, a stock drop explained" not in text
        assert "An earnings preview that only lists consensus estimates is `mention_only`." in text
        notices = text[text.index("**Filing and data-feed notices**"):]
        notices = notices[: notices.index("**Pure stock price/volume articles**")]
        for phrase in (
            "lists its exhibits",
            "a short earnings summary that only restates reported figures",
            "an earnings preview or options-activity page that only fills in consensus estimates",
            "a page whose headline promises an analyst's view that the body does not give",
        ):
            assert phrase in notices, phrase

    def test_r1s_one_sentence_threshold_covers_fit_and_keeps_praise_retail(self, v1_14_0):
        line = _paragraph(v1_14_0.system_prompt, "  - A sale listing or deal piece that also describes")
        assert RULING_PHRASES["r1_one_sentence"] in line
        assert "a blurb that only praises it (\"a gym-bag staple\"), is not." in line

    def test_r2_is_scoped_to_the_deal_and_names_no_content_type_in_social(self, v1_14_0):
        """R2 covers the deal itself, by either party, not a partner's revenue (code review P3);
        the Social endorsement question no longer names a content type, which would pull a
        signature product's release to business (P4)."""
        text = v1_14_0.system_prompt
        mention_only = text[text.index("**`mention_only`:"):text.index("1. **Template financial articles")]
        assert RULING_PHRASES["r2_scope"] in mention_only
        assert "a story about its sponsorship or an athlete's deal" not in text
        endorsement = text[text.index("3. **Is this about a celebrity/athlete endorsement"):]
        endorsement = endorsement[: endorsement.index("4. **")]
        assert "`business`" not in endorsement

    def test_r2_is_not_overridden_by_the_esg_first_paragraph(self, v1_14_0):
        """ESG-first decides first, so its marketing sentence must not make a campaign
        mention_only (architect B2); R2 is in mention_only's and business's definitions."""
        text = v1_14_0.system_prompt
        esg_first = _paragraph(text, "**ESG content decides first:**")
        assert "a marketing activation alone do not make an article `business`" not in esg_first
        assert "do not by themselves trigger this rule" in esg_first
        mention_only = text[text.index("**`mention_only`:"):text.index("1. **Template financial articles")]
        assert "`business`, not `mention_only`, even with no ESG category" in mention_only
        business = _paragraph(text, "**`business`:")
        assert "the brand's own marketing campaigns" in business
        assert "sourced consumer-demand data about the brand" in business

    def test_r6_narrows_job_cuts_where_they_are_defined_and_excepted(self, v1_14_0):
        """A severance-cost line is cost-cutting language (Social question 1), not the reported
        workforce reduction rule 8 excepts (narrows D040 ruling 1)."""
        text = v1_14_0.system_prompt
        assert "Cost-cutting language with no cuts reported" in _paragraph(text, "   - Job cuts are cuts the brand announces")
        assert RULING_PHRASES["r6"] in _paragraph(text, "   - Job cuts are cuts the brand announces")
        rule_8 = _paragraph(text, "8. A single passing sentence")
        assert "A severance-cost line in an earnings release's figures is not a reported workforce reduction" in rule_8

    def test_v1_13_0_is_unchanged(self):
        """The edits went to a copy; v1.13.0's hashes are pinned in tests/test_owner_rulings_prompt.py."""
        system = (PROMPTS_DIR / "v1.13.0" / "system_prompt.txt").read_text()
        assert "even when that line gives a reason" in system
        assert RULING_PHRASES["r2"] not in system


class TestCaseFile:
    def test_owner_rulings_pinned(self, specs):
        got = {
            k: (s.expected.brand, s.expected.is_sportswear_brand, s.expected.categories, s.expected.sentiments)
            for k, s in specs.items()
        }
        assert got == EXPECTED

    def test_every_case_is_an_owner_ruling_on_the_same_side(self, specs):
        """Each expectation is the article's ruling (D045): not sportswear is false_positive,
        sportswear with no category a judged skip, and categories labeled."""
        assert all(s.owner_ruled is True for s in specs.values())
        for article_id, spec in specs.items():
            expected = spec.expected
            if not expected.is_sportswear_brand:
                want = "false_positive"
            else:
                want = "labeled" if expected.categories else "skipped"
            assert HUMAN_RULINGS.get(article_id) == want, article_id

    def test_the_case_files_agree_with_the_human_rulings(self):
        merged = compare_prompts.read_case_files(ALL_CASE_FILES)
        assert compare_prompts.case_ruling_disagreements(merged, HUMAN_RULINGS) == []

    def test_corrections_and_guards_are_d046s_families(self):
        """Every case in owner_rulings_232.json but 57acb8be is in exactly one D046 stratum, the
        corrections and guards are the tables above, and every D046 stratum has a case across the
        case files (#238's R3 wraps are in the product file)."""
        families = {f.name: f.strata for f in RULING_FAMILIES}
        d046 = set(families["D046 corrections"]) | set(families["D046 guards"])
        merged = compare_prompts.read_case_files(ALL_CASE_FILES)
        by_stratum: dict[str, set[str]] = {}
        for article_id, spec in merged.items():
            for stratum in spec.strata & d046:
                by_stratum.setdefault(stratum, set()).add(article_id[:8])
        assert set(by_stratum) == d046
        assert {k: v for k, v in by_stratum.items() if k in families["D046 corrections"]} == CORRECTIONS
        assert {k: v for k, v in by_stratum.items() if k in families["D046 guards"]} == GUARDS
        for article_id, spec in specs_of(CASE_FILE).items():
            want = 0 if article_id.startswith("57acb8be") else 1
            assert len(spec.strata & d046) == want, article_id

    def test_the_digital_case_is_outside_every_family(self, specs):
        """No v1.14.0 text addresses #249's Digital ruling, so it is read, not barred (D046 §2,
        #280): its stratum is in no family, and its categories and sentiment are still ruled."""
        [(article_id, spec)] = [(k, s) for k, s in specs.items() if k.startswith("57acb8be")]
        in_a_family = {s for f in RULING_FAMILIES for s in f.strata}
        assert "r249_digital" in spec.strata and not spec.strata & in_a_family
        assert spec.expected.categories == ("digital_transformation",)

    def test_r1_cases_repeat_for_the_sale_listing_noise(self, cases):
        """D040 §6 measured a few percent per-reply noise on the sale-listing boundary."""
        r1 = [c for c in cases if {"v114_correction_r1", "v114_guard_r1"} & set(c["strata"])]
        assert len(r1) == 4 and all(c["repeats"] == 3 for c in r1)

    def test_r3_wraps_keep_one_expectation_across_files(self):
        product = specs_of(PRODUCT_CASE_FILE)
        for prefix, brand in (("a7eeaa13", "Nike"), ("b8891fbc", "Adidas")):
            [spec] = [s for k, s in product.items() if k.startswith(prefix)]
            assert "v114_correction_r3" in spec.strata and "financial" in spec.strata
            assert (spec.expected.brand, spec.expected.is_sportswear_brand) == (brand, True)

    def test_no_case_is_a_prompt_example_of_the_candidate(self, cases, v1_14_0):
        """A quoted example matching a case title would pull the case out of accuracy (D040 §2)."""
        quotes = prompt_examples(v1_14_0.system_prompt + v1_14_0.user_prompt)
        assert quotes
        hits = {c["article_id"]: matching_prompt_example(c["title"], quotes) for c in cases}
        assert {k: v for k, v in hits.items() if v} == {}

    def test_the_unruled_wrap_is_left_out(self, specs):
        """7f3d37d8 is named in #248 but not ruled; it joins only once it is in HUMAN_RULINGS."""
        assert not any(k.startswith("7f3d37d8") for k in specs)
        assert not any(k.startswith("7f3d37d8") for k in HUMAN_RULINGS)


def specs_of(path: Path):
    return compare_prompts.parse_case_file(json.loads(path.read_text()))
