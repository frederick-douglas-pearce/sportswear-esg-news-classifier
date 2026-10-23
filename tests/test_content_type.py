"""Tests for prompt v1.12.0's content type (D037, #219): the derivation in BrandAnalysis, the
reply schema, the request the labeler sends, and the disposition of a typed reply."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED, dispose_reply
from src.labeling.labeler import ArticleLabeler
from src.labeling.models import CONTENT_TYPE_RULES, BrandAnalysis, ContentType, LabelingResponse
from src.labeling.prompt_manager import PromptManager
from src.data_collection.models import SKIP_NO_ESG_CATEGORIES

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts" / "labeling"
CATEGORY_NAMES = ("environmental", "social", "governance", "digital_transformation")

# The v1.12.0 text and schema to be evaluated (#219). Edited in place only while no labeling_runs
# or brand_labels row records v1.12.0 (D036 §3); the hashes evaluated are the hashes promoted.
V1_12_0_SYSTEM_HASH = "6ab0c3af777b"
V1_12_0_USER_HASH = "f9e786d07439"
V1_12_0_SCHEMA_HASH = "96064689a1f3"


def _categories(*applying):
    return {
        name: {
            "applies": name in applying,
            "sentiment": 1 if name in applying else None,
            "evidence": ["quote"] if name in applying else [],
        }
        for name in CATEGORY_NAMES
    }


def _typed(content_type, brand="Nike", applying=("environmental",), reasoning="Why."):
    return BrandAnalysis.model_validate({
        "brand": brand,
        "reasoning": reasoning,
        "content_type": content_type,
        "categories": _categories(*applying),
        "confidence": 0.9,
    })


@pytest.fixture
def manager():
    return PromptManager(prompts_dir=PROMPTS_DIR)


@pytest.fixture
def schema():
    return json.loads((PROMPTS_DIR / "v1.12.0" / "reply_schema.json").read_text())


class TestDerivation:
    @pytest.mark.parametrize("content_type", list(ContentType))
    def test_verdict_comes_from_the_one_mapping(self, content_type):
        analysis = _typed(content_type.value)
        sportswear, keeps = CONTENT_TYPE_RULES[content_type]
        assert analysis.is_sportswear_brand is sportswear
        assert analysis.get_applicable_categories() == (["environmental"] if keeps else [])
        assert analysis.gated_categories == ([] if keeps else ["environmental"])

    def test_every_content_type_has_a_rule(self):
        assert set(CONTENT_TYPE_RULES) == set(ContentType)

    def test_only_business_keeps_categories(self):
        assert [t for t, (_, keeps) in CONTENT_TYPE_RULES.items() if keeps] == [ContentType.BUSINESS]

    def test_product_and_financial_are_sportswear_without_categories(self):
        for content_type in ("product", "financial"):
            analysis = _typed(content_type, applying=("governance",))
            assert analysis.is_sportswear_brand and analysis.categories == {}
            assert analysis.gated_categories == ["governance"]
            assert analysis.not_sportswear_reason is None

    def test_not_sportswear_reason_is_the_type_and_the_reasoning(self):
        analysis = _typed("retail_only", reasoning="Only a price and a link.")
        assert analysis.not_sportswear_reason == "retail_only: Only a price and a link."

    def test_the_reply_cannot_override_the_derived_verdict(self):
        """A stated is_sportswear_brand loses to the content type."""
        analysis = BrandAnalysis.model_validate({
            "brand": "Nike", "is_sportswear_brand": True, "content_type": "retail_only",
            "categories": {}, "reasoning": "r",
        })
        assert analysis.is_sportswear_brand is False

    def test_a_reply_without_a_content_type_parses_as_before(self):
        """v1.0-v1.11 replies: no derivation, no gate."""
        analysis = BrandAnalysis.model_validate({
            "brand": "Nike", "is_sportswear_brand": True, "categories": _categories("social"),
            "reasoning": "r",
        })
        assert analysis.content_type is None
        assert analysis.get_applicable_categories() == ["social"]
        assert analysis.gated_categories == [] and analysis.not_sportswear_reason is None

    def test_the_gate_survives_revalidation(self):
        """Placing a gated analysis in a response re-runs the validator on it."""
        analysis = _typed("product")
        response = LabelingResponse(brand_analyses=[analysis], article_summary="s")
        assert response.brand_analyses[0].gated_categories == ["environmental"]

    def test_gated_categories_are_not_part_of_a_dump(self):
        assert "gated_categories" not in _typed("product").model_dump()

    def test_an_unknown_content_type_fails_validation(self):
        with pytest.raises(ValueError):
            _typed("esg")


class TestDisposition:
    def test_retail_only_for_every_brand_is_a_false_positive_with_the_type_in_the_reason(self):
        d = dispose_reply(["Nike"], [_typed("retail_only", reasoning="Only prices.")])
        assert d.status == FALSE_POSITIVE
        assert d.reason == "Nike: retail_only: Only prices."

    def test_a_product_article_with_categories_given_is_a_judged_skip(self):
        d = dispose_reply(["Nike"], [_typed("product", applying=("environmental",))])
        assert (d.status, d.reason) == (SKIPPED, SKIP_NO_ESG_CATEGORIES)

    def test_business_with_a_category_is_labeled(self):
        assert dispose_reply(["Nike"], [_typed("business")]).status == LABELED

    def test_a_content_type_is_a_verdict(self):
        analysis = BrandAnalysis(brand="Nike", content_type=ContentType.MENTION_ONLY, reasoning="r")
        d = dispose_reply(["Nike"], [analysis])
        assert d.unset == [] and d.status == FALSE_POSITIVE


class TestReplySchema:
    def _objects(self, node):
        """Every object node in the schema."""
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from self._objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from self._objects(value)

    def _analysis(self, schema):
        return schema["properties"]["brand_analyses"]["items"]

    def test_content_type_enum_is_the_python_enum(self, schema):
        enum = self._analysis(schema)["properties"]["content_type"]["enum"]
        assert enum == [t.value for t in ContentType]

    def test_every_object_is_closed_and_requires_every_property(self, schema):
        """Structured outputs need additionalProperties false; required keeps the field order."""
        for node in self._objects(schema):
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])

    def test_reasoning_comes_before_content_type(self, schema):
        """The verdict is written after the reasoning, not before it (D037)."""
        order = list(self._analysis(schema)["properties"])
        assert order.index("reasoning") < order.index("content_type") < order.index("categories")

    def test_no_is_sportswear_brand_in_the_reply(self, schema):
        assert "is_sportswear_brand" not in self._analysis(schema)["properties"]
        assert "not_sportswear_reason" not in self._analysis(schema)["properties"]

    def test_sentiment_null_is_an_any_of(self, schema):
        """type [integer, null] with null in the enum is rejected with a 400."""
        category = self._analysis(schema)["properties"]["categories"]["properties"]["social"]
        assert category["properties"]["sentiment"] == {
            "anyOf": [{"type": "integer", "enum": [-1, 0, 1]}, {"type": "null"}]
        }

    def test_no_numeric_bounds(self, schema):
        """Structured outputs reject minimum/maximum; the Pydantic model keeps confidence's."""
        text = json.dumps(schema)
        assert "minimum" not in text and "maximum" not in text

    def test_a_schema_shaped_reply_parses(self, schema):
        reply = {
            "brand_analyses": [{
                "brand": "Nike", "reasoning": "r", "content_type": "business",
                "categories": _categories("social"), "confidence": 0.8,
            }],
            "article_summary": "s",
        }
        parsed = LabelingResponse.model_validate(reply)
        assert parsed.brand_analyses[0].get_applicable_categories() == ["social"]


class TestV1_12_0:
    def test_registered_as_candidate_and_not_production(self, manager):
        assert manager.get_version_info("v1.12.0")["tags"]["status"] == "candidate"
        assert manager.get_production_version() != "v1.12.0"

    def test_config_names_the_schema(self):
        config = json.loads((PROMPTS_DIR / "v1.12.0" / "config.json").read_text())
        assert config["version"] == "v1.12.0"
        assert config["reply_schema_file"] == "reply_schema.json"

    def test_earlier_versions_load_without_a_schema(self, manager):
        """v1.10.0 and v1.11.0 are called exactly as they were evaluated."""
        for version in ("v1.10.0", "v1.11.0"):
            loaded = manager.load_version(version)
            assert loaded.reply_schema is None and loaded.reply_schema_hash == ""

    def test_text_and_schema_are_at_the_pinned_hashes(self, manager):
        version = manager.load_version("v1.12.0")
        assert (
            version.system_prompt_hash, version.user_prompt_hash, version.reply_schema_hash
        ) == (V1_12_0_SYSTEM_HASH, V1_12_0_USER_HASH, V1_12_0_SCHEMA_HASH), (
            "v1.12.0 changed: allowed only while no row records it (D036 §3); rerun the eval"
        )

    def test_prompts_format_and_name_every_content_type(self, manager):
        system = manager.get_formatted_system_prompt(version="v1.12.0", brands=["Nike"])
        user = manager.get_formatted_user_prompt(version="v1.12.0", title="T", brands="Nike", content="C")
        for content_type in ContentType:
            assert f"`{content_type.value}`" in system
            assert f"`{content_type.value}`" in user

    def test_prompts_do_not_ask_for_is_sportswear_brand(self, manager):
        system = manager.get_formatted_system_prompt(version="v1.12.0", brands=["Nike"])
        user = manager.get_formatted_user_prompt(version="v1.12.0", title="T", brands="Nike", content="C")
        assert "is_sportswear_brand" not in system + user

    def test_sentiment_is_anchored_to_conduct(self, manager):
        """#117: not the article's stance, and symmetric."""
        system = manager.get_formatted_system_prompt(version="v1.12.0", brands=["Nike"])
        assert "**the brand's ESG conduct**" in system
        assert "A hostile source reporting proactive brand action is still positive" in system
        negative = next(line for line in system.splitlines() if line.startswith("- **Negative (-1)**"))
        assert "Criticism," not in negative and "credible criticism of conduct that fell short" in negative

    def test_example_reply_matches_the_schema_shape(self, manager, schema):
        """The JSON example in the user prompt has the schema's fields in the schema's order."""
        user = manager.get_formatted_user_prompt(version="v1.12.0", title="T", brands="Nike", content="C")
        example = json.loads(user[user.index("```json") + len("```json"):user.index("```", user.index("```json") + 7)])
        properties = list(schema["properties"]["brand_analyses"]["items"]["properties"])
        for analysis in example["brand_analyses"]:
            assert list(analysis) == properties
        LabelingResponse.model_validate(example)


class TestLabelerRequest:
    def _labeler(self, model, prompt_version, **kw):
        client = MagicMock()
        with patch("src.labeling.labeler.Anthropic", return_value=client):
            return ArticleLabeler(api_key="test-key", model=model, prompt_version=prompt_version, **kw)

    def test_v1_12_0_on_haiku_5_5_sends_the_schema_beside_effort(self, schema):
        labeler = self._labeler("claude-haiku-5-5", "v1.12.0", effort="medium")
        params = labeler._request_params("u", "s")
        assert params["extra_body"] == {
            "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": schema}}
        }

    def test_v1_12_0_on_haiku_4_5_sends_the_schema_alone(self, schema):
        labeler = self._labeler("claude-haiku-4-5-20251001", "v1.12.0")
        params = labeler._request_params("u", "s")
        assert params["extra_body"] == {"output_config": {"format": {"type": "json_schema", "schema": schema}}}
        assert params["temperature"] == 0.0

    def test_v1_10_0_sends_no_format(self):
        labeler = self._labeler("claude-haiku-5-5", "v1.10.0", effort="medium")
        assert labeler._request_params("u", "s")["extra_body"] == {"output_config": {"effort": "medium"}}

    def test_a_structured_reply_is_gated_at_parse_time(self):
        labeler = self._labeler("claude-haiku-5-5", "v1.12.0", effort="medium")
        reply = json.dumps({
            "brand_analyses": [{
                "brand": "Nike", "reasoning": "A review of the Pegasus.", "content_type": "product",
                "categories": _categories("environmental"), "confidence": 0.9,
            }],
            "article_summary": "s",
        })
        analysis = labeler._parse_response(reply).brand_analyses[0]
        assert analysis.is_sportswear_brand and analysis.categories == {}
        assert analysis.gated_categories == ["environmental"]
