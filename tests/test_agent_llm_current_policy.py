"""The daily LLM analysis judges labels against the current labeling policy, on Haiku 5.5 (#252)."""

import ast
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.agent.llm import (
    ANALYSIS_EFFORT,
    DEFAULT_MODEL,
    LabelingAnalyzer,
    SampleQueryError,
    attach_brand_records,
    get_recent_labeling_samples,
    labeling_instructions,
    mark_skip_judgment,
    stored_categories,
)
from src.data_collection.config import BRANDS
from src.data_collection.models import (
    JUDGED_SKIP_REASONS,
    SKIP_NO_ESG_CATEGORIES,
    SKIP_NO_ESG_CONTENT,
    UNJUDGED_SKIP_REASONS,
)

REPO = Path(__file__).resolve().parent.parent
ANALYSIS_JSON = '{"overall_assessment": "fine", "summary": "fine"}'


def _block(kind, text=""):
    block = MagicMock(type=kind, text=text)
    if kind != "text":
        del block.text
    return block


def _response(*blocks, stop_reason="end_turn"):
    response = MagicMock()
    response.content = list(blocks)
    response.stop_reason = stop_reason
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


def _analyzer(model=None, response=None):
    with patch("src.agent.llm.Anthropic") as anthropic:
        client = MagicMock()
        anthropic.return_value = client
        analyzer = LabelingAnalyzer(api_key="test-key", model=model)
    client.messages.create.return_value = response or _response(_block("text", ANALYSIS_JSON))
    return analyzer, client


def _analyze(analyzer, skipped=(), labeled=(), fp=()):
    return analyzer.analyze_labeling_results(
        stats={"articles_processed": len(skipped), "articles_skipped": len(skipped)},
        labeled_sample=list(labeled),
        fp_sample=list(fp),
        skipped_sample=list(skipped),
    )


def _production_version():
    return json.loads((REPO / "prompts/labeling/registry.json").read_text())["production"]


class TestModelAndRequest:
    def test_the_default_model_is_haiku_5_5(self):
        from src.agent.config import AgentSettings

        assert DEFAULT_MODEL == "claude-haiku-5-5"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_LLM_MODEL", None)
            assert AgentSettings().llm_analysis_model == "claude-haiku-5-5"

    def test_haiku_5_5_sends_effort_and_no_temperature(self):
        analyzer, client = _analyzer("claude-haiku-5-5")
        assert _analyze(analyzer).success
        kwargs = client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["extra_body"] == {"output_config": {"effort": ANALYSIS_EFFORT}}
        assert kwargs["max_tokens"] == 8000

    def test_haiku_4_5_sends_temperature_and_no_effort(self):
        analyzer, client = _analyzer("claude-haiku-4-5-20251001")
        assert _analyze(analyzer).success
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["temperature"] == 0.0
        assert "extra_body" not in kwargs

    def test_a_model_with_no_spec_is_refused_before_any_call(self):
        analyzer, client = _analyzer("claude-unknown-9")
        result = _analyze(analyzer)
        assert result.success is False
        assert "LABELING_MODELS" in result.error
        client.messages.create.assert_not_called()


class TestReplyReading:
    def test_a_reply_that_starts_with_thinking_parses(self):
        analyzer, _ = _analyzer(
            response=_response(_block("thinking"), _block("text", ANALYSIS_JSON))
        )
        result = _analyze(analyzer)
        assert result.success is True
        assert result.analysis["overall_assessment"] == "fine"

    @pytest.mark.parametrize("blocks", [(_block("thinking"),), (_block("text", "  "),), ()])
    def test_a_reply_with_no_text_is_a_failure(self, blocks):
        analyzer, _ = _analyzer(response=_response(*blocks))
        result = _analyze(analyzer)
        assert result.success is False
        assert result.error == "Analysis reply contained no text"

    def test_a_refusal_is_a_failure(self):
        analyzer, _ = _analyzer(
            response=_response(_block("text", ANALYSIS_JSON), stop_reason="refusal")
        )
        result = _analyze(analyzer)
        assert result.success is False
        assert result.error == "Analysis refused by the model"


class TestLabelingInstructions:
    def test_the_analyst_is_given_the_rendered_production_prompt(self):
        analyzer, client = _analyzer()
        assert _analyze(analyzer).success
        system = client.messages.create.call_args.kwargs["system"]
        version, rendered = labeling_instructions()
        assert version == _production_version()
        assert f'<labeling_instructions version="{version}">\n{rendered}\n</labeling_instructions>' in system
        # The rules the 2026-11-03 analysis lacked, and the content types (D037)
        assert "Governance is NOT Financial News" in system
        assert "CRITICAL: Content Type" in system
        assert "{brands}" not in system
        for brand in BRANDS:
            assert brand in system

    def test_the_labeling_prompt_version_override_is_followed(self):
        # LabelingSettings is frozen, so the module's instance is replaced
        with patch("src.labeling.config.labeling_settings", MagicMock(prompt_version="v1.12.0")):
            analyzer, client = _analyzer()
            assert _analyze(analyzer).success
        assert '<labeling_instructions version="v1.12.0">' in (
            client.messages.create.call_args.kwargs["system"]
        )

    def test_a_prompt_that_does_not_load_fails_the_analysis(self):
        # LabelingSettings is frozen, so the module's instance is replaced
        with patch("src.labeling.config.labeling_settings", MagicMock(prompt_version="v9.9.9")):
            analyzer, client = _analyzer()
            result = _analyze(analyzer)
        assert result.success is False
        assert "Labeling instructions could not be loaded" in result.error
        assert "v9.9.9" in result.error
        client.messages.create.assert_not_called()


class TestReplayOf20261103:
    """The day of the false alarm: product releases skipped, judged, with no category."""

    SKIPPED = [
        {"id": "a1", "title": "Nike Air Max 95 'Big Bubble' releases November 14",
         "source_name": "Sneaker News", "brands": ["Nike"], "reason": SKIP_NO_ESG_CATEGORIES,
         "content": "The sneaker releases next week."},
        {"id": "a2", "title": "Adidas Samba OG gets a suede colorway",
         "source_name": "Hypebeast", "brands": ["Adidas"], "reason": SKIP_NO_ESG_CATEGORIES,
         "content": "A new colorway of the Samba."},
    ]
    JUDGMENTS = [
        {"article_id": "a1", "brand": "Nike", "content_type": "product",
         "reasoning": "Reports the release date of a Nike sneaker.", "gated_categories": [],
         "prompt_version": "v1.13.0"},
        {"article_id": "a2", "brand": "Adidas", "content_type": "product",
         "reasoning": "Describes a new Adidas colorway.", "gated_categories": ["Environmental"],
         "prompt_version": "v1.13.0"},
    ]

    def test_the_prompt_carries_each_brand_s_content_type(self):
        skipped = [dict(sample) for sample in self.SKIPPED]
        attach_brand_records(skipped, self.JUDGMENTS, [])
        for sample in skipped:
            mark_skip_judgment(sample, JUDGED_SKIP_REASONS, UNJUDGED_SKIP_REASONS)
        analyzer, client = _analyzer()
        assert _analyze(analyzer, skipped=skipped).success
        kwargs = client.messages.create.call_args.kwargs
        user = kwargs["messages"][0]["content"]
        assert user.count(f"- Status: skipped (judged) — reason: {SKIP_NO_ESG_CATEGORIES}") == 2
        assert "Nike: product — Reports the release date of a Nike sneaker." in user
        assert "Adidas: product — Describes a new Adidas colorway." in user
        assert "dropped by the content type: Environmental" in user
        assert "no content type recorded" not in user
        assert "Governance is NOT Financial News" in kwargs["system"]


class TestSampleAssembly:
    def test_judgments_and_labels_attach_to_their_article_and_brand(self):
        samples = [{"id": "a1", "brands": ["Nike", "Puma"]}, {"id": "a2", "brands": ["Puma"]}]
        judgments = [
            {"article_id": "a1", "brand": "Nike", "content_type": "business", "reasoning": "r1",
             "gated_categories": None, "prompt_version": "v1.13.0"},
            {"article_id": "a1", "brand": "Puma", "content_type": "other_entity",
             "reasoning": "r2", "gated_categories": [], "prompt_version": "v1.13.0"},
            {"article_id": "zz", "brand": "Nike", "content_type": "product", "reasoning": "x",
             "gated_categories": [], "prompt_version": "v1.13.0"},
        ]
        labels = [
            {"article_id": "a1", "brand": "Nike", "environmental": False, "social": True,
             "governance": True, "digital_transformation": False,
             "social_sentiment": -1, "governance_sentiment": 0},
        ]
        attach_brand_records(samples, judgments, labels)
        assert [(j["brand"], j["content_type"]) for j in samples[0]["judgments"]] == [
            ("Nike", "business"), ("Puma", "other_entity")
        ]
        assert samples[0]["categories_by_brand"] == {"Nike": [("Social", -1), ("Governance", 0)]}
        assert samples[1]["judgments"] == [] and samples[1]["categories_by_brand"] == {}

    def test_stored_categories_follow_the_prompt_order(self):
        label = {"digital_transformation": True, "environmental": True,
                 "digital_sentiment": 1, "environmental_sentiment": None}
        assert stored_categories(label) == [("Environmental", None), ("Digital Transformation", 1)]

    @pytest.mark.parametrize("reason,judged", [
        (SKIP_NO_ESG_CATEGORIES, True), (SKIP_NO_ESG_CONTENT, False), ("something else", None),
    ])
    def test_a_skip_is_marked_judged_from_its_reason(self, reason, judged):
        sample = {"reason": reason}
        mark_skip_judgment(sample, JUDGED_SKIP_REASONS, UNJUDGED_SKIP_REASONS)
        assert sample["judged"] is judged

    def test_an_article_with_no_judgment_says_so(self):
        analyzer, _ = _analyzer()
        text = analyzer._format_article_sample(
            [{"id": "a", "brands": ["Nike"], "judgments": [], "categories_by_brand": {}}], "skipped"
        )
        assert "- Brands: Nike (no content type recorded)" in text


class TestFailedSampleQuery:
    def test_a_failed_query_raises_rather_than_returning_bare_samples(self):
        with patch("sqlalchemy.create_engine", side_effect=RuntimeError("connection refused")):
            with pytest.raises(SampleQueryError, match="connection refused"):
                get_recent_labeling_samples(database_url="postgresql://nowhere/db")

    def test_no_database_configured_returns_empty_samples(self):
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            assert get_recent_labeling_samples() == ([], [], [])

    def test_the_workflow_reports_the_analysis_failed(self):
        from src.agent.workflows import daily_labeling

        analyzer = MagicMock()
        with patch.object(daily_labeling.agent_settings, "llm_analysis_enabled", True), \
                patch.object(daily_labeling.agent_settings, "anthropic_api_key", "k"), \
                patch.object(daily_labeling.agent_settings, "llm_error_threshold", 0.0), \
                patch("src.agent.llm.LabelingAnalyzer", analyzer), \
                patch("src.agent.llm.get_recent_labeling_samples",
                      side_effect=SampleQueryError("Failed to get labeling samples: boom")):
            result = daily_labeling.run_llm_analysis(MagicMock(), {"articles_processed": 3})
        assert result == {
            "llm_analysis_completed": False,
            "llm_analysis_error": "Failed to get labeling samples: boom",
        }
        analyzer.assert_not_called()


def test_llm_module_imports_nothing_from_labeling_or_data_collection_at_module_level():
    """The #211 constraint on #252: those imports stay inside the functions that use them."""
    tree = ast.parse((REPO / "src/agent/llm.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert names, "no module-level imports found; the test is not reading the module"
    assert not [n for n in names if n.startswith(("src.labeling", "src.data_collection"))]


requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@requires_postgres
class TestSampleQueries:
    @pytest.fixture
    def database(self):
        from src.data_collection.database import Database

        # Never DATABASE_URL: it may point at production
        url = os.environ.get(
            "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news_test"
        )
        db = Database(database_url=url)
        db.init_db()
        created = []
        yield db, url, created
        from src.data_collection import models

        # Articles first: their judgments (deleted with them) reference the run
        with db.get_session() as session:
            for article_id, _ in created:
                session.query(models.Article).filter(models.Article.id == article_id).delete()
            for run_id in {run_id for _, run_id in created}:
                session.query(models.LabelingRun).filter(models.LabelingRun.id == run_id).delete()

    def test_samples_carry_their_judgments_labels_and_reasons(self, database):
        from datetime import datetime, timezone

        from src.data_collection import models

        db, url, created = database
        now = datetime.now(timezone.utc)
        with db.get_session() as session:
            run = models.LabelingRun()
            skipped = models.Article(
                article_id=f"test-{uuid4()}", url=f"https://example.com/{uuid4()}",
                title=f"Nike release {uuid4()}", brands_mentioned=["Nike"], labeling_status="skipped",
                labeling_error=SKIP_NO_ESG_CATEGORIES, skipped_at=now,
            )
            labeled = models.Article(
                article_id=f"test-{uuid4()}", url=f"https://example.com/{uuid4()}",
                title=f"Puma job cuts {uuid4()}", brands_mentioned=["Puma"], labeling_status="labeled",
                labeled_at=now,
            )
            session.add_all([run, skipped, labeled])
            session.flush()
            session.add_all([
                models.BrandJudgment(article_id=skipped.id, labeling_run_id=run.id, brand="Nike",
                                     content_type="product", reasoning="A release.",
                                     prompt_version="v1.13.0", model="claude-haiku-5-5"),
                models.BrandJudgment(article_id=labeled.id, labeling_run_id=run.id, brand="Puma",
                                     content_type="business", reasoning="Job cuts.",
                                     prompt_version="v1.13.0", model="claude-haiku-5-5"),
                models.BrandLabel(article_id=labeled.id, brand="Puma", social=True,
                                  social_sentiment=-1, labeled_by="claude-haiku-5-5"),
            ])
            ids = {"skipped": str(skipped.id), "labeled": str(labeled.id)}
            created += [(skipped.id, run.id), (labeled.id, run.id)]

        labeled_sample, _, skipped_sample = get_recent_labeling_samples(
            days=1, sample_size=50, database_url=url
        )
        mine = {s["id"]: s for s in labeled_sample + skipped_sample if s["id"] in ids.values()}
        skipped_row = mine[ids["skipped"]]
        assert skipped_row["reason"] == SKIP_NO_ESG_CATEGORIES and skipped_row["judged"] is True
        assert [(j["brand"], j["content_type"]) for j in skipped_row["judgments"]] == [
            ("Nike", "product")
        ]
        assert mine[ids["labeled"]]["categories_by_brand"] == {"Puma": [("Social", -1)]}
