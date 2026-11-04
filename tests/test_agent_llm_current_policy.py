"""The daily LLM analysis judges labels against the current labeling policy, on Haiku 5.5 (#252)."""

import ast
import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.llm import (
    DEFAULT_MODEL,
    LabelingAnalyzer,
    SampleQueryError,
    attach_brand_records,
    build_samples,
    get_recent_labeling_samples,
    stored_categories,
)
from src.data_collection import provenance
from src.data_collection.config import BRANDS
from src.data_collection.models import (
    FP_ACTION_CONTINUED_TO_LLM,
    FP_ACTION_SKIPPED_LLM,
    FP_REASON_AUTO_DETECTED_PREFIX,
    FP_REASON_MANUAL,
    MANUAL_SKIP_REASON,
    SKIP_INSUFFICIENT_CONTENT,
    SKIP_NO_ESG_CATEGORIES,
    SKIP_NO_ESG_CONTENT,
)
from src.labeling.config import LabelingModelSpec
from src.labeling.prompt_manager import PromptManager

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


def _settings(prompt_version=None, labeling_effort="medium"):
    """A stand-in for the frozen labeling_settings: the module's instance is replaced."""
    return patch(
        "src.labeling.config.labeling_settings",
        MagicMock(prompt_version=prompt_version, labeling_effort=labeling_effort),
    )


def _production_version():
    return json.loads((REPO / "prompts/labeling/registry.json").read_text())["production"]


def _rendered(version):
    return PromptManager().get_formatted_system_prompt(version=version, brands=list(BRANDS))


def _spec(**fields):
    base = dict(input_cost_per_mtok=1.0, output_cost_per_mtok=1.0, min_chars_per_token=2.0)
    return LabelingModelSpec(**{**base, **fields})


def _row(status_reason, *, brands=("Nike",), fp_action=FP_ACTION_CONTINUED_TO_LLM,
         fp_skip_reason=None, article_id=None, title="t"):
    return {
        "id": article_id or uuid.uuid4(),
        "title": title,
        "source_name": "s",
        "brands_mentioned": list(brands),
        "labeling_error": status_reason,
        "content": "c",
        "fp_action": fp_action,
        "fp_skip_reason": fp_skip_reason,
    }


def _judgment(article_id, brand, content_type, reasoning="r", gated=(), version="v1.13.0"):
    return {"article_id": article_id, "brand": brand, "content_type": content_type,
            "reasoning": reasoning, "gated_categories": list(gated), "prompt_version": version}


class TestModelAndRequest:
    def test_the_default_model_is_haiku_5_5(self):
        from src.agent.config import AgentSettings

        assert DEFAULT_MODEL == "claude-haiku-5-5"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_LLM_MODEL", None)
            assert AgentSettings().llm_analysis_model == "claude-haiku-5-5"

    def test_haiku_5_5_sends_medium_effort_and_no_temperature(self):
        analyzer, client = _analyzer("claude-haiku-5-5")
        assert _analyze(analyzer).success
        kwargs = client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["extra_body"] == {"output_config": {"effort": "medium"}}
        assert kwargs["max_tokens"] == 8000

    def test_haiku_4_5_sends_temperature_no_effort_and_8000_tokens(self):
        analyzer, client = _analyzer("claude-haiku-4-5-20251001")
        assert _analyze(analyzer).success
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["temperature"] == 0.0
        assert "extra_body" not in kwargs
        # Its spec's default is 2000, which truncated the JSON before #42
        assert kwargs["max_tokens"] == 8000

    def test_the_request_follows_the_spec_not_the_model_name(self):
        specs = {
            "claude-test-thinks": _spec(accepts_sampling=False, accepts_effort=True,
                                        default_max_tokens=12000),
            "claude-test-samples": _spec(accepts_sampling=True, accepts_effort=False,
                                         default_max_tokens=1000),
        }
        with patch.dict("src.labeling.config.LABELING_MODELS", specs):
            analyzer, client = _analyzer("claude-test-thinks")
            assert _analyze(analyzer).success
            thinks = client.messages.create.call_args.kwargs
            analyzer, client = _analyzer("claude-test-samples")
            assert _analyze(analyzer).success
            samples = client.messages.create.call_args.kwargs
        assert "temperature" not in thinks and thinks["max_tokens"] == 12000
        assert thinks["extra_body"] == {"output_config": {"effort": "medium"}}
        assert samples["temperature"] == 0.0 and samples["max_tokens"] == 8000
        assert "extra_body" not in samples

    def test_the_effort_does_not_follow_labeling_effort(self):
        with _settings(labeling_effort="high"):
            analyzer, client = _analyzer("claude-haiku-5-5")
            assert _analyze(analyzer).success
        assert client.messages.create.call_args.kwargs["extra_body"] == {
            "output_config": {"effort": "medium"}
        }

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

    def test_every_text_block_is_read(self):
        analyzer, _ = _analyzer(response=_response(
            _block("text", '{"overall_assessment": '), _block("thinking"), _block("text", '"fine"}')
        ))
        assert _analyze(analyzer).analysis == {"overall_assessment": "fine"}

    @pytest.mark.parametrize("blocks", [(_block("thinking"),), (_block("text", "  "),), ()])
    def test_a_reply_with_no_text_is_a_failure(self, blocks):
        analyzer, _ = _analyzer(response=_response(*blocks))
        result = _analyze(analyzer)
        assert result.success is False
        assert result.error == "Analysis reply contained no text"

    def test_thinking_that_hits_max_tokens_is_reported_as_truncated(self):
        analyzer, _ = _analyzer(response=_response(_block("thinking"), stop_reason="max_tokens"))
        result = _analyze(analyzer)
        assert result.success is False
        assert result.truncated is True
        assert "truncated" in result.error

    def test_a_refusal_is_a_failure(self):
        analyzer, _ = _analyzer(
            response=_response(_block("text", ANALYSIS_JSON), stop_reason="refusal")
        )
        result = _analyze(analyzer)
        assert result.success is False
        assert result.error == "Analysis refused by the model"


class TestLabelingInstructions:
    def test_the_analyst_is_given_the_rendered_production_prompt(self):
        version = _production_version()
        with _settings():
            analyzer, client = _analyzer()
            assert _analyze(analyzer).success
        system = client.messages.create.call_args.kwargs["system"]
        assert (
            f'<labeling_instructions version="{version}">\n{_rendered(version)}\n'
            "</labeling_instructions>"
        ) in system
        # The rules the 2026-11-03 analysis lacked, and the content types (D037)
        assert "Governance is NOT Financial News" in system
        assert "CRITICAL: Content Type" in system
        assert "{brands}" not in system
        for brand in BRANDS:
            assert brand in system

    def test_the_labeling_prompt_version_override_is_rendered(self):
        with _settings(prompt_version="v1.12.0"):
            analyzer, client = _analyzer()
            assert _analyze(analyzer).success
        system = client.messages.create.call_args.kwargs["system"]
        v12, v13 = _rendered("v1.12.0"), _rendered("v1.13.0")
        assert f'<labeling_instructions version="v1.12.0">\n{v12}\n</labeling_instructions>' in system
        only_v13 = [line for line in v13.splitlines() if line.strip() and line not in v12]
        assert only_v13, "v1.13.0 has no line v1.12.0 lacks; pick other versions"
        assert not [line for line in only_v13 if line in system]

    def test_a_prompt_that_does_not_load_fails_the_analysis(self):
        with _settings(prompt_version="v9.9.9"):
            analyzer, client = _analyzer()
            result = _analyze(analyzer)
        assert result.success is False
        assert "Labeling instructions could not be loaded" in result.error
        assert "v9.9.9" in result.error
        client.messages.create.assert_not_called()


class TestReplayOf20261103:
    """Synthetic rows shaped like 2026-11-03's product-release skips (AC 5's stubbed replay;
    plumbing only: the real rows went through the owner-approved live call)."""

    def test_the_prompt_carries_each_brand_s_content_type(self):
        nike, adidas = uuid.uuid4(), uuid.uuid4()
        rows = {"skipped": [
            _row(SKIP_NO_ESG_CATEGORIES, article_id=nike, title="Nike Air Max 95 releases Nov 14"),
            _row(SKIP_NO_ESG_CATEGORIES, brands=["Adidas"], article_id=adidas,
                 title="Adidas Samba OG gets a suede colorway"),
        ]}
        judgments = [
            _judgment(nike, "Nike", "product", "Reports the release date of a Nike sneaker."),
            _judgment(adidas, "Adidas", "product", "Describes a new Adidas colorway.",
                      gated=["environmental"]),
        ]
        _, _, skipped = build_samples(rows, judgments, [])
        with _settings():
            analyzer, client = _analyzer()
            assert _analyze(analyzer, skipped=skipped).success
        kwargs = client.messages.create.call_args.kwargs
        user = kwargs["messages"][0]["content"]
        assert user.count(f"- Status: skipped (labeler) — reason: {SKIP_NO_ESG_CATEGORIES}") == 2
        assert "Nike: product — Reports the release date of a Nike sneaker." in user
        assert "Adidas: product — Describes a new Adidas colorway." in user
        assert "dropped by the content type: Environmental" in user
        assert "not judged" not in user
        assert "Governance is NOT Financial News" in kwargs["system"]


class TestBuildSamples:
    def _deciders(self, rows):
        labeled, fp, skipped = build_samples(rows, [], [])
        return {s["reason"]: s["decider"] for s in labeled + fp + skipped}

    def test_who_decided_each_row_comes_from_provenance(self):
        fp_skip = "Likely false positive (low risk): probability 0.01 < threshold 0.6"
        rows = {
            "skipped": [_row(SKIP_NO_ESG_CATEGORIES), _row(SKIP_NO_ESG_CONTENT),
                        _row(SKIP_INSUFFICIENT_CONTENT), _row("left behind by a move")],
            "false_positive": [
                _row(fp_skip, fp_action=FP_ACTION_SKIPPED_LLM, fp_skip_reason=fp_skip),
                _row(f"{FP_REASON_AUTO_DETECTED_PREFIX} keyword"),
                _row("Nike: mention_only: named in a market wrap"),
            ],
        }
        assert self._deciders(rows) == {
            SKIP_NO_ESG_CATEGORIES: "labeler",
            SKIP_NO_ESG_CONTENT: "unjudged",
            SKIP_INSUFFICIENT_CONTENT: "unjudged",
            "left behind by a move": "unattributed",
            fp_skip: "FP classifier",
            f"{FP_REASON_AUTO_DETECTED_PREFIX} keyword": "keyword rule",
            "Nike: mention_only: named in a market wrap": "labeler",
        }

    def test_human_decided_rows_are_left_out(self):
        ruled_id, ruled_status = next(iter(provenance.HUMAN_RULINGS.items()))
        other_status = "labeled" if ruled_status != "labeled" else "skipped"
        rows = {
            "skipped": [_row(MANUAL_SKIP_REASON, title="manual skip")],
            "false_positive": [_row(FP_REASON_MANUAL, title="manual fp")],
            ruled_status: [_row(SKIP_NO_ESG_CATEGORIES if ruled_status == "skipped" else None,
                                article_id=uuid.UUID(ruled_id), title="ruled")],
        }
        rows.setdefault(other_status, []).append(
            _row(None, article_id=uuid.UUID(ruled_id), title="same id, another status")
        )
        labeled, fp, skipped = build_samples(rows, [], [])
        titles = [s["title"] for s in labeled + fp + skipped]
        # A ruling counts only while the article still has the ruled status
        assert titles == ["same id, another status"]

    def test_rows_attach_by_uuid_and_brand_spelling(self):
        article_id = uuid.uuid4()
        rows = {"labeled": [_row(None, brands=["Adidas", "Puma"], article_id=article_id)]}
        judgments = [_judgment(article_id, "Adidas", "business")]
        labels = [{"article_id": article_id, "brand": "adidas ", "social": True,
                   "social_sentiment": -1}]
        (sample,), _, _ = build_samples(rows, judgments, labels)
        assert sample["id"] == str(article_id)
        assert [(line["brand"], line["judgment"] and line["judgment"]["content_type"],
                 line["stored"]) for line in sample["brand_lines"]] == [
            ("Adidas", "business", [("Social", -1)]),
            ("Puma", None, []),
        ]

    def test_a_label_with_no_judgment_keeps_its_categories(self):
        samples = [{"id": "a1", "brands": ["Nike"]}]
        attach_brand_records(samples, [], [{"article_id": "a1", "brand": "Nike",
                                            "governance": True, "governance_sentiment": 1}])
        assert samples[0]["brand_lines"] == [
            {"brand": "Nike", "judgment": None, "stored": [("Governance", 1)]}
        ]

    def test_stored_categories_follow_the_prompt_order(self):
        label = {"digital_transformation": True, "environmental": True,
                 "digital_sentiment": 1, "environmental_sentiment": -1}
        assert stored_categories(label) == [("Environmental", -1), ("Digital Transformation", 1)]


class TestFormatting:
    def _text(self, article, status="skipped", prompt_version="v1.13.0"):
        analyzer, _ = _analyzer()
        return analyzer._format_article_sample([article], status, prompt_version)

    def _article(self, **fields):
        return {"id": "a", "title": "t", "source_name": "s", "brands": ["Nike"], **fields}

    @pytest.mark.parametrize("status,decider,reason", [
        ("skipped", "unjudged", SKIP_NO_ESG_CONTENT),
        ("skipped", "unattributed", "left behind"),
        ("false_positive", "FP classifier", "Likely false positive (low risk)"),
        ("false_positive", "labeler", "Nike: other_entity: the animal"),
    ])
    def test_the_status_line_names_the_decider_and_the_reason(self, status, decider, reason):
        text = self._text(self._article(decider=decider, reason=reason), status)
        assert f"- Status: {status} ({decider}) — reason: {reason}" in text

    def test_brand_lines(self):
        long_reasoning = "line one\nline two " + "x" * 300
        article = self._article(brand_lines=[
            {"brand": "Nike", "stored": [("Governance", 0), ("Social", None)],
             "judgment": {"brand": "Nike", "content_type": "business", "reasoning": long_reasoning,
                          "gated_categories": ["digital_transformation"],
                          "prompt_version": "v1.12.0"}},
            {"brand": "Puma", "judgment": None, "stored": [("Social", -1)]},
            {"brand": "Asics", "judgment": None, "stored": []},
        ])
        text = self._text(article, "labeled")
        snippet = long_reasoning.replace("\n", " ")[:200] + "..."
        assert f"  - Nike: business (judged under v1.12.0) — {snippet}" in text
        assert "    dropped by the content type: Digital Transformation" in text
        assert "    stored categories: Governance (0), Social (no sentiment)" in text
        assert "  - Puma: not judged (no content type recorded)\n    stored categories: Social (-1)" in text
        assert "  - Asics: not judged (no content type recorded)" in text

    def test_a_judgment_under_the_instructions_version_says_nothing_more(self):
        article = self._article(brand_lines=[{"brand": "Nike", "stored": [], "judgment": {
            "brand": "Nike", "content_type": "product", "reasoning": "r",
            "prompt_version": "v1.13.0"}}])
        assert "  - Nike: product — r" in self._text(article)

    def test_brands_without_attached_records(self):
        assert "- Brands: Nike" in self._text(self._article())
        assert "- Brands: none recorded" in self._text(self._article(brand_lines=[]))


class TestSampleQueries:
    """get_recent_labeling_samples's wiring, with the three queries stubbed."""

    def _run(self, article_rows=None, judgments=None, labels=None, **raises):
        engine = MagicMock()
        stubs = {
            "_article_rows": MagicMock(side_effect=lambda conn, status, since, limit:
                                       (article_rows or {}).get(status, [])),
            "_judgment_rows": MagicMock(return_value=judgments or []),
            "_label_rows": MagicMock(return_value=labels or []),
        }
        for name, error in raises.items():
            stubs[name].side_effect = error
        with patch("sqlalchemy.create_engine", return_value=engine), \
                patch.multiple("src.agent.llm", **stubs):
            try:
                return get_recent_labeling_samples(database_url="postgresql://x/y"), stubs
            finally:
                engine.dispose.assert_called_once()

    def test_samples_are_built_from_the_three_queries(self):
        labeled_id, skipped_id = uuid.uuid4(), uuid.uuid4()
        (labeled, fp, skipped), stubs = self._run(
            article_rows={"labeled": [_row(None, article_id=labeled_id)],
                          "skipped": [_row(SKIP_NO_ESG_CATEGORIES, article_id=skipped_id)]},
            judgments=[_judgment(skipped_id, "Nike", "product")],
            labels=[{"article_id": labeled_id, "brand": "Nike", "social": True,
                     "social_sentiment": -1}],
        )
        assert [s["decider"] for s in labeled + skipped] == ["labeler", "labeler"]
        assert skipped[0]["brand_lines"][0]["judgment"]["content_type"] == "product"
        assert labeled[0]["brand_lines"][0]["stored"] == [("Social", -1)]
        assert fp == []
        assert stubs["_judgment_rows"].call_args.args[1] == [labeled_id, skipped_id]
        assert stubs["_label_rows"].call_args.args[1] == [labeled_id]

    @pytest.mark.parametrize("query", ["_judgment_rows", "_label_rows"])
    def test_a_failed_brand_query_raises_rather_than_returning_bare_samples(self, query):
        rows = {"labeled": [_row(None)]}
        with pytest.raises(SampleQueryError, match="relation does not exist"):
            self._run(article_rows=rows, **{query: RuntimeError("relation does not exist")})

    def test_a_failed_connection_raises(self):
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


def test_experiment_reflection_keeps_its_own_model():
    """AGENT_LLM_MODEL is the daily analysis's; the reflector's request is Haiku 4.5's (#257)."""
    from src.agent.workflows import model_training

    tracker = MagicMock()
    reflector_class = MagicMock()
    with patch.object(model_training, "agent_settings") as settings, \
            patch.object(model_training, "_get_tracker_class", return_value=MagicMock(return_value=tracker)), \
            patch.object(model_training, "_get_reflector_class", return_value=reflector_class):
        settings.llm_analysis_enabled = True
        settings.anthropic_api_key = "k"
        settings.llm_analysis_model = "claude-haiku-5-5"
        model_training.finalize_experiments(
            MagicMock(), {"experiment_ids": {"fp": "fp_1"}, "promoted": ["fp"], "classifiers": {}}
        )
    reflector_class.assert_called_once_with(api_key="k")


FORBIDDEN = ("src.labeling", "src.data_collection")


def _module_level_imports(source: str, package: str = "src.agent") -> list[str]:
    """Every import outside a function body, relative ones resolved against ``package``."""
    names = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = package.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1]) if node.level else ""
            module = ".".join(p for p in (base, node.module or "") if p)
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return names


@pytest.mark.parametrize("source", [
    "from ..labeling.config import x",
    "from src import labeling",
    "try:\n    import src.data_collection.config\nexcept ImportError:\n    pass",
    "class C:\n    from src.labeling import models",
])
def test_the_import_check_sees_every_module_level_form(source):
    assert any(n.startswith(FORBIDDEN) for n in _module_level_imports(source))


def test_llm_module_imports_nothing_from_labeling_or_data_collection_at_module_level():
    """The #211 constraint on #252: those imports stay inside the functions that use them."""
    names = _module_level_imports((REPO / "src/agent/llm.py").read_text())
    assert "anthropic" in names, "the check is not reading the module's imports"
    assert not [n for n in names if n.startswith(FORBIDDEN)]
    # A function-level import is not module level
    assert not _module_level_imports("def f():\n    from src.labeling import x")


requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@requires_postgres
class TestSampleQueriesAgainstPostgres:
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
        db.engine.dispose()

    def test_samples_carry_deciders_judgments_and_labels_and_leave_out_humans(self, database):
        from datetime import datetime, timezone

        from src.data_collection import models

        db, url, created = database
        now = datetime.now(timezone.utc)

        def article(title, status, **fields):
            return models.Article(
                article_id=f"test-{uuid.uuid4()}", url=f"https://example.com/{uuid.uuid4()}",
                title=f"{title} {uuid.uuid4()}", labeling_status=status, **fields,
            )

        with db.get_session() as session:
            run = models.LabelingRun()
            skipped = article("release", "skipped", brands_mentioned=["Nike"],
                              labeling_error=SKIP_NO_ESG_CATEGORIES, skipped_at=now)
            labeled = article("job cuts", "labeled", brands_mentioned=["Puma"], labeled_at=now)
            manual = article("manual", "skipped", brands_mentioned=["Nike"],
                             labeling_error=MANUAL_SKIP_REASON, skipped_at=now)
            human_label = article("human", "labeled", brands_mentioned=["Adidas"], labeled_at=now)
            session.add_all([run, skipped, labeled, manual, human_label])
            session.flush()

            def judged(row, brand, content_type):
                return models.BrandJudgment(
                    article_id=row.id, labeling_run_id=run.id, brand=brand,
                    content_type=content_type, reasoning="r", prompt_version="v1.13.0",
                    model="claude-haiku-5-5",
                )

            session.add_all([
                judged(skipped, "Nike", "product"),
                judged(labeled, "Puma", "business"),
                judged(manual, "Nike", "business"),
                models.BrandLabel(article_id=labeled.id, brand="Puma", social=True,
                                  social_sentiment=-1, labeled_by="claude-haiku-5-5"),
                models.BrandLabel(article_id=human_label.id, brand="Adidas",
                                  digital_transformation=True, digital_sentiment=1,
                                  labeled_by="human", human_reviewed=True),
            ])
            ids = {name: str(row.id) for name, row in
                   (("skipped", skipped), ("labeled", labeled), ("manual", manual),
                    ("human_label", human_label))}
            created += [(row.id, run.id) for row in (skipped, labeled, manual, human_label)]

        labeled_sample, _, skipped_sample = get_recent_labeling_samples(
            days=1, sample_size=100, database_url=url
        )
        mine = {s["id"]: s for s in labeled_sample + skipped_sample if s["id"] in ids.values()}
        assert set(mine) == {ids["skipped"], ids["labeled"]}
        skipped_row, labeled_row = mine[ids["skipped"]], mine[ids["labeled"]]
        assert (skipped_row["decider"], skipped_row["reason"]) == ("labeler", SKIP_NO_ESG_CATEGORIES)
        assert [(line["brand"], line["judgment"]["content_type"], line["stored"])
                for line in skipped_row["brand_lines"]] == [("Nike", "product", [])]
        assert [(line["brand"], line["judgment"]["content_type"], line["stored"])
                for line in labeled_row["brand_lines"]] == [("Puma", "business", [("Social", -1)])]
