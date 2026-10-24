"""Tests for the per-brand content type stored from each labeling reply (D038, #192)."""

import os
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.data_collection import models
from src.labeling.database import LabelingDatabase
from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED, dispose_reply
from src.labeling.labeler import LabelingResult
from src.labeling.models import BrandAnalysis, CategoryLabel, ContentType, LabelingResponse
from src.labeling.pipeline import LabelingPipeline

CATEGORIES = ("environmental", "social", "governance", "digital_transformation")


def _categories(esg):
    return {
        c: CategoryLabel(
            applies=esg and c == "social",
            sentiment=1 if esg and c == "social" else None,
            evidence=["quote"] if esg and c == "social" else [],
        )
        for c in CATEGORIES
    }


def _typed(brand, content_type, esg=False):
    """A v1.12.0 analysis; a non-business type with esg=True has its category gated (D037)."""
    return BrandAnalysis(
        brand=brand,
        content_type=content_type,
        categories=_categories(esg),
        confidence=0.8,
        reasoning=f"{brand} reasoning",
    )


def _legacy(brand, sportswear=True):
    """A v1.11.0-style analysis: a verdict and no content type."""
    return BrandAnalysis(
        brand=brand,
        is_sportswear_brand=sportswear,
        not_sportswear_reason=None if sportswear else "the animal",
        confidence=0.9,
        reasoning="r",
    )


class TestJudged:
    def test_keyed_by_the_sent_spelling(self):
        d = dispose_reply(["The North Face"], [_typed("the north face", ContentType.PRODUCT)])
        assert list(d.judged) == ["The North Face"]
        assert d.judged["The North Face"].content_type == ContentType.PRODUCT

    def test_sportswear_wins_a_conflict_as_it_does_for_the_status(self):
        d = dispose_reply(
            ["Puma"],
            [_typed("Puma", ContentType.OTHER_ENTITY), _typed("Puma", ContentType.PRODUCT)],
        )
        assert d.conflicting == ["Puma"]
        assert d.judged["Puma"].content_type == ContentType.PRODUCT

    def test_otherwise_the_first_analysis_wins(self):
        d = dispose_reply(
            ["Nike"], [_typed("Nike", ContentType.PRODUCT), _typed("Nike", ContentType.FINANCIAL)]
        )
        assert d.judged["Nike"].content_type == ContentType.PRODUCT

    def test_left_out_unsent_and_unset_brands_have_no_entry(self):
        unset = BrandAnalysis(brand="Puma", reasoning="r")  # no verdict stated
        d = dispose_reply(
            ["Nike", "Adidas", "Puma"],
            [_typed("Nike", ContentType.MENTION_ONLY), _typed("Reebok", ContentType.PRODUCT), unset],
        )
        assert list(d.judged) == ["Nike"]
        assert (d.omitted, d.unsent, d.unset) == (["Adidas", "Puma"], ["Reebok"], ["Puma"])

    def test_empty_reply_judges_nothing(self):
        assert dispose_reply(["Nike"], []).judged == {}

    def test_legacy_analyses_are_judged_without_a_content_type(self):
        d = dispose_reply(["Nike"], [_legacy("Nike")])
        assert d.judged["Nike"].content_type is None


class TestSaveBrandJudgments:
    def _save(self, judged, session=None):
        session = session or MagicMock()
        rows = LabelingDatabase(MagicMock()).save_brand_judgments(
            session, uuid4(), uuid4(), judged, model="claude-haiku-5-5", prompt_version="v1.12.0"
        )
        return session, rows

    def test_writes_one_row_per_typed_brand_with_its_provenance(self):
        judged = dispose_reply(
            ["Nike", "Adidas"],
            [_typed("nike", ContentType.PRODUCT, esg=True), _typed("Adidas", ContentType.BUSINESS, esg=True)],
        ).judged
        session, rows = self._save(judged)
        by_brand = {r.brand: r for r in rows}
        assert set(by_brand) == {"Nike", "Adidas"}  # the sent spelling
        assert by_brand["Nike"].content_type == "product"
        assert by_brand["Nike"].gated_categories == ["social"]  # dropped by the product type
        assert by_brand["Adidas"].gated_categories == []
        assert by_brand["Nike"].reasoning == "nike reasoning"
        assert by_brand["Nike"].confidence == 0.8
        assert {(r.model, r.prompt_version) for r in rows} == {("claude-haiku-5-5", "v1.12.0")}
        session.add_all.assert_called_once_with(rows)

    def test_replaces_the_articles_rows_first(self):
        session = MagicMock()
        calls = []
        session.query.return_value.filter.return_value.delete.side_effect = lambda: calls.append("delete")
        session.add_all.side_effect = lambda rows: calls.append("add")
        self._save({"Nike": _typed("Nike", ContentType.PRODUCT)}, session)
        assert session.query.call_args.args == (models.BrandJudgment,)
        assert calls == ["delete", "add"]

    def test_a_legacy_reply_writes_no_rows(self):
        session, rows = self._save({"Nike": _legacy("Nike")})
        assert rows == []
        # its earlier rows are still removed: they no longer describe the latest reply
        session.query.return_value.filter.return_value.delete.assert_called_once()


class TestPipelineWritesJudgments:
    """Judgments are written once per outcome the LLM decides, in the status write's session."""

    def _article(self, brands=("Nike",), **overrides):
        article = {
            "id": uuid4(),
            "title": "Test Article",
            "full_content": "Test content " * 50,
            "description": "Test description",
            "brands_mentioned": list(brands),
            "published_at": datetime.now(),
            "source_name": "Test Source",
        }
        article.update(overrides)
        return article

    def _database(self):
        """A mocked database that opens a distinct session each time, recording the order."""
        database = MagicMock()
        database.save_brand_labels.return_value = []
        sessions = []

        @contextmanager
        def get_session():
            session = MagicMock(name=f"session{len(sessions)}")
            sessions.append(session)
            yield session

        database.db.get_session.side_effect = get_session
        return database

    def _labeler(self, analyses):
        labeler = MagicMock()
        labeler.label_article.return_value = LabelingResult(
            success=True,
            response=LabelingResponse(brand_analyses=analyses, article_summary="s"),
            model="claude-haiku-5-5",
            prompt_version="v1.12.0",
        )
        return labeler

    def _run(self, database, labeler, article=None, **kwargs):
        kwargs.setdefault("labeling_run_id", uuid4())
        return LabelingPipeline(database=database, labeler=labeler)._process_article(
            article or self._article(), skip_chunking=True, skip_embedding=True, **kwargs
        )

    @pytest.mark.parametrize(
        "brands,analyses,status,reason",
        [
            (["Nike"], [_typed("Nike", ContentType.BUSINESS, esg=True)], LABELED, None),
            (["Nike"], [_typed("Nike", ContentType.PRODUCT)], SKIPPED, models.SKIP_NO_ESG_CATEGORIES),
            (["Nike", "Puma"], [_typed("Nike", ContentType.MENTION_ONLY)], SKIPPED,
             models.SKIP_INCOMPLETE_JUDGMENT),
            (["Nike"], [_typed("Nike", ContentType.OTHER_ENTITY)], FALSE_POSITIVE, None),
            (["Nike"], [], SKIPPED, models.SKIP_NO_ESG_CONTENT),
        ],
    )
    def test_each_llm_outcome_writes_once_in_the_status_session(self, brands, analyses, status, reason):
        database = self._database()
        run_id = uuid4()
        self._run(database, self._labeler(analyses), self._article(brands), labeling_run_id=run_id)

        assert database.save_brand_judgments.call_count == 1
        judgment = database.save_brand_judgments.call_args
        update = database.update_article_labeling_status.call_args
        assert update.args[2] == status
        if reason is not None:
            assert update.args[3] == reason
        assert judgment.args[0] is update.args[0], "judgments and status share one session"
        assert judgment.args[2] == run_id
        expected = {b for b in brands if any(a.brand == b for a in analyses)}
        assert set(judgment.args[3]) == expected
        assert judgment.kwargs == {"model": "claude-haiku-5-5", "prompt_version": "v1.12.0"}

    @pytest.mark.parametrize(
        "article,kwargs",
        [
            ({"full_content": "Short", "description": None}, {}),
            ({"brands_mentioned": []}, {}),
            ({}, {"fp_prefilter_result": (False, None)}),
            ({}, {"dry_run": True}),
        ],
        ids=["insufficient-content", "no-brands", "fp-prefilter-skip", "dry-run"],
    )
    def test_paths_the_llm_does_not_decide_write_nothing(self, article, kwargs):
        database = self._database()
        self._run(database, self._labeler([_typed("Nike", ContentType.PRODUCT)]),
                  self._article(**article), **kwargs)
        database.save_brand_judgments.assert_not_called()

    def test_a_failed_call_writes_nothing(self):
        database = self._database()
        labeler = MagicMock()
        labeler.label_article.return_value = LabelingResult(
            success=False, error="Rate limited", error_type="rate_limit"
        )
        self._run(database, labeler)
        database.save_brand_judgments.assert_not_called()

    def test_saving_an_llm_outcome_without_a_run_id_raises_before_the_status_write(self):
        database = self._database()
        with pytest.raises(ValueError, match="labeling run"):
            self._run(database, self._labeler([_typed("Nike", ContentType.PRODUCT)]),
                      labeling_run_id=None)
        database.update_article_labeling_status.assert_not_called()

    def test_label_articles_passes_its_run_id(self):
        from unittest.mock import patch

        database = MagicMock()
        database.create_labeling_run.return_value.id = run_id = uuid4()
        pipeline = LabelingPipeline(database=database)
        article = self._article()
        with patch("src.labeling.pipeline.db"), \
             patch.object(pipeline, "_ensure_labeler"), \
             patch.object(pipeline, "_deduplicate_by_title", return_value=([article], [])), \
             patch.object(pipeline, "_compute_novelty_batch", return_value={}), \
             patch.object(pipeline, "_run_fp_prefilter_batch", return_value={}), \
             patch.object(pipeline, "_process_article", return_value={"labeled": True}) as process:
            pipeline.label_articles(article_ids=[article["id"]])
        assert process.call_args.kwargs["labeling_run_id"] == run_id


requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@requires_postgres
class TestBrandJudgmentsTable:
    @pytest.fixture
    def database(self):
        from src.data_collection.database import Database

        # Never DATABASE_URL: it may point at production
        db = Database(
            database_url=os.environ.get(
                "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news_test"
            )
        )
        db.init_db()
        created = []
        yield LabelingDatabase(db), created
        with db.get_session() as session:
            for article_id, run_id in created:
                session.query(models.Article).filter(models.Article.id == article_id).delete()
                session.query(models.LabelingRun).filter(models.LabelingRun.id == run_id).delete()

    def _article_and_run(self, labeling_db, created):
        with labeling_db.db.get_session() as session:
            article = models.Article(
                article_id=f"test-{uuid4()}", url=f"https://example.com/{uuid4()}", title="t",
                brands_mentioned=["Nike"],
                labeling_status="skipped", labeling_error=models.SKIP_NO_ESG_CONTENT,
            )
            run = models.LabelingRun()
            session.add_all([article, run])
            session.flush()
            ids = (article.id, run.id)
        created.append(ids)
        return ids

    def _rows(self, labeling_db, article_id):
        with labeling_db.db.get_session() as session:
            return sorted(
                (r.brand, r.content_type)
                for r in session.query(models.BrandJudgment).filter_by(article_id=article_id)
            )

    def _save(self, labeling_db, session, article_id, run_id, judged, prompt_version="v1.12.0"):
        labeling_db.save_brand_judgments(
            session, article_id, run_id, judged, model="claude-haiku-5-5",
            prompt_version=prompt_version,
        )

    def test_a_relabel_replaces_the_rows(self, database):
        labeling_db, created = database
        article_id, run_id = self._article_and_run(labeling_db, created)
        with labeling_db.db.get_session() as session:
            self._save(labeling_db, session, article_id, run_id,
                       {"Nike": _typed("Nike", ContentType.PRODUCT),
                        "Puma": _typed("Puma", ContentType.OTHER_ENTITY)})
        with labeling_db.db.get_session() as session:
            self._save(labeling_db, session, article_id, run_id,
                       {"Nike": _typed("Nike", ContentType.FINANCIAL)})
        assert self._rows(labeling_db, article_id) == [("Nike", "financial")]

    def test_a_failed_write_rolls_back_the_status_written_with_it(self, database):
        labeling_db, created = database
        article_id, run_id = self._article_and_run(labeling_db, created)
        with pytest.raises(Exception):
            with labeling_db.db.get_session() as session:
                labeling_db.update_article_labeling_status(session, article_id, "false_positive", "x")
                # prompt_version is NOT NULL
                self._save(labeling_db, session, article_id, run_id,
                           {"Nike": _typed("Nike", ContentType.OTHER_ENTITY)}, prompt_version=None)
        with labeling_db.db.get_session() as session:
            article = session.get(models.Article, article_id)
            assert (article.labeling_status, article.labeling_error) == (
                "skipped", models.SKIP_NO_ESG_CONTENT
            )
        assert self._rows(labeling_db, article_id) == []

    def test_deleting_the_article_deletes_its_judgments(self, database):
        labeling_db, created = database
        article_id, run_id = self._article_and_run(labeling_db, created)
        with labeling_db.db.get_session() as session:
            self._save(labeling_db, session, article_id, run_id,
                       {"Nike": _typed("Nike", ContentType.PRODUCT)})
        with labeling_db.db.get_session() as session:
            session.query(models.Article).filter(models.Article.id == article_id).delete()
        assert self._rows(labeling_db, article_id) == []
