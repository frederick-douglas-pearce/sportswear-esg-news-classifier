"""Stored rows whose text names none of their brands (#227, D044): the cohort and the triage."""

import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data_collection import models
from src.data_collection.models import (
    FP_ACTION_CONTINUED_TO_LLM,
    FP_ACTION_SKIPPED_LLM,
    FP_REASON_MANUAL,
    MANUAL_SKIP_REASON,
    SCRAPE_NO_BRAND_ERROR,
    SKIP_NO_ESG_CONTENT,
)
from src.data_collection.provenance import (
    EXCLUDED_HUMAN,
    EXCLUDED_HUMAN_LABELS,
    EXCLUDED_OTHER_STATUS,
    EXCLUDED_POSSIBLE_HAND_MOVE,
    HUMAN_RULINGS,
    NoBrandRow,
    no_brand_exclusion,
    select_no_brand_content,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import triage_no_brand_content as triage  # noqa: E402

FP_SKIP = "Low sportswear probability (0.12)"
RULED_ID, RULED_STATUS = next(iter(HUMAN_RULINGS.items()))


def row(status, reason=None, *, action=None, skip_reason=None, has_labels=False,
        human_labels=False, article_id=None):
    return NoBrandRow(
        article_id=article_id or str(uuid.uuid4()),
        labeling_status=status,
        labeling_error=reason,
        fp_action=action,
        fp_skip_reason=skip_reason,
        brands=["Nike"],
        has_labels=has_labels,
        human_labels=human_labels,
    )


class TestExclusion:
    @pytest.mark.parametrize(
        "candidate",
        [
            row("pending"),
            row("failed", "API error"),
            row("labeled", has_labels=True),
            row("skipped", SKIP_NO_ESG_CONTENT),
            row("false_positive", "Nike: not the sportswear brand", action=FP_ACTION_CONTINUED_TO_LLM),
            row("false_positive", "Nike: not the sportswear brand"),  # before the FP classifier
            # the FP classifier decided it: the stored reason is the skip's own
            row("false_positive", FP_SKIP, action=FP_ACTION_SKIPPED_LLM, skip_reason=FP_SKIP),
        ],
    )
    def test_moved(self, candidate):
        assert no_brand_exclusion(candidate) is None

    @pytest.mark.parametrize(
        "candidate, why",
        [
            (row(RULED_STATUS, article_id=RULED_ID), EXCLUDED_HUMAN),
            (row("pending", article_id=RULED_ID), EXCLUDED_HUMAN),
            (row("deduplicated", "Duplicate title of x"), EXCLUDED_OTHER_STATUS),
            (row("labeled", has_labels=True, human_labels=True), EXCLUDED_HUMAN_LABELS),
            (row("false_positive", FP_REASON_MANUAL, action=FP_ACTION_CONTINUED_TO_LLM), EXCLUDED_HUMAN),
            (row("skipped", MANUAL_SKIP_REASON), EXCLUDED_HUMAN),
            # an LLM-era negative whose reason is not a reply's "<brand>: <reason>"
            (row("false_positive", "moved after review", action=FP_ACTION_CONTINUED_TO_LLM),
             EXCLUDED_POSSIBLE_HAND_MOVE),
            (row("false_positive", None), EXCLUDED_POSSIBLE_HAND_MOVE),
            # a skip reason no pipeline writes
            (row("skipped", "Scrape failed: 404"), EXCLUDED_POSSIBLE_HAND_MOVE),
            (row("labeled", "left by fix_label", has_labels=True), EXCLUDED_POSSIBLE_HAND_MOVE),
            (row("labeled", has_labels=False), EXCLUDED_POSSIBLE_HAND_MOVE),
        ],
    )
    def test_left_out(self, candidate, why):
        assert no_brand_exclusion(candidate) == why


def test_eval_case_ids_reads_every_case_file():
    ids = triage.eval_case_ids()
    assert "fd48d9a7-4993-4651-b770-5d232e70d918" in ids  # owner_rulings_192.json


requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)

NO_BRAND = "When Jeff Bezos said that one technology would shape Amazon, analysts were caught off guard."
NAMES_BRAND = "Nike reported quarterly results and raised its guidance for the year."


@requires_postgres
class TestTriageDatabase:
    @pytest.fixture
    def test_db(self):
        from src.data_collection.database import Database

        # Never DATABASE_URL: it may point at production.
        database_url = os.environ.get(
            "TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/esg_news_test"
        )
        database = Database(database_url=database_url)
        database.init_db()
        yield database
        with database.get_session() as session:
            session.query(models.BrandLabel).delete()
            session.query(models.ClassifierPrediction).delete()
            session.query(models.Article).delete()

    def _article(self, session, key, status, reason=None, *, content=NO_BRAND, article_uuid=None,
                 brands=("Nike",), scrape_status="success"):
        article = models.Article(
            id=article_uuid or uuid.uuid4(),
            article_id=f"nobrand-{key}",
            title=f"Nike headline {key}",
            url=f"https://example.com/{key}",
            full_content=content,
            brands_mentioned=list(brands),
            scrape_status=scrape_status,
            labeling_status=status,
            labeling_error=reason,
        )
        session.add(article)
        session.flush()
        return article

    def _label(self, session, article, labeled_by="claude-sonnet", reviewed=False):
        session.add(models.BrandLabel(
            article_id=article.id, brand="Nike", labeled_by=labeled_by, human_reviewed=reviewed,
        ))
        session.flush()

    def _seed(self, session):
        moved = {
            "labeled": self._article(session, "labeled", "labeled"),
            "fp": self._article(session, "fp", "false_positive", "Nike: not the sportswear brand"),
            "skipped": self._article(session, "skipped", "skipped", SKIP_NO_ESG_CONTENT),
            "pending": self._article(session, "pending", "pending"),
        }
        self._label(session, moved["labeled"])
        left = {
            "names_brand": self._article(session, "names", "labeled", content=NAMES_BRAND),
            "unlabelable": self._article(session, "unlabelable", "unlabelable", "Scrape failed"),
            "not_scraped": self._article(session, "failed-scrape", "pending", scrape_status="failed"),
            "no_brands": self._article(session, "no-brands", "pending", brands=()),
            "ruled": self._article(session, "ruled", RULED_STATUS, article_uuid=uuid.UUID(RULED_ID)),
            "hand_move": self._article(session, "hand", "false_positive", "moved by hand"),
        }
        reviewed = self._article(session, "reviewed", "labeled")
        self._label(session, reviewed, reviewed=True)
        left["reviewed"] = reviewed
        # ids only: the rows are read again after the seeding session closes
        return (
            {k: SimpleNamespace(id=a.id) for k, a in moved.items()},
            {k: SimpleNamespace(id=a.id) for k, a in left.items()},
        )

    def test_selection(self, test_db):
        with test_db.get_session() as session:
            moved, left = self._seed(session)
            cohort = select_no_brand_content(session)

        assert set(cohort.selected) == {str(a.id) for a in moved.values()}
        assert cohort.left_out == {
            str(left["ruled"].id): EXCLUDED_HUMAN,
            str(left["hand_move"].id): EXCLUDED_POSSIBLE_HAND_MOVE,
            str(left["reviewed"].id): EXCLUDED_HUMAN_LABELS,
        }
        assert cohort.selected[str(moved["fp"].id)] == (
            "false_positive", "Nike: not the sportswear brand", "success", None,
        )

    def test_apply_moves_to_the_gated_scrape_state(self, test_db, tmp_path):
        from src.data_collection.database import scrape_failure_reason

        manifest = tmp_path / "manifest.jsonl"
        with test_db.get_session() as session:
            moved, left = self._seed(session)
            session.commit()
            # a scrape the gate skipped, for comparison
            gated_row = self._article(session, "gated", "pending", content=None, scrape_status="pending")
            test_db.update_article_content(
                session, gated_row.article_id, NO_BRAND, "skipped", SCRAPE_NO_BRAND_ERROR
            )
            gated = SimpleNamespace(id=gated_row.id)
            session.commit()
            assert triage.main(["--manifest", str(manifest)], database=test_db) == 0

        with test_db.get_session() as session:
            def state(article):
                a = session.get(models.Article, article.id)
                return (a.labeling_status, a.labeling_error, a.scrape_status, a.scrape_error,
                        a.full_content)

            gated_state = state(gated)
            assert gated_state == (
                "unlabelable", scrape_failure_reason("skipped", SCRAPE_NO_BRAND_ERROR),
                "skipped", SCRAPE_NO_BRAND_ERROR, NO_BRAND,
            )
            for article in moved.values():
                assert state(article) == gated_state
            for key in ("names_brand", "ruled", "hand_move", "reviewed"):
                assert state(left[key])[0] != "unlabelable"
            assert session.query(models.BrandLabel).filter_by(article_id=moved["labeled"].id).count() == 0
            assert session.query(models.BrandLabel).filter_by(article_id=left["reviewed"].id).count() == 1

            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert {r["article_id"]: r["labeling_status"] for r in records} == {
                str(a.id): status for status, a in
                [("labeled", moved["labeled"]), ("false_positive", moved["fp"]),
                 ("skipped", moved["skipped"]), ("pending", moved["pending"])]
            }
            by_id = {r["article_id"]: r for r in records}
            assert by_id[str(moved["labeled"].id)]["brand_labels_deleted"] == 1
            assert by_id[str(moved["fp"].id)]["labeling_error"] == "Nike: not the sportswear brand"
            assert by_id[str(moved["fp"].id)]["scrape_status"] == "success"

            # a second run finds nothing left to move
            assert select_no_brand_content(session).selected == {}

    def test_manifest_is_written_before_commit(self, test_db, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        with test_db.get_session() as session:
            moved, _ = self._seed(session)
            session.commit()
            counts = triage.apply_triage(session, select_no_brand_content(session), manifest)
            assert counts["moved"] == len(moved)
            assert len(manifest.read_text().splitlines()) == len(moved)
            session.rollback()
            # nothing committed: every row is as seeded
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"

    def test_row_changed_since_selection_is_not_moved(self, test_db, tmp_path):
        with test_db.get_session() as session:
            moved, _ = self._seed(session)
            session.commit()
            cohort = select_no_brand_content(session)
            # the labeling cron labels the pending row after selection
            session.get(models.Article, moved["pending"].id).labeling_status = "labeled"
            session.flush()
            counts = triage.apply_triage(session, cohort, tmp_path / "m.jsonl")
            session.commit()

            assert counts["changed"] == 1
            assert counts["moved"] == len(moved) - 1
            assert session.get(models.Article, moved["pending"].id).labeling_status == "labeled"

    def test_dry_run_writes_nothing(self, test_db, tmp_path, capsys):
        manifest = tmp_path / "manifest.jsonl"
        with test_db.get_session() as session:
            moved, _ = self._seed(session)
            session.commit()
        assert triage.main(["--dry-run", "--manifest", str(manifest)], database=test_db) == 0

        assert not manifest.exists()
        assert f"To move: {len(moved)}" in capsys.readouterr().out
        with test_db.get_session() as session:
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"
            assert session.query(models.BrandLabel).filter_by(article_id=moved["labeled"].id).count() == 1
