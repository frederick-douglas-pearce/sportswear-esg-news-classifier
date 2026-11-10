"""Stored rows whose text names none of their brands (#227, D044): the cohort and the triage."""

import json
import os
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data_collection import models
from src.data_collection.models import (
    FP_ACTION_CONTINUED_TO_LLM,
    FP_ACTION_SKIPPED_LLM,
    FP_REASON_AUTO_DETECTED_PREFIX,
    FP_REASON_MANUAL,
    MANUAL_SKIP_REASON,
    SCRAPE_NO_BRAND_ERROR,
    SKIP_NO_ESG_CATEGORIES,
    SKIP_NO_ESG_CONTENT,
)
from src.data_collection.provenance import (
    EXCLUDED_EVAL_CASE,
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
            # the keyword sweep wrote it (rule), D044
            row("false_positive", f"{FP_REASON_AUTO_DETECTED_PREFIX} vehicle keywords"),
            # a judged skip, before and after the FP classifier
            row("skipped", SKIP_NO_ESG_CATEGORIES),
            row("skipped", SKIP_NO_ESG_CATEGORIES, action=FP_ACTION_CONTINUED_TO_LLM),
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
            # a human marker is a human's whatever the status
            (row("pending", FP_REASON_MANUAL), EXCLUDED_HUMAN),
            (row("pending", MANUAL_SKIP_REASON), EXCLUDED_HUMAN),
            (row("failed", FP_REASON_MANUAL), EXCLUDED_HUMAN),
            (row("labeled", MANUAL_SKIP_REASON, has_labels=True), EXCLUDED_HUMAN),
            # the pipeline writes no reason and no label before labeling a pending row
            (row("pending", "Nike: not sportswear"), EXCLUDED_POSSIBLE_HAND_MOVE),
            (row("pending", has_labels=True), EXCLUDED_POSSIBLE_HAND_MOVE),
            (row("failed", "API error", has_labels=True), EXCLUDED_POSSIBLE_HAND_MOVE),
        ],
    )
    def test_left_out(self, candidate, why):
        assert no_brand_exclusion(candidate) == why


def test_eval_case_ids_reads_every_case_file():
    """Every article id written in every case file is excluded from the triage (D044 §3), so an
    id eval_case_ids misses would be moved."""
    root = Path(__file__).parent.parent
    paths = sorted(root.glob(triage.EVAL_CASES_GLOB))
    assert len(paths) >= 2
    ids = triage.eval_case_ids()
    for path in paths:
        written = set(re.findall(r'"article_id":\s*"([0-9a-f-]{36})"', path.read_text()))
        assert written, path.name
        assert written <= ids, path.name


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
                 brands=("Nike",), scrape_status="success", scrape_error=None):
        article = models.Article(
            id=article_uuid or uuid.uuid4(),
            article_id=f"nobrand-{key}",
            title=f"Nike headline {key}",
            url=f"https://example.com/{key}",
            full_content=content,
            brands_mentioned=list(brands),
            scrape_status=scrape_status,
            scrape_error=scrape_error,
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
            "fp": self._article(session, "fp", "false_positive", "Nike: not the sportswear brand",
                                scrape_error="prior scrape note"),
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
            # scraped as success with no content: not this cohort's
            "null_content": self._article(session, "null-content", "pending", content=None),
            "eval_case": self._article(session, "eval-case", "skipped", SKIP_NO_ESG_CONTENT),
        }
        written = self._article(session, "human-written", "labeled")
        self._label(session, written, labeled_by="human")
        left["human_written"] = written
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
            cohort = select_no_brand_content(
                session, eval_case_ids={str(left["eval_case"].id), str(left["ruled"].id)}
            )

        assert set(cohort.selected) == {str(a.id) for a in moved.values()}
        assert cohort.left_out == {
            # a ruled row in an eval case keeps its human reason
            str(left["ruled"].id): EXCLUDED_HUMAN,
            str(left["hand_move"].id): EXCLUDED_POSSIBLE_HAND_MOVE,
            str(left["reviewed"].id): EXCLUDED_HUMAN_LABELS,
            str(left["human_written"].id): EXCLUDED_HUMAN_LABELS,
            str(left["eval_case"].id): EXCLUDED_EVAL_CASE,
        }
        assert cohort.selected[str(moved["fp"].id)] == (
            "false_positive", "Nike: not the sportswear brand", "success", "prior scrape note",
        )

    @pytest.fixture
    def seeded(self, test_db, monkeypatch):
        """The seed, committed, with the seeded eval-case row as the only eval case main sees."""
        with test_db.get_session() as session:
            moved, left = self._seed(session)
        eval_ids = {str(left["eval_case"].id)}
        monkeypatch.setattr(triage, "eval_case_ids", lambda: eval_ids)
        return moved, left, eval_ids

    @staticmethod
    def _state(session, article):
        a = session.get(models.Article, article.id)
        return (a.labeling_status, a.labeling_error, a.scrape_status, a.scrape_error, a.full_content)

    def test_apply_moves_to_the_gated_scrape_state(self, test_db, seeded, tmp_path):
        from src.data_collection.database import scrape_failure_reason

        moved, left, _ = seeded
        manifest = tmp_path / "manifest.jsonl"
        with test_db.get_session() as session:
            before = {k: self._state(session, a) for k, a in left.items()}
            # a scrape the gate skipped, for comparison
            gated_row = self._article(session, "gated", "pending", content=None, scrape_status="pending")
            test_db.update_article_content(
                session, gated_row.article_id, NO_BRAND, "skipped", SCRAPE_NO_BRAND_ERROR
            )
            gated = SimpleNamespace(id=gated_row.id)
        assert triage.main(["--manifest", str(manifest)], database=test_db) == 0

        with test_db.get_session() as session:
            gated_state = self._state(session, gated)
            assert gated_state == (
                "unlabelable", scrape_failure_reason("skipped", SCRAPE_NO_BRAND_ERROR),
                "skipped", SCRAPE_NO_BRAND_ERROR, NO_BRAND,
            )
            for article in moved.values():
                assert self._state(session, article) == gated_state
            assert {k: self._state(session, a) for k, a in left.items()} == before
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
            assert by_id[str(moved["fp"].id)]["scrape_error"] == "prior scrape note"

            # a second run finds nothing left to move
            assert select_no_brand_content(session, triage.eval_case_ids()).selected == {}

    def test_manifest_is_complete_before_commit(self, test_db, seeded, tmp_path):
        moved, _, eval_ids = seeded
        manifest = tmp_path / "manifest.jsonl"
        with test_db.get_session() as session:
            counts = triage.apply_triage(session, select_no_brand_content(session, eval_ids), manifest)
            assert counts["moved"] == len(moved)
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            assert {r["article_id"] for r in records} == {str(a.id) for a in moved.values()}
            session.rollback()
            # nothing committed: every row is as seeded
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"

    def test_row_changed_since_selection_is_not_moved(self, test_db, seeded, tmp_path):
        moved, _, eval_ids = seeded
        manifest = tmp_path / "m.jsonl"
        with test_db.get_session() as session:
            cohort = select_no_brand_content(session, eval_ids)
            # after selection: the labeling cron labels the pending row; a human marks the
            # skipped row by hand at the same status; a human labels the false positive
            pending = session.get(models.Article, moved["pending"].id)
            pending.labeling_status = "labeled"
            session.add(models.BrandLabel(article_id=pending.id, brand="Nike", labeled_by="claude-sonnet"))
            session.get(models.Article, moved["skipped"].id).labeling_error = MANUAL_SKIP_REASON
            session.add(models.BrandLabel(article_id=moved["fp"].id, brand="Nike", labeled_by="human"))
            session.flush()
            counts = triage.apply_triage(session, cohort, manifest)

        assert counts == {"moved": 1, "changed": 3, "labels_deleted": 1}
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert [r["article_id"] for r in records] == [str(moved["labeled"].id)]
        with test_db.get_session() as session:
            assert session.get(models.Article, moved["pending"].id).labeling_status == "labeled"
            assert session.query(models.BrandLabel).filter_by(article_id=moved["pending"].id).count() == 1
            skipped = session.get(models.Article, moved["skipped"].id)
            assert (skipped.labeling_status, skipped.labeling_error) == ("skipped", MANUAL_SKIP_REASON)
            fp = session.get(models.Article, moved["fp"].id)
            assert (fp.labeling_status, fp.scrape_status) == ("false_positive", "success")
            assert session.query(models.BrandLabel).filter_by(article_id=moved["fp"].id).count() == 1

    def test_dry_run_writes_nothing_and_prints_the_cohort(self, test_db, seeded, tmp_path, capsys):
        moved, left, _ = seeded
        manifest = tmp_path / "manifest.jsonl"
        assert triage.main(["--dry-run", "--manifest", str(manifest)], database=test_db) == 0

        assert not manifest.exists()
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert f"To move: {len(moved)}" in lines
        for status in ("false_positive", "labeled", "pending", "skipped"):
            assert f"  {status}: 1" in lines
        assert "Left out: 5" in lines
        assert f"  {EXCLUDED_HUMAN}: 1" in lines
        assert f"  {EXCLUDED_HUMAN_LABELS}: 2" in lines
        assert f"  {EXCLUDED_POSSIBLE_HAND_MOVE}: 1" in lines
        assert f"  {EXCLUDED_EVAL_CASE}: 1" in lines
        assert f"  {left['hand_move'].id}" in lines
        assert f"  {left['eval_case'].id}" in lines
        with test_db.get_session() as session:
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"
            assert session.query(models.BrandLabel).filter_by(article_id=moved["labeled"].id).count() == 1

    def test_existing_manifest_is_refused_before_any_write(self, test_db, seeded, tmp_path):
        moved, _, _ = seeded
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"earlier": "run"}\n')
        assert triage.main(["--manifest", str(manifest)], database=test_db) == 2

        assert manifest.read_text() == '{"earlier": "run"}\n'
        with test_db.get_session() as session:
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"

    def test_failed_commit_says_the_manifest_did_not_commit(self, test_db, seeded, tmp_path,
                                                             monkeypatch, capsys):
        from contextlib import contextmanager

        moved, _, _ = seeded
        manifest = tmp_path / "manifest.jsonl"

        @contextmanager
        def failing_commit():
            session = test_db.SessionLocal()
            try:
                yield session
                raise RuntimeError("commit failed")
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        monkeypatch.setattr(test_db, "get_session", failing_commit)
        with pytest.raises(RuntimeError, match="commit failed"):
            triage.main(["--manifest", str(manifest)], database=test_db)

        assert f"Manifest {manifest} describes a transaction that did not commit." in capsys.readouterr().out
        monkeypatch.undo()
        with test_db.get_session() as session:
            assert session.get(models.Article, moved["fp"].id).labeling_status == "false_positive"
