"""Tests for FP label provenance and the FP training export's exclusions (#154 PR1, D030)."""

import json
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.export_training_data import (
    FPCandidate,
    build_fp_dataset,
    export_false_positive_data,
    parse_args,
    write_and_report,
)
from src.data_collection import models, provenance
from src.data_collection.provenance import (
    FP_CLASSIFIER,
    HUMAN,
    HUMAN_RULINGS,
    LLM,
    PRE_CLASSIFIER,
    RULE,
    UNATTRIBUTED,
    UNJUDGED,
    UnknownFPActionError,
    derive_label_source,
    human_verified_ids,
)

AUTO = "Auto-detected: Puma matched pattern 'puma biotechnology'"
CORRECTED_FP_ID = "aabc7f6e-8fc5-4c02-9880-b9e404bd840a"  # #182: human-ruled false_positive, latest FP action skipped_llm


class TestStoredLiterals:
    """Stored rows match only by text, so the constants must keep their exact wording."""

    def test_skip_reasons(self):
        assert models.SKIP_INSUFFICIENT_CONTENT == "Insufficient content"
        assert models.SKIP_NO_BRANDS == "No brands mentioned"
        assert models.SKIP_NO_ESG_CONTENT == "No ESG content found"
        assert models.SKIP_NO_SPORTSWEAR_BRAND == "No sportswear brand ESG content"
        assert models.SKIP_NO_ESG_CATEGORIES == "No ESG categories apply to sportswear brands"

    def test_unjudged_and_judged_sets(self):
        assert models.UNJUDGED_SKIP_REASONS == {
            "Insufficient content",
            "No brands mentioned",
            "No ESG content found",
            "No sportswear brand ESG content",
        }
        assert models.JUDGED_SKIP_REASONS == {"No ESG categories apply to sportswear brands"}

    def test_fp_actions(self):
        assert models.FP_ACTIONS == {
            "skipped_llm",
            "continued_to_llm",
            "failed",
            "fp_classifier_disabled",
        }

    def test_cleanup_script_reasons(self):
        assert models.FP_REASON_MANUAL == "Manually marked as false positive"
        assert models.FP_REASON_AUTO_DETECTED_PREFIX == "Auto-detected:"


class TestDeriveLabelSource:
    @pytest.mark.parametrize(
        "status, reason, action, expected",
        [
            # false_positive (negatives)
            ("false_positive", "High-confidence false positive: p=0.2", "skipped_llm", FP_CLASSIFIER),
            ("false_positive", "Puma: not sportswear", "continued_to_llm", LLM),
            ("false_positive", "Puma: not sportswear", "failed", LLM),
            ("false_positive", "Puma: not sportswear", "fp_classifier_disabled", LLM),
            ("false_positive", "Puma: not sportswear", None, PRE_CLASSIFIER),
            ("false_positive", "Manually marked as false positive", None, HUMAN),
            ("false_positive", "Manually marked as false positive", "skipped_llm", HUMAN),
            ("false_positive", AUTO, None, RULE),
            ("false_positive", AUTO, "skipped_llm", RULE),
            # skipped (positives): the reason decides, whatever the latest FP action
            ("skipped", "No ESG content found", "continued_to_llm", UNJUDGED),
            ("skipped", "No ESG content found", "fp_classifier_disabled", UNJUDGED),
            ("skipped", "Insufficient content", None, UNJUDGED),
            ("skipped", "No brands mentioned", None, UNJUDGED),
            ("skipped", "No sportswear brand ESG content", "continued_to_llm", UNJUDGED),
            ("skipped", "No ESG categories apply to sportswear brands", "continued_to_llm", LLM),
            ("skipped", "No ESG categories apply to sportswear brands", None, PRE_CLASSIFIER),
            ("skipped", None, "continued_to_llm", UNATTRIBUTED),
            ("skipped", "Product article (sale/review/release) - not ESG news content", None, UNATTRIBUTED),
            # a human-written reason that only starts with the pipeline's text is not unjudged
            ("skipped", "No ESG content found - article contains only financial data", None, UNATTRIBUTED),
            # a human moved a classifier skip to skipped; no human inference from state (D030)
            ("skipped", "High-confidence false positive: p=0.437", "skipped_llm", UNATTRIBUTED),
            ("skipped", "No ESG content found", "skipped_llm", UNJUDGED),
            # labeled (positives)
            ("labeled", None, "continued_to_llm", LLM),
            ("labeled", None, "fp_classifier_disabled", LLM),
            ("labeled", None, None, PRE_CLASSIFIER),
        ],
    )
    def test_table(self, status, reason, action, expected):
        assert derive_label_source(status, reason, action) == expected

    @pytest.mark.parametrize("status", ["false_positive", "skipped", "labeled"])
    def test_human_verified_wins(self, status):
        assert derive_label_source(status, "No ESG content found", "skipped_llm", True) == HUMAN

    def test_unknown_action_raises_with_value_and_article(self):
        with pytest.raises(UnknownFPActionError) as excinfo:
            derive_label_source("false_positive", None, "rerouted", article_id="abc-123")
        assert "'rerouted'" in str(excinfo.value)
        assert "abc-123" in str(excinfo.value)

    def test_unknown_action_raises_even_when_human_verified(self):
        with pytest.raises(UnknownFPActionError):
            derive_label_source("labeled", None, "rerouted", human_verified=True)

    def test_unexported_status_raises(self):
        with pytest.raises(ValueError, match="pending"):
            derive_label_source("pending", None, None)

    def test_classifier_only_negative(self):
        assert provenance.is_classifier_only_negative(FP_CLASSIFIER)
        assert not provenance.is_classifier_only_negative(LLM)


class TestHumanVerifiedIds:
    def test_ruling_counts_while_status_matches(self):
        assert human_verified_ids([(CORRECTED_FP_ID, "false_positive")]) == {CORRECTED_FP_ID}

    def test_ruling_lapses_after_status_changes(self):
        assert human_verified_ids([(CORRECTED_FP_ID, "skipped")]) == set()

    def test_unknown_id(self):
        assert human_verified_ids([(str(uuid.uuid4()), "false_positive")]) == set()

    def test_rulings_are_lowercase_uuids_with_exported_statuses(self):
        for article_id, status in HUMAN_RULINGS.items():
            assert str(uuid.UUID(article_id)) == article_id
            assert status in ("false_positive", "skipped", "labeled")


def cand(status, reason=None, action=None, at=None, article_id=None):
    return FPCandidate(
        article_id=article_id or str(uuid.uuid4()),
        title="t",
        content="c",
        brands=["Puma"],
        source_name="s",
        category=[],
        labeling_status=status,
        labeling_error=reason,
        fp_action=action,
        fp_action_at=at,
    )


JAN = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
MAR = datetime(2026, 3, 2, 8, tzinfo=timezone.utc)


def mixed_candidates():
    return [
        cand("labeled", action="continued_to_llm", article_id="pos-llm"),
        cand("skipped", "No ESG categories apply to sportswear brands", "continued_to_llm", article_id="pos-judged"),
        cand("skipped", "No ESG content found", "continued_to_llm", article_id="pos-unjudged"),
        cand("skipped", None, None, article_id="pos-unattributed"),
        cand("false_positive", "not sportswear", "continued_to_llm", article_id="neg-llm"),
        cand("false_positive", "High-confidence false positive", "skipped_llm", JAN, article_id="neg-fp-jan"),
        cand("false_positive", "High-confidence false positive", "skipped_llm", MAR, article_id="neg-fp-mar"),
        cand("false_positive", AUTO, None, article_id="neg-rule"),
        cand("false_positive", "High-confidence false positive", "skipped_llm", JAN, article_id=CORRECTED_FP_ID),
    ]


def ids(records):
    return {r["article_id"] for r in records}


class TestBuildFpDataset:
    def test_default_exclusions_and_counts(self):
        records, report = build_fp_dataset(mixed_candidates())

        assert ids(records) == {"pos-llm", "pos-judged", "pos-unattributed", "neg-llm", CORRECTED_FP_ID}
        assert report.excluded_classifier_only == 2
        assert report.excluded_classifier_only_by_month == {"2026-01": 1, "2026-03": 1}
        assert report.excluded_unjudged == 1
        assert report.excluded_unjudged_by_reason == {"No ESG content found": 1}
        assert report.excluded_rule == 1
        assert report.positives_by_source == {LLM: 2, UNATTRIBUTED: 1}
        assert report.negatives_by_source == {LLM: 1, HUMAN: 1}

    def test_every_record_carries_label_source(self):
        records, _ = build_fp_dataset(mixed_candidates(), True, True, True)
        assert len(records) == len(mixed_candidates())
        assert all(r["label_source"] in provenance.LABEL_SOURCES for r in records)

    def test_human_corrected_classifier_skip_is_kept_as_human(self):
        records, _ = build_fp_dataset(mixed_candidates())
        corrected = next(r for r in records if r["article_id"] == CORRECTED_FP_ID)
        assert corrected["label_source"] == HUMAN
        assert corrected["is_sportswear"] == 0

    def test_include_classifier_labels_restores_only_that_cohort(self):
        records, report = build_fp_dataset(mixed_candidates(), include_classifier_labels=True)
        assert {"neg-fp-jan", "neg-fp-mar"} <= ids(records)
        assert "pos-unjudged" not in ids(records) and "neg-rule" not in ids(records)
        assert report.excluded_classifier_only == 0
        assert report.negatives_by_source[FP_CLASSIFIER] == 2

    def test_include_unjudged_skips_restores_only_that_cohort(self):
        records, report = build_fp_dataset(mixed_candidates(), include_unjudged_skips=True)
        assert "pos-unjudged" in ids(records)
        assert "neg-fp-jan" not in ids(records)
        assert report.positives_by_source[UNJUDGED] == 1

    def test_include_rule_labels_restores_only_that_cohort(self):
        records, _ = build_fp_dataset(mixed_candidates(), include_rule_labels=True)
        assert "neg-rule" in ids(records)
        assert "pos-unjudged" not in ids(records)

    def test_include_all_reproduces_the_old_set(self):
        records, _ = build_fp_dataset(mixed_candidates(), True, True, True)
        assert ids(records) == ids([{"article_id": c.article_id} for c in mixed_candidates()])
        assert {r["article_id"]: r["is_sportswear"] for r in records} == {
            c.article_id: 0 if c.labeling_status == "false_positive" else 1
            for c in mixed_candidates()
        }

    def test_attested_tier_keeps_only_older_classifier_skips(self):
        records, report = build_fp_dataset(mixed_candidates(), attested_before=date(2026, 2, 1))
        kept = {r["article_id"]: r for r in records}
        assert "neg-fp-jan" in kept and "neg-fp-mar" not in kept
        assert kept["neg-fp-jan"]["label_source"] == FP_CLASSIFIER
        assert kept["neg-fp-jan"]["attested_cutoff"] == "2026-02-01"
        assert "attested_cutoff" not in kept["neg-llm"]
        assert report.attested_included == 1
        assert report.excluded_classifier_only == 1
        # the tier never touches unjudged skips
        assert "pos-unjudged" not in kept

    def test_attested_cutoff_is_exclusive_and_utc(self):
        just_before = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        at_cutoff = datetime(2026, 2, 1, tzinfo=timezone.utc)
        # 2026-02-01 00:30 in UTC+1 is 2026-01-31 23:30 UTC
        local = datetime(2026, 2, 1, 0, 30, tzinfo=timezone(timedelta(hours=1)))
        cands = [
            cand("false_positive", "x", "skipped_llm", just_before, article_id="a"),
            cand("false_positive", "x", "skipped_llm", at_cutoff, article_id="b"),
            cand("false_positive", "x", "skipped_llm", local, article_id="c"),
            cand("false_positive", "x", "skipped_llm", None, article_id="d"),
        ]
        records, report = build_fp_dataset(cands, attested_before=date(2026, 2, 1))
        assert ids(records) == {"a", "c"}
        assert report.excluded_classifier_only_by_month == {"2026-02": 1, "unknown": 1}

    def test_unknown_action_fails_the_export(self):
        with pytest.raises(UnknownFPActionError):
            build_fp_dataset([cand("labeled", action="rerouted")])


class TestStdoutContract:
    """model_training.export_training_data reads the record count from the export's stdout."""

    def test_workflow_reads_the_written_record_count(self, tmp_path, capsys, monkeypatch):
        records, report = build_fp_dataset(mixed_candidates(), attested_before=date(2026, 2, 1))
        write_and_report(records, "fp", tmp_path / "fp_out.jsonl", report)
        stdout = capsys.readouterr().out
        # the by-month line (a number that is not the count) is printed before the closing line
        assert "in 2026-03: 1" in stdout

        from src.agent.workflows import model_training

        # run from an empty directory so the step finds no data/ files to copy
        monkeypatch.chdir(tmp_path)
        result = SimpleNamespace(success=True, exit_code=0, duration_seconds=0.0, stdout=stdout, stderr="")
        with patch.object(model_training, "run_export_training_data", return_value=result):
            out = model_training.export_training_data(None, {"classifiers": ["fp"]})
        assert out["datasets"]["fp"]["record_count"] == len(records)

    def test_written_file_is_what_the_splits_read(self, tmp_path, capsys):
        records, report = build_fp_dataset(mixed_candidates())
        path = tmp_path / "fp_out.jsonl"
        write_and_report(records, "fp", path, report)
        written = [json.loads(line) for line in path.read_text().splitlines()]
        assert ids(written) == ids(records)
        assert not {"neg-fp-jan", "neg-fp-mar", "pos-unjudged", "neg-rule"} & ids(written)


class TestArgs:
    def test_defaults(self):
        args = parse_args(["--dataset", "fp"])
        assert not args.include_classifier_labels
        assert not args.include_unjudged_skips
        assert not args.include_rule_labels
        assert args.attested_before is None

    def test_attested_before_parses_to_date(self):
        assert parse_args(["--dataset", "fp", "--attested-before", "2026-01-25"]).attested_before == date(2026, 1, 25)

    def test_attested_before_rejects_bad_date(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "fp", "--attested-before", "Jan 25"])

    def test_attested_before_conflicts_with_include_classifier_labels(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "fp", "--attested-before", "2026-01-25", "--include-classifier-labels"])

    @pytest.mark.parametrize(
        "flag", ["--include-classifier-labels", "--include-unjudged-skips", "--include-rule-labels"]
    )
    def test_fp_flags_rejected_for_other_datasets(self, flag):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "esg-prefilter", flag])


requires_postgres = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 to run.",
)


@requires_postgres
class TestExportQuery:
    """The real query: latest FP prediction per article, joined to its status."""

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
            session.query(models.ClassifierPrediction).delete()
            session.query(models.Article).delete()

    def _article(self, session, key, status, reason=None):
        article = models.Article(
            article_id=f"prov-{key}",
            title=f"provenance test {key}",
            url=f"https://example.com/{key}",
            full_content="content",
            brands_mentioned=["Puma"],
            scrape_status="success",
            labeling_status=status,
            labeling_error=reason,
        )
        session.add(article)
        session.flush()
        return article

    def _prediction(self, session, article, action, at):
        session.add(models.ClassifierPrediction(
            article_id=article.id,
            classifier_type="fp",
            probability=0.5,
            prediction=False,
            threshold_used=0.5,
            action_taken=action,
            created_at=at,
        ))

    def test_latest_prediction_decides(self, test_db):
        with test_db.get_session() as session:
            relabeled = self._article(session, "relabeled", "false_positive", "not sportswear")
            self._prediction(session, relabeled, "skipped_llm", JAN)
            self._prediction(session, relabeled, "continued_to_llm", MAR)
            skipped_only = self._article(session, "skipped-only", "false_positive", "fp skip")
            self._prediction(session, skipped_only, "continued_to_llm", JAN)
            self._prediction(session, skipped_only, "skipped_llm", MAR)
            # another classifier's prediction must not count
            ep = models.ClassifierPrediction(
                article_id=relabeled.id, classifier_type="ep", probability=0.1, prediction=False,
                threshold_used=0.5, action_taken="skipped_llm", created_at=MAR + timedelta(days=1),
            )
            session.add(ep)
            self._article(session, "no-row", "false_positive", "not sportswear")
            self._article(session, "unjudged", "skipped", "No ESG content found")
            session.commit()

            records, report = export_false_positive_data(session)

        by_id = {r["title"]: r for r in records}
        assert by_id["provenance test relabeled"]["label_source"] == LLM
        assert by_id["provenance test no-row"]["label_source"] == PRE_CLASSIFIER
        assert "provenance test skipped-only" not in by_id
        assert "provenance test unjudged" not in by_id
        assert report.excluded_classifier_only == 1
        assert report.excluded_classifier_only_by_month == {"2026-03": 1}
        assert report.excluded_unjudged == 1
