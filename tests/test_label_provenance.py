"""Tests for FP label provenance and the FP training export's exclusions (#154 PR1, D030)."""

import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from scripts import export_training_data as export_script
from scripts.export_training_data import (
    FPCandidate,
    build_fp_dataset,
    export_false_positive_data,
    fp_export_query,
    parse_args,
    print_fp_report,
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
FP_SKIP = "High-confidence false positive: probability 0.234 < threshold 0.53"
# #182: in HUMAN_RULINGS as false_positive; the fixtures below give it a classifier skip
CORRECTED_FP_ID = "aabc7f6e-8fc5-4c02-9880-b9e404bd840a"
JUDGED = models.SKIP_NO_ESG_CATEGORIES
FM = provenance.EXCLUDED_STRATA[0]
STRATUM = FM.name
# #289: an FM row the owner ruled skipped, so it is in HUMAN_RULINGS at the stratum's status
RULED_FM_ID = "1fe66929-a227-4ce3-a7fb-78df8661c7c9"


class TestStoredLiterals:
    """Stored rows match only by text, so the constants must keep their exact wording."""

    def test_skip_reasons(self):
        assert models.SKIP_INSUFFICIENT_CONTENT == "Insufficient content"
        assert models.SKIP_NO_BRANDS == "No brands mentioned"
        assert models.SKIP_NO_ESG_CONTENT == "No ESG content found"
        assert models.SKIP_NO_SPORTSWEAR_BRAND == "No sportswear brand ESG content"
        assert models.SKIP_NO_ESG_CATEGORIES == "No ESG categories apply to sportswear brands"
        assert models.SKIP_INCOMPLETE_JUDGMENT == "Some brands not judged; the rest not sportswear"

    def test_unjudged_and_judged_sets(self):
        assert models.UNJUDGED_SKIP_REASONS == {
            "Insufficient content",
            "No brands mentioned",
            "No ESG content found",
            "No sportswear brand ESG content",
            "Some brands not judged; the rest not sportswear",
        }
        assert models.JUDGED_SKIP_REASONS == {"No ESG categories apply to sportswear brands"}

    def test_every_skip_constant_is_classified_exactly_once(self):
        """A new SKIP_* constant left out of both sets would read as an unattributed positive."""
        constants = {v for k, v in vars(models).items() if k.startswith("SKIP_")}
        assert constants == models.UNJUDGED_SKIP_REASONS | models.JUDGED_SKIP_REASONS
        assert not models.UNJUDGED_SKIP_REASONS & models.JUDGED_SKIP_REASONS

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

    def test_manual_skip_reason(self):
        """fix_label.py --manual's human marker for 'skipped' (#234); stored rows match its text."""
        assert models.MANUAL_SKIP_REASON == "Manually ruled: sportswear, no ESG category"
        assert models.MANUAL_SKIP_REASON not in models.UNJUDGED_SKIP_REASONS | models.JUDGED_SKIP_REASONS


class TestDeriveLabelSource:
    @pytest.mark.parametrize(
        "status, reason, action, skip_reason, expected",
        [
            # false_positive (negatives)
            ("false_positive", FP_SKIP, "skipped_llm", FP_SKIP, FP_CLASSIFIER),
            # relabeled by the LLM after the skip, no new prediction recorded: not classifier-only
            ("false_positive", "Puma: refers to the animal", "skipped_llm", FP_SKIP, LLM),
            ("false_positive", "Puma: not sportswear", "continued_to_llm", None, LLM),
            ("false_positive", "Puma: not sportswear", "failed", None, LLM),
            ("false_positive", "Puma: not sportswear", "fp_classifier_disabled", None, LLM),
            ("false_positive", "Puma: not sportswear", None, None, PRE_CLASSIFIER),
            ("false_positive", "Manually marked as false positive", None, None, HUMAN),
            ("false_positive", "Manually marked as false positive", "skipped_llm", FP_SKIP, HUMAN),
            ("false_positive", AUTO, None, None, RULE),
            ("false_positive", AUTO, "skipped_llm", FP_SKIP, RULE),
            # skipped (positives): the reason decides, whatever the latest FP action
            ("skipped", "No ESG content found", "continued_to_llm", None, UNJUDGED),
            ("skipped", "No ESG content found", "fp_classifier_disabled", None, UNJUDGED),
            ("skipped", "Insufficient content", None, None, UNJUDGED),
            ("skipped", "No brands mentioned", None, None, UNJUDGED),
            ("skipped", "No sportswear brand ESG content", "continued_to_llm", None, UNJUDGED),
            ("skipped", "Some brands not judged; the rest not sportswear", "continued_to_llm", None, UNJUDGED),
            ("skipped", "No ESG categories apply to sportswear brands", "continued_to_llm", None, LLM),
            ("skipped", "No ESG categories apply to sportswear brands", None, None, PRE_CLASSIFIER),
            ("skipped", None, "continued_to_llm", None, UNATTRIBUTED),
            ("skipped", "Product article (sale/review/release) - not ESG news content", None, None, UNATTRIBUTED),
            # a human-written reason that only starts with the pipeline's text is not unjudged
            ("skipped", "No ESG content found - article contains only financial data", None, None, UNATTRIBUTED),
            # a human moved a classifier skip to skipped; no human inference from state (D030)
            ("skipped", FP_SKIP, "skipped_llm", FP_SKIP, UNATTRIBUTED),
            ("skipped", "No ESG content found", "skipped_llm", FP_SKIP, UNJUDGED),
            # labeled (positives)
            ("labeled", None, "continued_to_llm", None, LLM),
            ("labeled", None, "fp_classifier_disabled", None, LLM),
            ("labeled", None, None, None, PRE_CLASSIFIER),
        ],
    )
    def test_table(self, status, reason, action, skip_reason, expected):
        assert derive_label_source(status, reason, action, skip_reason) == expected

    @pytest.mark.parametrize(
        "status, reason, action, skip_reason",
        [
            ("false_positive", FP_SKIP, "skipped_llm", FP_SKIP),  # would be fp_classifier
            ("false_positive", AUTO, None, None),  # would be rule
            ("skipped", "No ESG content found", "continued_to_llm", None),  # would be unjudged
            ("skipped", None, None, None),  # would be unattributed
            ("labeled", None, "continued_to_llm", None),  # would be llm
        ],
    )
    def test_human_verified_wins(self, status, reason, action, skip_reason):
        assert derive_label_source(status, reason, action, skip_reason, human_verified=True) == HUMAN

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

    def test_rulings_literal_has_no_duplicate_keys(self):
        """A dict literal keeps the last value for a repeated key, so a pasted id that repeats would
        override an earlier ruling, and the loaded dict could not show it. Read the source."""
        import ast

        tree = ast.parse(open(provenance.__file__, encoding="utf-8").read())
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "HUMAN_RULINGS"
        ]
        assert len(literals) == 1 and isinstance(literals[0], ast.Dict)
        keys = [key.value for key in literals[0].keys]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        assert repeated == []
        assert len(keys) == len(HUMAN_RULINGS)


def cand(status, reason=None, action=None, at=None, article_id=None, skip_reason="same"):
    """A candidate. For a classifier skip, ``skip_reason`` defaults to the stored reason, as the
    pipeline writes it; pass another value to model a relabel after the skip."""
    if skip_reason == "same":
        skip_reason = reason if action == "skipped_llm" else None
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
        fp_skip_reason=skip_reason,
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
        cand("false_positive", FP_SKIP, "skipped_llm", JAN, article_id="neg-fp-jan"),
        cand("false_positive", FP_SKIP, "skipped_llm", MAR, article_id="neg-fp-mar"),
        cand("false_positive", "Puma: the animal", "skipped_llm", JAN, article_id="neg-relabeled", skip_reason=FP_SKIP),
        cand("false_positive", AUTO, None, article_id="neg-rule"),
        cand("false_positive", FP_SKIP, "skipped_llm", JAN, article_id=CORRECTED_FP_ID),
    ]


DEFAULT_IDS = {"pos-llm", "pos-judged", "pos-unattributed", "neg-llm", "neg-relabeled", CORRECTED_FP_ID}
COHORTS = {
    "include_classifier_labels": ({"neg-fp-jan", "neg-fp-mar"}, "excluded_classifier_only"),
    "include_unjudged_skips": ({"pos-unjudged"}, "excluded_unjudged"),
    "include_rule_labels": ({"neg-rule"}, "excluded_rule"),
}
DEFAULT_COUNTS = {"excluded_classifier_only": 2, "excluded_unjudged": 1, "excluded_rule": 1}


@pytest.fixture
def host_tz():
    """Set the host time zone for one test, so a host-local conversion shows on any machine."""
    saved = os.environ.get("TZ")

    def set_zone(zone):
        os.environ["TZ"] = zone
        time.tzset()
        # an unknown zone falls back to UTC silently, which would make these tests host-blind
        assert time.timezone != 0, f"{zone} not loaded; is tzdata installed?"

    yield set_zone
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


def query_row(**values):
    """A row in the order fp_export_query selects its columns, so a reordered query reaches the mapping."""
    return tuple(values[c["name"]] for c in fp_export_query(Session()).column_descriptions)


def ids(records):
    return {r["article_id"] for r in records}


class TestBuildFpDataset:
    def test_default_exclusions_and_counts(self):
        records, report = build_fp_dataset(mixed_candidates())

        assert ids(records) == DEFAULT_IDS
        for counter, expected in DEFAULT_COUNTS.items():
            assert getattr(report, counter) == expected
        assert report.excluded_classifier_only_by_month == {"2026-01": 1, "2026-03": 1}
        assert report.excluded_unjudged_by_reason == {"No ESG content found": 1}
        assert report.positives_by_source == {LLM: 2, UNATTRIBUTED: 1}
        assert report.negatives_by_source == {LLM: 2, HUMAN: 1}

    def test_default_excluded_sources_are_exactly_the_policy_set(self):
        """Every source in EXCLUDED_BY_DEFAULT is left out by default, and no other source is."""
        records, _ = build_fp_dataset(mixed_candidates(), True, True, True)
        all_sources = {r["article_id"]: r["label_source"] for r in records}
        kept, _ = build_fp_dataset(mixed_candidates())
        dropped = {all_sources[i] for i in set(all_sources) - ids(kept)}
        assert dropped == provenance.EXCLUDED_BY_DEFAULT

    def test_every_record_carries_label_source_and_fp_reason_only_on_negatives(self):
        records, _ = build_fp_dataset(mixed_candidates(), True, True, True)
        assert len(records) == len(mixed_candidates())
        for r in records:
            assert r["label_source"] in provenance.LABEL_SOURCES
            assert ("fp_reason" in r) == (r["is_sportswear"] == 0)

    def test_human_corrected_classifier_skip_is_kept_as_human(self):
        records, _ = build_fp_dataset(mixed_candidates())
        corrected = next(r for r in records if r["article_id"] == CORRECTED_FP_ID)
        assert corrected["label_source"] == HUMAN
        assert corrected["is_sportswear"] == 0

    def test_ruling_lapses_through_the_build(self):
        """A ruled article whose status has since changed is not human, and is excluded on its own terms."""
        records, report = build_fp_dataset(
            [cand("skipped", "No ESG content found", "continued_to_llm", article_id=CORRECTED_FP_ID)]
        )
        assert records == []
        assert report.excluded_unjudged == 1

    def test_relabel_after_skip_is_kept_as_llm(self):
        records, _ = build_fp_dataset(mixed_candidates())
        relabeled = next(r for r in records if r["article_id"] == "neg-relabeled")
        assert relabeled["label_source"] == LLM

    @pytest.mark.parametrize("flag", sorted(COHORTS))
    def test_each_flag_restores_exactly_its_cohort(self, flag):
        cohort, counter = COHORTS[flag]
        records, report = build_fp_dataset(mixed_candidates(), **{flag: True})
        assert ids(records) == DEFAULT_IDS | cohort
        expected = dict(DEFAULT_COUNTS, **{counter: 0})
        assert {c: getattr(report, c) for c in DEFAULT_COUNTS} == expected

    def test_include_all_reproduces_the_old_set(self):
        records, _ = build_fp_dataset(mixed_candidates(), True, True, True)
        by_id = {r["article_id"]: r for r in records}
        for c in mixed_candidates():
            r = by_id[c.article_id]
            assert r["is_sportswear"] == (0 if c.labeling_status == "false_positive" else 1)
            assert r["source"] == c.labeling_status
            if c.labeling_status == "false_positive":
                assert r["fp_reason"] == c.labeling_error

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

    @pytest.mark.parametrize("zone, moment, expected", [
        # 2026-02-01 00:30 in UTC+1 is January in UTC, and February in Tokyo
        ("Asia/Tokyo", datetime(2026, 2, 1, 0, 30, tzinfo=timezone(timedelta(hours=1))), "2026-01"),
        # 2026-02-01 03:00 UTC is January in Los Angeles
        ("America/Los_Angeles", datetime(2026, 2, 1, 3, tzinfo=timezone.utc), "2026-02"),
    ])
    def test_month_key_is_utc(self, host_tz, zone, moment, expected):
        host_tz(zone)
        _, report = build_fp_dataset([cand("false_positive", "x", "skipped_llm", moment)])
        assert report.excluded_classifier_only_by_month == {expected: 1}

    @pytest.mark.parametrize("zone, moment, expected", [
        ("America/Los_Angeles", datetime(2026, 1, 31, 23, 59), "2026-01"),
        ("Europe/Berlin", datetime(2026, 2, 1, 0, 30), "2026-02"),
    ])
    def test_naive_month_key_is_utc(self, host_tz, zone, moment, expected):
        """Read as host-local, each of these would land in the other month."""
        host_tz(zone)
        _, report = build_fp_dataset([cand("false_positive", "x", "skipped_llm", moment)])
        assert report.excluded_classifier_only_by_month == {expected: 1}

    @pytest.mark.parametrize("zone", ["America/Los_Angeles", "Europe/Berlin"])
    def test_naive_timestamp_taken_as_utc(self, host_tz, zone):
        host_tz(zone)
        cands = [
            cand("false_positive", "x", "skipped_llm", datetime(2026, 1, 31, 23, 59), article_id="n1"),
            cand("false_positive", "x", "skipped_llm", datetime(2026, 2, 1), article_id="n2"),
        ]
        records, report = build_fp_dataset(cands, attested_before=date(2026, 2, 1))
        assert ids(records) == {"n1"}
        assert report.excluded_classifier_only_by_month == {"2026-02": 1}

    def test_unknown_action_fails_the_export(self):
        with pytest.raises(UnknownFPActionError):
            build_fp_dataset([cand("labeled", action="rerouted")])


class TestSpotCheckStrata:
    """#289: the unreviewed rows of a relabel stratum in the exclude band are left out."""

    def strata_candidates(self):
        return [
            cand("skipped", JUDGED, "continued_to_llm", article_id="fm-unreviewed"),
            cand("skipped", JUDGED, "continued_to_llm", article_id=RULED_FM_ID),
            cand("skipped", "No ESG content found", "continued_to_llm", article_id="fm-unjudged"),
            cand("skipped", JUDGED, "continued_to_llm", article_id="not-in-stratum"),
        ]

    def strata(self):
        return {i: STRATUM for i in ("fm-unreviewed", RULED_FM_ID, "fm-unjudged")}

    def test_the_fm_stratum_is_the_289_decision(self):
        assert FM.run_tag == provenance.MODE_RELABEL_248 == "248-relabel"
        assert (FM.prior_status, FM.status, FM.record) == ("false_positive", "skipped", "#289")
        assert HUMAN_RULINGS[RULED_FM_ID] == FM.status

    def test_unreviewed_rows_are_excluded_and_counted(self):
        records, report = build_fp_dataset(self.strata_candidates(), strata=self.strata())
        assert ids(records) == {RULED_FM_ID, "not-in-stratum"}
        assert report.excluded_strata == {STRATUM: 1}
        # a row a source exclusion already left out is counted there, not twice
        assert report.excluded_unjudged == 1
        assert all("spot_check_stratum" not in r for r in records)

    def test_a_human_ruling_is_kept_whatever_its_stratum(self):
        records, _ = build_fp_dataset(self.strata_candidates(), strata=self.strata())
        ruled = next(r for r in records if r["article_id"] == RULED_FM_ID)
        assert ruled["label_source"] == HUMAN

    def test_include_flag_keeps_and_stamps_them(self):
        records, report = build_fp_dataset(
            self.strata_candidates(), strata=self.strata(), include_spot_check_strata=True
        )
        stamped = {r["article_id"]: r.get("spot_check_stratum") for r in records}
        assert stamped == {"fm-unreviewed": STRATUM, RULED_FM_ID: None, "not-in-stratum": None}
        assert report.excluded_strata == {}

    def test_an_attested_row_a_stratum_excludes_is_not_counted_as_attested(self):
        row = cand("false_positive", FP_SKIP, "skipped_llm", JAN, article_id="attested-in-stratum")
        _, report = build_fp_dataset(
            [row], attested_before=date(2026, 2, 1), strata={"attested-in-stratum": STRATUM}
        )
        assert (report.attested_included, report.excluded_strata) == (0, {STRATUM: 1})

    def test_no_strata_changes_nothing(self):
        assert build_fp_dataset(mixed_candidates(), strata={}) == build_fp_dataset(mixed_candidates())

    def test_report_prints_each_stratum(self, capsys):
        _, report = build_fp_dataset(self.strata_candidates(), strata=self.strata())
        print_fp_report(report)
        lines = capsys.readouterr().out.splitlines()
        assert "Excluded spot-checked relabel strata: 1" in lines
        assert f"  {STRATUM} (#289): 1" in lines
        assert workflow_count_lines("\n".join(lines)) == []


class TestNotSportswearSkips:
    """#154 PR2: a skipped row storing a reply's not-sportswear reason for one of its brands."""

    NIKE_NEGATIVE = "Nike: purely a financial investment analysis article"

    @pytest.mark.parametrize(
        "status, reason, brands, expected",
        [
            ("skipped", NIKE_NEGATIVE, ["Nike"], True),
            ("skipped", "nike: lowercase brand; Puma: second brand", ["Nike", "Puma"], True),
            # not a brand the article mentions (a fix note, another brand)
            ("skipped", "Fixed: Removed Anta (Santa/Santana substring match)", ["Nike"], False),
            ("skipped", "Puma: the animal", ["Nike"], False),
            # pipeline and human reasons
            ("skipped", JUDGED, ["Nike"], False),
            ("skipped", models.MANUAL_SKIP_REASON, ["Nike"], False),
            ("skipped", None, ["Nike"], False),
            ("skipped", "", ["Nike"], False),
            # a negative's reply reason is where it belongs
            ("false_positive", NIKE_NEGATIVE, ["Nike"], False),
        ],
    )
    def test_predicate(self, status, reason, brands, expected):
        assert provenance.not_sportswear_reason_on_skip(status, reason, brands) is expected

    def candidates(self):
        nike = lambda **kw: FPCandidate(**{**cand("skipped").__dict__, "brands": ["Nike"], **kw})
        return [
            nike(article_id="contradicted", labeling_error=self.NIKE_NEGATIVE),
            nike(article_id="fix-note", labeling_error="Fixed: Removed Anta"),
            nike(article_id="no-reason", labeling_error=None),
            # ruled skipped by the owner (#289), so human, and kept
            nike(article_id=RULED_FM_ID, labeling_error=self.NIKE_NEGATIVE),
        ]

    def test_excluded_by_default_and_counted(self):
        records, report = build_fp_dataset(self.candidates())
        assert ids(records) == {"fix-note", "no-reason", RULED_FM_ID}
        assert report.excluded_not_sportswear_skips == 1
        sources = {r["article_id"]: r["label_source"] for r in records}
        assert sources == {"fix-note": UNATTRIBUTED, "no-reason": UNATTRIBUTED, RULED_FM_ID: HUMAN}

    def test_include_flag_keeps_them_as_unattributed(self):
        records, report = build_fp_dataset(self.candidates(), include_not_sportswear_skips=True)
        assert ids(records) == {"contradicted", "fix-note", "no-reason", RULED_FM_ID}
        assert report.excluded_not_sportswear_skips == 0

    def test_report_line(self, capsys):
        _, report = build_fp_dataset(self.candidates())
        print_fp_report(report)
        out = capsys.readouterr().out
        assert "Excluded skips with a not-sportswear reason: 1" in out.splitlines()
        assert workflow_count_lines(out) == []


class TestStratumMembers:
    PASS = {"run-a", "run-b"}

    def members(self, prior_status, status, reason=JUDGED, runs=frozenset({"run-a"})):
        prior = {"a": provenance.PriorRecord(prior_status, "old reason", "arm")}
        current = {} if status is None else {"a": provenance.StratumRow(status, reason, frozenset(runs))}
        return provenance.stratum_members(FM, prior, current, self.PASS)

    def test_member_holds_the_pass_verdict(self):
        assert self.members("false_positive", "skipped") == {"a"}

    @pytest.mark.parametrize(
        "prior_status, status, reason, runs",
        [
            # the pass recorded another prior status
            ("skipped", "skipped", JUDGED, {"run-a"}),
            # its status moved since (kept false_positive, a later relabel)
            ("false_positive", "false_positive", JUDGED, {"run-a"}),
            ("false_positive", "labeled", None, {"run-a"}),
            # a hand move to skipped: the reason is not one a reply writes (review of #292)
            ("false_positive", "skipped", "Nike: a running shoe sale", {"run-a"}),
            ("false_positive", "skipped", models.MANUAL_SKIP_REASON, {"run-a"}),
            # an unjudged skip: its source already excludes it
            ("false_positive", "skipped", models.SKIP_NO_ESG_CONTENT, frozenset()),
            ("false_positive", "skipped", models.SKIP_INCOMPLETE_JUDGMENT, {"run-a"}),
            # judged by a run outside the pass, or not judged at all
            ("false_positive", "skipped", JUDGED, {"run-a", "later"}),
            ("false_positive", "skipped", JUDGED, {"later"}),
            ("false_positive", "skipped", JUDGED, frozenset()),
            # gone from the database
            ("false_positive", None, JUDGED, {"run-a"}),
        ],
    )
    def test_not_a_member(self, prior_status, status, reason, runs):
        assert self.members(prior_status, status, reason, runs) == set()


class TestFirstPrior:
    def test_first_record_wins_and_tried_reasons_reads_it(self):
        configs = [
            {"run_config": {"prior": {"a": ["false_positive", "r1", "arm"]}}},
            None,
            {"run_config": {}},
            {"run_config": {"prior": {"a": ["failed", "r2", "arm"], "b": ["skipped", None, "arm"]}}},
        ]
        assert provenance.first_prior(configs) == {
            "a": ("false_positive", "r1", "arm"), "b": ("skipped", None, "arm"),
        }
        assert provenance.tried_reasons(configs) == {"a": "r1", "b": None}


def workflow_count_lines(stdout):
    """The lines model_training.export_training_data reads a record count from."""
    return [l for l in stdout.split("\n") if "records" in l.lower() or "exported" in l.lower()]


class TestReportOutput:
    def test_every_report_line_matches_the_report(self, capsys):
        _, report = build_fp_dataset(mixed_candidates(), attested_before=date(2026, 2, 1))
        print_fp_report(report)
        lines = capsys.readouterr().out.splitlines()
        assert "Excluded classifier-only negatives: 1" in lines
        assert "  skipped by the FP classifier in 2026-03: 1" in lines
        assert "Excluded unjudged skips: 1" in lines
        assert "  No ESG content found: 1" in lines
        assert "Excluded keyword-sweep negatives: 1" in lines
        assert any(l.startswith("Owner-attested tier") and l.endswith(": 1 kept") for l in lines)
        positives = lines.index("Positives by label source:")
        negatives = lines.index("Negatives by label source:")
        assert lines[positives + 1 : negatives] == ["  llm: 2", "  unattributed: 1"]
        assert lines[negatives + 1 : negatives + 4] == ["  fp_classifier: 1", "  human: 1", "  llm: 2"]

    def test_no_report_line_looks_like_a_record_count(self, capsys):
        _, report = build_fp_dataset(mixed_candidates(), attested_before=date(2026, 2, 1))
        print_fp_report(report)
        assert workflow_count_lines(capsys.readouterr().out) == []


class TestStdoutContract:
    """model_training.export_training_data reads the record count from the export's stdout."""

    def test_closing_line_is_last_and_every_count_line_carries_the_count(self, tmp_path, capsys):
        records, report = build_fp_dataset(mixed_candidates(), attested_before=date(2026, 2, 1))
        write_and_report(records, "fp", tmp_path / "fp_out.jsonl", report)
        stdout = capsys.readouterr().out

        assert stdout.rstrip().splitlines()[-1].startswith(f"Exported {len(records)} records to ")
        count_lines = workflow_count_lines(stdout)
        assert count_lines[-1].startswith("Exported ")
        for line in count_lines:
            assert int(re.findall(r"\d+", line)[0]) == len(records)

    def test_workflow_reads_the_written_record_count(self, tmp_path, capsys, monkeypatch):
        records, report = build_fp_dataset(mixed_candidates())
        write_and_report(records, "fp", tmp_path / "fp_out.jsonl", report)
        stdout = capsys.readouterr().out

        from src.agent.workflows import model_training

        # run from an empty directory so the step finds no data/ files to copy
        monkeypatch.chdir(tmp_path)
        result = SimpleNamespace(success=True, exit_code=0, duration_seconds=0.0, stdout=stdout, stderr="")
        with patch.object(model_training, "run_export_training_data", return_value=result):
            out = model_training.export_training_data(None, {"classifiers": ["fp"]})
        assert out["datasets"]["fp"]["record_count"] == len(records)

    def test_written_file_matches_records(self, tmp_path, capsys):
        records, report = build_fp_dataset(mixed_candidates())
        path = tmp_path / "fp_out.jsonl"
        write_and_report(records, "fp", path, report)
        written = [json.loads(line) for line in path.read_text().splitlines()]
        assert written == records


class TestArgs:
    def test_defaults(self):
        args = parse_args(["--dataset", "fp"])
        assert not args.include_classifier_labels
        assert not args.include_unjudged_skips
        assert not args.include_rule_labels
        assert not args.include_spot_check_strata
        assert not args.include_not_sportswear_skips
        assert args.attested_before is None

    def test_attested_before_parses_to_date(self):
        assert parse_args(["--dataset", "fp", "--attested-before", "2026-01-25"]).attested_before == date(2026, 1, 25)

    @pytest.mark.parametrize("value", ["Jan 25", ""])
    def test_attested_before_rejects_bad_date(self, value):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "fp", "--attested-before", value])

    def test_attested_before_conflicts_with_include_classifier_labels(self):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "fp", "--attested-before", "2026-01-25", "--include-classifier-labels"])

    @pytest.mark.parametrize(
        "flags",
        [
            ["--include-classifier-labels"],
            ["--include-unjudged-skips"],
            ["--include-rule-labels"],
            ["--include-spot-check-strata"],
            ["--include-not-sportswear-skips"],
            ["--attested-before", "2026-01-01"],
            ["--attested-before", ""],
        ],
    )
    def test_fp_flags_rejected_for_other_datasets(self, flags):
        with pytest.raises(SystemExit):
            parse_args(["--dataset", "esg-prefilter", *flags])


class TestMainWiring:
    """argv reaches the export function, and each dataset routes to its own output."""

    def _run_main(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["export_training_data.py", *argv])
        with patch.object(export_script, "db") as db:
            db.get_session.return_value.__enter__.return_value = "session"
            return export_script.main()

    def test_fp_flags_reach_the_export(self, monkeypatch, tmp_path, capsys):
        spy = MagicMock(return_value=build_fp_dataset(mixed_candidates()))
        monkeypatch.setattr(export_script, "export_false_positive_data", spy)
        out = tmp_path / "fp.jsonl"
        code = self._run_main(monkeypatch, [
            "--dataset", "fp", "-o", str(out), "--since", "2026-01-02",
            "--include-unjudged-skips", "--include-rule-labels", "--attested-before", "2026-02-01",
        ])
        assert code == 0
        spy.assert_called_once_with(
            "session",
            datetime(2026, 1, 2),
            include_classifier_labels=False,
            include_unjudged_skips=True,
            include_rule_labels=True,
            attested_before=date(2026, 2, 1),
            include_spot_check_strata=False,
            include_not_sportswear_skips=False,
        )
        assert "Excluded classifier-only negatives" in capsys.readouterr().out

    def test_include_classifier_labels_reaches_the_export(self, monkeypatch, tmp_path, capsys):
        spy = MagicMock(return_value=build_fp_dataset(mixed_candidates()))
        monkeypatch.setattr(export_script, "export_false_positive_data", spy)
        self._run_main(monkeypatch, ["--dataset", "fp", "-o", str(tmp_path / "fp.jsonl"), "--include-classifier-labels"])
        assert spy.call_args.kwargs["include_classifier_labels"] is True
        assert spy.call_args.kwargs["attested_before"] is None

    def test_include_spot_check_strata_reaches_the_export(self, monkeypatch, tmp_path, capsys):
        spy = MagicMock(return_value=build_fp_dataset(mixed_candidates()))
        monkeypatch.setattr(export_script, "export_false_positive_data", spy)
        self._run_main(monkeypatch, ["--dataset", "fp", "-o", str(tmp_path / "fp.jsonl"), "--include-spot-check-strata"])
        assert spy.call_args.kwargs["include_spot_check_strata"] is True

    @pytest.mark.parametrize(
        "dataset, exporter, record",
        [
            ("esg-prefilter", "export_esg_prefilter_data", {"has_esg": 1, "source": "labeled"}),
            ("esg-labels", "export_esg_labels_data", {
                "brand": "Nike", "environmental": 1, "social": 0, "governance": 0, "digital_transformation": 0,
            }),
        ],
    )
    def test_other_datasets_get_no_fp_report(self, monkeypatch, tmp_path, capsys, dataset, exporter, record):
        monkeypatch.setattr(export_script, exporter, MagicMock(return_value=[record]))
        fp_spy = MagicMock()
        monkeypatch.setattr(export_script, "export_false_positive_data", fp_spy)
        out = tmp_path / "out.jsonl"
        assert self._run_main(monkeypatch, ["--dataset", dataset, "-o", str(out)]) == 0
        stdout = capsys.readouterr().out
        fp_spy.assert_not_called()
        assert "label source" not in stdout
        assert stdout.rstrip().splitlines()[-1].startswith("Exported 1 records to ")
        assert [json.loads(l) for l in out.read_text().splitlines()] == [record]


class TestExportRowMapping:
    """export_false_positive_data turns query rows into candidates."""

    @pytest.fixture(autouse=True)
    def no_strata(self, monkeypatch):
        """The session here is a placeholder, so the strata query is stubbed."""
        stub = MagicMock(return_value={})
        monkeypatch.setattr(provenance, "excluded_stratum_ids", stub)
        return stub

    def test_rows_map_to_candidates(self):
        article_uuid = uuid.UUID(CORRECTED_FP_ID)
        article = SimpleNamespace(
            id=article_uuid, article_id="external-id-not-the-key", title="t", full_content=None,
            description="d", brands_mentioned=None, source_name="s", category=None,
            labeling_status="false_positive", labeling_error=FP_SKIP,
        )
        query = MagicMock()
        query.all.return_value = [
            query_row(Article=article, action_taken="skipped_llm", skip_reason=FP_SKIP, created_at=JAN)
        ]
        with patch.object(export_script, "fp_export_query", return_value=query) as built:
            records, report = export_false_positive_data("session", None)
        built.assert_called_once_with("session", None)
        assert records[0]["article_id"] == CORRECTED_FP_ID  # the UUID, which HUMAN_RULINGS keys on
        assert records[0]["label_source"] == HUMAN
        assert records[0]["content"] == "d"
        assert records[0]["brands"] == [] and records[0]["category"] == []

    def test_strata_are_read_from_the_session_and_applied(self, no_strata):
        article = SimpleNamespace(
            id=uuid.uuid4(), title="t", full_content="c", description=None, brands_mentioned=["Nike"],
            source_name="s", category=None, labeling_status="skipped", labeling_error=JUDGED,
        )
        no_strata.return_value = {str(article.id): STRATUM}
        query = MagicMock()
        query.all.return_value = [query_row(Article=article, action_taken=None, skip_reason=None, created_at=None)]
        with patch.object(export_script, "fp_export_query", return_value=query):
            records, report = export_false_positive_data("session", None)
            assert records == [] and report.excluded_strata == {STRATUM: 1}
            records, report = export_false_positive_data("session", None, include_spot_check_strata=True)
            assert [r["spot_check_stratum"] for r in records] == [STRATUM]
        no_strata.assert_called_with("session")

    def test_prediction_fields_reach_the_candidate(self):
        """The latest prediction's action, skip reason and time decide a classifier-only row; the
        ruled row above returns HUMAN before reading them."""
        article = SimpleNamespace(
            id=uuid.uuid4(), title="t", full_content="c", description=None, brands_mentioned=["Puma"],
            source_name="s", category=None, labeling_status="false_positive", labeling_error=FP_SKIP,
        )
        query = MagicMock()
        query.all.return_value = [
            query_row(Article=article, action_taken="skipped_llm", skip_reason=FP_SKIP, created_at=JAN)
        ]
        with patch.object(export_script, "fp_export_query", return_value=query):
            records, report = export_false_positive_data("session", None)
            assert records == []
            assert report.excluded_classifier_only == 1
            assert report.excluded_classifier_only_by_month == {"2026-01": 1}

            records, report = export_false_positive_data("session", None, attested_before=date(2026, 2, 1))
            assert [r["label_source"] for r in records] == [FP_CLASSIFIER]
            assert report.attested_included == 1


class TestCleanupScriptReasons:
    """cleanup_false_positives.py writes the reasons provenance reads as human and rule."""

    @staticmethod
    def _article():
        return SimpleNamespace(id=uuid.uuid4(), title="Puma Biotechnology reports", labeling_status="labeled",
                               labeling_error=None)

    def test_manual_mark_stores_the_manual_reason(self):
        from scripts import cleanup_false_positives as cleanup

        article = self._article()
        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = article
        assert cleanup.mark_articles_as_false_positive(session, [article.id]) == 1
        assert article.labeling_status == "false_positive"
        assert derive_label_source(article.labeling_status, article.labeling_error, None) == HUMAN

    def test_auto_detect_stores_the_rule_reason(self):
        from scripts import cleanup_false_positives as cleanup

        article = self._article()
        with patch.object(
            cleanup, "list_potential_false_positives", return_value=[(article, "Puma", "puma biotechnology")]
        ):
            assert cleanup.auto_detect_false_positives(MagicMock(), dry_run=False) == 1
        assert article.labeling_status == "false_positive"
        assert derive_label_source(article.labeling_status, article.labeling_error, None) == RULE


class TestExportQuerySql:
    """The query as PostgreSQL sees it; the RUN_DB_TESTS test below runs it for real."""

    @pytest.fixture
    def sql(self):
        statement = fp_export_query(Session(), since=datetime(2026, 1, 2)).statement
        return " ".join(str(statement.compile(dialect=postgresql.dialect())).split())

    def test_latest_prediction_ranking(self, sql):
        assert (
            "row_number() OVER (PARTITION BY classifier_predictions.article_id "
            "ORDER BY classifier_predictions.created_at DESC NULLS LAST, classifier_predictions.id DESC)"
        ) in sql
        assert re.search(r"WHERE classifier_predictions\.classifier_type = %\(classifier_type_\d+\)s", sql)
        assert re.search(r"\.rn = %\(rn_\d+\)s", sql)

    def test_outer_join_filters_and_order(self, sql):
        assert "LEFT OUTER JOIN" in sql
        assert "ON latest_prediction.article_id = articles.id" in sql
        assert re.search(r"articles\.scrape_status = %\(scrape_status_\d+\)s", sql)
        assert re.search(r"articles\.created_at >= %\(created_at_\d+\)s", sql)
        assert "articles.labeling_status IN" in sql
        assert sql.endswith("ORDER BY articles.id")

    def test_selects_skip_reason(self, sql):
        assert "latest_prediction.skip_reason" in sql

    def test_column_order(self):
        """export_false_positive_data unpacks rows by position, in this order, and the prediction's
        time is the latest prediction's, not the article's own created_at."""
        columns = fp_export_query(Session()).column_descriptions
        assert columns[0]["entity"] is models.Article
        assert [(c["name"], c["expr"].table.name) for c in columns[1:]] == [
            ("action_taken", "latest_prediction"),
            ("skip_reason", "latest_prediction"),
            ("created_at", "latest_prediction"),
        ]

    def test_bound_values(self, sql):
        """Each placeholder above carries the value the export depends on, not just the right shape."""
        compiled = fp_export_query(Session(), since=datetime(2026, 1, 2)).statement.compile(
            dialect=postgresql.dialect()
        )

        def bound(pattern):
            return compiled.params[re.search(pattern, sql).group(1)]

        assert bound(r"classifier_predictions\.classifier_type = %\((\w+)\)s") == "fp"
        assert bound(r"\.rn = %\((\w+)\)s") == 1
        assert bound(r"articles\.labeling_status IN \(__\[POSTCOMPILE_(\w+)\]\)") == [
            "labeled", "skipped", "false_positive",
        ]


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
            session.query(models.Article).delete()  # its brand_judgments rows go with it (CASCADE)

    def _article(self, session, key, status, reason=None, article_uuid=None):
        article = models.Article(
            id=article_uuid or uuid.uuid4(),
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

    def _prediction(self, session, article, action, at, skip_reason=None, classifier_type="fp"):
        session.add(models.ClassifierPrediction(
            article_id=article.id,
            classifier_type=classifier_type,
            probability=0.5,
            prediction=False,
            threshold_used=0.5,
            action_taken=action,
            skip_reason=skip_reason,
            created_at=at,
        ))

    def test_latest_prediction_decides(self, test_db):
        with test_db.get_session() as session:
            relabeled = self._article(session, "relabeled", "false_positive", "not sportswear")
            self._prediction(session, relabeled, "skipped_llm", JAN, FP_SKIP)
            self._prediction(session, relabeled, "continued_to_llm", MAR)
            # another classifier's later prediction must not count
            self._prediction(session, relabeled, "skipped_llm", MAR + timedelta(days=1), FP_SKIP, "ep")
            latest_skip = self._article(session, "latest-skip", "false_positive", FP_SKIP)
            self._prediction(session, latest_skip, "continued_to_llm", JAN)
            self._prediction(session, latest_skip, "skipped_llm", MAR, FP_SKIP)
            stale_skip = self._article(session, "stale-skip", "false_positive", "Puma: the animal")
            self._prediction(session, stale_skip, "skipped_llm", JAN, FP_SKIP)
            ruled = self._article(
                session, "ruled", "false_positive", FP_SKIP, article_uuid=uuid.UUID(CORRECTED_FP_ID)
            )
            self._prediction(session, ruled, "skipped_llm", JAN, FP_SKIP)
            self._article(session, "no-row", "false_positive", "not sportswear")
            self._article(session, "unjudged", "skipped", "No ESG content found")
            session.commit()

            records, report = export_false_positive_data(session)

        by_title = {r["title"]: r for r in records}
        assert by_title["provenance test relabeled"]["label_source"] == LLM
        assert by_title["provenance test stale-skip"]["label_source"] == LLM
        assert by_title["provenance test no-row"]["label_source"] == PRE_CLASSIFIER
        assert by_title["provenance test ruled"]["label_source"] == HUMAN
        assert by_title["provenance test ruled"]["article_id"] == CORRECTED_FP_ID
        assert "provenance test latest-skip" not in by_title
        assert "provenance test unjudged" not in by_title
        assert report.excluded_classifier_only == 1
        assert report.excluded_classifier_only_by_month == {"2026-03": 1}
        assert report.excluded_unjudged == 1
        assert [r["article_id"] for r in records] == sorted(r["article_id"] for r in records)

    def _judge(self, session, article, run):
        session.add(models.BrandJudgment(
            article_id=article.id, labeling_run_id=run.id, brand="Puma", content_type="product",
            reasoning="r", prompt_version="v1.14.0", model="m",
        ))

    def test_excluded_strata_from_the_tagged_runs(self, test_db):
        """#289: the FM stratum is read from the pass's runs and the rows' current state, and the
        export leaves its members out. Assertions are scoped to the rows made here, so other
        modules' rows in the shared test database do not reach them."""
        in_pass = FM.runs_before - timedelta(days=1)
        with test_db.get_session() as session:
            first = models.LabelingRun(started_at=in_pass, config={})
            second = models.LabelingRun(started_at=in_pass + timedelta(hours=1), config={})
            after_pass = models.LabelingRun(started_at=FM.runs_before + timedelta(days=1), config={})
            daily = models.LabelingRun(started_at=in_pass + timedelta(hours=2), config={})
            other_tag = models.LabelingRun(started_at=in_pass, config={})
            runs = [first, second, after_pass, daily, other_tag]
            session.add_all(runs)
            session.flush()
            run_ids = [run.id for run in runs]
        try:
            self._strata_case(test_db, run_ids)
        finally:
            with test_db.get_session() as session:
                session.query(models.BrandJudgment).filter(
                    models.BrandJudgment.labeling_run_id.in_(run_ids)
                ).delete(synchronize_session=False)
                session.query(models.LabelingRun).filter(
                    models.LabelingRun.id.in_(run_ids)
                ).delete(synchronize_session=False)

    def _strata_case(self, test_db, run_ids):
        first_id, second_id, after_id, daily_id, other_id = run_ids
        with test_db.get_session() as session:
            run = {r.id: r for r in session.query(models.LabelingRun).filter(models.LabelingRun.id.in_(run_ids))}
            member = self._article(session, "fm-member", "skipped", JUDGED)
            retried = self._article(session, "fm-retried", "skipped", JUDGED)
            rejudged = self._article(session, "fm-rejudged", "skipped", JUDGED)
            later_pass = self._article(session, "fm-later-pass", "skipped", JUDGED)
            hand_moved = self._article(session, "fk-hand-moved", "skipped", "Puma: a shoe sale")
            kept_fp = self._article(session, "fk-kept", "false_positive", "Puma: the animal")
            was_skip = self._article(session, "sk-kept", "skipped", JUDGED)
            untagged = self._article(session, "other-tag", "skipped", JUDGED)
            ruled = self._article(session, "fm-ruled", "skipped", JUDGED, article_uuid=uuid.UUID(RULED_FM_ID))
            prior = lambda status: [status, "reason", "arm"]
            tagged = lambda priors: {"run_tag": FM.run_tag, "run_config": {"prior": priors}}
            run[first_id].config = tagged({
                str(a.id): prior("false_positive") for a in (member, retried, rejudged, hand_moved, kept_fp, ruled)
            } | {str(was_skip.id): prior("skipped")})
            # a retry's later record is not its prior status
            run[second_id].config = tagged({str(retried.id): prior("failed")})
            # an invocation of the same mode after the pass adds no row
            run[after_id].config = tagged({str(later_pass.id): prior("false_positive")})
            run[other_id].config = {"run_tag": provenance.MODE_RELABEL, "run_config": {"prior": {
                str(untagged.id): prior("false_positive"),
            }}}
            for article in (member, hand_moved, kept_fp, was_skip, ruled):
                self._judge(session, article, run[first_id])
            self._judge(session, retried, run[second_id])
            self._judge(session, rejudged, run[daily_id])
            self._judge(session, later_pass, run[after_id])
            self._judge(session, untagged, run[other_id])
            session.commit()

            made = {str(a.id) for a in (
                member, retried, rejudged, later_pass, hand_moved, kept_fp, was_skip, untagged, ruled
            )}
            members = {i: n for i, n in provenance.excluded_stratum_ids(session).items() if i in made}
            records, report = export_false_positive_data(session)
            expected = {str(member.id): STRATUM, str(retried.id): STRATUM, RULED_FM_ID: STRATUM}

        assert members == expected
        by_title = {r["title"]: r for r in records}
        assert "provenance test fm-member" not in by_title
        assert "provenance test fm-retried" not in by_title
        assert by_title["provenance test fm-ruled"]["label_source"] == HUMAN
        for kept in ("fm-rejudged", "fm-later-pass", "fk-kept", "sk-kept", "other-tag"):
            assert f"provenance test {kept}" in by_title
        # not a stratum member, but its reply's not-sportswear reason on a skip leaves it out
        assert "provenance test fk-hand-moved" not in by_title
        assert report.excluded_not_sportswear_skips >= 1
