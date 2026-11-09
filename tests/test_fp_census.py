"""The FP census (#155 PR1): reply verdicts, the review queue, and the script's spend and resume
rules. The labeler is always a stub; no test calls the API or the database."""

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from scripts import fp_census as script
from src.data_collection import provenance
from src.data_collection.models import FP_ACTION_SKIPPED_LLM
from src.evaluation import fp_census as census
from src.labeling.labeler import REFUSAL_ERROR_TYPE, LabelingResult
from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
CUTOFF = date(2026, 2, 19)
PRE = datetime(2026, 1, 15, tzinfo=timezone.utc)
POST = datetime(2026, 5, 1, tzinfo=timezone.utc)


# --- replies ---------------------------------------------------------------------------------

def _esg(brand):
    categories = {
        c: CategoryLabel(applies=False, sentiment=None, evidence=[])
        for c in ("social", "governance", "digital_transformation")
    }
    categories["environmental"] = CategoryLabel(applies=True, sentiment=1, evidence=["e"])
    return BrandAnalysis(brand=brand, is_sportswear_brand=True, categories=categories, reasoning="esg")


def _no_esg(brand):
    return BrandAnalysis(brand=brand, is_sportswear_brand=True, categories={}, reasoning="product")


def _not_sportswear(brand):
    return BrandAnalysis(brand=brand, is_sportswear_brand=False, not_sportswear_reason="the animal", reasoning="no")


def _ok(analyses, input_tokens=1000, output_tokens=100):
    return LabelingResult(
        success=True,
        response=LabelingResponse(brand_analyses=analyses, article_summary="s"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="end_turn",
    )


def _failed(error_type="authentication", output_tokens=0):
    return LabelingResult(success=False, error="boom", error_type=error_type, output_tokens=output_tokens)


def _row(article_id="a1", brands=("Puma",), probability=0.3, threshold=0.53, predicted_at=POST, content="x" * 300):
    return census.CohortRow(
        article_id=article_id,
        title=f"T {article_id}",
        content=content,
        brands=list(brands),
        probability=probability,
        threshold=threshold,
        predicted_at=predicted_at,
    )


def _record(row, result, cost=0.01):
    return census.census_record(
        row, result, model="claude-sonnet-5-5", effort="medium", prompt_version="v1.13.0", cost_usd=cost, labeled_at=NOW
    )


class TestVerdict:
    """Each reply maps to a verdict through record_reply, so through dispose_reply (D-10.1)."""

    @pytest.mark.parametrize(
        "analyses,brands,verdict,agreement",
        [
            ([_esg("Puma")], ["Puma"], census.POSITIVE, census.DISAGREE),  # labeled
            ([_no_esg("Puma")], ["Puma"], census.POSITIVE, census.DISAGREE),  # judged skip
            ([_not_sportswear("Puma")], ["Puma"], census.NEGATIVE, census.AGREE),  # false_positive
            ([], ["Puma"], census.UNJUDGED, None),  # no analyses: the model never judged the brand
            # One sent brand left out of an all-negative reply: it may be the sportswear one
            ([_not_sportswear("Puma")], ["Puma", "Nike"], census.UNJUDGED, None),
        ],
    )
    def test_reply_verdicts(self, analyses, brands, verdict, agreement):
        record = _record(_row(brands=brands), _ok(analyses))
        assert record["verdict"] == verdict
        assert record["agreement"] == agreement

    def test_api_failure_is_failed_with_no_agreement(self):
        record = _record(_row(), _failed())
        assert (record["verdict"], record["agreement"], record["refusal"]) == (census.VERDICT_FAILED, None, False)

    def test_refusal_is_failed_and_marked(self):
        record = _record(_row(), _failed(error_type=REFUSAL_ERROR_TYPE, output_tokens=5))
        assert record["verdict"] == census.VERDICT_FAILED
        assert record["refusal"] is True

    def test_unrecognized_outcome_is_refused(self):
        with pytest.raises(ValueError):
            census.verdict_of("labelled")

    def test_record_carries_the_classifier_side_and_content_types(self):
        analysis = BrandAnalysis(brand="Puma", content_type="other_entity", reasoning="the animal")
        record = _record(_row(probability=0.45, threshold=0.4911), _ok([analysis]))
        assert record["probability"] == 0.45 and record["threshold"] == 0.4911
        assert record["band"] == census.BAND_HIGH
        assert record["content_types"] == {"puma": "other_entity"}
        assert record["predicted_at"] == POST.isoformat()
        json.dumps(record)  # serializable as written


def test_probability_bands_at_their_edges():
    assert census.probability_band(0.0) == census.BAND_LOW
    assert census.probability_band(0.1999) == census.BAND_LOW
    assert census.probability_band(0.2) == census.BAND_MID
    assert census.probability_band(0.3999) == census.BAND_MID
    assert census.probability_band(0.4) == census.BAND_HIGH


def test_before_cutoff_reads_naive_times_as_utc_and_none_as_not_before():
    assert census.before_cutoff(datetime(2026, 2, 18, 23, 59), CUTOFF)
    assert not census.before_cutoff(datetime(2026, 2, 19, tzinfo=timezone.utc), CUTOFF)
    assert not census.before_cutoff(None, CUTOFF)


def test_needs_call_retries_failures_but_not_refusals():
    assert census.needs_call(None)
    assert not census.needs_call({"verdict": census.NEGATIVE})
    assert not census.needs_call({"verdict": census.UNJUDGED})
    assert census.needs_call({"verdict": census.VERDICT_FAILED, "refusal": False})
    assert not census.needs_call({"verdict": census.VERDICT_FAILED, "refusal": True})


def test_latest_record_wins():
    lines = [{"article_id": "a", "verdict": "failed"}, {"article_id": "b"}, {"article_id": "a", "verdict": "negative"}]
    assert census.latest_records(lines)["a"]["verdict"] == "negative"


# --- summary and queue -----------------------------------------------------------------------

def _rec(article_id, agreement, probability=0.3, threshold=0.53, predicted_at=POST):
    verdict = {census.AGREE: census.NEGATIVE, census.DISAGREE: census.POSITIVE, None: census.UNJUDGED}[agreement]
    return {
        "article_id": article_id,
        "verdict": verdict,
        "agreement": agreement,
        "probability": probability,
        "threshold": threshold,
        "band": census.probability_band(probability),
        "predicted_at": predicted_at.isoformat() if predicted_at else None,
        "cost_usd": 0.1,
    }


class TestSummary:
    def test_rates_by_band_month_and_cutoff(self):
        records = [
            _rec("a", census.DISAGREE, 0.1, predicted_at=PRE),
            _rec("b", census.AGREE, 0.1, predicted_at=PRE),
            _rec("c", census.AGREE, 0.3),
            _rec("d", None, 0.45),
        ]
        s = census.summarize(records, cutoff=CUTOFF)
        assert s["verdicts"] == {census.POSITIVE: 1, census.NEGATIVE: 2, census.UNJUDGED: 1, census.VERDICT_FAILED: 0}
        assert s["overall"]["disagreements"] == 1 and s["overall"]["judged"] == 3
        assert s["by_band"][census.BAND_LOW]["disagreement_rate"] == 0.5
        # No disagreement in 1 judged row: the rule-of-three bound is reported
        assert s["by_band"][census.BAND_MID]["upper_bound_if_none"] == 3.0
        assert s["by_band"][census.BAND_HIGH]["judged"] == 0
        assert s["by_month"]["2026-01"]["disagreements"] == 1
        assert s["before_cutoff"]["rows"] == 2 and s["after_cutoff"]["rows"] == 2
        assert s["cost_usd"] == pytest.approx(0.4)


class TestReviewQueue:
    def test_disagreements_first_nearest_their_own_threshold(self):
        records = [
            _rec("far", census.DISAGREE, 0.10),
            # 0.48 is 0.0111 under its own threshold 0.4911, nearer than 0.51 is to 0.53 (0.02)
            _rec("own", census.DISAGREE, 0.48, threshold=0.4911),
            _rec("near", census.DISAGREE, 0.51),
            _rec("agree", census.AGREE, 0.52),
        ]
        queue = census.build_review_queue(records, cutoff=CUTOFF)
        assert [e["article_id"] for e in queue[:3]] == ["own", "near", "far"]
        assert all(e["queue_reasons"] == [census.REASON_DISAGREEMENT] for e in queue[:3])
        assert [e["queue_rank"] for e in queue] == list(range(1, len(queue) + 1))

    def test_agreements_are_sampled_per_band_and_reproducibly(self):
        records = [_rec(f"lo{i:02d}", census.AGREE, 0.01 * (i % 19)) for i in range(40)]
        records += [_rec(f"mi{i:02d}", census.AGREE, 0.2 + 0.001 * i) for i in range(40)]
        records += [_rec(f"hi{i:02d}", census.AGREE, 0.4 + 0.001 * i) for i in range(5)]
        queue = census.build_review_queue(records, cutoff=CUTOFF, agreements_per_band=10, seed=7)
        by_band = {}
        for e in queue:
            assert e["queue_reasons"] == [census.REASON_AGREEMENT_SAMPLE]
            by_band.setdefault(e["band"], []).append(e)
        # Each band gives min(k, its size), never another band's rows
        assert {b: len(v) for b, v in by_band.items()} == {census.BAND_LOW: 10, census.BAND_MID: 10, census.BAND_HIGH: 5}
        again = census.build_review_queue(list(reversed(records)), cutoff=CUTOFF, agreements_per_band=10, seed=7)
        assert [e["article_id"] for e in again] == [e["article_id"] for e in queue]
        other = census.build_review_queue(records, cutoff=CUTOFF, agreements_per_band=10, seed=8)
        assert {e["article_id"] for e in other} != {e["article_id"] for e in queue}
        # Uniform within the band, not the rows nearest the threshold
        top_mid = {f"mi{i:02d}" for i in range(30, 40)}
        assert {e["article_id"] for e in by_band[census.BAND_MID]} != top_mid

    def test_pre_cutoff_rows_are_a_census_when_few(self):
        records = [
            _rec("pre-disagree", census.DISAGREE, predicted_at=PRE),
            _rec("pre-unjudged", None, predicted_at=PRE),
            _rec("post-unjudged", None, predicted_at=POST),
        ]
        queue = {e["article_id"]: e for e in census.build_review_queue(records, cutoff=CUTOFF, pre_cutoff_census_max=5)}
        # A row queued twice keeps one entry with both reasons
        assert queue["pre-disagree"]["queue_reasons"] == [census.REASON_DISAGREEMENT, census.REASON_PRE_CUTOFF]
        assert queue["pre-unjudged"]["queue_reasons"] == [census.REASON_PRE_CUTOFF]
        # A row with no verdict is queued only by the pre-cutoff draw
        assert "post-unjudged" not in queue

    def test_pre_cutoff_rows_are_sampled_when_many(self):
        records = [_rec(f"p{i:03d}", census.AGREE, predicted_at=PRE) for i in range(30)]
        queue = census.build_review_queue(
            records, cutoff=CUTOFF, agreements_per_band=0, pre_cutoff_census_max=10, pre_cutoff_sample=6
        )
        assert len(queue) == 6
        assert all(e["queue_reasons"] == [census.REASON_PRE_CUTOFF] for e in queue)


# --- the script ------------------------------------------------------------------------------

class StubLabeler:
    def __init__(self, results):
        self.model = "claude-sonnet-5-5"
        self.results = list(results)
        self.calls = []

    def label_article(self, title, content, brands, published_at=None, source_name=None):
        self.calls.append(title)
        return self.results.pop(0)


@pytest.fixture
def env(tmp_path):
    """The script with its cohort, prompt and labeler stubbed; returns a runner."""
    cohort = [_row(f"a{i}", probability=0.1 * (i + 1)) for i in range(3)]
    state = SimpleNamespace(cohort=cohort, labeler=None, built=0, output=tmp_path / "census.jsonl")

    def build_labeler(args, version):
        state.built += 1
        return state.labeler

    manager = MagicMock()
    manager.get_production_version.return_value = "v1.13.0"
    with (
        patch.object(script, "read_cohort", side_effect=lambda: state.cohort),
        patch.object(script, "PromptManager", return_value=manager),
        patch.object(script, "prompt_input_chars", return_value=10_000),
        patch.object(script, "build_labeler", side_effect=build_labeler),
    ):
        state.run = lambda *argv: script.main(["--output", str(state.output), *argv])
        yield state


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _worst():
    return script.worst_case_call_cost_usd(10_000, 300, 8000, "claude-sonnet-5-5")


class TestScript:
    def test_dry_run_calls_nothing_and_writes_nothing(self, env, capsys):
        assert env.run("--dry-run") == script.EXIT_DONE
        assert env.built == 0
        assert not env.output.exists()
        assert "Cohort (label source fp_classifier): 3" in capsys.readouterr().out

    def test_a_run_needs_max_cost(self, env):
        assert env.run() == script.EXIT_REFUSED
        assert env.built == 0

    def test_refused_before_any_call_when_one_worst_case_passes_the_cap(self, env):
        assert env.run("--max-cost", str(_worst() / 2)) == script.EXIT_REFUSED
        assert env.built == 0 and not env.output.exists()

    def test_runs_every_article_and_records_each(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")]), _ok([_esg("Puma")]), _ok([])])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        lines = _lines(env.output)
        assert [line["verdict"] for line in lines] == [census.NEGATIVE, census.POSITIVE, census.UNJUDGED]
        # Priced at the census model's rates: 1000 input and 100 output tokens at $2/$10 per MTok
        assert lines[0]["cost_usd"] == pytest.approx(0.003)
        assert {line["model"] for line in lines} == {"claude-sonnet-5-5"}
        assert {line["effort"] for line in lines} == {"medium"}

    def test_stops_before_the_call_that_could_pass_the_cap(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 3)
        # Room for exactly one worst case once the first call has cost $0.003
        cap = 0.003 + _worst() + 0.0001
        assert env.run("--max-cost", str(cap)) == script.EXIT_STOPPED_AT_CAP
        assert len(_lines(env.output)) == 2

    def test_resume_skips_recorded_articles_and_retries_failures_but_not_refusals(self, env):
        prior = [
            _record(env.cohort[0], _ok([_not_sportswear("Puma")]), cost=1.0),
            _record(env.cohort[1], _failed(), cost=0.0),
            _record(env.cohort[2], _failed(error_type=REFUSAL_ERROR_TYPE, output_tokens=5), cost=0.5),
        ]
        env.output.write_text("".join(json.dumps(r) + "\n" for r in prior))
        env.labeler = StubLabeler([_ok([_esg("Puma")])])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert env.labeler.calls == ["T a1"]
        assert census.latest_records(_lines(env.output))["a1"]["verdict"] == census.POSITIVE

    def test_spend_already_in_the_file_counts_toward_the_cap(self, env):
        prior = [_record(env.cohort[0], _ok([_not_sportswear("Puma")]), cost=9.99)]
        env.output.write_text(json.dumps(prior[0]) + "\n")
        assert env.run("--max-cost", "10") == script.EXIT_REFUSED
        assert env.built == 0

    def test_stops_after_consecutive_calls_with_no_reply(self, env):
        env.cohort = [_row(f"a{i}") for i in range(5)]
        env.labeler = StubLabeler([_failed()] * 5)
        assert env.run("--max-cost", "10") == script.EXIT_ABORTED
        assert len(_lines(env.output)) == script.MAX_CONSECUTIVE_NO_REPLY

    def test_limit(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])])
        assert env.run("--max-cost", "10", "--limit", "1") == script.EXIT_DONE
        assert len(_lines(env.output)) == 1

    def test_queue_mode_writes_the_queue_without_calling(self, env, capsys):
        records = [
            _record(env.cohort[0], _ok([_esg("Puma")])),
            _record(env.cohort[1], _ok([_not_sportswear("Puma")])),
        ]
        env.output.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert env.run("--queue") == script.EXIT_DONE
        assert env.built == 0
        queue = _lines(script.queue_path(env.output))
        assert queue[0]["article_id"] == "a0" and queue[0]["queue_reasons"] == [census.REASON_DISAGREEMENT]
        assert "1/2 disagree" in capsys.readouterr().out


def test_read_cohort_rolls_back_and_never_commits():
    session = MagicMock()
    with (
        patch.object(script.db, "SessionLocal", return_value=session),
        patch.object(script, "load_cohort", return_value=[]),
    ):
        assert script.read_cohort() == []
    session.rollback.assert_called_once()
    session.commit.assert_not_called()


def test_load_cohort_keeps_classifier_only_negatives_with_their_latest_prediction():
    skip_reason = "Low-risk FP skip"
    ids = {name: uuid4() for name in ("classifier", "relabeled", "ruled")}

    def article(name, status="false_positive", error=skip_reason):
        return SimpleNamespace(
            id=ids[name], labeling_status=status, labeling_error=error, title=name, full_content="c" * 300,
            description=None, brands_mentioned=["Puma"], published_at=None, source_name=None,
        )

    rows = [
        (article("classifier"), FP_ACTION_SKIPPED_LLM, skip_reason, POST),
        # A different stored reason: something relabeled it after the skip, so the LLM decided it
        (article("relabeled", error="Puma: the animal"), FP_ACTION_SKIPPED_LLM, skip_reason, POST),
        (article("ruled"), FP_ACTION_SKIPPED_LLM, skip_reason, POST),
    ]
    session = MagicMock()
    session.execute.return_value = [
        SimpleNamespace(article_id=ids["classifier"], probability=0.1, threshold_used=0.53, created_at=PRE),
        SimpleNamespace(article_id=ids["classifier"], probability=0.45, threshold_used=0.4911, created_at=POST),
    ]
    query = MagicMock()
    query.all.return_value = rows
    with (
        patch.object(script.provenance, "fp_export_query", return_value=query),
        patch.dict(provenance.HUMAN_RULINGS, {str(ids["ruled"]): "false_positive"}),
    ):
        cohort = script.load_cohort(session)
    assert [r.article_id for r in cohort] == [str(ids["classifier"])]
    # The last prediction the ordered query returns is the latest
    assert (cohort[0].probability, cohort[0].threshold, cohort[0].predicted_at) == (0.45, 0.4911, POST)
