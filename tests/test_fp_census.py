"""The FP census (#155 PR1): reply verdicts, the review queue, and the script's spend, resume and
queue rules. The labeler is always a stub; no test calls the API or the database."""

import json
import random
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

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
SONNET = "claude-sonnet-5-5"


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


def _failed(error_type="authentication", output_tokens=0, input_tokens=0):
    return LabelingResult(
        success=False, error="boom", error_type=error_type, input_tokens=input_tokens, output_tokens=output_tokens
    )


def _billed_failure():
    """A reply the model was billed for that did not parse."""
    return _failed(error_type="parse_error", input_tokens=1000, output_tokens=50)


def _refusal():
    return _failed(error_type=REFUSAL_ERROR_TYPE, input_tokens=1000, output_tokens=5)


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


def _record(row, result, cost=0.01, effort="medium"):
    return census.census_record(
        row,
        result,
        model=SONNET,
        effort=effort,
        prompt_version="v1.13.0",
        cost_usd=cost,
        worst_case_usd=0.09,
        labeled_at=NOW,
    )


class TestVerdict:
    """A parsed reply maps to a verdict through record_reply, so through dispose_reply (D-10.1);
    a failed call is `failed` before dispose_reply."""

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

    def test_mention_only_is_negative(self):
        """A brand the article only mentions is not sportswear under the production policy."""
        record = _record(_row(), _ok([BrandAnalysis(brand="Puma", content_type="mention_only", reasoning="m")]))
        assert record["verdict"] == census.NEGATIVE

    def test_api_failure_is_failed_with_no_agreement(self):
        record = _record(_row(), _failed())
        assert (record["verdict"], record["agreement"], record["refusal"]) == (census.VERDICT_FAILED, None, False)

    def test_refusal_is_failed_and_marked(self):
        record = _record(_row(), _refusal())
        assert record["verdict"] == census.VERDICT_FAILED
        assert record["refusal"] is True

    def test_unrecognized_outcome_is_refused(self):
        with pytest.raises(ValueError):
            census.verdict_of("labelled")

    def test_record_carries_the_classifier_side_brand_lists_and_content_types(self):
        analysis = BrandAnalysis(brand="Puma", content_type="other_entity", reasoning="the animal")
        unsent = BrandAnalysis(brand="Reebok", content_type="other_entity", reasoning="x")
        record = _record(_row(brands=["Puma", "Nike"], probability=0.45, threshold=0.4911), _ok([analysis, unsent]))
        assert record["probability"] == 0.45 and record["threshold"] == 0.4911
        assert record["band"] == census.BAND_HIGH
        assert record["content_types"]["puma"] == "other_entity"
        assert record["omitted"] == ["Nike"] and record["unsent"] == ["Reebok"]
        assert record["conflicting"] == []
        assert record["predicted_at"] == POST.isoformat()
        assert record["worst_case_usd"] == 0.09
        json.dumps(record)  # serializable as written

    def test_record_converts_an_aware_time_to_utc(self):
        """A skip at 02:00 UTC on 1 March read in a -05:00 session is a March skip."""
        local = datetime(2026, 2, 28, 21, 0, tzinfo=timezone(timedelta(hours=-5)))
        record = _record(_row(predicted_at=local), _ok([_not_sportswear("Puma")]))
        assert record["predicted_at"] == "2026-03-01T02:00:00+00:00"
        assert census.summarize([record], cutoff=CUTOFF, spend_usd=0)["by_month"].keys() == {"2026-03"}

    def test_record_effort_is_what_was_passed(self):
        assert _record(_row(), _ok([_not_sportswear("Puma")]), effort=None)["effort"] is None


def test_probability_bands_at_their_edges():
    assert census.probability_band(0.0) == census.BAND_LOW
    assert census.probability_band(0.1999) == census.BAND_LOW
    assert census.probability_band(0.2) == census.BAND_MID
    assert census.probability_band(0.3999) == census.BAND_MID
    assert census.probability_band(0.4) == census.BAND_HIGH


def test_before_cutoff_reads_naive_times_as_utc_aware_times_by_instant_and_none_as_not_before():
    assert census.before_cutoff(datetime(2026, 2, 18, 23, 59), CUTOFF)
    assert not census.before_cutoff(datetime(2026, 2, 19, tzinfo=timezone.utc), CUTOFF)
    # 23:00 on the 18th at -05:00 is 04:00 UTC on the 19th
    assert not census.before_cutoff(datetime(2026, 2, 18, 23, 0, tzinfo=timezone(timedelta(hours=-5))), CUTOFF)
    assert not census.before_cutoff(None, CUTOFF)


def test_needs_call_retries_only_failures_with_no_reply():
    assert census.needs_call(None)
    assert not census.needs_call({"verdict": census.NEGATIVE})
    assert not census.needs_call({"verdict": census.UNJUDGED})
    assert census.needs_call({"verdict": census.VERDICT_FAILED, "refusal": False, "output_tokens": 0})
    # Billed failures and refusals are final
    assert not census.needs_call({"verdict": census.VERDICT_FAILED, "refusal": False, "output_tokens": 50})
    assert not census.needs_call({"verdict": census.VERDICT_FAILED, "refusal": True, "output_tokens": 0})


def test_latest_record_wins():
    lines = [{"article_id": "a", "verdict": "failed"}, {"article_id": "b"}, {"article_id": "a", "verdict": "negative"}]
    assert census.latest_records(lines)["a"]["verdict"] == "negative"


def test_rule_of_three_is_capped_at_one():
    assert census.rule_of_three(0) is None
    assert census.rule_of_three(1) == 1.0
    assert census.rule_of_three(3) == 1.0
    assert census.rule_of_three(30) == pytest.approx(0.1)


# --- summary and queue -----------------------------------------------------------------------

def _rec(article_id, agreement, probability=0.3, threshold=0.53, predicted_at=POST, verdict=None):
    if verdict is None:
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
    def test_rates_by_band_month_and_cutoff_side(self):
        records = [
            _rec("a", census.DISAGREE, 0.1, predicted_at=PRE),
            _rec("b", census.AGREE, 0.1, predicted_at=PRE),
            _rec("c", census.AGREE, 0.3),
            _rec("d", None, 0.45),
            _rec("e", census.AGREE, 0.3, predicted_at=None),
        ]
        s = census.summarize(records, cutoff=CUTOFF, spend_usd=1.25)
        assert s["verdicts"] == {census.POSITIVE: 1, census.NEGATIVE: 3, census.UNJUDGED: 1, census.VERDICT_FAILED: 0}
        assert s["overall"]["disagreements"] == 1 and s["overall"]["judged"] == 4
        assert s["by_band"][census.BAND_LOW]["disagreement_rate"] == 0.5
        # No disagreement in 2 judged rows: the bound is capped at 1
        assert s["by_band"][census.BAND_MID]["upper_bound_if_none"] == 1.0
        assert s["by_band"][census.BAND_HIGH]["judged"] == 0
        assert s["by_month"]["2026-01"]["disagreements"] == 1
        sides = s["by_cutoff_side"]
        assert (sides[census.BEFORE]["rows"], sides[census.AFTER]["rows"], sides[census.UNKNOWN]["rows"]) == (2, 2, 1)
        # Spend is what the caller says every call cost, not a sum over these records
        assert s["cost_usd"] == 1.25


def _queue(records, **kw):
    return census.build_review_queue(records, cutoff=CUTOFF, **kw)


class TestReviewQueue:
    def test_sections_in_order_each_nearest_its_own_threshold(self):
        records = [
            _rec("d-far", census.DISAGREE, 0.10),
            # 0.48 is 0.0111 under its own threshold 0.4911, nearer than 0.51 is to 0.53 (0.02)
            _rec("d-own", census.DISAGREE, 0.48, threshold=0.4911),
            _rec("d-near", census.DISAGREE, 0.51),
            _rec("n-far", None, 0.05, verdict=census.VERDICT_FAILED),
            _rec("n-near", None, 0.50),
            _rec("a-low", census.AGREE, 0.05),
            _rec("a-high", census.AGREE, 0.52),
            _rec("a-mid", census.AGREE, 0.30),
        ]
        queue, _ = _queue(records)
        assert [e["article_id"] for e in queue] == [
            "d-own", "d-near", "d-far", "n-near", "n-far", "a-high", "a-mid", "a-low",
        ]
        assert [e["queue_reasons"][0] for e in queue] == (
            [census.REASON_DISAGREEMENT] * 3 + [census.REASON_NO_VERDICT] * 2 + [census.REASON_AGREEMENT_SAMPLE] * 3
        )
        assert [e["queue_rank"] for e in queue] == list(range(1, len(queue) + 1))

    def test_agreements_are_the_seeded_uniform_draw_from_each_whole_band(self):
        records = [_rec(f"lo{i:02d}", census.AGREE, 0.01 * (i % 19)) for i in range(40)]
        records += [_rec(f"mi{i:02d}", census.AGREE, 0.2 + 0.001 * i) for i in range(40)]
        records += [_rec(f"hi{i:02d}", census.AGREE, 0.4 + 0.001 * i) for i in range(5)]
        queue, design = _queue(records, agreements_per_band=10, seed=7)
        # The oracle: one rng, each band's whole pool in article-id order, bands in BANDS order
        rng = random.Random(7)
        expected = set()
        for prefix in ("lo", "mi", "hi"):
            pool = sorted((r for r in records if r["article_id"].startswith(prefix)), key=lambda r: r["article_id"])
            expected |= {r["article_id"] for r in rng.sample(pool, min(10, len(pool)))}
        assert {e["article_id"] for e in queue} == expected
        assert design["agreements_by_band"] == {
            census.BAND_LOW: {"pool": 40, "drawn": 10},
            census.BAND_MID: {"pool": 40, "drawn": 10},
            census.BAND_HIGH: {"pool": 5, "drawn": 5},
        }
        again, _ = _queue(list(reversed(records)), agreements_per_band=10, seed=7)
        assert [e["article_id"] for e in again] == [e["article_id"] for e in queue]

    def test_pre_cutoff_rows_are_a_census_up_to_the_max(self):
        records = [
            _rec("pre-disagree", census.DISAGREE, predicted_at=PRE),
            _rec("pre-agree", census.AGREE, predicted_at=PRE),
            _rec("post-unjudged", None, predicted_at=POST),
        ]
        queue, design = _queue(records, agreements_per_band=0, pre_cutoff_census_max=2)
        entries = {e["article_id"]: e for e in queue}
        assert len(queue) == len(entries) == 3  # one entry per row
        # A row queued twice keeps one entry with both reasons
        assert entries["pre-disagree"]["queue_reasons"] == [census.REASON_DISAGREEMENT, census.REASON_PRE_CUTOFF]
        assert entries["pre-agree"]["queue_reasons"] == [census.REASON_PRE_CUTOFF]
        # A post-cutoff row with no verdict is queued as such
        assert entries["post-unjudged"]["queue_reasons"] == [census.REASON_NO_VERDICT]
        assert design["pre_cutoff"] == {"pool": 2, "drawn": 2, "mode": "census"}

    def test_pre_cutoff_rows_are_the_seeded_draw_from_every_pre_cutoff_row_when_many(self):
        kinds = [census.DISAGREE, census.AGREE, None]
        records = [_rec(f"p{i:03d}", kinds[i % 3], predicted_at=PRE) for i in range(30)]
        queue, design = _queue(records, agreements_per_band=2, pre_cutoff_census_max=10, pre_cutoff_sample=6, seed=3)
        rng = random.Random(3)
        ordered = sorted(records, key=lambda r: r["article_id"])
        for band in census.BANDS:
            pool = [r for r in ordered if r["agreement"] == census.AGREE and r["band"] == band]
            rng.sample(pool, min(2, len(pool)))
        expected = {r["article_id"] for r in rng.sample(ordered, 6)}
        drawn = {e["article_id"] for e in queue if census.REASON_PRE_CUTOFF in e["queue_reasons"]}
        assert drawn == expected
        assert design["pre_cutoff"] == {"pool": 30, "drawn": 6, "mode": "sample"}
        other, _ = _queue(records, agreements_per_band=2, pre_cutoff_census_max=10, pre_cutoff_sample=6, seed=4)
        assert {e["article_id"] for e in other if census.REASON_PRE_CUTOFF in e["queue_reasons"]} != drawn


# --- the script ------------------------------------------------------------------------------

class StubLabeler:
    def __init__(self, results):
        self.model = SONNET
        self.results = list(results)
        self.calls = []

    def label_article(self, title, content, brands, published_at=None, source_name=None):
        self.calls.append(title)
        return self.results.pop(0)


@pytest.fixture
def env(tmp_path):
    """The script with its cohort, prompt and labeler stubbed; returns a runner."""
    cohort = [_row(f"a{i}", probability=0.1 * (i + 1)) for i in range(3)]
    state = SimpleNamespace(cohort=cohort, labeler=None, built=[], output=tmp_path / "census.jsonl")

    def build_labeler(args, version):
        state.built.append((args.model, args.effort, version))
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


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _worst(row=None, model=SONNET, max_tokens=8000):
    return script.worst_case_call_cost_usd(10_000, script.call_chars(row or _row("a0")), max_tokens, model)


class TestRunRefusals:
    def test_dry_run_calls_nothing_and_writes_nothing(self, env, capsys):
        assert env.run("--dry-run") == script.EXIT_DONE
        assert env.built == []
        assert not env.output.exists()
        assert "Cohort (label source fp_classifier): 3" in capsys.readouterr().out

    def test_a_run_needs_max_cost(self, env):
        assert env.run() == script.EXIT_REFUSED
        assert env.built == []

    @pytest.mark.parametrize("cap", ["nan", "inf", "0", "-1"])
    def test_a_cap_that_is_not_a_positive_number_is_refused(self, env, cap):
        # With nothing to call, only the cap's own check can refuse
        env.cohort = []
        assert env.run("--max-cost", cap) == script.EXIT_REFUSED
        assert env.built == []

    def test_refused_before_any_call_when_one_worst_case_passes_the_cap(self, env):
        assert env.run("--max-cost", str(_worst() / 2)) == script.EXIT_REFUSED
        assert env.built == [] and not env.output.exists()

    def test_spend_on_every_line_counts_toward_the_cap(self, env):
        """Two lines for one article: the superseded billed failure still counts."""
        first = _record(env.cohort[0], _billed_failure(), cost=5.0)
        second = _record(env.cohort[0], _ok([_not_sportswear("Puma")]), cost=4.95)
        _write(env.output, [first, second])
        assert env.run("--max-cost", "10") == script.EXIT_REFUSED
        assert env.built == []

    def test_insufficient_content_rows_are_never_sent(self, env, capsys):
        env.cohort = [_row("a0"), _row("short", content="x" * 50)]
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert env.labeler.calls == ["T a0"]
        assert "Left out for insufficient content (never sent): 1" in capsys.readouterr().out

    def test_a_malformed_line_before_the_last_is_refused(self, env):
        env.output.write_text('{"article_id": "a0"\n' + json.dumps(_record(env.cohort[1], _ok([]))) + "\n")
        assert env.run("--max-cost", "10") == script.EXIT_REFUSED
        assert env.built == []


class TestRun:
    def test_runs_every_article_and_records_each(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")]), _ok([_esg("Puma")]), _ok([])])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        lines = _lines(env.output)
        assert [line["verdict"] for line in lines] == [census.NEGATIVE, census.POSITIVE, census.UNJUDGED]
        # Priced at the census model's rates: 1000 input and 100 output tokens at $2/$10 per MTok
        assert lines[0]["cost_usd"] == pytest.approx(0.003)
        assert lines[0]["worst_case_usd"] == pytest.approx(_worst(env.cohort[0]))
        assert {line["model"] for line in lines} == {SONNET}
        assert {line["effort"] for line in lines} == {"medium"}
        assert env.built == [(SONNET, "medium", "v1.13.0")]

    def test_a_model_that_takes_no_effort_records_none(self, env, capsys):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])])
        assert env.run("--model", "claude-haiku-4-5", "--max-cost", "10", "--limit", "1") == script.EXIT_DONE
        assert _lines(env.output)[0]["effort"] is None
        assert "effort not sent" in capsys.readouterr().out

    def test_stops_before_the_call_that_could_pass_the_cap(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 3)
        # Room for exactly one worst case once the first call has cost $0.003
        cap = 0.003 + _worst() + 0.0001
        assert env.run("--max-cost", str(cap)) == script.EXIT_STOPPED_AT_CAP
        assert len(_lines(env.output)) == 2

    def test_the_cap_check_inside_the_loop_counts_spend_already_in_the_file(self, env):
        env.cohort.append(_row("a3"))
        _write(env.output, [_record(env.cohort[3], _ok([_not_sportswear("Puma")]), cost=5.0)])
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 3)
        assert env.run("--max-cost", str(5.0 + 0.003 + _worst() + 0.0001)) == script.EXIT_STOPPED_AT_CAP
        assert len(env.labeler.calls) == 2

    def test_a_cap_equal_to_the_worst_case_allows_the_call(self, env):
        env.cohort = [_row("a0")]
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])])
        assert env.run("--max-cost", repr(_worst())) == script.EXIT_DONE
        assert env.labeler.calls == ["T a0"]

    def test_each_row_has_its_own_worst_case(self, env):
        long_row = _row("a1", content="x" * 200_000)
        env.cohort = [_row("a0"), long_row, _row("a2")]
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 3)
        # Room for two short worst cases, not for the long one
        cap = 0.003 + 2 * _worst() + 0.0001
        assert _worst(long_row) > 2 * _worst()
        assert env.run("--max-cost", str(cap)) == script.EXIT_STOPPED_AT_CAP
        assert env.labeler.calls == ["T a0"]

    def test_stops_after_a_call_that_cost_more_than_its_worst_case(self, env):
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")], input_tokens=10_000_000)] * 3)
        assert env.run("--max-cost", "100") == script.EXIT_ABORTED
        assert len(_lines(env.output)) == 1

    def test_limit_applies_after_the_resume_filter(self, env):
        _write(env.output, [_record(env.cohort[0], _ok([_not_sportswear("Puma")]))])
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])])
        assert env.run("--max-cost", "10", "--limit", "1") == script.EXIT_DONE
        assert env.labeler.calls == ["T a1"]


class TestResume:
    def test_only_failures_with_no_reply_are_called_again(self, env):
        env.cohort.append(_row("a3"))
        _write(
            env.output,
            [
                _record(env.cohort[0], _ok([_not_sportswear("Puma")])),
                _record(env.cohort[1], _failed()),  # no reply: called again
                _record(env.cohort[2], _refusal()),  # final
                _record(env.cohort[3], _billed_failure()),  # final
            ],
        )
        env.labeler = StubLabeler([_ok([_esg("Puma")])])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert env.labeler.calls == ["T a1"]
        assert census.latest_records(_lines(env.output))["a1"]["verdict"] == census.POSITIVE

    def test_articles_never_called_go_before_retries_and_retries_go_oldest_attempt_first(self, env):
        """Three articles that fail with no reply every time cannot starve the rest."""
        env.cohort = [_row(f"a{i}") for i in range(5)]
        failures = []
        for i, minute in ((0, 3), (1, 1), (2, 2)):
            record = _record(env.cohort[i], _failed())
            record["labeled_at"] = (NOW + timedelta(minutes=minute)).isoformat()
            failures.append(record)
        _write(env.output, failures)
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 2 + [_failed()] * 3)
        assert env.run("--max-cost", "10") == script.EXIT_ABORTED
        assert env.labeler.calls == ["T a3", "T a4", "T a1", "T a2", "T a0"]

    def test_the_latest_line_of_an_article_decides(self, env):
        _write(
            env.output,
            [_record(env.cohort[0], _failed()), _record(env.cohort[0], _ok([_not_sportswear("Puma")]))],
        )
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 2)
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert env.labeler.calls == ["T a1", "T a2"]

    def test_an_interrupted_write_is_ignored_then_removed_and_every_record_is_fsynced(self, env, caplog):
        good = json.dumps(_record(env.cohort[0], _ok([_not_sportswear("Puma")])))
        env.output.write_text(good + '\n{"article_id": "a1", "verd')
        env.labeler = StubLabeler([_ok([_not_sportswear("Puma")])] * 2)
        with patch.object(script.os, "fsync") as fsync:
            assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert env.labeler.calls == ["T a1", "T a2"]
        assert "malformed last line" in caplog.text
        assert fsync.call_count == 3  # the trim and each record
        lines = env.output.read_text().splitlines()
        assert lines[0] == good
        assert [json.loads(line)["article_id"] for line in lines] == ["a0", "a1", "a2"]
        # The next run reads the file cleanly
        assert len(script.read_lines(env.output)) == 3

    def test_a_complete_malformed_last_line_is_refused(self, env):
        env.output.write_text(json.dumps(_record(env.cohort[0], _ok([]))) + '\n{"article_id": "a1"\n')
        assert env.run("--max-cost", "10") == script.EXIT_REFUSED

    def test_a_record_after_an_unterminated_valid_line_starts_a_new_line(self, tmp_path):
        path = tmp_path / "f.jsonl"
        path.write_text('{"article_id": "a0"}')
        script.append_line(path, {"article_id": "a1"})
        assert [r["article_id"] for r in script.read_lines(path)] == ["a0", "a1"]


class TestAbort:
    def test_stops_after_consecutive_failed_calls_of_any_kind(self, env):
        env.cohort = [_row(f"a{i}") for i in range(5)]
        env.labeler = StubLabeler([_failed(), _billed_failure(), _refusal(), _failed(), _failed()])
        assert env.run("--max-cost", "10") == script.EXIT_ABORTED
        assert len(_lines(env.output)) == script.MAX_CONSECUTIVE_FAILED

    def test_a_success_resets_the_count(self, env):
        env.cohort = [_row(f"a{i}") for i in range(5)]
        env.labeler = StubLabeler([_failed(), _failed(), _ok([_not_sportswear("Puma")]), _failed(), _failed()])
        assert env.run("--max-cost", "10") == script.EXIT_DONE
        assert len(_lines(env.output)) == 5


class TestQueue:
    def _records(self, env):
        return [
            _record(env.cohort[0], _ok([_esg("Puma")])),
            _record(env.cohort[1], _ok([_not_sportswear("Puma")])),
            _record(env.cohort[2], _ok([_not_sportswear("Puma")])),
        ]

    def test_writes_the_queue_and_its_design_without_calling(self, env, capsys):
        _write(env.output, self._records(env))
        assert env.run("--queue") == script.EXIT_DONE
        assert env.built == []
        queue = _lines(script.queue_path(env.output))
        assert queue[0]["article_id"] == "a0" and queue[0]["queue_reasons"] == [census.REASON_DISAGREEMENT]
        design = json.loads(script.design_path(env.output).read_text())
        assert design["records_sha256"] == script.sha256_of(env.output)
        # a1 (0.2) and a2 (0.3) agree, both in the middle band
        assert design["agreements_by_band"][census.BAND_MID] == {"pool": 2, "drawn": 2}
        assert (design["cohort"], design["missing"], design["partial"]) == (3, 0, False)
        assert "1/3 disagree" in capsys.readouterr().out

    def test_spend_counts_every_line(self, env, capsys):
        records = self._records(env)
        superseded = _record(env.cohort[0], _billed_failure(), cost=0.5)
        _write(env.output, [superseded, *records])
        assert env.run("--queue") == script.EXIT_DONE
        assert "spend $0.53" in capsys.readouterr().out

    def test_refuses_when_a_cohort_row_has_no_usable_record_unless_partial(self, env):
        _write(env.output, self._records(env)[:2] + [_record(env.cohort[2], _failed())])
        assert env.run("--queue") == script.EXIT_REFUSED
        assert not script.queue_path(env.output).exists()
        assert env.run("--queue", "--partial") == script.EXIT_DONE
        queue = {e["article_id"]: e for e in _lines(script.queue_path(env.output))}
        # The no-reply failure is queued with no verdict
        assert set(queue) == {"a0", "a1", "a2"}
        assert queue["a2"]["queue_reasons"] == [census.REASON_NO_VERDICT]
        design = json.loads(script.design_path(env.output).read_text())
        assert (design["partial"], design["never_called"], design["retryable_failures"]) == (True, 0, 1)

    def test_partial_leaves_out_articles_never_called(self, env):
        _write(env.output, self._records(env)[:2])
        assert env.run("--queue") == script.EXIT_REFUSED
        assert env.run("--queue", "--partial") == script.EXIT_DONE
        assert {e["article_id"] for e in _lines(script.queue_path(env.output))} == {"a0", "a1"}
        assert json.loads(script.design_path(env.output).read_text())["never_called"] == 1

    def test_drops_records_no_longer_in_the_cohort_and_ignores_unsendable_rows(self, env, capsys):
        _write(env.output, self._records(env))
        env.cohort = [env.cohort[0], env.cohort[1], _row("short", content="x" * 50)]
        assert env.run("--queue") == script.EXIT_DONE
        assert {e["article_id"] for e in _lines(script.queue_path(env.output))} == {"a0", "a1"}
        out = capsys.readouterr().out
        assert "no longer in the cohort: 1" in out
        assert "insufficient content (never sent): 1" in out

    def test_an_existing_queue_is_replaced_only_with_force(self, env):
        _write(env.output, self._records(env))
        assert env.run("--queue") == script.EXIT_DONE
        script.queue_path(env.output).write_text("kept\n")
        assert env.run("--queue") == script.EXIT_REFUSED
        assert script.queue_path(env.output).read_text() == "kept\n"
        assert env.run("--queue", "--force") == script.EXIT_DONE
        assert script.queue_path(env.output).read_text() != "kept\n"


class TestBuildLabeler:
    """The real build_labeler, with ArticleLabeler replaced."""

    def _args(self, **kw):
        return script.parse_args([f"--{k.replace('_', '-')}={v}" for k, v in kw.items()])

    def _labeler(self, model=SONNET, effort="medium", version="v1.13.0"):
        return SimpleNamespace(model=model, effort=effort, prompt_version=version, loaded_prompt=object(), max_tokens=8000)

    def test_builds_the_requested_model_effort_and_prompt(self):
        with patch.object(script, "ArticleLabeler", return_value=self._labeler()) as cls:
            script.build_labeler(self._args(), "v1.13.0")
        cls.assert_called_once_with(model=SONNET, effort="medium", prompt_version="v1.13.0")

    def test_refuses_a_labeler_running_another_model(self):
        with patch.object(script, "ArticleLabeler", return_value=self._labeler(model="claude-haiku-5-5")):
            with pytest.raises(script.LabelerMismatchError):
                script.build_labeler(self._args(), "v1.13.0")

    def test_refuses_a_labeler_that_fell_back_to_another_prompt(self):
        with patch.object(script, "ArticleLabeler", return_value=self._labeler(version="legacy")):
            with pytest.raises(script.LabelerMismatchError):
                script.build_labeler(self._args(), "v1.13.0")

    def test_a_constructor_error_is_a_refusal(self):
        with patch.object(script, "ArticleLabeler", side_effect=ValueError("no key")):
            with pytest.raises(script.LabelerMismatchError):
                script.build_labeler(self._args(), "v1.13.0")


# --- the cohort ------------------------------------------------------------------------------

def _cohort_fixture():
    skip_reason = "Low-risk FP skip"
    ids = {name: uuid4() for name in ("classifier", "relabeled", "ruled")}

    def article(name, error=skip_reason):
        return SimpleNamespace(
            id=ids[name], labeling_status="false_positive", labeling_error=error, title=name, full_content="c" * 300,
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
        SimpleNamespace(article_id=ids["ruled"], probability=0.2, threshold_used=0.53, created_at=PRE),
        SimpleNamespace(article_id=ids["classifier"], probability=0.45, threshold_used=0.4911, created_at=POST),
    ]
    query = MagicMock()
    query.all.return_value = rows
    return ids, session, query


def test_load_cohort_keeps_classifier_only_negatives_with_the_last_prediction_returned():
    ids, session, query = _cohort_fixture()
    with (
        patch.object(script.provenance, "fp_export_query", return_value=query),
        patch.dict(provenance.HUMAN_RULINGS, {str(ids["ruled"]): "false_positive"}),
    ):
        cohort = script.load_cohort(session)
    assert [r.article_id for r in cohort] == [str(ids["classifier"])]
    # Last wins over the rows execute returns
    assert (cohort[0].probability, cohort[0].threshold, cohort[0].predicted_at) == (0.45, 0.4911, POST)
    # The query orders so that the last row is the latest, and reads FP predictions only
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "ORDER BY classifier_predictions.created_at ASC NULLS FIRST, classifier_predictions.id ASC" in sql
    assert "classifier_predictions.classifier_type = 'fp'" in sql


def test_read_cohort_reads_through_load_cohort_and_never_writes():
    ids, session, query = _cohort_fixture()
    with (
        patch.object(script.db, "SessionLocal", return_value=session),
        patch.object(script.provenance, "fp_export_query", return_value=query),
    ):
        assert len(script.read_cohort()) == 2
    assert {call[0] for call in session.method_calls} == {"execute", "rollback", "close"}
    session.rollback.assert_called_once()
