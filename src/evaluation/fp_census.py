"""A second opinion on the FP negatives only the classifier decided, and the queue a human reviews
(#155 PR1, D-10.1 to D-10.3 on #155).

The cohort is the FP export's rows whose label source is ``fp_classifier``: the FP pre-filter
skipped them before any LLM call, so no model has judged them. The census sends each to the
labeler once and records what the reply decides, through ``record_reply`` and so
``dispose_reply``, the function the pipeline decides an article's status with.

This module makes no database or API call. ``scripts/fp_census.py`` reads the cohort, calls the
labeler and writes the records; this module turns a reply into a census record and the records
into a summary and a review queue. Nothing here is ground truth: the records are inputs to a human
review, and the review's decisions are what #155 PR2 stores.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.labeling.disposition import FALSE_POSITIVE, LABELED
from src.labeling.labeler import LabelingResult
from src.evaluation.comparator import (
    FAILED,
    SKIPPED_JUDGED,
    SKIPPED_UNJUDGED,
    record_reply,
)

# What the reply says about the article, against the classifier's negative
POSITIVE = "positive"  # a sent brand is the sportswear company (labeled, or a judged skip)
NEGATIVE = "negative"  # every sent brand judged not the sportswear company
UNJUDGED = "unjudged"  # the reply did not establish sportswear status (D-10.1, 2026-10-09)
VERDICT_FAILED = "failed"  # no usable reply: an API error, a refusal, or one that did not parse
VERDICTS = (POSITIVE, NEGATIVE, UNJUDGED, VERDICT_FAILED)

_VERDICT_OF_OUTCOME = {
    LABELED: POSITIVE,
    SKIPPED_JUDGED: POSITIVE,
    FALSE_POSITIVE: NEGATIVE,
    SKIPPED_UNJUDGED: UNJUDGED,
    FAILED: VERDICT_FAILED,
}

AGREE = "agree"
DISAGREE = "disagree"

# Probability bands (D-10.3). The top band ends at the row's own threshold, the value its skip
# was decided against
BAND_LOW = "[0, 0.2)"
BAND_MID = "[0.2, 0.4)"
BAND_HIGH = "[0.4, threshold)"
BANDS = (BAND_LOW, BAND_MID, BAND_HIGH)

# Why a row is in the review queue
REASON_DISAGREEMENT = "disagreement"
REASON_AGREEMENT_SAMPLE = "agreement_sample"
REASON_PRE_CUTOFF = "pre_cutoff"

# The owner-attested cutoff's initial value (#155, 2026-09-25): the last commit of the FP notebooks
DEFAULT_CUTOFF = date(2026, 2, 19)
DEFAULT_AGREEMENTS_PER_BAND = 20
DEFAULT_PRE_CUTOFF_CENSUS_MAX = 100
DEFAULT_PRE_CUTOFF_SAMPLE = 60
DEFAULT_SEED = 155


@dataclass(frozen=True)
class CohortRow:
    """One classifier-only negative, with what its latest FP prediction recorded."""

    article_id: str
    title: str
    content: str
    brands: list[str]
    probability: float
    threshold: float
    predicted_at: datetime | None
    published_at: datetime | None = None
    source_name: str | None = None


def verdict_of(outcome: str) -> str:
    """The census verdict for a reply's outcome class (``record_reply``'s ``outcome``)."""
    try:
        return _VERDICT_OF_OUTCOME[outcome]
    except KeyError:
        raise ValueError(f"Unrecognized reply outcome {outcome!r}") from None


def agreement_of(verdict: str) -> str | None:
    """Whether the reply agrees with the classifier's negative; None when it gave no verdict."""
    if verdict == NEGATIVE:
        return AGREE
    if verdict == POSITIVE:
        return DISAGREE
    return None


def probability_band(probability: float) -> str:
    """The D-10.3 band of a skip's probability. Every row in the cohort is below its threshold."""
    if probability < 0.2:
        return BAND_LOW
    if probability < 0.4:
        return BAND_MID
    return BAND_HIGH


def as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def before_cutoff(predicted_at: datetime | None, cutoff: date) -> bool:
    """Whether the FP skip predates the cutoff (UTC), as the export's ``--attested-before`` reads
    it. A row with no prediction time is not before any cutoff."""
    if predicted_at is None:
        return False
    return as_utc(predicted_at) < datetime.combine(cutoff, datetime.min.time(), tzinfo=timezone.utc)


def census_record(
    row: CohortRow,
    result: LabelingResult,
    *,
    model: str,
    effort: str | None,
    prompt_version: str,
    cost_usd: float,
    labeled_at: datetime,
) -> dict:
    """The JSON-serializable record of one census call."""
    reply = record_reply(row.brands, result)
    verdict = verdict_of(reply.outcome)
    return {
        "article_id": row.article_id,
        "title": row.title,
        "brands": list(row.brands),
        "published_at": row.published_at.isoformat() if row.published_at else None,
        "source_name": row.source_name,
        "probability": row.probability,
        "threshold": row.threshold,
        "band": probability_band(row.probability),
        "predicted_at": as_utc(row.predicted_at).isoformat() if row.predicted_at else None,
        "verdict": verdict,
        "agreement": agreement_of(verdict),
        "status": reply.status,
        "reason": reply.reason,
        "content_types": reply.content_types,
        "explanations": reply.explanations,
        "error": reply.error,
        "refusal": reply.refusal,
        "parse_failure": reply.parse_failure,
        "stop_reason": reply.stop_reason,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "cost_usd": cost_usd,
        "model": model,
        "effort": effort,
        "prompt_version": prompt_version,
        "labeled_at": as_utc(labeled_at).isoformat(),
    }


def latest_records(lines: list[dict]) -> dict[str, dict]:
    """The latest record per article, in file order: a retried article's last line wins."""
    latest: dict[str, dict] = {}
    for record in lines:
        latest[record["article_id"]] = record
    return latest


def needs_call(record: dict | None) -> bool:
    """Whether an article still needs a census call: never recorded, or recorded as failed for a
    reason other than a refusal. A refusal is the model's answer and is not retried."""
    if record is None:
        return True
    return record["verdict"] == VERDICT_FAILED and not record.get("refusal")


def rule_of_three(n: int) -> float | None:
    """The 95% upper bound on an error rate when 0 errors are seen in n (3/n); None for n=0."""
    return 3 / n if n else None


def _rates(records: list[dict]) -> dict:
    judged = [r for r in records if r["agreement"] is not None]
    disagreements = sum(r["agreement"] == DISAGREE for r in judged)
    return {
        "rows": len(records),
        "judged": len(judged),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / len(judged) if judged else None,
        "upper_bound_if_none": rule_of_three(len(judged)) if not disagreements else None,
    }


def _month(record: dict) -> str:
    return record["predicted_at"][:7] if record.get("predicted_at") else "unknown"


def summarize(records: list[dict], *, cutoff: date) -> dict:
    """Counts by verdict, and disagreement rates overall, per band (for #141), per month of the FP
    skip and either side of the cutoff (the "early data is clean" hypothesis), plus spend."""
    by_band: dict[str, list[dict]] = defaultdict(list)
    by_month: dict[str, list[dict]] = defaultdict(list)
    sides: dict[str, list[dict]] = {"before": [], "after": []}
    for r in records:
        by_band[r["band"]].append(r)
        by_month[_month(r)].append(r)
        predicted = datetime.fromisoformat(r["predicted_at"]) if r.get("predicted_at") else None
        sides["before" if before_cutoff(predicted, cutoff) else "after"].append(r)
    verdicts = Counter(r["verdict"] for r in records)
    return {
        "rows": len(records),
        "verdicts": {v: verdicts.get(v, 0) for v in VERDICTS},
        "refusals": sum(bool(r.get("refusal")) for r in records),
        "overall": _rates(records),
        "by_band": {b: _rates(by_band[b]) for b in BANDS},
        "by_month": {m: _rates(by_month[m]) for m in sorted(by_month)},
        "cutoff": cutoff.isoformat(),
        "before_cutoff": _rates(sides["before"]),
        "after_cutoff": _rates(sides["after"]),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in records),
    }


def build_review_queue(
    records: list[dict],
    *,
    cutoff: date = DEFAULT_CUTOFF,
    agreements_per_band: int = DEFAULT_AGREEMENTS_PER_BAND,
    pre_cutoff_census_max: int = DEFAULT_PRE_CUTOFF_CENSUS_MAX,
    pre_cutoff_sample: int = DEFAULT_PRE_CUTOFF_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """The rows a human reviews, in review order, each once (D-10.3).

    1. Every disagreement, nearest its own threshold first. The order is triage only.
    2. A uniform random sample of agreements within each probability band, so the residual error
       estimate is not biased toward the threshold.
    3. Pre-cutoff rows: all of them when there are at most ``pre_cutoff_census_max``, otherwise a
       uniform random sample of ``pre_cutoff_sample``, drawn from every pre-cutoff row whatever
       its verdict. A row already queued gains the reason rather than a second entry.

    Rows with no verdict (unjudged or failed) are queued only by the pre-cutoff draw; the summary
    counts them. Sampling is reproducible for a given ``seed`` and record set.
    """
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda r: r["article_id"])
    queue: dict[str, dict] = {}

    def add(record: dict, reason: str) -> None:
        entry = queue.get(record["article_id"])
        if entry is None:
            queue[record["article_id"]] = {**record, "queue_reasons": [reason]}
        elif reason not in entry["queue_reasons"]:
            entry["queue_reasons"].append(reason)

    disagreements = [r for r in ordered if r["agreement"] == DISAGREE]
    for r in sorted(disagreements, key=lambda r: (r["threshold"] - r["probability"], r["article_id"])):
        add(r, REASON_DISAGREEMENT)

    for band in BANDS:
        pool = [r for r in ordered if r["agreement"] == AGREE and r["band"] == band]
        for r in rng.sample(pool, min(agreements_per_band, len(pool))):
            add(r, REASON_AGREEMENT_SAMPLE)

    pre = [
        r
        for r in ordered
        if r.get("predicted_at") and before_cutoff(datetime.fromisoformat(r["predicted_at"]), cutoff)
    ]
    chosen = pre if len(pre) <= pre_cutoff_census_max else rng.sample(pre, pre_cutoff_sample)
    for r in chosen:
        add(r, REASON_PRE_CUTOFF)

    return [{**entry, "queue_rank": rank} for rank, entry in enumerate(queue.values(), start=1)]
