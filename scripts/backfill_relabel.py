#!/usr/bin/env python3
"""Relabel the #192 backfill cohort under the production prompt (D039).

The cohort (``select_backfill_cohort`` in ``src/data_collection/provenance.py``) is the rows whose
status the labeling LLM decided, or could decide, with no stored judgment for each brand:
unjudged skips at a reason a reply writes, LLM-decided ``false_positive`` rows, and judged skips
written since the production version's first run. Human-ruled rows, rows a human may have moved
by hand, and rows with human-reviewed labels are left out. Each article goes through
``LabelingPipeline.label_articles`` with the FP pre-filter, novelty scoring and title
deduplication bypassed, so ``dispose_reply`` decides every status and no status is set here.

Runs are tagged ``192-backfill`` (``daily_labeling`` leaves tagged runs out of its metrics), and
each run's config records every article's status, reason and arm from before the relabel. The
cohort is derived again on each invocation: an article whose call failed transiently is still in
it, and every other outcome leaves it.

Usage:
    # Cohort, exclusions and cost bounds; no API call, no write:
    uv run python scripts/backfill_relabel.py --dry-run

    # A first slice, then the rest (stops before spend would pass --max-cost):
    uv run python scripts/backfill_relabel.py --limit 50 --max-cost 5
    uv run python scripts/backfill_relabel.py --max-cost 30

    # What the tagged runs did so far:
    uv run python scripts/backfill_relabel.py --report
"""

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func  # noqa: E402

from src.data_collection import provenance  # noqa: E402
from src.data_collection.database import db  # noqa: E402
from src.data_collection.models import Article, BrandJudgment, LabelingRun  # noqa: E402
from src.labeling.config import labeling_settings, worst_case_call_cost_usd  # noqa: E402
from src.labeling.labeler import (  # noqa: E402
    ArticleLabeler,
    LabelerMismatchError,
    check_labeler,
    prompt_input_chars,
)
from src.labeling.pipeline import LabelingPipeline  # noqa: E402
from src.labeling.prompt_manager import PromptManager  # noqa: E402

logger = logging.getLogger("backfill_relabel")

RUN_TAG = "192-backfill"
DEFAULT_BATCH_SIZE = 25

# Exit codes
EXIT_DONE = 0
EXIT_STOPPED_AT_CAP = 1  # some batches ran, the next would pass --max-cost
EXIT_REFUSED = 2  # nothing was sent to the API


class BackfillRefused(RuntimeError):
    """The backfill cannot run as asked; nothing has been sent to the API."""


@dataclass(frozen=True)
class RunSettings:
    """What the production version was evaluated on: the registry's model_recommendations."""

    version: str
    model: str
    effort: str | None
    max_tokens: int


@dataclass
class Spend:
    """Spend so far over the tagged runs."""

    usd: float
    # (run id, worst case counted) for each run with no completed_at, so no recorded cost
    unfinished: list[tuple[str, float]]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def registry_settings(manager: PromptManager) -> RunSettings:
    """The production version and its registry model_recommendations, the one source for them.

    Refuses when LABELING_PROMPT_VERSION names another version, since the cron would then label
    under a prompt the backfill does not.
    """
    version = manager.get_production_version()
    configured = labeling_settings.prompt_version
    if configured and configured != version:
        raise BackfillRefused(
            f"LABELING_PROMPT_VERSION={configured} is not the registry's production version "
            f"{version}; refusing to run"
        )
    recs = manager.get_version_info(version).get("model_recommendations") or {}
    if not recs.get("model") or not recs.get("max_tokens"):
        raise BackfillRefused(
            f"Registry entry {version} has no model_recommendations model and max_tokens; "
            "refusing to run"
        )
    return RunSettings(version, recs["model"], recs.get("effort"), int(recs["max_tokens"]))


def build_labeler(version: str) -> ArticleLabeler:
    """The labeler the cron would build: model and effort from the environment."""
    return ArticleLabeler(prompt_version=version)


def first_run_started_at(session, prompt_version: str) -> datetime | None:
    """When the first untagged run under the version started: arm J's cutoff."""
    return (
        session.query(func.min(LabelingRun.started_at))
        .filter(LabelingRun.prompt_version == prompt_version)
        .filter(LabelingRun.config["run_tag"].as_string().is_(None))
        .scalar()
    )


def tagged_runs(session, tag: str = RUN_TAG) -> list[LabelingRun]:
    return (
        session.query(LabelingRun)
        .filter(LabelingRun.config["run_tag"].as_string() == tag)
        .order_by(LabelingRun.started_at)
        .all()
    )


def spend_of(runs: list[LabelingRun]) -> Spend:
    """Recorded cost of the finished runs, plus each unfinished run at its batch's worst case.

    A run with no completed_at was cut off and recorded no cost, but its calls may have been
    billed. Its worst case is what its config recorded before the batch ran.
    """
    spend = Spend(0.0, [])
    for run in runs:
        if run.completed_at is not None:
            spend.usd += run.estimated_cost_usd or 0.0
            continue
        worst = ((run.config or {}).get("run_config") or {}).get("batch_worst_case_usd")
        if worst is None:
            raise BackfillRefused(
                f"Tagged run {run.id} is unfinished and recorded no worst case; refusing to run"
            )
        spend.usd += worst
        spend.unfinished.append((str(run.id), worst))
    return spend


def observed_cost_per_call(session, settings: RunSettings) -> float | None:
    """Recorded cost per LLM call over finished untagged runs of the version on its model."""
    cost, calls = (
        session.query(func.sum(LabelingRun.estimated_cost_usd), func.sum(LabelingRun.llm_calls_made))
        .filter(LabelingRun.prompt_version == settings.version)
        .filter(LabelingRun.completed_at.isnot(None))
        .filter(LabelingRun.llm_calls_made > 0)
        .filter(LabelingRun.config["model"].as_string() == settings.model)
        .filter(LabelingRun.config["run_tag"].as_string().is_(None))
        .one()
    )
    return float(cost) / int(calls) if calls else None


def batches(ids: list[str], size: int) -> list[list[str]]:
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def print_cohort(cohort: provenance.BackfillCohort, ids: list[str], per_call: dict[str, float]) -> None:
    print(f"\n{'=' * 60}\nBACKFILL COHORT ({RUN_TAG})\n{'=' * 60}")
    print(f"Arm J cutoff (first production-version run): {cohort.judged_skip_cutoff}")
    print(f"U/J skipped_at bound: {cohort.bound}")
    for arm in provenance.BACKFILL_ARMS:
        print(f"\n{arm}: {len(cohort.selected[arm])}")
        for (a, key), n in sorted(cohort.by_reason.items(), key=lambda kv: str(kv[0])):
            if a == arm:
                print(f"  {key}: {n}")
        months = sorted((m, n) for (a, m), n in cohort.by_month.items() if a == arm)
        if months:
            print("  by month collected: " + ", ".join(f"{m} {n}" for m, n in months))
    print("\nLeft out:")
    for reason, n in sorted(cohort.excluded.items()):
        print(f"  {reason}: {n}")
    print(f"Selected articles with non-human brand_labels rows (left as they are): {cohort.stale_labels}")
    if per_call:
        values = list(per_call.values())
        print(
            f"\nThis invocation: {len(ids)} articles; worst case per call "
            f"${sum(values) / len(values):.4f} mean, ${max(values):.4f} max; "
            f"total ${sum(values):.2f}"
        )


def run_batches(
    pipeline: LabelingPipeline,
    ids: list[str],
    cohort: provenance.BackfillCohort,
    per_call: dict[str, float],
    batch_size: int,
    max_cost: float,
) -> int:
    """Label the ids batch by batch, stopping before a batch whose worst case would pass the cap."""
    for n, batch in enumerate(batches(ids, batch_size)):
        worst = sum(per_call[i] for i in batch)
        with db.get_session() as session:
            spend = spend_of(tagged_runs(session))
        if spend.usd + worst > max_cost:
            message = (
                f"Spend so far ${spend.usd:.2f} + next batch's worst case ${worst:.2f} exceeds "
                f"--max-cost ${max_cost:.2f}"
            )
            if n == 0:
                raise BackfillRefused(f"{message}; refusing to run")
            logger.warning(f"{message}; stopping after {n} batches")
            return EXIT_STOPPED_AT_CAP
        run_config = {
            "judged_skip_cutoff": cohort.judged_skip_cutoff.isoformat()
            if cohort.judged_skip_cutoff
            else None,
            "bound": cohort.bound.isoformat(),
            "batch_worst_case_usd": worst,
            # Written with the run row, before any article in it is processed
            "prior": {i: cohort.prior[i] for i in batch},
        }
        logger.info(f"Batch {n + 1}: {len(batch)} articles, worst case ${worst:.2f}")
        pipeline.label_articles(
            article_ids=[UUID(i) for i in batch],
            prefilter=False,
            dedupe_titles=False,
            run_tag=RUN_TAG,
            run_config=run_config,
        )
    return EXIT_DONE


def backfill_report(session, tag: str = RUN_TAG) -> dict:
    """What the tagged runs did: prior to current status and source by arm, content types, failed
    outcomes by arm, and the articles still unjudged."""
    runs = tagged_runs(session, tag)
    prior: dict[str, list] = {}
    for run in runs:
        for article_id, values in (((run.config or {}).get("run_config") or {}).get("prior") or {}).items():
            # A retried article's first record is its status from before the backfill
            prior.setdefault(article_id, values)
    report = {
        "runs": len(runs),
        "transitions": Counter(),
        "content_types": Counter(),
        "failed_by_arm": Counter(),
        "still_unjudged": [],
    }
    if not prior:
        return report
    latest = provenance.latest_prediction_subquery("fp")
    rows = (
        session.query(Article.id, Article.labeling_status, Article.labeling_error, latest.c.action_taken, latest.c.skip_reason)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(Article.id.in_([UUID(i) for i in prior]))
        .all()
    )
    for article_id, status, reason, fp_action, fp_skip_reason in rows:
        old_status, _, arm = prior[str(article_id)]
        if status in (*provenance.FP_POSITIVE_STATUSES, provenance.FP_NEGATIVE_STATUS):
            source = provenance.derive_label_source(
                status, reason, fp_action, fp_skip_reason, article_id=article_id
            )
        else:
            source = None
        report["transitions"][(arm, old_status, status, source)] += 1
        if status == "failed":
            report["failed_by_arm"][arm] += 1
        if source == provenance.UNJUDGED:
            report["still_unjudged"].append(str(article_id))
    run_ids = [run.id for run in runs]
    report["content_types"] = Counter(
        dict(
            session.query(BrandJudgment.content_type, func.count())
            .filter(BrandJudgment.labeling_run_id.in_(run_ids))
            .group_by(BrandJudgment.content_type)
            .all()
        )
    )
    return report


def print_report(report: dict) -> None:
    print(f"\n{'=' * 60}\nBACKFILL REPORT ({RUN_TAG}, {report['runs']} runs)\n{'=' * 60}")
    print("arm: prior status -> status (source): articles")
    for (arm, old, new, source), n in sorted(report["transitions"].items(), key=lambda kv: str(kv[0])):
        print(f"  {arm}: {old} -> {new} ({source}): {n}")
    print("Brand judgments by content type:")
    for content_type, n in sorted(report["content_types"].items()):
        print(f"  {content_type}: {n}")
    print("Failed by arm: " + (", ".join(f"{a} {n}" for a, n in sorted(report["failed_by_arm"].items())) or "none"))
    print(f"Still unjudged (left out of later passes): {len(report['still_unjudged'])}")
    for article_id in report["still_unjudged"]:
        print(f"  {article_id}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="Cohort and cost bounds; no API call, no write")
    p.add_argument("--report", action="store_true", help="Print what the tagged runs did, and exit")
    p.add_argument("--limit", type=int, help="Label at most this many articles")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Articles per labeling run")
    p.add_argument("--max-cost", type=float, help="Spend cap in USD over every tagged run; required to run")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.report:
        with db.get_session() as session:
            print_report(backfill_report(session))
        return EXIT_DONE
    if not args.dry_run and args.max_cost is None:
        logger.error("--max-cost is required for a run (use --dry-run to see the bounds)")
        return EXIT_REFUSED
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        logger.error("--batch-size and --limit must be at least 1")
        return EXIT_REFUSED

    manager = PromptManager()
    started = datetime.now(timezone.utc)
    try:
        settings = registry_settings(manager)
        labeler = build_labeler(settings.version)
        check_labeler(
            labeler,
            settings.version,
            model=settings.model,
            effort=settings.effort,
            max_tokens=settings.max_tokens,
        )
        prompt_chars = prompt_input_chars(manager, settings.version)
        with db.get_session() as session:
            cutoff = first_run_started_at(session, settings.version)
            runs = tagged_runs(session)
            # The first tagged run's start, so the cohort does not move between invocations
            bound = runs[0].started_at if runs else started
            cohort = provenance.select_backfill_cohort(session, judged_skip_cutoff=cutoff, bound=bound)
            observed = observed_cost_per_call(session, settings)
            spend = spend_of(runs)
    except (BackfillRefused, LabelerMismatchError) as e:
        logger.error(str(e))
        return EXIT_REFUSED

    ids = cohort.ids[: args.limit] if args.limit else cohort.ids
    per_call = {
        i: worst_case_call_cost_usd(prompt_chars, cohort.content_chars[i], settings.max_tokens, settings.model)
        for i in ids
    }
    print_cohort(cohort, ids, per_call)
    print(
        f"Settings: {settings.version} on {settings.model}, effort {settings.effort}, "
        f"max_tokens {settings.max_tokens}; prompt {prompt_chars} chars"
    )
    print(f"Tagged spend so far: ${spend.usd:.2f}")
    for run_id, worst in spend.unfinished:
        print(f"  unfinished run {run_id}: counted at its worst case ${worst:.2f}")
    if observed is None:
        logger.warning(f"No finished run of {settings.version} on {settings.model} to compare the bound with")
    else:
        print(f"Observed cost per call ({settings.version} on {settings.model}): ${observed:.4f}")
        if per_call and observed > sum(per_call.values()) / len(per_call):
            logger.error(
                f"Observed cost per call ${observed:.4f} exceeds the mean worst case per call; "
                "the bound is not one. Refusing to run"
            )
            return EXIT_REFUSED

    if args.dry_run:
        print("\n[DRY RUN] No API call, nothing written")
        return EXIT_DONE
    if not ids:
        print("\nNothing to relabel")
        return EXIT_DONE
    first_worst = sum(per_call[i] for i in batches(ids, args.batch_size)[0])
    if spend.usd + first_worst > args.max_cost:
        logger.error(
            f"Spend so far ${spend.usd:.2f} + the first batch's worst case ${first_worst:.2f} "
            f"exceeds --max-cost ${args.max_cost:.2f}; refusing to run"
        )
        return EXIT_REFUSED

    pipeline = LabelingPipeline(labeler=labeler)
    try:
        code = run_batches(pipeline, ids, cohort, per_call, args.batch_size, args.max_cost)
    except BackfillRefused as e:
        logger.error(str(e))
        return EXIT_REFUSED
    with db.get_session() as session:
        print_report(backfill_report(session))
    return code


if __name__ == "__main__":
    sys.exit(main())
