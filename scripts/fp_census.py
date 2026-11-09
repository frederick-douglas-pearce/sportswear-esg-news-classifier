#!/usr/bin/env python
"""Second-opinion census of the FP negatives only the classifier decided (#155 PR1).

Sends each article whose FP training label source is ``fp_classifier`` to the labeler once,
under the production prompt, and appends what the reply decides to a JSONL file. Reads the
database and never writes to it: the replies are inputs to a human review (#155 PR2), not labels.

Usage:
    # Cohort, its probability bands and months, and the worst-case cost; no API call:
    uv run python scripts/fp_census.py --dry-run

    # A first slice, then the rest (stops before spend would pass --max-cost, counted over every
    # record in the output file):
    uv run python scripts/fp_census.py --limit 3 --max-cost 1
    uv run python scripts/fp_census.py --max-cost 60

    # Summary and review queue from the records; no API call, no database:
    uv run python scripts/fp_census.py --queue

A re-run skips articles already recorded, except a failed call that was not a refusal, which is
retried; the latest record of an article is the one read. A refusal is recorded as failed and no
other model is asked in its place.

Exit codes: 0 done, 2 refused before any API call, 3 stopped before an article whose worst case
would pass --max-cost, 4 stopped after consecutive failed calls that got no reply.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select  # noqa: E402

from src.data_collection import provenance  # noqa: E402
from src.data_collection.database import db  # noqa: E402
from src.data_collection.models import ClassifierPrediction  # noqa: E402
from src.evaluation import fp_census as census  # noqa: E402
from src.labeling.article_input import article_content, has_sufficient_content  # noqa: E402
from src.labeling.config import (  # noqa: E402
    labeling_model_spec,
    llm_cost_usd,
    worst_case_call_cost_usd,
)
from src.labeling.labeler import (  # noqa: E402
    ArticleLabeler,
    LabelerMismatchError,
    check_labeler,
    prompt_input_chars,
)
from src.labeling.prompt_manager import PromptManager  # noqa: E402

logger = logging.getLogger("fp_census")

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "medium"
DATA_DIR = Path(__file__).parent.parent / "data"
# Consecutive failed calls with no reply (no output tokens) after which the run stops: an
# exhausted credit balance or a bad key fails every call the same way
MAX_CONSECUTIVE_NO_REPLY = 3

EXIT_DONE = 0
EXIT_REFUSED = 2  # nothing was sent to the API
EXIT_STOPPED_AT_CAP = 3
EXIT_ABORTED = 4


def default_output(model: str) -> Path:
    """Gitignored by ``data/*.jsonl``."""
    return DATA_DIR / f"fp_census_{model}.jsonl"


def queue_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_queue.jsonl")


def load_cohort(session) -> list[census.CohortRow]:
    """The FP export's rows whose label source is ``fp_classifier``, each with its latest FP
    prediction's probability, threshold and time, ordered by article id. Read only."""
    rows = provenance.fp_export_query(session).all()
    verified = provenance.human_verified_ids((str(a.id), a.labeling_status) for a, *_ in rows)
    articles = [
        article
        for article, action, skip_reason, _ in rows
        if provenance.derive_label_source(
            article.labeling_status,
            article.labeling_error,
            action,
            skip_reason,
            human_verified=str(article.id) in verified,
            article_id=article.id,
        )
        == provenance.FP_CLASSIFIER
    ]
    latest = {}
    if articles:
        # Ascending, so the last prediction seen per article is the latest, with ties broken as
        # latest_prediction_subquery breaks them
        predictions = session.execute(
            select(
                ClassifierPrediction.article_id,
                ClassifierPrediction.probability,
                ClassifierPrediction.threshold_used,
                ClassifierPrediction.created_at,
            )
            .where(
                ClassifierPrediction.classifier_type == "fp",
                ClassifierPrediction.article_id.in_([a.id for a in articles]),
            )
            .order_by(
                ClassifierPrediction.created_at.asc().nulls_first(),
                ClassifierPrediction.id.asc(),
            )
        )
        for p in predictions:
            latest[p.article_id] = p
    cohort = []
    for article in articles:
        p = latest[article.id]  # a classifier-only negative has an FP prediction by definition
        cohort.append(
            census.CohortRow(
                article_id=str(article.id),
                title=article.title,
                content=article_content(article.full_content, article.description),
                brands=list(article.brands_mentioned or []),
                probability=p.probability,
                threshold=p.threshold_used,
                predicted_at=p.created_at,
                published_at=article.published_at,
                source_name=article.source_name,
            )
        )
    return cohort


def read_cohort() -> list[census.CohortRow]:
    """Read the cohort in a session that is rolled back, never committed."""
    session = db.SessionLocal()
    try:
        return load_cohort(session)
    finally:
        session.rollback()
        session.close()


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def print_cohort(cohort: list[census.CohortRow], todo: list[census.CohortRow], per_call: dict) -> None:
    bands = Counter(census.probability_band(r.probability) for r in cohort)
    months = Counter(
        census.as_utc(r.predicted_at).strftime("%Y-%m") if r.predicted_at else "unknown" for r in cohort
    )
    print(f"Cohort (label source fp_classifier): {len(cohort)}")
    print("  by probability band: " + ", ".join(f"{b} {bands.get(b, 0)}" for b in census.BANDS))
    print("  by month of the FP skip: " + ", ".join(f"{m} {n}" for m, n in sorted(months.items())))
    print(f"To call this invocation: {len(todo)}")
    print(f"Worst case for them: ${sum(per_call.values()):.2f}")


def print_summary(summary: dict) -> None:
    def rate(r: dict) -> str:
        if r["disagreement_rate"] is None:
            return f"{r['rows']} rows, none judged"
        text = f"{r['disagreements']}/{r['judged']} disagree ({r['disagreement_rate']:.1%})"
        if r["upper_bound_if_none"] is not None:
            text += f", 95% upper bound {r['upper_bound_if_none']:.1%}"
        return text

    print(f"Records: {summary['rows']}  spend ${summary['cost_usd']:.2f}")
    print("Verdicts: " + ", ".join(f"{v} {n}" for v, n in summary["verdicts"].items()))
    print(f"  refusals (counted in failed): {summary['refusals']}")
    print(f"Overall: {rate(summary['overall'])}")
    for band, r in summary["by_band"].items():
        print(f"  band {band}: {rate(r)}")
    for month, r in summary["by_month"].items():
        print(f"  month {month}: {rate(r)}")
    print(f"  before {summary['cutoff']}: {rate(summary['before_cutoff'])}")
    print(f"  from {summary['cutoff']}: {rate(summary['after_cutoff'])}")


def write_queue(args: argparse.Namespace, output: Path) -> int:
    records = list(census.latest_records(read_lines(output)).values())
    if not records:
        logger.error(f"No census records in {output}")
        return EXIT_REFUSED
    print_summary(census.summarize(records, cutoff=args.cutoff))
    queue = census.build_review_queue(
        records,
        cutoff=args.cutoff,
        agreements_per_band=args.agreements_per_band,
        seed=args.seed,
    )
    path = queue_path(output)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in queue))
    reasons = Counter(reason for entry in queue for reason in entry["queue_reasons"])
    print(f"\nReview queue: {len(queue)} rows -> {path}")
    print("  " + ", ".join(f"{reason} {n}" for reason, n in reasons.items()))
    return EXIT_DONE


def build_labeler(args: argparse.Namespace, version: str) -> ArticleLabeler:
    try:
        labeler = ArticleLabeler(model=args.model, effort=args.effort, prompt_version=version)
    except ValueError as e:  # no API key, a model with no spec, or an effort it does not take
        raise LabelerMismatchError(f"Cannot build the labeler: {e}; refusing to run") from e
    check_labeler(labeler, version, model=args.model, effort=args.effort)
    return labeler


def run(args: argparse.Namespace, output: Path) -> int:
    if not args.dry_run and args.max_cost is None:
        logger.error("--max-cost is required for a run (use --dry-run to see the bounds)")
        return EXIT_REFUSED
    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be at least 1")
        return EXIT_REFUSED

    manager = PromptManager()
    try:
        spec = labeling_model_spec(args.model)
        version = args.prompt_version or manager.get_production_version()
        prompt_chars = prompt_input_chars(manager, version)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"{e}; refusing to run")
        return EXIT_REFUSED

    cohort = read_cohort()
    lines = read_lines(output)
    latest = census.latest_records(lines)
    todo = [
        r for r in cohort if census.needs_call(latest.get(r.article_id)) and has_sufficient_content(r.content)
    ]
    insufficient = sum(not has_sufficient_content(r.content) for r in cohort)
    if args.limit:
        todo = todo[: args.limit]
    max_tokens = spec.default_max_tokens
    per_call = {
        r.article_id: worst_case_call_cost_usd(prompt_chars, len(r.content), max_tokens, args.model)
        for r in todo
    }
    spent = sum(line.get("cost_usd", 0.0) for line in lines)

    print_cohort(cohort, todo, per_call)
    if insufficient:
        print(f"Left out for insufficient content: {insufficient}")
    print(
        f"Settings: {version} on {args.model}, effort {args.effort}, max_tokens {max_tokens}; "
        f"prompt {prompt_chars} chars"
    )
    print(f"Output: {output} ({len(latest)} articles recorded, spend so far ${spent:.2f})")

    if args.dry_run:
        print("\n[DRY RUN] No API call, nothing written")
        return EXIT_DONE
    if not todo:
        print("\nNothing to call")
        return EXIT_DONE
    if spent + per_call[todo[0].article_id] > args.max_cost:
        logger.error(
            f"Spend so far ${spent:.2f} plus one call's worst case "
            f"${per_call[todo[0].article_id]:.2f} exceeds --max-cost ${args.max_cost:.2f}; refusing to run"
        )
        return EXIT_REFUSED
    try:
        labeler = build_labeler(args, version)
    except LabelerMismatchError as e:
        logger.error(str(e))
        return EXIT_REFUSED

    no_reply = 0
    verdicts: Counter = Counter()
    for done, row in enumerate(todo):
        if spent + per_call[row.article_id] > args.max_cost:
            logger.warning(
                f"Stopped after {done} calls: the next call's worst case would pass "
                f"--max-cost ${args.max_cost:.2f} (spend ${spent:.2f})"
            )
            print(f"Verdicts this invocation: {dict(verdicts)}")
            return EXIT_STOPPED_AT_CAP
        result = labeler.label_article(
            title=row.title,
            content=row.content,
            brands=row.brands,
            published_at=row.published_at,
            source_name=row.source_name,
        )
        cost = llm_cost_usd(result.input_tokens, result.output_tokens, args.model)
        record = census.census_record(
            row,
            result,
            model=args.model,
            effort=args.effort,
            prompt_version=version,
            cost_usd=cost,
            labeled_at=datetime.now(timezone.utc),
        )
        append_line(output, record)
        spent += cost
        verdicts[record["verdict"]] += 1
        no_reply = no_reply + 1 if record["verdict"] == census.VERDICT_FAILED and not result.output_tokens else 0
        if no_reply >= MAX_CONSECUTIVE_NO_REPLY:
            logger.error(
                f"Stopped: {no_reply} consecutive calls got no reply (last: {result.error_type}: "
                f"{result.error})"
            )
            print(f"Verdicts this invocation: {dict(verdicts)}")
            return EXIT_ABORTED

    print(f"\nDone: {len(todo)} calls, spend ${spent:.2f} in all")
    print(f"Verdicts this invocation: {dict(verdicts)}")
    return EXIT_DONE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="Cohort and cost bounds; no API call, no write")
    p.add_argument("--queue", action="store_true", help="Summarize the records and write the review queue")
    p.add_argument("--max-cost", type=float, help="Spend cap in USD over every record in the output; required to run")
    p.add_argument("--limit", type=int, help="Call at most this many articles")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Labeling model (default {DEFAULT_MODEL})")
    p.add_argument("--effort", default=DEFAULT_EFFORT, help=f"output_config.effort (default {DEFAULT_EFFORT})")
    p.add_argument("--prompt-version", help="Prompt version (default: the registry's production version)")
    p.add_argument("--output", type=Path, help="Records file (default data/fp_census_<model>.jsonl)")
    p.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        default=census.DEFAULT_CUTOFF,
        help=f"Owner-attested cutoff, YYYY-MM-DD (default {census.DEFAULT_CUTOFF.isoformat()})",
    )
    p.add_argument(
        "--agreements-per-band",
        type=int,
        default=census.DEFAULT_AGREEMENTS_PER_BAND,
        help=f"Agreements sampled per probability band (default {census.DEFAULT_AGREEMENTS_PER_BAND})",
    )
    p.add_argument("--seed", type=int, default=census.DEFAULT_SEED, help="Sampling seed")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output = args.output or default_output(args.model)
    if args.queue:
        return write_queue(args, output)
    return run(args, output)


if __name__ == "__main__":
    sys.exit(main())
