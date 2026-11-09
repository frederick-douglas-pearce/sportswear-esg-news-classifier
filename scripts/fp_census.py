#!/usr/bin/env python
"""Second-opinion census of the FP negatives only the classifier decided (#155 PR1).

Sends each article with sufficient content whose FP training label source is ``fp_classifier``
to the labeler, under the production prompt, and appends what the reply decides to a JSONL file.
Reads the database and never writes to it: the replies are inputs to a human review (#155 PR2),
not labels.

Usage:
    # Cohort, its probability bands and months, and the worst-case cost; no API call:
    uv run python scripts/fp_census.py --dry-run

    # A first slice, then the rest (stops before spend would pass --max-cost, counted over every
    # record in the output file):
    uv run python scripts/fp_census.py --limit 3 --max-cost 1
    uv run python scripts/fp_census.py --max-cost 60

    # Summary and review queue, checked against the live cohort (reads the database); no API call:
    uv run python scripts/fp_census.py --queue

Resume: a re-run skips articles already recorded, except a failed call that got no reply (no
output tokens) and was not a refusal, which is called again; the latest record of an article is
the one read. A failure the model was billed for (a reply that did not parse, or one cut at
max_tokens) and a refusal are final, and are queued for the reviewer. No other model answers in
place of a refused call.

What the cap does not see: a call interrupted (Ctrl-C, a kill) after the API billed it and before
its record was written is not counted, so the cap can be passed by at most one call's cost per
interruption; a final fragment such an interruption leaves is ignored with a warning, removed
before the next record is written, and that article is called again. A call that timed out is recorded at $0 although the API may have
billed it (#264).

Exit codes: 0 done, 2 refused before any API call (or, for --queue, before writing), 3 stopped
before an article whose worst case would pass --max-cost, 4 stopped after consecutive failed
calls or after a call that cost more than its worst case.
"""

import argparse
import hashlib
import json
import logging
import math
import os
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
# Consecutive failed calls, of any kind, after which the run stops: an exhausted credit balance,
# a bad key, or a reply shape the parser cannot read fails every call the same way
MAX_CONSECUTIVE_FAILED = 3

EXIT_DONE = 0
EXIT_REFUSED = 2  # nothing was sent to the API (for --queue: nothing was written)
EXIT_STOPPED_AT_CAP = 3
EXIT_ABORTED = 4


class CensusFileError(ValueError):
    """The records file has a malformed line before its last."""


def default_output(model: str) -> Path:
    """Gitignored by ``data/fp_census_*``."""
    return DATA_DIR / f"fp_census_{model}.jsonl"


def queue_path(output: Path) -> Path:
    """Beside the records file (gitignored by ``data/fp_census_*`` at the default path)."""
    return output.with_name(f"{output.stem}_queue.jsonl")


def design_path(output: Path) -> Path:
    """Beside the records file (gitignored by ``data/fp_census_*`` at the default path)."""
    return output.with_name(f"{output.stem}_queue_design.json")


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
        # Ascending, keeping the last row seen per article: the reverse of
        # latest_prediction_subquery's order (created_at DESC NULLS LAST, id DESC, first row)
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


def _fragment(text: str) -> str | None:
    """The text after the last newline when it does not parse: what an interrupted write leaves."""
    if not text or text.endswith("\n"):
        return None
    tail = text[text.rfind("\n") + 1 :]
    try:
        json.loads(tail)
    except json.JSONDecodeError:
        return tail
    return None


def read_lines(path: Path) -> list[dict]:
    """The records in the file. A final fragment with no newline that does not parse (an
    interrupted write) is ignored with a warning; any other malformed line raises
    CensusFileError."""
    if not path.exists():
        return []
    text = path.read_text()
    fragment = _fragment(text)
    if fragment is not None:
        logger.warning(f"Ignoring a malformed last line in {path} (an interrupted write?): {fragment[:80]!r}")
        text = text[: len(text) - len(fragment)]
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise CensusFileError(f"Malformed line {number} of {path}: {e}") from e
    return records


def trim_fragment(path: Path) -> None:
    """Drop a final fragment an interrupted write left, so the next record starts a clean line."""
    if not path.exists():
        return
    text = path.read_text()
    fragment = _fragment(text)
    if fragment is not None:
        with path.open("r+b") as f:
            f.truncate(len(text.encode()) - len(fragment.encode()))
            f.flush()
            os.fsync(f.fileno())


def append_line(path: Path, record: dict) -> None:
    """Append one record and fsync it; a file left without a final newline gets one first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0, os.SEEK_END)
        prefix = b""
        if f.tell():
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = b"\n"
        f.write(prefix + (json.dumps(record) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def total_spend(lines: list[dict]) -> float:
    """What every recorded call cost, retried calls included: the figure the cap counts."""
    return sum(line.get("cost_usd", 0.0) for line in lines)


def call_chars(row: census.CohortRow) -> int:
    """The characters a call adds to the prompt template: the content and the fields substituted
    into the user prompt."""
    return len(row.content) + len(row.title) + len(", ".join(row.brands)) + len(row.source_name or "") + 32


def print_cohort(cohort: list[census.CohortRow], todo: list[census.CohortRow], per_call: dict) -> None:
    bands = Counter(census.probability_band(r.probability) for r in cohort)
    months = Counter(
        census.as_utc(r.predicted_at).strftime("%Y-%m") if r.predicted_at else census.UNKNOWN for r in cohort
    )
    print(f"Cohort (label source fp_classifier): {len(cohort)}")
    print("  by probability band: " + ", ".join(f"{b} {bands.get(b, 0)}" for b in census.BANDS))
    print("  by month of the FP skip (UTC): " + ", ".join(f"{m} {n}" for m, n in sorted(months.items())))
    print(f"To call this invocation: {len(todo)}")
    print(f"Worst case for them: ${sum(per_call.values()):.2f}")


def print_summary(summary: dict) -> None:
    def rate(r: dict) -> str:
        if r["disagreement_rate"] is None:
            return f"{r['rows']} rows, none judged"
        text = f"{r['disagreements']}/{r['judged']} disagree ({r['disagreement_rate']:.1%})"
        if r["upper_bound_if_none"] is not None:
            text += f", rule-of-three 95% bound {r['upper_bound_if_none']:.1%}"
        return text

    print(f"Records: {summary['rows']}  spend ${summary['cost_usd']:.2f} (every call, retries included)")
    print("Verdicts: " + ", ".join(f"{v} {n}" for v, n in summary["verdicts"].items()))
    print(f"  refusals (counted in failed): {summary['refusals']}")
    print(f"Overall: {rate(summary['overall'])}")
    for band, r in summary["by_band"].items():
        print(f"  band {band}: {rate(r)}")
    for month, r in summary["by_month"].items():
        print(f"  month {month}: {rate(r)}")
    sides = summary["by_cutoff_side"]
    print(f"  before {summary['cutoff']}: {rate(sides[census.BEFORE])}")
    print(f"  from {summary['cutoff']}: {rate(sides[census.AFTER])}")
    if sides[census.UNKNOWN]["rows"]:
        print(f"  no prediction time: {rate(sides[census.UNKNOWN])}")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_queue(args: argparse.Namespace, output: Path) -> int:
    """Summarize the records of the live cohort and write the review queue and its design."""
    queue_file, design_file = queue_path(output), design_path(output)
    existing = [p for p in (queue_file, design_file) if p.exists()]
    if existing and not args.force:
        logger.error(f"{', '.join(map(str, existing))} exists; pass --force to replace it")
        return EXIT_REFUSED
    try:
        lines = read_lines(output)
    except CensusFileError as e:
        logger.error(str(e))
        return EXIT_REFUSED
    latest = census.latest_records(lines)
    if not latest:
        logger.error(f"No census records in {output}")
        return EXIT_REFUSED

    cohort = read_cohort()
    cohort_ids = {r.article_id for r in cohort}
    sendable = {r.article_id for r in cohort if has_sufficient_content(r.content)}
    insufficient = len(cohort_ids) - len(sendable)
    left_cohort = sorted(set(latest) - cohort_ids)
    missing = sorted(i for i in sendable if census.needs_call(latest.get(i)))
    print(f"Cohort: {len(cohort_ids)}; records for {len(sendable) - len(missing)} of {len(sendable)} with sufficient content")
    if insufficient:
        print(f"  left out for insufficient content (never sent): {insufficient}")
    if left_cohort:
        print(f"  records dropped, article no longer in the cohort: {len(left_cohort)}")
    if missing:
        message = f"{len(missing)} cohort articles have no usable record yet (never called, or a retryable failure)"
        if not args.partial:
            logger.error(f"{message}; run the census, or pass --partial to queue what is recorded")
            return EXIT_REFUSED
        logger.warning(f"{message}; --partial: queueing what is recorded")

    records = [r for article_id, r in latest.items() if article_id in cohort_ids and article_id not in missing]
    summary = census.summarize(records, cutoff=args.cutoff, spend_usd=total_spend(lines))
    print_summary(summary)
    queue, design = census.build_review_queue(
        records,
        cutoff=args.cutoff,
        agreements_per_band=args.agreements_per_band,
        seed=args.seed,
    )
    design.update(
        {
            "records_file": str(output),
            "records_sha256": sha256_of(output),
            "cohort": len(cohort_ids),
            "insufficient_content": insufficient,
            "dropped_not_in_cohort": len(left_cohort),
            "missing": len(missing),
            "partial": bool(missing),
            "summary": summary,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    queue_file.write_text("".join(json.dumps(entry) + "\n" for entry in queue))
    design_file.write_text(json.dumps(design, indent=2) + "\n")
    reasons = Counter(reason for entry in queue for reason in entry["queue_reasons"])
    print(f"\nReview queue: {len(queue)} rows -> {queue_file}")
    print("  " + ", ".join(f"{reason} {n}" for reason, n in reasons.items()))
    print(f"Design: {design_file}")
    return EXIT_DONE


def build_labeler(args: argparse.Namespace, version: str) -> ArticleLabeler:
    try:
        labeler = ArticleLabeler(model=args.model, effort=args.effort, prompt_version=version)
    except ValueError as e:  # no API key, or an effort level the model's effort setting refuses
        raise LabelerMismatchError(f"Cannot build the labeler: {e}; refusing to run") from e
    check_labeler(labeler, version, model=args.model, effort=args.effort)
    return labeler


def run(args: argparse.Namespace, output: Path) -> int:
    if not args.dry_run:
        if args.max_cost is None:
            logger.error("--max-cost is required for a run (use --dry-run to see the bounds)")
            return EXIT_REFUSED
        if not (math.isfinite(args.max_cost) and args.max_cost > 0):
            logger.error(f"--max-cost must be a positive number, got {args.max_cost}")
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
    # The effort a call is actually sent: none for a model that takes none
    effort = args.effort if spec.accepts_effort else None

    try:
        lines = read_lines(output)
    except CensusFileError as e:
        logger.error(str(e))
        return EXIT_REFUSED
    cohort = read_cohort()
    latest = census.latest_records(lines)
    sendable = [r for r in cohort if has_sufficient_content(r.content)]
    todo = [r for r in sendable if census.needs_call(latest.get(r.article_id))]
    if args.limit:
        todo = todo[: args.limit]
    max_tokens = spec.default_max_tokens
    per_call = {
        r.article_id: worst_case_call_cost_usd(prompt_chars, call_chars(r), max_tokens, args.model) for r in todo
    }
    spent = total_spend(lines)

    print_cohort(cohort, todo, per_call)
    if len(sendable) < len(cohort):
        print(f"Left out for insufficient content (never sent): {len(cohort) - len(sendable)}")
    print(
        f"Settings: {version} on {args.model}, effort {effort or 'not sent'}, max_tokens {max_tokens}; "
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
    trim_fragment(output)

    failed_in_a_row = 0
    verdicts: Counter = Counter()

    def stop(code: int, message: str) -> int:
        (logger.warning if code == EXIT_STOPPED_AT_CAP else logger.error)(message)
        print(f"Verdicts this invocation: {dict(verdicts)}")
        return code

    for done, row in enumerate(todo):
        worst = per_call[row.article_id]
        if spent + worst > args.max_cost:
            return stop(
                EXIT_STOPPED_AT_CAP,
                f"Stopped after {done} calls: the next call's worst case would pass "
                f"--max-cost ${args.max_cost:.2f} (spend ${spent:.2f})",
            )
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
            effort=effort,
            prompt_version=version,
            cost_usd=cost,
            worst_case_usd=worst,
            labeled_at=datetime.now(timezone.utc),
        )
        append_line(output, record)
        spent += cost
        verdicts[record["verdict"]] += 1
        if cost > worst:
            return stop(
                EXIT_ABORTED,
                f"Stopped: article {row.article_id} cost ${cost:.4f}, more than its worst case "
                f"${worst:.4f}; the bound is not one for {args.model}",
            )
        failed_in_a_row = failed_in_a_row + 1 if record["verdict"] == census.VERDICT_FAILED else 0
        if failed_in_a_row >= MAX_CONSECUTIVE_FAILED:
            return stop(
                EXIT_ABORTED,
                f"Stopped: {failed_in_a_row} consecutive failed calls (last: {result.error_type}: {result.error})",
            )

    print(f"\nDone: {len(todo)} calls, spend ${spent:.2f} in all")
    print(f"Verdicts this invocation: {dict(verdicts)}")
    return EXIT_DONE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="Cohort and cost bounds; no API call, no write")
    p.add_argument("--queue", action="store_true", help="Summarize the records and write the review queue")
    p.add_argument("--partial", action="store_true", help="--queue: queue what is recorded when cohort rows have no usable record")
    p.add_argument("--force", action="store_true", help="--queue: replace an existing queue and design file")
    p.add_argument("--max-cost", type=float, help="Spend cap in USD over every record in the output; required to run")
    p.add_argument("--limit", type=int, help="Call at most this many articles")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Labeling model (default {DEFAULT_MODEL})")
    p.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        help=f"output_config.effort, sent only to a model that takes it (default {DEFAULT_EFFORT})",
    )
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
