#!/usr/bin/env python3
"""Move stored articles whose text names none of their brands to 'unlabelable' (#227, D044).

The scraper now skips a scrape whose content names none of the article's brands
(``SCRAPE_NO_BRAND_ERROR``). This moves stored rows scraped as 'success' before that, which were
labeled, or queued for labeling, with content that does not name the brand, to the same state:
``scrape_status='skipped'``, that ``scrape_error``, ``labeling_status='unlabelable'``. Their
content is kept, and their brand labels are deleted (a row with a human's label is not moved). The
cohort is derived on every run (``provenance.select_no_brand_content``). Human rulings, human
labels, possible hand moves, eval-case articles and other statuses are left out and counted by
reason; possible hand moves and eval-case articles are also listed by id.

Usage:
    # What would move, by status and brand, and what is left out; writes nothing
    python scripts/triage_no_brand_content.py --dry-run

    # Move them (back up the database first: the manifest records each row's prior status,
    # reason and scrape fields and the number of labels deleted, not the labels themselves,
    # which only the backup restores). The manifest is written before the transaction commits,
    # and an existing manifest path is refused.
    python scripts/triage_no_brand_content.py

Run it outside the labeling cron's window. A row whose status or reason changed since selection,
or that gained a human label, is not moved and is counted; but the labeling pipeline's status
write is unconditional, so a labeling run that selected a row before this commits can still
label it afterwards. Check after the run with the scrape_error query in
queries/collection_queries.sql: every row with this scrape_error should be unlabelable.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import exists

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import provenance
from src.data_collection.database import db, scrape_failure_reason
from src.data_collection.models import SCRAPE_NO_BRAND_ERROR, Article, BrandLabel

EVAL_CASES_GLOB = "prompts/labeling/eval_cases/*.json"
TRIAGE_STATUS = "unlabelable"
TRIAGE_SCRAPE_STATUS = "skipped"


def eval_case_ids(root: Path = Path(__file__).parent.parent) -> set[str]:
    """Every ``article_id`` in the eval case files."""
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "article_id" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in sorted(root.glob(EVAL_CASES_GLOB)):
        walk(json.loads(path.read_text()))
    return found


def print_cohort(cohort: provenance.NoBrandCohort) -> None:
    print(f"To move: {len(cohort.selected)}")
    for status, count in sorted(cohort.by_status.items()):
        print(f"  {status}: {count}")
    print("By brand:")
    for brand, count in cohort.by_brand.most_common():
        print(f"  {brand}: {count}")
    print(f"Left out: {sum(cohort.excluded.values())}")
    for why, count in sorted(cohort.excluded.items()):
        print(f"  {why}: {count}")
    hand_moves = sorted(
        i for i, why in cohort.left_out.items() if why == provenance.EXCLUDED_POSSIBLE_HAND_MOVE
    )
    if hand_moves:
        print("Possible hand moves (left as stored; rule on them with fix_label.py):")
        for article_id in hand_moves:
            print(f"  {article_id}")
    in_cases = sorted(
        i for i, why in cohort.left_out.items() if why == provenance.EXCLUDED_EVAL_CASE
    )
    if in_cases:
        print("Eval-case articles (left as stored; the case's verdict is the owner's):")
        for article_id in in_cases:
            print(f"  {article_id}")


def apply_triage(session, cohort: provenance.NoBrandCohort, manifest_path: Path) -> dict[str, int]:
    """Move the cohort, in the caller's transaction, and write the manifest before returning.

    Each update is guarded on the row still having the status and reason it had at selection and
    no brand label a human wrote or reviewed; a row that fails the guard is not moved, keeps its
    labels, gets no manifest line, and is counted as ``changed``. The manifest holds one JSON line
    per moved row with its prior status, reason and scrape fields and the number of brand labels
    deleted (not the labels: restore those from the backup). It is created here, refusing an
    existing file, and flushed to disk, so the caller's commit follows it.
    """
    # Imported here: the labeling package loads the chunker and tokenizer, which --dry-run needs not
    from src.labeling.database import LabelingDatabase

    labeling_db = LabelingDatabase()
    reason = scrape_failure_reason(TRIAGE_SCRAPE_STATUS, SCRAPE_NO_BRAND_ERROR)
    records, counts = [], {"moved": 0, "changed": 0, "labels_deleted": 0}
    for article_id, (status, error, scrape_status, scrape_error) in cohort.selected.items():
        uid = UUID(article_id)
        updated = (
            session.query(Article)
            .filter(
                Article.id == uid,
                Article.labeling_status == status,
                Article.labeling_error.is_not_distinct_from(error),
                ~exists().where(
                    BrandLabel.article_id == Article.id, provenance.human_label_criterion()
                ),
            )
            .update(
                {
                    Article.labeling_status: TRIAGE_STATUS,
                    Article.labeling_error: reason,
                    Article.scrape_status: TRIAGE_SCRAPE_STATUS,
                    Article.scrape_error: SCRAPE_NO_BRAND_ERROR,
                },
                synchronize_session=False,
            )
        )
        if not updated:
            counts["changed"] += 1
            continue
        deleted = labeling_db.delete_llm_brand_labels(session, uid)
        counts["moved"] += 1
        counts["labels_deleted"] += deleted
        records.append({
            "article_id": article_id,
            "labeling_status": status,
            "labeling_error": error,
            "scrape_status": scrape_status,
            "scrape_error": scrape_error,
            "brand_labels_deleted": deleted,
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return counts


def default_manifest_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).parent.parent / "data" / f"no_brand_triage_{stamp}.jsonl"


def main(argv: list[str] | None = None, database=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print the cohort; write nothing")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Where to write each moved row's prior state (default: data/no_brand_triage_<UTC>.jsonl)",
    )
    args = parser.parse_args(argv)
    database = database or db
    manifest = args.manifest or default_manifest_path()
    if not args.dry_run and manifest.exists():
        print(f"Manifest {manifest} exists; refusing to overwrite it. Pass another --manifest.")
        return 2

    written = False
    try:
        with database.get_session() as session:
            cohort = provenance.select_no_brand_content(session, eval_case_ids())
            print_cohort(cohort)
            if args.dry_run:
                session.rollback()
                print("Dry run: nothing written.")
                return 0
            if not cohort.selected:
                print("Nothing to move.")
                return 0
            counts = apply_triage(session, cohort, manifest)
            written = True
    except Exception:
        if written:
            print(f"Manifest {manifest} describes a transaction that did not commit.")
        raise
    print(
        f"Moved {counts['moved']} to {TRIAGE_STATUS}; {counts['changed']} changed since selection "
        f"and left as stored; {counts['labels_deleted']} brand labels deleted. Manifest: {manifest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
