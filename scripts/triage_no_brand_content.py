#!/usr/bin/env python3
"""Move stored articles whose text names none of their brands to 'unlabelable' (#227, D044).

The scraper now skips a scrape whose content names none of the article's brands
(``SCRAPE_NO_BRAND_ERROR``). This moves stored rows scraped as 'success' before that, which the
labeling LLM decided from the headline, to the same state: ``scrape_status='skipped'``, that
``scrape_error``, ``labeling_status='unlabelable'``. Their content is kept, and their brand labels
are deleted unless a human wrote or reviewed them. The cohort is derived on every run
(``provenance.select_no_brand_content``); human rulings, human labels and possible hand moves are
left out and listed.

Usage:
    # What would move, by status and brand, and what is left out; writes nothing
    python scripts/triage_no_brand_content.py --dry-run

    # Move them (back up the database first). Each row's prior state is written to the
    # manifest before the transaction commits.
    python scripts/triage_no_brand_content.py

Run it outside the labeling cron's window: a row whose status changes between selection and write
is not moved, and is counted.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import provenance
from src.data_collection.database import db, scrape_failure_reason
from src.data_collection.models import SCRAPE_NO_BRAND_ERROR, Article

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


def print_cohort(cohort: provenance.NoBrandCohort, case_ids: set[str]) -> None:
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
    in_cases = sorted(set(cohort.selected) & case_ids)
    print(f"In eval case files: {len(in_cases)}")
    for article_id in in_cases:
        print(f"  {article_id}")


def apply_triage(session, cohort: provenance.NoBrandCohort, manifest_path: Path) -> dict[str, int]:
    """Move the cohort, in the caller's transaction, and write the manifest before returning.

    Each update is guarded on the row still having the status it had at selection; a row that
    changed is not moved and is counted as ``changed``. The manifest holds one JSON line per moved
    row with its prior state and the brand labels deleted, and is flushed to disk here, so the
    caller's commit follows it.
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
            .filter(Article.id == uid, Article.labeling_status == status)
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
    with manifest_path.open("w") as handle:
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

    with database.get_session() as session:
        cohort = provenance.select_no_brand_content(session)
        print_cohort(cohort, eval_case_ids())
        if args.dry_run:
            session.rollback()
            print("Dry run: nothing written.")
            return 0
        if not cohort.selected:
            print("Nothing to move.")
            return 0
        manifest = args.manifest or default_manifest_path()
        counts = apply_triage(session, cohort, manifest)
    print(
        f"Moved {counts['moved']} to {TRIAGE_STATUS}; {counts['changed']} changed since selection "
        f"and left as stored; {counts['labels_deleted']} brand labels deleted. Manifest: {manifest}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
