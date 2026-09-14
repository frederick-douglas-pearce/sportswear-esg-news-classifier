#!/usr/bin/env python3
"""One-shot sweep: archived runs that reported success while carrying a failure.

The Class-A half of #76. The epic's central finding is that the signal was
already on disk and simply never read -- runs archived as ``completed`` in the
same record as an embedded failure flag. This reads it.

**Deliberately a script and not a scheduled check.** Its value is a single
retroactive pass over the history that already exists; once #73 and #74 fix
propagation at the source, a *recurring* version of this would mostly re-report
the same history. It is not wired to cron and should not be given a schedule --
the recurring liveness check is ``run_audit``, which answers a different
question.

**It reports a list of instances and never a rate.** That is not a presentation
choice, it is what makes the output trustworthy without defining a corpus. The
archive contains runs written by the test suite under production workflow names,
so any denominator computed over the directory is wrong in a way that is hard to
see; a list has no denominator, and a spurious line is one a reader can dismiss.
Percentages measured over this directory have been wrong before, repeatedly.

Usage:
    uv run python scripts/audit_archive.py
    uv run python scripts/audit_archive.py --workflow drift_monitoring
    uv run python scripts/audit_archive.py --kind success_flag_false

Exit codes:
    0  no run reported success over a failure signal
    1  at least one did (they are listed on stdout)
    2  the archive could not be read at all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.archive import (  # noqa: E402
    iter_runs,
    vacuous_success_signals,
)
from src.agent.config import agent_settings  # noqa: E402

# The real workflow names. An explicit allowlist, never a heuristic over record
# contents: synthetic names written by the test suite are excluded because they
# are not workflows, which is a fact about the name and cannot misfire.
KNOWN_WORKFLOWS = (
    "daily_labeling",
    "drift_monitoring",
    "model_training",
    "run_audit",
    "website_export",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Archive directory to sweep (default: the agent's history dir)",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=KNOWN_WORKFLOWS,
        help="Restrict to this workflow (repeatable; default: all known)",
    )
    parser.add_argument(
        "--kind",
        action="append",
        help="Only report these signal kinds (repeatable)",
    )
    parser.add_argument(
        "--all-signals",
        action="store_true",
        help="Also show runs that reported failure honestly, for comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    history_dir = args.history_dir or agent_settings.history_dir
    workflows = tuple(args.workflow) if args.workflow else KNOWN_WORKFLOWS

    try:
        runs = list(iter_runs(history_dir, workflows=workflows))
    except OSError as exc:
        print(f"could not read the run archive at {history_dir}: {exc}", file=sys.stderr)
        return 2

    findings = 0
    for run in runs:
        signals = vacuous_success_signals(run)
        if args.kind:
            signals = [s for s in signals if s.kind in set(args.kind)]
        if not signals:
            continue

        findings += 1
        print(f"{run.workflow_name} {run.run_id}  ({run.path.name})")
        for signal in signals:
            print(f"    [{signal.kind}] {signal}")

    print()
    if findings:
        print(
            f"{findings} archived run(s) reported success while carrying a failure "
            f"signal. Listed above by run; this is a list of instances, not a rate "
            f"-- see the module docstring for why no proportion is reported."
        )
        return 1

    print("No archived run reported success over a failure signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
