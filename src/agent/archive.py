"""Read the run archive that scheduled workflows leave behind.

``StateManager._archive_workflow`` writes each terminal run's full
``WorkflowState.to_dict()`` to ``agent_settings.history_dir``. Nothing else
reads it back. This module is that reader, and it is the piece #75 is meant to
import rather than re-derive.

**This module writes nothing and holds no state.** It is not a persistence
layer, a store, or a framework -- epic #72 and ``CLAUDE.md`` both name
``src/experiment_log/`` as the precedent to avoid, and the whole point of #76 is
that the signal is already on disk.

**Import position.** This is a leaf below the workflow layer: it imports
``.config``, PyYAML and stdlib. It deliberately does not import ``.state``, even
though ``WorkflowState.to_dict`` defines the shape it parses -- see *Two readers
of one shape* below. ``workflows/run_audit`` imports *this*; the reverse
direction would be the cycle ``health.py`` documents for itself.

Two readers of one shape
------------------------
The record shape is defined by ``WorkflowState.to_dict``/``StepState.to_dict``.
This module re-states those key names rather than calling ``WorkflowState.
from_dict``, because ``from_dict`` raises on a malformed timestamp and a reader
of historical data must survive one bad file. That tolerance buys robustness and
costs a contract: a field renamed in ``state.py`` would silently degrade this
reader to "sees no signal" -- this epic's own defect, inside the instrument built
to detect it. ``tests/test_agent_archive.py`` pins the contract with a
round-trip over real ``WorkflowState`` objects, so the rename breaks CI instead
of the auditor.

Extraction is not policy
------------------------
``failure_signals`` extracts *kinded* evidence; ``run_succeeded`` is a policy
over which kinds disqualify a run. The two consumers need different breadth, and
merging them would break one of them:

* the Class-A sweep (``scripts/audit_archive.py``) is read by a human, once, and
  can afford to be broad -- it reports what it found and lets the reader judge;
* ``run_succeeded`` is the predicate #75's *automatic* consecutive-failure
  alerting is to be built on, so over-broadening it would turn into alert noise
  there. It must not read ordinary non-failure context as failure: workflows
  legitimately archive keys like
  ``alerts_sent: false``, ``alerts_skipped: true`` and
  ``reason: nothing_to_report``.

``run_succeeded`` therefore keys on the **current** contract -- a failed run, a
failed step, a step error, a run error, or a health verdict that never resolved
(#73, #74). The historical ``<name>_success: false`` form belongs to the
archaeology sweep: it is what runs archived *before* those issues landed, and
reading it as a live failure signal would make #75 re-alert on history.
"""

import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import agent_settings

logger = logging.getLogger(__name__)

# ``_archive_workflow`` names a file ``{workflow_name}_{run_id}.yaml`` where
# ``run_id`` is ``%Y%m%d_%H%M%S``. Anchor on that suffix and treat everything
# before it as the workflow name: every real workflow name contains underscores
# (``daily_labeling``, ``website_export``, ``drift_monitoring``,
# ``model_training``), so splitting on "_" misparses all of them.
_ARCHIVE_NAME = re.compile(r"^(?P<name>.+)_(?P<run_id>\d{8}_\d{6})\.yaml$")

_RUN_ID_FORMAT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class ArchivedRun:
    """One archived run, identified by its filename and carrying its record.

    ``run_at`` comes from the **filename**, not from the record body, and is the
    ordering key. The body's own ``started_at`` can be absent or unparseable in
    a historical record, while the filename is what the archiver always writes
    -- so ordering keyed on the body would be undefined for exactly the old runs
    this reader exists to read. The body remains available through ``data``.
    """

    workflow_name: str
    run_id: str
    run_at: datetime
    path: Path
    data: dict[str, Any]

    @property
    def status(self) -> str | None:
        """The archived lifecycle status, or None if the record omits it."""
        value = self.data.get("status")
        return value if isinstance(value, str) else None

    @property
    def context(self) -> dict[str, Any]:
        """The archived workflow context, or an empty mapping."""
        value = self.data.get("context")
        return value if isinstance(value, dict) else {}

    @property
    def steps(self) -> dict[str, Any]:
        """The archived per-step records, or an empty mapping."""
        value = self.data.get("steps")
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class FailureSignal:
    """One piece of evidence that a run did not do what it reported.

    ``kind`` is what the signal is; ``where`` is where it was found; ``detail``
    renders it for a human. A signal names its evidence so that a finding can be
    checked rather than believed.
    """

    kind: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.where}: {self.detail}"


#: Signal kinds that mean the run did not succeed, under the contract that
#: #73 and #74 established. Deliberately excludes ``success_flag_false`` and
#: ``context_errors`` -- see the module docstring.
DISQUALIFYING_KINDS = frozenset(
    {
        "status_failed",
        "run_error",
        "step_failed",
        "step_error",
        "verdict_unknown",
    }
)

#: Every kind ``failure_signals`` can emit. Exported so a CLI filtering on kind
#: can validate its argument against the emitter rather than against a second
#: hand-maintained list: an unrecognised ``--kind`` that silently matches
#: nothing prints "no findings" and exits clean, which is this epic's failure
#: mode wearing the sweep's own output.
SIGNAL_KINDS = DISQUALIFYING_KINDS | frozenset(
    {
        "success_flag_false",
        "context_errors",
    }
)


def iter_runs(
    history_dir: Path | None = None,
    *,
    workflows: Iterable[str] | None = None,
) -> Iterator[ArchivedRun]:
    """Yield archived runs, oldest first.

    Args:
        history_dir: Directory to read. Defaults to ``agent_settings.history_dir``.
        workflows: If given, only yield runs whose workflow name is in this set.
            This is how synthetic, test-written archives are excluded -- by an
            explicit allowlist of real workflow names, never by a heuristic over
            the record contents.

    Ordering is chronological by ``run_at`` and is part of the contract: #75
    needs a per-workflow run sequence in time order. Lexical filename order
    happens to agree because the timestamp is zero-padded, but this sorts
    explicitly rather than relying on that.

    A file that does not parse is logged and skipped, never raised: one bad
    record must not blind the reader to every other one. Raises ``OSError`` if
    the directory itself cannot be listed -- that is not a bad record, it is the
    reader being unable to tell anything at all, and the caller must be able to
    distinguish those.
    """
    directory = history_dir if history_dir is not None else agent_settings.history_dir
    allowed = set(workflows) if workflows is not None else None

    # Checked explicitly rather than left to glob(), which yields nothing for a
    # missing directory. "No archive directory" and "an archive directory with
    # no runs in it" must not share an answer: the first means the reader can
    # tell nothing, the second is a finding about every workflow in it.
    if not directory.is_dir():
        raise FileNotFoundError(f"run archive directory not found: {directory}")

    runs: list[ArchivedRun] = []
    for path in sorted(directory.glob("*.yaml")):
        match = _ARCHIVE_NAME.match(path.name)
        if not match:
            continue

        name = match.group("name")
        if allowed is not None and name not in allowed:
            continue

        run_id = match.group("run_id")
        try:
            run_at = datetime.strptime(run_id, _RUN_ID_FORMAT).replace(
                tzinfo=timezone.utc
            )
        except ValueError:  # pragma: no cover - the regex already pins the shape
            logger.warning(f"archive {path.name} has an unparseable run id; skipping")
            continue

        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"archive {path.name} could not be read; skipping ({exc})")
            continue

        if not isinstance(data, dict):
            logger.warning(f"archive {path.name} is not a mapping; skipping")
            continue

        runs.append(
            ArchivedRun(
                workflow_name=name,
                run_id=run_id,
                run_at=run_at,
                path=path,
                data=data,
            )
        )

    runs.sort(key=lambda run: (run.run_at, run.run_id))
    yield from runs


def latest_run_per_workflow(
    history_dir: Path | None = None,
    *,
    workflows: Iterable[str] | None = None,
) -> dict[str, ArchivedRun]:
    """Map each workflow name to its most recent archived run.

    A workflow with no archived run at all is simply absent from the result --
    the caller decides what that means. It is not an error here, and it is not
    "unknown" either: for a workflow the operator expects to be running, having
    never run is the strongest possible finding, not an absence of one.
    """
    latest: dict[str, ArchivedRun] = {}
    for run in iter_runs(history_dir, workflows=workflows):
        latest[run.workflow_name] = run  # iter_runs is oldest-first
    return latest


def failure_signals(run: ArchivedRun) -> list[FailureSignal]:
    """Every piece of failure evidence carried by this run's record.

    Broad by design: this is the extraction half, and the Class-A sweep that
    consumes it is read by a human. ``run_succeeded`` narrows it to a policy.
    """
    signals: list[FailureSignal] = []

    if run.status == "failed":
        signals.append(
            FailureSignal("status_failed", "status", "the run archived as failed")
        )

    run_error = run.data.get("error")
    if run_error:
        signals.append(FailureSignal("run_error", "error", str(run_error)))

    for step_name, step in sorted(run.steps.items()):
        if not isinstance(step, dict):
            continue
        if step.get("status") == "failed":
            signals.append(
                FailureSignal(
                    "step_failed", f"steps.{step_name}.status", "the step failed"
                )
            )
        step_error = step.get("error")
        if step_error:
            signals.append(
                FailureSignal("step_error", f"steps.{step_name}.error", str(step_error))
            )

    for key, value in sorted(run.context.items()):
        if key.endswith("_verdict") and value == "unknown":
            signals.append(
                FailureSignal(
                    "verdict_unknown",
                    f"context.{key}",
                    "the check produced no verdict",
                )
            )
        # The pre-#73/#74 form. Archaeology only: see the module docstring.
        if key.endswith("_success") and value is False:
            signals.append(
                FailureSignal(
                    "success_flag_false",
                    f"context.{key}",
                    "an embedded failure flag the run reported success over",
                )
            )

    errors = run.context.get("errors")
    if isinstance(errors, (list, tuple)) and errors:
        signals.append(
            FailureSignal(
                "context_errors",
                "context.errors",
                "; ".join(str(item) for item in errors),
            )
        )

    return signals


def reported_success(run: ArchivedRun) -> bool:
    """Did the run *claim* to have succeeded?

    The claim, not the truth of it. ``vacuous_success_signals`` is the gap
    between the two.
    """
    return run.status == "completed"


def run_succeeded(run: ArchivedRun) -> bool:
    """Did this run actually succeed, under the #73/#74 contract?

    The shared classifier #75 imports for its consecutive-failure counter. It is
    narrower than ``failure_signals`` on purpose -- see the module docstring for
    why the two differ and what over-broadening would cost.

    A run that reports no status at all is **not** counted as a success: absence
    of a failure signal is not evidence of success, which is the reasoning error
    this whole epic exists to remove.
    """
    if run.status != "completed":
        return False
    return not any(
        signal.kind in DISQUALIFYING_KINDS for signal in failure_signals(run)
    )


def vacuous_success_signals(run: ArchivedRun) -> list[FailureSignal]:
    """Failure evidence carried by a run that reported success.

    The Class-A predicate: ``status: completed`` in the same record as evidence
    that something inside it failed. Empty for a run that did not report
    success -- a run that reported failure is not lying about it.
    """
    if not reported_success(run):
        return []
    return failure_signals(run)
