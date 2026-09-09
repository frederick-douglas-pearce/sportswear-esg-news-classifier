"""Health verdicts for scheduled checks.

A health verdict answers "what did this check find?" and is **not** a workflow
lifecycle state. `WorkflowStatus` (pending/running/paused/completed/failed) says
where a workflow *is*; a `HealthVerdict` says what one of its checks *saw*.
Adding `unknown` to `WorkflowStatus` instead would write run archives carrying
`status: unknown`, which every existing archive reader would then need
migrating for -- so epic #72 forbids it explicitly.

The vocabulary exists because a check that could not run must have somewhere to
say so. Before issue #71 it did not: a failed drift check set no key, the
workflow read `context.get("fp_drift_detected", False)`, and absence resolved
to "no drift" and then to "all classifiers healthy" -- on 219 runs.

This module holds the vocabulary and the subject-agnostic operations over it:
`verdict_of` reads one, `summarize` aggregates a set, `unresolved` names the
ones that never resolved. **The mapping from a given check's signals to these
verdicts stays beside that check** -- each scheduled script has its own contract
(the drift monitor's exit codes live in `src/mlops/exit_codes.py` and are
applied in `src/agent/workflows/drift_monitoring.py`), and keeping that mapping
per-workflow is what stops #77/#78/#79 depending on drift's semantics. D008
originally phrased this boundary as "enum only"; D010 revised the phrasing and
kept the invariant, which was always about the mapping rather than about
functions.

**This module is an import leaf, deliberately.** It imports nothing from the
`agent` package, so `workflows/base.py` can import *it*. The reverse direction
would be a cycle that is latent today and fires the moment a second workflow
adopts the vocabulary: `workflows/__init__` eagerly imports every workflow
module, so `health` importing `workflows.base` makes `import src.agent.health`
raise `ImportError: cannot import name ... from partially initialized module`
as soon as `daily_labeling` imports `health` (D010.2). The escalation gate that
turns an unresolved verdict into a FAILED workflow therefore lives in
`workflows/base.py` as `fail_on_unresolved_verdicts`, not here.
"""

import logging
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class HealthVerdict(str, Enum):
    """What a scheduled check found.

    **Always store `verdict.value` in workflow context, never the member.**
    The `str` mixin does NOT make this YAML-safe: `yaml.dump` renders a member
    of a `str, Enum` as `!!python/object/apply:...HealthVerdict\\n- healthy`,
    and `yaml.safe_dump` refuses it outright. `WorkflowStatus` survives only
    because `WorkflowState.to_dict`/`StepState.to_dict` call `.value`
    explicitly -- but `context` and `StepState.result` are dumped raw, so a
    member stored there reaches the archive as a Python object tag, and the
    next `yaml.safe_load` in `StateManager._load` raises into a bare `except`
    that resets all workflow state to `{}`.

    The mixin is kept because it makes `HealthVerdict.HEALTHY == "healthy"`
    true, which keeps comparisons against archived strings readable. That is
    what it is for; it is not a serialization guarantee.
    """

    HEALTHY = "healthy"
    """The check ran and found nothing wrong."""

    DEGRADED = "degraded"
    """The check ran and found a problem -- a real result, and actionable."""

    UNKNOWN = "unknown"
    """The check did not produce a verdict.

    It raised, or it had nothing to compare, or it returned a result with no
    evidence behind it. Never a synonym for healthy: this is the state whose
    absence caused #71.
    """

    SKIPPED = "skipped"
    """The check was deliberately not run, for a reason that must be stated.

    Distinct from UNKNOWN: skipping is a decision someone made, while unknown
    is a failure to decide. A skipped check does not fail a workflow, which is
    exactly why the reason has to be recorded with it.
    """


class HealthSummary(TypedDict):
    """An aggregate verdict over several checks, in a form safe to archive.

    **A `TypedDict`, and not a dataclass or `NamedTuple`, for a reason that bites
    silently.** Callers put this in the workflow context, and the state file is
    written with `yaml.dump` but read back with `yaml.safe_load`
    (`StateManager._save` / `._load`). A dataclass serializes happily on write
    and then fails to load on the next run, landing in `_load`'s bare `except`
    -- **which resets every workflow's state to `{}`**. A `TypedDict` is a plain
    `dict` at runtime, so it round-trips, while still giving #77/#78/#79 a typed
    shape to bind to. Every field below is a primitive for the same reason;
    verdicts appear as their `.value` strings, never as members (see
    `HealthVerdict`).
    """

    all_checked_healthy: bool
    """True only if at least one check ran AND every check that ran passed.

    Never "no failures were found", which is vacuously true when nothing ran --
    see `summarize`.
    """

    checked: list[str]
    """Subjects whose check was not skipped, in the order given."""

    degraded: list[str]
    """Subjects whose check ran and found a real problem."""

    unknown: list[str]
    """Subjects whose check produced no verdict."""

    skipped: list[str]
    """Subjects deliberately not checked."""


def verdict_of(context: Mapping[str, Any], key: str) -> HealthVerdict:
    """Read one verdict out of a workflow context, defaulting to UNKNOWN.

    Absent, `None`, and unrecognised all coerce to `UNKNOWN`. That default is
    the whole point of the module: a step that returned a dict without its
    verdict key must not read as healthy, which is the shape that let a failed
    drift check report "all classifiers healthy" (#71).
    """
    raw = context.get(key)
    try:
        return HealthVerdict(raw)
    except ValueError:
        logger.error(
            f"health verdict at {key!r} is missing or unrecognised ({raw!r}); "
            f"treating as unknown"
        )
        return HealthVerdict.UNKNOWN


def summarize(verdicts: Mapping[str, HealthVerdict]) -> HealthSummary:
    """Aggregate per-subject verdicts, without the vacuous-truth trap.

    `all_checked_healthy` is **"at least one check ran and every check that ran
    passed"** -- `bool(checked) and not degraded and not unknown` -- and
    deliberately not `all(v is HEALTHY for v in non_skipped)`, which Python
    reports as `True` over an empty sequence. A run in which every check was
    skipped would otherwise satisfy "no failures found" and report healthy
    having checked nothing: this epic's own thesis reproduced inside the
    mechanism written to enforce it (#95).

    Subject *identity* stays with the caller. This returns the generic
    partition; a workflow maps it onto its own context keys.
    """
    degraded = [s for s, v in verdicts.items() if v is HealthVerdict.DEGRADED]
    unknown = [s for s, v in verdicts.items() if v is HealthVerdict.UNKNOWN]
    skipped = [s for s, v in verdicts.items() if v is HealthVerdict.SKIPPED]
    checked = [s for s, v in verdicts.items() if v is not HealthVerdict.SKIPPED]

    return HealthSummary(
        all_checked_healthy=bool(checked) and not degraded and not unknown,
        checked=checked,
        degraded=degraded,
        unknown=unknown,
        skipped=skipped,
    )


def unresolved(verdicts: Mapping[str, HealthVerdict]) -> list[str]:
    """Subjects whose check never produced a verdict.

    The test is for an *explicit* healthy/degraded/skipped; anything else is
    unresolved. Because `verdict_of` maps an absent or unrecognised value to
    `UNKNOWN`, a handler that returned a dict without its verdict key is caught
    here rather than sailing past the one gate placed to catch it.
    """
    return [s for s, v in verdicts.items() if v is HealthVerdict.UNKNOWN]
