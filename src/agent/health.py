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
`as_verdict` coerces one value, `verdict_of` reads one out of a context,
`summarize` aggregates a set, and `unresolved` names the ones that never
resolved -- the last three all route through the first. **The mapping from a given check's signals to these
verdicts stays beside that check** -- each scheduled script has its own contract
(the drift monitor's exit codes live in `src/mlops/exit_codes.py` and are
applied in `src/agent/workflows/drift_monitoring.py`), and keeping that mapping
per-workflow is what stops #77/#78/#79 depending on drift's semantics. D008
originally phrased this boundary as "enum only"; D010 revised the phrasing and
kept the invariant, which was always about the mapping rather than about
functions.

**This module is an import leaf, deliberately.** It imports nothing from the
`agent` package, so `workflows/base.py` can import *it*. The reverse direction
is a cycle **today** -- not a hazard that arrives with a future adopter.
`workflows/__init__` eagerly imports every workflow module, and
`drift_monitoring` already imports this one, so adding `health -> workflows.base`
makes `import src.agent.health` fail immediately with `ImportError: cannot
import name 'HealthVerdict' from partially initialized module`, with
`drift_monitoring` alone closing the loop. The
escalation gate that turns an unresolved verdict into a FAILED workflow
therefore lives in `workflows/base.py` as `fail_on_unresolved_verdicts`, not
here (D010.2).
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

    **A `TypedDict`, and not a dataclass or `NamedTuple`, so that a caller may
    put it in the workflow context.** The state file is written with `yaml.dump`
    but read back with `yaml.safe_load` (`StateManager._save` / `._load`), so a
    non-primitive here fails to load on the next run. A `TypedDict` is a plain
    `dict` at runtime, and gives #77/#78/#79 a typed shape to bind to.

    The fields hold the caller's own subject names, so those must be strings.
    No verdict appears in this structure -- only subject names and a bool.
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


def _describe(value: object, limit: int = 120) -> str:
    """A bounded repr that does not itself raise.

    `repr()` is not total -- a very large int and a raising `__repr__` both
    escape it -- and a log line explaining a bad value must not become one.
    """
    try:
        shown = repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


def as_verdict(value: object, *, label: str | None = None) -> HealthVerdict:
    """Coerce anything to a verdict, resolving what we cannot read to UNKNOWN.

    This is the module's core rule in one place, so that every entry point
    applies it identically. `None`, an unrecognised string, an int, an
    unhashable object -- all become `UNKNOWN` and are logged. A member passes
    through, and so does a stored `.value` string, which matters because the
    context stores `.value` strings by mandate (see `HealthVerdict`): reading a
    verdict back out of a run archive and handing it here has to work.

    `UNKNOWN` is the route for "we cannot tell": it fails the run at the
    terminal gate. The `except` is deliberately broad so that a value this
    cannot interpret comes back as a verdict rather than as an exception; the
    fallback is logged, not silent.
    """
    try:
        return HealthVerdict(value)
    except Exception:
        where = f" at {label!r}" if label else ""
        logger.error(
            f"health verdict{where} is missing or unrecognised ({_describe(value)}); "
            f"treating as unknown"
        )
        return HealthVerdict.UNKNOWN


def verdict_of(context: Mapping[str, Any], key: str) -> HealthVerdict:
    """Read one verdict out of a workflow context, defaulting to UNKNOWN.

    Absent, `None`, and unrecognised all coerce to `UNKNOWN` (via `as_verdict`).
    That default is the whole point of the module: a step that returned a dict
    without its verdict key must not read as healthy, which is the shape that
    let a failed drift check report "all classifiers healthy" (#71).
    """
    return as_verdict(context.get(key), label=key)


def summarize(verdicts: Mapping[str, object]) -> HealthSummary:
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

    **Values are normalized through `as_verdict`**, so stored `.value` strings
    are accepted and anything unrecognised becomes `UNKNOWN`. Identity
    comparison against the raw input would misclassify the stored form, which
    is what the context actually holds.
    """
    normalized = {s: as_verdict(v, label=s) for s, v in verdicts.items()}

    degraded = [s for s, v in normalized.items() if v is HealthVerdict.DEGRADED]
    unknown = [s for s, v in normalized.items() if v is HealthVerdict.UNKNOWN]
    skipped = [s for s, v in normalized.items() if v is HealthVerdict.SKIPPED]
    checked = [s for s, v in normalized.items() if v is not HealthVerdict.SKIPPED]

    return HealthSummary(
        all_checked_healthy=bool(checked) and not degraded and not unknown,
        checked=checked,
        degraded=degraded,
        unknown=unknown,
        skipped=skipped,
    )


def unresolved(verdicts: Mapping[str, object]) -> list[str]:
    """Subjects whose check never produced a verdict.

    The test is for an *explicit* healthy/degraded/skipped; anything else is
    unresolved. Values are normalized through `as_verdict`, so a handler that
    returned a dict without its verdict key is caught here rather than sailing
    past the one gate placed to catch it -- and so is a caller who passes stored
    `.value` strings, which is the shape a run archive holds.
    """
    return [
        s for s, v in verdicts.items() if as_verdict(v, label=s) is HealthVerdict.UNKNOWN
    ]
