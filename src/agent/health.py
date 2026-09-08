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

This module holds the vocabulary and nothing else. The mapping from a given
check's exit codes to these verdicts belongs beside that check, because each
scheduled script has its own exit-code contract (the drift monitor's lives in
`src/mlops/exit_codes.py` and is applied in
`src/agent/workflows/drift_monitoring.py`). A shared enum plus a per-workflow
mapping keeps #77/#78/#79 from having to depend on drift's semantics.
"""

from enum import Enum


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
