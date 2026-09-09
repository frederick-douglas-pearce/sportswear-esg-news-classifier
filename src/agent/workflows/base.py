"""Base workflow class and registry."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from ..config import agent_settings
from ..health import unresolved, verdict_of
from ..state import (
    StateManager,
    StepState,
    WorkflowState,
    WorkflowStatus,
    state_manager,
)

logger = logging.getLogger(__name__)


def _describe_exception(exc: BaseException) -> str:
    """A never-empty description of an exception.

    `str(RuntimeError())` is `""`, which would record a failed step whose error
    names nothing at all. Used by both the recording side and the summary side
    so the two agree -- the summary dedups by exact equality against what was
    recorded.
    """
    return str(exc) or type(exc).__name__


@dataclass
class StepFailure:
    """Returned by a step handler to mark the step -- and the run -- FAILED.

    A handler that catches its own error and returns a plain dict is recorded
    COMPLETED, which is how a run whose work failed archives as
    ``status: completed, error: null``. Returning this instead marks the step
    FAILED without the handler having to raise.

    Three properties of the contract that callers depend on:

    1. **``error`` is always on the step**, as ``StepState.error``. Read it there;
       it is the one field guaranteed to carry the failure detail.
    2. **``context`` goes to both places.** It is merged into the workflow
       context, exactly as a returned dict would be, *and* recorded as this
       step's ``StepState.result`` -- the workflow context is a flat namespace
       that several steps write the same keys into (``website_export`` has three
       writers of ``"error"`` alone), so per-step attribution would otherwise be
       lost to the run archive. A step that fails by *raising* has no payload and
       leaves ``result`` None.
    3. **``context`` fully replaces the dict return**, so it must carry every key
       downstream steps read. Dropping one is silent.

    Returning this does NOT halt the remaining steps: a failure dict does not
    halt them today, and aggregate-then-notify terminal steps require the
    earlier steps to have run. Guarding on context flags is therefore a
    requirement on downstream steps -- not a property they already have. It does
    not halt them, but it does decide the run: a FAILED step fails the workflow
    even if a later step pauses.
    """

    error: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDefinition:
    """Definition of a workflow step."""

    name: str
    description: str
    handler: Callable[["Workflow", dict[str, Any]], dict[str, Any] | StepFailure | None]
    skip_on_dry_run: bool = False
    requires_approval: bool = False


class UnresolvedVerdictReport(TypedDict):
    """What the terminal verdict gate writes into the workflow context.

    **Both paths return this whole shape**, so the archive carries the same keys
    whether the run passed or failed. The success path used to return only
    `verdicts_confirmed`, which meant the type annotated onto it described a
    payload it did not produce -- a type introduced to stop prose drifting,
    drifting itself.

    A `TypedDict` for the same reason `HealthSummary` is one: this reaches the
    run archive, which is written with `yaml.dump` and read back with
    `yaml.safe_load`, so every field has to be a plain primitive or the next
    load resets all workflow state. Stating the shape as a type rather than in
    prose is deliberate -- an earlier docstring here described these three
    fields as "a `str` or a `bool`", which two of them are not.

    The runtime guarantee is separate and lives at the boundary: the factory
    coerces subjects and reason values with `str()` on the way in. This type
    says what the shape is; the coercion is what makes it true.
    """

    verdicts_confirmed: bool
    """True when every verdict resolved, False when any did not."""

    unresolved_verdicts: list[str]
    """Subjects whose check produced no verdict; empty on the success path."""

    unresolved_reasons: dict[str, str]
    """Why each of those is unresolved, or "no reason recorded"; empty on success."""


def _default_unresolved_message(subjects: list[str], reasons: dict[str, str]) -> str:
    """Fallback wording for a gate whose workflow supplied none."""
    names = ", ".join(subjects)
    detail = "; ".join(f"{s}: {reasons[s]}" for s in subjects)
    return (
        f"No verdict was produced for {names} - this run must not be recorded "
        f"as successful ({detail})"
    )


def fail_on_unresolved_verdicts(
    subjects: Sequence[str],
    *,
    verdict_key: str = "{subject}_verdict",
    reason_key: str = "{subject}_error",
    describe: Callable[[list[str], dict[str, str]], str] | None = None,
) -> Callable[["Workflow", dict[str, Any]], UnresolvedVerdictReport | StepFailure]:
    """Build the terminal step that fails a run whose checks produced no verdict.

    This is the first-class escalation path issue #74 owes: before it, the only
    implementation was hand-rolled inside ``drift_monitoring`` and every other
    workflow adopting the vocabulary would have written its own (D010.2).

    **Register the step it returns after every step that writes a verdict** --
    in practice, last. The gate reads verdicts out of the context, so a subject
    whose check has not run yet reads as ``unknown`` and fails the run
    spuriously. That is the whole constraint.

    It is *not* an archive-ordering constraint, and an earlier draft of this
    docstring said it was: it claimed a gate placed before the reporting step
    would discard that report, which is the behaviour of the **raising** gate
    this factory replaced. Returning ``StepFailure`` does not halt the loop
    (see ``StepFailure``), so later steps still run and are still recorded --
    measured, with the gate registered second of four: the alert step still
    sent and the report step still recorded ``COMPLETED`` with its result.

    **It returns ``StepFailure`` rather than raising** (D010.3). Both reach
    FAILED through ``_finalize`` -- the raising path via ``run()``'s ``except``,
    this one via ``any_failed`` -- and ``_failure_summary`` wraps the text as
    ``"Step '<name>' failed: <error>"`` either way, so the archived
    ``WorkflowState.error`` keeps its shape. What ``StepFailure`` adds is the
    payload on ``StepState.result``, which is the per-step attribution #75/#76
    want, and the absence of a traceback that describes nothing.

    The gate passes only on an **explicit** healthy/degraded/skipped: because
    ``verdict_of`` maps an absent, ``None`` or unrecognised value to ``UNKNOWN``,
    a handler that returned a dict without its verdict key is caught here rather
    than slipping past the one gate placed to catch it.

    Args:
        subjects: What was checked -- classifiers, feeds, stages. Used to build
            each context key and reported in the failure message.
        verdict_key: Template for a subject's verdict key in the workflow
            context. Formatted with ``subject=``.
        reason_key: Template for a subject's failure-reason key. The reason is
            threaded into the message and the payload rather than dropped: a
            gate that names *which* checks are unresolved but not *why* thins
            the detail #75/#76 read out of the archive (D010.5).
        describe: Builds the failure text from the unresolved subjects and their
            reasons. Workflows with their own wording pass one; the default
            covers the rest.

    Returns:
        A step handler whose failure payload is an ``UnresolvedVerdictReport``
        -- see that type for the shape. It round-trips ``yaml.safe_load``
        because subjects are *validated* as ``str`` at construction and reason
        values are coerced with ``str()`` at read time. The asymmetry is
        deliberate: a subject is supplied by the workflow author, so a wrong one
        is a bug worth refusing outright; a reason is whatever happened to be in
        the context at runtime, which the gate cannot refuse and so must make
        safe. An uncoerced reason -- an exception object, a ``Path`` -- reaches
        ``yaml.dump`` intact and fails only on the *next* run's load.

    Raises:
        TypeError: if ``subjects`` is a ``str``/``bytes``/``bytearray`` itself,
            or if any element is not a ``str``.
        ValueError: if ``subjects`` is empty. A gate over no subjects would
            report ``verdicts_confirmed`` having confirmed nothing -- the same
            vacuous truth ``summarize`` exists to refuse.
    """
    if isinstance(subjects, (str, bytes, bytearray)):
        raise TypeError(
            f"subjects must be a sequence of names, not {subjects!r} itself -- "
            f"iterating it yields one subject per character or per byte"
        )

    subjects = tuple(subjects)
    if not subjects:
        raise ValueError(
            "fail_on_unresolved_verdicts() needs at least one subject: a gate over "
            "no subjects always passes, which is the vacuous truth this contract refuses"
        )
    # Validate rather than coerce. str() on a non-str subject would manufacture a
    # plausible-looking name -- a memoryview or an array of bytes yields ints, and
    # str() turns those into subjects called '102', '101'. Requiring str is both
    # stronger and shorter than enumerating the sequence types that misbehave,
    # which the previous guard tried to do and got wrong.
    if not all(isinstance(subject, str) for subject in subjects):
        raise TypeError(
            f"every subject must be a str; got "
            f"{[type(s).__name__ for s in subjects if not isinstance(s, str)]}"
        )

    def handler(
        workflow: "Workflow", context: dict[str, Any]
    ) -> UnresolvedVerdictReport | StepFailure:
        verdicts = {
            subject: verdict_of(context, verdict_key.format(subject=subject))
            for subject in subjects
        }
        missing = unresolved(verdicts)

        if not missing:
            return UnresolvedVerdictReport(
                verdicts_confirmed=True, unresolved_verdicts=[], unresolved_reasons={}
            )

        reasons = {
            # str() because the value is whatever the workflow stored: an
            # exception object or a Path survives yaml.dump and then fails the
            # next yaml.safe_load, taking all workflow state with it.
            subject: str(context.get(reason_key.format(subject=subject)) or "no reason recorded")
            for subject in missing
        }
        message = (describe or _default_unresolved_message)(missing, reasons)
        logger.error(message)

        report = UnresolvedVerdictReport(
            verdicts_confirmed=False,
            unresolved_verdicts=missing,
            unresolved_reasons=reasons,
        )
        # dict() because StepFailure.context is a plain dict[str, Any]; a
        # TypedDict is one at runtime but is not assignable to it. Building the
        # report first is what keeps both paths pinned to the same declared shape.
        return StepFailure(error=message, context=dict(report))

    return handler


class Workflow(ABC):
    """Base class for workflow definitions.

    Subclasses define steps and their execution logic. The base class
    handles state management, error handling, and step transitions.
    """

    # Subclasses must define these
    name: str = ""
    description: str = ""
    steps: list[StepDefinition] = []

    # Per-step cap on how much of a step's error reaches the workflow-level
    # summary built by _failure_summary(). The full text stays in step.error.
    _ERROR_SUMMARY_LIMIT = 500

    def __init__(
        self,
        state_manager: StateManager | None = None,
        dry_run: bool | None = None,
    ):
        """Initialize workflow.

        Args:
            state_manager: State manager instance (default: global state_manager)
            dry_run: Override dry_run setting
        """
        self.state = state_manager or globals()["state_manager"]
        self.dry_run = dry_run if dry_run is not None else agent_settings.dry_run
        self._workflow_state: WorkflowState | None = None

    @property
    def step_names(self) -> list[str]:
        """Get list of step names."""
        return [step.name for step in self.steps]

    def run(self, context: dict[str, Any] | None = None) -> WorkflowState:
        """Execute the workflow.

        Args:
            context: Initial context data

        Returns:
            Final workflow state
        """
        context = context or {}
        context["dry_run"] = self.dry_run

        # Create workflow state
        self._workflow_state = self.state.create_workflow(
            name=self.name,
            steps=self.step_names,
            context=context,
        )
        self.state.start_workflow(self.name)

        logger.info(f"Starting workflow: {self.name} (dry_run={self.dry_run})")

        try:
            for step in self.steps:
                if self._workflow_state.status == WorkflowStatus.PAUSED:
                    logger.info(f"Workflow paused at step: {step.name}")
                    break

                # Skip steps in dry-run mode if configured
                if self.dry_run and step.skip_on_dry_run:
                    logger.info(f"Skipping step (dry-run): {step.name}")
                    self.state.complete_step(
                        self.name,
                        step.name,
                        result={"skipped": True, "reason": "dry_run"},
                    )
                    continue

                # Handle approval-required steps
                if step.requires_approval:
                    # Set current_step so resume knows where we paused
                    self._workflow_state.current_step = step.name
                    self.state.pause_workflow(
                        self.name,
                        reason=f"Approval required for: {step.name}",
                    )
                    logger.info(f"Workflow paused - approval required for: {step.name}")
                    break

                # Execute step
                self._execute_step(step)

            self._finalize()

        except Exception as e:
            logger.exception(f"Workflow failed: {e}")
            self._finalize(exception=e)

        return self.state.get_workflow(self.name)  # type: ignore

    def _recorded_steps(self) -> list[tuple[str, StepState | None]]:
        """Pair every defined step name with its persisted state, if any.

        `.get()` rather than indexing, because `resume()` loads state written by
        an earlier process: a step added to the workflow class since that run
        started has no record. Indexing raised `KeyError` from inside
        `_finalize` -- including from the `except` handler that calls it -- so
        the exception escaped `run()`/`resume()` entirely and left the run
        RUNNING and unarchived. That is the very state this class exists to make
        impossible, so the finalizer must not be the thing that produces it.
        """
        steps = self._workflow_state.steps if self._workflow_state else {}
        return [(name, steps.get(name)) for name in self.step_names]

    def _failure_summary(self, exception: Exception | None = None) -> str:
        """Build the workflow-level error from the steps that actually failed.

        Each failed step contributes its name and its recorded error, truncated
        to ``_ERROR_SUMMARY_LIMIT`` -- several steps can fail in one run (a
        ``StepFailure`` does not halt the loop) and a step error can be large,
        e.g. a captured stderr tail. The untruncated text always remains in
        ``StepState.error``. The cap is per step, so the summary grows with the
        number of failed steps.
        """
        parts = []
        for name, step in self._recorded_steps():
            if step is None:
                parts.append(
                    f"Step '{name}' has no recorded state "
                    "(the workflow definition changed since this run started)"
                )
                continue
            if step.status != WorkflowStatus.FAILED:
                continue
            error = step.error or "no error recorded"
            if len(error) > self._ERROR_SUMMARY_LIMIT:
                error = (
                    f"{error[: self._ERROR_SUMMARY_LIMIT]}... "
                    "(truncated; full text in step.error)"
                )
            parts.append(f"Step '{name}' failed: {error}")

        if exception is not None:
            # A step that raises is recorded by _execute_step with exactly
            # str(e), so an exact match means this exception is already
            # represented above. Anything else -- including an exception raised
            # outside any step, which matches nothing -- is appended, since this
            # is the only route by which it reaches the workflow error.
            # Exact equality can in principle drop a runner-level exception whose
            # message coincides with an unrelated step's error; that costs one
            # omitted duplicate at worst and never loses a step's own failure.
            described = _describe_exception(exception)
            if not any(
                step is not None and step.error == described
                for _, step in self._recorded_steps()
            ):
                parts.append(f"Workflow error: {described}")

        return "; ".join(parts) or "Not all steps completed"

    def _finalize(self, exception: Exception | None = None) -> None:
        """Turn step outcomes into the workflow verdict.

        The single place step statuses become a workflow status. ``run()`` and
        ``resume()`` both call it, and so do both of their ``except`` branches,
        so the two paths cannot drift apart -- two copies of this logic is the
        divergence this change exists to remove (see D009). ``resume()``
        previously had no *non-exception* failure branch: it completed the
        workflow when every step had completed and did nothing otherwise, so a
        resumed run carrying a failed step stayed RUNNING and was never archived
        (the archive is written only by ``complete_workflow``/``fail_workflow``).
        A raised failure on resume was always caught and failed by its
        ``except``, so that gap was latent until ``StepFailure`` made a
        non-raising failure possible.
        """
        recorded = self._recorded_steps()
        all_completed = all(
            step is not None and step.status == WorkflowStatus.COMPLETED
            for _, step in recorded
        )
        any_failed = any(
            step is not None and step.status == WorkflowStatus.FAILED
            for _, step in recorded
        )

        if exception is None:
            if all_completed:
                self.state.complete_workflow(self.name)
                logger.info(f"Workflow completed: {self.name}")
                return
            if self._workflow_state.status == WorkflowStatus.PAUSED and not any_failed:  # type: ignore[union-attr]
                # Waiting on a human, not a failure. Left for resume().
                return
            # A pause does NOT mask a failed step. Without this, a step that
            # returned StepFailure followed by any pausing step archived
            # nowhere, reported `paused` with error None, and exited 0 -- a
            # failed run that reads as waiting. Reachable as soon as a workflow
            # with an approval step adopts StepFailure.

        self.state.fail_workflow(self.name, self._failure_summary(exception))

    def _execute_step(self, step: StepDefinition) -> None:
        """Execute a single step with error handling."""
        logger.info(f"Executing step: {step.name} - {step.description}")
        self.state.start_step(self.name, step.name)

        try:
            result = step.handler(self, self._workflow_state.context)  # type: ignore

            if isinstance(result, StepFailure):
                # Must precede complete_step, which records COMPLETED
                # unconditionally. Independently, a StepFailure is truthy, so
                # the `if result:` line below would reach
                # update_context(..., StepFailure) and raise TypeError -- a
                # dataclass is not a mapping. Two separate reasons; either one
                # requires this branch to come first.
                logger.error(f"Step reported failure: {step.name} - {result.error}")
                self.state.fail_step(
                    self.name, step.name, result.error, result=result.context or None
                )
                if result.context:
                    self.state.update_context(self.name, result.context)
                # Deliberately no raise: the loop continues, exactly as it does
                # for a handler that returns a failure dict today.
                return

            self.state.complete_step(self.name, step.name, result=result)
            if result:
                self.state.update_context(self.name, result)
            logger.info(f"Step completed: {step.name}")

        except Exception as e:
            logger.error(f"Step failed: {step.name} - {_describe_exception(e)}")
            self.state.fail_step(self.name, step.name, _describe_exception(e))
            raise

    def resume(self) -> WorkflowState:
        """Resume a paused workflow.

        Returns:
            Final workflow state
        """
        self._workflow_state = self.state.get_workflow(self.name)
        if not self._workflow_state:
            raise ValueError(f"No workflow state found for: {self.name}")

        if self._workflow_state.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow is not paused: {self._workflow_state.status}")

        self.state.resume_workflow(self.name)
        logger.info(f"Resuming workflow: {self.name}")

        # Find the next pending step after current
        current_step = self._workflow_state.current_step
        found_current = False
        remaining_steps = []

        for step in self.steps:
            if step.name == current_step:
                found_current = True
                # Mark the approval step as completed (user resumed = manual work done)
                if step.requires_approval:
                    self.state.complete_step(
                        self.name,
                        step.name,
                        result={"approved": True, "manual_completion": True},
                    )
                continue
            if found_current:
                remaining_steps.append(step)

        # Execute remaining steps
        try:
            for step in remaining_steps:
                if self._workflow_state.status == WorkflowStatus.PAUSED:
                    break

                step_state = self._workflow_state.steps.get(step.name)
                if step_state and step_state.status == WorkflowStatus.COMPLETED:
                    continue

                if step.requires_approval:
                    # Set current_step so next resume knows where we paused
                    self._workflow_state.current_step = step.name
                    self.state.pause_workflow(
                        self.name,
                        reason=f"Approval required for: {step.name}",
                    )
                    break

                self._execute_step(step)

            self._finalize()

        except Exception as e:
            logger.exception(f"Workflow failed on resume: {e}")
            self._finalize(exception=e)

        return self.state.get_workflow(self.name)  # type: ignore


class WorkflowRegistry:
    """Registry of available workflows."""

    _workflows: dict[str, type[Workflow]] = {}

    @classmethod
    def register(cls, workflow_class: type[Workflow]) -> type[Workflow]:
        """Register a workflow class.

        Can be used as a decorator:
            @WorkflowRegistry.register
            class MyWorkflow(Workflow):
                ...
        """
        cls._workflows[workflow_class.name] = workflow_class
        return workflow_class

    @classmethod
    def get(cls, name: str) -> type[Workflow] | None:
        """Get a workflow class by name."""
        return cls._workflows.get(name)

    @classmethod
    def list(cls) -> list[str]:
        """List registered workflow names."""
        return list(cls._workflows.keys())

    @classmethod
    def create(
        cls,
        name: str,
        state_manager: StateManager | None = None,
        dry_run: bool | None = None,
    ) -> Workflow:
        """Create a workflow instance by name.

        Args:
            name: Workflow name
            state_manager: State manager instance
            dry_run: Override dry_run setting

        Returns:
            Workflow instance

        Raises:
            ValueError: If workflow not found
        """
        workflow_class = cls.get(name)
        if not workflow_class:
            available = ", ".join(cls.list())
            raise ValueError(f"Unknown workflow: {name}. Available: {available}")

        return workflow_class(state_manager=state_manager, dry_run=dry_run)
