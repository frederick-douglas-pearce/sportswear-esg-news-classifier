"""CLI utilities for experiment log queries and recording.

Designed for replay-time use from SKILL.md steps or agent invocations.

Usage:
    uv run python -m src.experiment_log heuristics --classifier fp
    uv run python -m src.experiment_log record-decision --classifier fp --experiment-id EXP_ID ...
    uv run python -m src.experiment_log update-heuristic --classifier fp --trigger "..." --success true
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from .models import Decision, DecisionOption, Heuristic
from .store import ExperimentStore

logger = logging.getLogger(__name__)


def lookup_heuristics(
    classifier: str,
    phase: str | None = None,
    store: ExperimentStore | None = None,
) -> list[dict]:
    """Look up heuristics from the knowledge base.

    Args:
        classifier: Classifier type (fp, ep, esg).
        phase: Optional phase filter (not yet used, reserved for future).
        store: Optional ExperimentStore instance.

    Returns:
        List of heuristic dicts with trigger, action, confidence, stats.
    """
    store = store or ExperimentStore()
    kb = store.load_knowledge(classifier)

    results = []
    for h in kb.heuristics:
        results.append({
            "id": h.id,
            "trigger": h.trigger,
            "action": h.action,
            "confidence": h.confidence,
            "times_applied": h.times_applied,
            "times_successful": h.times_successful,
            "success_rate": (
                round(h.times_successful / h.times_applied, 2)
                if h.times_applied > 0
                else None
            ),
        })

    return results


def record_replay_decision(
    classifier: str,
    experiment_id: str,
    phase: str,
    trigger: str,
    chosen: str,
    reasoning: str,
    trigger_type: str = "metric_observation",
    options: list[dict[str, str]] | None = None,
    store: ExperimentStore | None = None,
) -> str:
    """Record a decision made during skill replay.

    Args:
        classifier: Classifier type.
        experiment_id: ID of the current experiment.
        phase: Workflow phase (e.g., feature_engineering).
        trigger: What prompted the decision.
        chosen: The option chosen.
        reasoning: Why this option was chosen.
        trigger_type: Category of trigger.
        options: Optional list of option dicts with id/description.
        store: Optional ExperimentStore instance.

    Returns:
        The decision_id of the saved decision.
    """
    store = store or ExperimentStore()

    timestamp = datetime.now(timezone.utc)
    decision_id = f"dec_{classifier}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    decision_options = []
    if options:
        for opt in options:
            decision_options.append(
                DecisionOption(
                    id=opt.get("id", ""),
                    description=opt.get("description", ""),
                    expected_outcome=opt.get("expected_outcome", ""),
                )
            )

    decision = Decision(
        decision_id=decision_id,
        experiment_id=experiment_id,
        classifier=classifier,
        timestamp=timestamp,
        phase=phase,
        trigger_type=trigger_type,
        trigger_description=trigger,
        options=decision_options,
        chosen=chosen,
        reasoning=reasoning,
    )

    store.save_decision(decision)
    return decision_id


def update_heuristic_outcome(
    classifier: str,
    trigger: str,
    success: bool,
    store: ExperimentStore | None = None,
) -> bool:
    """Update a heuristic's outcome counters based on trigger match.

    Finds heuristics whose trigger contains the given trigger text,
    increments times_applied and (if success) times_successful.
    Upgrades confidence when enough successful applications accumulate.

    Args:
        classifier: Classifier type.
        trigger: Trigger text to match against heuristic triggers.
        success: Whether the heuristic's action led to a good outcome.
        store: Optional ExperimentStore instance.

    Returns:
        True if a matching heuristic was found and updated, False otherwise.
    """
    store = store or ExperimentStore()
    kb = store.load_knowledge(classifier)

    found = False
    for h in kb.heuristics:
        if trigger.lower() in h.trigger.lower() or h.trigger.lower() in trigger.lower():
            h.times_applied += 1
            if success:
                h.times_successful += 1
            # Upgrade confidence based on track record
            if h.times_applied >= 3 and h.times_successful / h.times_applied >= 0.7:
                h.confidence = "high"
            elif h.times_applied >= 2 and h.times_successful / h.times_applied >= 0.5:
                h.confidence = "medium"
            found = True

    if found:
        store.save_knowledge(kb)

    return found


def main() -> int:
    """CLI entry point for experiment log utilities."""
    parser = argparse.ArgumentParser(
        prog="experiment_log",
        description="Experiment log CLI for knowledge base queries and decision recording",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # heuristics command
    heur_parser = subparsers.add_parser(
        "heuristics", help="Look up heuristics from the knowledge base"
    )
    heur_parser.add_argument(
        "--classifier", required=True, help="Classifier type (fp, ep, esg)"
    )
    heur_parser.add_argument(
        "--phase", help="Optional phase filter (reserved for future use)"
    )

    # record-decision command
    rec_parser = subparsers.add_parser(
        "record-decision", help="Record a decision made during replay"
    )
    rec_parser.add_argument(
        "--classifier", required=True, help="Classifier type"
    )
    rec_parser.add_argument(
        "--experiment-id", required=True, help="Current experiment ID"
    )
    rec_parser.add_argument(
        "--phase", required=True, help="Workflow phase"
    )
    rec_parser.add_argument(
        "--trigger", required=True, help="What prompted the decision"
    )
    rec_parser.add_argument(
        "--chosen", required=True, help="The option chosen"
    )
    rec_parser.add_argument(
        "--reasoning", required=True, help="Why this option was chosen"
    )
    rec_parser.add_argument(
        "--trigger-type",
        default="metric_observation",
        help="Category of trigger (default: metric_observation)",
    )

    # update-heuristic command
    upd_parser = subparsers.add_parser(
        "update-heuristic", help="Update heuristic outcome counters"
    )
    upd_parser.add_argument(
        "--classifier", required=True, help="Classifier type"
    )
    upd_parser.add_argument(
        "--trigger", required=True, help="Trigger text to match"
    )
    upd_parser.add_argument(
        "--success",
        required=True,
        choices=["true", "false"],
        help="Whether the outcome was successful",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "heuristics":
        results = lookup_heuristics(
            classifier=args.classifier,
            phase=getattr(args, "phase", None),
        )
        if not results:
            print(f"No heuristics found for {args.classifier} classifier.")
            return 0

        print(f"Heuristics for {args.classifier} classifier:")
        print("-" * 60)
        for h in results:
            success_info = ""
            if h["success_rate"] is not None:
                success_info = f" ({h['times_successful']}/{h['times_applied']} successful)"
            print(f"  [{h['confidence'].upper()}] {h['trigger']}")
            print(f"    → {h['action']}{success_info}")
            print()
        return 0

    elif args.command == "record-decision":
        decision_id = record_replay_decision(
            classifier=args.classifier,
            experiment_id=args.experiment_id,
            phase=args.phase,
            trigger=args.trigger,
            chosen=args.chosen,
            reasoning=args.reasoning,
            trigger_type=args.trigger_type,
        )
        print(f"Recorded decision: {decision_id}")
        return 0

    elif args.command == "update-heuristic":
        success = args.success.lower() == "true"
        found = update_heuristic_outcome(
            classifier=args.classifier,
            trigger=args.trigger,
            success=success,
        )
        if found:
            print(f"Updated heuristic matching trigger: {args.trigger}")
        else:
            print(f"No heuristic found matching trigger: {args.trigger}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
