"""Bridge between workflow learning analysis and experiment log.

Converts extracted decisions from recording analysis into experiment log
Decision entries and seeds Pattern/Heuristic entries in the knowledge base.
"""

import logging
from datetime import datetime, timezone

from src.experiment_log.models import (
    Decision,
    DecisionOption,
    Heuristic,
    Pattern,
)
from src.experiment_log.store import ExperimentStore

from .models import AnalysisResult, ExtractedDecision

logger = logging.getLogger(__name__)


def _infer_classifier(workflow_name: str) -> str | None:
    """Infer classifier type from workflow name.

    Examples:
        "model-training-fp" → "fp"
        "model-training-ep" → "ep"
        "fp-retraining" → "fp"
    """
    name_lower = workflow_name.lower()
    for classifier in ("fp", "ep", "esg"):
        if classifier in name_lower:
            return classifier
    return None


def save_decisions_from_analysis(
    analysis: AnalysisResult,
    workflow_name: str,
    experiment_id: str | None = None,
    classifier: str | None = None,
    store: ExperimentStore | None = None,
) -> list[str]:
    """Save extracted decisions from analysis as experiment log Decision entries.

    Args:
        analysis: The analysis result containing extracted decisions.
        workflow_name: Name of the workflow recording.
        experiment_id: ID of linked experiment. If None, uses placeholder.
        classifier: Classifier type. If None, inferred from workflow_name.
        store: Optional ExperimentStore instance.

    Returns:
        List of saved decision IDs.
    """
    if not analysis.decisions:
        return []

    try:
        store = store or ExperimentStore()
        classifier = classifier or _infer_classifier(workflow_name) or "unknown"

        # Use placeholder experiment ID if none provided
        if not experiment_id:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            experiment_id = f"recording_{timestamp}"

        saved_ids = []
        for i, dec in enumerate(analysis.decisions):
            timestamp = datetime.now(timezone.utc)
            decision_id = f"dec_{classifier}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i}"

            # Convert options
            options = []
            for opt in dec.options:
                options.append(
                    DecisionOption(
                        id=opt.get("id", f"opt_{len(options)}"),
                        description=opt.get("description", ""),
                        expected_outcome=opt.get("expected_outcome", ""),
                    )
                )

            decision = Decision(
                decision_id=decision_id,
                experiment_id=experiment_id,
                classifier=classifier,
                timestamp=timestamp,
                phase=dec.phase or "unknown",
                trigger_type=dec.trigger_type,
                trigger_description=dec.trigger,
                trigger_evidence=dec.evidence,
                options=options,
                chosen=dec.chosen,
                reasoning=dec.reasoning,
                outcome_result=dec.outcome,
            )

            store.save_decision(decision)
            saved_ids.append(decision_id)

        return saved_ids

    except Exception as e:
        logger.warning(f"Failed to save decisions from analysis: {e}")
        return []


def seed_knowledge_from_decisions(
    analysis: AnalysisResult,
    workflow_name: str,
    classifier: str | None = None,
    store: ExperimentStore | None = None,
) -> dict[str, int]:
    """Seed knowledge base patterns and heuristics from extracted decisions.

    Creates Pattern entries when a decision has trigger + outcome (observed cause-effect).
    Creates Heuristic entries when a decision has trigger + chosen + reasoning (actionable rule).
    Deduplicates against existing knowledge base entries.

    Args:
        analysis: The analysis result containing extracted decisions.
        workflow_name: Name of the workflow recording.
        classifier: Classifier type. If None, inferred from workflow_name.
        store: Optional ExperimentStore instance.

    Returns:
        Dict with counts: {"patterns_added": N, "heuristics_added": N}
    """
    result = {"patterns_added": 0, "heuristics_added": 0}

    if not analysis.decisions:
        return result

    try:
        store = store or ExperimentStore()
        classifier = classifier or _infer_classifier(workflow_name) or "unknown"

        kb = store.load_knowledge(classifier)

        # Collect existing triggers for deduplication
        existing_pattern_texts = {p.pattern.lower() for p in kb.patterns}
        existing_heuristic_triggers = {h.trigger.lower() for h in kb.heuristics}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for dec in analysis.decisions:
            # Create Pattern if trigger + outcome exist
            if dec.trigger and dec.outcome:
                pattern_text = f"When {dec.trigger}, then {dec.outcome}"
                if pattern_text.lower() not in existing_pattern_texts:
                    pattern_id = f"pat_{classifier}_{len(kb.patterns) + 1}"
                    kb.patterns.append(
                        Pattern(
                            id=pattern_id,
                            category=dec.phase or "general",
                            pattern=pattern_text,
                            confidence="low",
                            evidence=[{
                                "source": f"recording:{workflow_name}",
                                "detail": dec.reasoning,
                            }],
                            first_observed=today,
                            last_confirmed=today,
                        )
                    )
                    existing_pattern_texts.add(pattern_text.lower())
                    result["patterns_added"] += 1

            # Create Heuristic if trigger + chosen + reasoning exist
            if dec.trigger and dec.chosen and dec.reasoning:
                if dec.trigger.lower() not in existing_heuristic_triggers:
                    heuristic_id = f"heur_{classifier}_{len(kb.heuristics) + 1}"
                    kb.heuristics.append(
                        Heuristic(
                            id=heuristic_id,
                            trigger=dec.trigger,
                            action=f"{dec.chosen}: {dec.reasoning}",
                            confidence="low",
                            derived_from=[f"recording:{workflow_name}"],
                            times_applied=0,
                            times_successful=0,
                        )
                    )
                    existing_heuristic_triggers.add(dec.trigger.lower())
                    result["heuristics_added"] += 1

        # Save if anything changed
        if result["patterns_added"] > 0 or result["heuristics_added"] > 0:
            store.save_knowledge(kb)

        return result

    except Exception as e:
        logger.warning(f"Failed to seed knowledge from decisions: {e}")
        return result
