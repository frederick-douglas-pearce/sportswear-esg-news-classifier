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


def _resolve_classifier(classifier: str | None, workflow_name: str) -> str:
    """Resolve classifier from explicit value or infer from workflow name."""
    resolved = classifier or _infer_classifier(workflow_name) or "unknown"
    if resolved == "unknown":
        logger.warning(
            "Could not infer classifier from workflow '%s', using 'unknown'",
            workflow_name,
        )
    return resolved


def _build_decision_options(raw_options: list[dict[str, str]]) -> list[DecisionOption]:
    """Convert raw option dicts from extracted decisions to DecisionOption models.

    Handles missing keys gracefully with sensible defaults.
    """
    options = []
    for idx, opt in enumerate(raw_options):
        if not isinstance(opt, dict):
            logger.warning(
                "Skipping non-dict option at index %d: %s", idx, type(opt).__name__
            )
            continue
        try:
            options.append(
                DecisionOption(
                    id=opt.get("id", f"opt_{idx}"),
                    description=opt.get("description", ""),
                    expected_outcome=opt.get("expected_outcome", ""),
                )
            )
        except (TypeError, ValueError) as e:
            logger.warning("Skipping invalid option at index %d: %s", idx, e)
    return options


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
        logger.debug("No decisions to save for workflow '%s'", workflow_name)
        return []

    classifier = _resolve_classifier(classifier, workflow_name)

    try:
        store = store or ExperimentStore()
    except Exception as e:
        logger.error("Failed to initialize ExperimentStore: %s", e)
        return []

    if not experiment_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"recording_{timestamp}"
        logger.debug(
            "No experiment_id provided, using placeholder: %s", experiment_id
        )

    saved_ids = []
    for i, dec in enumerate(analysis.decisions):
        try:
            timestamp = datetime.now(timezone.utc)
            decision_id = f"dec_{classifier}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i}"

            options = _build_decision_options(dec.options)

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
            logger.debug(
                "Saved decision %d/%d: %s (trigger: %s)",
                i + 1,
                len(analysis.decisions),
                decision_id,
                dec.trigger[:80],
            )
        except (TypeError, ValueError) as e:
            logger.error(
                "Invalid data for decision %d (trigger: '%s'): %s",
                i,
                dec.trigger[:80] if dec.trigger else "empty",
                e,
            )
        except OSError as e:
            logger.error(
                "File I/O error saving decision %d (%s): %s",
                i,
                decision_id,
                e,
            )
        except Exception as e:
            logger.error(
                "Unexpected error saving decision %d: %s", i, e, exc_info=True
            )

    logger.info(
        "Saved %d/%d decisions for workflow '%s'",
        len(saved_ids),
        len(analysis.decisions),
        workflow_name,
    )
    return saved_ids


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
        logger.debug("No decisions to seed knowledge from for workflow '%s'", workflow_name)
        return result

    classifier = _resolve_classifier(classifier, workflow_name)

    try:
        store = store or ExperimentStore()
    except Exception as e:
        logger.error("Failed to initialize ExperimentStore: %s", e)
        return result

    try:
        kb = store.load_knowledge(classifier)
    except Exception as e:
        logger.error(
            "Failed to load knowledge base for classifier '%s': %s",
            classifier,
            e,
        )
        return result

    existing_pattern_texts = {p.pattern.lower() for p in kb.patterns}
    existing_heuristic_triggers = {h.trigger.lower() for h in kb.heuristics}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for i, dec in enumerate(analysis.decisions):
        # Create Pattern if trigger + outcome exist
        if dec.trigger and dec.outcome:
            pattern_text = f"When {dec.trigger}, then {dec.outcome}"
            if pattern_text.lower() not in existing_pattern_texts:
                try:
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
                    logger.debug(
                        "Created pattern from decision %d: %s",
                        i,
                        pattern_text[:100],
                    )
                except (TypeError, ValueError) as e:
                    logger.error(
                        "Invalid data for pattern from decision %d: %s", i, e
                    )
            else:
                logger.debug(
                    "Skipping duplicate pattern from decision %d: %s",
                    i,
                    pattern_text[:80],
                )

        # Create Heuristic if trigger + chosen + reasoning exist
        if dec.trigger and dec.chosen and dec.reasoning:
            if dec.trigger.lower() not in existing_heuristic_triggers:
                try:
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
                    logger.debug(
                        "Created heuristic from decision %d: trigger='%s'",
                        i,
                        dec.trigger[:80],
                    )
                except (TypeError, ValueError) as e:
                    logger.error(
                        "Invalid data for heuristic from decision %d: %s", i, e
                    )
            else:
                logger.debug(
                    "Skipping duplicate heuristic from decision %d: trigger='%s'",
                    i,
                    dec.trigger[:80],
                )

    # Save if anything changed
    if result["patterns_added"] > 0 or result["heuristics_added"] > 0:
        try:
            store.save_knowledge(kb)
            logger.info(
                "Saved knowledge base for '%s': %d patterns, %d heuristics added",
                classifier,
                result["patterns_added"],
                result["heuristics_added"],
            )
        except OSError as e:
            logger.error(
                "File I/O error saving knowledge base for '%s': %s",
                classifier,
                e,
            )
            # Reset counts since save failed - entries were not persisted
            result["patterns_added"] = 0
            result["heuristics_added"] = 0
        except Exception as e:
            logger.error(
                "Unexpected error saving knowledge base for '%s': %s",
                classifier,
                e,
                exc_info=True,
            )
            result["patterns_added"] = 0
            result["heuristics_added"] = 0
    else:
        logger.debug(
            "No new knowledge to seed for workflow '%s'",
            workflow_name,
        )

    return result
