"""Tests for experiment log CLI utilities."""

from datetime import datetime, timezone

import pytest

from src.experiment_log.cli import (
    lookup_heuristics,
    record_replay_decision,
    update_heuristic_outcome,
)
from src.experiment_log.models import Heuristic, KnowledgeBase
from src.experiment_log.store import ExperimentStore


@pytest.fixture
def store(tmp_path):
    """Create an ExperimentStore with a temporary directory."""
    return ExperimentStore(base_dir=tmp_path)


@pytest.fixture
def populated_store(store):
    """Store with pre-populated heuristics."""
    kb = KnowledgeBase(
        classifier="fp",
        last_updated=datetime.now(timezone.utc),
        heuristics=[
            Heuristic(
                id="heur_fp_1",
                trigger="NER features not contributing",
                action="remove_ner: Remove NER features to simplify model",
                confidence="low",
                derived_from=["recording:model-training-fp"],
                times_applied=1,
                times_successful=1,
            ),
            Heuristic(
                id="heur_fp_2",
                trigger="High class imbalance",
                action="class_weight: Use class_weight='balanced'",
                confidence="medium",
                derived_from=["recording:model-training-fp"],
                times_applied=2,
                times_successful=1,
            ),
        ],
    )
    store.save_knowledge(kb)
    return store


class TestLookupHeuristics:
    def test_empty(self, store):
        results = lookup_heuristics("fp", store=store)
        assert results == []

    def test_returns_all(self, populated_store):
        results = lookup_heuristics("fp", store=populated_store)
        assert len(results) == 2
        assert results[0]["trigger"] == "NER features not contributing"
        assert results[0]["confidence"] == "low"
        assert results[0]["success_rate"] == 1.0
        assert results[1]["trigger"] == "High class imbalance"
        assert results[1]["success_rate"] == 0.5


class TestRecordReplayDecision:
    def test_creates_decision(self, store):
        decision_id = record_replay_decision(
            classifier="fp",
            experiment_id="fp_20260226_143000",
            phase="feature_engineering",
            trigger="NER features not contributing",
            chosen="remove_ner",
            reasoning="F2 unchanged with NER",
            store=store,
        )
        assert decision_id.startswith("dec_fp_")

        # Verify it was saved
        decisions = store.load_decisions("fp_20260226_143000")
        assert len(decisions) == 1
        assert decisions[0].trigger_description == "NER features not contributing"
        assert decisions[0].chosen == "remove_ner"
        assert decisions[0].reasoning == "F2 unchanged with NER"


class TestUpdateHeuristicOutcome:
    def test_success(self, populated_store):
        found = update_heuristic_outcome(
            classifier="fp",
            trigger="NER features not contributing",
            success=True,
            store=populated_store,
        )
        assert found is True

        # Verify counters updated
        kb = populated_store.load_knowledge("fp")
        h = kb.heuristics[0]
        assert h.times_applied == 2
        assert h.times_successful == 2

    def test_failure_outcome(self, populated_store):
        found = update_heuristic_outcome(
            classifier="fp",
            trigger="NER features not contributing",
            success=False,
            store=populated_store,
        )
        assert found is True

        kb = populated_store.load_knowledge("fp")
        h = kb.heuristics[0]
        assert h.times_applied == 2
        assert h.times_successful == 1  # unchanged

    def test_not_found(self, populated_store):
        found = update_heuristic_outcome(
            classifier="fp",
            trigger="completely unrelated trigger",
            success=True,
            store=populated_store,
        )
        assert found is False

    def test_confidence_upgrade(self, store):
        """After enough successful applications, confidence should upgrade."""
        kb = KnowledgeBase(
            classifier="fp",
            last_updated=datetime.now(timezone.utc),
            heuristics=[
                Heuristic(
                    id="heur_fp_1",
                    trigger="test trigger pattern",
                    action="test action",
                    confidence="low",
                    times_applied=2,
                    times_successful=2,
                ),
            ],
        )
        store.save_knowledge(kb)

        # Third successful application should upgrade to high
        update_heuristic_outcome(
            classifier="fp",
            trigger="test trigger pattern",
            success=True,
            store=store,
        )

        kb = store.load_knowledge("fp")
        assert kb.heuristics[0].confidence == "high"
        assert kb.heuristics[0].times_applied == 3
        assert kb.heuristics[0].times_successful == 3
