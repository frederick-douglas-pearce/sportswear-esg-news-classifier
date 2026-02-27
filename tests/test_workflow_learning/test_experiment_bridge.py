"""Tests for workflow learning ↔ experiment log bridge."""

from datetime import datetime, timezone

import pytest

from src.experiment_log.store import ExperimentStore
from src.workflow_learning.experiment_bridge import (
    _infer_classifier,
    save_decisions_from_analysis,
    seed_knowledge_from_decisions,
)
from src.workflow_learning.models import AnalysisResult, ExtractedDecision


@pytest.fixture
def store(tmp_path):
    """Create an ExperimentStore with a temporary directory."""
    return ExperimentStore(base_dir=tmp_path)


@pytest.fixture
def sample_decisions():
    """Create sample extracted decisions."""
    return [
        ExtractedDecision(
            trigger="NER features not contributing to F2 score",
            trigger_type="metric_observation",
            options=[
                {"id": "keep_ner", "description": "Keep NER features", "expected_outcome": "No change"},
                {"id": "remove_ner", "description": "Remove NER features", "expected_outcome": "Faster training"},
            ],
            chosen="remove_ner",
            reasoning="F2 unchanged with NER, removing simplifies model",
            phase="feature_engineering",
            evidence=["NER features show zero importance in SHAP analysis"],
            outcome="F2 remained at 0.974 after removal",
        ),
        ExtractedDecision(
            trigger="High class imbalance in training data",
            trigger_type="error_pattern",
            options=[
                {"id": "smote", "description": "Apply SMOTE oversampling"},
                {"id": "class_weight", "description": "Use class_weight='balanced'"},
            ],
            chosen="class_weight",
            reasoning="SMOTE can introduce noise, class_weight is simpler",
            phase="model_selection",
            evidence=["Class ratio is 8:1"],
            outcome="",  # No outcome visible
        ),
    ]


@pytest.fixture
def analysis_with_decisions(sample_decisions):
    """Create an AnalysisResult with decisions."""
    return AnalysisResult(
        success=True,
        summary="Test analysis",
        decisions=sample_decisions,
    )


@pytest.fixture
def analysis_without_decisions():
    """Create an AnalysisResult without decisions."""
    return AnalysisResult(
        success=True,
        summary="Test analysis",
        decisions=[],
    )


class TestInferClassifier:
    def test_infer_fp(self):
        assert _infer_classifier("model-training-fp") == "fp"

    def test_infer_ep(self):
        assert _infer_classifier("ep-retraining") == "ep"

    def test_infer_esg(self):
        assert _infer_classifier("esg-multi-label") == "esg"

    def test_infer_unknown(self):
        assert _infer_classifier("generic-workflow") is None


class TestSaveDecisionsFromAnalysis:
    def test_save_decisions(self, analysis_with_decisions, store):
        saved_ids = save_decisions_from_analysis(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        assert len(saved_ids) == 2
        assert all(id.startswith("dec_fp_") for id in saved_ids)

    def test_save_decisions_empty_list(self, analysis_without_decisions, store):
        saved_ids = save_decisions_from_analysis(
            analysis=analysis_without_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        assert saved_ids == []

    def test_save_decisions_infer_classifier(self, analysis_with_decisions, store):
        saved_ids = save_decisions_from_analysis(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        # Verify decisions were saved under fp classifier
        decisions = store.load_decisions("recording_00000000_000000")
        # They won't match the dummy experiment_id, but let's verify files exist
        decisions_dir = store._decisions_dir("fp")
        decision_files = list(decisions_dir.glob("*.yaml"))
        assert len(decision_files) == 2

    def test_save_decisions_explicit_experiment_id(self, analysis_with_decisions, store):
        saved_ids = save_decisions_from_analysis(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            experiment_id="fp_20260226_143000",
            store=store,
        )
        assert len(saved_ids) == 2
        # Verify decisions link to the provided experiment
        decisions = store.load_decisions("fp_20260226_143000")
        assert len(decisions) == 2
        assert all(d.experiment_id == "fp_20260226_143000" for d in decisions)

    def test_save_decisions_explicit_classifier(self, analysis_with_decisions, store):
        saved_ids = save_decisions_from_analysis(
            analysis=analysis_with_decisions,
            workflow_name="generic-workflow",
            classifier="ep",
            store=store,
        )
        assert len(saved_ids) == 2
        assert all(id.startswith("dec_ep_") for id in saved_ids)


class TestSeedKnowledgeFromDecisions:
    def test_seed_patterns(self, analysis_with_decisions, store):
        """Decisions with trigger + outcome should create Pattern entries."""
        result = seed_knowledge_from_decisions(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        # Only first decision has an outcome
        assert result["patterns_added"] == 1

        kb = store.load_knowledge("fp")
        assert len(kb.patterns) == 1
        assert "NER features not contributing" in kb.patterns[0].pattern
        assert "remained at 0.974" in kb.patterns[0].pattern

    def test_seed_heuristics(self, analysis_with_decisions, store):
        """Decisions with trigger + chosen + reasoning should create Heuristic entries."""
        result = seed_knowledge_from_decisions(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        assert result["heuristics_added"] == 2

        kb = store.load_knowledge("fp")
        assert len(kb.heuristics) == 2
        # Check first heuristic
        h0 = kb.heuristics[0]
        assert "NER features not contributing" in h0.trigger
        assert h0.confidence == "low"
        assert h0.times_applied == 0

    def test_seed_knowledge_deduplicates(self, analysis_with_decisions, store):
        """Running seed twice should not create duplicates."""
        seed_knowledge_from_decisions(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        result = seed_knowledge_from_decisions(
            analysis=analysis_with_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        assert result["patterns_added"] == 0
        assert result["heuristics_added"] == 0

        kb = store.load_knowledge("fp")
        assert len(kb.patterns) == 1
        assert len(kb.heuristics) == 2

    def test_seed_empty_decisions(self, analysis_without_decisions, store):
        result = seed_knowledge_from_decisions(
            analysis=analysis_without_decisions,
            workflow_name="model-training-fp",
            store=store,
        )
        assert result == {"patterns_added": 0, "heuristics_added": 0}
