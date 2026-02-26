"""Tests for experiment log store."""

from datetime import datetime, timezone

import pytest
import yaml

from src.experiment_log.models import (
    Decision,
    DecisionOption,
    ExperimentAction,
    ExperimentChange,
    ExperimentEntry,
    ExperimentObservation,
    ExperimentReflection,
    ExperimentReward,
    ExperimentState,
    KnowledgeBase,
    Pattern,
    SharedKnowledgeBase,
    SharedPattern,
)
from src.experiment_log.store import ExperimentStore


@pytest.fixture
def store(tmp_path):
    """Create an ExperimentStore with a temporary directory."""
    return ExperimentStore(base_dir=tmp_path)


@pytest.fixture
def sample_entry():
    """Create a sample ExperimentEntry."""
    return ExperimentEntry(
        experiment_id="fp_20260226_143000",
        classifier="fp",
        created_at=datetime(2026, 2, 26, 14, 30, tzinfo=timezone.utc),
        status="completed",
        tags=["ablation", "ner"],
        state=ExperimentState(
            production_version="v2.1.0",
            production_metrics={"test_f2": 0.974},
        ),
        action=ExperimentAction(
            type="ablation",
            summary="Remove NER features",
            changes=[
                ExperimentChange(
                    component="feature_config",
                    field="ner_enabled",
                    old_value=True,
                    new_value=False,
                ),
            ],
            hypothesis="NER features are dispensable",
        ),
        observation=ExperimentObservation(
            metrics={"test_f2": 0.960, "test_recall": 0.975},
            delta={"test_f2": -0.014},
        ),
        reward=ExperimentReward(
            primary_value=0.960,
            primary_target=0.974,
            target_met=False,
        ),
        reflection=ExperimentReflection(
            summary="NER features are critical",
            hypothesis_result="refuted",
        ),
    )


@pytest.fixture
def sample_decision():
    """Create a sample Decision."""
    return Decision(
        decision_id="dec_fp_20260226_150000",
        experiment_id="fp_20260226_143000",
        classifier="fp",
        timestamp=datetime(2026, 2, 26, 15, 0, tzinfo=timezone.utc),
        phase="feature_engineering",
        trigger_type="metric_observation",
        trigger_description="SHAP shows low NER importance",
        options=[
            DecisionOption(id="keep", description="Keep NER"),
            DecisionOption(id="remove", description="Remove NER"),
        ],
        chosen="keep",
        reasoning="Critical for edge cases",
    )


class TestCreateExperiment:
    def test_writes_yaml_at_correct_path(self, store, sample_entry):
        path = store.create_experiment(sample_entry)
        assert path.exists()
        assert path.name == "exp_20260226_143000.yaml"
        assert path.parent.name == "fp"

    def test_file_contains_correct_data(self, store, sample_entry):
        path = store.create_experiment(sample_entry)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["experiment_id"] == "fp_20260226_143000"
        assert data["classifier"] == "fp"
        assert data["action"]["type"] == "ablation"

    def test_updates_index(self, store, sample_entry):
        store.create_experiment(sample_entry)
        index = store.load_index()
        assert "fp" in index.experiments
        assert len(index.experiments["fp"]) == 1
        assert index.experiments["fp"][0].id == "fp_20260226_143000"
        assert index.summary["fp"].total_experiments == 1

    def test_duplicate_raises(self, store, sample_entry):
        store.create_experiment(sample_entry)
        with pytest.raises(FileExistsError):
            store.create_experiment(sample_entry)


class TestLoadExperiment:
    def test_roundtrip(self, store, sample_entry):
        store.create_experiment(sample_entry)
        loaded = store.load_experiment("fp_20260226_143000")
        assert loaded.experiment_id == sample_entry.experiment_id
        assert loaded.classifier == "fp"
        assert loaded.status == "completed"
        assert loaded.state.production_metrics["test_f2"] == 0.974
        assert loaded.action.changes[0].old_value is True
        assert loaded.observation.metrics["test_f2"] == 0.960

    def test_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_experiment("fp_20260226_999999")


class TestUpdateExperiment:
    def test_modifies_file_and_index(self, store, sample_entry):
        store.create_experiment(sample_entry)

        # Update entry
        sample_entry.status = "completed"
        sample_entry.observation.metrics["test_f2"] = 0.965
        sample_entry.reward.promoted = True
        sample_entry.reward.promoted_as = "v2.2.0"
        store.update_experiment(sample_entry)

        # Verify file
        loaded = store.load_experiment("fp_20260226_143000")
        assert loaded.observation.metrics["test_f2"] == 0.965
        assert loaded.reward.promoted is True

        # Verify index
        index = store.load_index()
        entry = index.experiments["fp"][0]
        assert entry.promoted is True
        assert index.summary["fp"].total_promoted == 1
        assert index.summary["fp"].current_production == "v2.2.0"


class TestListExperiments:
    def test_newest_first(self, store):
        for ts in ["100000", "120000", "140000"]:
            entry = ExperimentEntry(
                experiment_id=f"fp_20260226_{ts}",
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary=f"Run {ts}"),
            )
            store.create_experiment(entry)

        entries = store.list_experiments("fp")
        assert len(entries) == 3
        assert entries[0].id == "fp_20260226_140000"
        assert entries[2].id == "fp_20260226_100000"

    def test_limit(self, store):
        for i in range(5):
            entry = ExperimentEntry(
                experiment_id=f"fp_20260226_{i:06d}",
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary=f"Run {i}"),
            )
            store.create_experiment(entry)

        entries = store.list_experiments("fp", limit=3)
        assert len(entries) == 3

    def test_empty_classifier(self, store):
        entries = store.list_experiments("ep")
        assert entries == []


class TestKnowledge:
    def test_load_returns_empty_for_new_classifier(self, store):
        kb = store.load_knowledge("fp")
        assert kb.classifier == "fp"
        assert kb.version == 1
        assert kb.patterns == []

    def test_save_and_load_roundtrip(self, store):
        kb = store.load_knowledge("fp")
        kb.patterns.append(
            Pattern(id="p1", category="data_quality", pattern="Test pattern")
        )
        store.save_knowledge(kb)

        loaded = store.load_knowledge("fp")
        assert loaded.version == 2  # incremented
        assert len(loaded.patterns) == 1
        assert loaded.patterns[0].id == "p1"

    def test_save_increments_version(self, store):
        kb = store.load_knowledge("ep")
        store.save_knowledge(kb)  # v1 -> v2
        store.save_knowledge(kb)  # v2 -> v3 (kb object mutated)

        loaded = store.load_knowledge("ep")
        assert loaded.version == 3

    def test_shared_knowledge(self, store):
        skb = store.load_shared_knowledge()
        assert skb.version == 1
        assert skb.patterns == []

        skb.patterns.append(
            SharedPattern(
                id="sp1",
                pattern="TF-IDF works well for < 3000 samples",
                applies_to=["fp"],
                source_classifier="fp",
            )
        )
        store.save_shared_knowledge(skb)

        loaded = store.load_shared_knowledge()
        assert loaded.version == 2
        assert len(loaded.patterns) == 1
        assert "fp" in loaded.patterns[0].applies_to


class TestDecisions:
    def test_save_creates_file(self, store, sample_decision):
        path = store.save_decision(sample_decision)
        assert path.exists()
        assert "decisions" in str(path)
        assert path.name == "dec_fp_20260226_150000.yaml"

    def test_load_filters_by_experiment(self, store, sample_decision):
        store.save_decision(sample_decision)

        # Add another decision for a different experiment
        other = Decision(
            decision_id="dec_fp_20260226_160000",
            experiment_id="fp_20260226_OTHER",
            classifier="fp",
            timestamp=datetime(2026, 2, 26, 16, 0, tzinfo=timezone.utc),
            phase="model_selection",
            trigger_type="comparison",
            trigger_description="Compare models",
            chosen="rf",
            reasoning="Better F2",
        )
        store.save_decision(other)

        decisions = store.load_decisions("fp_20260226_143000")
        assert len(decisions) == 1
        assert decisions[0].decision_id == "dec_fp_20260226_150000"

    def test_load_empty(self, store):
        decisions = store.load_decisions("fp_20260226_NONEXIST")
        assert decisions == []


class TestGetRecentExperiments:
    def test_returns_full_entries(self, store):
        for ts in ["100000", "120000", "140000"]:
            entry = ExperimentEntry(
                experiment_id=f"fp_20260226_{ts}",
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary=f"Run {ts}"),
                observation=ExperimentObservation(metrics={"test_f2": 0.95}),
            )
            store.create_experiment(entry)

        recent = store.get_recent_experiments("fp", n=2)
        assert len(recent) == 2
        assert recent[0].experiment_id == "fp_20260226_140000"
        assert recent[0].observation.metrics["test_f2"] == 0.95


class TestGetParentChain:
    def test_follows_parents(self, store):
        # Create chain: child -> parent -> grandparent
        for eid, parent in [
            ("fp_20260226_100000", None),
            ("fp_20260226_120000", "fp_20260226_100000"),
            ("fp_20260226_140000", "fp_20260226_120000"),
        ]:
            entry = ExperimentEntry(
                experiment_id=eid,
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                parent_experiment=parent,
                action=ExperimentAction(type="test", summary=f"Run {eid}"),
            )
            store.create_experiment(entry)

        chain = store.get_parent_chain("fp_20260226_140000", depth=5)
        assert len(chain) == 3
        assert chain[0].experiment_id == "fp_20260226_140000"
        assert chain[1].experiment_id == "fp_20260226_120000"
        assert chain[2].experiment_id == "fp_20260226_100000"

    def test_depth_limit(self, store):
        for eid, parent in [
            ("fp_20260226_100000", None),
            ("fp_20260226_120000", "fp_20260226_100000"),
            ("fp_20260226_140000", "fp_20260226_120000"),
        ]:
            entry = ExperimentEntry(
                experiment_id=eid,
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                parent_experiment=parent,
                action=ExperimentAction(type="test", summary=f"Run {eid}"),
            )
            store.create_experiment(entry)

        chain = store.get_parent_chain("fp_20260226_140000", depth=2)
        assert len(chain) == 2

    def test_missing_parent(self, store):
        entry = ExperimentEntry(
            experiment_id="fp_20260226_140000",
            classifier="fp",
            created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
            parent_experiment="fp_20260226_MISSING",
            action=ExperimentAction(type="test", summary="Orphan"),
        )
        store.create_experiment(entry)

        chain = store.get_parent_chain("fp_20260226_140000")
        assert len(chain) == 1


class TestFindExperimentsByTag:
    def test_filters_by_tag(self, store):
        for eid, tags in [
            ("fp_20260226_100000", ["ablation", "ner"]),
            ("fp_20260226_120000", ["retraining"]),
            ("fp_20260226_140000", ["ablation", "tfidf"]),
        ]:
            entry = ExperimentEntry(
                experiment_id=eid,
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                tags=tags,
                action=ExperimentAction(type="test", summary=f"Run {eid}"),
            )
            store.create_experiment(entry)

        ablation_entries = store.find_experiments_by_tag("fp", "ablation")
        assert len(ablation_entries) == 2
        ids = {e.id for e in ablation_entries}
        assert "fp_20260226_100000" in ids
        assert "fp_20260226_140000" in ids

    def test_no_matches(self, store):
        entry = ExperimentEntry(
            experiment_id="fp_20260226_100000",
            classifier="fp",
            created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
            tags=["retraining"],
            action=ExperimentAction(type="test", summary="Run"),
        )
        store.create_experiment(entry)

        results = store.find_experiments_by_tag("fp", "nonexistent")
        assert results == []


class TestIndexTracking:
    def test_best_f2_tracks_maximum(self, store):
        for eid, f2 in [
            ("fp_20260226_100000", 0.95),
            ("fp_20260226_120000", 0.98),
            ("fp_20260226_140000", 0.96),
        ]:
            entry = ExperimentEntry(
                experiment_id=eid,
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary="Run"),
                observation=ExperimentObservation(metrics={"test_f2": f2}),
            )
            store.create_experiment(entry)

        index = store.load_index()
        assert index.summary["fp"].best_test_f2_ever == 0.98

    def test_promoted_count(self, store):
        for eid, promoted in [
            ("fp_20260226_100000", True),
            ("fp_20260226_120000", False),
            ("fp_20260226_140000", True),
        ]:
            entry = ExperimentEntry(
                experiment_id=eid,
                classifier="fp",
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary="Run"),
                reward=ExperimentReward(promoted=promoted),
            )
            store.create_experiment(entry)

        index = store.load_index()
        assert index.summary["fp"].total_promoted == 2

    def test_multiple_classifiers_isolated(self, store):
        for classifier in ["fp", "ep"]:
            entry = ExperimentEntry(
                experiment_id=f"{classifier}_20260226_100000",
                classifier=classifier,
                created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
                action=ExperimentAction(type="test", summary="Run"),
            )
            store.create_experiment(entry)

        index = store.load_index()
        assert len(index.experiments["fp"]) == 1
        assert len(index.experiments["ep"]) == 1
        assert index.summary["fp"].total_experiments == 1
        assert index.summary["ep"].total_experiments == 1


class TestEdgeCases:
    def test_missing_index_returns_empty(self, store):
        index = store.load_index()
        assert index.experiments == {}

    def test_missing_knowledge_returns_empty(self, store):
        kb = store.load_knowledge("nonexistent")
        assert kb.classifier == "nonexistent"
        assert kb.patterns == []

    def test_missing_shared_knowledge_returns_empty(self, store):
        skb = store.load_shared_knowledge()
        assert skb.patterns == []

    def test_atomic_write_creates_parent_dirs(self, store):
        entry = ExperimentEntry(
            experiment_id="ep_20260226_100000",
            classifier="ep",
            created_at=datetime(2026, 2, 26, tzinfo=timezone.utc),
            action=ExperimentAction(type="test", summary="Run"),
        )
        path = store.create_experiment(entry)
        assert path.exists()
        assert path.parent.name == "ep"
