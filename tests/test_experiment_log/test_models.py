"""Tests for experiment log models."""

from datetime import datetime, timezone

from src.experiment_log.models import (
    ClassifierSummary,
    Decision,
    DecisionOption,
    DiagnosticStep,
    DiagnosticTree,
    ExperimentAction,
    ExperimentChange,
    ExperimentEntry,
    ExperimentIndex,
    ExperimentObservation,
    ExperimentReflection,
    ExperimentReward,
    ExperimentState,
    Heuristic,
    IndexEntry,
    KnowledgeBase,
    Pattern,
    SharedKnowledgeBase,
    SharedPattern,
)


class TestExperimentState:
    def test_defaults(self):
        state = ExperimentState()
        assert state.production_version is None
        assert state.production_metrics == {}
        assert state.training_data == {}
        assert state.feature_config == {}
        assert state.model_config_snapshot == {}
        assert state.known_issues == []

    def test_flexible_dicts(self):
        state = ExperimentState(
            production_metrics={"test_f2": 0.974, "test_recall": 0.988},
            training_data={"record_count": 2318, "positive_pct": 0.42},
            feature_config={"tfidf": {"max_features": 5000}},
            model_config_snapshot={"model_name": "RandomForest", "params": {"n_estimators": 200}},
        )
        assert state.production_metrics["test_f2"] == 0.974
        assert state.training_data["record_count"] == 2318
        assert state.feature_config["tfidf"]["max_features"] == 5000


class TestExperimentChange:
    def test_captures_fields(self):
        change = ExperimentChange(
            component="feature_config",
            field="tfidf.max_features",
            old_value=3000,
            new_value=5000,
            rationale="Increase vocabulary for better recall",
        )
        assert change.component == "feature_config"
        assert change.old_value == 3000
        assert change.new_value == 5000

    def test_none_values(self):
        change = ExperimentChange(component="labels", field="corrections")
        assert change.old_value is None
        assert change.new_value is None
        assert change.rationale == ""


class TestExperimentAction:
    def test_with_changes(self):
        action = ExperimentAction(
            type="feature_engineering",
            summary="Add NER features",
            changes=[
                ExperimentChange(component="feature_config", field="ner_enabled", new_value=True),
            ],
            hypothesis="NER features improve edge case detection",
        )
        assert action.type == "feature_engineering"
        assert len(action.changes) == 1
        assert action.hypothesis != ""


class TestExperimentEntry:
    def _make_entry(self, **kwargs):
        defaults = dict(
            experiment_id="fp_20260226_143000",
            classifier="fp",
            created_at=datetime(2026, 2, 26, 14, 30, tzinfo=timezone.utc),
            action=ExperimentAction(type="test", summary="Test run"),
        )
        defaults.update(kwargs)
        return ExperimentEntry(**defaults)

    def test_full_roundtrip(self):
        now = datetime(2026, 2, 26, 14, 30, tzinfo=timezone.utc)
        entry = ExperimentEntry(
            experiment_id="fp_20260226_143000",
            classifier="fp",
            created_at=now,
            completed_at=now,
            status="completed",
            parent_experiment="fp_20260225_100000",
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
                delta={"test_f2": -0.014, "is_improvement": False},
                error_analysis={"total_errors": 5, "false_positives": 3},
                artifacts={"pipeline": "models/fp_v2.2.0.pkl"},
            ),
            reward=ExperimentReward(
                primary_metric="test_f2",
                primary_value=0.960,
                primary_target=0.974,
                target_met=False,
                constraints_met={"recall_above_99pct": False},
                promoted=False,
            ),
            reflection=ExperimentReflection(
                summary="NER features are critical for edge cases",
                hypothesis_result="refuted",
                hypothesis_evidence="F2 dropped 0.014 without NER",
                surprises=["Recall dropped more than F2"],
                next_steps=["Keep NER, try adding new NER types"],
                confidence="high",
            ),
            extensions={"notebook": "fp2_model_selection_tuning.ipynb"},
        )

        # Serialize and deserialize
        data = entry.model_dump(mode="json")
        restored = ExperimentEntry.model_validate(data)

        assert restored.experiment_id == entry.experiment_id
        assert restored.classifier == "fp"
        assert restored.status == "completed"
        assert restored.parent_experiment == "fp_20260225_100000"
        assert restored.tags == ["ablation", "ner"]
        assert restored.state.production_metrics["test_f2"] == 0.974
        assert restored.action.changes[0].old_value is True
        assert restored.observation.metrics["test_f2"] == 0.960
        assert restored.reward.target_met is False
        assert restored.reflection.hypothesis_result == "refuted"
        assert restored.extensions["notebook"] == "fp2_model_selection_tuning.ipynb"

    def test_minimal_defaults(self):
        entry = self._make_entry()
        assert entry.status == "running"
        assert entry.parent_experiment is None
        assert entry.tags == []
        assert entry.state.production_version is None
        assert entry.observation.metrics == {}
        assert entry.reward.primary_metric == "test_f2"
        assert entry.reflection.summary == ""
        assert entry.extensions == {}

    def test_extensions_accept_arbitrary_keys(self):
        entry = self._make_entry(
            extensions={
                "custom_plot": "/path/to/plot.png",
                "notes": ["first", "second"],
                "nested": {"key": "value"},
            }
        )
        assert entry.extensions["custom_plot"] == "/path/to/plot.png"
        assert len(entry.extensions["notes"]) == 2
        assert entry.extensions["nested"]["key"] == "value"


class TestDecision:
    def test_full_roundtrip(self):
        now = datetime(2026, 2, 26, 15, 0, tzinfo=timezone.utc)
        decision = Decision(
            decision_id="dec_fp_20260226_150000",
            experiment_id="fp_20260226_143000",
            classifier="fp",
            timestamp=now,
            phase="feature_engineering",
            trigger_type="metric_observation",
            trigger_description="SHAP shows NER features have low individual importance",
            trigger_evidence=["SHAP top 10 excludes all NER features"],
            options=[
                DecisionOption(
                    id="keep_ner",
                    description="Keep NER features",
                    expected_outcome="Maintain current performance",
                    risk="low",
                ),
                DecisionOption(
                    id="remove_ner",
                    description="Remove NER features",
                    expected_outcome="Simplify model, may lose edge cases",
                    risk="medium",
                ),
            ],
            chosen="keep_ner",
            reasoning="NER features are critical for brand disambiguation edge cases",
            precedents=[
                {
                    "decision_id": "dec_fp_20260110_120000",
                    "summary": "Previously confirmed NER helps with ambiguous brand names",
                    "relevance": "Same feature group, same classifier",
                }
            ],
            outcome_success=True,
            outcome_result="Correct — ablation confirmed NER loss drops F2 by 0.014",
            lesson_learned="Low SHAP importance != dispensable, check interaction effects",
        )

        data = decision.model_dump(mode="json")
        restored = Decision.model_validate(data)

        assert restored.decision_id == decision.decision_id
        assert len(restored.options) == 2
        assert restored.chosen == "keep_ner"
        assert len(restored.precedents) == 1
        assert restored.outcome_success is True
        assert restored.lesson_learned != ""


class TestKnowledgeBase:
    def test_patterns_accumulate(self):
        now = datetime(2026, 2, 26, tzinfo=timezone.utc)
        kb = KnowledgeBase(classifier="fp", last_updated=now)
        assert len(kb.patterns) == 0

        kb.patterns.append(
            Pattern(
                id="pat_001",
                category="feature_engineering",
                pattern="NER features critical for brand disambiguation",
                confidence="high",
                evidence=[{"experiment": "fp_20260226_143000", "detail": "Ablation -0.014 F2"}],
                first_observed="2026-02-26",
                last_confirmed="2026-02-26",
            )
        )
        assert len(kb.patterns) == 1

    def test_heuristics_track_counts(self):
        h = Heuristic(
            id="heur_001",
            trigger="SHAP shows low NER importance",
            action="Run ablation before removing",
            confidence="high",
            derived_from=["pat_001"],
            times_applied=3,
            times_successful=3,
        )
        assert h.times_applied == 3
        assert h.times_successful == 3

    def test_diagnostic_tree(self):
        tree = DiagnosticTree(
            id="diag_001",
            name="F2 regression",
            trigger="test_f2 dropped > 0.01",
            steps=[
                DiagnosticStep(
                    check="Did training data change?",
                    if_yes="Check label quality",
                    if_no="Check feature config changes",
                ),
            ],
            derived_from=["pat_001"],
        )
        assert len(tree.steps) == 1
        assert tree.steps[0].if_yes == "Check label quality"

    def test_roundtrip(self):
        now = datetime(2026, 2, 26, tzinfo=timezone.utc)
        kb = KnowledgeBase(
            classifier="fp",
            last_updated=now,
            version=3,
            patterns=[
                Pattern(id="p1", category="data_quality", pattern="Test pattern")
            ],
            baselines={"expected_ranges": {"test_f2": {"min": 0.95, "max": 0.99}}},
        )
        data = kb.model_dump(mode="json")
        restored = KnowledgeBase.model_validate(data)
        assert restored.version == 3
        assert restored.baselines["expected_ranges"]["test_f2"]["min"] == 0.95


class TestSharedKnowledgeBase:
    def test_cross_classifier_patterns(self):
        now = datetime(2026, 2, 26, tzinfo=timezone.utc)
        skb = SharedKnowledgeBase(last_updated=now)
        skb.patterns.append(
            SharedPattern(
                id="shared_001",
                pattern="TF-IDF + LSA outperforms sentence transformers for < 3000 samples",
                applies_to=["fp", "ep"],
                confidence="high",
                source_classifier="fp",
            )
        )
        assert len(skb.patterns) == 1
        assert "ep" in skb.patterns[0].applies_to


class TestExperimentIndex:
    def test_entries_per_classifier(self):
        now = datetime(2026, 2, 26, tzinfo=timezone.utc)
        index = ExperimentIndex(
            last_updated=now,
            experiments={
                "fp": [
                    IndexEntry(
                        id="fp_20260226_143000",
                        status="completed",
                        action_type="ablation",
                        action_summary="Remove NER",
                        test_f2=0.960,
                        delta_f2=-0.014,
                        tags=["ablation"],
                    ),
                    IndexEntry(
                        id="fp_20260225_100000",
                        status="completed",
                        action_type="retraining",
                        action_summary="Retrain after label fixes",
                        test_f2=0.974,
                        promoted=True,
                    ),
                ],
            },
            summary={
                "fp": ClassifierSummary(
                    total_experiments=2,
                    total_promoted=1,
                    current_production="v2.1.0",
                    best_test_f2_ever=0.974,
                ),
            },
        )
        assert len(index.experiments["fp"]) == 2
        assert index.experiments["fp"][0].id > index.experiments["fp"][1].id  # newest first
        assert index.summary["fp"].best_test_f2_ever == 0.974

    def test_summary_aggregation(self):
        summary = ClassifierSummary(
            total_experiments=12,
            total_promoted=3,
            current_production="v2.2.0",
            best_test_f2_ever=0.987,
        )
        assert summary.total_experiments == 12
        assert summary.total_promoted == 3
