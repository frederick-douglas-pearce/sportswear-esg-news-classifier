"""Tests for ExperimentTracker."""

import json

import pytest

from src.experiment_log.models import ExperimentReflection
from src.experiment_log.store import ExperimentStore
from src.experiment_log.tracker import ExperimentTracker


@pytest.fixture
def store(tmp_path):
    """Create an ExperimentStore with a temp directory."""
    return ExperimentStore(base_dir=tmp_path)


@pytest.fixture
def models_dir(tmp_path):
    """Create a models directory with test config files."""
    models = tmp_path / "models"
    models.mkdir()

    # Registry with production version
    registry = {
        "fp": {
            "production": "v2.1.0",
            "versions": {
                "v2.1.0": {
                    "metrics": {
                        "test_f2": 0.974,
                        "test_recall": 0.988,
                        "test_precision": 0.92,
                        "cv_f2": 0.965,
                    }
                }
            },
        }
    }
    with open(models / "registry.json", "w") as f:
        json.dump(registry, f)

    # Feature config
    feature_config = {"features": ["tfidf", "ner"], "max_features": 5000}
    with open(models / "fp_feature_config.json", "w") as f:
        json.dump(feature_config, f)

    # Classifier config
    classifier_config = {
        "model_name": "RandomForest",
        "n_estimators": 200,
        "test_f2": 0.974,
    }
    with open(models / "fp_classifier_config.json", "w") as f:
        json.dump(classifier_config, f)

    return models


@pytest.fixture
def context():
    """Standard workflow context with dataset info."""
    return {
        "datasets": {
            "fp": {
                "success": True,
                "total_records": 2318,
                "positive_count": 1800,
                "negative_count": 518,
                "imbalance_ratio": 3.47,
            }
        }
    }


@pytest.fixture
def comparison_results():
    """Standard compare_models output."""
    return {
        "comparison_complete": True,
        "classifiers": {
            "fp": {
                "production": {"version": "v2.1.0", "test_f2": 0.974},
                "new_model": {
                    "test_f2": 0.976,
                    "test_recall": 0.990,
                    "test_precision": 0.925,
                    "cv_f2": 0.968,
                    "threshold": 0.45,
                },
                "improvement": {
                    "test_f2_delta": 0.002,
                    "test_f2_pct": 0.21,
                    "is_better": True,
                },
            }
        },
    }


class TestCreateExperiment:
    def test_snapshots_state(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        exp_id = tracker.create_experiment(context)

        loaded = store.load_experiment(exp_id)
        assert loaded.state.production_version == "v2.1.0"
        assert loaded.state.production_metrics["test_f2"] == 0.974
        assert loaded.state.feature_config["max_features"] == 5000
        assert loaded.state.model_config_snapshot["model_name"] == "RandomForest"

    def test_with_training_data_context(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        exp_id = tracker.create_experiment(context)

        loaded = store.load_experiment(exp_id)
        assert loaded.state.training_data["total_records"] == 2318
        assert loaded.state.training_data["positive_count"] == 1800
        assert loaded.action.type == "automated_retraining"
        assert "2318" in loaded.action.summary

    def test_links_parent(self, store, models_dir, context):
        from datetime import datetime, timezone
        from unittest.mock import patch

        # Use fixed timestamps to avoid same-second collision
        t1 = datetime(2026, 2, 26, 14, 30, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 26, 14, 30, 1, tzinfo=timezone.utc)

        tracker1 = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        with patch("src.experiment_log.tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            first_id = tracker1.create_experiment(context)

        tracker2 = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        with patch("src.experiment_log.tracker.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            second_id = tracker2.create_experiment(context)

        loaded = store.load_experiment(second_id)
        assert loaded.parent_experiment == first_id

    def test_missing_configs(self, store, tmp_path, context):
        """Works when config files don't exist."""
        empty_models = tmp_path / "empty_models"
        empty_models.mkdir()
        tracker = ExperimentTracker(
            store=store, classifier="fp", models_dir=empty_models
        )
        exp_id = tracker.create_experiment(context)

        loaded = store.load_experiment(exp_id)
        assert loaded.state.production_version is None
        assert loaded.state.production_metrics == {}
        assert loaded.state.feature_config == {}

    def test_tags_and_status(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        exp_id = tracker.create_experiment(context)

        loaded = store.load_experiment(exp_id)
        assert loaded.status == "running"
        assert "automated" in loaded.tags
        assert "workflow" in loaded.tags


class TestResume:
    def test_loads_experiment(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        exp_id = tracker.create_experiment(context)

        # Create new tracker and resume
        tracker2 = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker2.resume(exp_id)

        assert tracker2.experiment is not None
        assert tracker2.experiment.experiment_id == exp_id


class TestRecordObservation:
    def test_from_comparison(self, store, models_dir, context, comparison_results):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker.create_experiment(context)
        tracker.record_observation(comparison_results)

        loaded = store.load_experiment(tracker.experiment.experiment_id)
        assert loaded.observation.metrics["test_f2"] == 0.976
        assert loaded.observation.metrics["test_recall"] == 0.990
        assert loaded.observation.delta["test_f2_delta"] == 0.002
        assert loaded.observation.delta["is_better"] is True


class TestRecordReward:
    def test_promoted(self, store, models_dir, context, comparison_results):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker.create_experiment(context)
        tracker.record_observation(comparison_results)
        tracker.record_reward(promoted=True, promoted_as="v2.2.0", reason="improvement")

        loaded = store.load_experiment(tracker.experiment.experiment_id)
        assert loaded.reward.promoted is True
        assert loaded.reward.promoted_as == "v2.2.0"
        assert loaded.reward.target_met is True
        assert loaded.reward.primary_value == 0.976

    def test_not_promoted(self, store, models_dir, context, comparison_results):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker.create_experiment(context)
        tracker.record_observation(comparison_results)
        tracker.record_reward(promoted=False, reason="no_improvement")

        loaded = store.load_experiment(tracker.experiment.experiment_id)
        assert loaded.reward.promoted is False
        assert loaded.reward.promotion_reason == "no_improvement"


class TestCompleteAndFail:
    def test_complete_sets_status_and_timestamp(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker.create_experiment(context)
        tracker.complete()

        loaded = store.load_experiment(tracker.experiment.experiment_id)
        assert loaded.status == "completed"
        assert loaded.completed_at is not None

    def test_fail_sets_status_and_reason(self, store, models_dir, context):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)
        tracker.create_experiment(context)
        tracker.fail(error="notebook crashed")

        loaded = store.load_experiment(tracker.experiment.experiment_id)
        assert loaded.status == "failed"
        assert loaded.extensions["failure_reason"] == "notebook crashed"
        assert loaded.completed_at is not None


class TestGracefulDegradation:
    def test_no_experiment_no_ops(self):
        """All methods are no-ops when experiment is None."""
        tracker = ExperimentTracker(classifier="fp")
        # None of these should raise
        tracker.record_observation({"classifiers": {"fp": {}}})
        tracker.record_reward(promoted=False)
        tracker.record_reflection(ExperimentReflection(summary="test"))
        tracker.complete()
        tracker.fail(error="test")


class TestFullLifecycle:
    def test_create_observe_reward_reflect_complete(
        self, store, models_dir, context, comparison_results
    ):
        tracker = ExperimentTracker(store=store, classifier="fp", models_dir=models_dir)

        # Create
        exp_id = tracker.create_experiment(context)
        assert exp_id.startswith("fp_")

        # Observe
        tracker.record_observation(comparison_results)

        # Reward
        tracker.record_reward(promoted=True, promoted_as="v2.2.0", reason="improvement")

        # Reflect
        reflection = ExperimentReflection(
            summary="Model improved slightly",
            hypothesis_result="confirmed",
            next_steps=["Monitor drift"],
            confidence="high",
        )
        tracker.record_reflection(reflection)

        # Complete
        tracker.complete()

        # Verify full entry
        loaded = store.load_experiment(exp_id)
        assert loaded.status == "completed"
        assert loaded.observation.metrics["test_f2"] == 0.976
        assert loaded.reward.promoted is True
        assert loaded.reflection.summary == "Model improved slightly"
        assert loaded.completed_at is not None
