"""Experiment lifecycle tracker for model training workflows.

Reads existing data sources (registry.json, config files, workflow context)
and creates/updates experiment entries via ExperimentStore. One instance
per classifier.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    ExperimentAction,
    ExperimentEntry,
    ExperimentObservation,
    ExperimentReflection,
    ExperimentReward,
    ExperimentState,
)
from .store import ExperimentStore

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models")


class ExperimentTracker:
    """Tracks experiment lifecycle from creation through completion.

    All public methods are no-ops when self._experiment is None,
    making it safe to call even if creation failed.
    """

    def __init__(
        self,
        store: ExperimentStore | None = None,
        classifier: str = "fp",
        models_dir: Path | None = None,
    ):
        self.store = store or ExperimentStore()
        self.classifier = classifier
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self._experiment: ExperimentEntry | None = None

    # --- Lifecycle ---

    def create_experiment(self, context: dict) -> str:
        """Create a new experiment entry, returning the experiment_id.

        Reads registry.json for production version/metrics, feature_config,
        and classifier_config to snapshot the current state.
        """
        now = datetime.now(timezone.utc)
        experiment_id = f"{self.classifier}_{now.strftime('%Y%m%d_%H%M%S')}"

        state = self._snapshot_state()
        self._enrich_training_data(state, context)
        action = self._build_action(context)

        # Link to most recent prior experiment as parent
        parent_id = None
        recent = self.store.list_experiments(self.classifier, limit=1)
        if recent:
            parent_id = recent[0].id

        self._experiment = ExperimentEntry(
            experiment_id=experiment_id,
            classifier=self.classifier,
            created_at=now,
            status="running",
            parent_experiment=parent_id,
            tags=["automated", "workflow"],
            state=state,
            action=action,
        )

        self.store.create_experiment(self._experiment)
        logger.info(f"Created experiment {experiment_id}")
        return experiment_id

    def resume(self, experiment_id: str) -> None:
        """Load an experiment from store (for use after workflow pause/resume)."""
        self._experiment = self.store.load_experiment(experiment_id)
        logger.info(f"Resumed experiment {experiment_id}")

    def record_observation(self, comparison: dict) -> None:
        """Record metrics from compare_models output.

        Expected format: {"classifiers": {"fp": {"new_model": {...}, "improvement": {...}}}}
        """
        if self._experiment is None:
            return

        classifier_data = comparison.get("classifiers", {}).get(self.classifier, {})
        new_model = classifier_data.get("new_model", {})
        improvement = classifier_data.get("improvement", {})

        metrics = {}
        for key in ("test_f2", "test_recall", "test_precision", "cv_f2", "threshold"):
            if key in new_model:
                metrics[key] = new_model[key]

        delta = {}
        if improvement:
            for key, value in improvement.items():
                delta[key] = value

        self._experiment.observation = ExperimentObservation(
            metrics=metrics,
            delta=delta,
        )

        self.store.update_experiment(self._experiment)
        logger.info(f"Recorded observation for {self._experiment.experiment_id}")

    def record_reward(
        self,
        promoted: bool,
        promoted_as: str | None = None,
        reason: str = "",
    ) -> None:
        """Record reward (promotion outcome)."""
        if self._experiment is None:
            return

        primary_value = self._experiment.observation.metrics.get("test_f2")
        prod_f2 = self._experiment.state.production_metrics.get("test_f2")

        target_met = False
        if primary_value is not None and prod_f2 is not None:
            target_met = primary_value > prod_f2

        self._experiment.reward = ExperimentReward(
            primary_metric="test_f2",
            primary_value=primary_value,
            primary_target=prod_f2,
            target_met=target_met,
            promoted=promoted,
            promoted_as=promoted_as,
            promotion_reason=reason,
        )

        self.store.update_experiment(self._experiment)
        logger.info(
            f"Recorded reward for {self._experiment.experiment_id}: "
            f"promoted={promoted}"
        )

    def record_reflection(self, reflection: ExperimentReflection) -> None:
        """Attach LLM-generated reflection to the experiment."""
        if self._experiment is None:
            return

        self._experiment.reflection = reflection
        self.store.update_experiment(self._experiment)
        logger.info(f"Recorded reflection for {self._experiment.experiment_id}")

    def complete(self, status: str = "completed") -> None:
        """Mark experiment as completed."""
        if self._experiment is None:
            return

        self._experiment.status = status
        self._experiment.completed_at = datetime.now(timezone.utc)
        self.store.update_experiment(self._experiment)
        logger.info(f"Completed experiment {self._experiment.experiment_id}")

    def fail(self, error: str = "") -> None:
        """Mark experiment as failed."""
        if self._experiment is None:
            return

        self._experiment.status = "failed"
        self._experiment.completed_at = datetime.now(timezone.utc)
        if error:
            self._experiment.extensions["failure_reason"] = error
        self.store.update_experiment(self._experiment)
        logger.info(f"Failed experiment {self._experiment.experiment_id}: {error}")

    @property
    def experiment(self) -> ExperimentEntry | None:
        """Access the current experiment entry."""
        return self._experiment

    # --- Private helpers ---

    def _snapshot_state(self) -> ExperimentState:
        """Read registry + config files to build current state snapshot."""
        registry = self._load_json(self.models_dir / "registry.json") or {}
        feature_config = (
            self._load_json(
                self.models_dir / f"{self.classifier}_feature_config.json"
            )
            or {}
        )
        classifier_config = (
            self._load_json(
                self.models_dir / f"{self.classifier}_classifier_config.json"
            )
            or {}
        )

        # Extract production version and metrics from registry
        classifier_registry = registry.get(self.classifier, {})
        prod_version = classifier_registry.get("production")
        prod_metrics: dict = {}
        if prod_version:
            version_info = classifier_registry.get("versions", {}).get(
                prod_version, {}
            )
            prod_metrics = version_info.get("metrics", {})

        return ExperimentState(
            production_version=prod_version,
            production_metrics=prod_metrics,
            feature_config=feature_config,
            model_config_snapshot=classifier_config,
        )

    def _build_action(self, context: dict) -> ExperimentAction:
        """Build action from workflow context."""
        datasets = context.get("datasets", {})
        classifier_data = datasets.get(self.classifier, {})
        record_count = classifier_data.get("total_records") or classifier_data.get(
            "record_count", "unknown"
        )

        return ExperimentAction(
            type="automated_retraining",
            summary=(
                f"Automated retraining of {self.classifier} classifier "
                f"with {record_count} records"
            ),
        )

    def _enrich_training_data(
        self, state: ExperimentState, context: dict
    ) -> None:
        """Add training data stats from context to state."""
        datasets = context.get("datasets", {})
        classifier_data = datasets.get(self.classifier, {})
        if classifier_data:
            state.training_data = {
                k: v
                for k, v in classifier_data.items()
                if k
                in (
                    "total_records",
                    "record_count",
                    "positive_count",
                    "negative_count",
                    "imbalance_ratio",
                )
            }

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        """Load a JSON file, returning None on any error."""
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
