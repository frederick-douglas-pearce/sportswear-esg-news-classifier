"""MLflow experiment tracking wrapper with graceful degradation."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from .config import mlops_settings

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wrapper for MLflow experiment tracking with graceful degradation.

    When MLflow is disabled (MLFLOW_ENABLED=false), all methods become no-ops.
    This allows the training code to use tracking unconditionally.

    Usage:
        tracker = ExperimentTracker("fp")
        with tracker.start_run(run_name="fp-v1.1.0"):
            tracker.log_params({"n_estimators": 100})
            tracker.log_metrics({"f2_score": 0.95})
            tracker.log_artifact("models/fp_classifier_pipeline.joblib")
    """

    def __init__(self, classifier_type: str):
        """Initialize tracker for a classifier type.

        Args:
            classifier_type: Type of classifier (fp, ep, esg)
        """
        self.classifier_type = classifier_type
        self.experiment_name = mlops_settings.get_experiment_name(classifier_type)
        self.enabled = mlops_settings.mlflow_enabled
        self._run = None
        self._mlflow = None

        if self.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow connection."""
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(mlops_settings.mlflow_tracking_uri)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)

            logger.info(
                f"MLflow tracking enabled: {mlops_settings.mlflow_tracking_uri} "
                f"(experiment: {self.experiment_name})"
            )
        except ImportError:
            logger.warning("MLflow not installed, tracking disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"MLflow setup failed, tracking disabled: {e}")
            self.enabled = False

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator["ExperimentTracker", None, None]:
        """Start an MLflow run as a context manager.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run

        Yields:
            Self for method chaining
        """
        if not self.enabled:
            yield self
            return

        # Default run name with timestamp
        if run_name is None:
            run_name = f"{self.classifier_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Default tags
        run_tags = {
            "classifier_type": self.classifier_type,
            "timestamp": datetime.now().isoformat(),
        }
        if tags:
            run_tags.update(tags)

        try:
            self._run = self._mlflow.start_run(run_name=run_name, tags=run_tags)
            logger.info(f"Started MLflow run: {run_name}")
            yield self
        finally:
            if self._run:
                self._mlflow.end_run()
                self._run = None
                logger.info(f"Ended MLflow run: {run_name}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled or not self._run:
            return

        # Flatten nested dicts and convert values to strings
        flat_params = self._flatten_dict(params)
        self._mlflow.log_params(flat_params)
        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for iterative metrics
        """
        if not self.enabled or not self._run:
            return

        self._mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            local_path: Path to the file or directory to log
            artifact_path: Optional subdirectory in the artifact store
        """
        if not self.enabled or not self._run:
            return

        self._mlflow.log_artifact(str(local_path), artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_model_config(self, config: dict[str, Any]) -> None:
        """Log model configuration as a JSON artifact.

        Args:
            config: Model configuration dictionary
        """
        if not self.enabled or not self._run:
            return

        # Save config to temp file and log
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f, indent=2, default=str)
            temp_path = f.name

        self._mlflow.log_artifact(temp_path, "config")
        Path(temp_path).unlink()
        logger.debug("Logged model config")

    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> None:
        """Log a scikit-learn model.

        Args:
            model: Trained sklearn model or pipeline
            artifact_path: Path within the artifact store
            registered_name: Optional name for model registry
        """
        if not self.enabled or not self._run:
            return

        try:
            import mlflow.sklearn

            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_name,
            )
            logger.info(f"Logged sklearn model to {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log sklearn model: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        if not self.enabled or not self._run:
            return

        self._mlflow.set_tag(key, value)

    def get_run_id(self) -> str | None:
        """Get the current run ID.

        Returns:
            Run ID if a run is active, None otherwise
        """
        if not self.enabled or not self._run:
            return None
        return self._run.info.run_id

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for keys
            sep: Separator between nested keys

        Returns:
            Flattened dictionary with string keys and values
        """
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep).items())
            else:
                # Convert value to string for MLflow
                items.append((new_key, str(v)))
        return dict(items)
