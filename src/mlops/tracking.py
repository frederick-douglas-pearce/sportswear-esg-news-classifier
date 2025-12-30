"""MLflow experiment tracking wrapper with graceful degradation."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from .config import mlops_settings

logger = logging.getLogger(__name__)

# MLflow Model Registry stage constants
STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


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

    # =========================================================================
    # Model Registry Methods
    # =========================================================================

    def get_registered_model_name(self) -> str:
        """Get the registered model name for this classifier type.

        Returns:
            Model name for the Model Registry (e.g., 'esg-classifier-fp')
        """
        return f"{mlops_settings.mlflow_experiment_prefix}-{self.classifier_type}"

    def register_model(
        self,
        model_uri: str | None = None,
        model_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> str | None:
        """Register a model in the MLflow Model Registry.

        If model_uri is not provided, uses the model logged in the current run.

        Args:
            model_uri: URI to the model artifact (e.g., 'runs:/<run_id>/model')
            model_name: Name for the registered model (defaults to classifier name)
            tags: Optional tags to add to the model version
            description: Optional description for the model version

        Returns:
            The model version number if successful, None otherwise
        """
        if not self.enabled:
            return None

        if model_name is None:
            model_name = self.get_registered_model_name()

        # If no URI provided, construct from current run
        if model_uri is None:
            run_id = self.get_run_id()
            if run_id is None:
                logger.warning("No active run - cannot register model without URI")
                return None
            model_uri = f"runs:/{run_id}/model"

        try:
            # Register the model
            result = self._mlflow.register_model(model_uri, model_name)
            version = result.version

            logger.info(f"Registered model '{model_name}' version {version}")

            # Add tags if provided
            if tags:
                client = self._mlflow.tracking.MlflowClient()
                for key, value in tags.items():
                    client.set_model_version_tag(model_name, version, key, value)

            # Add description if provided
            if description:
                client = self._mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description,
                )

            return version

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            return None

    def transition_model_stage(
        self,
        version: str,
        stage: str,
        model_name: str | None = None,
        archive_existing: bool = True,
    ) -> bool:
        """Transition a model version to a new stage.

        Args:
            version: Model version to transition
            stage: Target stage (Staging, Production, Archived, None)
            model_name: Name of registered model (defaults to classifier name)
            archive_existing: If True, archive existing models in target stage

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        if model_name is None:
            model_name = self.get_registered_model_name()

        valid_stages = {STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED}
        if stage not in valid_stages:
            logger.warning(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")
            return False

        try:
            client = self._mlflow.tracking.MlflowClient()

            # Archive existing models in target stage if requested
            if archive_existing and stage in (STAGE_STAGING, STAGE_PRODUCTION):
                existing = self.get_model_versions_by_stage(stage, model_name)
                for existing_version in existing:
                    if existing_version != version:
                        client.transition_model_version_stage(
                            name=model_name,
                            version=existing_version,
                            stage=STAGE_ARCHIVED,
                        )
                        logger.info(
                            f"Archived '{model_name}' version {existing_version}"
                        )

            # Transition the model
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )

            logger.info(f"Transitioned '{model_name}' version {version} to {stage}")
            return True

        except Exception as e:
            logger.warning(f"Failed to transition model stage: {e}")
            return False

    def get_latest_model_version(
        self,
        model_name: str | None = None,
        stages: list[str] | None = None,
    ) -> str | None:
        """Get the latest version of a registered model.

        Args:
            model_name: Name of registered model (defaults to classifier name)
            stages: List of stages to filter by (default: all stages)

        Returns:
            Latest version number if found, None otherwise
        """
        if not self.enabled:
            return None

        if model_name is None:
            model_name = self.get_registered_model_name()

        try:
            client = self._mlflow.tracking.MlflowClient()

            # Get all versions
            versions = client.search_model_versions(f"name='{model_name}'")

            if not versions:
                return None

            # Filter by stages if specified
            if stages:
                versions = [v for v in versions if v.current_stage in stages]

            if not versions:
                return None

            # Return the highest version number
            return str(max(int(v.version) for v in versions))

        except Exception as e:
            logger.warning(f"Failed to get latest model version: {e}")
            return None

    def get_model_versions_by_stage(
        self,
        stage: str,
        model_name: str | None = None,
    ) -> list[str]:
        """Get all model versions in a specific stage.

        Args:
            stage: Stage to filter by (Staging, Production, Archived, None)
            model_name: Name of registered model (defaults to classifier name)

        Returns:
            List of version numbers in the specified stage
        """
        if not self.enabled:
            return []

        if model_name is None:
            model_name = self.get_registered_model_name()

        try:
            client = self._mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")

            return [v.version for v in versions if v.current_stage == stage]

        except Exception as e:
            logger.warning(f"Failed to get model versions by stage: {e}")
            return []

    def get_production_model_version(
        self,
        model_name: str | None = None,
    ) -> str | None:
        """Get the current production model version.

        Args:
            model_name: Name of registered model (defaults to classifier name)

        Returns:
            Production version number if found, None otherwise
        """
        versions = self.get_model_versions_by_stage(STAGE_PRODUCTION, model_name)
        return versions[0] if versions else None

    def promote_to_production(
        self,
        version: str,
        model_name: str | None = None,
    ) -> bool:
        """Promote a model version to production.

        This is a convenience method that:
        1. Archives any existing production models
        2. Transitions the specified version to Production

        Args:
            version: Model version to promote
            model_name: Name of registered model (defaults to classifier name)

        Returns:
            True if successful, False otherwise
        """
        return self.transition_model_stage(
            version=version,
            stage=STAGE_PRODUCTION,
            model_name=model_name,
            archive_existing=True,
        )

    def get_model_info(
        self,
        version: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Get information about a registered model version.

        Args:
            version: Model version (default: latest production or latest overall)
            model_name: Name of registered model (defaults to classifier name)

        Returns:
            Dictionary with model information or None if not found
        """
        if not self.enabled:
            return None

        if model_name is None:
            model_name = self.get_registered_model_name()

        if version is None:
            # Try production first, then latest
            version = self.get_production_model_version(model_name)
            if version is None:
                version = self.get_latest_model_version(model_name)
            if version is None:
                return None

        try:
            client = self._mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(model_name, version)

            return {
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "status": model_version.status,
                "run_id": model_version.run_id,
                "source": model_version.source,
                "description": model_version.description,
                "tags": dict(model_version.tags) if model_version.tags else {},
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
            }

        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None
