"""MLOps configuration settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# The comparison window a drift check reads: the last N days of predictions.
# One constant, not a literal per caller, because a reference built for that
# check excludes exactly this many recent days (issue #97). If the two numbers
# came apart, the reference would silently start overlapping the window it is
# compared against again.
DEFAULT_DRIFT_WINDOW_DAYS = 7


@dataclass
class MLOpsSettings:
    """Configuration for MLOps features (MLflow, Evidently, alerts)."""

    # MLflow settings
    mlflow_enabled: bool = field(
        default_factory=lambda: os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    )
    mlflow_experiment_prefix: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_PREFIX", "esg-classifier")
    )

    # Evidently settings
    evidently_enabled: bool = field(
        default_factory=lambda: os.getenv("EVIDENTLY_ENABLED", "false").lower() == "true"
    )
    evidently_reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("EVIDENTLY_REPORTS_DIR", "reports/monitoring"))
    )
    drift_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_THRESHOLD", "0.1"))
    )
    # The floor below which a reading is not trusted (issue #94). Enough rows
    # to compute a statistic is not enough rows for it to mean anything: a
    # p-value from three rows reports as confidently as one from three thousand.
    #
    # Two scopes, two outcomes. A whole FRAME below the floor -- the reference
    # and the current window are checked independently -- makes the verdict
    # `indeterminate` rather than healthy. A single COLUMN below it, counted
    # after its NaN are dropped, is recorded in `details["columns_skipped"]` and
    # left out of the score. For a `brand_*` column the check still returns a
    # verdict from whatever else it could measure; for a core column it is
    # `indeterminate`, on both paths (#105, D023).
    drift_min_sample_size: int = field(
        default_factory=lambda: int(os.getenv("DRIFT_MIN_SAMPLE_SIZE", "30"))
    )

    # Reference data settings
    reference_data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REFERENCE_DATA_DIR", "data/reference"))
    )
    reference_window_days: int = field(
        default_factory=lambda: int(os.getenv("REFERENCE_WINDOW_DAYS", "30"))
    )

    # Alert settings
    alert_webhook_url: str | None = field(
        default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL") or None
    )
    alert_on_drift: bool = field(
        default_factory=lambda: os.getenv("ALERT_ON_DRIFT", "true").lower() == "true"
    )
    alert_on_training: bool = field(
        default_factory=lambda: os.getenv("ALERT_ON_TRAINING", "false").lower() == "true"
    )

    # Model artifacts
    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODELS_DIR", "models"))
    )

    def get_experiment_name(self, classifier_type: str) -> str:
        """Get MLflow experiment name for a classifier type."""
        return f"{self.mlflow_experiment_prefix}-{classifier_type}"

    def get_reference_data_path(self, classifier_type: str) -> Path:
        """Get path to reference data for a classifier type."""
        return self.reference_data_dir / f"{classifier_type}_reference.parquet"

    def get_reports_dir(self, classifier_type: str) -> Path:
        """Get path to monitoring reports for a classifier type."""
        return self.evidently_reports_dir / classifier_type


mlops_settings = MLOpsSettings()
