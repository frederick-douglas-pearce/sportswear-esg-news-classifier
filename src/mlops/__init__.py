"""MLOps module for experiment tracking and production monitoring."""

from .config import mlops_settings
from .tracking import (
    ExperimentTracker,
    STAGE_NONE,
    STAGE_STAGING,
    STAGE_PRODUCTION,
    STAGE_ARCHIVED,
)
from .monitoring import DriftMonitor, DriftReport, run_drift_analysis
from .alerts import AlertSender, AlertType, send_drift_alert, send_training_alert
from .reference_data import (
    create_reference_dataset,
    load_prediction_logs,
    load_reference_dataset,
    get_reference_stats,
)

__all__ = [
    # Config
    "mlops_settings",
    # Tracking
    "ExperimentTracker",
    # Model Registry Stages
    "STAGE_NONE",
    "STAGE_STAGING",
    "STAGE_PRODUCTION",
    "STAGE_ARCHIVED",
    # Monitoring
    "DriftMonitor",
    "DriftReport",
    "run_drift_analysis",
    # Alerts
    "AlertSender",
    "AlertType",
    "send_drift_alert",
    "send_training_alert",
    # Reference data
    "create_reference_dataset",
    "load_prediction_logs",
    "load_reference_dataset",
    "get_reference_stats",
]
