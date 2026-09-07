"""MLOps module for experiment tracking and production monitoring."""

from .config import mlops_settings
from .exit_codes import (
    EXIT_DRIFT_DETECTED,
    EXIT_INDETERMINATE,
    EXIT_NO_DRIFT,
    NON_RETRYABLE_EXIT_CODES,
)
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
from .importance_tracking import (
    ImportanceHistory,
    ImportanceRecord,
    StabilityAnalysis,
)

__all__ = [
    # Config
    "mlops_settings",
    # Exit-code contract (scripts/monitor_drift.py <-> agent runner)
    "EXIT_NO_DRIFT",
    "EXIT_DRIFT_DETECTED",
    "EXIT_INDETERMINATE",
    "NON_RETRYABLE_EXIT_CODES",
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
    # Importance tracking
    "ImportanceHistory",
    "ImportanceRecord",
    "StabilityAnalysis",
]
