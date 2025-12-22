"""Deployment module for FP Brand Classifier."""

from .config import (
    DEFAULT_THRESHOLD,
    MODEL_NAME,
    PIPELINE_PATH,
    CONFIG_PATH,
    RISK_LEVELS,
    TARGET_RECALL,
    get_risk_level,
    load_config,
)
from .data import (
    create_text_features,
    load_training_data,
    split_data,
)
from .prediction import FPClassifier
from .preprocessing import clean_text, prepare_input

__all__ = [
    # Config
    "DEFAULT_THRESHOLD",
    "MODEL_NAME",
    "PIPELINE_PATH",
    "CONFIG_PATH",
    "RISK_LEVELS",
    "TARGET_RECALL",
    "get_risk_level",
    "load_config",
    # Data
    "create_text_features",
    "load_training_data",
    "split_data",
    # Prediction
    "FPClassifier",
    # Preprocessing
    "clean_text",
    "prepare_input",
]
