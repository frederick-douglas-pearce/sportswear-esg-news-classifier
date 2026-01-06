"""Configuration management for multi-classifier deployment."""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple


class ClassifierType(Enum):
    """Enumeration of available classifier types."""

    FP = "fp"  # False Positive Brand Classifier
    EP = "ep"  # ESG Pre-filter Classifier
    ESG = "esg"  # ESG Multi-label Classifier (future)


# Multi-classifier configuration
CLASSIFIER_CONFIG: Dict[ClassifierType, Dict[str, Any]] = {
    ClassifierType.FP: {
        "pipeline_path": "models/fp_classifier_pipeline.joblib",
        "config_path": "models/fp_classifier_config.json",
        "default_threshold": 0.605,
        "target_recall": 0.98,
        "model_name": "FP_Classifier",
        "description": "False Positive Brand Classifier - filters non-sportswear articles",
    },
    ClassifierType.EP: {
        "pipeline_path": "models/ep_classifier_pipeline.joblib",
        "config_path": "models/ep_classifier_config.json",
        "default_threshold": 0.156,
        "target_recall": 0.99,
        "model_name": "EP_Classifier",
        "description": "ESG Pre-filter Classifier - identifies ESG-related content",
    },
    ClassifierType.ESG: {
        "pipeline_path": "models/esg_classifier_pipeline.joblib",
        "config_path": "models/esg_classifier_config.json",
        "default_threshold": 0.5,
        "target_recall": 0.95,
        "model_name": "ESG_Classifier",
        "description": "ESG Multi-label Classifier - detailed ESG categorization",
    },
}


def get_classifier_config(classifier_type: ClassifierType) -> Dict[str, Any]:
    """Get configuration for a specific classifier type.

    Args:
        classifier_type: The classifier type to get config for

    Returns:
        Dictionary containing classifier configuration

    Raises:
        ValueError: If classifier type is not recognized
    """
    if classifier_type not in CLASSIFIER_CONFIG:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    return CLASSIFIER_CONFIG[classifier_type]


def get_classifier_paths(classifier_type: ClassifierType) -> Tuple[str, str]:
    """Get pipeline and config paths for a classifier type.

    Args:
        classifier_type: The classifier type

    Returns:
        Tuple of (pipeline_path, config_path)
    """
    config = get_classifier_config(classifier_type)
    return config["pipeline_path"], config["config_path"]


# Legacy FP-specific exports (for backwards compatibility)
DEFAULT_THRESHOLD = CLASSIFIER_CONFIG[ClassifierType.FP]["default_threshold"]
TARGET_RECALL = CLASSIFIER_CONFIG[ClassifierType.FP]["target_recall"]
MODEL_NAME = CLASSIFIER_CONFIG[ClassifierType.FP]["model_name"]
PIPELINE_PATH = CLASSIFIER_CONFIG[ClassifierType.FP]["pipeline_path"]
CONFIG_PATH = CLASSIFIER_CONFIG[ClassifierType.FP]["config_path"]

# Confidence levels for FP classifier (probability-based)
# Higher probability = higher confidence it's a true sportswear article
# Lower probability = lower confidence (likely false positive)
CONFIDENCE_LEVELS: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 0.3),  # Likely false positive
    "medium": (0.3, 0.6),  # Uncertain
    "high": (0.6, 1.0),  # Likely true sportswear article
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load classifier configuration from JSON file.

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        Dictionary containing classifier configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = json.load(f)

    return config


def get_confidence_level(probability: float) -> str:
    """Map probability to confidence level category (FP classifier specific).

    The confidence level indicates how confident the classifier is that the
    article is about a sportswear brand:
    - "high": High confidence it's a true sportswear article (prob >= 0.6)
    - "medium": Uncertain, may need manual review (0.3 <= prob < 0.6)
    - "low": Low confidence, likely a false positive (prob < 0.3)

    Args:
        probability: Probability from classifier (0.0 to 1.0)

    Returns:
        Confidence level string: "low", "medium", or "high"

    Raises:
        ValueError: If probability is not between 0 and 1
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    for level, (low, high) in CONFIDENCE_LEVELS.items():
        if low <= probability < high:
            return level

    # Handle edge case of probability == 1.0
    return "high"


def save_config(
    config_path: str,
    threshold: float,
    metrics: Dict[str, float],
    model_name: str = MODEL_NAME,
    transformer_method: str = "sentence_transformer_ner",
    best_params: Dict[str, Any] = None,
    target_recall: float = None,
) -> None:
    """Save classifier configuration to JSON file.

    Args:
        config_path: Path to save configuration
        threshold: Optimal threshold for classification
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        transformer_method: Feature transformation method used
        best_params: Best hyperparameters from tuning
        target_recall: Target recall used for threshold optimization
    """
    config = {
        "threshold": threshold,
        "target_recall": target_recall or TARGET_RECALL,
        "model_name": model_name,
        "transformer_method": transformer_method,
        "best_params": best_params or {},
        **metrics,
    }

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
