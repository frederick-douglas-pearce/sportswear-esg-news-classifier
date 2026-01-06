"""Multi-classifier deployment module.

This module provides production-ready classifier wrappers for:
- FP (False Positive) Brand Classifier
- EP (ESG Pre-filter) Classifier
- ESG (Multi-label) Classifier (future)

Example usage:
    # FP Classifier
    from src.deployment import FPClassifier
    classifier = FPClassifier()
    result = classifier.predict_from_fields(title="...", content="...")

    # EP Classifier
    from src.deployment import EPClassifier
    classifier = EPClassifier()
    result = classifier.predict_from_fields(title="...", content="...")

    # Using ClassifierType enum
    from src.deployment import ClassifierType, create_classifier
    classifier = create_classifier(ClassifierType.FP)
"""

from .base import BaseClassifier
from .config import (
    # Multi-classifier
    ClassifierType,
    CLASSIFIER_CONFIG,
    get_classifier_config,
    get_classifier_paths,
    # Legacy FP-specific (backwards compatibility)
    DEFAULT_THRESHOLD,
    MODEL_NAME,
    PIPELINE_PATH,
    CONFIG_PATH,
    CONFIDENCE_LEVELS,
    TARGET_RECALL,
    get_confidence_level,
    load_config,
    save_config,
)
from .data import (
    create_text_features,
    load_training_data,
    split_data,
)
from .preprocessing import clean_text, prepare_input, truncate_text
from .training_config import (
    TrainingConfig,
    FeatureEngineeringConfig,
    ModelConfig,
    CVMetrics,
    load_training_config,
    training_config_exists,
    get_training_config_path,
    SUPPORTED_MODELS,
)

# Import classifiers
from .fp import FPClassifier
from .ep import EPClassifier


def create_classifier(classifier_type: ClassifierType) -> BaseClassifier:
    """Factory function to create a classifier by type.

    Args:
        classifier_type: The type of classifier to create

    Returns:
        Instantiated classifier

    Raises:
        ValueError: If classifier type is not implemented

    Example:
        >>> from src.deployment import ClassifierType, create_classifier
        >>> fp = create_classifier(ClassifierType.FP)
        >>> ep = create_classifier(ClassifierType.EP)
    """
    if classifier_type == ClassifierType.FP:
        return FPClassifier()
    elif classifier_type == ClassifierType.EP:
        return EPClassifier()
    elif classifier_type == ClassifierType.ESG:
        raise ValueError("ESG classifier not yet implemented")
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


__all__ = [
    # Base class
    "BaseClassifier",
    # Classifiers
    "FPClassifier",
    "EPClassifier",
    "create_classifier",
    # Multi-classifier config
    "ClassifierType",
    "CLASSIFIER_CONFIG",
    "get_classifier_config",
    "get_classifier_paths",
    # Training config (from notebooks)
    "TrainingConfig",
    "FeatureEngineeringConfig",
    "ModelConfig",
    "CVMetrics",
    "load_training_config",
    "training_config_exists",
    "get_training_config_path",
    "SUPPORTED_MODELS",
    # Legacy FP-specific (backwards compatibility)
    "DEFAULT_THRESHOLD",
    "MODEL_NAME",
    "PIPELINE_PATH",
    "CONFIG_PATH",
    "CONFIDENCE_LEVELS",
    "TARGET_RECALL",
    "get_confidence_level",
    "load_config",
    "save_config",
    # Data utilities
    "create_text_features",
    "load_training_data",
    "split_data",
    # Preprocessing (shared)
    "clean_text",
    "prepare_input",
    "truncate_text",
]
