"""Configuration management for FP Classifier deployment."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

# Default configuration values
DEFAULT_THRESHOLD = 0.605
TARGET_RECALL = 0.98
MODEL_NAME = "FP_Classifier"

# Paths relative to project root
PIPELINE_PATH = "models/fp_classifier_pipeline.joblib"
CONFIG_PATH = "models/fp_classifier_config.json"

# Risk levels based on probability of being sportswear-related
# Higher probability = higher confidence it's a true positive (sportswear)
# Lower probability = higher risk of being a false positive
RISK_LEVELS: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 0.3),      # Likely false positive
    "medium": (0.3, 0.6),   # Uncertain
    "high": (0.6, 1.0),     # Likely true sportswear article
}


def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
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


def get_risk_level(probability: float) -> str:
    """Map probability to risk level category.

    The risk level indicates confidence that the article is about sportswear:
    - "high": High confidence it's a true sportswear article (prob >= 0.6)
    - "medium": Uncertain, may need manual review (0.3 <= prob < 0.6)
    - "low": Likely a false positive (prob < 0.3)

    Args:
        probability: Probability from classifier (0.0 to 1.0)

    Returns:
        Risk level string: "low", "medium", or "high"

    Raises:
        ValueError: If probability is not between 0 and 1
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")

    for level, (low, high) in RISK_LEVELS.items():
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
) -> None:
    """Save classifier configuration to JSON file.

    Args:
        config_path: Path to save configuration
        threshold: Optimal threshold for classification
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        transformer_method: Feature transformation method used
        best_params: Best hyperparameters from tuning
    """
    config = {
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "model_name": model_name,
        "transformer_method": transformer_method,
        "best_params": best_params or {},
        **metrics,
    }

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
