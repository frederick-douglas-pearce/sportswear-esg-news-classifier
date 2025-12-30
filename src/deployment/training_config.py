"""Training configuration schema and loader.

This module defines the schema for training configs exported from notebooks
and provides utilities to load and validate them.

Training configs are exported from the model selection notebooks (fp2, ep2)
and contain all parameters needed to train a classifier from scratch:
- Feature engineering method and parameters
- Model type and hyperparameters
- Cross-validation metrics (for reference)

Example config structure:
{
    "classifier_type": "fp",
    "feature_engineering": {
        "method": "tfidf_lsa_ner_proximity_brands",
        "params": {
            "max_features": 10000,
            "lsa_n_components": 100,
            ...
        }
    },
    "model": {
        "type": "RandomForest",
        "params": {
            "n_estimators": 100,
            "max_depth": 15,
            ...
        },
        "calibrated": false
    },
    "cv_metrics": {
        "f2": 0.985,
        "recall": 0.995,
        ...
    },
    "exported_at": "2024-12-28T...",
    "notebook": "fp2_model_selection_tuning.ipynb"
}
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Supported model types
SUPPORTED_MODELS = {
    "fp": ["RandomForest", "LogisticRegression", "GradientBoosting", "SVM"],
    "ep": ["LogisticRegression", "RandomForest", "SVM", "GradientBoosting"],
    "esg": ["LogisticRegression", "RandomForest"],  # Future
}


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEngineeringConfig":
        return cls(
            method=data["method"],
            params=data.get("params", {}),
        )


@dataclass
class ModelConfig:
    """Model configuration."""

    type: str  # RandomForest, LogisticRegression, SVM, etc.
    params: dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False  # For SVM probability calibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "params": self.params,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            type=data["type"],
            params=data.get("params", {}),
            calibrated=data.get("calibrated", False),
        )


@dataclass
class CVMetrics:
    """Cross-validation metrics from model selection."""

    f2: float
    recall: float
    precision: float
    f1: float = 0.0
    accuracy: float = 0.0
    pr_auc: float = 0.0
    n_folds: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "f2": self.f2,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "pr_auc": self.pr_auc,
            "n_folds": self.n_folds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CVMetrics":
        return cls(
            f2=data.get("f2", data.get("cv_f2", 0.0)),
            recall=data.get("recall", data.get("cv_recall", 0.0)),
            precision=data.get("precision", data.get("cv_precision", 0.0)),
            f1=data.get("f1", data.get("cv_f1", 0.0)),
            accuracy=data.get("accuracy", data.get("cv_accuracy", 0.0)),
            pr_auc=data.get("pr_auc", data.get("cv_pr_auc", 0.0)),
            n_folds=data.get("n_folds", 3),
        )


@dataclass
class TrainingConfig:
    """Complete training configuration exported from notebooks."""

    classifier_type: str  # fp, ep, esg
    feature_engineering: FeatureEngineeringConfig
    model: ModelConfig
    cv_metrics: CVMetrics
    exported_at: str = ""
    notebook: str = ""

    def __post_init__(self):
        if not self.exported_at:
            self.exported_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier_type": self.classifier_type,
            "feature_engineering": self.feature_engineering.to_dict(),
            "model": self.model.to_dict(),
            "cv_metrics": self.cv_metrics.to_dict(),
            "exported_at": self.exported_at,
            "notebook": self.notebook,
        }

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(
            classifier_type=data["classifier_type"],
            feature_engineering=FeatureEngineeringConfig.from_dict(
                data["feature_engineering"]
            ),
            model=ModelConfig.from_dict(data["model"]),
            cv_metrics=CVMetrics.from_dict(data.get("cv_metrics", {})),
            exported_at=data.get("exported_at", ""),
            notebook=data.get("notebook", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_training_config_path(classifier_type: str, models_dir: str = "models") -> Path:
    """Get the path to training config for a classifier type."""
    return Path(models_dir) / f"{classifier_type}_training_config.json"


def load_training_config(
    classifier_type: str,
    models_dir: str = "models",
) -> TrainingConfig:
    """Load training config for a classifier type.

    Args:
        classifier_type: One of 'fp', 'ep', 'esg'
        models_dir: Directory containing model artifacts

    Returns:
        TrainingConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = get_training_config_path(classifier_type, models_dir)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {config_path}\n"
            f"Please run the {classifier_type}2 notebook to export the training config."
        )

    config = TrainingConfig.load(config_path)

    # Validate classifier type matches
    if config.classifier_type != classifier_type:
        raise ValueError(
            f"Config classifier_type '{config.classifier_type}' "
            f"doesn't match requested '{classifier_type}'"
        )

    # Validate model type is supported
    if config.model.type not in SUPPORTED_MODELS.get(classifier_type, []):
        raise ValueError(
            f"Unsupported model type '{config.model.type}' for {classifier_type} classifier. "
            f"Supported: {SUPPORTED_MODELS.get(classifier_type, [])}"
        )

    return config


def training_config_exists(classifier_type: str, models_dir: str = "models") -> bool:
    """Check if training config exists for a classifier type."""
    return get_training_config_path(classifier_type, models_dir).exists()
