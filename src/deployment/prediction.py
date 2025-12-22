"""Prediction logic for FP Classifier deployment."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

from .config import (
    CONFIG_PATH,
    DEFAULT_THRESHOLD,
    PIPELINE_PATH,
    get_risk_level,
    load_config,
)
from .preprocessing import prepare_input, truncate_text


class FPClassifier:
    """Production classifier wrapper for False Positive Brand detection.

    This class wraps the trained sklearn Pipeline and provides a clean
    interface for making predictions in production.

    Attributes:
        pipeline: Loaded sklearn Pipeline (transformer + classifier)
        config: Configuration dictionary with threshold and metrics
        threshold: Classification threshold for positive class
        model_name: Name of the model

    Example:
        >>> classifier = FPClassifier()
        >>> result = classifier.predict(
        ...     "Nike announces new sustainability initiative..."
        ... )
        >>> print(result["is_sportswear"], result["probability"])
    """

    def __init__(
        self,
        pipeline_path: str = PIPELINE_PATH,
        config_path: str = CONFIG_PATH,
    ):
        """Initialize the classifier by loading the pipeline and config.

        Args:
            pipeline_path: Path to the joblib-saved sklearn Pipeline
            config_path: Path to the JSON configuration file

        Raises:
            FileNotFoundError: If pipeline or config file doesn't exist
        """
        # Validate paths
        if not Path(pipeline_path).exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load pipeline
        self.pipeline = joblib.load(pipeline_path)

        # Load config
        self.config = load_config(config_path)
        self.threshold = self.config.get("threshold", DEFAULT_THRESHOLD)
        self.model_name = self.config.get("model_name", "FP_Classifier")

    def predict(
        self,
        text: str,
        return_proba: bool = True,
    ) -> Dict[str, Any]:
        """Make a prediction for a single text input.

        Args:
            text: Combined text input (title + content + brands)
            return_proba: Whether to include probability in response

        Returns:
            Dictionary containing:
                - is_sportswear: Boolean indicating if article is about sportswear
                - probability: Probability of positive class (if return_proba=True)
                - risk_level: "low", "medium", or "high"
                - threshold: Classification threshold used
        """
        # Truncate if needed for model limits
        text = truncate_text(text)

        # Get probability from pipeline
        proba = self.pipeline.predict_proba([text])[0, 1]

        # Apply threshold
        is_sportswear = proba >= self.threshold

        result = {
            "is_sportswear": bool(is_sportswear),
            "risk_level": get_risk_level(proba),
            "threshold": self.threshold,
        }

        if return_proba:
            result["probability"] = float(proba)

        return result

    def predict_from_fields(
        self,
        title: str,
        content: str,
        brands: Optional[List[str]] = None,
        source_name: Optional[str] = None,
        category: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Make a prediction from separate article fields.

        Convenience method that combines fields before prediction.

        Args:
            title: Article title
            content: Article content/body
            brands: List of brand names
            source_name: News source name (e.g., "ESPN", "National Geographic")
            category: List of article categories (e.g., ["sports", "business"])

        Returns:
            Prediction result dictionary
        """
        text = prepare_input(title, content, brands, source_name, category)
        return self.predict(text)

    def predict_batch(
        self,
        texts: List[str],
        return_proba: bool = True,
    ) -> List[Dict[str, Any]]:
        """Make predictions for multiple texts.

        More efficient than calling predict() in a loop as it uses
        the pipeline's batch processing capabilities.

        Args:
            texts: List of combined text inputs
            return_proba: Whether to include probabilities

        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []

        # Truncate all texts
        texts = [truncate_text(t) for t in texts]

        # Get probabilities for all texts at once
        probas = self.pipeline.predict_proba(texts)[:, 1]

        # Build results
        results = []
        for proba in probas:
            is_sportswear = proba >= self.threshold
            result = {
                "is_sportswear": bool(is_sportswear),
                "risk_level": get_risk_level(proba),
                "threshold": self.threshold,
            }
            if return_proba:
                result["probability"] = float(proba)
            results.append(result)

        return results

    def predict_batch_from_fields(
        self,
        articles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Make predictions for multiple articles from field dictionaries.

        Args:
            articles: List of dicts with "title", "content", and optional
                      "brands", "source_name", "category"

        Returns:
            List of prediction dictionaries
        """
        texts = [
            prepare_input(
                article.get("title", ""),
                article.get("content", ""),
                article.get("brands"),
                article.get("source_name"),
                article.get("category"),
            )
            for article in articles
        ]
        return self.predict_batch(texts)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance metrics.

        Returns:
            Dictionary with model information including:
                - model_name
                - threshold
                - target_recall
                - cv_metrics
                - test_metrics
                - transformer_method
                - best_params
        """
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "target_recall": self.config.get("target_recall", 0.98),
            "transformer_method": self.config.get(
                "transformer_method", "sentence_transformer_ner"
            ),
            "best_params": self.config.get("best_params", {}),
            "metrics": {
                "cv_f2": self.config.get("cv_f2"),
                "cv_recall": self.config.get("cv_recall"),
                "cv_precision": self.config.get("cv_precision"),
                "test_f2": self.config.get("test_f2"),
                "test_recall": self.config.get("test_recall"),
                "test_precision": self.config.get("test_precision"),
                "threshold_f2": self.config.get("threshold_f2"),
                "threshold_recall": self.config.get("threshold_recall"),
                "threshold_precision": self.config.get("threshold_precision"),
            },
        }

    def __repr__(self) -> str:
        return f"FPClassifier(model={self.model_name}, threshold={self.threshold:.3f})"
