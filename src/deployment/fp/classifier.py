"""FP (False Positive) Brand Classifier implementation."""

from typing import Any, Dict, List, Optional

from ..base import BaseClassifier
from ..config import (
    ClassifierType,
    CLASSIFIER_CONFIG,
    get_risk_level,
)
from .preprocessing import prepare_input


class FPClassifier(BaseClassifier):
    """Production classifier for False Positive Brand detection.

    This classifier determines if an article is about sportswear brands
    (true positive) or is a false positive where brand names appear in
    non-sportswear contexts (e.g., "Puma" the animal, "Patagonia" the region).

    Attributes:
        pipeline: Loaded sklearn Pipeline (transformer + classifier)
        config: Configuration dictionary with threshold and metrics
        threshold: Classification threshold for positive class
        model_name: Name of the model

    Example:
        >>> classifier = FPClassifier()
        >>> result = classifier.predict_from_fields(
        ...     title="Nike announces new sustainability initiative",
        ...     content="The athletic footwear giant unveiled plans...",
        ...     brands=["Nike"],
        ...     source_name="ESPN"
        ... )
        >>> print(result["is_sportswear"], result["probability"])
    """

    CLASSIFIER_TYPE = "fp"

    def _get_default_paths(self) -> tuple[str, str]:
        """Return default paths for FP pipeline and config files."""
        config = CLASSIFIER_CONFIG[ClassifierType.FP]
        return config["pipeline_path"], config["config_path"]

    def _prepare_input(
        self,
        title: str,
        content: str,
        brands: Optional[List[str]] = None,
        source_name: Optional[str] = None,
        category: Optional[List[str]] = None,
    ) -> str:
        """Prepare input text from article fields using FP-specific preprocessing."""
        return prepare_input(title, content, brands, source_name, category)

    def _build_result(self, probability: float, is_positive: bool) -> Dict[str, Any]:
        """Build FP-specific result dictionary.

        Args:
            probability: Probability from classifier (0.0 to 1.0)
            is_positive: Whether prediction exceeds threshold

        Returns:
            Dictionary with FP-specific fields:
                - is_sportswear: Whether article is about sportswear
                - probability: Model confidence
                - risk_level: "low", "medium", or "high"
                - threshold: Classification threshold used
        """
        return {
            "is_sportswear": bool(is_positive),
            "probability": float(probability),
            "risk_level": get_risk_level(probability),
            "threshold": self.threshold,
        }
