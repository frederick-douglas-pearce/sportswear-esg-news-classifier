"""EP (ESG Pre-filter) Classifier implementation."""

from typing import Any, Dict, List, Optional

from ..base import BaseClassifier
from ..config import ClassifierType, CLASSIFIER_CONFIG
from .preprocessing import prepare_input


class EPClassifier(BaseClassifier):
    """Production classifier for ESG Pre-filter detection.

    This classifier determines if an article contains ESG (Environmental,
    Social, Governance) content that should be passed to the detailed
    ESG labeling pipeline.

    Attributes:
        pipeline: Loaded sklearn Pipeline (transformer + classifier)
        config: Configuration dictionary with threshold and metrics
        threshold: Classification threshold for positive class
        model_name: Name of the model

    Example:
        >>> classifier = EPClassifier()
        >>> result = classifier.predict_from_fields(
        ...     title="Nike announces carbon neutrality goals",
        ...     content="The sportswear giant unveiled plans to reduce emissions...",
        ...     brands=["Nike"],
        ...     source_name="Reuters"
        ... )
        >>> print(result["has_esg"], result["probability"])
    """

    CLASSIFIER_TYPE = "ep"

    def _get_default_paths(self) -> tuple[str, str]:
        """Return default paths for EP pipeline and config files."""
        config = CLASSIFIER_CONFIG[ClassifierType.EP]
        return config["pipeline_path"], config["config_path"]

    def _prepare_input(
        self,
        title: str,
        content: str,
        brands: Optional[List[str]] = None,
        source_name: Optional[str] = None,
        category: Optional[List[str]] = None,
    ) -> str:
        """Prepare input text from article fields using EP-specific preprocessing."""
        return prepare_input(title, content, brands, source_name, category)

    def _build_result(self, probability: float, is_positive: bool) -> Dict[str, Any]:
        """Build EP-specific result dictionary.

        Args:
            probability: Probability from classifier (0.0 to 1.0)
            is_positive: Whether prediction exceeds threshold

        Returns:
            Dictionary with EP-specific fields:
                - has_esg: Whether article contains ESG content
                - probability: Model confidence
                - threshold: Classification threshold used
        """
        return {
            "has_esg": bool(is_positive),
            "probability": float(probability),
            "threshold": self.threshold,
        }
