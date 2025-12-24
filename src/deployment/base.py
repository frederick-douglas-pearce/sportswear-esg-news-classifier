"""Abstract base class for all classifiers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .config import load_config


class BaseClassifier(ABC):
    """Abstract base class for production classifier wrappers.

    This class defines the common interface that all classifiers must implement.
    It provides shared functionality for loading pipelines and configurations.

    Subclasses must implement:
        - CLASSIFIER_TYPE: str class attribute identifying the classifier
        - _get_default_paths(): Returns default pipeline and config paths
        - _build_result(): Builds classifier-specific result dictionary
    """

    CLASSIFIER_TYPE: str = ""  # Override in subclasses: "fp", "ep", "esg"

    def __init__(
        self,
        pipeline_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the classifier by loading the pipeline and config.

        Args:
            pipeline_path: Path to the joblib-saved sklearn Pipeline.
                          If None, uses default path for this classifier type.
            config_path: Path to the JSON configuration file.
                        If None, uses default path for this classifier type.

        Raises:
            FileNotFoundError: If pipeline or config file doesn't exist
        """
        # Get default paths if not provided
        default_pipeline, default_config = self._get_default_paths()
        pipeline_path = pipeline_path or default_pipeline
        config_path = config_path or default_config

        # Validate paths
        if not Path(pipeline_path).exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load pipeline
        self.pipeline = joblib.load(pipeline_path)

        # Load config
        self.config = load_config(config_path)
        self.threshold = self.config.get("threshold", 0.5)
        self.model_name = self.config.get("model_name", self.CLASSIFIER_TYPE.upper())

    @abstractmethod
    def _get_default_paths(self) -> tuple[str, str]:
        """Return default paths for pipeline and config files.

        Returns:
            Tuple of (pipeline_path, config_path)
        """
        pass

    @abstractmethod
    def _build_result(self, probability: float, is_positive: bool) -> Dict[str, Any]:
        """Build classifier-specific result dictionary.

        Args:
            probability: Probability from classifier (0.0 to 1.0)
            is_positive: Whether prediction exceeds threshold

        Returns:
            Dictionary with classifier-specific fields
        """
        pass

    @abstractmethod
    def _prepare_input(
        self,
        title: str,
        content: str,
        brands: Optional[List[str]] = None,
        source_name: Optional[str] = None,
        category: Optional[List[str]] = None,
    ) -> str:
        """Prepare input text from article fields.

        Args:
            title: Article title
            content: Article content/body
            brands: List of brand names
            source_name: News source name
            category: List of article categories

        Returns:
            Combined text string ready for classification
        """
        pass

    def _truncate_text(self, text: str, max_length: int = 10000) -> str:
        """Truncate text to maximum length.

        Args:
            text: Input text
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Truncate at word boundary
        truncated = text[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated

    def predict(
        self,
        text: str,
        return_proba: bool = True,
    ) -> Dict[str, Any]:
        """Make a prediction for a single text input.

        Args:
            text: Combined text input
            return_proba: Whether to include probability in response

        Returns:
            Dictionary containing prediction results
        """
        # Truncate if needed
        text = self._truncate_text(text)

        # Get probability from pipeline
        proba = self.pipeline.predict_proba([text])[0, 1]

        # Apply threshold
        is_positive = proba >= self.threshold

        # Build classifier-specific result
        result = self._build_result(proba, is_positive)

        if not return_proba:
            result.pop("probability", None)

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

        Args:
            title: Article title
            content: Article content/body
            brands: List of brand names
            source_name: News source name
            category: List of article categories

        Returns:
            Prediction result dictionary
        """
        text = self._prepare_input(title, content, brands, source_name, category)
        return self.predict(text)

    def predict_batch(
        self,
        texts: List[str],
        return_proba: bool = True,
    ) -> List[Dict[str, Any]]:
        """Make predictions for multiple texts.

        Args:
            texts: List of combined text inputs
            return_proba: Whether to include probabilities

        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []

        # Truncate all texts
        texts = [self._truncate_text(t) for t in texts]

        # Get probabilities for all texts at once
        probas = self.pipeline.predict_proba(texts)[:, 1]

        # Build results
        results = []
        for proba in probas:
            is_positive = proba >= self.threshold
            result = self._build_result(proba, is_positive)
            if not return_proba:
                result.pop("probability", None)
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
            self._prepare_input(
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
            Dictionary with model information
        """
        return {
            "classifier_type": self.CLASSIFIER_TYPE,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "target_recall": self.config.get("target_recall", 0.98),
            "transformer_method": self.config.get("transformer_method", "unknown"),
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
        return f"{self.__class__.__name__}(model={self.model_name}, threshold={self.threshold:.3f})"
