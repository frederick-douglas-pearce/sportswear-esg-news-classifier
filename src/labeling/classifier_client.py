"""HTTP client for ML classifier APIs.

This module provides a client for calling the FP (False Positive) classifier
and other ML classifiers deployed as FastAPI services.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class FPPredictionResult:
    """Result from FP classifier API."""

    is_sportswear: bool
    probability: float
    risk_level: str
    threshold: float


@dataclass
class ClassifierPredictionRecord:
    """Record of a classifier prediction for database storage.

    This is a data transfer object that holds all information about
    a classifier prediction before it's saved to the database.
    """

    classifier_type: str  # 'fp', 'ep', 'esg'
    model_version: str
    probability: float
    prediction: bool
    threshold_used: float
    action_taken: str  # 'skipped_llm', 'continued_to_llm', 'failed'

    # FP-specific
    risk_level: str | None = None

    # ESG-specific (future)
    esg_categories: dict[str, Any] | None = None

    # Decision tracking
    skip_reason: str | None = None
    error_message: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ClassifierClient:
    """HTTP client for calling classifier APIs.

    Provides methods to call the FP classifier (and future EP/ESG classifiers)
    deployed as FastAPI services. Handles connection management, retries,
    and error handling.

    Example:
        client = ClassifierClient("http://localhost:8000")
        if client.health_check():
            result = client.predict_fp(
                title="Nike releases new shoe",
                content="...",
                brands=["Nike"],
                source_name="ESPN",
                category=["sports"]
            )
            print(f"Is sportswear: {result.is_sportswear}, prob: {result.probability}")
        client.close()
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ):
        """Initialize the classifier client.

        Args:
            base_url: Base URL of the classifier API (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self._model_info: dict[str, Any] | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._model_info = None

    def health_check(self) -> bool:
        """Check if classifier service is healthy.

        Returns:
            True if service is healthy and model is loaded, False otherwise.
        """
        try:
            client = self._get_client()
            response = client.get("/health")
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy" and data.get("model_loaded", False)
        except Exception as e:
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with model information including name, version, metrics.

        Raises:
            httpx.HTTPError: On API errors.
        """
        if self._model_info is not None:
            return self._model_info

        try:
            client = self._get_client()
            response = client.get("/model/info")
            response.raise_for_status()
            self._model_info = response.json()
            return self._model_info
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {"model_name": "unknown", "version": "unknown"}

    def predict_fp(
        self,
        title: str,
        content: str,
        brands: list[str] | None = None,
        source_name: str | None = None,
        category: list[str] | None = None,
    ) -> FPPredictionResult:
        """Make a single FP classifier prediction.

        Args:
            title: Article title
            content: Article content
            brands: List of brand names mentioned
            source_name: News source name (e.g., "ESPN", "Reuters")
            category: Article categories (e.g., ["sports", "business"])

        Returns:
            FPPredictionResult with prediction details.

        Raises:
            httpx.HTTPError: On API errors.
            httpx.TimeoutException: On request timeout.
        """
        client = self._get_client()

        payload = {
            "title": title,
            "content": content,
            "brands": brands or [],
            "source_name": source_name,
            "category": category or [],
        }

        response = client.post("/predict", json=payload)
        response.raise_for_status()

        data = response.json()
        return FPPredictionResult(
            is_sportswear=data["is_sportswear"],
            probability=data["probability"],
            risk_level=data["risk_level"],
            threshold=data["threshold"],
        )

    def predict_fp_batch(
        self,
        articles: list[dict[str, Any]],
    ) -> list[FPPredictionResult]:
        """Make batch FP classifier predictions.

        Args:
            articles: List of article dicts, each containing:
                - title (str): Article title
                - content (str): Article content
                - brands (list[str], optional): Brand names
                - source_name (str, optional): News source
                - category (list[str], optional): Categories

        Returns:
            List of FPPredictionResult objects.

        Raises:
            httpx.HTTPError: On API errors.
        """
        client = self._get_client()

        # Ensure all required fields are present
        normalized_articles = []
        for article in articles:
            normalized_articles.append({
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "brands": article.get("brands") or [],
                "source_name": article.get("source_name"),
                "category": article.get("category") or [],
            })

        payload = {"articles": normalized_articles}
        response = client.post("/predict/batch", json=payload)
        response.raise_for_status()

        data = response.json()
        return [
            FPPredictionResult(
                is_sportswear=p["is_sportswear"],
                probability=p["probability"],
                risk_level=p["risk_level"],
                threshold=p["threshold"],
            )
            for p in data["predictions"]
        ]
