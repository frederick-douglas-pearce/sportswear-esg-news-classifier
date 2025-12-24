#!/usr/bin/env python3
"""Unified FastAPI service for all classifiers.

This script provides a REST API for classifier predictions. The classifier
type is determined by the CLASSIFIER_TYPE environment variable.

Usage:
    # FP Classifier (default)
    CLASSIFIER_TYPE=fp uv run python scripts/predict.py

    # EP Classifier
    CLASSIFIER_TYPE=ep uv run python scripts/predict.py

    # With custom port
    CLASSIFIER_TYPE=fp PORT=8001 uv run python scripts/predict.py

    # With uvicorn directly
    CLASSIFIER_TYPE=ep uv run uvicorn scripts.predict:app --host 0.0.0.0 --port 8001

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import (
    ClassifierType,
    create_classifier,
    BaseClassifier,
)

# Prediction logging configuration
LOGS_DIR = Path(os.getenv("PREDICTION_LOGS_DIR", "logs/predictions"))
ENABLE_PREDICTION_LOGGING = os.getenv("ENABLE_PREDICTION_LOGGING", "true").lower() == "true"


def log_prediction(classifier_type: str, prediction: Dict[str, Any]) -> None:
    """Log a single prediction to JSONL file for drift monitoring.

    Logs are written to logs/predictions/{classifier}_predictions_{date}.jsonl

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        prediction: Prediction result dictionary
    """
    if not ENABLE_PREDICTION_LOGGING:
        return

    try:
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate log file path for today
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = LOGS_DIR / f"{classifier_type}_predictions_{date_str}.jsonl"

        # Create log entry with timestamp
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **prediction,
        }

        # Append to log file
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        # Don't let logging failures affect predictions
        print(f"Warning: Failed to log prediction: {e}")


def log_predictions_batch(classifier_type: str, predictions: List[Dict[str, Any]]) -> None:
    """Log multiple predictions to JSONL file.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        predictions: List of prediction result dictionaries
    """
    for prediction in predictions:
        log_prediction(classifier_type, prediction)

# Get classifier type from environment
CLASSIFIER_TYPE_STR = os.getenv("CLASSIFIER_TYPE", "fp").lower()
try:
    CLASSIFIER_TYPE = ClassifierType(CLASSIFIER_TYPE_STR)
except ValueError:
    print(f"Invalid CLASSIFIER_TYPE: {CLASSIFIER_TYPE_STR}")
    print(f"Valid options: {[c.value for c in ClassifierType]}")
    sys.exit(1)

# Global classifier instance
classifier: Optional[BaseClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    global classifier

    try:
        classifier = create_classifier(CLASSIFIER_TYPE)
        print(f"Loaded {CLASSIFIER_TYPE.value.upper()} classifier")
        print(f"Model: {classifier.model_name}")
        print(f"Threshold: {classifier.threshold:.4f}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model: {e}")
        raise
    except ValueError as e:
        print(f"ERROR: {e}")
        raise

    yield

    classifier = None


# Classifier-specific metadata
CLASSIFIER_METADATA = {
    ClassifierType.FP: {
        "title": "FP Brand Classifier API",
        "description": (
            "Classifies whether news articles are about sportswear brands "
            "(true positive) or false positive brand mentions "
            "(e.g., 'Puma' animal, 'Patagonia' region)."
        ),
    },
    ClassifierType.EP: {
        "title": "EP ESG Pre-filter API",
        "description": (
            "Classifies whether news articles contain ESG "
            "(Environmental, Social, Governance) content that should be "
            "passed to the detailed labeling pipeline."
        ),
    },
    ClassifierType.ESG: {
        "title": "ESG Multi-label Classifier API",
        "description": (
            "Multi-label ESG classification with sentiment for detailed "
            "article categorization."
        ),
    },
}

metadata = CLASSIFIER_METADATA.get(CLASSIFIER_TYPE, CLASSIFIER_METADATA[ClassifierType.FP])

app = FastAPI(
    title=metadata["title"],
    description=metadata["description"],
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for request/response validation
class ArticleRequest(BaseModel):
    """Request model for single article prediction."""

    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content/body text")
    brands: List[str] = Field(
        default_factory=list,
        description="List of brand names mentioned in the article",
    )
    source_name: Optional[str] = Field(
        default=None,
        description="News source name (e.g., 'ESPN', 'National Geographic')",
    )
    category: Optional[List[str]] = Field(
        default=None,
        description="Article categories (e.g., ['sports', 'business'])",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Nike announces new sustainability initiative",
                    "content": "The athletic footwear giant unveiled plans to reduce carbon emissions by 50% in its manufacturing process.",
                    "brands": ["Nike"],
                    "source_name": "ESPN",
                    "category": ["sports", "business"],
                }
            ]
        }
    }


class FPPredictionResponse(BaseModel):
    """Response model for FP classifier predictions."""

    is_sportswear: bool = Field(
        ..., description="True if article is about sportswear brands"
    )
    probability: float = Field(
        ..., description="Probability of being sportswear-related (0.0 to 1.0)"
    )
    risk_level: str = Field(
        ..., description="Risk level: 'low', 'medium', or 'high'"
    )
    threshold: float = Field(..., description="Classification threshold used")


class EPPredictionResponse(BaseModel):
    """Response model for EP classifier predictions."""

    has_esg: bool = Field(
        ..., description="True if article contains ESG content"
    )
    probability: float = Field(
        ..., description="Probability of containing ESG content (0.0 to 1.0)"
    )
    threshold: float = Field(..., description="Classification threshold used")


class BatchArticleRequest(BaseModel):
    """Request model for batch prediction."""

    articles: List[ArticleRequest] = Field(
        ..., description="List of articles to classify"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[Dict[str, Any]] = Field(
        ..., description="List of prediction results"
    )
    count: int = Field(..., description="Number of articles processed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    classifier_type: str = Field(..., description="Type of classifier loaded")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    classifier_type: str
    model_name: str
    threshold: float
    target_recall: float
    transformer_method: str
    best_params: Dict[str, Any]
    metrics: Dict[str, Optional[float]]


# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint for container orchestration.

    Returns:
        Service status and whether model is loaded
    """
    return HealthResponse(
        status="healthy" if classifier is not None else "unhealthy",
        model_loaded=classifier is not None,
        classifier_type=CLASSIFIER_TYPE.value,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics.

    Returns:
        Model information including name, threshold, and metrics
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = classifier.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", tags=["Prediction"])
async def predict(
    request: ArticleRequest,
    background_tasks: BackgroundTasks,
) -> Union[FPPredictionResponse, EPPredictionResponse, Dict[str, Any]]:
    """Classify an article.

    The response format depends on the classifier type:
    - FP: is_sportswear, probability, risk_level, threshold
    - EP: has_esg, probability, threshold

    Args:
        request: Article with title, content, brands, and optional metadata

    Returns:
        Prediction result with classifier-specific fields
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = classifier.predict_from_fields(
        title=request.title,
        content=request.content,
        brands=request.brands,
        source_name=request.source_name,
        category=request.category,
    )

    # Log prediction asynchronously for drift monitoring
    background_tasks.add_task(log_prediction, CLASSIFIER_TYPE.value, result)

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchArticleRequest,
    background_tasks: BackgroundTasks,
):
    """Classify multiple articles in a single request.

    More efficient than calling /predict multiple times.

    Args:
        request: Batch of articles to classify

    Returns:
        List of prediction results
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.articles:
        return BatchPredictionResponse(predictions=[], count=0)

    # Convert to format expected by classifier
    articles = [
        {
            "title": article.title,
            "content": article.content,
            "brands": article.brands,
            "source_name": article.source_name,
            "category": article.category,
        }
        for article in request.articles
    ]

    results = classifier.predict_batch_from_fields(articles)

    # Log predictions asynchronously for drift monitoring
    background_tasks.add_task(log_predictions_batch, CLASSIFIER_TYPE.value, results)

    return BatchPredictionResponse(predictions=results, count=len(results))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    print(f"Starting {CLASSIFIER_TYPE.value.upper()} classifier API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
