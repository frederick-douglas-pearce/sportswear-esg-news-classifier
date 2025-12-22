"""FastAPI service for FP Brand Classifier.

Provides REST API endpoints for classifying whether articles are about
sportswear brands (true positive) or false positive brand mentions.

Run locally:
    uv run python predict.py
    # Or with uvicorn:
    uv run uvicorn predict:app --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.deployment import FPClassifier

# Global classifier instance (loaded on startup)
classifier: Optional[FPClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    global classifier

    # Load model on startup
    try:
        classifier = FPClassifier()
        print(f"Loaded model: {classifier.model_name}")
        print(f"Threshold: {classifier.threshold:.4f}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model: {e}")
        raise

    yield

    # Cleanup on shutdown (if needed)
    classifier = None


# Create FastAPI app
app = FastAPI(
    title="FP Brand Classifier API",
    description=(
        "Classifies whether news articles are about sportswear brands "
        "(true positive) or false positive brand mentions "
        "(e.g., 'Puma' animal, 'Patagonia' region)."
    ),
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Nike announces new sustainability initiative",
                    "content": "The athletic footwear giant unveiled plans to reduce carbon emissions by 50% in its manufacturing process.",
                    "brands": ["Nike"],
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for prediction results."""

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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_sportswear": True,
                    "probability": 0.92,
                    "risk_level": "high",
                    "threshold": 0.605,
                }
            ]
        }
    }


class BatchArticleRequest(BaseModel):
    """Request model for batch prediction."""

    articles: List[ArticleRequest] = Field(
        ..., description="List of articles to classify"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction results"
    )
    count: int = Field(..., description="Number of articles processed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

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


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: ArticleRequest):
    """Classify if an article is about sportswear brands.

    Args:
        request: Article with title, content, and brands

    Returns:
        Prediction result with probability and risk level
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = classifier.predict_from_fields(
        title=request.title,
        content=request.content,
        brands=request.brands,
    )

    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchArticleRequest):
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
        }
        for article in request.articles
    ]

    results = classifier.predict_batch_from_fields(articles)
    predictions = [PredictionResponse(**r) for r in results]

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
