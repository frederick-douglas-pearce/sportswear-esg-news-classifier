"""Pytest fixtures for ESG News Classifier tests."""

import hashlib

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data_collection.api_client import ArticleData


@pytest.fixture
def fake_embed_texts():
    """Deterministic stand-in for sentence-transformer embedding.

    Maps each unique text to a fixed pseudo-random unit vector, so identical
    texts get cosine similarity 1.0 and distinct texts are near-orthogonal
    (~0 in high dimensions). This lets tests exercise embedding-dependent logic
    (e.g. deduplication clustering) without downloading the real
    ``all-MiniLM-L6-v2`` model from HuggingFace Hub, which makes CI flaky when
    the Hub rate-limits the runner (see issue #47).

    Returns a callable ``embed(texts, dim=384) -> np.ndarray`` of shape
    ``(len(texts), dim)`` suitable for feeding straight into
    ``sklearn.metrics.pairwise.cosine_similarity``.
    """

    def _embed(texts: list[str], dim: int = 384) -> np.ndarray:
        vectors = []
        for text in texts:
            # Stable, process-independent seed derived from the text itself.
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim)
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return np.array(vectors)

    return _embed


@pytest.fixture
def sample_article_data() -> ArticleData:
    """Create a sample ArticleData for testing."""
    return ArticleData(
        article_id="test_article_123",
        title="Nike Announces New Sustainability Initiative",
        description="Nike commits to carbon neutrality by 2030",
        content="Full article content about Nike's sustainability efforts...",
        url="https://example.com/nike-sustainability",
        image_url="https://example.com/image.jpg",
        published_at=datetime(2024, 12, 14, 10, 0, 0),
        source_name="Example News",
        source_url="https://example.com",
        language="en",
        country=["us"],
        category=["business"],
        keywords=["nike", "sustainability"],
        brands_mentioned=["Nike"],
        raw_response={"article_id": "test_article_123"},
    )


@pytest.fixture
def sample_raw_api_response() -> dict:
    """Create a sample raw API response for testing."""
    return {
        "article_id": "raw_article_456",
        "title": "Adidas and Puma Compete on ESG Goals",
        "description": "Sportswear rivals race to meet sustainability targets",
        "content": "Both Adidas and Puma have announced ambitious ESG goals...",
        "link": "https://example.com/adidas-puma-esg",
        "image_url": "https://example.com/image2.jpg",
        "pubDate": "2024-12-14T12:00:00Z",
        "source_name": "Sports Business News",
        "source_url": "https://sportsbusiness.com",
        "language": "en",
        "country": ["us", "de"],
        "category": ["business", "sports"],
        "keywords": ["adidas", "puma", "esg"],
    }


@pytest.fixture
def multiple_article_data() -> list[ArticleData]:
    """Create multiple ArticleData objects for testing deduplication."""
    return [
        ArticleData(
            article_id=f"article_{i}",
            title=f"Test Article {i}",
            description=f"Description {i}",
            url=f"https://example.com/article-{i}",
            brands_mentioned=["Nike"] if i % 2 == 0 else ["Adidas"],
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_newsdata_client():
    """Create a mock NewsData API client."""
    with patch("src.data_collection.api_client.NewsDataApiClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance
