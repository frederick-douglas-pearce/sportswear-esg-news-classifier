"""Pytest fixtures for ESG News Classifier tests."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data_collection.api_client import ArticleData


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
