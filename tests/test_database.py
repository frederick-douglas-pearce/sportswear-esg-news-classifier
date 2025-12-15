"""Tests for database operations.

These tests require PostgreSQL with pgvector extension.
They are skipped by default and can be run with:
    pytest tests/test_database.py --run-db-tests

Or run against the Docker PostgreSQL:
    DATABASE_URL=postgresql://postgres:postgres@localhost:5434/esg_news pytest tests/test_database.py --run-db-tests
"""

import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data_collection.database import Database
from src.data_collection.api_client import ArticleData
from src.data_collection.models import Article, CollectionRun, Base


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "db: marks tests as requiring PostgreSQL database"
    )


# Check if we should run database tests
def should_run_db_tests():
    """Check if database tests should run."""
    # Check for explicit flag or DATABASE_URL environment variable
    return os.environ.get("RUN_DB_TESTS") == "1" or os.environ.get("DATABASE_URL")


# Skip marker for database tests
requires_postgres = pytest.mark.skipif(
    not should_run_db_tests(),
    reason="Database tests require PostgreSQL. Set RUN_DB_TESTS=1 or DATABASE_URL to run."
)


@pytest.fixture
def test_db():
    """Create a test database connection.

    Uses the DATABASE_URL environment variable or defaults to the Docker PostgreSQL.
    """
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5434/esg_news_test"
    )

    db = Database(database_url=database_url)
    db.init_db()

    yield db

    # Cleanup: remove test data
    with db.get_session() as session:
        session.query(Article).delete()
        session.query(CollectionRun).delete()


@pytest.fixture
def sample_article_data():
    """Create sample ArticleData for testing."""
    return ArticleData(
        article_id="test_123",
        title="Test Article Title",
        description="Test description",
        content="Full test content",
        url="https://example.com/test",
        image_url="https://example.com/image.jpg",
        published_at=datetime(2024, 12, 14),
        source_name="Test Source",
        source_url="https://example.com",
        language="en",
        country=["us"],
        category=["business"],
        keywords=["test", "article"],
        brands_mentioned=["Nike", "Adidas"],
        raw_response={"article_id": "test_123"},
    )


@requires_postgres
class TestUpsertArticle:
    """Tests for article upsert operations."""

    def test_insert_new_article(self, test_db, sample_article_data):
        """Should insert a new article and return is_new=True."""
        with test_db.get_session() as session:
            article, is_new = test_db.upsert_article(session, sample_article_data)

            assert is_new is True
            assert article.article_id == "test_123"
            assert article.title == "Test Article Title"
            assert article.scrape_status == "pending"

    def test_duplicate_returns_existing(self, test_db, sample_article_data):
        """Should return existing article with is_new=False for duplicates."""
        with test_db.get_session() as session:
            # Insert first time
            article1, is_new1 = test_db.upsert_article(session, sample_article_data)

            # Insert same article again
            article2, is_new2 = test_db.upsert_article(session, sample_article_data)

            assert is_new1 is True
            assert is_new2 is False
            assert article1.id == article2.id

    def test_article_fields_populated(self, test_db, sample_article_data):
        """Should populate all article fields correctly."""
        with test_db.get_session() as session:
            article, _ = test_db.upsert_article(session, sample_article_data)

            assert article.description == "Test description"
            assert article.url == "https://example.com/test"
            assert article.source_name == "Test Source"
            assert article.language == "en"
            assert "Nike" in article.brands_mentioned
            assert "Adidas" in article.brands_mentioned


@requires_postgres
class TestGetArticlesPendingScrape:
    """Tests for fetching articles pending scrape."""

    def test_returns_pending_articles(self, test_db, sample_article_data):
        """Should return articles with pending scrape status."""
        with test_db.get_session() as session:
            # Insert article (default status is pending)
            test_db.upsert_article(session, sample_article_data)

        with test_db.get_session() as session:
            pending = test_db.get_articles_pending_scrape(session, limit=10)

            assert len(pending) == 1
            assert pending[0].article_id == "test_123"

    def test_excludes_scraped_articles(self, test_db, sample_article_data):
        """Should not return already scraped articles."""
        with test_db.get_session() as session:
            article, _ = test_db.upsert_article(session, sample_article_data)
            article.scrape_status = "success"

        with test_db.get_session() as session:
            pending = test_db.get_articles_pending_scrape(session, limit=10)

            assert len(pending) == 0

    def test_respects_limit(self, test_db):
        """Should respect the limit parameter."""
        # Insert multiple articles
        with test_db.get_session() as session:
            for i in range(5):
                article_data = ArticleData(
                    article_id=f"article_{i}",
                    title=f"Article {i}",
                    url=f"https://example.com/{i}",
                )
                test_db.upsert_article(session, article_data)

        with test_db.get_session() as session:
            pending = test_db.get_articles_pending_scrape(session, limit=3)

            assert len(pending) == 3


@requires_postgres
class TestUpdateArticleContent:
    """Tests for updating article content after scraping."""

    def test_update_success(self, test_db, sample_article_data):
        """Should update article with scraped content."""
        with test_db.get_session() as session:
            test_db.upsert_article(session, sample_article_data)

        with test_db.get_session() as session:
            test_db.update_article_content(
                session,
                "test_123",
                "Full scraped content here",
                "success",
            )

        with test_db.get_session() as session:
            article = session.query(Article).filter_by(article_id="test_123").first()

            assert article.full_content == "Full scraped content here"
            assert article.scrape_status == "success"
            assert article.scraped_at is not None

    def test_update_failure(self, test_db, sample_article_data):
        """Should update article with failure status and error."""
        with test_db.get_session() as session:
            test_db.upsert_article(session, sample_article_data)

        with test_db.get_session() as session:
            test_db.update_article_content(
                session,
                "test_123",
                None,
                "failed",
                "Connection timeout",
            )

        with test_db.get_session() as session:
            article = session.query(Article).filter_by(article_id="test_123").first()

            assert article.full_content is None
            assert article.scrape_status == "failed"
            assert article.scrape_error == "Connection timeout"


@requires_postgres
class TestCollectionRun:
    """Tests for collection run tracking."""

    def test_create_collection_run(self, test_db):
        """Should create a new collection run."""
        with test_db.get_session() as session:
            run = test_db.create_collection_run(session)

            assert run.id is not None
            assert run.status == "running"
            assert run.started_at is not None

    def test_complete_collection_run(self, test_db):
        """Should update collection run with final statistics."""
        with test_db.get_session() as session:
            run = test_db.create_collection_run(session)
            run_id = run.id

        with test_db.get_session() as session:
            run = session.get(CollectionRun, run_id)
            test_db.complete_collection_run(
                session,
                run,
                api_calls=10,
                articles_fetched=50,
                articles_duplicates=5,
                articles_scraped=45,
                articles_scrape_failed=3,
                status="success",
            )

        with test_db.get_session() as session:
            run = session.get(CollectionRun, run_id)

            assert run.api_calls_made == 10
            assert run.articles_fetched == 50
            assert run.articles_duplicates == 5
            assert run.articles_scraped == 45
            assert run.articles_scrape_failed == 3
            assert run.status == "success"
            assert run.completed_at is not None


@requires_postgres
class TestArticleCounts:
    """Tests for article count queries."""

    def test_get_article_count(self, test_db):
        """Should return total article count."""
        with test_db.get_session() as session:
            for i in range(3):
                article_data = ArticleData(
                    article_id=f"article_{i}",
                    title=f"Article {i}",
                    url=f"https://example.com/{i}",
                )
                test_db.upsert_article(session, article_data)

        with test_db.get_session() as session:
            count = test_db.get_article_count(session)

            assert count == 3

    def test_get_scraped_article_count(self, test_db):
        """Should return count of successfully scraped articles."""
        with test_db.get_session() as session:
            for i in range(3):
                article_data = ArticleData(
                    article_id=f"article_{i}",
                    title=f"Article {i}",
                    url=f"https://example.com/{i}",
                )
                article, _ = test_db.upsert_article(session, article_data)
                if i < 2:  # Mark first 2 as scraped
                    article.scrape_status = "success"

        with test_db.get_session() as session:
            count = test_db.get_scraped_article_count(session)

            assert count == 2
