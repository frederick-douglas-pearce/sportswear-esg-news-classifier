"""Tests for the news collector."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.data_collection.collector import NewsCollector, CollectionStats
from src.data_collection.api_client import ArticleData


class TestCollectionStats:
    """Tests for CollectionStats dataclass."""

    def test_default_values(self):
        """Should initialize with zero counts and empty errors."""
        stats = CollectionStats()

        assert stats.api_calls == 0
        assert stats.articles_fetched == 0
        assert stats.articles_duplicates == 0
        assert stats.articles_scraped == 0
        assert stats.articles_scrape_failed == 0
        assert stats.errors == []

    def test_errors_list_independent(self):
        """Each instance should have independent errors list."""
        stats1 = CollectionStats()
        stats2 = CollectionStats()

        stats1.errors.append("error1")

        assert "error1" in stats1.errors
        assert "error1" not in stats2.errors


class TestInMemoryDeduplication:
    """Tests for in-memory article deduplication."""

    def test_duplicate_articles_counted(self):
        """Should count duplicate articles from same run."""
        # Create mock dependencies
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        # Configure mock API to return articles with duplicates
        articles_batch1 = [
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1"),
            ArticleData(article_id="article_2", title="Article 2", url="http://example.com/2"),
        ]
        articles_batch2 = [
            ArticleData(article_id="article_2", title="Article 2", url="http://example.com/2"),  # Duplicate
            ArticleData(article_id="article_3", title="Article 3", url="http://example.com/3"),
        ]

        mock_api.generate_search_queries.return_value = [
            ("query1", None),
            ("query2", None),
        ]

        # Track api_calls_made properly
        call_count = [0]

        def search_side_effect(*args, **kwargs):
            result = [articles_batch1, articles_batch2][call_count[0]]
            call_count[0] += 1
            return (result, None)

        mock_api.search_news.side_effect = search_side_effect
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=2, dry_run=True)

        # Should have 3 unique articles, 1 duplicate
        assert stats.articles_fetched == 3
        assert stats.articles_duplicates == 1

    def test_dry_run_does_not_save_to_database(self):
        """Dry run should not call database methods."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1"),
        ]

        mock_api.search_news.return_value = (articles, None)
        mock_api.generate_search_queries.return_value = [("query", None)]

        # Use property to track calls
        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            return (articles, None)

        mock_api.search_news.side_effect = search_side_effect
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=1, dry_run=True)

        # Database upsert should not be called in dry run
        mock_db.upsert_article.assert_not_called()
        assert stats.articles_fetched == 1


class TestCollectFromApi:
    """Tests for the collect_from_api method."""

    def test_respects_max_calls_limit(self):
        """Should stop when max_calls is reached."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        mock_api.generate_search_queries.return_value = [
            ("query1", None),
            ("query2", None),
            ("query3", None),
            ("query4", None),
            ("query5", None),
        ]

        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            return ([], None)

        mock_api.search_news.side_effect = search_side_effect
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=3, dry_run=True)

        assert stats.api_calls == 3
        assert mock_api.search_news.call_count == 3

    def test_handles_empty_api_responses(self):
        """Should handle queries that return no articles."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            return ([], None)

        mock_api.search_news.side_effect = search_side_effect
        mock_api.generate_search_queries.return_value = [("query", None)]
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=1, dry_run=True)

        assert stats.articles_fetched == 0
        assert stats.articles_duplicates == 0

    def test_production_mode_saves_to_database(self):
        """Non-dry-run should save articles to database."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1"),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            return (articles, None)

        mock_api.search_news.side_effect = search_side_effect
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        # Mock database session context manager
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.upsert_article.return_value = (MagicMock(), True)  # (article, is_new)

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=1, dry_run=False)

        # Database upsert should be called
        mock_db.upsert_article.assert_called_once()
        assert stats.articles_fetched == 1

    def test_database_duplicates_counted(self):
        """Should count articles that already exist in database."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(article_id="existing_article", title="Existing", url="http://example.com/1"),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

        call_count = [0]

        def search_side_effect(*args, **kwargs):
            call_count[0] += 1
            return (articles, None)

        mock_api.search_news.side_effect = search_side_effect
        type(mock_api).api_calls_made = property(lambda self: call_count[0])

        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.upsert_article.return_value = (MagicMock(), False)  # is_new=False means duplicate

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.collect_from_api(max_calls=1, dry_run=False)

        assert stats.articles_fetched == 0
        assert stats.articles_duplicates == 1


class TestScrapePendingArticles:
    """Tests for the scrape_pending_articles method."""

    def test_dry_run_does_not_scrape(self):
        """Dry run should not actually scrape articles."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        # Mock pending articles
        mock_article = MagicMock()
        mock_article.url = "http://example.com/article"

        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.get_articles_pending_scrape.return_value = [mock_article]

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        stats = collector.scrape_pending_articles(limit=10, dry_run=True)

        # Scraper should not be called in dry run
        mock_scraper.scrape_with_delay.assert_not_called()
        assert stats.articles_scraped == 1  # Counted but not actually scraped

    def test_respects_limit(self):
        """Should respect the article limit parameter."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.get_articles_pending_scrape.return_value = []

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        collector.scrape_pending_articles(limit=50, dry_run=True)

        # Check that limit was passed to database query
        mock_db.get_articles_pending_scrape.assert_called_once()
        call_args = mock_db.get_articles_pending_scrape.call_args
        assert call_args[1]["limit"] == 50
