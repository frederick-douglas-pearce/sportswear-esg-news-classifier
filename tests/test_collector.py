"""Tests for the news collector."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.data_collection.collector import NewsCollector, CollectionStats, is_blocked_domain
from src.data_collection.api_client import ArticleData


class TestCollectionStats:
    """Tests for CollectionStats dataclass."""

    def test_default_values(self):
        """Should initialize with zero counts and empty errors."""
        stats = CollectionStats()

        assert stats.api_calls == 0
        assert stats.articles_fetched == 0
        assert stats.articles_duplicates == 0
        assert stats.articles_no_brand == 0
        assert stats.articles_blocked_domain == 0
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
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1", brands_mentioned=["Nike"]),
            ArticleData(article_id="article_2", title="Article 2", url="http://example.com/2", brands_mentioned=["Adidas"]),
        ]
        articles_batch2 = [
            ArticleData(article_id="article_2", title="Article 2", url="http://example.com/2", brands_mentioned=["Adidas"]),  # Duplicate
            ArticleData(article_id="article_3", title="Article 3", url="http://example.com/3", brands_mentioned=["Nike"]),
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

    def test_duplicate_titles_filtered(self):
        """Should filter out articles with duplicate titles from different sources."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        # Same title from different sources (different article_id)
        articles = [
            ArticleData(article_id="article_1", title="Nike Reports Record Sales", url="http://source1.com/1", brands_mentioned=["Nike"]),
            ArticleData(article_id="article_2", title="Nike Reports Record Sales", url="http://source2.com/1", brands_mentioned=["Nike"]),  # Dup title
            ArticleData(article_id="article_3", title="Adidas Launches New Line", url="http://source1.com/2", brands_mentioned=["Adidas"]),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

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

        # Should have 2 unique articles (by title), 1 duplicate title
        assert stats.articles_fetched == 2
        assert stats.articles_duplicate_title == 1
        assert stats.articles_duplicates == 0  # No ID duplicates

    def test_title_dedup_case_insensitive(self):
        """Should treat titles with different casing as duplicates."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(article_id="article_1", title="Nike News Today", url="http://source1.com/1", brands_mentioned=["Nike"]),
            ArticleData(article_id="article_2", title="NIKE NEWS TODAY", url="http://source2.com/1", brands_mentioned=["Nike"]),  # Same title, diff case
            ArticleData(article_id="article_3", title="  Nike News Today  ", url="http://source3.com/1", brands_mentioned=["Nike"]),  # With whitespace
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

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

        # Should have 1 unique article, 2 duplicate titles
        assert stats.articles_fetched == 1
        assert stats.articles_duplicate_title == 2

    def test_dry_run_does_not_save_to_database(self):
        """Dry run should not call database methods."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1", brands_mentioned=["Nike"]),
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


class TestBrandFiltering:
    """Tests for brand filtering functionality."""

    def test_articles_without_brands_are_filtered(self):
        """Should filter out articles that don't mention any tracked brand."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        # Mix of articles with and without brands
        articles = [
            ArticleData(article_id="article_1", title="Nike News", url="http://example.com/1", brands_mentioned=["Nike"]),
            ArticleData(article_id="article_2", title="Generic News", url="http://example.com/2", brands_mentioned=[]),
            ArticleData(article_id="article_3", title="Adidas News", url="http://example.com/3", brands_mentioned=["Adidas"]),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

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

        # Should have 2 articles with brands, 1 filtered out
        assert stats.articles_fetched == 2
        assert stats.articles_no_brand == 1


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
            ArticleData(article_id="article_1", title="Article 1", url="http://example.com/1", brands_mentioned=["Nike"]),
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
        mock_db.upsert_article.return_value = (MagicMock(), "new")  # (article, status)

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
            ArticleData(article_id="existing_article", title="Existing", url="http://example.com/1", brands_mentioned=["Nike"]),
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
        mock_db.upsert_article.return_value = (MagicMock(), "duplicate_id")  # ID duplicate

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


class TestIsBlockedDomain:
    """Tests for the is_blocked_domain helper function."""

    def test_blocked_source_url(self):
        """Should detect blocked domain in source_url."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            assert is_blocked_domain(None, "openpr.com") is True
            assert is_blocked_domain(None, "https://openpr.com") is True
            assert is_blocked_domain(None, "www.openpr.com") is True

    def test_blocked_article_url(self):
        """Should detect blocked domain in article URL."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            assert is_blocked_domain("https://www.openpr.com/news/12345", None) is True
            assert is_blocked_domain("http://openpr.com/article", None) is True

    def test_case_insensitive(self):
        """Should match blocked domains case-insensitively."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            assert is_blocked_domain(None, "OPENPR.COM") is True
            assert is_blocked_domain(None, "OpenPR.com") is True
            assert is_blocked_domain("https://OPENPR.COM/news", None) is True

    def test_allowed_domains(self):
        """Should allow non-blocked domains."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            assert is_blocked_domain("https://reuters.com/article", "reuters.com") is False
            assert is_blocked_domain("https://bbc.com/news", "bbc.com") is False

    def test_empty_blocklist(self):
        """Should allow all domains when blocklist is empty."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", []):
            assert is_blocked_domain("https://openpr.com/news", "openpr.com") is False

    def test_none_inputs(self):
        """Should handle None inputs gracefully."""
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            assert is_blocked_domain(None, None) is False


class TestDomainFiltering:
    """Tests for domain filtering in article collection."""

    def test_blocked_domain_articles_filtered(self):
        """Should filter out articles from blocked domains."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        # Mix of articles from allowed and blocked domains
        articles = [
            ArticleData(
                article_id="article_1",
                title="Nike News",
                url="http://reuters.com/1",
                source_url="reuters.com",
                brands_mentioned=["Nike"],
            ),
            ArticleData(
                article_id="article_2",
                title="Market Report",
                url="http://openpr.com/2",
                source_url="openpr.com",
                brands_mentioned=["Adidas"],
            ),
            ArticleData(
                article_id="article_3",
                title="Puma Update",
                url="http://bbc.com/3",
                source_url="bbc.com",
                brands_mentioned=["Puma"],
            ),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

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

        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            stats = collector.collect_from_api(max_calls=1, dry_run=True)

        # Should have 2 allowed articles, 1 blocked
        assert stats.articles_fetched == 2
        assert stats.articles_blocked_domain == 1

    def test_multiple_blocked_domains(self):
        """Should filter articles from multiple blocked domains."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(
                article_id="article_1",
                title="News 1",
                url="http://openpr.com/1",
                source_url="openpr.com",
                brands_mentioned=["Nike"],
            ),
            ArticleData(
                article_id="article_2",
                title="News 2",
                url="http://spamsite.com/2",
                source_url="spamsite.com",
                brands_mentioned=["Adidas"],
            ),
            ArticleData(
                article_id="article_3",
                title="News 3",
                url="http://reuters.com/3",
                source_url="reuters.com",
                brands_mentioned=["Puma"],
            ),
        ]

        mock_api.generate_search_queries.return_value = [("query", None)]

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

        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com", "spamsite.com"]):
            stats = collector.collect_from_api(max_calls=1, dry_run=True)

        # Should have 1 allowed article, 2 blocked
        assert stats.articles_fetched == 1
        assert stats.articles_blocked_domain == 2

    def test_blocked_articles_not_saved_to_database(self):
        """Should not save blocked domain articles to database."""
        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_scraper = MagicMock()

        articles = [
            ArticleData(
                article_id="article_1",
                title="Good Article",
                url="http://reuters.com/1",
                source_url="reuters.com",
                brands_mentioned=["Nike"],
            ),
            ArticleData(
                article_id="article_2",
                title="Blocked Article",
                url="http://openpr.com/2",
                source_url="openpr.com",
                brands_mentioned=["Adidas"],
            ),
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
        mock_db.upsert_article.return_value = (MagicMock(), "new")

        collector = NewsCollector(
            database=mock_db,
            api_client=mock_api,
            scraper=mock_scraper,
        )

        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["openpr.com"]):
            stats = collector.collect_from_api(max_calls=1, dry_run=False)

        # Only one article should be saved (the non-blocked one)
        assert mock_db.upsert_article.call_count == 1
        assert stats.articles_fetched == 1
        assert stats.articles_blocked_domain == 1
