"""Tests for the NewsData API client."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data_collection.api_client import NewsDataClient, ArticleData
from src.data_collection.config import BRANDS, KEYWORDS, MAX_QUERY_LENGTH


class TestBrandExtraction:
    """Tests for brand extraction from text."""

    def test_extract_single_brand(self):
        """Should extract a single brand from text."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            text = "Nike announced new sustainability goals today."
            brands = client._extract_brands(text)

            assert "Nike" in brands
            assert len(brands) == 1

    def test_extract_multiple_brands(self):
        """Should extract multiple brands from text."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            text = "Nike and Adidas are competing with Puma on sustainability."
            brands = client._extract_brands(text)

            assert "Nike" in brands
            assert "Adidas" in brands
            assert "Puma" in brands
            assert len(brands) == 3

    def test_extract_brand_case_insensitive(self):
        """Should extract brands regardless of case."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            text = "NIKE and adidas are leading sportswear brands."
            brands = client._extract_brands(text)

            assert "Nike" in brands
            assert "Adidas" in brands

    def test_extract_brand_with_spaces(self):
        """Should extract brands with spaces like 'Under Armour'."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            text = "Under Armour reported strong ESG progress."
            brands = client._extract_brands(text)

            assert "Under Armour" in brands

    def test_extract_no_brands(self):
        """Should return empty list when no brands found."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            text = "Generic sportswear company announces plans."
            brands = client._extract_brands(text)

            assert brands == []

    def test_extract_brands_empty_text(self):
        """Should handle empty text."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            assert client._extract_brands("") == []
            assert client._extract_brands(None) == []


class TestArticleParsing:
    """Tests for parsing raw API responses into ArticleData."""

    def test_parse_complete_article(self, sample_raw_api_response):
        """Should parse a complete API response."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            article = client._parse_article(sample_raw_api_response)

            assert article.article_id == "raw_article_456"
            assert article.title == "Adidas and Puma Compete on ESG Goals"
            assert article.url == "https://example.com/adidas-puma-esg"
            assert "Adidas" in article.brands_mentioned
            assert "Puma" in article.brands_mentioned
            assert article.language == "en"

    def test_parse_article_with_missing_fields(self):
        """Should handle missing optional fields gracefully."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            raw = {
                "article_id": "minimal_123",
                "title": "Minimal Article",
                "link": "https://example.com/minimal",
            }

            article = client._parse_article(raw)

            assert article.article_id == "minimal_123"
            assert article.title == "Minimal Article"
            assert article.description is None
            assert article.content is None
            assert article.country == []
            assert article.category == []

    def test_parse_article_date_formats(self):
        """Should parse ISO date formats correctly."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            # Test with Z suffix
            raw = {
                "article_id": "date_test",
                "title": "Date Test",
                "link": "https://example.com",
                "pubDate": "2024-12-14T10:30:00Z",
            }

            article = client._parse_article(raw)
            assert article.published_at is not None
            assert article.published_at.year == 2024
            assert article.published_at.month == 12
            assert article.published_at.day == 14

    def test_parse_article_invalid_date(self):
        """Should handle invalid date gracefully."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            raw = {
                "article_id": "bad_date",
                "title": "Bad Date Test",
                "link": "https://example.com",
                "pubDate": "not-a-date",
            }

            article = client._parse_article(raw)
            assert article.published_at is None


class TestQueryGeneration:
    """Tests for search query generation."""

    def test_generate_queries_not_empty(self):
        """Should generate at least one query."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()

            assert len(queries) > 0

    def test_all_queries_under_max_length(self):
        """All generated queries should be under MAX_QUERY_LENGTH."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()

            for query, category in queries:
                assert len(query) <= MAX_QUERY_LENGTH, f"Query too long: {query}"

    def test_queries_use_or_operator(self):
        """Queries should use OR to combine brands."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()

            # At least some queries should contain OR
            has_or = any(" OR " in query for query, _ in queries)
            assert has_or, "Expected queries to use OR operator for brand grouping"

    def test_queries_contain_keywords(self):
        """Each query should end with a keyword."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()

            for query, _ in queries:
                # Query format: (Brand1 OR Brand2) keyword
                has_keyword = any(query.endswith(f") {kw}") for kw in KEYWORDS)
                assert has_keyword, f"Query missing keyword: {query}"

    def test_queries_no_category_filter(self):
        """Queries should have no category filter (None)."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()

            for _, category in queries:
                assert category is None, "Expected no category filtering"

    def test_all_keywords_covered(self):
        """All keywords should be covered by at least one query."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client.generate_search_queries()
            query_texts = [q for q, _ in queries]

            for keyword in KEYWORDS:
                has_keyword = any(f") {keyword}" in q for q in query_texts)
                assert has_keyword, f"Keyword not covered: {keyword}"


class TestBrandGrouping:
    """Tests for the brand grouping helper method."""

    def test_group_brands_creates_valid_queries(self):
        """Should create queries with parentheses and OR operators."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client._group_brands_for_query("sustainability")

            for query in queries:
                assert query.startswith("(")
                assert ") sustainability" in query
                assert " OR " in query or query.count("(") == 1

    def test_group_brands_respects_length_limit(self):
        """All grouped queries should be under MAX_QUERY_LENGTH."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            # Test with longest keyword
            longest_keyword = max(KEYWORDS, key=len)
            queries = client._group_brands_for_query(longest_keyword)

            for query in queries:
                assert len(query) <= MAX_QUERY_LENGTH

    def test_group_brands_covers_all_brands(self):
        """All brands should appear in at least one query group."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            client = NewsDataClient(api_key="test_key")

            queries = client._group_brands_for_query("test")
            combined = " ".join(queries)

            for brand in BRANDS:
                assert brand in combined, f"Brand not covered: {brand}"


class TestApiKeyValidation:
    """Tests for API key validation."""

    def test_missing_api_key_raises_error(self):
        """Should raise ValueError when API key is missing."""
        with patch("src.data_collection.api_client.NewsDataApiClient"):
            with patch("src.data_collection.api_client.settings") as mock_settings:
                mock_settings.newsdata_api_key = ""

                with pytest.raises(ValueError, match="API key is required"):
                    NewsDataClient(api_key=None)

    def test_provided_api_key_used(self):
        """Should use provided API key over settings."""
        with patch("src.data_collection.api_client.NewsDataApiClient") as mock_client:
            client = NewsDataClient(api_key="my_test_key")

            assert client.api_key == "my_test_key"
