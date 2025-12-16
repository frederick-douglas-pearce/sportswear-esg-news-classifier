"""Tests for the GDELT DOC 2.0 API client."""

import hashlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.data_collection.config import BRANDS
from src.data_collection.gdelt_client import (
    GDELT_MAX_QUERY_LENGTH,
    GDELTClient,
)


class TestArticleIdGeneration:
    """Tests for article ID generation from URLs."""

    def test_generate_article_id_consistent(self):
        """Same URL should always produce same ID."""
        client = GDELTClient()
        url = "https://example.com/article/123"

        id1 = client._generate_article_id(url)
        id2 = client._generate_article_id(url)

        assert id1 == id2

    def test_generate_article_id_unique(self):
        """Different URLs should produce different IDs."""
        client = GDELTClient()

        id1 = client._generate_article_id("https://example.com/article/1")
        id2 = client._generate_article_id("https://example.com/article/2")

        assert id1 != id2

    def test_generate_article_id_is_md5(self):
        """ID should be MD5 hash of URL."""
        client = GDELTClient()
        url = "https://example.com/article/123"

        expected = hashlib.md5(url.encode()).hexdigest()
        actual = client._generate_article_id(url)

        assert actual == expected


class TestBrandExtraction:
    """Tests for brand extraction with word boundaries."""

    def test_extract_brand_exact_match(self):
        """Should extract exact brand matches."""
        client = GDELTClient()
        text = "Nike announces new sustainability program"

        brands = client._extract_brands(text)

        assert "Nike" in brands

    def test_extract_brand_case_insensitive(self):
        """Should match brands regardless of case."""
        client = GDELTClient()
        text = "ADIDAS and nike partnership"

        brands = client._extract_brands(text)

        assert "Nike" in brands
        assert "Adidas" in brands

    def test_no_false_positive_anta(self):
        """Should not match 'Anta' in 'Santa' or 'amenities'."""
        client = GDELTClient()

        assert client._extract_brands("Santa Claus") == []
        assert client._extract_brands("basic amenities") == []

    def test_no_false_positive_asics(self):
        """Should not match 'ASICS' in 'basic'."""
        client = GDELTClient()

        assert client._extract_brands("basic training") == []

    def test_no_false_positive_fila(self):
        """Should not match 'Fila' in 'Philadelphia'."""
        client = GDELTClient()

        assert client._extract_brands("Philadelphia news") == []

    def test_extract_multiple_brands(self):
        """Should extract multiple brands from text."""
        client = GDELTClient()
        text = "Nike and Adidas compete in the Puma market"

        brands = client._extract_brands(text)

        assert "Nike" in brands
        assert "Adidas" in brands
        assert "Puma" in brands


class TestSeendateParsing:
    """Tests for GDELT seendate format parsing."""

    def test_parse_valid_seendate(self):
        """Should parse GDELT timestamp format."""
        client = GDELTClient()

        result = client._parse_seendate("20251203T143000Z")

        assert result == datetime(2025, 12, 3, 14, 30, 0)

    def test_parse_empty_seendate(self):
        """Should return None for empty seendate."""
        client = GDELTClient()

        assert client._parse_seendate("") is None
        assert client._parse_seendate(None) is None

    def test_parse_invalid_seendate(self):
        """Should return None for invalid format."""
        client = GDELTClient()

        assert client._parse_seendate("invalid") is None
        assert client._parse_seendate("2025-12-03") is None


class TestArticleParsing:
    """Tests for parsing GDELT API responses."""

    def test_parse_complete_article(self):
        """Should parse all fields from GDELT response."""
        client = GDELTClient()
        raw = {
            "url": "https://example.com/article",
            "title": "Nike sustainability report",
            "seendate": "20251203T143000Z",
            "socialimage": "https://example.com/image.jpg",
            "domain": "example.com",
            "language": "English",
            "sourcecountry": "United States",
        }

        article = client._parse_article(raw)

        assert article is not None
        assert article.url == "https://example.com/article"
        assert article.title == "Nike sustainability report"
        assert article.image_url == "https://example.com/image.jpg"
        assert article.source_name == "example.com"
        assert article.language == "English"
        assert article.country == ["United States"]
        assert "Nike" in article.brands_mentioned

    def test_parse_article_missing_url(self):
        """Should return None if URL is missing."""
        client = GDELTClient()
        raw = {"title": "Test article"}

        article = client._parse_article(raw)

        assert article is None

    def test_parse_article_missing_title(self):
        """Should return None if title is missing."""
        client = GDELTClient()
        raw = {"url": "https://example.com/article"}

        article = client._parse_article(raw)

        assert article is None

    def test_parse_article_no_description_content(self):
        """GDELT doesn't provide description or content."""
        client = GDELTClient()
        raw = {
            "url": "https://example.com/article",
            "title": "Test article",
        }

        article = client._parse_article(raw)

        assert article.description is None
        assert article.content is None


class TestQueryGeneration:
    """Tests for GDELT query generation."""

    def test_generate_queries_not_empty(self):
        """Should generate at least one query."""
        client = GDELTClient()

        queries = client.generate_search_queries()

        assert len(queries) > 0

    def test_brand_only_queries_have_parentheses(self):
        """Brand-only queries should have parentheses for grouping."""
        client = GDELTClient()

        queries = client.generate_search_queries(brand_only=True)

        for query, _ in queries:
            assert query.startswith("(")
            assert ")" in query

    def test_queries_under_max_length(self):
        """All queries should be under max length."""
        client = GDELTClient()

        queries = client.generate_search_queries(brand_only=True)

        for query, _ in queries:
            assert len(query) <= GDELT_MAX_QUERY_LENGTH

    def test_multi_word_brands_quoted(self):
        """Multi-word brands should be quoted in queries."""
        client = GDELTClient()

        queries = client.generate_search_queries(brand_only=True)
        all_queries = " ".join(q for q, _ in queries)

        # Check that multi-word brands are quoted
        assert '"Under Armour"' in all_queries or "Under Armour" not in BRANDS

    def test_keyword_mode_generates_more_queries(self):
        """Keyword mode should generate more queries than brand-only."""
        client = GDELTClient()

        brand_only = client.generate_search_queries(brand_only=True)
        with_keywords = client.generate_search_queries(brand_only=False)

        assert len(with_keywords) > len(brand_only)


class TestDatetimeFormatting:
    """Tests for datetime formatting for GDELT API."""

    def test_format_datetime(self):
        """Should format datetime in GDELT format."""
        client = GDELTClient()
        dt = datetime(2025, 12, 3, 14, 30, 0)

        result = client._format_datetime(dt)

        assert result == "20251203143000"

    def test_format_datetime_midnight(self):
        """Should format midnight correctly."""
        client = GDELTClient()
        dt = datetime(2025, 1, 1, 0, 0, 0)

        result = client._format_datetime(dt)

        assert result == "20250101000000"


class TestSearchNews:
    """Tests for the search_news API call."""

    def test_search_news_increments_api_calls(self):
        """API calls counter should increment."""
        client = GDELTClient()

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": []}
            mock_get.return_value = mock_response

            client.search_news("Nike")

            assert client.api_calls_made == 1

    def test_search_news_parses_articles(self):
        """Should parse articles from API response."""
        client = GDELTClient()

        mock_articles = [
            {
                "url": "https://example.com/1",
                "title": "Nike news article",
                "seendate": "20251203T143000Z",
                "domain": "example.com",
            },
            {
                "url": "https://example.com/2",
                "title": "Adidas news article",
                "seendate": "20251203T150000Z",
                "domain": "example.com",
            },
        ]

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": mock_articles}
            mock_get.return_value = mock_response

            articles, next_page = client.search_news("Nike OR Adidas")

            assert len(articles) == 2
            assert next_page is None  # GDELT doesn't have pagination

    def test_search_news_handles_error(self):
        """Should handle API errors gracefully."""
        client = GDELTClient()

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            articles, next_page = client.search_news("Nike")

            assert articles == []
            assert next_page is None

    def test_search_news_with_timespan(self):
        """Should include timespan in URL."""
        client = GDELTClient()

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": []}
            mock_get.return_value = mock_response

            client.search_news("Nike", timespan="6h")

            # Check that timespan is in the URL
            call_args = mock_get.call_args
            url = call_args[0][0]
            assert "timespan=6h" in url

    def test_search_news_with_date_range(self):
        """Should include start and end datetime in URL."""
        client = GDELTClient()

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": []}
            mock_get.return_value = mock_response

            start = datetime(2025, 10, 1, 0, 0, 0)
            end = datetime(2025, 10, 7, 23, 59, 59)
            client.search_news("Nike", start_datetime=start, end_datetime=end)

            # Check that dates are in the URL
            call_args = mock_get.call_args
            url = call_args[0][0]
            assert "startdatetime=20251001000000" in url
            assert "enddatetime=20251007235959" in url

    def test_search_news_date_range_overrides_timespan(self):
        """Date range should take precedence over timespan."""
        client = GDELTClient()

        with patch.object(client.client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": []}
            mock_get.return_value = mock_response

            start = datetime(2025, 10, 1, 0, 0, 0)
            client.search_news("Nike", timespan="6h", start_datetime=start)

            # Check that dates are in URL but timespan is not
            call_args = mock_get.call_args
            url = call_args[0][0]
            assert "startdatetime=20251001000000" in url
            assert "timespan" not in url


class TestGetRemainingCalls:
    """Tests for remaining calls tracking."""

    def test_remaining_calls_decrements(self):
        """Remaining calls should decrease as calls are made."""
        client = GDELTClient()
        client.api_calls_made = 5

        remaining = client.get_remaining_calls(max_calls=10)

        assert remaining == 5

    def test_remaining_calls_not_negative(self):
        """Remaining calls should not go negative."""
        client = GDELTClient()
        client.api_calls_made = 15

        remaining = client.get_remaining_calls(max_calls=10)

        assert remaining == 0
