"""NewsData.io API client wrapper."""

import logging
import re
from datetime import datetime
from typing import Any

from newsdataapi import NewsDataApiClient
from pydantic import BaseModel, Field

from .config import BRANDS, KEYWORDS, LANGUAGE, MAX_QUERY_LENGTH, settings

logger = logging.getLogger(__name__)


class ArticleData(BaseModel):
    """Validated article data from API response."""

    article_id: str
    title: str
    description: str | None = None
    content: str | None = None
    url: str
    image_url: str | None = None
    published_at: datetime | None = None
    source_name: str | None = None
    source_url: str | None = None
    language: str | None = None
    country: list[str] = Field(default_factory=list)
    category: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    brands_mentioned: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class NewsDataClient:
    """Wrapper around NewsData.io API client."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.newsdata_api_key
        if not self.api_key:
            raise ValueError("NewsData API key is required")
        self.client = NewsDataApiClient(apikey=self.api_key)
        self.api_calls_made = 0

    def _extract_brands(self, text: str) -> list[str]:
        """Extract mentioned brands from article text using word boundary matching.

        This prevents false positives like:
        - 'Anta' matching 'Santa' or 'Himanta'
        - 'ASICS' matching 'basic'
        """
        if not text:
            return []
        text_lower = text.lower()
        found_brands = []
        for brand in BRANDS:
            pattern = r"\b" + re.escape(brand.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found_brands.append(brand)
        return found_brands

    def _parse_article(self, raw: dict[str, Any]) -> ArticleData:
        """Parse raw API response into validated ArticleData."""
        combined_text = f"{raw.get('title', '')} {raw.get('description', '')} {raw.get('content', '')}"
        brands = self._extract_brands(combined_text)

        pub_date = None
        if raw.get("pubDate"):
            try:
                pub_date = datetime.fromisoformat(raw["pubDate"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return ArticleData(
            article_id=raw.get("article_id", ""),
            title=raw.get("title", ""),
            description=raw.get("description"),
            content=raw.get("content"),
            url=raw.get("link", ""),
            image_url=raw.get("image_url"),
            published_at=pub_date,
            source_name=raw.get("source_name"),
            source_url=raw.get("source_url"),
            language=raw.get("language"),
            country=raw.get("country") or [],
            category=raw.get("category") or [],
            keywords=raw.get("keywords") or [],
            brands_mentioned=brands,
            raw_response=raw,
        )

    def search_news(
        self,
        query: str,
        category: str | None = None,
        language: str = LANGUAGE,
        page: str | None = None,
    ) -> tuple[list[ArticleData], str | None]:
        """
        Search for news articles.

        Returns:
            Tuple of (list of articles, next_page token or None)
        """
        try:
            response = self.client.news_api(
                q=query,
                language=language,
                category=category,
                page=page,
            )
            self.api_calls_made += 1

            if response.get("status") != "success":
                logger.error(f"API error: {response.get('results', {}).get('message', 'Unknown error')}")
                return [], None

            articles = []
            for raw_article in response.get("results", []):
                try:
                    article = self._parse_article(raw_article)
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")

            next_page = response.get("nextPage")
            return articles, next_page

        except Exception as e:
            logger.error(f"API request failed: {e}")
            self.api_calls_made += 1
            return [], None

    def _group_brands_for_query(self, keyword: str | None = None) -> list[str]:
        """
        Group brands using OR to maximize coverage per query while staying under limit.

        Args:
            keyword: Optional keyword to append to brand groups. If None, returns brand-only queries.

        Returns:
            List of query strings like "(Nike OR Adidas OR Puma) keyword" or "Nike OR Adidas OR Puma"
        """
        queries = []
        current_brands: list[str] = []

        if keyword:
            # Reserve space for: "(" + ") " + keyword
            overhead = 3 + len(keyword)
        else:
            # No keyword, no overhead needed
            overhead = 0

        for brand in BRANDS:
            # Calculate what the query would look like with this brand added
            test_brands = current_brands + [brand]
            brand_part = " OR ".join(test_brands)
            query_len = len(brand_part) + overhead

            if query_len <= MAX_QUERY_LENGTH:
                current_brands.append(brand)
            else:
                # Save current group and start new one
                if current_brands:
                    brand_part = " OR ".join(current_brands)
                    if keyword:
                        queries.append(f"({brand_part}) {keyword}")
                    else:
                        queries.append(brand_part)
                current_brands = [brand]

        # Don't forget the last group
        if current_brands:
            brand_part = " OR ".join(current_brands)
            if keyword:
                queries.append(f"({brand_part}) {keyword}")
            else:
                queries.append(brand_part)

        return queries

    def generate_search_queries(self, brand_only: bool = True) -> list[tuple[str, str | None]]:
        """
        Generate optimized queries combining multiple brands per API call.

        Args:
            brand_only: If True, generate queries with only brand names (no keywords).
                       If False, generate queries combining brands with keywords.

        Strategy:
        1. Group brands using OR operators to maximize coverage per query
        2. All queries stay under MAX_QUERY_LENGTH (100 chars for free tier)
        3. No category filtering to maximize results

        Returns list of tuples: (search_query, category_or_none)
        """
        queries = []

        if brand_only:
            # Generate brand-only queries (no keywords)
            grouped_queries = self._group_brands_for_query(keyword=None)
            for query in grouped_queries:
                queries.append((query, None))
        else:
            # Generate grouped brand queries for each keyword
            for keyword in KEYWORDS:
                grouped_queries = self._group_brands_for_query(keyword)
                for query in grouped_queries:
                    queries.append((query, None))

        return queries

    def get_remaining_calls(self, max_calls: int | None = None) -> int:
        """Get number of API calls remaining for the day."""
        max_calls = max_calls or settings.max_api_calls_per_day
        return max(0, max_calls - self.api_calls_made)
