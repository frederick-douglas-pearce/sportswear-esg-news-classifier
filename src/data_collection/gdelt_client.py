"""GDELT DOC 2.0 API client wrapper."""

import hashlib
import logging
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx

from .api_client import ArticleData
from .config import BRANDS, KEYWORDS, LANGUAGE, settings

logger = logging.getLogger(__name__)

# GDELT API base URL
GDELT_API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT has an undocumented query length limit of ~250 chars
# We use 230 to leave room for sourcelang filter and safety margin
GDELT_MAX_QUERY_LENGTH = 230

# Language code mapping (GDELT uses full names)
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


class GDELTClient:
    """Wrapper around GDELT DOC 2.0 API."""

    def __init__(self):
        """Initialize the GDELT client."""
        self.api_calls_made = 0
        self.client = httpx.Client(timeout=30.0)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def _generate_article_id(self, url: str) -> str:
        """Generate a unique article ID from URL hash."""
        return hashlib.md5(url.encode()).hexdigest()

    def _extract_brands(self, text: str) -> list[str]:
        """
        Extract mentioned brands from article text using word boundary matching.

        This prevents false positives like:
        - 'Anta' matching 'Santa' or 'amenities'
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

    def _parse_seendate(self, seendate: str) -> datetime | None:
        """Parse GDELT seendate format (YYYYMMDDTHHMMSSZ) into datetime."""
        if not seendate:
            return None
        try:
            # Format: 20251203T143000Z
            return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
        except (ValueError, AttributeError):
            logger.warning(f"Failed to parse seendate: {seendate}")
            return None

    def _parse_article(self, raw: dict[str, Any]) -> ArticleData | None:
        """Parse raw GDELT API response into validated ArticleData."""
        url = raw.get("url", "")
        if not url:
            return None

        title = raw.get("title", "")
        if not title:
            return None

        # Generate article_id from URL hash
        article_id = self._generate_article_id(url)

        # Extract brands from title (GDELT doesn't provide description/content)
        brands = self._extract_brands(title)

        # Parse publication date
        pub_date = self._parse_seendate(raw.get("seendate", ""))

        # Handle country - GDELT returns single country string
        country = raw.get("sourcecountry", "")
        country_list = [country] if country else []

        return ArticleData(
            article_id=article_id,
            title=title,
            description=None,  # GDELT doesn't provide description
            content=None,  # GDELT doesn't provide content
            url=url,
            image_url=raw.get("socialimage"),
            published_at=pub_date,
            source_name=raw.get("domain"),
            source_url=f"https://{raw.get('domain', '')}" if raw.get("domain") else None,
            language=raw.get("language"),
            country=country_list,
            category=[],  # GDELT doesn't provide categories
            keywords=[],  # GDELT doesn't provide keywords
            brands_mentioned=brands,
            raw_response=raw,
        )

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for GDELT API (YYYYMMDDHHMMSS)."""
        return dt.strftime("%Y%m%d%H%M%S")

    def search_news(
        self,
        query: str,
        language: str = LANGUAGE,
        max_records: int = 250,
        timespan: str | None = None,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
    ) -> tuple[list[ArticleData], str | None]:
        """
        Search for news articles using GDELT DOC 2.0 API.

        Args:
            query: Search query string
            language: Language code (e.g., "en")
            max_records: Maximum results to return (up to 250)
            timespan: Relative time window (e.g., "6h", "1d", "1w", "3m")
            start_datetime: Absolute start time (overrides timespan if provided)
            end_datetime: Absolute end time (use with start_datetime)

        Note: If start_datetime/end_datetime are provided, they override timespan.
              Both must be within the last 3 months.

        Returns:
            Tuple of (list of articles, next_page token or None)
            Note: GDELT doesn't support pagination in the same way, so next_page is always None
        """
        try:
            # Build query parameters
            gdelt_lang = LANGUAGE_MAP.get(language, "English")

            # Construct the API URL
            params = {
                "query": f"{query} sourcelang:{gdelt_lang.lower()}",
                "mode": "artlist",
                "format": "json",
                "maxrecords": str(min(max_records, 250)),
                "sort": "datedesc",
            }

            # Build URL with encoded query
            url = f"{GDELT_API_BASE}?query={quote_plus(params['query'])}"
            url += f"&mode={params['mode']}&format={params['format']}"
            url += f"&maxrecords={params['maxrecords']}&sort={params['sort']}"

            # Add time parameters - absolute dates take precedence over timespan
            if start_datetime:
                url += f"&startdatetime={self._format_datetime(start_datetime)}"
                if end_datetime:
                    url += f"&enddatetime={self._format_datetime(end_datetime)}"
            elif timespan:
                url += f"&timespan={timespan}"

            logger.debug(f"GDELT API request: {url}")

            response = self.client.get(url)
            self.api_calls_made += 1

            if response.status_code != 200:
                logger.error(f"GDELT API error: HTTP {response.status_code}")
                return [], None

            data = response.json()

            # GDELT returns {"articles": [...]} structure
            raw_articles = data.get("articles", [])

            articles = []
            for raw_article in raw_articles:
                try:
                    article = self._parse_article(raw_article)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse GDELT article: {e}")

            logger.info(f"GDELT returned {len(articles)} articles for query: {query[:50]}...")
            return articles, None  # GDELT doesn't have pagination tokens

        except Exception as e:
            logger.error(f"GDELT API request failed: {e}")
            self.api_calls_made += 1
            return [], None

    def _group_brands_for_query(self, keyword: str | None = None) -> list[str]:
        """
        Group brands using OR to maximize coverage per query.

        Args:
            keyword: Optional keyword to append to brand groups.

        Returns:
            List of query strings like "(Nike OR Adidas OR Puma) keyword"
        """
        queries = []
        current_brands: list[str] = []

        # Reserve space for sourcelang filter added in search_news
        # Format: " sourcelang:english" = 19 chars
        sourcelang_overhead = 19

        if keyword:
            # Reserve space for: "(" + ") " + keyword + sourcelang
            overhead = 3 + len(keyword) + sourcelang_overhead
        else:
            # Reserve space for: "(" + ")" + sourcelang
            overhead = 2 + sourcelang_overhead

        for brand in BRANDS:
            # Quote brands with spaces or special characters (like dashes) for GDELT
            needs_quotes = " " in brand or "-" in brand
            brand_query = f'"{brand}"' if needs_quotes else brand

            test_brands = current_brands + [brand_query]
            brand_part = " OR ".join(test_brands)
            query_len = len(brand_part) + overhead

            if query_len <= GDELT_MAX_QUERY_LENGTH:
                current_brands.append(brand_query)
            else:
                if current_brands:
                    brand_part = " OR ".join(current_brands)
                    if keyword:
                        queries.append(f"({brand_part}) {keyword}")
                    else:
                        queries.append(f"({brand_part})")
                current_brands = [brand_query]

        if current_brands:
            brand_part = " OR ".join(current_brands)
            if keyword:
                queries.append(f"({brand_part}) {keyword}")
            else:
                queries.append(f"({brand_part})")

        return queries

    def generate_search_queries(self, brand_only: bool = True) -> list[tuple[str, str | None]]:
        """
        Generate optimized queries for GDELT API.

        Args:
            brand_only: If True, generate queries with only brand names.
                       If False, generate queries combining brands with keywords.

        Returns:
            List of tuples: (search_query, None)
        """
        queries = []

        if brand_only:
            grouped_queries = self._group_brands_for_query(keyword=None)
            for query in grouped_queries:
                queries.append((query, None))
        else:
            for keyword in KEYWORDS:
                grouped_queries = self._group_brands_for_query(keyword)
                for query in grouped_queries:
                    queries.append((query, None))

        return queries

    def get_remaining_calls(self, max_calls: int | None = None) -> int:
        """
        Get number of API calls remaining.

        GDELT doesn't have stated rate limits, but we track calls for consistency.
        """
        max_calls = max_calls or settings.max_api_calls_per_day
        return max(0, max_calls - self.api_calls_made)
