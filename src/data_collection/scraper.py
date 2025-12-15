"""Article scraper for extracting full text from URLs."""

import logging
import random
import time
from dataclasses import dataclass

from newspaper import Article as NewspaperArticle
from newspaper import ArticleException

from .config import settings

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

PAYWALL_INDICATORS = [
    "subscribe to continue",
    "subscription required",
    "sign up to read",
    "become a member",
    "premium content",
    "for subscribers only",
]


@dataclass
class ScrapeResult:
    """Result of scraping an article."""

    success: bool
    content: str | None = None
    error: str | None = None
    status: str = "pending"


class ArticleScraper:
    """Scraper for extracting full article content from URLs."""

    def __init__(self, delay_seconds: float | None = None):
        self.delay_seconds = delay_seconds or settings.scrape_delay_seconds
        self.articles_scraped = 0
        self.articles_failed = 0

    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(USER_AGENTS)

    def _is_paywall_content(self, text: str) -> bool:
        """Check if the content indicates a paywall."""
        if not text:
            return False
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in PAYWALL_INDICATORS)

    def scrape_article(self, url: str) -> ScrapeResult:
        """
        Scrape full article content from URL.

        Returns:
            ScrapeResult with content or error details
        """
        try:
            article = NewspaperArticle(url)
            article.config.browser_user_agent = self._get_random_user_agent()
            article.config.request_timeout = 30

            article.download()
            article.parse()

            content = article.text

            if not content or len(content.strip()) < 100:
                self.articles_failed += 1
                return ScrapeResult(
                    success=False,
                    error="Content too short or empty",
                    status="failed",
                )

            if self._is_paywall_content(content):
                self.articles_failed += 1
                return ScrapeResult(
                    success=False,
                    error="Paywall detected",
                    status="skipped",
                )

            self.articles_scraped += 1
            return ScrapeResult(
                success=True,
                content=content,
                status="success",
            )

        except ArticleException as e:
            self.articles_failed += 1
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                return ScrapeResult(success=False, error="Article not found (404)", status="failed")
            if "403" in error_msg or "forbidden" in error_msg.lower():
                return ScrapeResult(success=False, error="Access forbidden (403)", status="skipped")
            return ScrapeResult(success=False, error=f"Article exception: {error_msg}", status="failed")

        except Exception as e:
            self.articles_failed += 1
            return ScrapeResult(success=False, error=f"Scrape error: {str(e)}", status="failed")

    def scrape_with_delay(self, url: str) -> ScrapeResult:
        """Scrape article with polite delay between requests."""
        result = self.scrape_article(url)
        time.sleep(self.delay_seconds)
        return result

    def get_stats(self) -> dict[str, int]:
        """Get scraping statistics."""
        return {
            "scraped": self.articles_scraped,
            "failed": self.articles_failed,
        }
