"""Article scraper for extracting full text from URLs."""

import logging
import random
import re
import time
import warnings
from dataclasses import dataclass, field

from dateutil.parser import UnknownTimezoneWarning
from langdetect import LangDetectException, detect
from newspaper import Article as NewspaperArticle

# Suppress timezone warnings from dateutil (used by newspaper internally)
# We already have publication dates from the API, so we don't need parsed dates from scraping
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)
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
    brands_found: list[str] = field(default_factory=list)
    brands_expected: list[str] = field(default_factory=list)
    brand_validation_warning: str | None = None


class ArticleScraper:
    """Scraper for extracting full article content from URLs."""

    def __init__(self, delay_seconds: float | None = None):
        self.delay_seconds = delay_seconds or settings.scrape_delay_seconds
        self.articles_scraped = 0
        self.articles_failed = 0
        self.brand_validation_warnings = 0

    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(USER_AGENTS)

    def _is_paywall_content(self, text: str) -> bool:
        """Check if the content indicates a paywall."""
        if not text:
            return False
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in PAYWALL_INDICATORS)

    def _is_english(self, text: str, min_length: int = 100) -> bool:
        """
        Check if text is English using language detection.

        Args:
            text: Text to check
            min_length: Minimum text length for reliable detection

        Returns:
            True if English or detection uncertain, False if definitely non-English
        """
        if not text or len(text) < min_length:
            return True  # Can't reliably detect, assume English
        try:
            return detect(text) == "en"
        except LangDetectException:
            return True  # Detection failed, assume English

    def _validate_brand_mentions(
        self, content: str, expected_brands: list[str]
    ) -> tuple[list[str], str | None]:
        """
        Validate that expected brands appear in scraped content.

        This helps detect when the scraper grabbed boilerplate/ads instead of
        actual article content.

        Args:
            content: Scraped article text
            expected_brands: List of brand names expected to appear

        Returns:
            Tuple of (brands_found, warning_message or None)
        """
        if not expected_brands:
            return [], None

        # Brand name patterns (handles common variations)
        # Maps canonical brand name to regex patterns
        brand_patterns: dict[str, list[str]] = {
            # Standard brands - case insensitive word boundary match
            "Nike": [r"\bNike\b"],
            "Adidas": [r"\bAdidas\b", r"\badidas\b"],
            "Puma": [r"\bPuma\b"],
            "Under Armour": [r"\bUnder\s+Armour\b", r"\bUnderArmour\b"],
            "Lululemon": [r"\bLululemon\b", r"\blululemon\b"],
            "Patagonia": [r"\bPatagonia\b"],
            "Columbia Sportswear": [r"\bColumbia\s+Sportswear\b", r"\bColumbia\b"],
            "New Balance": [r"\bNew\s+Balance\b"],
            "ASICS": [r"\bASICS\b", r"\bAsics\b"],
            "Reebok": [r"\bReebok\b"],
            "Skechers": [r"\bSkechers\b"],
            "Fila": [r"\bFila\b", r"\bFILA\b"],
            "The North Face": [r"\bThe\s+North\s+Face\b", r"\bNorth\s+Face\b"],
            "Vans": [r"\bVans\b"],
            "Converse": [r"\bConverse\b"],
            "Salomon": [r"\bSalomon\b"],
            "Mammut": [r"\bMammut\b"],
            "Umbro": [r"\bUmbro\b"],
            "Anta": [r"\bAnta\s+Sports\b", r"\bAnta\b"],
            "Li-Ning": [r"\bLi-Ning\b", r"\bLi\s*Ning\b"],
            "Brooks Running": [r"\bBrooks\s+Running\b", r"\bBrooks\b"],
            "Decathlon": [r"\bDecathlon\b"],
            "Deckers": [r"\bDeckers\b"],
            "Yonex": [r"\bYonex\b"],
            "Mizuno": [r"\bMizuno\b"],
            "K-Swiss": [r"\bK-Swiss\b"],
            "Altra Running": [r"\bAltra\s+Running\b", r"\bAltra\b"],
            "Hoka": [r"\bHoka\b", r"\bHOKA\b"],
            "Saucony": [r"\bSaucony\b"],
            "Merrell": [r"\bMerrell\b"],
            "Timberland": [r"\bTimberland\b"],
            "Spyder": [r"\bSpyder\b"],
            "On Running": [r"\bOn\s+Running\b", r"\bOn\s+Cloud\b"],
            "Allbirds": [r"\bAllbirds\b"],
            "Gymshark": [r"\bGymshark\b"],
            "Everlast": [r"\bEverlast\b"],
            "Arc'teryx": [r"\bArc'teryx\b", r"\bArcteryx\b"],
            "Jack Wolfskin": [r"\bJack\s+Wolfskin\b"],
            "Athleta": [r"\bAthleta\b"],
            "Vuori": [r"\bVuori\b"],
            "Cotopaxi": [r"\bCotopaxi\b"],
            "Prana": [r"\bPrana\b", r"\bprAna\b"],
            "Eddie Bauer": [r"\bEddie\s+Bauer\b"],
            "361 Degrees": [r"\b361\s*Degrees\b", r"\b361°\b"],
            "Xtep": [r"\bXtep\b"],
            "Peak Sport": [r"\bPeak\s+Sport\b"],
            "Mountain Hardwear": [r"\bMountain\s+Hardwear\b"],
            "Black Diamond": [r"\bBlack\s+Diamond\b"],
            "Outdoor Voices": [r"\bOutdoor\s+Voices\b"],
            "Diadora": [r"\bDiadora\b"],
        }

        brands_found = []
        content_lower = content.lower()

        for brand in expected_brands:
            # Get patterns for this brand, or create a default one
            patterns = brand_patterns.get(brand, [rf"\b{re.escape(brand)}\b"])

            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    brands_found.append(brand)
                    break

        # Generate warning if no expected brands found
        warning = None
        if expected_brands and not brands_found:
            warning = (
                f"None of the expected brands ({', '.join(expected_brands[:3])}"
                f"{', ...' if len(expected_brands) > 3 else ''}) found in scraped content. "
                f"Scraper may have grabbed boilerplate/ads instead of article."
            )
            logger.warning(f"Brand validation: {warning}")

        return brands_found, warning

    def scrape_article(self, url: str, expected_brands: list[str] | None = None) -> ScrapeResult:
        """
        Scrape full article content from URL.

        Args:
            url: Article URL to scrape
            expected_brands: List of brand names expected to appear in content
                (used for validation warning, not failure)

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

            if not self._is_english(content):
                self.articles_failed += 1
                return ScrapeResult(
                    success=False,
                    error="Non-English content detected",
                    status="skipped",
                )

            # Validate brand mentions in content
            brands_found, validation_warning = self._validate_brand_mentions(
                content, expected_brands or []
            )
            if validation_warning:
                self.brand_validation_warnings += 1

            self.articles_scraped += 1
            return ScrapeResult(
                success=True,
                content=content,
                status="success",
                brands_found=brands_found,
                brands_expected=expected_brands or [],
                brand_validation_warning=validation_warning,
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

    def scrape_with_delay(
        self, url: str, expected_brands: list[str] | None = None
    ) -> ScrapeResult:
        """Scrape article with polite delay between requests."""
        result = self.scrape_article(url, expected_brands)
        time.sleep(self.delay_seconds)
        return result

    def get_stats(self) -> dict[str, int]:
        """Get scraping statistics."""
        return {
            "scraped": self.articles_scraped,
            "failed": self.articles_failed,
            "brand_validation_warnings": self.brand_validation_warnings,
        }
