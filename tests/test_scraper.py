"""Tests for the article scraper."""

from unittest.mock import MagicMock, patch

import pytest

from src.data_collection.scraper import ArticleScraper, ScrapeResult


class TestLanguageDetection:
    """Tests for language detection in the scraper."""

    def test_is_english_with_english_text(self):
        """Should return True for English text."""
        scraper = ArticleScraper()
        text = "This is a long English article about Nike and sustainability. The company has announced new initiatives to reduce carbon emissions and improve their environmental footprint."

        assert scraper._is_english(text) is True

    def test_is_english_with_german_text(self):
        """Should return False for German text."""
        scraper = ArticleScraper()
        text = "Dies ist ein langer deutscher Artikel über Nike und Nachhaltigkeit. Das Unternehmen hat neue Initiativen zur Reduzierung von Kohlenstoffemissionen angekündigt."

        assert scraper._is_english(text) is False

    def test_is_english_with_italian_text(self):
        """Should return False for Italian text."""
        scraper = ArticleScraper()
        text = "Questo è un lungo articolo italiano su Nike e sostenibilità. L'azienda ha annunciato nuove iniziative per ridurre le emissioni di carbonio."

        assert scraper._is_english(text) is False

    def test_is_english_with_french_text(self):
        """Should return False for French text."""
        scraper = ArticleScraper()
        text = "Ceci est un long article français sur Nike et la durabilité. L'entreprise a annoncé de nouvelles initiatives pour réduire les émissions de carbone."

        assert scraper._is_english(text) is False

    def test_is_english_with_spanish_text(self):
        """Should return False for Spanish text."""
        scraper = ArticleScraper()
        text = "Este es un largo artículo en español sobre Nike y la sostenibilidad. La empresa ha anunciado nuevas iniciativas para reducir las emisiones de carbono."

        assert scraper._is_english(text) is False

    def test_is_english_with_short_text_returns_true(self):
        """Should return True for text shorter than min_length (can't reliably detect)."""
        scraper = ArticleScraper()
        text = "Short text"

        assert scraper._is_english(text, min_length=100) is True

    def test_is_english_with_empty_text_returns_true(self):
        """Should return True for empty text."""
        scraper = ArticleScraper()

        assert scraper._is_english("") is True
        assert scraper._is_english(None) is True

    def test_is_english_with_custom_min_length(self):
        """Should respect custom min_length parameter."""
        scraper = ArticleScraper()
        # German text that's longer than 50 chars but shorter than 100
        text = "Dies ist ein deutscher Text über Nike Nachhaltigkeit."

        # With default min_length=100, should return True (too short to detect)
        assert scraper._is_english(text, min_length=100) is True

        # With min_length=20, should detect as German
        assert scraper._is_english(text, min_length=20) is False


class TestScrapeArticleLanguageValidation:
    """Tests for language validation during article scraping."""

    def test_scrape_article_skips_non_english_content(self):
        """Should skip articles with non-English content."""
        scraper = ArticleScraper()

        # Mock the newspaper Article to return German content
        with patch('src.data_collection.scraper.NewspaperArticle') as mock_article_class:
            mock_article = MagicMock()
            mock_article.text = (
                "Dies ist ein langer deutscher Artikel über Nike und Nachhaltigkeit. "
                "Das Unternehmen hat neue Initiativen zur Reduzierung von Kohlenstoffemissionen angekündigt. "
                "Diese Maßnahmen sollen dazu beitragen, die Umweltbelastung zu reduzieren."
            )
            mock_article_class.return_value = mock_article

            result = scraper.scrape_article("https://example.com/german-article")

            assert result.success is False
            assert result.status == "skipped"
            assert "Non-English" in result.error

    def test_scrape_article_accepts_english_content(self):
        """Should accept articles with English content."""
        scraper = ArticleScraper()

        # Mock the newspaper Article to return English content
        with patch('src.data_collection.scraper.NewspaperArticle') as mock_article_class:
            mock_article = MagicMock()
            mock_article.text = (
                "This is a long English article about Nike and sustainability. "
                "The company has announced new initiatives to reduce carbon emissions. "
                "These measures are designed to help reduce environmental impact."
            )
            mock_article_class.return_value = mock_article

            result = scraper.scrape_article("https://example.com/english-article")

            assert result.success is True
            assert result.status == "success"
            assert result.content is not None


class TestPaywallDetection:
    """Tests for paywall detection in the scraper."""

    def test_is_paywall_content_detects_subscription_required(self):
        """Should detect paywall indicators."""
        scraper = ArticleScraper()

        assert scraper._is_paywall_content("Please subscribe to continue reading") is True
        assert scraper._is_paywall_content("This is premium content") is True
        assert scraper._is_paywall_content("Sign up to read the full article") is True

    def test_is_paywall_content_allows_normal_content(self):
        """Should not flag normal content as paywall."""
        scraper = ArticleScraper()

        assert scraper._is_paywall_content("Nike announced new sustainability goals") is False

    def test_is_paywall_content_handles_empty(self):
        """Should handle empty content."""
        scraper = ArticleScraper()

        assert scraper._is_paywall_content("") is False
        assert scraper._is_paywall_content(None) is False


class TestScrapeResult:
    """Tests for the ScrapeResult dataclass."""

    def test_scrape_result_success(self):
        """Should create successful result."""
        result = ScrapeResult(success=True, content="Article content", status="success")

        assert result.success is True
        assert result.content == "Article content"
        assert result.status == "success"
        assert result.error is None

    def test_scrape_result_failure(self):
        """Should create failed result."""
        result = ScrapeResult(success=False, error="Connection timeout", status="failed")

        assert result.success is False
        assert result.content is None
        assert result.error == "Connection timeout"
        assert result.status == "failed"

    def test_scrape_result_skipped(self):
        """Should create skipped result."""
        result = ScrapeResult(success=False, error="Paywall detected", status="skipped")

        assert result.success is False
        assert result.status == "skipped"


class TestBrandValidation:
    """Tests for post-scrape brand validation."""

    def test_finds_brand_in_content(self):
        """Should find brand when it appears in content."""
        scraper = ArticleScraper()
        content = "Nike announced new sustainability initiatives today."

        brands_found, warning = scraper._validate_brand_mentions(content, ["Nike"])

        assert "Nike" in brands_found
        assert warning is None

    def test_finds_multiple_brands(self):
        """Should find multiple brands in content."""
        scraper = ArticleScraper()
        content = "Nike and Adidas are competing on sustainability metrics."

        brands_found, warning = scraper._validate_brand_mentions(content, ["Nike", "Adidas"])

        assert "Nike" in brands_found
        assert "Adidas" in brands_found
        assert warning is None

    def test_case_insensitive_matching(self):
        """Should match brands case-insensitively."""
        scraper = ArticleScraper()
        content = "NIKE announced new goals. The adidas team responded."

        brands_found, warning = scraper._validate_brand_mentions(content, ["Nike", "Adidas"])

        assert "Nike" in brands_found
        assert "Adidas" in brands_found

    def test_warns_when_brand_not_found(self):
        """Should warn when expected brand not found in content."""
        scraper = ArticleScraper()
        content = "This article is about general sports industry news."

        brands_found, warning = scraper._validate_brand_mentions(content, ["Nike"])

        assert len(brands_found) == 0
        assert warning is not None
        assert "Nike" in warning
        assert "not found" in warning.lower() or "None of" in warning

    def test_handles_multi_word_brands(self):
        """Should handle multi-word brand names."""
        scraper = ArticleScraper()
        content = "Under Armour released quarterly earnings. The North Face expanded."

        brands_found, warning = scraper._validate_brand_mentions(
            content, ["Under Armour", "The North Face"]
        )

        assert "Under Armour" in brands_found
        assert "The North Face" in brands_found

    def test_handles_brand_variations(self):
        """Should handle brand name variations."""
        scraper = ArticleScraper()
        content = "lululemon reported strong sales. ASICS earnings exceeded expectations."

        brands_found, warning = scraper._validate_brand_mentions(
            content, ["Lululemon", "ASICS"]
        )

        assert "Lululemon" in brands_found
        assert "ASICS" in brands_found

    def test_empty_expected_brands(self):
        """Should handle empty expected brands list."""
        scraper = ArticleScraper()
        content = "Some article content here."

        brands_found, warning = scraper._validate_brand_mentions(content, [])

        assert len(brands_found) == 0
        assert warning is None

    def test_partial_brand_not_matched(self):
        """Should not match partial brand names (word boundary required)."""
        scraper = ArticleScraper()
        content = "The punishing workout was tough."  # Contains 'puma' but not as brand

        brands_found, warning = scraper._validate_brand_mentions(content, ["Puma"])

        assert "Puma" not in brands_found


class TestScraperStats:
    """Tests for scraper statistics tracking."""

    def test_initial_stats_zero(self):
        """Stats should start at zero."""
        scraper = ArticleScraper()
        stats = scraper.get_stats()

        assert stats["scraped"] == 0
        assert stats["failed"] == 0
        assert stats["brand_validation_warnings"] == 0

    def test_stats_increment_on_success(self):
        """Should increment scraped count on success."""
        scraper = ArticleScraper()

        with patch('src.data_collection.scraper.NewspaperArticle') as mock_article_class:
            mock_article = MagicMock()
            mock_article.text = "This is a sufficiently long English article about Nike sustainability initiatives and environmental programs."
            mock_article_class.return_value = mock_article

            scraper.scrape_article("https://example.com/article")

            assert scraper.articles_scraped == 1
            assert scraper.articles_failed == 0

    def test_stats_increment_on_failure(self):
        """Should increment failed count on failure."""
        scraper = ArticleScraper()

        with patch('src.data_collection.scraper.NewspaperArticle') as mock_article_class:
            mock_article = MagicMock()
            mock_article.text = "Short"  # Too short, will fail
            mock_article_class.return_value = mock_article

            scraper.scrape_article("https://example.com/article")

            assert scraper.articles_scraped == 0
            assert scraper.articles_failed == 1
