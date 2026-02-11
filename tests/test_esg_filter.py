"""Tests for ESG keyword filter module."""

import pytest

from src.labeling.esg_filter import (
    ESG_KEYWORDS,
    NON_ESG_KEYWORDS,
    ESGFilterResult,
    check_esg_keywords,
    filter_articles_for_labeling,
    should_label_article,
)


class TestESGFilterResult:
    """Tests for ESGFilterResult dataclass."""

    def test_total_esg_keywords_empty(self):
        """Should return 0 when no keywords found."""
        result = ESGFilterResult(
            has_esg_keywords=False,
            has_non_esg_keywords=False,
            esg_keywords_found={},
            non_esg_patterns_matched=[],
            recommendation="review",
        )
        assert result.total_esg_keywords == 0

    def test_total_esg_keywords_single_category(self):
        """Should count keywords in single category."""
        result = ESGFilterResult(
            has_esg_keywords=True,
            has_non_esg_keywords=False,
            esg_keywords_found={"environmental": ["carbon", "emission"]},
            non_esg_patterns_matched=[],
            recommendation="label",
        )
        assert result.total_esg_keywords == 2

    def test_total_esg_keywords_multiple_categories(self):
        """Should count keywords across all categories."""
        result = ESGFilterResult(
            has_esg_keywords=True,
            has_non_esg_keywords=False,
            esg_keywords_found={
                "environmental": ["carbon", "emission"],
                "social": ["diversity", "inclusion"],
                "governance": ["transparency"],
            },
            non_esg_patterns_matched=[],
            recommendation="label",
        )
        assert result.total_esg_keywords == 5


class TestCheckESGKeywords:
    """Tests for check_esg_keywords function."""

    def test_environmental_keywords_detected(self):
        """Should detect environmental keywords."""
        text = "Nike announces carbon neutral commitment with sustainable materials"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is True
        assert "environmental" in result.esg_keywords_found
        assert "carbon" in result.esg_keywords_found["environmental"]
        assert "sustainab" in result.esg_keywords_found["environmental"]

    def test_social_keywords_detected(self):
        """Should detect social keywords."""
        text = "Company improves worker rights and promotes diversity and inclusion"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is True
        assert "social" in result.esg_keywords_found
        assert "worker right" in result.esg_keywords_found["social"]
        assert "diversity" in result.esg_keywords_found["social"]
        assert "inclusion" in result.esg_keywords_found["social"]

    def test_governance_keywords_detected(self):
        """Should detect governance keywords."""
        text = "Board of directors releases ESG disclosure and transparency report"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is True
        assert "governance" in result.esg_keywords_found
        assert "board of director" in result.esg_keywords_found["governance"]
        assert "ESG disclosure" in result.esg_keywords_found["governance"]

    def test_case_insensitive_matching(self):
        """Should match keywords regardless of case."""
        text = "CARBON emissions and SUSTAINABILITY initiatives"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is True
        assert "carbon" in result.esg_keywords_found["environmental"]
        assert "emission" in result.esg_keywords_found["environmental"]
        assert "sustainab" in result.esg_keywords_found["environmental"]

    def test_no_esg_keywords(self):
        """Should return no ESG keywords for non-ESG content."""
        text = "Nike releases new sneaker collection featuring popular colorways"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is False
        assert result.esg_keywords_found == {}

    def test_non_esg_patterns_detected(self):
        """Should detect non-ESG financial patterns."""
        text = "NKE stock surges 5% after Q4 earnings beat analyst expectations"
        result = check_esg_keywords(text)

        assert result.has_non_esg_keywords is True
        assert len(result.non_esg_patterns_matched) > 0

    def test_stock_price_pattern(self):
        """Should detect stock price patterns."""
        text = "Nike stock price climbed 3% today"
        result = check_esg_keywords(text)

        assert result.has_non_esg_keywords is True

    def test_analyst_pattern(self):
        """Should detect analyst patterns."""
        text = "Analysts expect Nike to beat estimates"
        result = check_esg_keywords(text)

        assert result.has_non_esg_keywords is True

    def test_product_launch_pattern(self):
        """Should detect product launch patterns."""
        text = "Nike launches new shoe collection"
        result = check_esg_keywords(text)

        assert result.has_non_esg_keywords is True

    def test_recommendation_label_for_clear_esg(self):
        """Should recommend 'label' for clear ESG content."""
        text = "Nike commits to carbon neutral manufacturing with sustainable materials"
        result = check_esg_keywords(text)

        assert result.recommendation == "label"

    def test_recommendation_label_for_mixed_content(self):
        """Should recommend 'label' when ESG keywords present even with financial content."""
        text = "Nike stock climbs on sustainability announcement with carbon reduction goals"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is True
        assert result.has_non_esg_keywords is True
        assert result.recommendation == "label"

    def test_recommendation_skip_for_pure_financial(self):
        """Should recommend 'skip' for pure financial content."""
        text = "NKE stock surges 5% as analysts upgrade to buy rating"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is False
        assert result.has_non_esg_keywords is True
        assert result.recommendation == "skip"

    def test_recommendation_review_for_unclear(self):
        """Should recommend 'review' when no clear signals."""
        text = "Nike announces new partnership with sports team"
        result = check_esg_keywords(text)

        assert result.has_esg_keywords is False
        assert result.has_non_esg_keywords is False
        assert result.recommendation == "review"


class TestShouldLabelArticle:
    """Tests for should_label_article function."""

    def test_returns_true_for_esg_content(self):
        """Should return True for articles with ESG keywords."""
        title = "Nike Sustainability Report"
        content = "Nike announces carbon reduction goals and sustainable materials initiative"

        should_label, reason = should_label_article(title, content)

        assert should_label is True
        assert "ESG keywords found" in reason

    def test_returns_false_for_financial_only(self):
        """Should return False for pure financial content."""
        title = "Nike Stock Analysis"
        content = "NKE stock price surged 5% after analysts upgraded to buy rating"

        should_label, reason = should_label_article(title, content)

        assert should_label is False
        assert "financial" in reason.lower() or "No ESG" in reason

    def test_returns_true_for_unclear_content(self):
        """Should return True for unclear content (let LLM decide)."""
        title = "Nike Partners with Organization"
        content = "Nike announces partnership with international sports organization"

        should_label, reason = should_label_article(title, content)

        assert should_label is True
        assert "LLM review" in reason or "ESG keywords" in reason

    def test_threshold_parameter(self):
        """Should respect threshold parameter."""
        title = "News"
        content = "This article mentions carbon once"

        # With threshold=1, should label
        should_label1, _ = should_label_article(title, content, threshold=1)
        assert should_label1 is True

        # With threshold=3, should not label (only 1 keyword found)
        should_label3, _ = should_label_article(title, content, threshold=3)
        assert should_label3 is False

    def test_combines_title_and_content(self):
        """Should check both title and content for keywords."""
        title = "Carbon Neutral Initiative"
        content = "The company made an announcement today"

        should_label, reason = should_label_article(title, content)

        assert should_label is True
        assert "ESG keywords found" in reason


class TestFilterArticlesForLabeling:
    """Tests for filter_articles_for_labeling function."""

    def test_separates_articles_correctly(self):
        """Should separate articles into label and skip lists."""
        articles = [
            {
                "title": "Nike Sustainability Report",
                "content": "Carbon neutral goals announced",
            },
            {
                "title": "Stock Analysis",
                "content": "NKE stock surges after earnings beat",
            },
            {
                "title": "Adidas ESG Initiative",
                "content": "Worker rights improvements in supply chain",
            },
        ]

        to_label, to_skip = filter_articles_for_labeling(articles)

        # Two ESG articles should be labeled
        assert len(to_label) == 2
        # One financial article should be skipped
        assert len(to_skip) == 1

    def test_adds_filter_reason_to_articles(self):
        """Should add _filter_reason field to all articles."""
        articles = [
            {
                "title": "Sustainability News",
                "content": "Carbon reduction initiative",
            },
        ]

        to_label, to_skip = filter_articles_for_labeling(articles)

        assert "_filter_reason" in to_label[0]

    def test_custom_key_names(self):
        """Should support custom title and content keys."""
        articles = [
            {
                "headline": "Carbon Neutral Initiative",
                "body": "Sustainable manufacturing goals",
            },
        ]

        to_label, to_skip = filter_articles_for_labeling(
            articles, title_key="headline", content_key="body"
        )

        assert len(to_label) == 1

    def test_handles_missing_keys(self):
        """Should handle articles with missing title or content."""
        articles = [
            {"title": "Some Title"},  # Missing content
            {"content": "Some content about carbon emissions"},  # Missing title
            {},  # Missing both
        ]

        to_label, to_skip = filter_articles_for_labeling(articles)

        # Should not crash, articles should be processed
        assert len(to_label) + len(to_skip) == 3

    def test_empty_list(self):
        """Should handle empty article list."""
        to_label, to_skip = filter_articles_for_labeling([])

        assert to_label == []
        assert to_skip == []


class TestESGKeywordsConfig:
    """Tests for ESG_KEYWORDS configuration."""

    def test_all_categories_present(self):
        """Should have all expected ESG categories."""
        expected_categories = ["environmental", "social", "governance"]
        for cat in expected_categories:
            assert cat in ESG_KEYWORDS

    def test_categories_have_keywords(self):
        """Each category should have at least some keywords."""
        for category, keywords in ESG_KEYWORDS.items():
            assert len(keywords) > 0, f"Category {category} has no keywords"

    def test_environmental_contains_key_terms(self):
        """Environmental category should contain key ESG terms."""
        env_keywords = ESG_KEYWORDS["environmental"]
        assert "carbon" in env_keywords
        assert "sustainab" in env_keywords
        assert "climate" in env_keywords
        assert "emission" in env_keywords

    def test_social_contains_key_terms(self):
        """Social category should contain key ESG terms."""
        social_keywords = ESG_KEYWORDS["social"]
        assert "diversity" in social_keywords
        assert "inclusion" in social_keywords
        assert "human rights" in social_keywords

    def test_governance_contains_key_terms(self):
        """Governance category should contain key ESG terms."""
        gov_keywords = ESG_KEYWORDS["governance"]
        assert "board of director" in gov_keywords
        assert "transparency" in gov_keywords or "transparency report" in gov_keywords


class TestNonESGPatterns:
    """Tests for NON_ESG_KEYWORDS patterns."""

    def test_patterns_are_valid_regex(self):
        """All non-ESG patterns should be valid regex."""
        import re

        for pattern in NON_ESG_KEYWORDS:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern '{pattern}': {e}")

    def test_stock_patterns_match_expected_text(self):
        """Stock patterns should match expected financial text."""
        import re

        stock_texts = [
            "stock tumbles on news",
            "stock surges after earnings",
            "stock price rises",
            "share price drops",
        ]

        for text in stock_texts:
            matched = any(
                re.search(p, text, re.IGNORECASE) for p in NON_ESG_KEYWORDS
            )
            assert matched, f"Expected '{text}' to match a non-ESG pattern"

    def test_earnings_patterns_match_expected_text(self):
        """Earnings patterns should match expected financial text."""
        import re

        earnings_texts = [
            "Q4 revenue beats estimates",
            "quarterly earnings miss expectations",
            "EPS beat by 5 cents",
        ]

        for text in earnings_texts:
            matched = any(
                re.search(p, text, re.IGNORECASE) for p in NON_ESG_KEYWORDS
            )
            assert matched, f"Expected '{text}' to match a non-ESG pattern"
