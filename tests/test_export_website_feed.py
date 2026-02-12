"""Tests for the website feed export script."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from export_website_feed import (
    SCORECARD_BOTTOM_N,
    SCORECARD_TOP_N,
    calculate_brand_scores,
    extract_snippet,
    extract_snippet_char_based,
    extract_snippet_with_sentences,
)


class TestExtractSnippetWithSentences:
    """Tests for sentence-aware snippet extraction."""

    def test_basic_sentence_context(self):
        """Should include sentence before and after the excerpt."""
        chunk_text = (
            "First sentence about background. "
            "Nike announced new sustainability goals. "
            "This will impact their operations."
        )
        excerpt = "Nike announced new sustainability goals"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        assert "First sentence about background" in result
        assert "Nike announced new sustainability goals" in result
        assert "This will impact their operations" in result

    def test_excerpt_at_beginning(self):
        """Should handle excerpt in first sentence (no sentence before)."""
        chunk_text = (
            "Nike announced major changes. "
            "The company will reduce emissions. "
            "Stakeholders are pleased."
        )
        excerpt = "Nike announced major changes"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should not have leading ellipsis since excerpt is in first sentence
        assert not result.startswith("...")
        assert "Nike announced major changes" in result
        assert "The company will reduce emissions" in result

    def test_excerpt_at_end(self):
        """Should handle excerpt in last sentence (no sentence after)."""
        chunk_text = (
            "Background information here. "
            "More context provided. "
            "Nike sets new emission targets."
        )
        excerpt = "Nike sets new emission targets"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should not have trailing ellipsis since excerpt is in last sentence
        assert not result.endswith("...")
        assert "Nike sets new emission targets" in result
        assert "More context provided" in result

    def test_excerpt_in_middle_with_truncation(self):
        """Should add ellipsis when truncating sentences."""
        chunk_text = (
            "Sentence one. "
            "Sentence two. "
            "Nike is the subject. "
            "Sentence four. "
            "Sentence five."
        )
        excerpt = "Nike is the subject"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should have ellipsis at start and end
        assert result.startswith("...")
        assert result.endswith("...")
        assert "Nike is the subject" in result

    def test_case_insensitive_matching(self):
        """Should find excerpt regardless of case."""
        chunk_text = "Background info. NIKE ANNOUNCED new goals. Future plans."
        excerpt = "nike announced"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        assert "NIKE ANNOUNCED" in result
        assert "Background info" in result

    def test_empty_chunk_text(self):
        """Should handle empty chunk text."""
        result = extract_snippet_with_sentences("", "some excerpt")
        assert result == ""

    def test_empty_excerpt(self):
        """Should handle empty excerpt."""
        result = extract_snippet_with_sentences("Some chunk text.", "")
        assert result == "Some chunk text."

    def test_none_chunk_text(self):
        """Should handle None chunk text."""
        result = extract_snippet_with_sentences(None, "excerpt")
        assert result == ""

    def test_none_excerpt(self):
        """Should handle None excerpt."""
        result = extract_snippet_with_sentences("Some text.", None)
        assert result == "Some text."

    def test_excerpt_not_found_falls_back(self):
        """Should fall back to char-based when excerpt not found."""
        chunk_text = "This is about Adidas and sustainability goals."
        excerpt = "Nike environmental targets"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should return something (fallback behavior)
        assert len(result) > 0

    def test_partial_match_with_first_words(self):
        """Should find excerpt using first few words when exact match fails."""
        chunk_text = (
            "Context sentence here. "
            "Nike announced major changes to their policy today. "
            "More details follow."
        )
        excerpt = "Nike announced major changes to their"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        assert "Nike announced" in result
        assert "Context sentence here" in result

    def test_custom_sentence_context(self):
        """Should respect custom sentences_before and sentences_after."""
        chunk_text = (
            "Sentence one. "
            "Sentence two. "
            "Sentence three. "
            "Nike target sentence. "
            "Sentence five. "
            "Sentence six. "
            "Sentence seven."
        )
        excerpt = "Nike target sentence"

        # Request 2 sentences before and after
        result = extract_snippet_with_sentences(
            chunk_text, excerpt, sentences_before=2, sentences_after=2
        )

        assert "Sentence two" in result
        assert "Sentence three" in result
        assert "Nike target sentence" in result
        assert "Sentence five" in result
        assert "Sentence six" in result

    def test_single_sentence_chunk(self):
        """Should handle chunk with only one sentence."""
        chunk_text = "Nike announced sustainability goals."
        excerpt = "Nike announced"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        assert result == "Nike announced sustainability goals."

    def test_excerpt_spanning_sentences(self):
        """Should handle excerpt that might span sentence boundaries."""
        chunk_text = "First part here. Second part there. Third part follows."
        excerpt = "here. Second part"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should return meaningful content
        assert len(result) > 0


class TestExtractSnippetCharBased:
    """Tests for character-based snippet extraction (fallback)."""

    def test_basic_extraction(self):
        """Should extract context around excerpt."""
        chunk_text = "A" * 200 + "Nike sustainability" + "B" * 200
        excerpt = "Nike sustainability"

        result = extract_snippet_char_based(chunk_text, excerpt)

        assert "Nike sustainability" in result
        assert "..." in result  # Should have truncation indicators

    def test_excerpt_at_start(self):
        """Should handle excerpt at the start of text."""
        chunk_text = "Nike sustainability" + "A" * 200
        excerpt = "Nike sustainability"

        result = extract_snippet_char_based(chunk_text, excerpt)

        assert result.startswith("Nike")
        assert "..." in result

    def test_short_text_no_truncation(self):
        """Should not truncate short texts."""
        chunk_text = "Nike announced goals."
        excerpt = "Nike announced"

        result = extract_snippet_char_based(chunk_text, excerpt)

        assert result == "Nike announced goals."
        assert "..." not in result

    def test_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        assert extract_snippet_char_based("", "excerpt") == ""
        assert extract_snippet_char_based("text", "") == "text"


class TestExtractSnippet:
    """Tests for the main extract_snippet function."""

    def test_uses_sentence_extraction(self):
        """Should use sentence-aware extraction by default."""
        chunk_text = (
            "Background context. "
            "Nike announced new goals. "
            "Impact discussion."
        )
        excerpt = "Nike announced new goals"

        result = extract_snippet(chunk_text, excerpt)

        # Should include surrounding sentences
        assert "Background context" in result
        assert "Nike announced new goals" in result
        assert "Impact discussion" in result

    def test_handles_edge_cases(self):
        """Should handle edge cases gracefully."""
        assert extract_snippet("", "excerpt") == ""
        assert extract_snippet("text", "") == "text"
        assert extract_snippet(None, "excerpt") == ""


class TestRealWorldScenarios:
    """Tests with realistic article text."""

    def test_esg_article_excerpt(self):
        """Should provide meaningful context for ESG evidence."""
        chunk_text = (
            "The sportswear industry faces increasing pressure from investors. "
            "Nike has committed to reducing carbon emissions by 30% by 2030. "
            "This target aligns with the Paris Agreement goals. "
            "Other brands are expected to follow suit."
        )
        excerpt = "Nike has committed to reducing carbon emissions by 30% by 2030"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should provide context before and after
        assert "increasing pressure from investors" in result
        assert "Nike has committed" in result
        assert "Paris Agreement" in result

    def test_governance_evidence(self):
        """Should handle governance-related evidence."""
        chunk_text = (
            "Corporate governance remains a key concern for stakeholders. "
            "Adidas announced changes to their board composition. "
            "The new board will include more independent directors. "
            "Transparency in reporting was also addressed."
        )
        excerpt = "Adidas announced changes to their board composition"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        assert "governance remains a key concern" in result
        assert "Adidas announced" in result
        assert "independent directors" in result

    def test_long_article_chunk(self):
        """Should handle longer article chunks efficiently."""
        sentences = [f"Sentence number {i} with some filler content." for i in range(20)]
        sentences[10] = "Nike revealed their environmental strategy for 2030."
        chunk_text = " ".join(sentences)
        excerpt = "Nike revealed their environmental strategy"

        result = extract_snippet_with_sentences(chunk_text, excerpt)

        # Should have context but not the whole chunk
        assert "Nike revealed their environmental strategy" in result
        assert "Sentence number 9" in result
        assert "Sentence number 11" in result
        # First sentences should not be included
        assert "Sentence number 0" not in result


# ============================================================================
# Scorecard Tie Handling Tests
# ============================================================================


def create_mock_article(brand_scores: dict[str, dict]) -> MagicMock:
    """Create a mock Article with brand labels for testing.

    Args:
        brand_scores: Dict mapping brand names to their category sentiments.
            Example: {"Nike": {"environmental": 1, "social": 0}}
            Sentiment values: 1 (positive), 0 (neutral), -1 (negative)

    Returns:
        Mock Article object with brand_labels populated
    """
    article = MagicMock()
    article.published_at = datetime.now()
    article.brand_labels = []

    for brand_name, categories in brand_scores.items():
        brand_label = MagicMock()
        brand_label.brand = brand_name
        brand_label.is_sportswear_brand = True

        # Set category sentiments (None if not in dict)
        brand_label.environmental_sentiment = categories.get("environmental")
        brand_label.social_sentiment = categories.get("social")
        brand_label.governance_sentiment = categories.get("governance")
        brand_label.digital_transformation_sentiment = categories.get("digital_transformation")

        article.brand_labels.append(brand_label)

    return article


class TestScorecardTieHandling:
    """Tests for scorecard tie handling in top and bottom performers."""

    def test_top_performers_no_ties(self):
        """Top performers with distinct scores should show exactly N brands."""
        # Create articles with distinct positive scores
        articles = [
            create_mock_article({"Nike": {"environmental": 1, "social": 1}}),  # +4
            create_mock_article({"Adidas": {"environmental": 1}}),  # +2
            create_mock_article({"Puma": {"social": 0}}),  # +1
            create_mock_article({"Lululemon": {"governance": 0}}),  # +1 (tie for 3rd)
        ]

        # But wait - Puma and Lululemon are tied, so this test should show 4 brands
        # Let's adjust to have distinct scores
        articles = [
            create_mock_article({"Nike": {"environmental": 1, "social": 1}}),  # +4
            create_mock_article({"Adidas": {"environmental": 1, "social": 0}}),  # +3
            create_mock_article({"Puma": {"environmental": 1}}),  # +2
            create_mock_article({"Lululemon": {"social": 0}}),  # +1
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        assert len(result["top_brands"]) == SCORECARD_TOP_N  # Should be exactly 3
        assert result["top_brands"][0]["brand"] == "Nike"
        assert result["top_brands"][0]["total"] == 4
        assert result["top_brands"][1]["brand"] == "Adidas"
        assert result["top_brands"][1]["total"] == 3
        assert result["top_brands"][2]["brand"] == "Puma"
        assert result["top_brands"][2]["total"] == 2

    def test_top_performers_with_tie_at_cutoff(self):
        """Top performers with tie at cutoff should include all tied brands."""
        # Create articles where 3rd and 4th place are tied
        articles = [
            create_mock_article({"Nike": {"environmental": 1, "social": 1}}),  # +4
            create_mock_article({"Adidas": {"environmental": 1, "social": 0}}),  # +3
            create_mock_article({"Puma": {"environmental": 1}}),  # +2
            create_mock_article({"Lululemon": {"social": 1}}),  # +2 (tied with Puma)
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should include all 4 brands since Puma and Lululemon are tied for 3rd
        assert len(result["top_brands"]) == 4
        top_brand_names = {b["brand"] for b in result["top_brands"]}
        assert "Nike" in top_brand_names
        assert "Adidas" in top_brand_names
        assert "Puma" in top_brand_names
        assert "Lululemon" in top_brand_names

    def test_top_performers_medals_with_ties(self):
        """Brands tied for a medal position get the same medal."""
        # Create articles with tie at 3rd place
        articles = [
            create_mock_article({"Nike": {"environmental": 1, "social": 1}}),  # +4
            create_mock_article({"Adidas": {"environmental": 1, "social": 0}}),  # +3
            create_mock_article({"Puma": {"environmental": 1}}),  # +2
            create_mock_article({"Lululemon": {"social": 1}}),  # +2 (tied with Puma)
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # First 3 get medals, 4th gets bronze too since tied with 3rd
        assert result["top_brands"][0]["medal"] == "gold"
        assert result["top_brands"][1]["medal"] == "silver"
        assert result["top_brands"][2]["medal"] == "bronze"
        assert result["top_brands"][3]["medal"] == "bronze"  # Tied for 3rd, gets bronze

    def test_top_performers_medals_tied_for_gold(self):
        """Brands tied for gold both get gold, next tier gets silver."""
        # Two brands tied for first place (+6), one for second (+4)
        # All 3 make the cutoff (SCORECARD_TOP_N=3)
        articles = [
            create_mock_article({"Adidas": {"environmental": 1, "social": 1, "governance": 1}}),  # +6
            create_mock_article({"Anta": {"environmental": 1, "social": 1, "governance": 1}}),  # +6 (tied)
            create_mock_article({"Nike": {"environmental": 1, "social": 1}}),  # +4
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Both +6 brands get gold (tied for 1st)
        adidas = next(b for b in result["top_brands"] if b["brand"] == "Adidas")
        anta = next(b for b in result["top_brands"] if b["brand"] == "Anta")
        assert adidas["medal"] == "gold"
        assert anta["medal"] == "gold"

        # +4 brand gets silver (2nd tier, not bronze!)
        nike = next(b for b in result["top_brands"] if b["brand"] == "Nike")
        assert nike["medal"] == "silver"

    def test_bottom_performers_no_ties(self):
        """Bottom performers with distinct scores should show exactly N brands."""
        # Create articles with distinct negative scores
        # Use multiple articles per brand to get different totals
        articles = [
            create_mock_article({"Nike": {"environmental": -1, "social": -1}}),  # -2
            create_mock_article({"Adidas": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
            create_mock_article({"Puma": {"social": -1}}),  # -1
            # Lululemon: -4 (use two articles: -3 + -1 = -4)
            create_mock_article({"Lululemon": {"governance": -1, "social": -1, "environmental": -1}}),  # -3
            create_mock_article({"Lululemon": {"governance": -1}}),  # -1 more = -4 total
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should show exactly 3 (Lululemon, Adidas, Nike)
        assert len(result["bottom_brands"]) == SCORECARD_BOTTOM_N
        # Worst first
        assert result["bottom_brands"][0]["brand"] == "Lululemon"
        assert result["bottom_brands"][0]["total"] == -4
        assert result["bottom_brands"][1]["brand"] == "Adidas"
        assert result["bottom_brands"][1]["total"] == -3
        assert result["bottom_brands"][2]["brand"] == "Nike"
        assert result["bottom_brands"][2]["total"] == -2

    def test_bottom_performers_with_tie_at_cutoff(self):
        """Bottom performers with tie at cutoff should include all tied brands.

        Example: Nike: -6, Adidas: -4, Puma: -4, Lululemon: -4
        Should show all 4 brands since Adidas, Puma, Lululemon are tied for 2nd-worst.
        """
        articles = [
            create_mock_article({"Nike": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
            create_mock_article({"Nike": {"environmental": -1, "social": -1, "governance": -1}}),  # -3 more = -6 total
            create_mock_article({"Adidas": {"environmental": -1, "social": -1, "governance": -1, "digital_transformation": -1}}),  # -4
            create_mock_article({"Puma": {"environmental": -1, "social": -1, "governance": -1, "digital_transformation": -1}}),  # -4
            create_mock_article({"Lululemon": {"environmental": -1, "social": -1, "governance": -1, "digital_transformation": -1}}),  # -4
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should include all 4 brands since 2nd, 3rd, 4th are tied
        assert len(result["bottom_brands"]) == 4
        bottom_brand_names = {b["brand"] for b in result["bottom_brands"]}
        assert "Nike" in bottom_brand_names
        assert "Adidas" in bottom_brand_names
        assert "Puma" in bottom_brand_names
        assert "Lululemon" in bottom_brand_names

        # Nike should be first (worst)
        assert result["bottom_brands"][0]["brand"] == "Nike"
        assert result["bottom_brands"][0]["total"] == -6

    def test_bottom_performers_excludes_better_scores_when_no_tie(self):
        """Bottom performers should exclude brands with better scores when no tie.

        Example: Nike: -6, Adidas: -4, Puma: -4, Lululemon: -3
        Should show Nike, Adidas, Puma (Lululemon excluded since -3 > -4).
        """
        articles = [
            # Nike: -6 total (two articles of -3 each)
            create_mock_article({"Nike": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
            create_mock_article({"Nike": {"environmental": -1, "social": -1, "governance": -1}}),  # -3 more = -6 total
            # Adidas: -4 total (two articles: -3 + -1)
            create_mock_article({"Adidas": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
            create_mock_article({"Adidas": {"environmental": -1}}),  # -1 more = -4 total
            # Puma: -4 total (two articles: -3 + -1)
            create_mock_article({"Puma": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
            create_mock_article({"Puma": {"environmental": -1}}),  # -1 more = -4 total
            # Lululemon: -3 total
            create_mock_article({"Lululemon": {"environmental": -1, "social": -1, "governance": -1}}),  # -3
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should show exactly 3: Nike (-6), Adidas (-4), Puma (-4)
        # Lululemon (-3) is excluded since it's not tied with the 3rd worst
        assert len(result["bottom_brands"]) == 3
        bottom_brand_names = {b["brand"] for b in result["bottom_brands"]}
        assert "Nike" in bottom_brand_names
        assert "Adidas" in bottom_brand_names
        assert "Puma" in bottom_brand_names
        assert "Lululemon" not in bottom_brand_names

    def test_fewer_than_n_positive_brands(self):
        """Should handle fewer than N brands with positive scores."""
        # Only 2 brands with positive scores
        articles = [
            create_mock_article({"Nike": {"environmental": 1}}),  # +2
            create_mock_article({"Adidas": {"social": 0}}),  # +1
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should show only the 2 positive brands
        assert len(result["top_brands"]) == 2
        assert result["top_brands"][0]["brand"] == "Nike"
        assert result["top_brands"][1]["brand"] == "Adidas"

    def test_fewer_than_n_negative_brands(self):
        """Should handle fewer than N brands with negative scores."""
        # Only 2 brands with negative scores
        articles = [
            create_mock_article({"Nike": {"environmental": -1, "social": -1}}),  # -2
            create_mock_article({"Adidas": {"social": -1}}),  # -1
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # Should show only the 2 negative brands
        assert len(result["bottom_brands"]) == 2
        assert result["bottom_brands"][0]["brand"] == "Nike"  # Worst first
        assert result["bottom_brands"][1]["brand"] == "Adidas"

    def test_alphabetical_tiebreaker(self):
        """Brands with same score should be sorted alphabetically."""
        # All brands have the same positive score
        articles = [
            create_mock_article({"Zebra": {"environmental": 1}}),  # +2
            create_mock_article({"Apple": {"environmental": 1}}),  # +2
            create_mock_article({"Mango": {"environmental": 1}}),  # +2
        ]

        result = calculate_brand_scores(articles, period_days=14, dedupe=False)

        # All 3 should be included (all tied), sorted alphabetically
        assert len(result["top_brands"]) == 3
        assert result["top_brands"][0]["brand"] == "Apple"
        assert result["top_brands"][1]["brand"] == "Mango"
        assert result["top_brands"][2]["brand"] == "Zebra"

    def test_empty_articles_list(self):
        """Should handle empty articles list gracefully."""
        result = calculate_brand_scores([], period_days=14, dedupe=False)

        assert result["top_brands"] == []
        assert result["bottom_brands"] == []
        assert result["all_brand_scores"] == []
