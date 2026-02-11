"""Tests for evidence summarizer module."""

from uuid import uuid4

import pytest

from src.labeling.evidence_matcher import EvidenceMatch
from src.labeling.evidence_summarizer import (
    EvidenceSummary,
    EvidenceSummarizer,
    generate_article_summary,
)


@pytest.fixture
def summarizer():
    """Create an EvidenceSummarizer instance."""
    return EvidenceSummarizer()


@pytest.fixture
def sample_evidence_match():
    """Create a sample EvidenceMatch for testing."""
    return EvidenceMatch(
        excerpt="Nike commits to reducing carbon emissions by 50% by 2030",
        chunk_id=uuid4(),
        chunk_index=0,
        similarity_score=0.95,
        match_method="exact",
        confidence="high",
        context_snippet="Nike commits to reducing carbon emissions by 50% by 2030...",
    )


@pytest.fixture
def unmatched_evidence():
    """Create an unmatched EvidenceMatch."""
    return EvidenceMatch(
        excerpt="Some excerpt that wasn't matched",
        chunk_id=None,
        chunk_index=None,
        similarity_score=0.0,
        match_method="none",
        confidence="unmatched",
    )


class TestEvidenceSummary:
    """Tests for EvidenceSummary dataclass."""

    def test_summary_fields(self):
        """Should have all expected fields."""
        summary = EvidenceSummary(
            category="environmental",
            category_name="Environmental",
            brand="Nike",
            excerpt="Test excerpt about carbon",
            detected_topics=["carbon"],
            explanation="Test explanation",
            confidence="high",
        )

        assert summary.category == "environmental"
        assert summary.category_name == "Environmental"
        assert summary.brand == "Nike"
        assert summary.excerpt == "Test excerpt about carbon"
        assert summary.detected_topics == ["carbon"]
        assert summary.explanation == "Test explanation"
        assert summary.confidence == "high"


class TestEvidenceSummarizerInit:
    """Tests for EvidenceSummarizer initialization."""

    def test_loads_esg_keywords(self, summarizer):
        """Should load ESG keywords from config."""
        assert "environmental" in summarizer.esg_keywords
        assert "social" in summarizer.esg_keywords
        assert "governance" in summarizer.esg_keywords
        assert isinstance(summarizer.esg_keywords["environmental"], list)

    def test_loads_esg_names(self, summarizer):
        """Should load ESG category names from config."""
        assert "environmental" in summarizer.esg_names
        assert summarizer.esg_names["environmental"] == "Environmental"

    def test_loads_esg_descriptions(self, summarizer):
        """Should load ESG descriptions from config."""
        assert "environmental" in summarizer.esg_descriptions
        assert len(summarizer.esg_descriptions["environmental"]) > 0


class TestDetectTopics:
    """Tests for _detect_topics method."""

    def test_detects_environmental_keywords(self, summarizer):
        """Should detect environmental keywords in text."""
        text = "Nike aims for carbon neutrality with emissions reduction targets"
        topics = summarizer._detect_topics(text, "environmental")

        assert "carbon" in topics
        assert "emissions" in topics

    def test_detects_social_keywords(self, summarizer):
        """Should detect social keywords in text."""
        text = "Company improves worker safety and promotes diversity"
        topics = summarizer._detect_topics(text, "social")

        assert "diversity" in topics
        assert "safety" in topics

    def test_case_insensitive_detection(self, summarizer):
        """Should detect keywords regardless of case."""
        text = "CARBON emissions and SUSTAINABLE initiatives"
        topics = summarizer._detect_topics(text, "environmental")

        assert "carbon" in topics
        assert "emissions" in topics
        assert "sustainable" in topics

    def test_returns_empty_for_no_matches(self, summarizer):
        """Should return empty list when no keywords found."""
        text = "Generic news article about sports"
        topics = summarizer._detect_topics(text, "environmental")

        assert topics == []

    def test_unknown_category_returns_empty(self, summarizer):
        """Should return empty list for unknown category."""
        text = "Some text with carbon and emissions"
        topics = summarizer._detect_topics(text, "unknown_category")

        assert topics == []


class TestFormatTopics:
    """Tests for _format_topics method."""

    def test_empty_list(self, summarizer):
        """Should return empty string for empty list."""
        result = summarizer._format_topics([])
        assert result == ""

    def test_single_topic(self, summarizer):
        """Should return single topic as-is."""
        result = summarizer._format_topics(["carbon"])
        assert result == "carbon"

    def test_two_topics(self, summarizer):
        """Should join two topics with 'and'."""
        result = summarizer._format_topics(["carbon", "emissions"])
        assert result == "carbon and emissions"

    def test_three_or_more_topics(self, summarizer):
        """Should use Oxford comma for three or more topics."""
        result = summarizer._format_topics(["carbon", "emissions", "sustainable"])
        assert result == "carbon, emissions, and sustainable"

    def test_four_topics(self, summarizer):
        """Should handle four topics correctly."""
        result = summarizer._format_topics(["a", "b", "c", "d"])
        assert result == "a, b, c, and d"


class TestSummarizeEvidence:
    """Tests for summarize_evidence method."""

    def test_generates_summary_with_detected_topics(
        self, summarizer, sample_evidence_match
    ):
        """Should generate summary with detected topics."""
        summary = summarizer.summarize_evidence(
            category="environmental",
            match=sample_evidence_match,
            brand="Nike",
        )

        assert summary.category == "environmental"
        assert summary.category_name == "Environmental"
        assert summary.brand == "Nike"
        assert summary.excerpt == sample_evidence_match.excerpt
        assert "carbon" in summary.detected_topics
        assert summary.confidence == "high"
        assert "carbon" in summary.explanation.lower()
        assert "Nike" in summary.explanation

    def test_generates_generic_explanation_without_topics(self, summarizer):
        """Should generate generic explanation when no keywords detected."""
        match = EvidenceMatch(
            excerpt="The company announced new initiatives",
            chunk_id=uuid4(),
            chunk_index=0,
            similarity_score=0.85,
            match_method="embedding",
            confidence="medium",
        )

        summary = summarizer.summarize_evidence(
            category="environmental",
            match=match,
            brand="Adidas",
        )

        assert summary.detected_topics == []
        assert "semantic similarity" in summary.explanation.lower()
        assert "Adidas" in summary.explanation

    def test_handles_unknown_category(self, summarizer, sample_evidence_match):
        """Should handle unknown category gracefully."""
        summary = summarizer.summarize_evidence(
            category="unknown",
            match=sample_evidence_match,
            brand="Nike",
        )

        assert summary.category == "unknown"
        # Should title-case unknown category
        assert summary.category_name == "Unknown"


class TestSummarizeAllEvidence:
    """Tests for summarize_all_evidence method."""

    def test_summarizes_matched_evidence_only(
        self, summarizer, sample_evidence_match, unmatched_evidence
    ):
        """Should only summarize matched evidence, not unmatched."""
        category_matches = {
            "environmental": [sample_evidence_match, unmatched_evidence],
        }

        summaries = summarizer.summarize_all_evidence("Nike", category_matches)

        # Only matched evidence should be summarized
        assert len(summaries) == 1

    def test_summarizes_multiple_categories(self, summarizer):
        """Should summarize evidence across multiple categories."""
        env_match = EvidenceMatch(
            excerpt="Carbon reduction goals",
            chunk_id=uuid4(),
            chunk_index=0,
            similarity_score=0.9,
            match_method="exact",
            confidence="high",
        )
        social_match = EvidenceMatch(
            excerpt="Worker safety improvements",
            chunk_id=uuid4(),
            chunk_index=1,
            similarity_score=0.85,
            match_method="fuzzy",
            confidence="medium",
        )

        category_matches = {
            "environmental": [env_match],
            "social": [social_match],
        }

        summaries = summarizer.summarize_all_evidence("Nike", category_matches)

        assert len(summaries) == 2
        categories = [s.category for s in summaries]
        assert "environmental" in categories
        assert "social" in categories

    def test_empty_category_matches(self, summarizer):
        """Should return empty list for no matches."""
        summaries = summarizer.summarize_all_evidence("Nike", {})
        assert summaries == []

    def test_all_unmatched_evidence(self, summarizer, unmatched_evidence):
        """Should return empty list when all evidence is unmatched."""
        category_matches = {
            "environmental": [unmatched_evidence],
            "social": [unmatched_evidence],
        }

        summaries = summarizer.summarize_all_evidence("Nike", category_matches)
        assert summaries == []


class TestGenerateArticleSummary:
    """Tests for generate_article_summary function."""

    def test_generates_summaries_for_multiple_brands(self):
        """Should generate summaries for all brands with evidence."""
        nike_match = EvidenceMatch(
            excerpt="Nike carbon goals",
            chunk_id=uuid4(),
            chunk_index=0,
            similarity_score=0.9,
            match_method="exact",
            confidence="high",
        )
        adidas_match = EvidenceMatch(
            excerpt="Adidas sustainable materials",
            chunk_id=uuid4(),
            chunk_index=1,
            similarity_score=0.85,
            match_method="exact",
            confidence="high",
        )

        brand_evidence = {
            "Nike": {"environmental": [nike_match]},
            "Adidas": {"environmental": [adidas_match]},
        }

        result = generate_article_summary(brand_evidence)

        assert "Nike" in result
        assert "Adidas" in result
        assert len(result["Nike"]) == 1
        assert len(result["Adidas"]) == 1

    def test_excludes_brands_with_no_matched_evidence(self):
        """Should exclude brands that have no matched evidence."""
        unmatched = EvidenceMatch(
            excerpt="Some text",
            chunk_id=None,
            chunk_index=None,
            similarity_score=0.0,
            match_method="none",
            confidence="unmatched",
        )

        brand_evidence = {
            "Nike": {"environmental": [unmatched]},
        }

        result = generate_article_summary(brand_evidence)

        assert "Nike" not in result or len(result.get("Nike", [])) == 0

    def test_empty_brand_evidence(self):
        """Should return empty dict for empty input."""
        result = generate_article_summary({})
        assert result == {}

    def test_brands_with_empty_categories(self):
        """Should handle brands with empty category lists."""
        brand_evidence = {
            "Nike": {"environmental": [], "social": []},
        }

        result = generate_article_summary(brand_evidence)

        # Nike should not be in result since no matched evidence
        assert result == {} or "Nike" not in result


class TestEvidenceSummaryIntegration:
    """Integration tests for evidence summarization workflow."""

    def test_full_workflow(self):
        """Test complete evidence summarization workflow."""
        # Create realistic evidence
        env_evidence = EvidenceMatch(
            excerpt="Nike has committed to achieving 100% renewable energy in all facilities by 2025",
            chunk_id=uuid4(),
            chunk_index=0,
            similarity_score=0.98,
            match_method="exact",
            confidence="high",
        )

        social_evidence = EvidenceMatch(
            excerpt="The company enhanced worker safety protections across supply chain",
            chunk_id=uuid4(),
            chunk_index=1,
            similarity_score=0.89,
            match_method="fuzzy",
            confidence="medium",
        )

        brand_evidence = {
            "Nike": {
                "environmental": [env_evidence],
                "social": [social_evidence],
            },
        }

        result = generate_article_summary(brand_evidence)

        assert "Nike" in result
        summaries = result["Nike"]
        assert len(summaries) == 2

        # Check environmental summary
        env_summary = next(s for s in summaries if s.category == "environmental")
        assert "renewable" in env_summary.detected_topics
        assert env_summary.confidence == "high"
        assert "Nike" in env_summary.explanation

        # Check social summary
        social_summary = next(s for s in summaries if s.category == "social")
        assert social_summary.confidence == "medium"
        assert "Nike" in social_summary.explanation
