"""Tests for the website feed export script."""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from export_website_feed import (
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
