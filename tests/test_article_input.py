"""Tests for what the labeling LLM is sent as an article's content (#192 PR2)."""

from src.labeling.article_input import MIN_CONTENT_CHARS, article_content, has_sufficient_content


def test_scraped_text_first_then_description():
    assert article_content("full", "desc") == "full"
    assert article_content(None, "desc") == "desc"
    assert article_content("", "desc") == "desc"
    assert article_content(None, None) == ""


def test_sufficiency_counts_stripped_characters():
    assert MIN_CONTENT_CHARS == 100
    assert has_sufficient_content("x" * 100)
    assert not has_sufficient_content("x" * 99)
    assert not has_sufficient_content("  " + "x" * 99 + "\n\n")
    assert not has_sufficient_content("")
