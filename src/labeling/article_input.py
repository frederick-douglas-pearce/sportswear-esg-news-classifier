"""What the labeling LLM is sent for an article's content.

The pipeline and the prompt evaluation (`scripts/compare_prompts.py`) both read this, so an
evaluation sends the same content the pipeline would (#192 PR2).
"""

# Shorter content is skipped as SKIP_INSUFFICIENT_CONTENT before any LLM call
MIN_CONTENT_CHARS = 100


def article_content(full_content: str | None, description: str | None) -> str:
    """The text sent as the article's content: the scraped text, else the description."""
    return full_content or description or ""


def has_sufficient_content(content: str) -> bool:
    """Whether the content is long enough to be sent to the LLM."""
    return len(content.strip()) >= MIN_CONTENT_CHARS
