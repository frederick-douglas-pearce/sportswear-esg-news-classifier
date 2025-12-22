"""Text preprocessing for FP Classifier API."""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text for prediction.

    Applies standard text cleaning:
    - Converts to lowercase
    - Removes URLs
    - Removes email addresses
    - Removes excessive whitespace
    - Removes special characters (keeps alphanumeric and basic punctuation)

    Args:
        text: Raw input text

    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r"[^\w\s.,!?;:\-\'\"()]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_input(
    title: str,
    content: str,
    brands: Optional[List[str]] = None,
) -> str:
    """Prepare raw API input for the classifier pipeline.

    Combines title, content, and brands into a single text string
    that matches the format expected by the trained model.

    Args:
        title: Article title
        content: Article content/body
        brands: List of brand names mentioned in the article

    Returns:
        Combined text string ready for classification

    Example:
        >>> text = prepare_input(
        ...     title="Nike releases new running shoe",
        ...     content="The athletic giant unveiled...",
        ...     brands=["Nike"]
        ... )
    """
    # Handle None/empty values
    title = str(title) if title else ""
    content = str(content) if content else ""

    # Handle brands list
    if brands:
        if isinstance(brands, str):
            brands_str = brands
        else:
            brands_str = " ".join(str(b) for b in brands)
    else:
        brands_str = ""

    # Combine fields with spaces
    parts = [title, content, brands_str]
    combined = " ".join(part for part in parts if part)

    return combined


def truncate_text(text: str, max_length: int = 10000) -> str:
    """Truncate text to maximum length.

    The sentence transformer model has input limits, so very long
    articles need to be truncated to avoid errors.

    Args:
        text: Input text
        max_length: Maximum character length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]

    return truncated
