"""Text preprocessing for EP (ESG Pre-filter) Classifier."""

import re
from typing import List, Optional


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Clean and normalize text for EP classifier.

    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        remove_punctuation: Remove punctuation

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    if lowercase:
        text = text.lower()

    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    if remove_emails:
        text = re.sub(r"\S+@\S+", " ", text)

    if remove_punctuation:
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_input(
    title: str,
    content: str,
    brands: Optional[List[str]] = None,
    source_name: Optional[str] = None,
    category: Optional[List[str]] = None,
    include_metadata: bool = True,
) -> str:
    """Prepare raw API input for the EP classifier pipeline.

    Combines title, content, brands, and optional metadata into a single
    text string that matches the format expected by the trained model.

    The EP classifier uses a slightly different format than FP:
    - Title is repeated for emphasis
    - Metadata is included as natural text (no brackets)
    - Brand names are appended

    Args:
        title: Article title
        content: Article content/body
        brands: List of brand names mentioned in the article
        source_name: News source name (e.g., "wwd.com", "espn.com")
        category: List of article categories (e.g., ["business", "environment"])
        include_metadata: Whether to prepend metadata to text (default True)

    Returns:
        Combined text string ready for classification
    """
    parts = []

    # Add metadata prefix if enabled (as natural text, no brackets)
    if include_metadata:
        metadata_parts = []
        if source_name:
            metadata_parts.append(str(source_name))
        if category:
            if isinstance(category, list):
                metadata_parts.extend(str(c) for c in category if c)
            else:
                metadata_parts.append(str(category))
        if metadata_parts:
            parts.append(" ".join(metadata_parts))

    # Add title (repeated for emphasis, matching training)
    if title:
        title_clean = clean_text(str(title))
        parts.append(title_clean)
        parts.append(title_clean)  # Repeat for emphasis

    # Add brands
    if brands:
        if isinstance(brands, str):
            brands_str = brands
        else:
            brands_str = " ".join(str(b) for b in brands)
        parts.append(clean_text(brands_str))

    # Add main content
    if content:
        parts.append(clean_text(str(content)))

    return " ".join(part for part in parts if part)
