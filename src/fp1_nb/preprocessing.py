"""Text preprocessing utilities for FP classifier notebook."""

import re
from typing import Callable, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    remove_punctuation: bool = True,
    remove_extra_whitespace: bool = True,
    min_word_length: int = 2
) -> str:
    """Clean and normalize text.

    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_urls: Remove URLs
        remove_emails: Remove email addresses
        remove_numbers: Remove standalone numbers
        remove_punctuation: Remove punctuation
        remove_extra_whitespace: Collapse multiple spaces
        min_word_length: Remove words shorter than this

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove emails
    if remove_emails:
        text = re.sub(r'\S+@\S+', ' ', text)

    # Remove punctuation (keep letters, numbers, spaces)
    if remove_punctuation:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Remove standalone numbers
    if remove_numbers:
        text = re.sub(r'\b\d+\b', ' ', text)

    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    # Remove short words
    if min_word_length > 1:
        words = text.split()
        words = [w for w in words if len(w) >= min_word_length]
        text = ' '.join(words)

    return text


def create_text_features(
    df: pd.DataFrame,
    text_col: str,
    title_col: Optional[str] = None,
    brands_col: Optional[str] = None,
    clean_func: Optional[Callable] = None
) -> pd.Series:
    """Create combined text features for classification.

    Args:
        df: DataFrame containing the data
        text_col: Name of the main text column
        title_col: Optional title column to prepend
        brands_col: Optional brands column to include
        clean_func: Optional text cleaning function

    Returns:
        Series with combined text features
    """
    if clean_func is None:
        clean_func = clean_text

    texts = []
    for idx, row in df.iterrows():
        parts = []

        # Add title if provided (with extra weight via repetition)
        if title_col and title_col in row and pd.notna(row[title_col]):
            title_text = clean_func(str(row[title_col]))
            parts.append(title_text)
            parts.append(title_text)  # Repeat for emphasis

        # Add brands if provided
        if brands_col and brands_col in row:
            brands = row[brands_col]
            if isinstance(brands, list):
                brand_text = ' '.join(brands)
            else:
                brand_text = str(brands) if pd.notna(brands) else ''
            parts.append(clean_func(brand_text))

        # Add main content
        content = row[text_col] if pd.notna(row[text_col]) else ''
        parts.append(clean_func(str(content)))

        texts.append(' '.join(parts))

    return pd.Series(texts, index=df.index)


def build_tfidf_pipeline(
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    classifier: Optional[object] = None
) -> Pipeline:
    """Build a TF-IDF based text classification pipeline.

    Args:
        max_features: Maximum number of features
        ngram_range: N-gram range (min_n, max_n)
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        sublinear_tf: Apply sublinear TF scaling
        classifier: Optional classifier to append to pipeline

    Returns:
        Sklearn Pipeline object
    """
    steps = [
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Words with 2+ letters
        ))
    ]

    if classifier is not None:
        steps.append(('classifier', classifier))

    return Pipeline(steps)


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract feature names from a fitted pipeline.

    Args:
        pipeline: Fitted sklearn pipeline with TF-IDF vectorizer

    Returns:
        List of feature names
    """
    tfidf = pipeline.named_steps.get('tfidf')
    if tfidf is None:
        raise ValueError("Pipeline does not contain a 'tfidf' step")

    return tfidf.get_feature_names_out().tolist()


def get_top_tfidf_features(
    pipeline: Pipeline,
    text: str,
    top_n: int = 10
) -> List[tuple]:
    """Get top TF-IDF features for a given text.

    Args:
        pipeline: Fitted sklearn pipeline with TF-IDF vectorizer
        text: Input text
        top_n: Number of top features to return

    Returns:
        List of (feature_name, score) tuples
    """
    tfidf = pipeline.named_steps.get('tfidf')
    if tfidf is None:
        raise ValueError("Pipeline does not contain a 'tfidf' step")

    # Transform text
    vector = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()

    # Get non-zero features
    indices = vector.nonzero()[1]
    scores = vector.toarray()[0]

    # Sort by score
    feature_scores = [(feature_names[i], scores[i]) for i in indices]
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    return feature_scores[:top_n]
