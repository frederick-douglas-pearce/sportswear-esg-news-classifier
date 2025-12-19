"""Text preprocessing utilities for FP classifier notebook."""

import re
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# Sportswear vocabulary for domain-specific features
SPORTSWEAR_VOCAB = {
    'footwear': {
        'shoes', 'sneakers', 'boots', 'sandals', 'cleats', 'trainers',
        'running', 'basketball', 'tennis', 'soccer', 'football', 'golf',
        'hiking', 'skateboarding', 'sneaker', 'footwear', 'shoe', 'boot',
    },
    'apparel': {
        'jacket', 'pants', 'shirt', 'hoodie', 'shorts', 'jersey', 'leggings',
        'sweatshirt', 'joggers', 'tights', 'vest', 'coat', 'apparel',
        'clothing', 'wear', 'outfit', 'gear', 'uniform', 'sportswear',
        'activewear', 'athleisure', 'athletic',
    },
    'sports_activity': {
        'running', 'training', 'fitness', 'workout', 'exercise', 'gym',
        'sports', 'athletic', 'marathon', 'yoga', 'crossfit', 'cycling',
        'swimming', 'basketball', 'football', 'soccer', 'tennis', 'golf',
    },
    'retail_business': {
        'store', 'retail', 'shop', 'brand', 'collection', 'launch',
        'release', 'collaboration', 'collab', 'partnership', 'sponsor',
        'endorsement', 'campaign', 'advertisement', 'commercial', 'sale',
        'discount', 'price', 'buy', 'purchase', 'ecommerce',
    },
}


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


def extract_brand_context(
    text: str,
    brands: List[str],
    window_size: int = 10,
    lowercase: bool = True
) -> str:
    """Extract words within a context window around brand mentions.

    Args:
        text: Full article text
        brands: List of brand names to find
        window_size: Number of words before/after brand to extract
        lowercase: Whether to lowercase the text

    Returns:
        String of context words around all brand mentions
    """
    if not text or not brands:
        return ""

    if lowercase:
        text_lower = text.lower()
    else:
        text_lower = text

    # Tokenize into words (preserving positions)
    words = re.findall(r'\b\w+\b', text_lower)

    context_words = []

    for brand in brands:
        brand_lower = brand.lower() if lowercase else brand
        brand_words = brand_lower.split()

        # Find all occurrences of the brand
        for i, word in enumerate(words):
            # Check for single-word brand match
            if len(brand_words) == 1 and word == brand_lower:
                # Extract context window
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context = words[start:i] + words[i+1:end]
                context_words.extend(context)

            # Check for multi-word brand match (e.g., "Under Armour")
            elif len(brand_words) > 1:
                if i + len(brand_words) <= len(words):
                    if words[i:i+len(brand_words)] == brand_words:
                        start = max(0, i - window_size)
                        end = min(len(words), i + len(brand_words) + window_size)
                        context = words[start:i] + words[i+len(brand_words):end]
                        context_words.extend(context)

    return ' '.join(context_words)


def compute_sportswear_vocab_features(
    text: str,
    brands: List[str],
    window_size: int = 15,
    vocab: Dict[str, Set[str]] = None
) -> Dict[str, float]:
    """Compute sportswear vocabulary features for a text.

    Args:
        text: Article text
        brands: List of brand names
        window_size: Context window for brand-adjacent features
        vocab: Vocabulary dict (defaults to SPORTSWEAR_VOCAB)

    Returns:
        Dictionary of feature names to values
    """
    if vocab is None:
        vocab = SPORTSWEAR_VOCAB

    if not text:
        return {f'vocab_{cat}': 0.0 for cat in vocab.keys()}

    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    # Get context around brands
    brand_context = extract_brand_context(text, brands, window_size=window_size)
    context_words = set(brand_context.split()) if brand_context else set()

    features = {}

    for category, category_words in vocab.items():
        # Count matches in full text
        full_matches = len(words & category_words)
        features[f'vocab_{category}_count'] = full_matches

        # Binary: any match in full text
        features[f'vocab_{category}_any'] = 1.0 if full_matches > 0 else 0.0

        # Count matches near brand mentions
        context_matches = len(context_words & category_words)
        features[f'vocab_{category}_near_brand'] = context_matches

    # Combined score: total sportswear vocab matches
    all_vocab = set()
    for cat_words in vocab.values():
        all_vocab.update(cat_words)

    features['vocab_total_matches'] = len(words & all_vocab)
    features['vocab_near_brand_total'] = len(context_words & all_vocab)

    return features


def create_sportswear_vocab_df(
    df: pd.DataFrame,
    text_col: str,
    brands_col: str,
    window_size: int = 15
) -> pd.DataFrame:
    """Create sportswear vocabulary features for a DataFrame.

    Args:
        df: DataFrame with text and brands columns
        text_col: Name of text column
        brands_col: Name of brands column

    Returns:
        DataFrame with vocabulary features
    """
    features_list = []

    for idx, row in df.iterrows():
        text = row[text_col] if pd.notna(row[text_col]) else ""
        brands = row[brands_col] if isinstance(row[brands_col], list) else []

        features = compute_sportswear_vocab_features(text, brands, window_size)
        features_list.append(features)

    return pd.DataFrame(features_list, index=df.index)


def create_enhanced_text_features(
    df: pd.DataFrame,
    text_col: str,
    title_col: Optional[str] = None,
    brands_col: Optional[str] = None,
    include_context: bool = True,
    context_window: int = 10,
    clean_func: Optional[Callable] = None
) -> pd.Series:
    """Create enhanced text features with brand context emphasis.

    This extends create_text_features by:
    1. Adding brand context window (words near brand mentions)
    2. Repeating context words for emphasis

    Args:
        df: DataFrame containing the data
        text_col: Name of the main text column
        title_col: Optional title column to prepend
        brands_col: Optional brands column
        include_context: Whether to add brand context window
        context_window: Size of context window around brands
        clean_func: Optional text cleaning function

    Returns:
        Series with enhanced text features
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
        brands = []
        if brands_col and brands_col in row:
            brands_val = row[brands_col]
            if isinstance(brands_val, list):
                brands = brands_val
                brand_text = ' '.join(brands_val)
            else:
                brand_text = str(brands_val) if pd.notna(brands_val) else ''
                brands = [brand_text] if brand_text else []
            parts.append(clean_func(brand_text))

        # Add main content
        content = row[text_col] if pd.notna(row[text_col]) else ''
        content_str = str(content)
        parts.append(clean_func(content_str))

        # Add brand context (words near brand mentions) - repeated for emphasis
        if include_context and brands and content_str:
            context = extract_brand_context(content_str, brands, window_size=context_window)
            if context:
                cleaned_context = clean_func(context)
                parts.append(cleaned_context)  # Add context once more for emphasis

        texts.append(' '.join(parts))

    return pd.Series(texts, index=df.index)
