"""EDA utilities for FP classifier notebook."""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided."""
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def analyze_text_length_stats(
    df: pd.DataFrame,
    text_col: str,
    target_col: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Analyze text length statistics.

    Args:
        df: DataFrame containing the data
        text_col: Name of the text column
        target_col: Optional target column for grouped analysis
        verbose: Whether to print statistics

    Returns:
        DataFrame with length statistics
    """
    # Calculate lengths
    df = df.copy()
    df['_char_length'] = df[text_col].fillna('').str.len()
    df['_word_count'] = df[text_col].fillna('').str.split().str.len()

    if target_col:
        stats = df.groupby(target_col).agg({
            '_char_length': ['mean', 'median', 'std', 'min', 'max'],
            '_word_count': ['mean', 'median', 'std', 'min', 'max']
        }).round(1)
    else:
        stats = df.agg({
            '_char_length': ['mean', 'median', 'std', 'min', 'max'],
            '_word_count': ['mean', 'median', 'std', 'min', 'max']
        }).round(1)

    if verbose:
        print("=" * 50)
        print("TEXT LENGTH STATISTICS")
        print("=" * 50)
        print(f"\nColumn: '{text_col}'")
        print(f"Total records: {len(df):,}")
        print(f"\nOverall statistics:")
        print(f"  Character length: mean={df['_char_length'].mean():.0f}, "
              f"median={df['_char_length'].median():.0f}, "
              f"range=[{df['_char_length'].min()}, {df['_char_length'].max()}]")
        print(f"  Word count: mean={df['_word_count'].mean():.0f}, "
              f"median={df['_word_count'].median():.0f}, "
              f"range=[{df['_word_count'].min()}, {df['_word_count'].max()}]")

        if target_col:
            print(f"\nBy {target_col}:")
            for cls in df[target_col].unique():
                subset = df[df[target_col] == cls]
                print(f"  Class {cls}:")
                print(f"    Chars: mean={subset['_char_length'].mean():.0f}, "
                      f"median={subset['_char_length'].median():.0f}")
                print(f"    Words: mean={subset['_word_count'].mean():.0f}, "
                      f"median={subset['_word_count'].median():.0f}")

        print("=" * 50)

    return stats


def plot_text_length_distributions(
    df: pd.DataFrame,
    text_col: str,
    target_col: Optional[str] = None,
    label_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    bins: int = 50,
    save_path: Optional[str] = None
) -> None:
    """Plot text length distributions.

    Args:
        df: DataFrame containing the data
        text_col: Name of the text column
        target_col: Optional target column for grouped distributions
        label_names: Optional mapping of target values to labels
        figsize: Figure size tuple
        bins: Number of histogram bins
        save_path: Optional path to save the figure
    """
    df = df.copy()
    df['_char_length'] = df[text_col].fillna('').str.len()
    df['_word_count'] = df[text_col].fillna('').str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if target_col:
        colors = ['#e74c3c', '#2ecc71']
        for i, cls in enumerate(sorted(df[target_col].unique())):
            subset = df[df[target_col] == cls]
            label = label_names.get(cls, f'Class {cls}') if label_names else f'Class {cls}'

            axes[0].hist(subset['_char_length'], bins=bins, alpha=0.6,
                        label=label, color=colors[i % len(colors)], edgecolor='black')
            axes[1].hist(subset['_word_count'], bins=bins, alpha=0.6,
                        label=label, color=colors[i % len(colors)], edgecolor='black')

        axes[0].legend()
        axes[1].legend()
    else:
        axes[0].hist(df['_char_length'], bins=bins, color='steelblue', edgecolor='black')
        axes[1].hist(df['_word_count'], bins=bins, color='steelblue', edgecolor='black')

    axes[0].set_title('Character Length Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Character Length')
    axes[0].set_ylabel('Frequency')

    axes[1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def analyze_brand_distribution(
    df: pd.DataFrame,
    brands_col: str,
    target_col: Optional[str] = None,
    top_n: int = 15,
    verbose: bool = True
) -> pd.DataFrame:
    """Analyze brand distribution in the dataset.

    Args:
        df: DataFrame containing the data
        brands_col: Name of the brands column (should contain lists)
        target_col: Optional target column for grouped analysis
        top_n: Number of top brands to display
        verbose: Whether to print statistics

    Returns:
        DataFrame with brand counts
    """
    # Flatten brand lists
    all_brands = []
    for brands in df[brands_col]:
        if isinstance(brands, list):
            all_brands.extend(brands)
        elif isinstance(brands, str):
            all_brands.append(brands)

    brand_counts = pd.Series(Counter(all_brands)).sort_values(ascending=False)

    if verbose:
        print("=" * 50)
        print("BRAND DISTRIBUTION")
        print("=" * 50)
        print(f"\nTotal brand mentions: {len(all_brands):,}")
        print(f"Unique brands: {len(brand_counts):,}")
        print(f"\nTop {top_n} brands:")
        for brand, count in brand_counts.head(top_n).items():
            pct = count / len(all_brands) * 100
            print(f"  {brand}: {count:,} ({pct:.1f}%)")

        if target_col:
            print(f"\nBrand distribution by {target_col}:")
            for cls in sorted(df[target_col].unique()):
                subset = df[df[target_col] == cls]
                cls_brands = []
                for brands in subset[brands_col]:
                    if isinstance(brands, list):
                        cls_brands.extend(brands)
                    elif isinstance(brands, str):
                        cls_brands.append(brands)
                cls_counts = Counter(cls_brands)
                top_3 = cls_counts.most_common(3)
                top_str = ', '.join([f"{b} ({c})" for b, c in top_3])
                print(f"  Class {cls}: {len(cls_brands)} mentions, top: {top_str}")

        print("=" * 50)

    return brand_counts.to_frame(name='count')


def plot_brand_distribution(
    df: pd.DataFrame,
    brands_col: str,
    target_col: Optional[str] = None,
    label_names: Optional[Dict[int, str]] = None,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """Plot brand distribution.

    Args:
        df: DataFrame containing the data
        brands_col: Name of the brands column
        target_col: Optional target column for grouped distributions
        label_names: Optional mapping of target values to labels
        top_n: Number of top brands to display
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    if target_col:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colors = ['#e74c3c', '#2ecc71']

        for i, cls in enumerate(sorted(df[target_col].unique())):
            subset = df[df[target_col] == cls]
            cls_brands = []
            for brands in subset[brands_col]:
                if isinstance(brands, list):
                    cls_brands.extend(brands)
                elif isinstance(brands, str):
                    cls_brands.append(brands)

            counts = pd.Series(Counter(cls_brands)).sort_values(ascending=True).tail(top_n)
            label = label_names.get(cls, f'Class {cls}') if label_names else f'Class {cls}'

            counts.plot(kind='barh', ax=axes[i], color=colors[i], edgecolor='black')
            axes[i].set_title(f'Top {top_n} Brands - {label}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Count')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        all_brands = []
        for brands in df[brands_col]:
            if isinstance(brands, list):
                all_brands.extend(brands)
            elif isinstance(brands, str):
                all_brands.append(brands)

        counts = pd.Series(Counter(all_brands)).sort_values(ascending=True).tail(top_n)
        counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Top {top_n} Brands', fontsize=12, fontweight='bold')
        ax.set_xlabel('Count')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def analyze_word_frequencies(
    df: pd.DataFrame,
    text_col: str,
    target_col: Optional[str] = None,
    top_n: int = 20,
    min_word_length: int = 3,
    stop_words: Optional[set] = None
) -> Dict[str, Counter]:
    """Analyze word frequencies in text data.

    Args:
        df: DataFrame containing the data
        text_col: Name of the text column
        target_col: Optional target column for grouped analysis
        top_n: Number of top words to return
        min_word_length: Minimum word length to include
        stop_words: Optional set of words to exclude

    Returns:
        Dictionary mapping class labels to word Counter objects
    """
    if stop_words is None:
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
            'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her', 'i', 'me',
            'my', 'who', 'which', 'what', 'where', 'when', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very',
            'just', 'also', 'now', 'here', 'there', 'then', 'if', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'once', 'while', 'said',
            'says', 'new', 'one', 'two', 'first', 'last', 'many', 'much',
        }

    def count_words(text_series):
        words = []
        for text in text_series.fillna(''):
            # Simple tokenization
            tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            words.extend([w for w in tokens
                         if len(w) >= min_word_length and w not in stop_words])
        return Counter(words)

    results = {}

    if target_col:
        for cls in sorted(df[target_col].unique()):
            subset = df[df[target_col] == cls]
            results[f'class_{cls}'] = count_words(subset[text_col])
    else:
        results['all'] = count_words(df[text_col])

    # Print top words
    print("=" * 50)
    print("WORD FREQUENCY ANALYSIS")
    print("=" * 50)

    for label, counter in results.items():
        print(f"\nTop {top_n} words for {label}:")
        for word, count in counter.most_common(top_n):
            print(f"  {word}: {count:,}")

    print("=" * 50)

    return results


def plot_word_cloud(
    word_freq: Counter,
    title: str = "Word Cloud",
    figsize: Tuple[int, int] = (12, 6),
    max_words: int = 100,
    save_path: Optional[str] = None
) -> None:
    """Plot word cloud from word frequencies.

    Args:
        word_freq: Counter object with word frequencies
        title: Plot title
        figsize: Figure size tuple
        max_words: Maximum number of words to display
        save_path: Optional path to save the figure
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud package not installed. Install with: pip install wordcloud")
        return

    fig, ax = plt.subplots(figsize=figsize)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap='viridis'
    ).generate_from_frequencies(word_freq)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()
