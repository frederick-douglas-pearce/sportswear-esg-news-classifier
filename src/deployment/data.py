"""Data loading and splitting utilities for FP Classifier deployment."""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_training_data(
    path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load JSONL training data.

    Args:
        path: Path to the JSONL file
        verbose: Whether to print dataset info

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    df = pd.read_json(file_path, lines=True)

    if verbose:
        print(f"Loaded {len(df):,} records from {file_path.name}")
        print(f"Columns: {list(df.columns)}")

    return df


def create_text_features(
    df: pd.DataFrame,
    title_col: str = "title",
    content_col: str = "content",
    brands_col: str = "brands",
    source_name_col: Optional[str] = "source_name",
    category_col: Optional[str] = "category",
    include_metadata: bool = True,
) -> pd.Series:
    """Create combined text features from title, content, brands, and metadata.

    Combines article title, content, brand names, and optionally metadata
    (source name, categories) into a single text feature string for model input.

    When include_metadata=True, prepends a structured prefix like:
    "[Source: wwd.com] [Category: business, sports] Title content brands"

    This allows sentence transformers to learn semantic relationships between
    publishers and content, improving generalization to new publishers.

    Args:
        df: DataFrame containing the article data
        title_col: Name of column containing article title
        content_col: Name of column containing article content
        brands_col: Name of column containing brands list
        source_name_col: Name of column containing publisher name (or None to skip)
        category_col: Name of column containing categories list (or None to skip)
        include_metadata: Whether to include source/category prefix in text

    Returns:
        Series of combined text features
    """

    def format_metadata_prefix(source_name: Optional[str], categories: Optional[List]) -> str:
        """Format metadata as a natural text prefix.

        Uses plain text without brackets to work well with both TF-IDF
        (after punctuation removal) and sentence transformers.
        """
        parts = []
        if source_name:
            # Just include the domain name as natural text
            parts.append(source_name)
        if categories:
            if isinstance(categories, list):
                # Include categories as space-separated words
                parts.extend(str(c) for c in categories)
            else:
                parts.append(str(categories))
        if parts:
            return " ".join(parts) + " "
        return ""

    def combine_text(row) -> str:
        """Combine text fields for a single row."""
        # Metadata prefix (if enabled)
        prefix = ""
        if include_metadata:
            source = row.get(source_name_col) if source_name_col and source_name_col in row.index else None
            cats = row.get(category_col) if category_col and category_col in row.index else None
            prefix = format_metadata_prefix(source, cats)

        title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
        content = str(row.get(content_col, "")) if pd.notna(row.get(content_col)) else ""

        # Handle brands - could be list or string
        brands = row.get(brands_col, [])
        if isinstance(brands, str):
            brands_str = brands
        elif isinstance(brands, list):
            brands_str = " ".join(brands)
        else:
            brands_str = ""

        # Combine with spaces
        parts = [title, content, brands_str]
        text = " ".join(part for part in parts if part)

        return prefix + text

    return df.apply(combine_text, axis=1)


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_sportswear",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into stratified train, validation, and test sets.

    Args:
        df: DataFrame to split
        target_col: Column to stratify by
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        verbose: Whether to print split information

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=df[target_col],
    )

    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_df[target_col],
    )

    if verbose:
        print("=" * 50)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 50)
        print(f"\nTotal samples: {len(df):,}")
        print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
        print(f"\nResulting sizes:")
        print(f"  Train:      {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

        print(f"\nClass distribution (stratified by '{target_col}'):")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df[target_col].value_counts(normalize=True)
            dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in dist.items()])
            print(f"  {name}: {dist_str}")

        print("=" * 50)

    return train_df, val_df, test_df
