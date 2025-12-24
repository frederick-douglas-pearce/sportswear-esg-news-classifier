"""Data loading and splitting utilities for FP classifier notebook."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def load_jsonl_data(
    file_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Load JSONL file into a pandas DataFrame.

    Args:
        file_path: Path to the JSONL file
        verbose: Whether to print dataset info

    Returns:
        DataFrame with loaded data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_json(file_path, lines=True)

    if verbose:
        print(f"Loaded {len(df):,} records from {file_path.name}")
        print(f"Columns: {list(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """Visualize target variable distribution with count and percentage plots.

    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        label_names: Optional list of label names for display
        figsize: Figure size tuple
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count plot
    counts = df[target_col].value_counts().sort_index()
    if label_names:
        counts.index = label_names

    colors = ['#e74c3c', '#2ecc71'] if len(counts) == 2 else None
    counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].tick_params(axis='x', rotation=0)

    # Add count labels
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts) * 0.02, f'{v:,}', ha='center', fontsize=10)

    # Percentage plot
    percentages = (counts / len(df) * 100)
    percentages.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
    axes[1].set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class', fontsize=10)
    axes[1].set_ylabel('Percentage (%)', fontsize=10)
    axes[1].tick_params(axis='x', rotation=0)

    # Add percentage labels
    for i, v in enumerate(percentages.values):
        axes[1].text(i, v + max(percentages) * 0.02, f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def analyze_target_stats(
    df: pd.DataFrame,
    target_col: str,
    label_names: Optional[List[str]] = None,
    imbalance_threshold: float = 10.0,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> dict:
    """Analyze target variable distribution and check for class imbalance.

    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        label_names: Optional list of label names for display
        imbalance_threshold: Ratio threshold to flag imbalance
        plot: Whether to display distribution plots
        figsize: Figure size tuple
        save_path: Optional path to save the figure

    Returns:
        Dictionary with distribution metrics and imbalance analysis
    """
    counts = df[target_col].value_counts().sort_index()
    percentages = counts / len(df) * 100

    # Calculate imbalance ratio
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    is_imbalanced = imbalance_ratio >= imbalance_threshold

    # Build results dictionary
    results = {
        'total_samples': len(df),
        'class_counts': counts.to_dict(),
        'class_percentages': percentages.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': is_imbalanced,
        'majority_class': counts.idxmax(),
        'minority_class': counts.idxmin(),
    }

    # Print summary
    print("=" * 50)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 50)
    print(f"\nTotal samples: {len(df):,}")
    print(f"\nClass distribution:")

    for i, (cls, count) in enumerate(counts.items()):
        label = label_names[i] if label_names else cls
        pct = percentages[cls]
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

    if is_imbalanced:
        print(f"\n[WARNING] Dataset is imbalanced (ratio >= {imbalance_threshold})")
        print("Consider using:")
        print("  - Stratified sampling for train/val/test splits")
        print("  - Class weights or oversampling/undersampling")
        print("  - Appropriate metrics (PR-AUC, F1) over accuracy")
    else:
        print(f"\n[OK] Dataset is reasonably balanced")

    print("=" * 50)

    if plot:
        plot_target_distribution(
            df, target_col, label_names, figsize, save_path
        )

    return results


def split_train_val_test(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train, validation, and test sets with stratification.

    Args:
        df: DataFrame to split
        target_col: Column to stratify by (if None, no stratification)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        verbose: Whether to print split information

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    # First split: separate test set
    stratify = df[target_col] if target_col else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify
    )

    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify = train_val_df[target_col] if target_col else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify
    )

    if verbose:
        print("=" * 50)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 50)
        print(f"\nTotal samples: {len(df):,}")
        print(f"\nSplit ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
        print(f"\nResulting sizes:")
        print(f"  Train:      {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

        if target_col:
            print(f"\nClass distribution (stratified by '{target_col}'):")
            for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                dist = split_df[target_col].value_counts(normalize=True)
                dist_str = ', '.join([f"{k}: {v:.1%}" for k, v in dist.items()])
                print(f"  {name}: {dist_str}")

        print("=" * 50)

    return train_df, val_df, test_df
