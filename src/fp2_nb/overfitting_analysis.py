"""Overfitting analysis utilities for model selection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def plot_train_val_gap(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    metric: str = 'f2',
    title: Optional[str] = None,
    gap_threshold: float = 0.05,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot train vs validation score gap for all hyperparameter combinations.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV object
        metric: Metric name to analyze (must match scoring keys)
        title: Plot title (auto-generated if None)
        gap_threshold: Threshold above which to flag overfitting
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Tuple of (figure, gap_dataframe)
    """
    cv_results = search_object.cv_results_

    # Extract train and validation scores
    train_key = f'mean_train_{metric}'
    val_key = f'mean_test_{metric}'

    if train_key not in cv_results:
        raise ValueError(
            f"Train scores not found for metric '{metric}'. "
            "Ensure return_train_score=True in search object."
        )

    train_scores = cv_results[train_key]
    val_scores = cv_results[val_key]
    gaps = train_scores - val_scores

    # Create gap dataframe
    gap_df = pd.DataFrame({
        'param_index': range(len(gaps)),
        'train_score': train_scores,
        'val_score': val_scores,
        'gap': gaps,
        'overfitting': gaps > gap_threshold,
    })

    # Sort by gap for better visualization
    gap_df_sorted = gap_df.sort_values('gap', ascending=False).reset_index(drop=True)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Gap by parameter combination
    ax1 = axes[0]
    colors = ['red' if g > gap_threshold else 'green' for g in gap_df_sorted['gap']]
    bars = ax1.bar(range(len(gap_df_sorted)), gap_df_sorted['gap'], color=colors, alpha=0.7)
    ax1.axhline(y=gap_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Overfitting threshold ({gap_threshold})')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Parameter Combination (sorted by gap)', fontsize=11)
    ax1.set_ylabel(f'Train-Val Gap ({metric})', fontsize=11)
    ax1.set_title('Train-Validation Gap by Hyperparameter Combination', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right plot: Train vs Val scores
    ax2 = axes[1]
    x = range(len(gap_df_sorted))
    ax2.scatter(x, gap_df_sorted['train_score'], alpha=0.7, label='Train', marker='o', s=30)
    ax2.scatter(x, gap_df_sorted['val_score'], alpha=0.7, label='Validation', marker='x', s=30)
    ax2.set_xlabel('Parameter Combination (sorted by gap)', fontsize=11)
    ax2.set_ylabel(f'{metric.upper()} Score', fontsize=11)
    ax2.set_title('Train vs Validation Scores', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f'Overfitting Analysis: {metric.upper()} Score'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    _save_figure(fig, save_path)

    # Print summary
    n_overfitting = gap_df['overfitting'].sum()
    print(f"\nOverfitting Analysis Summary ({metric}):")
    print(f"  Total combinations: {len(gap_df)}")
    print(f"  Overfitting (gap > {gap_threshold}): {n_overfitting} ({n_overfitting/len(gap_df)*100:.1f}%)")
    print(f"  Mean gap: {gaps.mean():.4f}")
    print(f"  Max gap: {gaps.max():.4f}")
    print(f"  Best model gap: {gaps[search_object.best_index_]:.4f}")

    return fig, gap_df


def plot_iteration_performance(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    param_name: str = 'n_estimators',
    metric: str = 'f2',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot train/val scores with error bars as function of a parameter.

    This function expects a search object where only one parameter varies
    (the param_name) while all others are fixed.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV object
        param_name: Name of the varying parameter (e.g., 'n_estimators')
        metric: Metric name to plot
        title: Plot title (auto-generated if None)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    cv_results = search_object.cv_results_

    # Extract scores
    train_mean = cv_results[f'mean_train_{metric}']
    train_std = cv_results[f'std_train_{metric}']
    val_mean = cv_results[f'mean_test_{metric}']
    val_std = cv_results[f'std_test_{metric}']

    # Extract parameter values
    param_key = f'param_{param_name}'
    if param_key not in cv_results:
        raise ValueError(f"Parameter '{param_name}' not found in cv_results")

    param_values = cv_results[param_key].data
    # Handle masked arrays
    if hasattr(param_values, 'compressed'):
        param_values = param_values.compressed()

    # Sort by parameter value
    sort_idx = np.argsort(param_values)
    param_values = np.array(param_values)[sort_idx]
    train_mean = np.array(train_mean)[sort_idx]
    train_std = np.array(train_std)[sort_idx]
    val_mean = np.array(val_mean)[sort_idx]
    val_std = np.array(val_std)[sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot train scores with error bars
    ax.errorbar(
        param_values, train_mean, yerr=train_std,
        marker='o', markersize=8, linewidth=2, capsize=5,
        label='Train', color='steelblue', alpha=0.8
    )

    # Plot validation scores with error bars
    ax.errorbar(
        param_values, val_mean, yerr=val_std,
        marker='s', markersize=8, linewidth=2, capsize=5,
        label='Validation', color='coral', alpha=0.8
    )

    # Mark best parameter value
    best_idx = search_object.best_index_
    best_param = cv_results[param_key][best_idx]
    best_val_score = val_mean[np.where(param_values == best_param)[0][0]] if best_param in param_values else None

    if best_val_score is not None:
        ax.axvline(x=best_param, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Best ({param_name}={best_param})')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)

    if title is None:
        title = f'Model Performance vs {param_name}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set x-ticks to parameter values
    ax.set_xticks(param_values)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def analyze_overfitting(
    tuned_models: Dict[str, Union[GridSearchCV, RandomizedSearchCV]],
    metric: str = 'f2',
    gap_threshold: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """Analyze overfitting for all tuned models.

    Args:
        tuned_models: Dictionary mapping model names to fitted search objects
        metric: Metric to analyze
        gap_threshold: Threshold above which to flag overfitting
        verbose: Whether to print analysis

    Returns:
        DataFrame with overfitting analysis for each model
    """
    results = []

    for name, search in tuned_models.items():
        cv_results = search.cv_results_
        best_idx = search.best_index_

        train_key = f'mean_train_{metric}'
        val_key = f'mean_test_{metric}'
        train_std_key = f'std_train_{metric}'
        val_std_key = f'std_test_{metric}'

        # Best model scores
        best_train = cv_results[train_key][best_idx]
        best_val = cv_results[val_key][best_idx]
        best_train_std = cv_results[train_std_key][best_idx]
        best_val_std = cv_results[val_std_key][best_idx]
        best_gap = best_train - best_val

        # All combinations stats
        all_gaps = cv_results[train_key] - cv_results[val_key]
        n_overfitting = (all_gaps > gap_threshold).sum()

        results.append({
            'model': name,
            'train_score': best_train,
            'train_std': best_train_std,
            'val_score': best_val,
            'val_std': best_val_std,
            'gap': best_gap,
            'overfitting': best_gap > gap_threshold,
            'n_combinations': len(all_gaps),
            'n_overfitting_combos': n_overfitting,
            'pct_overfitting': n_overfitting / len(all_gaps) * 100,
        })

    df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print(f"OVERFITTING ANALYSIS: Train vs Validation ({metric.upper()})")
        print("=" * 70)
        print(f"\nGap threshold: {gap_threshold}")
        print("\nBest Model Performance:")
        print("-" * 70)

        for _, row in df.iterrows():
            status = "OVERFITTING" if row['overfitting'] else "OK"
            print(f"\n{row['model']}:")
            print(f"  Train {metric}: {row['train_score']:.4f} (+/- {row['train_std']:.4f})")
            print(f"  Val {metric}:   {row['val_score']:.4f} (+/- {row['val_std']:.4f})")
            print(f"  Gap:        {row['gap']:.4f} [{status}]")
            print(f"  Overfitting combos: {row['n_overfitting_combos']}/{row['n_combinations']} "
                  f"({row['pct_overfitting']:.1f}%)")

        print("\n" + "=" * 70)

    return df
