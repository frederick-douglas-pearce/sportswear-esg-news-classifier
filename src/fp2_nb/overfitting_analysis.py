"""
Cross-validation train-validation gap analysis utilities.

This module provides functions for analyzing overfitting by comparing training
and validation scores from GridSearchCV/RandomizedSearchCV results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided, creating directories as needed."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved: {save_path}")


def analyze_cv_train_val_gap(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    metric: str = 'f2',
    gap_threshold_warning: float = 0.05,
    gap_threshold_severe: float = 0.10,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 5),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze train-validation gap from GridSearchCV/RandomizedSearchCV results.

    This function examines the difference between training and validation scores
    to detect overfitting. Requires that the search object was created with
    `return_train_score=True`.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV object
        metric: The metric to analyze (default: 'f2')
        gap_threshold_warning: Gap percentage above which to show warning (default: 0.05 = 5%)
        gap_threshold_severe: Gap percentage above which to show severe warning (default: 0.10 = 10%)
        model_name: Name of the model for display purposes
        figsize: Figure size for visualization
        verbose: If True, display analysis and create visualizations
        save_path: Optional path to save the figure

    Returns:
        Dictionary with:
        - best_train_score: Training score for best model
        - best_val_score: Validation score for best model
        - gap: Absolute gap (train - val)
        - gap_pct: Gap as percentage of training score
        - diagnosis: 'Good fit', 'MODERATE OVERFITTING', or 'SEVERE OVERFITTING'
        - overfitting_detected: Boolean flag
        - recommendation: Actionable recommendation string

    Example:
        >>> gap_analysis = analyze_cv_train_val_gap(
        ...     rf_search,
        ...     metric='f2',
        ...     model_name='Random Forest'
        ... )
        >>> if gap_analysis['overfitting_detected']:
        ...     print(gap_analysis['recommendation'])
    """
    cv_results = search_object.cv_results_

    # Determine column names
    mean_train_col = f'mean_train_{metric}'
    mean_val_col = f'mean_test_{metric}'
    rank_col = f'rank_test_{metric}'

    # Check if training scores are available
    if mean_train_col not in cv_results:
        raise ValueError(
            f"Training scores not found in CV results. "
            f"Column '{mean_train_col}' not found. "
            f"Ensure GridSearchCV/RandomizedSearchCV was created with return_train_score=True."
        )

    # Get best model index
    best_idx = search_object.best_index_

    # Calculate gap metrics
    best_train_score = cv_results[mean_train_col][best_idx]
    best_val_score = cv_results[mean_val_col][best_idx]
    gap = best_train_score - best_val_score
    gap_pct = gap / best_train_score if best_train_score > 0 else 0

    # Determine diagnosis
    if gap_pct >= gap_threshold_severe:
        diagnosis = 'SEVERE OVERFITTING'
        overfitting_detected = True
    elif gap_pct >= gap_threshold_warning:
        diagnosis = 'MODERATE OVERFITTING'
        overfitting_detected = True
    else:
        diagnosis = 'Good fit'
        overfitting_detected = False

    # Generate recommendation
    recommendation = _generate_gap_recommendation(
        model_name, gap_pct, diagnosis, gap_threshold_warning, gap_threshold_severe
    )

    result = {
        'best_train_score': float(best_train_score),
        'best_val_score': float(best_val_score),
        'gap': float(gap),
        'gap_pct': float(gap_pct),
        'diagnosis': diagnosis,
        'overfitting_detected': overfitting_detected,
        'recommendation': recommendation,
        'model_name': model_name,
        'metric': metric
    }

    if verbose:
        _print_cv_gap_analysis(result, gap_threshold_warning, gap_threshold_severe)
        _plot_cv_gap_analysis(
            cv_results, best_idx, mean_train_col, mean_val_col,
            rank_col, model_name, metric, figsize, save_path
        )

    return result


def _generate_gap_recommendation(
    model_name: str,
    gap_pct: float,
    diagnosis: str,
    gap_threshold_warning: float,
    gap_threshold_severe: float
) -> str:
    """Generate actionable recommendation based on gap analysis."""
    if diagnosis == 'Good fit':
        return f"{model_name} shows healthy generalization with minimal overfitting."

    model_name_upper = model_name.upper()

    if 'RANDOM FOREST' in model_name_upper or 'RF' in model_name_upper:
        if diagnosis == 'SEVERE OVERFITTING':
            return (
                f"{model_name} shows severe overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Reduce model complexity:\n"
                "     - Decrease max_depth (e.g., 20 -> 15 or 10)\n"
                "     - Increase min_samples_leaf (e.g., 1 -> 5 or 10)\n"
                "  2. Increase regularization:\n"
                "     - Increase min_samples_split (e.g., 2 -> 10 or 20)\n"
                "  3. Consider using class_weight='balanced_subsample' if not already"
            )
        else:
            return (
                f"{model_name} shows moderate overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Consider reducing max_depth slightly\n"
                "  2. Increase min_samples_leaf if possible"
            )

    elif 'HISTGRADIENT' in model_name_upper or 'HGB' in model_name_upper:
        if diagnosis == 'SEVERE OVERFITTING':
            return (
                f"{model_name} shows severe overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Increase regularization:\n"
                "     - Increase l2_regularization\n"
                "     - Decrease learning_rate (and increase max_iter)\n"
                "  2. Reduce model complexity:\n"
                "     - Decrease max_depth\n"
                "     - Increase min_samples_leaf\n"
                "  3. Use early_stopping=True with validation_fraction"
            )
        else:
            return (
                f"{model_name} shows moderate overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Consider increasing l2_regularization\n"
                "  2. Enable early_stopping if not already"
            )

    elif 'LOGISTIC' in model_name_upper or 'LR' in model_name_upper:
        if diagnosis == 'SEVERE OVERFITTING':
            return (
                f"{model_name} shows severe overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Increase regularization:\n"
                "     - Decrease C (e.g., 1.0 -> 0.1 or 0.01)\n"
                "  2. Consider using penalty='l1' or 'elasticnet' for feature selection"
            )
        else:
            return (
                f"{model_name} shows moderate overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Consider decreasing C slightly"
            )

    else:
        return (
            f"{model_name} shows {diagnosis.lower()} ({gap_pct:.1%} train-val gap).\n"
            "Consider reducing model complexity or increasing regularization."
        )


def _print_cv_gap_analysis(
    result: Dict[str, Any],
    gap_threshold_warning: float,
    gap_threshold_severe: float
) -> None:
    """Print formatted train-validation gap analysis."""
    model_name = result['model_name']
    metric = result['metric']
    gap_pct = result['gap_pct']
    diagnosis = result['diagnosis']

    print("\n" + "=" * 80)

    if result['overfitting_detected']:
        print(f"!!! OVERFITTING WARNING - {model_name} !!!")
    else:
        print(f"{model_name} - Train-Validation Gap Analysis")

    print("=" * 80)
    print(f"\nMetric: {metric}")
    print(f"  Training score:   {result['best_train_score']:.4f}")
    print(f"  Validation score: {result['best_val_score']:.4f}")
    print(f"  Gap:              {result['gap']:.4f} ({gap_pct:.1%})")
    print(f"\nDiagnosis: {diagnosis}")
    print(f"  (Warning threshold: {gap_threshold_warning:.0%}, Severe threshold: {gap_threshold_severe:.0%})")

    if result['overfitting_detected']:
        print("\n" + "-" * 80)
        print(result['recommendation'])
        print("-" * 80)

    print("=" * 80)


def _plot_cv_gap_analysis(
    cv_results: Dict[str, Any],
    best_idx: int,
    mean_train_col: str,
    mean_val_col: str,
    rank_col: str,
    model_name: str,
    metric: str,
    figsize: Tuple[int, int],
    save_path: Optional[str] = None
) -> None:
    """Create visualization of train-validation gap across all candidates."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(cv_results)

    # Sort by validation score in descending order (best on left)
    sorted_df = df.sort_values(mean_val_col, ascending=False).reset_index(drop=True)

    # Find position of best model in the sorted dataframe
    # The best model is the one with minimum rank
    best_sorted_idx = sorted_df[rank_col].idxmin()

    x = np.arange(len(sorted_df))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Train vs Validation scores
    ax1 = axes[0]
    ax1.plot(x, sorted_df[mean_train_col], 'b-', label='Training', alpha=0.7, linewidth=2)
    ax1.plot(x, sorted_df[mean_val_col], 'g-', label='Validation', alpha=0.7, linewidth=2)
    ax1.fill_between(x, sorted_df[mean_val_col], sorted_df[mean_train_col],
                     alpha=0.3, color='red', label='Gap (Overfitting)')

    # Highlight best model
    ax1.axvline(x=best_sorted_idx, color='red', linestyle='--', linewidth=2,
                label='Best Model (Rank 1)', alpha=0.8)

    ax1.set_xlabel('Candidate (sorted by validation score, best on left)', fontsize=11)
    ax1.set_ylabel(f'{metric.upper()}', fontsize=11)
    ax1.set_title(f'{model_name}: Training vs Validation Scores', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1, 1.02), loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Gap distribution
    ax2 = axes[1]
    gaps = sorted_df[mean_train_col] - sorted_df[mean_val_col]
    gap_pcts = gaps / sorted_df[mean_train_col] * 100

    colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gap_pcts]
    ax2.bar(x, gap_pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add threshold lines
    ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Warning (5%)')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Severe (10%)')

    # Highlight best model
    ax2.bar(best_sorted_idx, gap_pcts.iloc[best_sorted_idx], color='purple',
            alpha=0.9, edgecolor='black', linewidth=2, label='Best Model')

    ax2.set_xlabel('Candidate (sorted by validation score, best on left)', fontsize=11)
    ax2.set_ylabel('Train-Val Gap (%)', fontsize=11)
    ax2.set_title(f'{model_name}: Overfitting Gap by Candidate', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1, 1.02), loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def analyze_iteration_performance(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    param_name: str = 'n_estimators',
    metric: str = 'f2',
    tuned_value: Optional[int] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 6),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze model performance across different parameter values from GridSearchCV results.

    This function examines how training and validation scores change as a parameter
    (typically n_estimators) increases, helping identify the optimal value and detect
    overfitting. The search object should have been created with only this parameter
    varying while other hyperparameters are fixed at their optimal values.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV object
        param_name: Name of the parameter to analyze (default: 'n_estimators')
        metric: The metric used for evaluation (default: 'f2')
        tuned_value: Optional parameter value from main hyperparameter tuning
            to mark on the plot for comparison
        model_name: Name of the model for display purposes
        figsize: Figure size for visualization
        verbose: If True, display analysis and create visualizations
        save_path: Optional path to save the figure

    Returns:
        Dictionary with:
        - optimal_value: Parameter value with best validation score
        - optimal_train_score: Training score at optimal value
        - optimal_val_score: Validation score at optimal value
        - optimal_gap: Train-val gap at optimal value
        - optimal_gap_pct: Gap as percentage at optimal value
        - tracking_df: DataFrame with all iteration results
        - tuned_value: The tuned value if provided

    Example:
        >>> result = analyze_iteration_performance(
        ...     rf_iteration_search,
        ...     param_name='n_estimators',
        ...     model_name='Random Forest',
        ...     tuned_value=200
        ... )
        >>> print(f"Optimal n_estimators: {result['optimal_value']}")
    """
    cv_results = search_object.cv_results_

    # Determine column names
    mean_train_col = f'mean_train_{metric}'
    mean_val_col = f'mean_test_{metric}'
    std_train_col = f'std_train_{metric}'
    std_val_col = f'std_test_{metric}'
    param_col = f'param_{param_name}'

    # Check required columns exist
    if mean_train_col not in cv_results:
        raise ValueError(
            f"Training scores not found. Column '{mean_train_col}' not found. "
            f"Ensure GridSearchCV was created with return_train_score=True."
        )

    if param_col not in cv_results:
        available_params = [k for k in cv_results.keys() if k.startswith('param_')]
        raise ValueError(
            f"Parameter '{param_name}' not found. Column '{param_col}' not found. "
            f"Available parameters: {available_params}"
        )

    # Convert to DataFrame and sort by parameter value
    df = pd.DataFrame(cv_results)
    df = df.sort_values(param_col)

    # Extract values
    param_values = df[param_col].values
    train_scores = df[mean_train_col].values
    val_scores = df[mean_val_col].values
    train_stds = df[std_train_col].values if std_train_col in df.columns else np.zeros_like(train_scores)
    val_stds = df[std_val_col].values if std_val_col in df.columns else np.zeros_like(val_scores)

    # Calculate gaps
    gaps = train_scores - val_scores
    gap_pcts = gaps / train_scores * 100

    # Find optimal value (best validation score)
    optimal_idx = np.argmax(val_scores)
    optimal_value = param_values[optimal_idx]
    # Handle numpy types
    if hasattr(optimal_value, 'item'):
        optimal_value = optimal_value.item()

    # Create tracking DataFrame
    tracking_df = pd.DataFrame({
        param_name: param_values,
        'train_score_mean': train_scores,
        'train_score_std': train_stds,
        'val_score_mean': val_scores,
        'val_score_std': val_stds,
        'gap': gaps,
        'gap_pct': gap_pcts
    })

    result = {
        'optimal_value': optimal_value,
        'optimal_train_score': float(train_scores[optimal_idx]),
        'optimal_val_score': float(val_scores[optimal_idx]),
        'optimal_gap': float(gaps[optimal_idx]),
        'optimal_gap_pct': float(gap_pcts[optimal_idx]),
        'tracking_df': tracking_df,
        'tuned_value': tuned_value,
        'param_name': param_name
    }

    if verbose:
        _print_iteration_analysis(result, model_name, metric)
        _plot_iteration_performance(
            param_values, train_scores, val_scores,
            train_stds, val_stds, optimal_value, tuned_value,
            param_name, model_name, metric, figsize, save_path
        )

    return result


def _print_iteration_analysis(
    result: Dict[str, Any],
    model_name: str,
    metric: str
) -> None:
    """Print formatted iteration analysis."""
    param_name = result['param_name']

    print(f"\n{'=' * 80}")
    print(f"{model_name} - {param_name} Performance Analysis")
    print("=" * 80)
    print(f"\nMetric: {metric}")
    print(f"\nOptimal {param_name}: {result['optimal_value']}")
    print(f"  Training {metric.upper()}:   {result['optimal_train_score']:.4f}")
    print(f"  Validation {metric.upper()}: {result['optimal_val_score']:.4f}")
    print(f"  Gap:               {result['optimal_gap']:.4f} ({result['optimal_gap_pct']:.1f}%)")

    if result['tuned_value'] is not None:
        tuned_val = result['tuned_value']
        df = result['tracking_df']
        tuned_row = df[df[param_name] == tuned_val]
        if len(tuned_row) > 0:
            tuned_row = tuned_row.iloc[0]
            print(f"\nTuned {param_name} from hyperparameter search: {tuned_val}")
            print(f"  Training {metric.upper()}:   {tuned_row['train_score_mean']:.4f}")
            print(f"  Validation {metric.upper()}: {tuned_row['val_score_mean']:.4f}")
            print(f"  Gap:               {tuned_row['gap']:.4f} ({tuned_row['gap_pct']:.1f}%)")

    print("=" * 80)


def _plot_iteration_performance(
    param_values: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_stds: np.ndarray,
    val_stds: np.ndarray,
    optimal_value: Any,
    tuned_value: Optional[Any],
    param_name: str,
    model_name: str,
    metric: str,
    figsize: Tuple[int, int],
    save_path: Optional[str] = None
) -> None:
    """Create iteration performance visualization."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training and validation curves with confidence bands
    ax.plot(param_values, train_scores, 'b-', label='Training', linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(param_values, train_scores - train_stds, train_scores + train_stds,
                    alpha=0.2, color='blue')

    ax.plot(param_values, val_scores, 'g-', label='Validation', linewidth=2.5, marker='s', markersize=8)
    ax.fill_between(param_values, val_scores - val_stds, val_scores + val_stds,
                    alpha=0.2, color='green')

    # Mark optimal value (best validation)
    optimal_idx = np.where(param_values == optimal_value)[0]
    if len(optimal_idx) > 0:
        optimal_idx = optimal_idx[0]
        ax.axvline(x=optimal_value, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter([optimal_value], [val_scores[optimal_idx]], color='green', s=200, zorder=5,
                   edgecolors='black', linewidths=2, label=f'Best Val ({param_name}={optimal_value})')

        # Add gap annotation at optimal point
        gap_at_optimal = train_scores[optimal_idx] - val_scores[optimal_idx]
        gap_pct = gap_at_optimal / train_scores[optimal_idx] * 100
        ax.annotate(f'Gap: {gap_pct:.1f}%',
                    xy=(optimal_value, (train_scores[optimal_idx] + val_scores[optimal_idx]) / 2),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=12, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Mark tuned value if provided and different from optimal
    if tuned_value is not None and tuned_value != optimal_value:
        tuned_idx_arr = np.where(param_values == tuned_value)[0]
        if len(tuned_idx_arr) > 0:
            tuned_idx = tuned_idx_arr[0]
            ax.axvline(x=tuned_value, color='purple', linestyle=':', linewidth=2, alpha=0.7)
            ax.scatter([tuned_value], [val_scores[tuned_idx]], color='purple', s=200, zorder=5,
                       edgecolors='black', linewidths=2, marker='D', label=f'Tuned ({param_name}={tuned_value})')

    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel(f'{metric.upper()}', fontsize=14)
    ax.set_title(f'{model_name}: Performance by {param_name}', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1, 1.02), loc='lower right', fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def get_top_hyperparameter_runs(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    n_top: int = 10,
    metric: str = 'f2'
) -> Tuple[pd.DataFrame, list]:
    """
    Extract top n hyperparameter combinations from a GridSearchCV object.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV object
        n_top: Number of top results to return (default: 10)
        metric: Metric to rank by (default: 'f2')

    Returns:
        Tuple of:
        - DataFrame with top n results including rank, scores, gap, and parameters
        - List of parameter names

    Example:
        >>> top_df, param_names = get_top_hyperparameter_runs(rf_search, n_top=10)
        >>> print(top_df[['rank', 'val_f2', 'train_f2', 'gap'] + param_names])
    """
    cv_results = search_object.cv_results_

    # Get parameter names (remove 'param_' prefix)
    param_names = [k.replace('param_', '') for k in cv_results.keys() if k.startswith('param_')]

    # Build dataframe with all results
    results_data = []
    for i in range(len(cv_results[f'mean_test_{metric}'])):
        row = {
            'rank': cv_results[f'rank_test_{metric}'][i],
            f'val_{metric}': cv_results[f'mean_test_{metric}'][i],
            f'val_{metric}_std': cv_results[f'std_test_{metric}'][i],
            f'train_{metric}': cv_results[f'mean_train_{metric}'][i],
            'gap': cv_results[f'mean_train_{metric}'][i] - cv_results[f'mean_test_{metric}'][i],
        }
        # Add parameter values
        for param in param_names:
            row[param] = cv_results[f'param_{param}'][i]
        results_data.append(row)

    df = pd.DataFrame(results_data)
    df = df.sort_values('rank').head(n_top)

    return df, param_names


def display_top_hyperparameter_runs(
    tuned_models: Dict[str, Union[GridSearchCV, RandomizedSearchCV]],
    n_top: int = 10,
    metric: str = 'f2'
) -> None:
    """
    Display top hyperparameter combinations for multiple tuned models.

    Args:
        tuned_models: Dictionary mapping model names to fitted search objects
        n_top: Number of top results to display per model (default: 10)
        metric: Metric to rank by (default: 'f2')

    Example:
        >>> tuned_models = {'LR_tuned': lr_search, 'RF_tuned': rf_search}
        >>> display_top_hyperparameter_runs(tuned_models, n_top=10, metric='f2')
    """
    for model_name, search in tuned_models.items():
        print("=" * 80)
        print(f"TOP {n_top} HYPERPARAMETER COMBINATIONS: {model_name}")
        print("=" * 80)

        top_df, param_names = get_top_hyperparameter_runs(search, n_top=n_top, metric=metric)

        # Display with formatted output
        display_cols = ['rank', f'val_{metric}', f'val_{metric}_std', f'train_{metric}', 'gap'] + param_names
        print(top_df[display_cols].to_string(index=False))
        print()

        # Also show best parameters explicitly
        print(f"Best parameters: {search.best_params_}")
        print()


def get_gap_summary(
    tuned_models: Dict[str, Union[GridSearchCV, RandomizedSearchCV]],
    metric: str = 'f2',
    gap_threshold: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get a summary DataFrame of train-val gaps for multiple tuned models.

    Args:
        tuned_models: Dictionary mapping model names to fitted search objects
        metric: Metric to analyze
        gap_threshold: Threshold above which to flag overfitting
        verbose: Whether to print summary

    Returns:
        DataFrame with gap analysis for each model
    """
    results = []

    for name, search in tuned_models.items():
        cv_results = search.cv_results_
        best_idx = search.best_index_

        train_key = f'mean_train_{metric}'
        val_key = f'mean_test_{metric}'
        train_std_key = f'std_train_{metric}'
        val_std_key = f'std_test_{metric}'

        if train_key not in cv_results:
            continue

        # Best model scores
        best_train = cv_results[train_key][best_idx]
        best_val = cv_results[val_key][best_idx]
        best_train_std = cv_results[train_std_key][best_idx]
        best_val_std = cv_results[val_std_key][best_idx]
        best_gap = best_train - best_val
        gap_pct = best_gap / best_train if best_train > 0 else 0

        # Determine diagnosis
        if gap_pct >= 0.10:
            diagnosis = 'SEVERE'
        elif gap_pct >= 0.05:
            diagnosis = 'MODERATE'
        else:
            diagnosis = 'Good'

        results.append({
            'model': name,
            'train_score': best_train,
            'train_std': best_train_std,
            'val_score': best_val,
            'val_std': best_val_std,
            'gap': best_gap,
            'gap_pct': gap_pct,
            'diagnosis': diagnosis,
        })

    df = pd.DataFrame(results)

    if verbose and len(df) > 0:
        print("\n" + "=" * 80)
        print(f"TRAIN-VALIDATION GAP SUMMARY ({metric.upper()})")
        print("=" * 80)
        print(f"\nGap thresholds: Warning >= 5%, Severe >= 10%")
        print("\n" + "-" * 80)

        for _, row in df.iterrows():
            status_icon = "✓" if row['diagnosis'] == 'Good' else "⚠" if row['diagnosis'] == 'MODERATE' else "✗"
            print(f"\n{status_icon} {row['model']}:")
            print(f"    Train {metric}: {row['train_score']:.4f} (+/- {row['train_std']:.4f})")
            print(f"    Val {metric}:   {row['val_score']:.4f} (+/- {row['val_std']:.4f})")
            print(f"    Gap:        {row['gap']:.4f} ({row['gap_pct']:.1%}) [{row['diagnosis']}]")

        print("\n" + "=" * 80)

    return df
