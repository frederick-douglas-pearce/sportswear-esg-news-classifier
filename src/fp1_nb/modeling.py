"""Modeling utilities for FP classifier notebook."""

import sys
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# F2 scorer for recall-focused optimization
f2_scorer = make_scorer(fbeta_score, beta=2)


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def create_search_object(
    search_type: str,
    estimator: Any,
    param_grid: Dict,
    scoring: Union[str, Dict, List] = None,
    refit: Union[str, bool] = 'average_precision',
    cv: Any = None,
    n_iter: Optional[int] = None,
    verbose: int = 1,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Create GridSearchCV or RandomizedSearchCV object.

    Args:
        search_type: 'grid' or 'random'
        estimator: Sklearn estimator or pipeline
        param_grid: Parameter grid dictionary
        scoring: Scoring metric(s)
        refit: Metric to use for refitting (for multi-metric)
        cv: Cross-validation strategy
        n_iter: Number of iterations (for RandomizedSearchCV)
        verbose: Verbosity level
        random_state: Random seed
        n_jobs: Number of parallel jobs

    Returns:
        Configured search object
    """
    if scoring is None:
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'f2': f2_scorer,
            'precision': 'precision',
            'recall': 'recall',
            'average_precision': 'average_precision',
        }

    common_params = {
        'estimator': estimator,
        'param_grid': param_grid,
        'scoring': scoring,
        'refit': refit,
        'cv': cv if cv is not None else 5,
        'verbose': verbose,
        'n_jobs': n_jobs,
        'return_train_score': True,
    }

    if search_type.lower() == 'grid':
        return GridSearchCV(**common_params)
    elif search_type.lower() == 'random':
        if n_iter is None:
            n_iter = 50
        return RandomizedSearchCV(
            **common_params,
            n_iter=n_iter,
            random_state=random_state
        )
    else:
        raise ValueError(f"search_type must be 'grid' or 'random', got '{search_type}'")


def _calculate_total_combinations(param_grid: Dict) -> int:
    """Calculate total number of parameter combinations."""
    total = 1
    for values in param_grid.values():
        if hasattr(values, '__len__'):
            total *= len(values)
    return total


def tune_with_logging(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    log_dir: str = 'models/logs',
    verbose: bool = True
) -> Tuple[Union[GridSearchCV, RandomizedSearchCV], str, str]:
    """Execute hyperparameter search with logging.

    Args:
        search_object: Configured search object
        X: Feature matrix
        y: Target vector
        model_name: Name for logging
        log_dir: Directory for log files
        verbose: Whether to print progress

    Returns:
        Tuple of (fitted search object, log file path, csv file path)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    csv_file = log_dir / f"{model_name}_{timestamp}_cv_results.csv"

    if verbose:
        total_combinations = _calculate_total_combinations(search_object.param_grid)
        n_splits = search_object.cv if isinstance(search_object.cv, int) else 5
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'='*60}")
        print(f"Total parameter combinations: {total_combinations}")
        print(f"Cross-validation folds: {n_splits}")
        print(f"Total fits: {total_combinations * n_splits}")
        print(f"Log file: {log_file}")
        print(f"{'='*60}\n")

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search_object.fit(X, y)
    finally:
        sys.stdout = old_stdout

    # Save log
    with open(log_file, 'w') as f:
        f.write(captured_output.getvalue())

    # Save CV results
    cv_results = pd.DataFrame(search_object.cv_results_)
    cv_results.to_csv(csv_file, index=False)

    if verbose:
        print(f"Tuning complete!")
        print(f"Best score ({search_object.refit}): {search_object.best_score_:.4f}")
        print(f"Results saved to: {csv_file}")

    return search_object, str(log_file), str(csv_file)


def extract_cv_metrics(
    search_object: Union[GridSearchCV, RandomizedSearchCV]
) -> Dict[str, float]:
    """Extract CV metrics for best model.

    Args:
        search_object: Fitted search object

    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}
    cv_results = search_object.cv_results_
    best_idx = search_object.best_index_

    # Get all scoring metrics
    for key in cv_results.keys():
        if key.startswith('mean_test_'):
            metric_name = key.replace('mean_test_', '')
            metrics[f'cv_{metric_name}'] = cv_results[key][best_idx]
        elif key.startswith('std_test_'):
            metric_name = key.replace('std_test_', '')
            metrics[f'cv_{metric_name}_std'] = cv_results[key][best_idx]

    return metrics


def get_best_params_summary(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    model_name: str = "Model",
    refit_metric: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Get summary of best parameters and scores.

    Args:
        search_object: Fitted search object
        model_name: Model name for display
        refit_metric: Primary metric used for refitting
        verbose: Whether to print summary

    Returns:
        Dictionary with best parameters and metrics
    """
    summary = {
        'model_name': model_name,
        'best_params': search_object.best_params_,
        'best_score': search_object.best_score_,
        'refit_metric': refit_metric or search_object.refit,
        'cv_metrics': extract_cv_metrics(search_object),
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"BEST PARAMETERS: {model_name}")
        print(f"{'='*60}")
        print(f"\nRefit metric: {summary['refit_metric']}")
        print(f"Best CV score: {summary['best_score']:.4f}")
        print(f"\nBest parameters:")
        for param, value in summary['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nCV Metrics:")
        for metric, value in summary['cv_metrics'].items():
            if not metric.endswith('_std'):
                std_key = f"{metric}_std"
                std = summary['cv_metrics'].get(std_key, 0)
                print(f"  {metric}: {value:.4f} (+/- {std:.4f})")
        print(f"{'='*60}")

    return summary


def compare_models(
    model_metrics: List[Dict[str, float]],
    metrics_to_display: Optional[List[str]] = None,
    metrics_to_highlight: Optional[List[str]] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (16, 6),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Compare multiple models on key metrics.

    Args:
        model_metrics: List of dictionaries with model metrics
        metrics_to_display: Metrics to include in comparison
        metrics_to_highlight: Metrics to highlight best values
        title: Display title
        figsize: Figure size
        verbose: Whether to display table and plots
        save_path: Optional path to save figure

    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(model_metrics)

    if 'model_name' in df.columns:
        df = df.set_index('model_name')

    if metrics_to_display:
        display_cols = [c for c in metrics_to_display if c in df.columns]
        df = df[display_cols]

    if verbose:
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
        print(df.round(4).to_string())
        print(f"{'='*60}\n")

        # Plot comparison
        if len(df.columns) > 0:
            fig, ax = plt.subplots(figsize=figsize)
            df.plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)

            plt.tight_layout()
            _save_figure(fig, save_path)
            plt.show()

    return df


def get_best_model(
    comparison_df: pd.DataFrame,
    primary_metric: str = 'cv_average_precision'
) -> Tuple[str, Dict[str, float]]:
    """Get the best model based on a primary metric.

    Args:
        comparison_df: Model comparison DataFrame
        primary_metric: Metric to rank by

    Returns:
        Tuple of (model_name, metrics_dict)
    """
    if primary_metric not in comparison_df.columns:
        raise ValueError(f"Metric '{primary_metric}' not found in comparison")

    best_model = comparison_df[primary_metric].idxmax()
    best_metrics = comparison_df.loc[best_model].to_dict()

    print(f"\nBest model by {primary_metric}: {best_model}")
    print(f"Score: {best_metrics[primary_metric]:.4f}")

    return best_model, best_metrics


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    dataset_name: str = "Test",
    threshold: float = 0.5,
    verbose: bool = True,
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate classification model with comprehensive metrics.

    Args:
        model: Fitted model with predict and predict_proba methods
        X: Feature matrix
        y: True labels
        model_name: Model name for display
        dataset_name: Dataset name (e.g., 'Validation', 'Test')
        threshold: Classification threshold
        verbose: Whether to print metrics
        plot: Whether to show ROC and PR curves
        figsize: Figure size for plots
        save_path: Optional path to save figure

    Returns:
        Dictionary with all metrics
    """
    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'dataset': dataset_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_proba)
        metrics['pr_auc'] = average_precision_score(y, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp

    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION: {model_name} on {dataset_name}")
        print(f"{'='*60}")
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if y_proba is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:,}  FP: {fp:,}")
        print(f"  FN: {fn:,}  TP: {tp:,}")
        print(f"\n{classification_report(y, y_pred, target_names=['Not Sportswear', 'Sportswear'])}")
        print(f"{'='*60}")

    if plot and y_proba is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        axes[0].plot(fpr, tpr, 'b-', linewidth=2,
                    label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y, y_proba)
        axes[1].plot(recall, precision, 'b-', linewidth=2,
                    label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
        axes[1].axhline(y=y.mean(), color='k', linestyle='--', linewidth=1,
                       label=f'Baseline ({y.mean():.3f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3)

        # Confusion Matrix Heatmap
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   xticklabels=['Not Sportswear', 'Sportswear'],
                   yticklabels=['Not Sportswear', 'Sportswear'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

        plt.suptitle(f'{model_name} - {dataset_name} Set Evaluation',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        _save_figure(fig, save_path)
        plt.show()

    return metrics


def compare_val_test_performance(
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    metrics_to_compare: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Compare validation vs test set performance.

    Args:
        val_metrics: Validation set metrics
        test_metrics: Test set metrics
        metrics_to_compare: List of metrics to compare
        verbose: Whether to print comparison

    Returns:
        Comparison DataFrame
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    comparison = []
    for metric in metrics_to_compare:
        if metric in val_metrics and metric in test_metrics:
            val_score = val_metrics[metric]
            test_score = test_metrics[metric]
            diff = test_score - val_score
            pct_change = (diff / val_score * 100) if val_score != 0 else 0

            comparison.append({
                'metric': metric,
                'validation': val_score,
                'test': test_score,
                'difference': diff,
                'pct_change': pct_change,
            })

    df = pd.DataFrame(comparison)

    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION vs TEST PERFORMANCE")
        print(f"{'='*60}")
        print(df.round(4).to_string(index=False))

        # Check for overfitting
        avg_drop = df['difference'].mean()
        if avg_drop < -0.05:
            print(f"\n[WARNING] Average performance drop: {avg_drop:.4f}")
            print("Model may be overfitting to validation set.")
        else:
            print(f"\n[OK] Model generalizes well (avg diff: {avg_drop:.4f})")
        print(f"{'='*60}")

    return df
