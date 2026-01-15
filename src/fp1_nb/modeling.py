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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

# Suppress sklearn parallel warning (harmless - just about config propagation)
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*should be used with.*sklearn.utils.parallel.Parallel.*",
    category=UserWarning
)

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
        'scoring': scoring,
        'refit': refit,
        'cv': cv if cv is not None else 5,
        'verbose': verbose,
        'n_jobs': n_jobs,
        'return_train_score': True,
    }

    if search_type.lower() == 'grid':
        return GridSearchCV(param_grid=param_grid, **common_params)
    elif search_type.lower() == 'random':
        if n_iter is None:
            n_iter = 50
        return RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_state,
            **common_params
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
        n_splits = search_object.cv if isinstance(search_object.cv, int) else getattr(search_object.cv, 'n_splits', 5)
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


def plot_fe_comparison(
    fe_df: pd.DataFrame,
    classifiers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot feature engineering comparison for multiple classifiers.

    Creates a side-by-side horizontal bar chart comparing F2 and Recall scores
    for each feature engineering method across specified classifiers.

    Args:
        fe_df: DataFrame with columns 'classifier', 'name', 'cv_f2', 'cv_recall'
        classifiers: List of classifier names to display (default: all three classifiers)
        figsize: Figure size tuple
        save_path: Optional path to save figure
    """
    if classifiers is None:
        classifiers = ['LogisticRegression', 'RandomForest', 'HistGradientBoosting']

    fig, axes = plt.subplots(1, len(classifiers), figsize=figsize)

    # Handle single classifier case
    if len(classifiers) == 1:
        axes = [axes]

    for idx, clf_name in enumerate(classifiers):
        ax = axes[idx]
        clf_df = fe_df[fe_df['classifier'] == clf_name].set_index('name')
        clf_df = clf_df.sort_values('cv_f2', ascending=True)

        x = range(len(clf_df))
        width = 0.35

        ax.barh([i - width / 2 for i in x], clf_df['cv_f2'], width, label='F2', color='steelblue')
        ax.barh([i + width / 2 for i in x], clf_df['cv_recall'], width, label='Recall', color='coral')

        ax.set_yticks(list(x))
        ax.set_yticklabels(clf_df.index)
        ax.set_xlabel('Score')
        ax.set_title(f'{clf_name} Performance by Feature Engineering')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def analyze_tuning_results(
    tuning_df: pd.DataFrame,
    param_name: str,
    default_value: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Any, float]:
    """Analyze hyperparameter tuning results and find optimal value.

    Prints a summary of tuning results sorted by CV F2 score and identifies
    the best parameter value. Optionally compares against a default value.

    Args:
        tuning_df: DataFrame with columns: param_name, cv_f2, cv_f2_std, cv_recall, cv_precision
        param_name: Name of the tuned parameter column
        default_value: Optional default value to compare improvement against
        verbose: Whether to print results summary

    Returns:
        Tuple of (optimal_param_value, best_cv_f2)
    """
    tuning_df_sorted = tuning_df.sort_values('cv_f2', ascending=False)

    # Find best value
    best_idx = tuning_df['cv_f2'].idxmax()
    optimal_param_value = tuning_df.loc[best_idx, param_name]

    # Handle different types (int vs float)
    if isinstance(optimal_param_value, (np.integer, int)):
        optimal_param_value = int(optimal_param_value)

    best_tuned_f2 = tuning_df.loc[best_idx, 'cv_f2']

    # Calculate improvement vs default if provided
    default_f2 = None
    improvement = None
    if default_value is not None:
        default_f2_rows = tuning_df[tuning_df[param_name] == default_value]
        if len(default_f2_rows) > 0:
            default_f2 = default_f2_rows['cv_f2'].values[0]
            improvement = (best_tuned_f2 - default_f2) * 100

    if verbose:
        print(f"{param_name.upper()} TUNING RESULTS")
        print("=" * 70)
        print(tuning_df_sorted.to_string(index=False))
        print("\n" + "=" * 70)
        print(f"Optimal {param_name}: {optimal_param_value}")
        print(f"Best CV F2: {best_tuned_f2:.4f}")
        if default_f2 is not None:
            print(f"Default ({default_value}) CV F2: {default_f2:.4f}")
            print(f"Improvement: {improvement:+.2f}%")
        print("=" * 70)

    return optimal_param_value, best_tuned_f2


def plot_tuning_results(
    tuning_df: pd.DataFrame,
    param_name: str,
    param_values: List[Any],
    optimal_param_value: Any,
    method_name: str,
    description: str,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot hyperparameter tuning results with error bars.

    Creates a line plot showing CV F2 scores across parameter values,
    with error bars for standard deviation and the best value highlighted.

    Args:
        tuning_df: DataFrame with columns: param_name, cv_f2, cv_f2_std
        param_name: Name of the tuned parameter (used as x-axis label and for sorting)
        param_values: List of parameter values tested (used for x-axis ticks)
        optimal_param_value: Best parameter value to highlight
        method_name: Feature engineering method name (for title)
        description: Human-readable description of the parameter (for title)
        figsize: Figure size tuple
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by parameter value for plotting
    plot_df = tuning_df.sort_values(param_name)

    ax.errorbar(
        plot_df[param_name],
        plot_df['cv_f2'],
        yerr=plot_df['cv_f2_std'],
        marker='o',
        markersize=8,
        capsize=5,
        linewidth=2,
        color='steelblue'
    )

    # Highlight best value
    best_tuned_f2 = tuning_df.loc[tuning_df['cv_f2'].idxmax(), 'cv_f2']
    ax.scatter(
        [optimal_param_value], [best_tuned_f2],
        s=150, c='red', zorder=5,
        label=f'Best: {optimal_param_value}'
    )

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('CV F2 Score', fontsize=12)
    ax.set_title(f'{method_name}: {description} Tuning', fontsize=14)
    ax.set_xticks(param_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def tune_feature_transformer(
    transformer_class: Any,
    base_config: Dict[str, Any],
    param_name: str,
    param_values: List[Any],
    X_fe: pd.Series,
    classifier: Any,
    cv: Any,
    X_cv: pd.Series,
    y_cv: np.ndarray,
    scorer: Any,
    fe_source_names: Optional[List[str]] = None,
    fe_categories: Optional[List[str]] = None,
    cv_source_names: Optional[List[str]] = None,
    cv_categories: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tune a single hyperparameter for a feature transformer.

    Evaluates different values of a hyperparameter using cross-validation
    with a specified classifier and scoring function.

    To prevent data leakage, the feature transformer is fitted on training data
    only, but cross-validation is performed on train+val combined for more
    reliable estimates.

    Args:
        transformer_class: Feature transformer class (e.g., FPFeatureTransformer)
        base_config: Base configuration dictionary for the transformer
        param_name: Name of the hyperparameter to tune
        param_values: List of values to try for the hyperparameter
        X_fe: Training text features for fitting transformer (pd.Series)
        classifier: Classifier to use for evaluation
        cv: Cross-validation splitter
        X_cv: Train+val text features for CV evaluation
        y_cv: Train+val labels for CV evaluation
        scorer: Scoring function (e.g., f2_scorer)
        fe_source_names: Optional source names for transformer fitting
        fe_categories: Optional categories for transformer fitting
        cv_source_names: Optional source names for CV transformation
        cv_categories: Optional categories for CV transformation
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: param_name, cv_f2, cv_f2_std, cv_recall, cv_precision
    """
    results = []

    for value in param_values:
        print(f"Testing {param_name}={value}...")

        # Create config with tuned parameter
        config = base_config.copy()
        config[param_name] = value

        # Fit transformer on training data only (to prevent data leakage)
        transformer = transformer_class(**config, random_state=random_state)
        transformer.fit_transform(
            X_fe,
            source_names=fe_source_names,
            categories=fe_categories
        )

        # Transform train+val data for CV evaluation
        X_cv_transformed = transformer.transform(
            X_cv,
            source_names=cv_source_names,
            categories=cv_categories
        )

        cv_scores = cross_validate(
            classifier, X_cv_transformed, y_cv,
            cv=cv,
            scoring={'f2': scorer, 'recall': 'recall', 'precision': 'precision'},
            return_train_score=False
        )

        results.append({
            param_name: value,
            'cv_f2': cv_scores['test_f2'].mean(),
            'cv_f2_std': cv_scores['test_f2'].std(),
            'cv_recall': cv_scores['test_recall'].mean(),
            'cv_precision': cv_scores['test_precision'].mean(),
        })

        print(f"  CV F2: {results[-1]['cv_f2']:.4f} (+/- {results[-1]['cv_f2_std']:.4f})")

    return pd.DataFrame(results)


def tune_lsa_components_fast(
    transformer_class: Any,
    base_config: Dict[str, Any],
    lsa_values: List[int],
    X_fe: pd.Series,
    classifier: Any,
    cv: Any,
    X_cv: pd.Series,
    y_cv: np.ndarray,
    scorer: Any,
    fe_source_names: Optional[List[str]] = None,
    fe_categories: Optional[List[str]] = None,
    cv_source_names: Optional[List[str]] = None,
    cv_categories: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tune LSA n_components efficiently by fitting TF-IDF only once.

    This is an optimized version of tune_feature_transformer() specifically for
    tuning lsa_n_components. It provides ~40-50% speedup by:
    1. Fitting TF-IDF vectorizer once (it doesn't depend on lsa_n_components)
    2. Precomputing all non-LSA features once (NER, proximity, brand indicators)
    3. Only re-fitting the SVD decomposition for each lsa_n_components value

    Args:
        transformer_class: Feature transformer class (e.g., FPFeatureTransformer)
        base_config: Base configuration dictionary for the transformer
            (must have method='tfidf_lsa_ner_proximity_brands' or similar LSA method)
        lsa_values: List of LSA n_components values to try
        X_fe: Training text features for fitting transformer (pd.Series)
        classifier: Classifier to use for evaluation
        cv: Cross-validation splitter
        X_cv: Train+val text features for CV evaluation (pd.Series)
        y_cv: Train+val labels for CV evaluation
        scorer: Scoring function (e.g., f2_scorer)
        fe_source_names: Optional source names for transformer fitting
        fe_categories: Optional categories for transformer fitting
        cv_source_names: Optional source names for CV transformation
        cv_categories: Optional categories for CV transformation
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: lsa_n_components, cv_f2, cv_f2_std, cv_recall, cv_precision
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    results = []

    # Create transformer with max LSA value to fit TF-IDF and all non-LSA features
    config = base_config.copy()
    config['lsa_n_components'] = max(lsa_values)

    print(f"Fitting TF-IDF and precomputing non-LSA features...")
    base_transformer = transformer_class(**config, random_state=random_state)

    # Preprocess texts once
    texts_fe = base_transformer._preprocess_texts(X_fe)
    texts_cv = base_transformer._preprocess_texts(X_cv)

    # Fit TF-IDF once - this is the expensive part that doesn't change
    base_transformer._tfidf = base_transformer._create_tfidf_word()
    tfidf_fe = base_transformer._tfidf.fit_transform(texts_fe)
    tfidf_cv = base_transformer._tfidf.transform(texts_cv)

    # Precompute all non-LSA features once (they don't depend on lsa_n_components)
    print("Precomputing NER and other features...")

    # NER features
    ner_features_fe = base_transformer._compute_ner_features(texts_fe)
    ner_features_cv = base_transformer._compute_ner_features(texts_cv)

    # Brand-specific NER features
    brand_ner_features_fe = base_transformer._compute_brand_specific_ner_features(texts_fe)
    brand_ner_features_cv = base_transformer._compute_brand_specific_ner_features(texts_cv)

    # Proximity features
    proximity_features_fe = base_transformer._compute_proximity_features(texts_fe)
    proximity_features_cv = base_transformer._compute_proximity_features(texts_cv)

    # Negative context features
    neg_context_fe = base_transformer._compute_negative_context_features(texts_fe)
    neg_context_cv = base_transformer._compute_negative_context_features(texts_cv)

    # FP indicator features
    fp_indicator_fe = base_transformer._compute_fp_indicator_features(texts_fe)
    fp_indicator_cv = base_transformer._compute_fp_indicator_features(texts_cv)

    # Brand indicator features
    brand_indicators_fe = base_transformer._compute_brand_indicators(texts_fe)
    brand_indicators_cv = base_transformer._compute_brand_indicators(texts_cv)

    # Fit scalers on training features
    ner_scaler = StandardScaler()
    ner_scaler.fit(ner_features_fe)
    ner_scaled_fe = ner_scaler.transform(ner_features_fe)
    ner_scaled_cv = ner_scaler.transform(ner_features_cv)

    brand_ner_scaler = StandardScaler()
    brand_ner_scaler.fit(brand_ner_features_fe)
    brand_ner_scaled_fe = brand_ner_scaler.transform(brand_ner_features_fe)
    brand_ner_scaled_cv = brand_ner_scaler.transform(brand_ner_features_cv)

    proximity_scaler = StandardScaler()
    proximity_scaler.fit(proximity_features_fe)
    proximity_scaled_fe = proximity_scaler.transform(proximity_features_fe)
    proximity_scaled_cv = proximity_scaler.transform(proximity_features_cv)

    neg_context_scaler = StandardScaler()
    neg_context_scaler.fit(neg_context_fe)
    neg_context_scaled_fe = neg_context_scaler.transform(neg_context_fe)
    neg_context_scaled_cv = neg_context_scaler.transform(neg_context_cv)

    fp_indicator_scaler = StandardScaler()
    fp_indicator_scaler.fit(fp_indicator_fe)
    fp_indicator_scaled_fe = fp_indicator_scaler.transform(fp_indicator_fe)
    fp_indicator_scaled_cv = fp_indicator_scaler.transform(fp_indicator_cv)

    # Brand indicators don't need scaling (already 0/1)

    print(f"Testing {len(lsa_values)} LSA component values...")

    # Now iterate only over LSA values - this is fast since we only fit SVD
    for n_components in lsa_values:
        print(f"Testing lsa_n_components={n_components}...")

        # Fit new LSA with this n_components value
        lsa = TruncatedSVD(n_components=n_components, random_state=random_state)
        lsa_fe = lsa.fit_transform(tfidf_fe)
        lsa_cv = lsa.transform(tfidf_cv)

        # Scale LSA features
        lsa_scaler = StandardScaler()
        lsa_scaled_fe = lsa_scaler.fit_transform(lsa_fe)
        lsa_scaled_cv = lsa_scaler.transform(lsa_cv)

        # Combine all features
        X_cv_transformed = np.hstack([
            lsa_scaled_cv,
            ner_scaled_cv,
            brand_ner_scaled_cv,
            proximity_scaled_cv,
            neg_context_scaled_cv,
            fp_indicator_scaled_cv,
            brand_indicators_cv,
        ])

        # Run cross-validation
        cv_scores = cross_validate(
            classifier, X_cv_transformed, y_cv,
            cv=cv,
            scoring={'f2': scorer, 'recall': 'recall', 'precision': 'precision'},
            return_train_score=False
        )

        results.append({
            'lsa_n_components': n_components,
            'cv_f2': cv_scores['test_f2'].mean(),
            'cv_f2_std': cv_scores['test_f2'].std(),
            'cv_recall': cv_scores['test_recall'].mean(),
            'cv_precision': cv_scores['test_precision'].mean(),
        })

        print(f"  CV F2: {results[-1]['cv_f2']:.4f} (+/- {results[-1]['cv_f2_std']:.4f})")

    return pd.DataFrame(results)


def evaluate_feature_engineering(
    fe_configs: Dict[str, Dict[str, Any]],
    classifiers: Dict[str, Dict[str, Any]],
    transformer_class: Any,
    X_train: pd.Series,
    X_trainval: pd.Series,
    y_trainval: np.ndarray,
    cv: Any,
    scorer: Any = None,
    train_source_names: Optional[List[str]] = None,
    train_categories: Optional[List[str]] = None,
    trainval_source_names: Optional[List[str]] = None,
    trainval_categories: Optional[List[str]] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate multiple feature engineering approaches with multiple classifiers.

    Fits each transformer on training data only (to prevent data leakage),
    then evaluates with cross-validation on train+val combined.

    Args:
        fe_configs: Dictionary mapping FE method names to config dictionaries.
            Each config should contain 'method' and other transformer parameters.
        classifiers: Dictionary mapping classifier names to config dictionaries.
            Each config should contain 'model' (sklearn estimator) and
            'requires_dense' (bool indicating if dense arrays are needed).
        transformer_class: Feature transformer class (e.g., FPFeatureTransformer)
        X_train: Training text features for fitting transformer (pd.Series)
        X_trainval: Train+val text features for CV evaluation (pd.Series)
        y_trainval: Train+val labels for CV evaluation (np.ndarray)
        cv: Cross-validation splitter (e.g., StratifiedKFold)
        scorer: Scoring function for F2 (default: f2_scorer)
        train_source_names: Optional source names for transformer fitting
        train_categories: Optional categories for transformer fitting
        trainval_source_names: Optional source names for CV transformation
        trainval_categories: Optional categories for CV transformation
        random_state: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        List of result dictionaries, each containing:
            - name: FE method name
            - classifier: Classifier name
            - n_features: Number of features
            - cv_f2: Mean CV F2 score
            - cv_f2_std: Std of CV F2 score
            - cv_recall: Mean CV recall
            - cv_precision: Mean CV precision

    Example:
        >>> fe_configs = {
        ...     'tfidf_lsa': {'method': 'tfidf_lsa', 'max_features': 10000},
        ...     'sentence_transformer': {'method': 'sentence_transformer_ner'},
        ... }
        >>> classifiers = {
        ...     'LogisticRegression': {
        ...         'model': LogisticRegression(max_iter=2000),
        ...         'requires_dense': False,
        ...     },
        ...     'RandomForest': {
        ...         'model': RandomForestClassifier(n_estimators=100),
        ...         'requires_dense': False,
        ...     },
        ... }
        >>> results = evaluate_feature_engineering(
        ...     fe_configs, classifiers, FPFeatureTransformer,
        ...     X_train, X_trainval, y_trainval, cv
        ... )
    """
    from scipy import sparse

    if scorer is None:
        scorer = f2_scorer

    fe_results = []

    for name, config in fe_configs.items():
        if verbose:
            print(f"Evaluating {name}...")

        # Filter out 'tuning' key from config as it's not a transformer parameter
        transformer_config = {k: v for k, v in config.items() if k != 'tuning'}

        # Create transformer
        transformer = transformer_class(**transformer_config, random_state=random_state)

        # Fit transformer on TRAINING data only (to prevent data leakage)
        transformer.fit_transform(
            X_train,
            source_names=train_source_names,
            categories=train_categories
        )

        # Transform train+val data for CV evaluation
        X_trainval_fe = transformer.transform(
            X_trainval,
            source_names=trainval_source_names,
            categories=trainval_categories
        )

        if verbose:
            print(f"  Feature shape: {X_trainval_fe.shape}, sparse: {sparse.issparse(X_trainval_fe)}")

        # Evaluate with each classifier
        for clf_name, clf_info in classifiers.items():
            clf = clf_info['model']

            # Convert to dense if classifier requires it
            if clf_info['requires_dense'] and sparse.issparse(X_trainval_fe):
                X_for_cv = X_trainval_fe.toarray()
            else:
                X_for_cv = X_trainval_fe

            # Cross-validation on train+val combined
            cv_scores = cross_validate(
                clf, X_for_cv, y_trainval,
                cv=cv,
                scoring={
                    'f2': scorer,
                    'recall': 'recall',
                    'precision': 'precision',
                },
                return_train_score=False
            )

            result = {
                'name': name,
                'classifier': clf_name,
                'n_features': X_trainval_fe.shape[1],
                'cv_f2': cv_scores['test_f2'].mean(),
                'cv_f2_std': cv_scores['test_f2'].std(),
                'cv_recall': cv_scores['test_recall'].mean(),
                'cv_precision': cv_scores['test_precision'].mean(),
            }
            fe_results.append(result)

            if verbose:
                print(f"  [{clf_name}] CV F2: {result['cv_f2']:.4f} (+/- {result['cv_f2_std']:.4f})")

    return fe_results


def run_transformer_tuning(
    best_fe: str,
    best_clf: str,
    fe_configs: Dict[str, Dict[str, Any]],
    tuning_configs: Dict[str, Dict[str, Any]],
    classifiers: Dict[str, Dict[str, Any]],
    transformer_class: Any,
    X_train: pd.Series,
    X_trainval: pd.Series,
    y_trainval: np.ndarray,
    cv: Any,
    scorer: Any,
    images_dir: Path,
    train_source_names: Optional[List[str]] = None,
    train_categories: Optional[List[str]] = None,
    trainval_source_names: Optional[List[str]] = None,
    trainval_categories: Optional[List[str]] = None,
    random_state: int = 42,
    n_folds: int = 3,
) -> Tuple[Optional[pd.DataFrame], Optional[Any], Optional[float]]:
    """Run hyperparameter tuning for the best feature transformer.

    Tunes the key hyperparameter for the best-performing feature engineering
    method using cross-validation.

    Args:
        best_fe: Name of the best feature engineering method
        best_clf: Name of the best classifier
        fe_configs: Feature engineering configurations dictionary
        tuning_configs: Tuning configurations dictionary with param_name, param_values, description
        classifiers: Classifiers dictionary with 'model' and 'requires_dense' keys
        transformer_class: Feature transformer class (e.g., FPFeatureTransformer)
        X_train: Training text features for fitting transformer
        X_trainval: Train+val text features for CV evaluation
        y_trainval: Train+val labels for CV evaluation
        cv: Cross-validation splitter
        scorer: Scoring function for optimization
        images_dir: Directory to save tuning plots
        train_source_names: Optional source names for transformer fitting (FP only)
        train_categories: Optional categories for transformer fitting (FP only)
        trainval_source_names: Optional source names for CV transformation (FP only)
        trainval_categories: Optional categories for CV transformation (FP only)
        random_state: Random seed for reproducibility
        n_folds: Number of CV folds (for display only)

    Returns:
        Tuple of (tuning_df, optimal_param_value, best_tuned_f2)
        Returns (None, None, None) if no tuning config exists for best_fe
    """
    # Get the baseline classifier
    baseline_clf = classifiers[best_clf]['model']

    # Check if tuning configuration exists for the best method
    if best_fe not in tuning_configs:
        print(f"No tuning configuration defined for {best_fe}")
        print("No tuning was performed.")
        return None, None, None

    tuning_config = tuning_configs[best_fe]
    param_name = tuning_config['param_name']
    param_values = tuning_config['param_values']
    description = tuning_config['description']

    print("=" * 70)
    print(f"TUNING {param_name.upper()} FOR {best_fe}")
    print("=" * 70)
    print(f"\nTesting values: {param_values}")
    print(f"Classifier: {best_clf} (baseline)")
    print(f"Transformer fitted on: TRAINING data only ({len(X_train)} samples)")
    print(f"CV evaluated on: TRAIN+VAL combined ({len(X_trainval)} samples)")
    print(f"CV: {n_folds}-fold stratified\n")

    # Build kwargs for tune_feature_transformer
    # Filter out 'tuning' key from config as it's not a transformer parameter
    base_config = {k: v for k, v in fe_configs[best_fe].items() if k != 'tuning'}
    tune_kwargs = {
        'transformer_class': transformer_class,
        'base_config': base_config,
        'param_name': param_name,
        'param_values': param_values,
        'X_fe': X_train,
        'classifier': baseline_clf,
        'cv': cv,
        'X_cv': X_trainval,
        'y_cv': y_trainval,
        'scorer': scorer,
        'random_state': random_state,
    }

    # Add optional source_names and categories if provided (FP transformer)
    if train_source_names is not None:
        tune_kwargs['fe_source_names'] = train_source_names
    if train_categories is not None:
        tune_kwargs['fe_categories'] = train_categories
    if trainval_source_names is not None:
        tune_kwargs['cv_source_names'] = trainval_source_names
    if trainval_categories is not None:
        tune_kwargs['cv_categories'] = trainval_categories

    # Run tuning - use optimized LSA tuning if tuning lsa_n_components
    if param_name == 'lsa_n_components':
        # Use optimized function that fits TF-IDF once
        print("Using optimized LSA tuning (TF-IDF fitted once)...")
        tuning_df = tune_lsa_components_fast(
            transformer_class=transformer_class,
            base_config=base_config,
            lsa_values=param_values,
            X_fe=X_train,
            classifier=baseline_clf,
            cv=cv,
            X_cv=X_trainval,
            y_cv=y_trainval,
            scorer=scorer,
            fe_source_names=tune_kwargs.get('fe_source_names'),
            fe_categories=tune_kwargs.get('fe_categories'),
            cv_source_names=tune_kwargs.get('cv_source_names'),
            cv_categories=tune_kwargs.get('cv_categories'),
            random_state=random_state,
        )
    else:
        tuning_df = tune_feature_transformer(**tune_kwargs)

    print("\n" + "=" * 70)

    if tuning_df is None:
        return None, None, None

    # Analyze tuning results
    default_value = base_config.get(param_name)
    optimal_param_value, best_tuned_f2 = analyze_tuning_results(
        tuning_df=tuning_df,
        param_name=param_name,
        default_value=default_value,
    )

    # Plot the tuning results
    # Determine prefix based on transformer class name
    prefix = 'fp' if 'FP' in transformer_class.__name__ else 'ep'
    plot_tuning_results(
        tuning_df=tuning_df,
        param_name=param_name,
        param_values=param_values,
        optimal_param_value=optimal_param_value,
        method_name=best_fe,
        description=description,
        save_path=images_dir / f'{prefix}_tuning_{best_fe}.png',
    )

    return tuning_df, optimal_param_value, best_tuned_f2


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
