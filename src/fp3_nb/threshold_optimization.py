"""Threshold optimization utilities for model evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.98
) -> Tuple[float, Dict[str, Any]]:
    """Find optimal classification threshold for target recall.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        target_recall: Minimum recall to achieve

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold that achieves target recall
    idx = np.where(recalls >= target_recall)[0]
    if len(idx) == 0:
        print(f"Warning: Cannot achieve target recall of {target_recall}")
        optimal_threshold = 0.5
        actual_recall = recalls[np.argmin(np.abs(thresholds - 0.5))]
        precision_at_threshold = precisions[np.argmin(np.abs(thresholds - 0.5))]
    else:
        # Take the highest threshold that still achieves target recall
        best_idx = idx[-1]
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
        actual_recall = recalls[best_idx]
        precision_at_threshold = precisions[best_idx]

    # Calculate metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    f2_score = fbeta_score(y_true, y_pred, beta=2)

    # Confusion matrix values
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    metrics = {
        'threshold': optimal_threshold,
        'target_recall': target_recall,
        'actual_recall': actual_recall,
        'precision': precision_at_threshold,
        'f2_score': f2_score,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }

    return optimal_threshold, metrics


def analyze_threshold_tradeoffs(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recalls: List[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Analyze precision-recall trade-offs at different thresholds.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        target_recalls: List of target recall values to analyze
        verbose: Whether to print analysis

    Returns:
        DataFrame with threshold analysis
    """
    if target_recalls is None:
        target_recalls = [0.90, 0.95, 0.97, 0.98, 0.99]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    total_positives = y_true.sum()
    total_negatives = (y_true == 0).sum()

    results = []

    for target in target_recalls:
        idx = np.where(recalls >= target)[0]
        if len(idx) > 0:
            best_idx = idx[-1]
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
            actual_recall = recalls[best_idx]
            precision = precisions[best_idx]

            # Calculate FP and FN at this threshold
            y_pred = (y_proba >= threshold).astype(int)
            fp_count = ((y_pred == 1) & (y_true == 0)).sum()
            fn_count = ((y_pred == 0) & (y_true == 1)).sum()
            f2 = fbeta_score(y_true, y_pred, beta=2)

            results.append({
                'target_recall': target,
                'threshold': threshold,
                'actual_recall': actual_recall,
                'precision': precision,
                'f2_score': f2,
                'fp_passed': fp_count,
                'fn_missed': fn_count,
                'fp_rate': fp_count / total_negatives,
                'fn_rate': fn_count / total_positives,
            })

    df = pd.DataFrame(results)

    if verbose:
        print("=" * 80)
        print("THRESHOLD ANALYSIS: Recall vs Precision Trade-off")
        print("=" * 80)
        print(f"\nTotal positives: {total_positives}, Total negatives: {total_negatives}")
        print("\n" + "-" * 80)
        print(f"{'Target':>8} | {'Threshold':>9} | {'Recall':>8} | {'Precision':>9} | "
              f"{'F2':>6} | {'FPs':>6} | {'FNs':>6}")
        print("-" * 80)

        for _, row in df.iterrows():
            print(f"{row['target_recall']:>7.0%} | {row['threshold']:>9.4f} | "
                  f"{row['actual_recall']:>7.1%} | {row['precision']:>8.1%} | "
                  f"{row['f2_score']:>6.4f} | {row['fp_passed']:>6.0f} | {row['fn_missed']:>6.0f}")

        print("-" * 80)
        print("\nLower threshold = Higher recall but more FPs pass to downstream processing")
        print("=" * 80)

    return df


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recalls: List[float] = None,
    optimal_threshold: Optional[float] = None,
    title: str = 'Precision-Recall Trade-off',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot precision-recall curve with threshold markers.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        target_recalls: Target recall values to mark on curve
        optimal_threshold: Selected optimal threshold to highlight
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    if target_recalls is None:
        target_recalls = [0.95, 0.98]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: PR curve with markers
    ax1 = axes[0]
    ax1.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')

    # Mark target recall points
    colors = plt.cm.Set1(np.linspace(0, 1, len(target_recalls)))
    for i, target in enumerate(target_recalls):
        idx = np.where(recalls >= target)[0]
        if len(idx) > 0:
            best_idx = idx[-1]
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
            ax1.scatter(recalls[best_idx], precisions[best_idx], s=120, c=[colors[i]],
                        zorder=5, label=f'Recall >= {target:.0%} (t={threshold:.2f})')

    # Mark default threshold (0.5)
    if len(thresholds) > 0:
        default_idx = np.argmin(np.abs(thresholds - 0.5))
        ax1.scatter(recalls[default_idx], precisions[default_idx], s=120, c='red',
                    marker='x', zorder=5, linewidths=3, label='Default (t=0.5)')

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.7, 1.01])
    ax1.set_ylim([0.5, 1.01])

    # Right plot: Threshold vs metrics
    ax2 = axes[1]

    # Create arrays for plotting
    # Note: thresholds has one less element than precisions/recalls
    plot_thresholds = thresholds
    plot_precisions = precisions[:-1]
    plot_recalls = recalls[:-1]

    # Calculate F2 at each threshold
    f2_scores = []
    for t in plot_thresholds:
        y_pred = (y_proba >= t).astype(int)
        f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
    f2_scores = np.array(f2_scores)

    ax2.plot(plot_thresholds, plot_recalls, 'b-', linewidth=2, label='Recall')
    ax2.plot(plot_thresholds, plot_precisions, 'g-', linewidth=2, label='Precision')
    ax2.plot(plot_thresholds, f2_scores, 'r--', linewidth=2, label='F2 Score')

    # Mark optimal threshold if provided
    if optimal_threshold is not None:
        ax2.axvline(x=optimal_threshold, color='purple', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Optimal (t={optimal_threshold:.3f})')

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Metrics vs Threshold', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig
