"""Visualization utilities for deduplication analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_threshold_analysis(
    sweep_results: list[dict],
    figsize: tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot threshold sweep analysis showing reduction vs quality.

    Args:
        sweep_results: Results from threshold_sweep()
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    df = pd.DataFrame(sweep_results)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Reduction rate vs threshold
    ax1 = axes[0]
    ax1.plot(df["threshold"], df["reduction_rate"] * 100, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Similarity Threshold", fontsize=11)
    ax1.set_ylabel("Duplicates Removed (%)", fontsize=11)
    ax1.set_title("Deduplication Rate vs Threshold", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.45, 1.0)

    # Add current production threshold marker
    current_threshold = 0.75
    ax1.axvline(x=current_threshold, color="red", linestyle="--", alpha=0.7, label=f"Current ({current_threshold})")
    ax1.legend()

    # Plot 2: Number of clusters vs threshold
    ax2 = axes[1]
    ax2.plot(df["threshold"], df["n_clusters"], "o-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("Similarity Threshold", fontsize=11)
    ax2.set_ylabel("Number of Clusters", fontsize=11)
    ax2.set_title("Cluster Count vs Threshold", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.45, 1.0)
    ax2.axvline(x=current_threshold, color="red", linestyle="--", alpha=0.7)

    # Plot 3: Silhouette score vs threshold (if available)
    ax3 = axes[2]
    if "silhouette_score" in df.columns and df["silhouette_score"].notna().any():
        valid_df = df[df["silhouette_score"].notna()]
        ax3.plot(
            valid_df["threshold"],
            valid_df["silhouette_score"],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax3.set_ylabel("Silhouette Score", fontsize=11)
        ax3.set_title("Cluster Quality vs Threshold", fontsize=12, fontweight="bold")
    else:
        # If no silhouette, show articles kept
        ax3.plot(df["threshold"], df["n_kept"], "o-", linewidth=2, markersize=8, color="orange")
        ax3.set_ylabel("Articles Kept", fontsize=11)
        ax3.set_title("Articles Retained vs Threshold", fontsize=12, fontweight="bold")

    ax3.set_xlabel("Similarity Threshold", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.45, 1.0)
    ax3.axvline(x=current_threshold, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_embedding_space(
    embeddings: np.ndarray,
    cluster_labels: list[int],
    method: str = "umap",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    highlight_indices: Optional[list[int]] = None,
) -> None:
    """Plot 2D projection of embedding space with cluster colors.

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        cluster_labels: Cluster assignment for each article
        method: Dimensionality reduction method ('umap', 'tsne', 'pca')
        figsize: Figure size
        save_path: Optional path to save the figure
        title: Optional custom title
        highlight_indices: Optional indices to highlight as representatives
    """
    # Reduce to 2D
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            print("Warning: umap-learn not installed. Falling back to PCA.")
            print("Install with: pip install umap-learn")
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            method = "pca"  # Update method name for labels
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        coords = reducer.fit_transform(embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    labels = np.array(cluster_labels)
    unique_labels = sorted(set(labels))

    # Color palette
    if len(unique_labels) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Plot points by cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # Noise points
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c="gray",
                alpha=0.3,
                s=20,
                label="Noise",
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[i % len(colors)]],
                alpha=0.6,
                s=30,
                label=f"Cluster {label}" if len(unique_labels) <= 10 else None,
            )

    # Highlight representatives if provided
    if highlight_indices:
        ax.scatter(
            coords[highlight_indices, 0],
            coords[highlight_indices, 1],
            c="red",
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=1,
            label="Representatives",
            zorder=5,
        )

    ax.set_xlabel(f"{method.upper()} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontsize=11)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Article Embeddings ({method.upper()} projection)", fontsize=12, fontweight="bold")

    if len(unique_labels) <= 10:
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_cluster_sizes(
    cluster_labels: list[int],
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot cluster size distribution.

    Args:
        cluster_labels: Cluster assignment for each article
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    labels = np.array(cluster_labels)

    # Count cluster sizes (excluding noise)
    cluster_sizes = []
    for label in set(labels):
        if label >= 0:
            size = sum(1 for l in labels if l == label)
            cluster_sizes.append(size)

    cluster_sizes = sorted(cluster_sizes, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Cluster size distribution
    ax1 = axes[0]
    ax1.bar(range(len(cluster_sizes)), cluster_sizes, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Cluster Rank", fontsize=11)
    ax1.set_ylabel("Cluster Size", fontsize=11)
    ax1.set_title("Cluster Sizes (Sorted)", fontsize=12, fontweight="bold")
    ax1.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Singleton")
    ax1.legend()

    # Plot 2: Histogram of cluster sizes
    ax2 = axes[1]
    max_size = max(cluster_sizes) if cluster_sizes else 1
    bins = range(1, min(max_size + 2, 20))
    ax2.hist(cluster_sizes, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Cluster Size", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Cluster Size Histogram", fontsize=12, fontweight="bold")

    # Add statistics
    n_singleton = sum(1 for s in cluster_sizes if s == 1)
    n_clusters = len(cluster_sizes)
    stats_text = f"Clusters: {n_clusters}\nSingletons: {n_singleton} ({n_singleton / n_clusters * 100:.1f}%)"
    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_method_comparison(
    results: list[dict],
    metric: str = "f1",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison of different deduplication methods.

    Args:
        results: List of dicts with 'method', 'embedder', and metrics
        metric: Primary metric to compare ('f1', 'precision', 'recall', 'silhouette_score')
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    df = pd.DataFrame(results)

    # Create method labels
    df["label"] = df.apply(
        lambda r: f"{r.get('embedder', 'unknown')}\n{r.get('method', 'unknown')}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(df))
    values = df[metric].fillna(0)

    bars = ax.bar(x, values, color="steelblue", alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Deduplication Method Comparison ({metric})", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_similarity_distribution(
    embeddings: np.ndarray,
    labeled_pairs: Optional[list[dict]] = None,
    article_ids: Optional[list[str]] = None,
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """Plot distribution of pairwise similarities.

    Args:
        embeddings: Embedding matrix
        labeled_pairs: Optional labeled pairs to overlay
        article_ids: Article IDs (required if labeled_pairs provided)
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)

    # Get upper triangle (excluding diagonal)
    upper_tri = np.triu_indices_from(sim_matrix, k=1)
    all_similarities = sim_matrix[upper_tri]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Overall similarity distribution
    ax1 = axes[0]
    ax1.hist(all_similarities, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(x=0.75, color="red", linestyle="--", alpha=0.7, label="Current threshold (0.75)")
    ax1.set_xlabel("Cosine Similarity", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Pairwise Similarity Distribution", fontsize=12, fontweight="bold")
    ax1.legend()

    # Plot 2: Cumulative distribution
    ax2 = axes[1]
    sorted_sims = np.sort(all_similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax2.plot(sorted_sims, cumulative, color="steelblue", linewidth=2)
    ax2.axvline(x=0.75, color="red", linestyle="--", alpha=0.7, label="Current threshold (0.75)")

    # Add percentage markers
    for thresh in [0.7, 0.75, 0.8, 0.85]:
        pct_above = (all_similarities >= thresh).mean() * 100
        ax2.axhline(y=1 - pct_above / 100, color="gray", linestyle=":", alpha=0.5)
        ax2.text(0.52, 1 - pct_above / 100 + 0.02, f"{pct_above:.1f}% >= {thresh}", fontsize=8)

    ax2.set_xlabel("Cosine Similarity", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.set_title("Cumulative Similarity Distribution", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()

    # Print summary statistics
    print("\nSimilarity Statistics:")
    print(f"  Mean: {all_similarities.mean():.4f}")
    print(f"  Std:  {all_similarities.std():.4f}")
    print(f"  Min:  {all_similarities.min():.4f}")
    print(f"  Max:  {all_similarities.max():.4f}")
    print(f"  Pairs >= 0.75: {(all_similarities >= 0.75).sum():,} ({(all_similarities >= 0.75).mean() * 100:.2f}%)")
    print(f"  Pairs >= 0.80: {(all_similarities >= 0.80).sum():,} ({(all_similarities >= 0.80).mean() * 100:.2f}%)")
