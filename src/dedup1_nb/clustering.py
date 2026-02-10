"""Clustering algorithms for article deduplication."""

from typing import Callable, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def greedy_cosine_clustering(
    embeddings: np.ndarray,
    threshold: float = 0.75,
    article_labels: Optional[list[frozenset]] = None,
    require_label_match: bool = False,
) -> tuple[list[int], dict]:
    """Greedy clustering based on cosine similarity threshold.

    This is the current production algorithm. It iterates through articles
    in order, keeping each article that isn't similar enough to any
    previously kept article.

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        threshold: Cosine similarity above which articles are considered duplicates
        article_labels: Optional list of label sets (frozenset of (brand, category) tuples)
                       for each article. If provided with require_label_match=True,
                       articles are only considered duplicates if they have identical labels.
        require_label_match: If True and article_labels provided, require identical labels
                            for deduplication (Option A label matching)

    Returns:
        Tuple of:
        - List of indices of articles to keep (representatives)
        - Dict with clustering metadata (cluster_labels, n_clusters, etc.)
    """
    n_articles = len(embeddings)
    if n_articles == 0:
        return [], {"cluster_labels": [], "n_clusters": 0, "duplicates_removed": 0}

    if n_articles == 1:
        return [0], {"cluster_labels": [0], "n_clusters": 1, "duplicates_removed": 0}

    # Compute pairwise similarities
    sim_matrix = cosine_similarity(embeddings)

    # Greedy clustering
    keep_indices = []
    seen = set()
    cluster_labels = [-1] * n_articles  # -1 means "duplicate of another"
    label_mismatch_preserved = 0

    cluster_id = 0
    for i in range(n_articles):
        if i in seen:
            continue

        # This article becomes a cluster representative
        keep_indices.append(i)
        cluster_labels[i] = cluster_id

        # Mark all similar articles as duplicates (assign to same cluster)
        for j in range(i + 1, n_articles):
            if j in seen:
                continue
            if sim_matrix[i, j] >= threshold:
                # Check label matching if required
                if require_label_match and article_labels:
                    if article_labels[i] != article_labels[j]:
                        # Different labels - not a duplicate, preserve this article
                        label_mismatch_preserved += 1
                        continue
                seen.add(j)
                cluster_labels[j] = cluster_id

        cluster_id += 1

    duplicates_removed = n_articles - len(keep_indices)

    metadata = {
        "cluster_labels": cluster_labels,
        "n_clusters": cluster_id,
        "duplicates_removed": duplicates_removed,
        "threshold": threshold,
        "method": "greedy_cosine",
        "require_label_match": require_label_match,
        "label_mismatch_preserved": label_mismatch_preserved,
    }

    return keep_indices, metadata


def hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "cosine",
) -> tuple[list[int], dict]:
    """HDBSCAN clustering for duplicate detection.

    HDBSCAN is a density-based algorithm that doesn't require specifying
    the number of clusters. It can identify noise points (articles that
    don't belong to any cluster).

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood for core points
        cluster_selection_epsilon: Distance threshold for cluster merging
        metric: Distance metric ('cosine', 'euclidean', etc.)

    Returns:
        Tuple of:
        - List of indices of representative articles (one per cluster + noise)
        - Dict with clustering metadata
    """
    import hdbscan
    from sklearn.metrics.pairwise import cosine_distances

    n_articles = len(embeddings)
    if n_articles == 0:
        return [], {"cluster_labels": [], "n_clusters": 0, "duplicates_removed": 0}

    if n_articles < min_cluster_size:
        # Not enough articles to form a cluster, keep all
        return list(range(n_articles)), {
            "cluster_labels": list(range(n_articles)),
            "n_clusters": n_articles,
            "duplicates_removed": 0,
        }

    # For cosine metric, use precomputed distance matrix
    # (HDBSCAN's BallTree doesn't support cosine for high-dim data)
    if metric == "cosine":
        distance_matrix = cosine_distances(embeddings).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="precomputed",
        )
        cluster_labels = clusterer.fit_predict(distance_matrix)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
        )
        cluster_labels = clusterer.fit_predict(embeddings)

    # Select representatives: first article in each cluster, plus all noise (-1)
    keep_indices = []
    seen_clusters = set()

    for i, label in enumerate(cluster_labels):
        if label == -1:
            # Noise point - keep it (unique article)
            keep_indices.append(i)
        elif label not in seen_clusters:
            # First article in this cluster - make it the representative
            keep_indices.append(i)
            seen_clusters.add(label)

    # Count actual clusters (excluding noise)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = sum(1 for l in cluster_labels if l == -1)
    duplicates_removed = n_articles - len(keep_indices)

    metadata = {
        "cluster_labels": cluster_labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "duplicates_removed": duplicates_removed,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "method": "hdbscan",
    }

    return keep_indices, metadata


def agglomerative_clustering(
    embeddings: np.ndarray,
    distance_threshold: float = 0.25,
    linkage: str = "average",
    metric: str = "cosine",
) -> tuple[list[int], dict]:
    """Agglomerative (hierarchical) clustering for duplicate detection.

    Uses bottom-up clustering with a distance threshold to form clusters.
    Distance threshold of 0.25 with cosine metric corresponds to
    similarity threshold of 0.75.

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        distance_threshold: Maximum distance for merging clusters
                           (1 - similarity for cosine metric)
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        metric: Distance metric ('cosine', 'euclidean')

    Returns:
        Tuple of:
        - List of indices of representative articles (one per cluster)
        - Dict with clustering metadata
    """
    from sklearn.cluster import AgglomerativeClustering

    n_articles = len(embeddings)
    if n_articles == 0:
        return [], {"cluster_labels": [], "n_clusters": 0, "duplicates_removed": 0}

    if n_articles == 1:
        return [0], {"cluster_labels": [0], "n_clusters": 1, "duplicates_removed": 0}

    # Ward linkage requires euclidean metric
    if linkage == "ward":
        metric = "euclidean"

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric=metric if linkage != "ward" else None,
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # Select representatives: first article in each cluster
    keep_indices = []
    seen_clusters = set()

    for i, label in enumerate(cluster_labels):
        if label not in seen_clusters:
            keep_indices.append(i)
            seen_clusters.add(label)

    n_clusters = len(set(cluster_labels))
    duplicates_removed = n_articles - len(keep_indices)

    metadata = {
        "cluster_labels": cluster_labels.tolist(),
        "n_clusters": n_clusters,
        "duplicates_removed": duplicates_removed,
        "distance_threshold": distance_threshold,
        "linkage": linkage,
        "metric": metric,
        "method": "agglomerative",
    }

    return keep_indices, metadata


def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    cluster_ratio: float = 0.8,
    random_state: int = 42,
) -> tuple[list[int], dict]:
    """K-Means clustering for duplicate detection.

    Note: K-Means requires specifying the number of clusters, which is
    typically unknown for deduplication. Use n_clusters or cluster_ratio
    to estimate.

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        n_clusters: Number of clusters (if None, use cluster_ratio)
        cluster_ratio: Fraction of articles to use as clusters if n_clusters not specified
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
        - List of indices of representative articles (nearest to centroids)
        - Dict with clustering metadata
    """
    from sklearn.cluster import KMeans

    n_articles = len(embeddings)
    if n_articles == 0:
        return [], {"cluster_labels": [], "n_clusters": 0, "duplicates_removed": 0}

    if n_articles == 1:
        return [0], {"cluster_labels": [0], "n_clusters": 1, "duplicates_removed": 0}

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = max(1, int(n_articles * cluster_ratio))

    n_clusters = min(n_clusters, n_articles)

    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = clusterer.fit_predict(embeddings)
    centroids = clusterer.cluster_centers_

    # Select representatives: article nearest to each centroid
    keep_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        # Find article nearest to centroid
        cluster_embeddings = embeddings[cluster_mask]
        centroid = centroids[cluster_id].reshape(1, -1)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        nearest_idx = cluster_indices[np.argmin(distances)]
        keep_indices.append(nearest_idx)

    duplicates_removed = n_articles - len(keep_indices)

    metadata = {
        "cluster_labels": cluster_labels.tolist(),
        "n_clusters": n_clusters,
        "duplicates_removed": duplicates_removed,
        "cluster_ratio": cluster_ratio,
        "method": "kmeans",
    }

    return sorted(keep_indices), metadata


def get_clustering_method(
    method: str = "greedy",
) -> Callable:
    """Factory function to get a clustering method by name.

    Args:
        method: Clustering method name ('greedy', 'hdbscan', 'agglomerative', 'kmeans')

    Returns:
        Clustering function
    """
    methods = {
        "greedy": greedy_cosine_clustering,
        "greedy_cosine": greedy_cosine_clustering,
        "hdbscan": hdbscan_clustering,
        "agglomerative": agglomerative_clustering,
        "hierarchical": agglomerative_clustering,
        "kmeans": kmeans_clustering,
    }

    if method not in methods:
        raise ValueError(f"Unknown clustering method: {method}. Options: {list(methods.keys())}")

    return methods[method]
