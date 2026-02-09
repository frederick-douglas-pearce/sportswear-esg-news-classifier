"""Evaluation metrics for deduplication analysis."""

from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def split_validation_pairs(
    labeled_pairs: list[dict],
    test_size: float = 0.3,
    random_state: int = 42,
    stratify_by_label: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Split labeled pairs into tuning and holdout sets.

    Args:
        labeled_pairs: List of labeled pair dicts with 'is_duplicate' field
        test_size: Fraction of pairs to use for holdout (default 0.3)
        random_state: Random seed for reproducibility
        stratify_by_label: If True, maintain duplicate/non-duplicate ratio in both sets

    Returns:
        Tuple of (tuning_pairs, holdout_pairs)
    """
    np.random.seed(random_state)

    # Filter to only labeled pairs
    labeled = [p for p in labeled_pairs if p.get("is_duplicate") is not None]

    if stratify_by_label:
        # Split by label to maintain ratio
        duplicates = [p for p in labeled if p["is_duplicate"]]
        non_duplicates = [p for p in labeled if not p["is_duplicate"]]

        # Shuffle each group
        np.random.shuffle(duplicates)
        np.random.shuffle(non_duplicates)

        # Split each group
        n_dup_test = max(1, int(len(duplicates) * test_size))
        n_nondup_test = max(1, int(len(non_duplicates) * test_size))

        holdout = duplicates[:n_dup_test] + non_duplicates[:n_nondup_test]
        tuning = duplicates[n_dup_test:] + non_duplicates[n_nondup_test:]
    else:
        # Simple random split
        indices = np.random.permutation(len(labeled))
        n_test = int(len(labeled) * test_size)

        holdout = [labeled[i] for i in indices[:n_test]]
        tuning = [labeled[i] for i in indices[n_test:]]

    return tuning, holdout


def find_optimal_threshold_with_recall_constraint(
    results: list[dict],
    min_recall: float = 0.98,
    recall_key: str = "pair_recall",
    precision_key: str = "pair_precision",
) -> dict:
    """Find optimal config that maximizes precision while meeting recall constraint.

    Args:
        results: List of result dicts from threshold sweep or clustering comparison
        min_recall: Minimum required recall (default 0.98)
        recall_key: Key for recall metric in results
        precision_key: Key for precision metric in results

    Returns:
        Dict with optimal config and metrics, or best recall config if constraint not met
    """
    # Filter configs that meet recall constraint
    valid_configs = [r for r in results if r.get(recall_key, 0) >= min_recall]

    if valid_configs:
        # Among valid configs, find one with highest precision
        best = max(valid_configs, key=lambda x: x.get(precision_key, 0))
        return {
            "config": best,
            "meets_recall_constraint": True,
            "min_recall_target": min_recall,
            "achieved_recall": best.get(recall_key),
            "achieved_precision": best.get(precision_key),
        }
    else:
        # No config meets constraint - return best recall
        best_recall = max(results, key=lambda x: x.get(recall_key, 0))
        return {
            "config": best_recall,
            "meets_recall_constraint": False,
            "min_recall_target": min_recall,
            "achieved_recall": best_recall.get(recall_key),
            "achieved_precision": best_recall.get(precision_key),
            "note": f"No config achieved recall >= {min_recall}. Returning best recall.",
        }


def calculate_cluster_metrics(
    embeddings: np.ndarray,
    cluster_labels: list[int],
) -> dict:
    """Calculate cluster quality metrics (no ground truth required).

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        cluster_labels: Cluster assignment for each article

    Returns:
        Dict with silhouette_score, calinski_harabasz, cluster sizes, etc.
    """
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    labels = np.array(cluster_labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Handle edge cases
    if n_clusters < 2:
        return {
            "silhouette_score": None,
            "calinski_harabasz_score": None,
            "n_clusters": n_clusters,
            "cluster_sizes": [],
            "mean_cluster_size": None,
            "max_cluster_size": None,
            "min_cluster_size": None,
            "singleton_clusters": None,
        }

    # Filter out noise points (-1) for metrics
    valid_mask = labels >= 0
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]

    n_unique_labels = len(set(valid_labels))
    # Silhouette requires: 2 <= n_labels <= n_samples - 1
    if n_unique_labels < 2 or n_unique_labels >= len(valid_labels):
        silhouette = None
        calinski = None
    else:
        silhouette = silhouette_score(valid_embeddings, valid_labels)
        calinski = calinski_harabasz_score(valid_embeddings, valid_labels)

    # Cluster size statistics
    cluster_sizes = []
    for cluster_id in set(labels):
        if cluster_id >= 0:  # Exclude noise
            size = sum(1 for l in labels if l == cluster_id)
            cluster_sizes.append(size)

    cluster_sizes = sorted(cluster_sizes, reverse=True)
    singleton_clusters = sum(1 for s in cluster_sizes if s == 1)

    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else None,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else None,
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else None,
        "singleton_clusters": singleton_clusters,
        "singleton_ratio": singleton_clusters / n_clusters if n_clusters > 0 else None,
    }


def evaluate_on_labeled_pairs(
    embeddings: np.ndarray,
    article_ids: list[str],
    labeled_pairs: list[dict],
    cluster_labels: list[int],
) -> dict:
    """Evaluate clustering against human-labeled duplicate pairs.

    Args:
        embeddings: Embedding matrix (n_articles x embedding_dim)
        article_ids: List of article IDs matching embedding rows
        labeled_pairs: List of dicts with 'article_id_1', 'article_id_2', 'is_duplicate'
        cluster_labels: Cluster assignment for each article

    Returns:
        Dict with precision, recall, f1, and confusion matrix info
    """
    # Build article_id to index mapping
    id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    # Evaluate pairs
    true_positives = 0  # Labeled duplicate, same cluster
    false_positives = 0  # Labeled not duplicate, same cluster
    true_negatives = 0  # Labeled not duplicate, different cluster
    false_negatives = 0  # Labeled duplicate, different cluster

    evaluated_pairs = 0
    skipped_pairs = 0

    for pair in labeled_pairs:
        aid1 = pair.get("article_id_1")
        aid2 = pair.get("article_id_2")
        is_duplicate = pair.get("is_duplicate")

        # Skip pairs with missing data
        if is_duplicate is None:
            skipped_pairs += 1
            continue

        if aid1 not in id_to_idx or aid2 not in id_to_idx:
            skipped_pairs += 1
            continue

        idx1 = id_to_idx[aid1]
        idx2 = id_to_idx[aid2]

        # Check if articles are in same cluster
        same_cluster = cluster_labels[idx1] == cluster_labels[idx2]

        # Handle noise points (-1) - they're never in same cluster
        if cluster_labels[idx1] == -1 or cluster_labels[idx2] == -1:
            same_cluster = False

        evaluated_pairs += 1

        if is_duplicate and same_cluster:
            true_positives += 1
        elif is_duplicate and not same_cluster:
            false_negatives += 1
        elif not is_duplicate and same_cluster:
            false_positives += 1
        else:  # not duplicate and not same cluster
            true_negatives += 1

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate accuracy
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "evaluated_pairs": evaluated_pairs,
        "skipped_pairs": skipped_pairs,
    }


def compute_scorecard_stability(
    brand_scores_baseline: dict,
    brand_scores_alternative: dict,
    top_n: int = 3,
) -> dict:
    """Compute stability metrics between two scorecard configurations.

    Measures how much brand rankings change between configurations.

    Args:
        brand_scores_baseline: Baseline scorecard output from calculate_brand_scores()
        brand_scores_alternative: Alternative scorecard output
        top_n: Number of top/bottom brands to compare

    Returns:
        Dict with Spearman correlation, Jaccard similarity for top/bottom brands
    """
    from scipy.stats import spearmanr

    # Extract brand rankings
    baseline_scores = {b["brand"]: b["total_score"] for b in brand_scores_baseline.get("all_brand_scores", [])}
    alt_scores = {b["brand"]: b["total_score"] for b in brand_scores_alternative.get("all_brand_scores", [])}

    # Get common brands
    common_brands = set(baseline_scores.keys()) & set(alt_scores.keys())

    if len(common_brands) < 2:
        return {
            "spearman_correlation": None,
            "top_jaccard": None,
            "bottom_jaccard": None,
            "n_common_brands": len(common_brands),
        }

    # Calculate Spearman correlation on scores
    baseline_vals = [baseline_scores[b] for b in common_brands]
    alt_vals = [alt_scores[b] for b in common_brands]
    spearman_corr, _ = spearmanr(baseline_vals, alt_vals)

    # Compare top N brands
    baseline_top = set(b["brand"] for b in brand_scores_baseline.get("top_brands", [])[:top_n])
    alt_top = set(b["brand"] for b in brand_scores_alternative.get("top_brands", [])[:top_n])

    if baseline_top and alt_top:
        top_jaccard = len(baseline_top & alt_top) / len(baseline_top | alt_top)
    else:
        top_jaccard = 1.0 if baseline_top == alt_top else 0.0

    # Compare bottom N brands
    baseline_bottom = set(b["brand"] for b in brand_scores_baseline.get("bottom_brands", [])[:top_n])
    alt_bottom = set(b["brand"] for b in brand_scores_alternative.get("bottom_brands", [])[:top_n])

    if baseline_bottom and alt_bottom:
        bottom_jaccard = len(baseline_bottom & alt_bottom) / len(baseline_bottom | alt_bottom)
    else:
        bottom_jaccard = 1.0 if baseline_bottom == alt_bottom else 0.0

    return {
        "spearman_correlation": spearman_corr,
        "top_jaccard": top_jaccard,
        "bottom_jaccard": bottom_jaccard,
        "n_common_brands": len(common_brands),
        "baseline_top_brands": list(baseline_top),
        "alt_top_brands": list(alt_top),
        "baseline_bottom_brands": list(baseline_bottom),
        "alt_bottom_brands": list(alt_bottom),
    }


def create_validation_pairs(
    embeddings: np.ndarray,
    df,
    similarity_ranges: Optional[list[tuple[float, float]]] = None,
    n_pairs_per_range: int = 15,
    random_state: int = 42,
) -> list[dict]:
    """Create validation pairs sampled across similarity levels.

    Args:
        embeddings: Embedding matrix
        df: DataFrame with article data (must have title, source_name, article_id)
        similarity_ranges: List of (min_sim, max_sim) tuples.
                          Defaults to ranges covering 0.5-0.95
        n_pairs_per_range: Number of pairs to sample per range
        random_state: Random seed

    Returns:
        List of pair dicts ready for manual labeling
    """
    if similarity_ranges is None:
        # Default ranges that cover interesting similarity levels
        similarity_ranges = [
            (0.50, 0.60),  # Low similarity - likely not duplicates
            (0.60, 0.70),  # Medium-low
            (0.70, 0.75),  # Medium - borderline cases
            (0.75, 0.80),  # Current threshold region
            (0.80, 0.85),  # Medium-high
            (0.85, 0.90),  # High similarity - likely duplicates
            (0.90, 0.95),  # Very high - almost certainly duplicates
        ]

    np.random.seed(random_state)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    pairs = []
    n_articles = len(embeddings)

    for min_sim, max_sim in similarity_ranges:
        # Find all pairs in this similarity range
        candidates = []
        for i in range(n_articles):
            for j in range(i + 1, n_articles):
                sim = sim_matrix[i, j]
                if min_sim <= sim < max_sim:
                    candidates.append((i, j, sim))

        # Sample pairs
        if len(candidates) > n_pairs_per_range:
            indices = np.random.choice(len(candidates), n_pairs_per_range, replace=False)
            sampled = [candidates[idx] for idx in indices]
        else:
            sampled = candidates

        # Create pair records
        for i, j, sim in sampled:
            pairs.append(
                {
                    "article_id_1": str(df.iloc[i]["article_id"]),
                    "article_id_2": str(df.iloc[j]["article_id"]),
                    "title_1": df.iloc[i]["title"],
                    "title_2": df.iloc[j]["title"],
                    "source_1": df.iloc[i]["source_name"],
                    "source_2": df.iloc[j]["source_name"],
                    "similarity": float(sim),
                    "similarity_range": f"{min_sim:.2f}-{max_sim:.2f}",
                    "is_duplicate": None,  # To be filled by human labeler
                }
            )

    return pairs


def threshold_sweep(
    embeddings: np.ndarray,
    thresholds: Optional[list[float]] = None,
    labeled_pairs: Optional[list[dict]] = None,
    article_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Sweep through similarity thresholds and compute metrics.

    Args:
        embeddings: Embedding matrix
        thresholds: List of thresholds to test. Defaults to 0.5-0.95 in 0.05 steps
        labeled_pairs: Optional labeled pairs for precision/recall
        article_ids: Article IDs (required if labeled_pairs provided)

    Returns:
        List of dicts with threshold and metrics for each threshold
    """
    from src.dedup1_nb.clustering import greedy_cosine_clustering

    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results = []

    for threshold in thresholds:
        keep_indices, cluster_meta = greedy_cosine_clustering(embeddings, threshold=threshold)

        # Cluster quality metrics
        cluster_metrics = calculate_cluster_metrics(embeddings, cluster_meta["cluster_labels"])

        result = {
            "threshold": threshold,
            "n_kept": len(keep_indices),
            "n_removed": cluster_meta["duplicates_removed"],
            "n_clusters": cluster_meta["n_clusters"],
            "reduction_rate": cluster_meta["duplicates_removed"] / len(embeddings) if len(embeddings) > 0 else 0,
            **cluster_metrics,
        }

        # Evaluate on labeled pairs if available
        if labeled_pairs and article_ids:
            pair_metrics = evaluate_on_labeled_pairs(
                embeddings, article_ids, labeled_pairs, cluster_meta["cluster_labels"]
            )
            result.update({f"pair_{k}": v for k, v in pair_metrics.items()})

        results.append(result)

    return results
