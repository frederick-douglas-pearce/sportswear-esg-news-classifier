"""Deduplication Analysis Notebook utilities."""

from src.dedup1_nb.data_utils import (
    load_articles_from_db,
    create_article_dataframe,
    prepare_text_representations,
)
from src.dedup1_nb.embedders import (
    SentenceTransformerEmbedder,
    TfidfLsaEmbedder,
    get_embedder,
)
from src.dedup1_nb.clustering import (
    greedy_cosine_clustering,
    hdbscan_clustering,
    agglomerative_clustering,
    get_clustering_method,
)
from src.dedup1_nb.evaluation import (
    calculate_cluster_metrics,
    evaluate_on_labeled_pairs,
    compute_scorecard_stability,
    create_validation_pairs,
    threshold_sweep,
    split_validation_pairs,
    find_optimal_threshold_with_recall_constraint,
    score_pairs_with_cross_encoder,
    evaluate_cross_encoder_dedup,
    cross_encoder_threshold_sweep,
)
from src.dedup1_nb.visualization import (
    plot_threshold_analysis,
    plot_embedding_space,
    plot_cluster_sizes,
    plot_method_comparison,
    plot_similarity_distribution,
)

__all__ = [
    # Data utilities
    "load_articles_from_db",
    "create_article_dataframe",
    "prepare_text_representations",
    # Embedders
    "SentenceTransformerEmbedder",
    "TfidfLsaEmbedder",
    "get_embedder",
    # Clustering
    "greedy_cosine_clustering",
    "hdbscan_clustering",
    "agglomerative_clustering",
    "get_clustering_method",
    # Evaluation
    "calculate_cluster_metrics",
    "evaluate_on_labeled_pairs",
    "compute_scorecard_stability",
    "create_validation_pairs",
    "threshold_sweep",
    "split_validation_pairs",
    "find_optimal_threshold_with_recall_constraint",
    "score_pairs_with_cross_encoder",
    "evaluate_cross_encoder_dedup",
    "cross_encoder_threshold_sweep",
    # Visualization
    "plot_threshold_analysis",
    "plot_embedding_space",
    "plot_cluster_sizes",
    "plot_method_comparison",
    "plot_similarity_distribution",
]
