"""FP Classifier Notebook utilities."""

from src.fp1_nb.data_utils import (
    load_jsonl_data,
    analyze_target_stats,
    plot_target_distribution,
    split_train_val_test,
)
from src.fp1_nb.eda_utils import (
    analyze_text_length_stats,
    plot_text_length_distributions,
    analyze_brand_distribution,
    plot_brand_distribution,
    analyze_word_frequencies,
    plot_word_cloud,
)
from src.fp1_nb.preprocessing import (
    clean_text,
    create_text_features,
    build_tfidf_pipeline,
)
from src.fp1_nb.modeling import (
    create_search_object,
    tune_with_logging,
    extract_cv_metrics,
    get_best_params_summary,
    compare_models,
    get_best_model,
    evaluate_model,
    compare_val_test_performance,
)

__all__ = [
    # Data utilities
    "load_jsonl_data",
    "analyze_target_stats",
    "plot_target_distribution",
    "split_train_val_test",
    # EDA utilities
    "analyze_text_length_stats",
    "plot_text_length_distributions",
    "analyze_brand_distribution",
    "plot_brand_distribution",
    "analyze_word_frequencies",
    "plot_word_cloud",
    # Preprocessing
    "clean_text",
    "create_text_features",
    "build_tfidf_pipeline",
    # Modeling
    "create_search_object",
    "tune_with_logging",
    "extract_cv_metrics",
    "get_best_params_summary",
    "compare_models",
    "get_best_model",
    "evaluate_model",
    "compare_val_test_performance",
]
