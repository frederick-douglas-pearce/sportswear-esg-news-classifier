"""EP3 notebook utilities for model evaluation and deployment."""

from .deployment import (
    create_deployment_pipeline,
    save_deployment_artifacts,
    validate_pipeline,
    validate_pipeline_with_articles,
    load_deployment_artifacts,
)
from .threshold_optimization import (
    analyze_threshold_tradeoffs,
    find_optimal_threshold,
    plot_threshold_analysis,
)
# Import explainability utilities from fp3_nb (shared module)
from src.fp3_nb.explainability import (
    TextExplainer,
    LIMEExplanation,
    FeatureGroupImportance,
    PrototypeExplanation,
    get_fp_feature_groups,
    get_ep_feature_groups,
    explain_prediction,
)

__all__ = [
    # Deployment
    'create_deployment_pipeline',
    'save_deployment_artifacts',
    'validate_pipeline',
    'validate_pipeline_with_articles',
    'load_deployment_artifacts',
    # Threshold optimization
    'analyze_threshold_tradeoffs',
    'find_optimal_threshold',
    'plot_threshold_analysis',
    # Explainability (from fp3_nb)
    'TextExplainer',
    'LIMEExplanation',
    'FeatureGroupImportance',
    'PrototypeExplanation',
    'get_fp_feature_groups',
    'get_ep_feature_groups',
    'explain_prediction',
]
