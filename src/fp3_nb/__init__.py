"""FP3 notebook utilities for model evaluation and deployment."""

from .deployment import (
    create_deployment_pipeline,
    save_deployment_artifacts,
    validate_pipeline,
)
from .threshold_optimization import (
    analyze_threshold_tradeoffs,
    find_optimal_threshold,
    plot_threshold_analysis,
)
from .explainability import (
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
    # Threshold optimization
    'analyze_threshold_tradeoffs',
    'find_optimal_threshold',
    'plot_threshold_analysis',
    # Explainability
    'TextExplainer',
    'LIMEExplanation',
    'FeatureGroupImportance',
    'PrototypeExplanation',
    'get_fp_feature_groups',
    'get_ep_feature_groups',
    'explain_prediction',
]
