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

__all__ = [
    'analyze_threshold_tradeoffs',
    'create_deployment_pipeline',
    'find_optimal_threshold',
    'load_deployment_artifacts',
    'plot_threshold_analysis',
    'save_deployment_artifacts',
    'validate_pipeline',
    'validate_pipeline_with_articles',
]
