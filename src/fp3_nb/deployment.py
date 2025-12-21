"""Deployment utilities for exporting classifier pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline


def create_deployment_pipeline(
    transformer: Any,
    classifier: Any,
    pipeline_name: str = 'fp_classifier'
) -> Pipeline:
    """Create sklearn Pipeline for deployment.

    Args:
        transformer: Fitted feature transformer
        classifier: Fitted classifier
        pipeline_name: Name for the pipeline

    Returns:
        Sklearn Pipeline with transformer and classifier
    """
    pipeline = Pipeline([
        ('features', transformer),
        ('classifier', classifier)
    ])

    # Store pipeline name as attribute
    pipeline._pipeline_name = pipeline_name

    return pipeline


def save_deployment_artifacts(
    pipeline: Pipeline,
    config: Dict[str, Any],
    models_dir: Path,
    pipeline_name: str = 'fp_classifier'
) -> Dict[str, Path]:
    """Save pipeline and configuration for deployment.

    Args:
        pipeline: Fitted sklearn Pipeline
        config: Configuration dictionary with threshold, metrics, etc.
        models_dir: Directory to save artifacts
        pipeline_name: Base name for saved files

    Returns:
        Dictionary with paths to saved artifacts
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    pipeline_path = models_dir / f'{pipeline_name}_pipeline.joblib'
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to: {pipeline_path}")

    # Save configuration
    # Convert numpy types to Python types for JSON serialization
    config_serializable = _make_json_serializable(config)

    config_path = models_dir / f'{pipeline_name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    return {
        'pipeline': pipeline_path,
        'config': config_path,
    }


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def validate_pipeline(
    pipeline: Pipeline,
    test_texts: List[str],
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate pipeline works correctly on sample texts.

    Args:
        pipeline: Fitted sklearn Pipeline
        test_texts: List of sample texts to test
        threshold: Classification threshold
        verbose: Whether to print results

    Returns:
        Dictionary with validation results
    """
    # Test predict
    try:
        predictions = pipeline.predict(test_texts)
        probabilities = pipeline.predict_proba(test_texts)[:, 1]
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

    # Apply threshold
    predictions_at_threshold = (probabilities >= threshold).astype(int)

    results = {
        'success': True,
        'n_samples': len(test_texts),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'predictions_at_threshold': predictions_at_threshold.tolist(),
        'threshold': threshold,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION")
        print("=" * 60)
        print(f"\nThreshold: {threshold}")
        print(f"Samples tested: {len(test_texts)}")
        print("\nResults:")
        print("-" * 60)

        for i, (text, prob, pred) in enumerate(zip(test_texts, probabilities, predictions_at_threshold)):
            label = "Sportswear" if pred == 1 else "False Positive"
            text_preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  [{prob:.4f}] {label}: {text_preview}")

        print("-" * 60)
        print("Validation PASSED")
        print("=" * 60)

    return results


def load_deployment_artifacts(
    models_dir: Path,
    pipeline_name: str = 'fp_classifier'
) -> Dict[str, Any]:
    """Load pipeline and configuration for deployment.

    Args:
        models_dir: Directory containing saved artifacts
        pipeline_name: Base name for saved files

    Returns:
        Dictionary with 'pipeline' and 'config' keys
    """
    models_dir = Path(models_dir)

    pipeline_path = models_dir / f'{pipeline_name}_pipeline.joblib'
    config_path = models_dir / f'{pipeline_name}_config.json'

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline = joblib.load(pipeline_path)
    with open(config_path) as f:
        config = json.load(f)

    print(f"Loaded pipeline from: {pipeline_path}")
    print(f"Loaded config from: {config_path}")

    return {
        'pipeline': pipeline,
        'config': config,
    }
