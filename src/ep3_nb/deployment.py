"""Deployment utilities for exporting classifier pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline


def create_deployment_pipeline(
    transformer: Any,
    classifier: Any,
    pipeline_name: str = 'ep_classifier'
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
    pipeline_name: str = 'ep_classifier'
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
            label = "Has ESG" if pred == 1 else "No ESG"
            text_preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  [{prob:.4f}] {label}: {text_preview}")

        print("-" * 60)
        print("Validation PASSED")
        print("=" * 60)

    return results


def validate_pipeline_with_articles(
    pipeline: Pipeline,
    articles: List[Dict[str, Any]],
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """Validate pipeline with structured article data including metadata.

    Each article should have: title, content, brands, source_name, category,
    and optionally expected_label for verification.

    Args:
        pipeline: Fitted sklearn Pipeline
        articles: List of article dicts with metadata
        threshold: Classification threshold
        verbose: Whether to print results

    Returns:
        Dictionary with validation results including accuracy if expected labels provided
    """
    from src.ep1_nb.preprocessing import create_text_features, clean_text
    import pandas as pd

    # Convert articles to DataFrame for create_text_features
    df = pd.DataFrame(articles)

    # Create text features with metadata (same as training)
    texts = create_text_features(
        df,
        text_col='content',
        title_col='title',
        brands_col='brands',
        source_name_col='source_name',
        category_col='category',
        include_metadata=True,
        clean_func=clean_text
    ).tolist()

    # Test predict
    try:
        predictions = pipeline.predict(texts)
        probabilities = pipeline.predict_proba(texts)[:, 1]
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }

    # Apply threshold
    predictions_at_threshold = (probabilities >= threshold).astype(int)

    # Check accuracy if expected labels provided
    expected_labels = [a.get('expected_label') for a in articles]
    has_labels = all(label is not None for label in expected_labels)

    if has_labels:
        correct = sum(1 for pred, exp in zip(predictions_at_threshold, expected_labels) if pred == exp)
        accuracy = correct / len(articles)
    else:
        accuracy = None

    results = {
        'success': True,
        'n_samples': len(articles),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'predictions_at_threshold': predictions_at_threshold.tolist(),
        'threshold': threshold,
        'accuracy': accuracy,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE VALIDATION WITH ARTICLES")
        print("=" * 80)
        print(f"\nThreshold: {threshold:.4f}")
        print(f"Samples tested: {len(articles)}")
        if accuracy is not None:
            print(f"Accuracy: {accuracy:.1%} ({correct}/{len(articles)})")
        print("\nResults:")
        print("-" * 80)

        for i, (article, prob, pred) in enumerate(zip(articles, probabilities, predictions_at_threshold)):
            pred_label = "Has ESG" if pred == 1 else "No ESG"
            expected = article.get('expected_label')

            # Format expected vs actual
            if expected is not None:
                expected_str = "Has ESG" if expected == 1 else "No ESG"
                match = "+" if pred == expected else "X"
            else:
                expected_str = "?"
                match = " "

            # Article info
            title = article.get('title', '')[:50]
            source = article.get('source_name', 'Unknown')
            brands = article.get('brands', [])
            brands_str = ', '.join(brands) if brands else 'None'

            print(f"\n[{i+1}] {match} Prob: {prob:.3f} | Pred: {pred_label} | Expected: {expected_str}")
            print(f"    Title: {title}...")
            print(f"    Source: {source} | Brands: {brands_str}")

        print("\n" + "-" * 80)
        if accuracy is not None and accuracy == 1.0:
            print("Validation PASSED - All predictions match expected labels")
        elif accuracy is not None:
            print(f"Validation COMPLETED - {accuracy:.1%} accuracy")
        else:
            print("Validation COMPLETED - No expected labels to verify")
        print("=" * 80)

    return results


def load_deployment_artifacts(
    models_dir: Path,
    pipeline_name: str = 'ep_classifier'
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
