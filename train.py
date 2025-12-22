"""Training script for FP Brand Classifier.

Retrains the False Positive Brand Classifier from scratch using the same
process as the notebooks but in script form.

Usage:
    uv run python train.py
    uv run python train.py --data-path data/fp_training_data.jsonl
    uv run python train.py --target-recall 0.98 --verbose

The script will:
1. Load training data from JSONL
2. Split into train/val/test sets (60/20/20)
3. Train feature transformer (sentence-transformer + NER)
4. Train Random Forest with best hyperparameters
5. Create sklearn Pipeline
6. Optimize threshold for target recall
7. Save pipeline and configuration to models/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src.deployment import (
    create_text_features,
    load_training_data,
    split_data,
)
from src.deployment.config import CONFIG_PATH, PIPELINE_PATH, TARGET_RECALL
from src.fp1_nb.feature_transformer import FPFeatureTransformer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the FP Brand Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/fp_training_data.jsonl",
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=TARGET_RECALL,
        help="Target recall for threshold optimization",
    )
    parser.add_argument(
        "--transformer-method",
        type=str,
        default="sentence_transformer_ner",
        choices=FPFeatureTransformer.VALID_METHODS,
        help="Feature transformation method",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    return parser.parse_args()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.98,
) -> Tuple[float, Dict[str, Any]]:
    """Find optimal classification threshold for target recall.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        target_recall: Minimum recall to achieve

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold that achieves target recall
    idx = np.where(recalls >= target_recall)[0]
    if len(idx) == 0:
        print(f"Warning: Cannot achieve target recall of {target_recall}")
        optimal_threshold = 0.5
        best_idx = np.argmin(np.abs(thresholds - 0.5))
    else:
        # Take the highest threshold that still achieves target recall
        best_idx = idx[-1]
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0

    actual_recall = recalls[best_idx]
    precision_at_threshold = precisions[best_idx]

    # Calculate metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    f2_score = fbeta_score(y_true, y_pred, beta=2)

    metrics = {
        "threshold": optimal_threshold,
        "threshold_recall": actual_recall,
        "threshold_precision": precision_at_threshold,
        "threshold_f2": f2_score,
    }

    return optimal_threshold, metrics


def train_model(
    data_path: str,
    output_dir: str = "models",
    target_recall: float = 0.98,
    transformer_method: str = "sentence_transformer_ner",
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train the FP classifier model.

    Args:
        data_path: Path to training data JSONL
        output_dir: Directory to save artifacts
        target_recall: Target recall for threshold
        transformer_method: Feature transformation method
        random_state: Random seed
        verbose: Print progress

    Returns:
        Dictionary with training results and metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best hyperparameters from notebook tuning
    best_params = {
        "class_weight": "balanced_subsample",
        "max_depth": 20,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 200,
    }

    if verbose:
        print("=" * 60)
        print("FP BRAND CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"\nData path: {data_path}")
        print(f"Output dir: {output_dir}")
        print(f"Target recall: {target_recall}")
        print(f"Transformer method: {transformer_method}")
        print(f"Random state: {random_state}")
        print("=" * 60)

    # Step 1: Load data
    if verbose:
        print("\n[1/8] Loading training data...")
    df = load_training_data(data_path, verbose=verbose)

    # Step 2: Create text features
    if verbose:
        print("\n[2/8] Creating text features...")
    df["text_features"] = create_text_features(df)

    # Step 3: Split data
    if verbose:
        print("\n[3/8] Splitting data...")
    train_df, val_df, test_df = split_data(
        df,
        target_col="is_sportswear",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state,
        verbose=verbose,
    )

    # Combine train + val for final training
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    X_trainval = trainval_df["text_features"]
    y_trainval = trainval_df["is_sportswear"]
    X_test = test_df["text_features"]
    y_test = test_df["is_sportswear"]

    if verbose:
        print(f"\nTrainval size: {len(trainval_df)}")
        print(f"Test size: {len(test_df)}")

    # Step 4: Create and fit feature transformer
    if verbose:
        print(f"\n[4/8] Fitting feature transformer ({transformer_method})...")
    transformer = FPFeatureTransformer(
        method=transformer_method,
        random_state=random_state,
    )
    transformer.fit(X_trainval)

    # Step 5: Transform features
    if verbose:
        print("\n[5/8] Transforming features...")
    X_trainval_transformed = transformer.transform(X_trainval)
    X_test_transformed = transformer.transform(X_test)

    if verbose:
        print(f"Feature matrix shape: {X_trainval_transformed.shape}")

    # Step 6: Train classifier
    if verbose:
        print("\n[6/8] Training Random Forest classifier...")
        print(f"Hyperparameters: {best_params}")

    clf = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_trainval_transformed, y_trainval)

    # Step 7: Evaluate on test set
    if verbose:
        print("\n[7/8] Evaluating on test set...")

    y_proba_test = clf.predict_proba(X_test_transformed)[:, 1]
    y_pred_test = clf.predict(X_test_transformed)

    test_f2 = fbeta_score(y_test, y_pred_test, beta=2)
    test_recall = (y_pred_test[y_test == 1] == 1).mean()
    test_precision = (y_test[y_pred_test == 1] == 1).mean()

    if verbose:
        print(f"Test F2 Score: {test_f2:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Precision: {test_precision:.4f}")

    # Step 8: Optimize threshold and create pipeline
    if verbose:
        print(f"\n[8/8] Optimizing threshold for {target_recall:.0%} recall...")

    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test, y_proba_test, target_recall
    )

    if verbose:
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Threshold recall: {threshold_metrics['threshold_recall']:.4f}")
        print(f"Threshold precision: {threshold_metrics['threshold_precision']:.4f}")
        print(f"Threshold F2: {threshold_metrics['threshold_f2']:.4f}")

    # Create sklearn Pipeline
    pipeline = Pipeline([
        ("features", transformer),
        ("classifier", clf),
    ])

    # Save artifacts
    pipeline_path = output_dir / "fp_classifier_pipeline.joblib"
    config_path = output_dir / "fp_classifier_config.json"

    if verbose:
        print(f"\nSaving pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)

    # Save configuration
    config = {
        "threshold": optimal_threshold,
        "target_recall": target_recall,
        "model_name": "RF_tuned",
        "transformer_method": transformer_method,
        "best_params": best_params,
        "test_f2": test_f2,
        "test_recall": test_recall,
        "test_precision": test_precision,
        **threshold_metrics,
        "trained_at": datetime.now().isoformat(),
        "data_path": str(data_path),
        "n_samples": len(df),
        "n_trainval": len(trainval_df),
        "n_test": len(test_df),
    }

    if verbose:
        print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Also save standalone transformer and classifier for flexibility
    transformer_path = output_dir / "fp_feature_transformer.joblib"
    classifier_path = output_dir / "fp_best_classifier.joblib"

    if verbose:
        print(f"Saving transformer to {transformer_path}...")
    joblib.dump(transformer, transformer_path)

    if verbose:
        print(f"Saving classifier to {classifier_path}...")
    joblib.dump(clf, classifier_path)

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nArtifacts saved to {output_dir}/:")
        print(f"  - fp_classifier_pipeline.joblib ({pipeline_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  - fp_classifier_config.json")
        print(f"  - fp_feature_transformer.joblib")
        print(f"  - fp_best_classifier.joblib")
        print("=" * 60)

    return config


def main():
    """Main entry point."""
    args = parse_args()

    try:
        config = train_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            target_recall=args.target_recall,
            transformer_method=args.transformer_method,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
