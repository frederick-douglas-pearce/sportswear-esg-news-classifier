#!/usr/bin/env python3
"""Unified training script for all classifiers.

This script trains the specified classifier from scratch using the same
process as the notebooks but in script form.

Usage:
    # Train FP classifier
    uv run python scripts/train.py --classifier fp

    # Train EP classifier
    uv run python scripts/train.py --classifier ep --target-recall 0.99

    # Train with custom data path
    uv run python scripts/train.py --classifier fp --data-path data/custom_data.jsonl

    # Verbose output
    uv run python scripts/train.py --classifier ep --verbose
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import ClassifierType
from src.deployment.data import load_training_data, split_data
from src.fp1_nb.preprocessing import create_text_features as fp_create_text_features, clean_text as fp_clean_text
from src.ep1_nb.preprocessing import create_text_features as ep_create_text_features, clean_text as ep_clean_text
from src.fp1_nb.feature_transformer import FPFeatureTransformer
from src.ep1_nb.feature_transformer import EPFeatureTransformer


# Default data paths per classifier
DEFAULT_DATA_PATHS = {
    ClassifierType.FP: "data/fp_training_data.jsonl",
    ClassifierType.EP: "data/ep_training_data.jsonl",
    ClassifierType.ESG: "data/esg_training_data.jsonl",
}

# Default target column per classifier
TARGET_COLUMNS = {
    ClassifierType.FP: "is_sportswear",
    ClassifierType.EP: "has_esg",
    ClassifierType.ESG: "esg_labels",  # Future
}

# Default target recall per classifier
DEFAULT_TARGET_RECALL = {
    ClassifierType.FP: 0.98,
    ClassifierType.EP: 0.99,
    ClassifierType.ESG: 0.95,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--classifier", "-c",
        type=str,
        choices=["fp", "ep", "esg"],
        required=True,
        help="Classifier type to train",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to training data JSONL file (uses default if not specified)",
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
        help="Target recall for threshold optimization",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
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

    # Confusion matrix values
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    metrics = {
        "threshold": float(optimal_threshold),
        "threshold_recall": float(actual_recall),
        "threshold_precision": float(precision_at_threshold),
        "threshold_f2": float(f2_score),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return optimal_threshold, metrics


def train_fp_classifier(
    data_path: str,
    output_dir: str,
    target_recall: float,
    random_state: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Train the FP (False Positive) classifier.

    Uses Random Forest with sentence-transformer + NER features.
    """
    target_col = TARGET_COLUMNS[ClassifierType.FP]

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
        print("FP CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"\nData path: {data_path}")
        print(f"Output dir: {output_dir}")
        print(f"Target recall: {target_recall}")

    # Load data
    if verbose:
        print("\n[1/8] Loading training data...")
    df = load_training_data(data_path, verbose=verbose)

    # Create text features
    if verbose:
        print("\n[2/8] Creating text features...")
    df["text_features"] = fp_create_text_features(
        df,
        text_col="content",
        title_col="title",
        brands_col="brands",
        source_name_col="source_name",
        category_col="category",
        include_metadata=True,
        clean_func=fp_clean_text,
    )

    # Split data
    if verbose:
        print("\n[3/8] Splitting data...")
    train_df, val_df, test_df = split_data(
        df,
        target_col=target_col,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state,
        verbose=verbose,
    )

    # Combine train + val for final training
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_trainval = trainval_df["text_features"]
    y_trainval = trainval_df[target_col]
    X_test = test_df["text_features"]
    y_test = test_df[target_col]

    if verbose:
        print(f"\nTrainval size: {len(trainval_df)}")
        print(f"Test size: {len(test_df)}")

    # Create and fit feature transformer
    if verbose:
        print("\n[4/8] Fitting feature transformer (sentence_transformer_ner)...")
    transformer = FPFeatureTransformer(
        method="sentence_transformer_ner",
        random_state=random_state,
    )
    transformer.fit(X_trainval)

    # Transform features
    if verbose:
        print("\n[5/8] Transforming features...")
    X_trainval_fe = transformer.transform(X_trainval)
    X_test_fe = transformer.transform(X_test)

    if verbose:
        print(f"Feature matrix shape: {X_trainval_fe.shape}")

    # Train classifier
    if verbose:
        print("\n[6/8] Training Random Forest classifier...")
        print(f"Hyperparameters: {best_params}")

    clf = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_trainval_fe, y_trainval)

    # Evaluate on test set
    if verbose:
        print("\n[7/8] Evaluating on test set...")

    y_proba_test = clf.predict_proba(X_test_fe)[:, 1]
    y_pred_test = clf.predict(X_test_fe)

    test_f2 = fbeta_score(y_test, y_pred_test, beta=2)
    test_recall = (y_pred_test[y_test == 1] == 1).mean()
    test_precision = (y_test[y_pred_test == 1] == 1).mean() if y_pred_test.sum() > 0 else 0.0

    if verbose:
        print(f"Test F2 Score: {test_f2:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Precision: {test_precision:.4f}")

    # Optimize threshold
    if verbose:
        print(f"\n[8/8] Optimizing threshold for {target_recall:.0%} recall...")

    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test.values, y_proba_test, target_recall
    )

    if verbose:
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Threshold recall: {threshold_metrics['threshold_recall']:.4f}")
        print(f"Threshold precision: {threshold_metrics['threshold_precision']:.4f}")

    # Create and save pipeline
    pipeline = Pipeline([
        ("features", transformer),
        ("classifier", clf),
    ])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_path / "fp_classifier_pipeline.joblib"
    config_path = output_path / "fp_classifier_config.json"

    if verbose:
        print(f"\nSaving pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)

    # Save configuration
    config = {
        "threshold": float(optimal_threshold),
        "target_recall": target_recall,
        "model_name": "RF_tuned",
        "transformer_method": "sentence_transformer_ner",
        "best_params": best_params,
        "test_f2": float(test_f2),
        "test_recall": float(test_recall),
        "test_precision": float(test_precision),
        **threshold_metrics,
        "trained_at": datetime.now().isoformat(),
        "data_path": str(data_path),
        "n_samples": len(df),
    }

    if verbose:
        print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

    return config


def train_ep_classifier(
    data_path: str,
    output_dir: str,
    target_recall: float,
    random_state: int,
    verbose: bool,
) -> Dict[str, Any]:
    """Train the EP (ESG Pre-filter) classifier.

    Uses Logistic Regression with TF-IDF + LSA features.
    """
    target_col = TARGET_COLUMNS[ClassifierType.EP]
    n_folds = 3

    if verbose:
        print("=" * 60)
        print("EP CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"\nData path: {data_path}")
        print(f"Output dir: {output_dir}")
        print(f"Target recall: {target_recall}")

    # Load data
    if verbose:
        print("\n[1/8] Loading training data...")
    df = load_training_data(data_path, verbose=verbose)

    # Create text features
    if verbose:
        print("\n[2/8] Creating text features...")
    df["text_features"] = ep_create_text_features(
        df,
        text_col="content",
        title_col="title",
        brands_col="brands",
        source_name_col="source_name",
        category_col="category",
        include_metadata=True,
        clean_func=ep_clean_text,
    )

    # Split data
    if verbose:
        print("\n[3/8] Splitting data...")
    train_df, val_df, test_df = split_data(
        df,
        target_col=target_col,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state,
        verbose=verbose,
    )

    X_train = train_df["text_features"]
    y_train = train_df[target_col].values
    X_test = test_df["text_features"]
    y_test = test_df[target_col].values

    if verbose:
        print(f"\nTrain size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")

    # Create and fit feature transformer
    if verbose:
        print("\n[4/8] Fitting feature transformer (tfidf_lsa)...")
    transformer = EPFeatureTransformer(
        method="tfidf_lsa",
        max_features=10000,
        lsa_n_components=200,
        include_metadata_features=False,
        random_state=random_state,
    )
    X_train_fe = transformer.fit_transform(X_train)

    if verbose:
        print(f"Feature matrix shape: {X_train_fe.shape}")

    # Transform test data
    X_test_fe = transformer.transform(X_test)

    # Hyperparameter tuning
    if verbose:
        print("\n[5/8] Tuning classifier hyperparameters...")

    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=random_state),
        param_grid,
        scoring=f2_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    grid_search.fit(X_train_fe, y_train)

    best_model = grid_search.best_estimator_
    if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F2: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    if verbose:
        print("\n[6/8] Evaluating on test set...")

    y_pred_test = best_model.predict(X_test_fe)
    y_proba_test = best_model.predict_proba(X_test_fe)[:, 1]

    test_f2 = fbeta_score(y_test, y_pred_test, beta=2)
    test_recall = (y_pred_test[y_test == 1] == 1).mean()
    test_precision = (y_test[y_pred_test == 1] == 1).mean() if y_pred_test.sum() > 0 else 0.0

    if verbose:
        print(f"Test F2: {test_f2:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Precision: {test_precision:.4f}")

    # Optimize threshold
    if verbose:
        print(f"\n[7/8] Optimizing threshold for {target_recall:.0%} recall...")

    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test, y_proba_test, target_recall
    )

    if verbose:
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Threshold recall: {threshold_metrics['threshold_recall']:.4f}")
        print(f"Threshold precision: {threshold_metrics['threshold_precision']:.4f}")

    # Create and save pipeline
    if verbose:
        print("\n[8/8] Saving artifacts...")

    pipeline = Pipeline([
        ("features", transformer),
        ("classifier", best_model),
    ])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_path / "ep_classifier_pipeline.joblib"
    config_path = output_path / "ep_classifier_config.json"

    joblib.dump(pipeline, pipeline_path)

    # Save configuration
    config = {
        "threshold": float(optimal_threshold),
        "target_recall": target_recall,
        "model_name": "LR_tuned",
        "transformer_method": "tfidf_lsa",
        "best_params": grid_search.best_params_,
        "cv_f2": float(grid_search.best_score_),
        "test_f2": float(test_f2),
        "test_recall": float(test_recall),
        "test_precision": float(test_precision),
        **threshold_metrics,
        "trained_at": datetime.now().isoformat(),
        "data_path": str(data_path),
        "n_samples": len(df),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nSaved pipeline to {pipeline_path}")
        print(f"Saved config to {config_path}")
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()

    classifier_type = ClassifierType(args.classifier)

    # Get defaults
    data_path = args.data_path or DEFAULT_DATA_PATHS[classifier_type]
    target_recall = args.target_recall or DEFAULT_TARGET_RECALL[classifier_type]

    # Validate data path
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    # Train appropriate classifier
    try:
        if classifier_type == ClassifierType.FP:
            config = train_fp_classifier(
                data_path=data_path,
                output_dir=args.output_dir,
                target_recall=target_recall,
                random_state=args.random_state,
                verbose=args.verbose,
            )
        elif classifier_type == ClassifierType.EP:
            config = train_ep_classifier(
                data_path=data_path,
                output_dir=args.output_dir,
                target_recall=target_recall,
                random_state=args.random_state,
                verbose=args.verbose,
            )
        elif classifier_type == ClassifierType.ESG:
            print("Error: ESG classifier training not yet implemented")
            return 1

        if args.verbose:
            print(f"\nFinal metrics:")
            print(f"  Test F2: {config.get('test_f2', 'N/A'):.4f}")
            print(f"  Threshold: {config.get('threshold', 'N/A'):.4f}")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
