#!/usr/bin/env python3
"""Unified training script for all classifiers.

This script trains the specified classifier using configuration exported from
the model selection notebooks (fp2, ep2). All model types, hyperparameters,
and feature engineering methods are loaded from the training config.

Usage:
    # Train FP classifier (uses config from models/fp_training_config.json)
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
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Load environment variables from .env file (before importing modules that use them)
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment import ClassifierType
from src.deployment.training_config import (
    TrainingConfig,
    load_training_config,
    training_config_exists,
)
from src.mlops import ExperimentTracker
from src.deployment.data import load_training_data, split_data
from src.fp1_nb.preprocessing import (
    create_text_features as fp_create_text_features,
    clean_text as fp_clean_text,
)
from src.ep1_nb.preprocessing import (
    create_text_features as ep_create_text_features,
    clean_text as ep_clean_text,
)
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
    ClassifierType.FP: 0.99,
    ClassifierType.EP: 0.98,
    ClassifierType.ESG: 0.95,
}

# Model type to class mapping
MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVM": LinearSVC,
    "HistGradientBoosting": HistGradientBoostingClassifier,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--classifier",
        "-c",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress",
    )

    return parser.parse_args()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.98,
) -> tuple[float, dict[str, Any]]:
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


def create_model(
    model_type: str,
    params: dict[str, Any],
    random_state: int,
    calibrated: bool = False,
) -> Any:
    """Create a model instance from type and parameters.

    Args:
        model_type: Model class name (e.g., "RandomForest", "SVM")
        params: Model hyperparameters
        random_state: Random seed
        calibrated: Whether to wrap in CalibratedClassifierCV (for SVM)

    Returns:
        Fitted model instance
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_CLASSES[model_type]

    # Filter params to only those accepted by the model
    # Remove any None values and convert numpy types
    clean_params = {}
    for k, v in params.items():
        if v is not None:
            # Convert numpy types to Python types
            if hasattr(v, "item"):
                v = v.item()
            clean_params[k] = v

    # Add random_state if the model accepts it
    if model_type in ["RandomForest", "LogisticRegression", "SVM", "HistGradientBoosting"]:
        clean_params["random_state"] = random_state

    # Add n_jobs for parallelizable models
    if model_type == "RandomForest":
        clean_params["n_jobs"] = -1

    # Add max_iter for iterative models if not specified
    if model_type in ["LogisticRegression", "SVM"] and "max_iter" not in clean_params:
        clean_params["max_iter"] = 2000

    model = model_class(**clean_params)

    # Wrap SVM in CalibratedClassifierCV if needed
    if model_type == "SVM" and calibrated:
        model = CalibratedClassifierCV(model, cv=5, method="sigmoid")

    return model


def create_transformer(
    classifier_type: ClassifierType,
    method: str,
    params: dict[str, Any],
    random_state: int,
) -> Any:
    """Create a feature transformer from config.

    Args:
        classifier_type: FP or EP
        method: Feature engineering method name
        params: Transformer parameters
        random_state: Random seed

    Returns:
        Transformer instance
    """
    # Extract relevant params
    transformer_params = {
        "method": method,
        "random_state": random_state,
    }

    # Add common params if present
    param_keys = [
        "max_features",
        "lsa_n_components",
        "include_metadata_features",
        "include_vocab_features",
        "include_brand_indicators",
        "include_brand_summary",
        "proximity_window_size",
        "vocab_window_size",
    ]

    for key in param_keys:
        if key in params:
            transformer_params[key] = params[key]

    if classifier_type == ClassifierType.FP:
        return FPFeatureTransformer(**transformer_params)
    elif classifier_type == ClassifierType.EP:
        return EPFeatureTransformer(**transformer_params)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def train_classifier(
    classifier_type: ClassifierType,
    training_config: TrainingConfig,
    data_path: str,
    output_dir: str,
    target_recall: float,
    random_state: int,
    verbose: bool,
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    """Train a classifier using the provided training config.

    Args:
        classifier_type: FP or EP
        training_config: Configuration loaded from notebook export
        data_path: Path to training data
        output_dir: Directory to save artifacts
        target_recall: Target recall for threshold optimization
        random_state: Random seed
        verbose: Print detailed progress
        tracker: MLflow experiment tracker

    Returns:
        Training result config
    """
    target_col = TARGET_COLUMNS[classifier_type]
    classifier_name = classifier_type.value.upper()

    # Extract config values
    model_type = training_config.model.type
    model_params = training_config.model.params
    fe_method = training_config.feature_engineering.method
    fe_params = training_config.feature_engineering.params
    is_calibrated = training_config.model.calibrated

    if verbose:
        print("=" * 60)
        print(f"{classifier_name} CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"\nData path: {data_path}")
        print(f"Output dir: {output_dir}")
        print(f"Target recall: {target_recall}")
        print(f"\nTraining config from: {training_config.notebook}")
        print(f"  Model: {model_type}")
        print(f"  Feature method: {fe_method}")
        print(f"  Calibrated: {is_calibrated}")

    # Log training parameters to MLflow
    if tracker:
        tracker.log_params(
            {
                "data_path": data_path,
                "target_recall": target_recall,
                "random_state": random_state,
                "model_type": model_type,
                "transformer_method": fe_method,
                "calibrated": is_calibrated,
                **{f"model_{k}": v for k, v in model_params.items() if v is not None},
            }
        )

    # Load data
    if verbose:
        print("\n[1/7] Loading training data...")
    df = load_training_data(data_path, verbose=verbose)

    # Create text features
    if verbose:
        print("\n[2/7] Creating text features...")

    # Use appropriate preprocessing for classifier type
    if classifier_type == ClassifierType.FP:
        df["text_features"] = fp_create_text_features(
            df,
            text_col="content",
            title_col="title",
            brands_col="brands",
            source_name_col="source_name",
            category_col="category",
            include_metadata=fe_params.get("include_metadata_in_text", True),
            clean_func=fp_clean_text,
        )
    else:
        df["text_features"] = ep_create_text_features(
            df,
            text_col="content",
            title_col="title",
            brands_col="brands",
            source_name_col=None,
            category_col=None,
            include_metadata=fe_params.get("include_metadata_in_text", False),
            clean_func=ep_clean_text,
        )

    # Split data
    if verbose:
        print("\n[3/7] Splitting data...")
    train_df, val_df, test_df = split_data(
        df,
        target_col=target_col,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state,
        verbose=verbose,
    )

    # Extract text features and targets for each split
    X_train = train_df["text_features"]
    y_train = train_df[target_col].values
    X_val = val_df["text_features"]
    y_val = val_df[target_col].values
    X_test = test_df["text_features"]
    y_test = test_df[target_col].values

    if verbose:
        print(f"\nTrain size: {len(train_df)}")
        print(f"Val size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")

    # Create and fit feature transformer on TRAIN ONLY (matches notebook methodology)
    # This prevents data leakage from validation set into feature learning
    if verbose:
        print(f"\n[4/7] Fitting feature transformer on train data ({fe_method})...")

    transformer = create_transformer(
        classifier_type=classifier_type,
        method=fe_method,
        params=fe_params,
        random_state=random_state,
    )

    # For FP classifier, need to pass metadata for discrete features
    if classifier_type == ClassifierType.FP and fe_params.get("include_metadata_features"):
        train_source_names = train_df["source_name"].tolist()
        train_categories = train_df["category"].tolist()
        val_source_names = val_df["source_name"].tolist()
        val_categories = val_df["category"].tolist()
        test_source_names = test_df["source_name"].tolist()
        test_categories = test_df["category"].tolist()

        # Fit on train only
        transformer.fit(X_train, source_names=train_source_names, categories=train_categories)

        # Transform all splits
        X_train_fe = transformer.transform(X_train, source_names=train_source_names, categories=train_categories)
        X_val_fe = transformer.transform(X_val, source_names=val_source_names, categories=val_categories)
        X_test_fe = transformer.transform(X_test, source_names=test_source_names, categories=test_categories)
    else:
        # Fit on train only
        X_train_fe = transformer.fit_transform(X_train)

        # Transform val and test
        X_val_fe = transformer.transform(X_val)
        X_test_fe = transformer.transform(X_test)

    # Combine train + val features for classifier training
    import scipy.sparse as sp

    if sp.issparse(X_train_fe):
        X_trainval_fe = sp.vstack([X_train_fe, X_val_fe])
    else:
        X_trainval_fe = np.vstack([X_train_fe, X_val_fe])
    y_trainval = np.concatenate([y_train, y_val])

    if verbose:
        print(f"Feature matrix shape: {X_trainval_fe.shape}")

    # Create and train model
    if verbose:
        print(f"\n[5/7] Training {model_type} classifier...")
        print(f"Hyperparameters: {model_params}")

    model = create_model(
        model_type=model_type,
        params=model_params,
        random_state=random_state,
        calibrated=is_calibrated,
    )

    # HistGradientBoosting requires dense arrays
    if model_type == "HistGradientBoosting":
        if sp.issparse(X_trainval_fe):
            X_trainval_fe = X_trainval_fe.toarray()
            X_test_fe = X_test_fe.toarray()

    model.fit(X_trainval_fe, y_trainval)

    # Evaluate on test set
    if verbose:
        print("\n[6/7] Evaluating on test set...")

    y_pred_test = model.predict(X_test_fe)

    # Get probabilities (handle SVM decision_function)
    if hasattr(model, "predict_proba"):
        y_proba_test = model.predict_proba(X_test_fe)[:, 1]
    else:
        # Use decision function and convert to pseudo-probability
        decision = model.decision_function(X_test_fe)
        y_proba_test = 1 / (1 + np.exp(-decision))  # Sigmoid

    test_f2 = fbeta_score(y_test, y_pred_test, beta=2)
    test_recall = (y_pred_test[y_test == 1] == 1).mean()
    test_precision = (y_test[y_pred_test == 1] == 1).mean() if y_pred_test.sum() > 0 else 0.0

    if verbose:
        print(f"Test F2 Score: {test_f2:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test Precision: {test_precision:.4f}")

    # Optimize threshold
    if verbose:
        print(f"\n[7/7] Optimizing threshold for {target_recall:.0%} recall...")

    optimal_threshold, threshold_metrics = find_optimal_threshold(y_test, y_proba_test, target_recall)

    if verbose:
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Threshold recall: {threshold_metrics['threshold_recall']:.4f}")
        print(f"Threshold precision: {threshold_metrics['threshold_precision']:.4f}")

    # Create and save pipeline
    pipeline = Pipeline(
        [
            ("features", transformer),
            ("classifier", model),
        ]
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_path / f"{classifier_type.value}_classifier_pipeline.joblib"
    config_path = output_path / f"{classifier_type.value}_classifier_config.json"

    if verbose:
        print(f"\nSaving pipeline to {pipeline_path}...")
    joblib.dump(pipeline, pipeline_path)

    # Save configuration
    config = {
        "threshold": float(optimal_threshold),
        "target_recall": target_recall,
        "model_name": f"{model_type}_tuned",
        "transformer_method": fe_method,
        "best_params": model_params,
        "calibrated": is_calibrated,
        "test_f2": float(test_f2),
        "test_recall": float(test_recall),
        "test_precision": float(test_precision),
        **threshold_metrics,
        "trained_at": datetime.now().isoformat(),
        "data_path": str(data_path),
        "n_samples": len(df),
        "training_config": training_config.notebook,
    }

    if verbose:
        print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Log metrics and artifacts to MLflow
    if tracker:
        tracker.log_metrics(
            {
                "test_f2": test_f2,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "threshold": optimal_threshold,
                "threshold_recall": threshold_metrics["threshold_recall"],
                "threshold_precision": threshold_metrics["threshold_precision"],
                "threshold_f2": threshold_metrics["threshold_f2"],
                "n_samples": len(df),
            }
        )
        tracker.log_artifact(pipeline_path)
        tracker.log_model_config(config)

        # Log sklearn model for Model Registry
        tracker.log_sklearn_model(pipeline, artifact_path="model")

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

    return config


def main() -> int:
    """Main entry point."""
    args = parse_args()

    classifier_type = ClassifierType(args.classifier)

    # Check for training config
    if not training_config_exists(classifier_type.value, args.output_dir):
        print(f"Error: Training config not found for {classifier_type.value} classifier.")
        print(f"Please run the {classifier_type.value}2 notebook to export the training config.")
        print(f"Expected file: {args.output_dir}/{classifier_type.value}_training_config.json")
        return 1

    # Load training config
    try:
        training_config = load_training_config(classifier_type.value, args.output_dir)
    except Exception as e:
        print(f"Error loading training config: {e}")
        return 1

    # Get defaults
    data_path = args.data_path or DEFAULT_DATA_PATHS[classifier_type]
    target_recall = args.target_recall or DEFAULT_TARGET_RECALL[classifier_type]

    # Validate data path
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    # Create experiment tracker (gracefully degrades when disabled)
    tracker = ExperimentTracker(classifier_type.value)

    # Train classifier with MLflow tracking
    try:
        with tracker.start_run(tags={"target_recall": str(target_recall)}):
            if classifier_type in [ClassifierType.FP, ClassifierType.EP]:
                config = train_classifier(
                    classifier_type=classifier_type,
                    training_config=training_config,
                    data_path=data_path,
                    output_dir=args.output_dir,
                    target_recall=target_recall,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    tracker=tracker,
                )
            elif classifier_type == ClassifierType.ESG:
                print("Error: ESG classifier training not yet implemented")
                return 1

            # Register model in MLflow Model Registry
            model_version = None
            if tracker.enabled:
                model_version = tracker.register_model(
                    description=f"Trained on {data_path} with target recall {target_recall}",
                    tags={
                        "test_f2": str(config.get("test_f2", "")),
                        "test_recall": str(config.get("test_recall", "")),
                        "threshold": str(config.get("threshold", "")),
                    },
                )

            if args.verbose:
                print(f"\nFinal metrics:")
                print(f"  Test F2: {config.get('test_f2', 'N/A'):.4f}")
                print(f"  Threshold: {config.get('threshold', 'N/A'):.4f}")
                if tracker.enabled:
                    print(f"  MLflow run ID: {tracker.get_run_id()}")
                    if model_version:
                        print(f"  MLflow model version: {model_version}")
                        print(f"  Registered model: {tracker.get_registered_model_name()}")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
