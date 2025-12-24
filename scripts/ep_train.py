#!/usr/bin/env python3
"""Train the ESG Pre-filter (EP) classifier.

This script trains the EP classifier pipeline using the training data
and exports the model artifacts for deployment.

Usage:
    python scripts/ep_train.py
    python scripts/ep_train.py --data data/ep_training_data.jsonl
    python scripts/ep_train.py --target-recall 0.99
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ep1_nb.data_utils import load_jsonl_data, split_train_val_test
from src.ep1_nb.preprocessing import clean_text, create_text_features
from src.ep1_nb.feature_transformer import EPFeatureTransformer
from src.ep3_nb.deployment import create_deployment_pipeline, save_deployment_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the ESG Pre-filter classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ep_training_data.jsonl",
        help="Path to training data file",
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
        default=0.99,
        help="Target recall for threshold optimization",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.99
) -> tuple[float, dict]:
    """Find optimal threshold for target recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    idx = np.where(recalls >= target_recall)[0]
    if len(idx) == 0:
        print(f"Warning: Cannot achieve target recall of {target_recall}")
        optimal_threshold = 0.5
        best_idx = np.argmin(np.abs(thresholds - 0.5))
    else:
        best_idx = idx[-1]
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0

    actual_recall = recalls[best_idx]
    precision_at_threshold = precisions[best_idx]

    y_pred = (y_proba >= optimal_threshold).astype(int)
    f2_score = fbeta_score(y_true, y_pred, beta=2)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    metrics = {
        'threshold': float(optimal_threshold),
        'target_recall': target_recall,
        'actual_recall': float(actual_recall),
        'precision': float(precision_at_threshold),
        'f2_score': float(f2_score),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }

    return optimal_threshold, metrics


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configuration
    target_col = 'has_esg'
    random_state = args.random_state
    n_folds = 3

    print("=" * 60)
    print("EP CLASSIFIER TRAINING")
    print("=" * 60)

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print(f"\nLoading data from {data_path}...")
    df = load_jsonl_data(data_path)

    # Create text features
    print("Creating text features...")
    df['text_features'] = create_text_features(
        df,
        text_col='content',
        title_col='title',
        brands_col='brands',
        source_name_col='source_name',
        category_col='category',
        include_metadata=True,
        clean_func=clean_text
    )

    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = split_train_val_test(
        df,
        target_col=target_col,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=random_state
    )

    X_train = train_df['text_features']
    y_train = train_df[target_col].values
    X_test = test_df['text_features']
    y_test = test_df[target_col].values
    test_source_names = test_df['source_name'].tolist()
    test_categories = test_df['category'].tolist()

    # Create feature transformer with best configuration from notebooks
    print("\nCreating feature transformer...")
    transformer = EPFeatureTransformer(
        method='tfidf_lsa',
        max_features=10000,
        lsa_n_components=200,
        include_metadata_features=False,
        random_state=random_state
    )

    # Fit transformer and transform training data
    print("Fitting transformer...")
    X_train_fe = transformer.fit_transform(X_train)
    print(f"Training features shape: {X_train_fe.shape}")

    # Transform test data
    X_test_fe = transformer.transform(
        X_test,
        source_names=test_source_names,
        categories=test_categories
    )

    # Hyperparameter tuning
    print("\nTuning classifier hyperparameters...")
    f2_scorer = make_scorer(fbeta_score, beta=2)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced'],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=random_state),
        param_grid,
        scoring=f2_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_fe, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F2: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_test = best_model.predict(X_test_fe)
    y_proba_test = best_model.predict_proba(X_test_fe)[:, 1]

    test_f2 = fbeta_score(y_test, y_pred_test, beta=2)
    test_recall = (y_pred_test[y_test == 1] == 1).mean()
    test_precision = (y_test[y_pred_test == 1] == 1).mean()

    print(f"Test F2: {test_f2:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

    # Find optimal threshold
    print(f"\nFinding optimal threshold for {args.target_recall:.0%} target recall...")
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_test, y_proba_test,
        target_recall=args.target_recall
    )
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"At threshold - Recall: {threshold_metrics['actual_recall']:.4f}, "
          f"Precision: {threshold_metrics['precision']:.4f}")

    # Create deployment pipeline
    print("\nCreating deployment pipeline...")
    pipeline = create_deployment_pipeline(
        transformer=transformer,
        classifier=best_model,
        pipeline_name='ep_classifier'
    )

    # Prepare configuration
    config = {
        'threshold': float(optimal_threshold),
        'target_recall': args.target_recall,
        'model_name': 'LogisticRegression',
        'transformer_method': transformer.method,
        'best_params': grid_search.best_params_,
        'cv_f2': float(grid_search.best_score_),
        'test_f2': float(test_f2),
        'test_recall': float(test_recall),
        'test_precision': float(test_precision),
        'threshold_recall': float(threshold_metrics['actual_recall']),
        'threshold_precision': float(threshold_metrics['precision']),
        'threshold_f2': float(threshold_metrics['f2_score']),
    }

    # Save artifacts
    output_dir = Path(args.output_dir)
    print(f"\nSaving artifacts to {output_dir}...")
    saved_paths = save_deployment_artifacts(
        pipeline=pipeline,
        config=config,
        models_dir=output_dir,
        pipeline_name='ep_classifier'
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nSaved artifacts:")
    for name, path in saved_paths.items():
        print(f"  - {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
