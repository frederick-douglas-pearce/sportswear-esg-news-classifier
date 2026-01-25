#!/usr/bin/env python3
"""Feature group ablation study for FP classifier.

Tests the impact of removing feature groups that showed negative permutation
importance (proximity_features, negative_context, brand_ner_features).

Approach:
1. Baseline: Remove all three suspicious feature groups
2. Add back each one at a time to measure marginal contribution
3. Compare to full model (all features)

For each configuration, we measure:
- CV F2 score (5-fold)
- CV F2 standard deviation (stability)
- SHAP feature group importance
- Permutation importance on validation fold
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fp1_nb.data_utils import load_jsonl_data, split_train_val_test
from src.fp1_nb.preprocessing import clean_text, create_text_features
from src.fp1_nb.feature_transformer import FPFeatureTransformer
from src.fp3_nb.explainability import TextExplainer, get_fp_feature_groups

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
DATA_PATH = Path("data/fp_training_data.jsonl")

# Best model params from fp2
BEST_PARAMS = {
    "n_estimators": 120,
    "max_depth": 15,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Feature groups to potentially exclude
# These showed negative permutation importance in the full model
SUSPECT_GROUPS = ["proximity_features", "negative_context", "brand_ner_features"]

# Ablation configurations: which groups to EXCLUDE
ABLATION_CONFIGS = {
    "full": {
        "description": "All features (current model)",
        "exclude": [],
    },
    "baseline": {
        "description": "Remove proximity, negative_context, brand_ner",
        "exclude": ["proximity_features", "negative_context", "brand_ner_features"],
    },
    "baseline+proximity": {
        "description": "Baseline + proximity_features",
        "exclude": ["negative_context", "brand_ner_features"],
    },
    "baseline+neg_context": {
        "description": "Baseline + negative_context",
        "exclude": ["proximity_features", "brand_ner_features"],
    },
    "baseline+brand_ner": {
        "description": "Baseline + brand_ner_features",
        "exclude": ["proximity_features", "negative_context"],
    },
}


def get_feature_mask(feature_groups: dict, exclude_groups: list, n_features: int) -> np.ndarray:
    """Create a boolean mask to select features (True = keep, False = exclude)."""
    mask = np.ones(n_features, dtype=bool)

    for group_name in exclude_groups:
        if group_name in feature_groups:
            indices = list(feature_groups[group_name])
            for idx in indices:
                if idx < n_features:
                    mask[idx] = False

    return mask


def apply_feature_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply feature mask to select only included features."""
    return X[:, mask]


def get_masked_feature_groups(
    original_groups: dict,
    mask: np.ndarray,
    exclude_groups: list
) -> dict:
    """Recompute feature group indices after masking."""
    # Build mapping from old indices to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx, keep in enumerate(mask):
        if keep:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    # Remap feature groups
    new_groups = {}
    for group_name, indices in original_groups.items():
        if group_name in exclude_groups:
            continue  # Skip excluded groups

        new_indices = []
        for old_idx in indices:
            if old_idx in old_to_new:
                new_indices.append(old_to_new[old_idx])

        if new_indices:
            new_groups[group_name] = range(min(new_indices), max(new_indices) + 1)

    return new_groups


def run_cv_with_config(
    X_full: np.ndarray,
    y: np.ndarray,
    feature_groups: dict,
    config: dict,
    config_name: str,
    verbose: bool = True,
) -> dict:
    """Run cross-validation with a specific feature configuration."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {config_name}")
        print(f"  {config['description']}")
        print(f"{'='*70}")

    exclude_groups = config["exclude"]

    # Create feature mask
    mask = get_feature_mask(feature_groups, exclude_groups, X_full.shape[1])
    X = apply_feature_mask(X_full, mask)
    masked_groups = get_masked_feature_groups(feature_groups, mask, exclude_groups)

    if verbose:
        print(f"  Features: {X_full.shape[1]} → {X.shape[1]} (excluded {X_full.shape[1] - X.shape[1]})")
        print(f"  Groups: {list(masked_groups.keys())}")

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_f2_scores = []
    fold_shap_importances = []
    fold_perm_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = RandomForestClassifier(**BEST_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)
        f2 = fbeta_score(y_val, y_pred, beta=2)
        fold_f2_scores.append(f2)

        # Compute importance on last fold only (for speed)
        if fold_idx == N_FOLDS - 1:
            explainer = TextExplainer(
                pipeline=model,
                feature_groups=masked_groups,
                class_names=["false_positive", "sportswear"],
            )

            # SHAP importance
            shap_importance = explainer.explain_feature_groups(
                X_val, sample_size=min(100, len(X_val)), use_tree_explainer=True
            )
            fold_shap_importances.append(shap_importance)

            # Permutation importance
            perm_importance = explainer.compute_permutation_importance(
                X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, scoring="f2"
            )
            fold_perm_importances.append(perm_importance)

        if verbose:
            print(f"  Fold {fold_idx + 1}: F2 = {f2:.4f}")

    # Aggregate results
    mean_f2 = np.mean(fold_f2_scores)
    std_f2 = np.std(fold_f2_scores)

    if verbose:
        print(f"\n  CV F2: {mean_f2:.4f} ± {std_f2:.4f}")

        if fold_shap_importances:
            shap = fold_shap_importances[0]
            print(f"\n  SHAP Feature Group Importance:")
            for group, importance in shap.top_groups[:5]:
                contrib = shap.group_contribution.get(group, 0) * 100
                print(f"    {group:<20} {contrib:>6.1f}%")

        if fold_perm_importances:
            perm = fold_perm_importances[0]
            print(f"\n  Permutation Importance (ΔF2):")
            for group, mean_decrease, std_decrease in perm.top_groups[:5]:
                contrib = perm.group_contribution.get(group, 0) * 100
                sign = "+" if mean_decrease >= 0 else ""
                print(f"    {group:<20} {sign}{mean_decrease:.4f} ± {std_decrease:.4f} ({contrib:.1f}%)")

    return {
        "config_name": config_name,
        "description": config["description"],
        "exclude": exclude_groups,
        "cv_f2_mean": mean_f2,
        "cv_f2_std": std_f2,
        "n_features": X.shape[1],
        "n_groups": len(masked_groups),
        "shap_importance": fold_shap_importances[0] if fold_shap_importances else None,
        "perm_importance": fold_perm_importances[0] if fold_perm_importances else None,
        "fold_scores": fold_f2_scores,
    }


def main():
    print("=" * 70)
    print("FEATURE GROUP ABLATION STUDY")
    print("=" * 70)
    print("\nGoal: Test if removing features with negative permutation importance")
    print("improves model stability without sacrificing performance.")
    print("\nSuspect features (negative perm importance in full model):")
    for g in SUSPECT_GROUPS:
        print(f"  - {g}")

    # Load and prepare data
    print("\n[1/3] Loading and preparing data...")
    df = load_jsonl_data(DATA_PATH)

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

    # Use train+val for CV (same as notebook methodology)
    train_df, val_df, test_df = split_train_val_test(
        df,
        target_col='is_sportswear',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=RANDOM_STATE
    )

    # Combine train and val for CV
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_texts = trainval_df['text_features']
    y = trainval_df['is_sportswear'].values
    source_names = trainval_df['source_name'].tolist()
    categories = trainval_df['category'].tolist()

    print(f"\nUsing {len(trainval_df)} samples for {N_FOLDS}-fold CV")
    print(f"(Test set of {len(test_df)} samples held out)")

    # Transform all data once with full feature set
    print("\n[2/3] Transforming features...")
    transformer = FPFeatureTransformer(
        method="tfidf_lsa_ner_proximity_brands",
        max_features=10000,
        lsa_n_components=100,
        include_metadata_features=True,
        include_brand_indicators=True,
        include_brand_summary=False,
        random_state=RANDOM_STATE,
    )

    X_full = transformer.fit_transform(X_texts, source_names=source_names, categories=categories)
    print(f"Full feature matrix: {X_full.shape}")

    # Get feature groups
    feature_groups = get_fp_feature_groups(
        method="tfidf_lsa_ner_proximity_brands",
        lsa_n_components=100,
        include_fp_indicators=True,
    )
    print(f"Feature groups: {list(feature_groups.keys())}")

    # Run ablation study
    print("\n[3/3] Running ablation experiments...")

    results = []
    for config_name, config in ABLATION_CONFIGS.items():
        result = run_cv_with_config(
            X_full, y, feature_groups,
            config, config_name, verbose=True
        )
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Configuration':<22} {'CV F2':>10} {'Std':>8} {'Feats':>7} {'Groups':>7}")
    print("-" * 60)

    baseline_f2 = None
    full_f2 = None
    for r in results:
        if r["config_name"] == "baseline":
            baseline_f2 = r["cv_f2_mean"]
        if r["config_name"] == "full":
            full_f2 = r["cv_f2_mean"]

        delta = ""
        if baseline_f2 and r["config_name"] not in ["baseline", "full"]:
            diff = r["cv_f2_mean"] - baseline_f2
            delta = f" ({diff:+.4f} vs baseline)"

        print(f"{r['config_name']:<22} {r['cv_f2_mean']:>10.4f} {r['cv_f2_std']:>8.4f} {r['n_features']:>7} {r['n_groups']:>7}{delta}")

    print("-" * 60)

    # Detailed comparison
    full_result = next(r for r in results if r["config_name"] == "full")
    baseline_result = next(r for r in results if r["config_name"] == "baseline")

    f2_diff = baseline_result["cv_f2_mean"] - full_result["cv_f2_mean"]
    std_diff = baseline_result["cv_f2_std"] - full_result["cv_f2_std"]

    print(f"\n{'='*70}")
    print("DETAILED COMPARISON: Full vs Baseline")
    print(f"{'='*70}")

    print(f"\n{'Metric':<25} {'Full':>15} {'Baseline':>15} {'Change':>15}")
    print("-" * 70)
    print(f"{'CV F2 Mean':<25} {full_result['cv_f2_mean']:>15.4f} {baseline_result['cv_f2_mean']:>15.4f} {f2_diff:>+15.4f}")
    print(f"{'CV F2 Std':<25} {full_result['cv_f2_std']:>15.4f} {baseline_result['cv_f2_std']:>15.4f} {std_diff:>+15.4f}")
    print(f"{'Features':<25} {full_result['n_features']:>15} {baseline_result['n_features']:>15} {baseline_result['n_features'] - full_result['n_features']:>+15}")
    print(f"{'Feature Groups':<25} {full_result['n_groups']:>15} {baseline_result['n_groups']:>15} {baseline_result['n_groups'] - full_result['n_groups']:>+15}")

    # SHAP comparison
    print(f"\n{'='*70}")
    print("SHAP IMPORTANCE COMPARISON (% contribution)")
    print(f"{'='*70}")

    all_groups = set()
    for r in results:
        if r["shap_importance"]:
            all_groups.update(r["shap_importance"].group_contribution.keys())

    print(f"\n{'Group':<22}", end="")
    for r in results:
        print(f" {r['config_name'][:10]:>10}", end="")
    print()
    print("-" * (22 + 11 * len(results)))

    for group in sorted(all_groups):
        print(f"{group:<22}", end="")
        for r in results:
            if r["shap_importance"] and group in r["shap_importance"].group_contribution:
                val = r["shap_importance"].group_contribution[group] * 100
                print(f" {val:>9.1f}%", end="")
            else:
                print(f" {'---':>10}", end="")
        print()

    # Permutation importance comparison
    print(f"\n{'='*70}")
    print("PERMUTATION IMPORTANCE COMPARISON (ΔF2)")
    print(f"{'='*70}")

    all_groups = set()
    for r in results:
        if r["perm_importance"]:
            all_groups.update(r["perm_importance"].group_importance.keys())

    print(f"\n{'Group':<22}", end="")
    for r in results:
        print(f" {r['config_name'][:10]:>10}", end="")
    print()
    print("-" * (22 + 11 * len(results)))

    for group in sorted(all_groups):
        print(f"{group:<22}", end="")
        for r in results:
            if r["perm_importance"] and group in r["perm_importance"].group_importance:
                val = r["perm_importance"].group_importance[group]
                print(f" {val:>+10.4f}", end="")
            else:
                print(f" {'---':>10}", end="")
        print()

    # Conclusions
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")

    print(f"\nBaseline vs Full model:")
    print(f"  F2 change:  {f2_diff:+.4f} ({'BETTER' if f2_diff > 0 else 'worse' if f2_diff < 0 else 'same'})")
    print(f"  Std change: {std_diff:+.4f} ({'MORE STABLE' if std_diff < 0 else 'less stable' if std_diff > 0 else 'same'})")

    # Check individual contributions
    print(f"\nMarginal contribution of each suspect feature group:")
    for r in results:
        if r["config_name"].startswith("baseline+"):
            group_added = r["config_name"].replace("baseline+", "")
            marginal = r["cv_f2_mean"] - baseline_result["cv_f2_mean"]
            print(f"  {group_added}: {marginal:+.4f} F2")

    # Recommendation
    if f2_diff >= -0.002 and std_diff <= 0:
        print("\n✓ RECOMMENDATION: Use BASELINE configuration")
        print("  - Similar or better performance with fewer features")
        print("  - More stable (lower variance across folds)")
        print("  - Simpler model, less prone to overfitting")
    elif f2_diff >= -0.005:
        print("\n~ RECOMMENDATION: Baseline is COMPETITIVE")
        print("  - Small F2 difference may not be significant")
        print("  - Consider baseline for production simplicity")
    else:
        print("\n✗ RECOMMENDATION: Keep FULL model")
        print("  - Baseline sacrifices too much performance")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = main()
