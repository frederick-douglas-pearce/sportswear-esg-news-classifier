#!/usr/bin/env python3
"""Retrain classifier with new data and compare to production.

This script retrains a classifier model with new data and compares
its performance to the current production model before promoting.

Semantic Versioning:
    - Major (v2.0.0): Breaking changes, new model architecture, schema changes
    - Minor (v1.1.0): New training data, hyperparameter tuning, improvements
    - Patch (v1.0.1): Bug fixes, threshold adjustments, config corrections

Usage:
    # Daily retraining (default: minor version bump)
    uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl

    # Major version bump (new model architecture)
    uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --major

    # Patch version bump (threshold fix)
    uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --patch

    # Auto-promote if better
    uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --auto-promote
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (before importing modules that use them)
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops import ExperimentTracker, STAGE_PRODUCTION


@dataclass
class SemanticVersion:
    """Semantic version representation (major.minor.patch)."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string into a SemanticVersion.

        Supports formats:
            - v1.2.3 (semantic)
            - v1 (legacy, treated as v1.0.0)
        """
        if not version_str:
            return cls(0, 0, 0)

        # Remove 'v' prefix if present
        version_str = version_str.lstrip("v")

        # Try semantic version pattern
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
        if match:
            return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # Try legacy single-number pattern (v1 -> v1.0.0)
        match = re.match(r"^(\d+)$", version_str)
        if match:
            return cls(int(match.group(1)), 0, 0)

        raise ValueError(f"Invalid version format: {version_str}")

    def bump_major(self) -> "SemanticVersion":
        """Increment major version, reset minor and patch."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Increment minor version, reset patch."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Increment patch version."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __gt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)


def get_current_version(classifier: str, registry_path: Path) -> SemanticVersion | None:
    """Get the current production version for a classifier."""
    if not registry_path.exists():
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    if classifier not in registry:
        return None

    prod_version = registry[classifier].get("production")
    if not prod_version:
        # No production version, find the highest version
        versions = list(registry[classifier].get("versions", {}).keys())
        if not versions:
            return None
        # Parse all versions and find the highest
        parsed = [SemanticVersion.parse(v) for v in versions]
        return max(parsed, key=lambda v: (v.major, v.minor, v.patch))

    return SemanticVersion.parse(prod_version)


def get_next_version(
    classifier: str,
    registry_path: Path,
    bump_type: str = "minor",
) -> str:
    """Get the next version number for a classifier.

    Args:
        classifier: Classifier type (fp, ep, esg)
        registry_path: Path to the registry JSON file
        bump_type: One of 'major', 'minor', 'patch'

    Returns:
        Next version string (e.g., 'v1.1.0')
    """
    current = get_current_version(classifier, registry_path)

    if current is None:
        # First version
        return "v1.0.0"

    if bump_type == "major":
        return str(current.bump_major())
    elif bump_type == "patch":
        return str(current.bump_patch())
    else:  # minor (default)
        return str(current.bump_minor())


def train_new_version(
    classifier: str,
    data_path: Path,
    output_dir: Path,
    target_recall: float = 0.99,
) -> dict:
    """Train a new version of the classifier."""
    print(f"\nTraining new {classifier.upper()} classifier version...")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the unified training script
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--classifier", classifier,
        "--data-path", str(data_path),
        "--target-recall", str(target_recall),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Training failed:\n{result.stderr}")
        return {"success": False, "error": result.stderr}

    # Load the new config
    config_path = output_dir / f"{classifier}_classifier_config.json"
    if not config_path.exists():
        return {"success": False, "error": "Config file not created"}

    with open(config_path) as f:
        config = json.load(f)

    # Get the latest MLflow model version (registered during training)
    mlflow_version = None
    tracker = ExperimentTracker(classifier)
    if tracker.enabled:
        mlflow_version = tracker.get_latest_model_version()

    return {
        "success": True,
        "config_path": str(config_path),
        "pipeline_path": str(output_dir / f"{classifier}_classifier_pipeline.joblib"),
        "mlflow_version": mlflow_version,
        "metrics": {
            "test_f2": config.get("test_f2"),
            "test_recall": config.get("test_recall"),
            "test_precision": config.get("test_precision"),
            "threshold": config.get("threshold"),
        },
    }


def load_production_metrics(classifier: str, registry_path: Path) -> dict | None:
    """Load metrics for the current production model."""
    if not registry_path.exists():
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    if classifier not in registry:
        return None

    prod_version = registry[classifier].get("production")
    if not prod_version:
        return None

    return registry[classifier]["versions"].get(prod_version, {}).get("metrics")


def compare_versions(production_metrics: dict | None, new_metrics: dict) -> dict:
    """Compare production and new version metrics."""
    if production_metrics is None:
        return {
            "comparison": "no_production",
            "improvement": True,
            "details": "No production model exists - new version will be first production model",
        }

    prod_f2 = production_metrics.get("test_f2") or production_metrics.get("cv_f2", 0)
    new_f2 = new_metrics.get("test_f2", 0)

    f2_diff = new_f2 - prod_f2
    f2_pct_change = (f2_diff / prod_f2 * 100) if prod_f2 > 0 else 0

    prod_recall = production_metrics.get("test_recall") or production_metrics.get("cv_recall", 0)
    new_recall = new_metrics.get("test_recall", 0)

    return {
        "comparison": "completed",
        "production_f2": prod_f2,
        "new_f2": new_f2,
        "f2_difference": f2_diff,
        "f2_pct_change": f2_pct_change,
        "production_recall": prod_recall,
        "new_recall": new_recall,
        "improvement": new_f2 >= prod_f2,
        "significant_improvement": f2_pct_change > 1.0,  # >1% improvement
        "details": f"F2 change: {f2_diff:+.4f} ({f2_pct_change:+.2f}%)",
    }


def promote_version(
    classifier: str,
    version: str,
    new_metrics: dict,
    data_path: str,
    output_dir: Path,
    models_dir: Path,
    registry_path: Path,
    mlflow_version: str | None = None,
) -> bool:
    """Promote new version to production.

    Args:
        classifier: Classifier type (fp, ep, esg)
        version: Version string (e.g., 'v1', 'v2')
        new_metrics: Performance metrics from training
        data_path: Path to training data used
        output_dir: Directory containing trained artifacts
        models_dir: Production models directory
        registry_path: Path to JSON registry file
        mlflow_version: MLflow Model Registry version to promote (optional)

    Returns:
        True if promotion succeeded
    """
    print(f"\nPromoting {classifier.upper()} {version} to production...")

    # Copy model files to production location
    src_pipeline = output_dir / f"{classifier}_classifier_pipeline.joblib"
    src_config = output_dir / f"{classifier}_classifier_config.json"
    dst_pipeline = models_dir / f"{classifier}_classifier_pipeline.joblib"
    dst_config = models_dir / f"{classifier}_classifier_config.json"

    shutil.copy(src_pipeline, dst_pipeline)
    shutil.copy(src_config, dst_config)

    # Update JSON registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {}

    if classifier not in registry:
        registry[classifier] = {"production": None, "versions": {}}

    registry[classifier]["production"] = version
    registry[classifier]["versions"][version] = {
        "created_at": datetime.utcnow().isoformat(),
        "trained_on": data_path,
        "threshold": new_metrics.get("threshold"),
        "metrics": {
            "test_f2": new_metrics.get("test_f2"),
            "test_recall": new_metrics.get("test_recall"),
            "test_precision": new_metrics.get("test_precision"),
        },
        "mlflow_version": mlflow_version,
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"  Copied pipeline to: {dst_pipeline}")
    print(f"  Copied config to: {dst_config}")
    print(f"  Updated registry: {registry_path}")

    # Promote in MLflow Model Registry
    if mlflow_version:
        tracker = ExperimentTracker(classifier)
        if tracker.enabled:
            promoted = tracker.promote_to_production(mlflow_version)
            if promoted:
                print(f"  Promoted MLflow model version {mlflow_version} to Production stage")
            else:
                print(f"  Warning: Failed to promote MLflow model version {mlflow_version}")

    print(f"\n{classifier.upper()} {version} is now in production!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrain classifier with new data and compare to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Semantic Versioning:
  --major    Breaking changes, new model architecture, schema changes
  --minor    New training data, hyperparameter tuning (default)
  --patch    Bug fixes, threshold adjustments, config corrections

Examples:
  # Daily retraining with new data (minor bump)
  uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl

  # New model architecture (major bump)
  uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --major

  # Threshold fix (patch bump)
  uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --patch
        """,
    )
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["fp", "ep", "esg"],
        help="Classifier to retrain",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.99,
        help="Target recall for threshold optimization (default: 0.99)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory for production models",
    )

    # Version bump type (mutually exclusive)
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--major",
        action="store_true",
        help="Major version bump (breaking changes, new architecture)",
    )
    version_group.add_argument(
        "--minor",
        action="store_true",
        help="Minor version bump (new data, tuning) - default",
    )
    version_group.add_argument(
        "--patch",
        action="store_true",
        help="Patch version bump (bug fixes, threshold adjustments)",
    )

    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote if new version is better",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote even if new version is not better",
    )

    args = parser.parse_args()

    # Determine version bump type
    if args.major:
        bump_type = "major"
    elif args.patch:
        bump_type = "patch"
    else:
        bump_type = "minor"  # default

    registry_path = args.models_dir / "registry.json"

    # Get current and next version
    current_version = get_current_version(args.classifier, registry_path)
    version = get_next_version(args.classifier, registry_path, bump_type)
    output_dir = args.models_dir / args.classifier / version

    print(f"=" * 60)
    print(f"Retraining {args.classifier.upper()} Classifier")
    print(f"=" * 60)
    print(f"Current version: {current_version or 'None'}")
    print(f"New version:     {version} ({bump_type} bump)")

    # Train new version
    training_result = train_new_version(
        args.classifier,
        args.data,
        output_dir,
        args.target_recall,
    )

    if not training_result["success"]:
        print(f"\nTraining failed: {training_result.get('error', 'Unknown error')}")
        return 1

    # Show training result
    print(f"\nTraining completed successfully!")
    print(f"  Test F2: {training_result['metrics'].get('test_f2', 'N/A'):.4f}")
    if training_result.get("mlflow_version"):
        print(f"  MLflow model version: {training_result['mlflow_version']}")

    # Compare to production
    print("\n" + "=" * 60)
    print("Version Comparison")
    print("=" * 60)

    production_metrics = load_production_metrics(args.classifier, registry_path)
    comparison = compare_versions(production_metrics, training_result["metrics"])

    if production_metrics:
        print(f"Production F2:  {comparison['production_f2']:.4f}")
        print(f"New version F2: {comparison['new_f2']:.4f}")
        print(f"Difference:     {comparison['f2_difference']:+.4f} ({comparison['f2_pct_change']:+.2f}%)")
    else:
        print("No production model found - this will be the first production model")

    print(f"\nVerdict: {comparison['details']}")

    # Decide on promotion
    should_promote = False
    if args.force_promote:
        should_promote = True
        print("\nForce promote requested - promoting regardless of comparison")
    elif comparison["improvement"]:
        if args.auto_promote:
            should_promote = True
            print("\nAuto-promote enabled - promoting new version")
        else:
            # Ask for confirmation
            response = input(f"\nPromote {version} to production? [y/N]: ")
            should_promote = response.lower() in ("y", "yes")

    if should_promote:
        promote_version(
            args.classifier,
            version,
            training_result["metrics"],
            str(args.data),
            output_dir,
            args.models_dir,
            registry_path,
            mlflow_version=training_result.get("mlflow_version"),
        )
    else:
        print(f"\nNew version saved to: {output_dir}")
        if training_result.get("mlflow_version"):
            print(f"MLflow model version: {training_result['mlflow_version']}")
        print("Run with --force-promote to promote anyway")

    return 0


if __name__ == "__main__":
    exit(main())
