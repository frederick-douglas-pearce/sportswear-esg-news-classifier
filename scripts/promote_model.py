#!/usr/bin/env python3
"""Promote a trained model to the registry for deployment.

This script reads the model config from the artifacts and updates the registry
with a new version entry.

Usage:
    # Promote FP classifier to v2
    python scripts/promote_model.py --classifier fp --version v2

    # Promote and set as production
    python scripts/promote_model.py --classifier fp --version v2 --production

    # Dry run to see what would change
    python scripts/promote_model.py --classifier fp --version v2 --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_model_config(classifier_type: str, models_dir: Path) -> dict:
    """Load the classifier config from the models directory."""
    config_path = models_dir / f"{classifier_type}_classifier_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def load_registry(models_dir: Path) -> dict:
    """Load the model registry."""
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        return {"fp": {"production": None, "versions": {}},
                "ep": {"production": None, "versions": {}},
                "esg": {"production": None, "versions": {}}}

    with open(registry_path) as f:
        return json.load(f)


def save_registry(registry: dict, models_dir: Path) -> None:
    """Save the model registry."""
    registry_path = models_dir / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")


def create_version_entry(config: dict, data_path: str | None = None) -> dict:
    """Create a registry version entry from model config."""
    entry = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trained_on": data_path or f"data/{config.get('classifier_type', 'unknown')}_training_data.jsonl",
        "model_name": config.get("model_name", "unknown"),
        "transformer_method": config.get("transformer_method", "unknown"),
        "threshold": round(config.get("threshold", 0.5), 4),
        "metrics": {
            "cv_f2": round(config.get("cv_f2", 0), 4),
            "cv_recall": round(config.get("cv_recall", 0), 4),
            "cv_precision": round(config.get("cv_precision", 0), 4),
            "test_f2": round(config.get("test_f2", 0), 4),
            "test_recall": round(config.get("test_recall", 0), 4),
            "test_precision": round(config.get("test_precision", 0), 4),
        }
    }

    # Add notes about dependencies
    method = config.get("transformer_method", "")
    if "sentence_transformer" in method:
        entry["notes"] = "Requires sentence-transformers + spaCy"
    elif "ner" in method:
        entry["notes"] = "Requires spaCy for NER"
    else:
        entry["notes"] = "Lightweight - no extra dependencies"

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Promote a trained model to the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--classifier", "-c",
        required=True,
        choices=["fp", "ep", "esg"],
        help="Classifier type to promote"
    )
    parser.add_argument(
        "--version", "-v",
        required=True,
        help="Version string (e.g., v2, v3)"
    )
    parser.add_argument(
        "--production", "-p",
        action="store_true",
        help="Set this version as production"
    )
    parser.add_argument(
        "--data-path",
        help="Training data path (for registry metadata)"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Models directory (default: models)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without saving"
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    # Load current state
    try:
        config = load_model_config(args.classifier, models_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nMake sure you have trained and exported the {args.classifier} classifier.", file=sys.stderr)
        return 1

    registry = load_registry(models_dir)

    # Check if version already exists
    if args.version in registry[args.classifier]["versions"]:
        print(f"Warning: Version {args.version} already exists for {args.classifier}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return 0

    # Create new version entry
    entry = create_version_entry(config, args.data_path)

    print(f"\n=== Promoting {args.classifier} classifier to {args.version} ===")
    print(f"Model: {entry['model_name']}")
    print(f"Transformer: {entry['transformer_method']}")
    print(f"Threshold: {entry['threshold']}")
    print(f"Test F2: {entry['metrics']['test_f2']}")
    print(f"Test Recall: {entry['metrics']['test_recall']}")
    print(f"Notes: {entry.get('notes', 'N/A')}")

    if args.production:
        print(f"\n→ Will set {args.version} as PRODUCTION")

    if args.dry_run:
        print("\n[DRY RUN] No changes saved.")
        return 0

    # Update registry
    registry[args.classifier]["versions"][args.version] = entry
    if args.production:
        registry[args.classifier]["production"] = args.version

    save_registry(registry, models_dir)

    print(f"\n✓ Registry updated: models/registry.json")
    print(f"✓ {args.classifier} {args.version} is now registered")
    if args.production:
        print(f"✓ {args.classifier} production version is now {args.version}")

    print("\nNext steps:")
    print("  1. Commit the registry: git add models/registry.json && git commit -m 'Promote model'")
    print("  2. Build Docker image: docker build -t {}-classifier-api .".format(args.classifier))
    print("  3. The Dockerfile will auto-detect dependencies from the config")

    return 0


if __name__ == "__main__":
    sys.exit(main())
