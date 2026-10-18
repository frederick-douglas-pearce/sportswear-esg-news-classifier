#!/usr/bin/env python3
"""Promote a trained model to the registry for deployment.

This script reads the model config from the artifacts, adds a new version entry
to the registry recording its `pipeline_sha256`, and stamps `version` and the same
hash into the artifact config (D026). A version name already in the registry is
refused (D033). `--pointer-only` instead moves the `production` pointer to an
existing version whose bytes are already in models/, and writes no entry.

Usage:
    # Promote FP classifier to v2.7.0
    uv run python scripts/promote_model.py --classifier fp --version v2.7.0

    # Promote and set as production
    uv run python scripts/promote_model.py --classifier fp --version v2.7.0 --production

    # Dry run to see what would change
    uv run python scripts/promote_model.py --classifier fp --version v2.7.0 --dry-run

    # Move the production pointer to an existing version (e.g. a rollback)
    uv run python scripts/promote_model.py --classifier fp --version v2.4.0 --pointer-only
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.registry import (  # noqa: E402
    PointerMoveRefused,
    check_pointer_move,
    move_pointer,
    require_new_version,
    write_version,
)

# promote_model.py has no bump flag; a refusal names the next free minor version.
REFUSAL_BUMP = "minor"


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
        help="Version string, vMAJOR.MINOR.PATCH (e.g., v2.7.0)"
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
    parser.add_argument(
        "--pointer-only",
        action="store_true",
        help="Move the production pointer to an existing version; write no entry",
    )

    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    if args.pointer_only:
        return pointer_only(args, models_dir)

    # Load current state
    try:
        config = load_model_config(args.classifier, models_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nMake sure you have trained and exported the {args.classifier} classifier.", file=sys.stderr)
        return 1

    # The version stamp hashes the pipeline; check it exists before anything is written.
    pipeline_path = models_dir / f"{args.classifier}_classifier_pipeline.joblib"
    if not pipeline_path.exists():
        print(f"Error: Pipeline not found: {pipeline_path}", file=sys.stderr)
        return 1

    registry = load_registry(models_dir)

    # A registered version name is never rewritten (D033).
    try:
        require_new_version(registry, args.classifier, args.version, REFUSAL_BUMP)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

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

    write_version(
        models_dir / "registry.json",
        args.classifier,
        args.version,
        entry,
        models_dir / f"{args.classifier}_classifier_config.json",
        pipeline_path,
        REFUSAL_BUMP,
        set_production=args.production,
    )

    print(f"\n✓ Registry updated: models/registry.json")
    print(f"✓ {args.classifier} {args.version} is now registered")
    if args.production:
        print(f"✓ {args.classifier} production version is now {args.version}")

    print("\nNext steps:")
    print(
        "  1. Commit the registry and the stamped config (deploy.yml builds from the committed tree): "
        f"git add models/registry.json models/{args.classifier}_classifier_config.json "
        f"models/{args.classifier}_classifier_pipeline.joblib && git commit -m 'Promote model'"
    )
    print("  2. Build Docker image: docker build -t {}-classifier-api .".format(args.classifier))
    print("  3. The Dockerfile will auto-detect dependencies from the config")

    return 0


def pointer_only(args: argparse.Namespace, models_dir: Path) -> int:
    """Move the production pointer to an existing version, writing no entry."""
    if args.data_path or args.production:
        print(
            "Error: --pointer-only writes no entry, so it takes neither --data-path nor "
            "--production (the move itself sets production).",
            file=sys.stderr,
        )
        return 1

    registry_path = models_dir / "registry.json"
    pipeline_path = models_dir / f"{args.classifier}_classifier_pipeline.joblib"
    config_path = models_dir / f"{args.classifier}_classifier_config.json"

    try:
        if args.dry_run:
            check_pointer_move(
                load_registry(models_dir), args.classifier, args.version, pipeline_path, config_path
            )
            print(f"[DRY RUN] Would point {args.classifier} production at {args.version}.")
            return 0
        previous = move_pointer(
            registry_path, args.classifier, args.version, pipeline_path, config_path
        )
    except PointerMoveRefused as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"✓ {args.classifier} production: {previous} → {args.version}")
    print(f"  Commit {registry_path}; deploy.yml builds from the committed tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
