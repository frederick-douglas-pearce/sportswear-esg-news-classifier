#!/usr/bin/env python3
"""Register a trained model in MLflow and optionally update the local registry.

Use this after training via notebooks to register the model without retraining.
Reads from existing config files (created by notebook's deployment cell). With
--update-registry it also adds the version to models/registry.json, recording its
`pipeline_sha256`, and stamps `version` and the same hash into
models/<classifier>_classifier_config.json (D026). A version name already in the
registry is refused before anything is written (D033).

Usage:
    # Register FP classifier with auto-detected version
    uv run python scripts/register_model.py --classifier fp

    # Register with specific version
    uv run python scripts/register_model.py --classifier fp --version v2.7.0

    # Also update local registry.json (e.g., after notebook training)
    uv run python scripts/register_model.py --classifier fp --version v2.7.0 --update-registry

    # Add notes for the version
    uv run python scripts/register_model.py --classifier fp --version v2.7.0 --update-registry \
        --notes "Added new feature X"

    # Register and add to MLflow Model Registry
    uv run python scripts/register_model.py --classifier fp --version v2.7.0 --register-model

Note:
    MLflow uses SQLite backend (mlruns.db) by default. The file-based backend
    (mlruns/) is deprecated and will be removed in a future MLflow release.
    To view the UI: uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deployment.registry import (  # noqa: E402
    next_version,
    require_new_version,
    write_version,
)


def register_mlflow(
    classifier: str,
    config: dict,
    version: str,
    notes: str = "",
    pipeline_path: Path = None,
    config_path: Path = None,
    register_model: bool = False,
) -> tuple[str, str]:
    """Register model run in MLflow.

    Returns:
        Tuple of (run_id, model_version) or (None, None) if failed.
    """
    try:
        import mlflow
    except ImportError:
        print("⚠️  MLflow not installed, skipping MLflow registration")
        return None, None

    # Use SQLite backend (file-based backend is deprecated)
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(f"esg-classifier-{classifier}")

    run_name = f"{classifier}-{version}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log params
        params = {
            "model_name": config.get("model_name"),
            "transformer_method": config.get("transformer_method"),
            "threshold": config.get("threshold"),
            "target_recall": config.get("target_recall", 0.99),
            "version": version,
        }
        # Add best_params with prefix
        for k, v in config.get("best_params", {}).items():
            params[f"param_{k}"] = v
        mlflow.log_params(params)

        # Log metrics
        metrics = {}
        for key in [
            "cv_f2", "cv_recall", "cv_precision",
            "test_f2", "test_recall", "test_precision",
            "threshold_f2", "threshold_recall", "threshold_precision",
        ]:
            if key in config:
                metrics[key] = config[key]
        mlflow.log_metrics(metrics)

        # Log artifacts
        if pipeline_path and pipeline_path.exists():
            mlflow.log_artifact(str(pipeline_path))
        if config_path and config_path.exists():
            mlflow.log_artifact(str(config_path))

        # Tags
        mlflow.set_tag("version", version)
        if notes:
            mlflow.set_tag("notes", notes)

        run_id = run.info.run_id
        model_version = None

        # Register in Model Registry if requested
        if register_model and pipeline_path and pipeline_path.exists():
            client = mlflow.tracking.MlflowClient()
            model_name = f"{classifier}-classifier"

            # Create registered model if it doesn't exist
            try:
                client.create_registered_model(
                    model_name,
                    description=f"{classifier.upper()} Classifier for ESG News"
                )
            except mlflow.exceptions.MlflowException:
                pass  # Model already exists

            # Register this version
            mv = client.create_model_version(
                name=model_name,
                source=f"{run.info.artifact_uri}/{pipeline_path.name}",
                run_id=run_id,
                description=f"{version} - {notes}" if notes else version,
            )
            model_version = mv.version

        return run_id, model_version


def update_registry(
    registry_path: Path,
    classifier: str,
    version: str,
    config: dict,
    config_path: Path,
    pipeline_path: Path,
    notes: str = "",
    set_production: bool = False,
    bump: str = "patch",
) -> None:
    """Add a new version to registry.json and stamp the config (refuses an existing name)."""
    entry = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trained_on": f"data/{classifier}_training_data.jsonl",
        "model_name": config.get("model_name"),
        "transformer_method": config.get("transformer_method"),
        "threshold": round(config.get("threshold", 0), 4),
        "metrics": {
            "cv_f2": round(config.get("cv_f2", 0), 4),
            "cv_recall": round(config.get("cv_recall", 0), 4),
            "cv_precision": round(config.get("cv_precision", 0), 4),
            "test_f2": round(config.get("test_f2", 0), 4),
            "test_recall": round(config.get("test_recall", 0), 4),
            "test_precision": round(config.get("test_precision", 0), 4),
        },
    }
    if notes:
        entry["notes"] = notes

    write_version(
        registry_path,
        classifier,
        version,
        entry,
        config_path,
        pipeline_path,
        bump,
        set_production=set_production,
    )


def main():
    parser = argparse.ArgumentParser(description="Register trained model in MLflow")
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["fp", "ep", "esg"],
        help="Classifier type",
    )
    parser.add_argument(
        "--version",
        help="Version string (e.g., v2.1.0). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Version bump type if auto-detecting version",
    )
    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Also update models/registry.json",
    )
    parser.add_argument(
        "--set-production",
        action="store_true",
        help="Set this version as production in registry",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Notes about this version",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Also register in MLflow Model Registry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    # Paths
    models_dir = Path("models")
    config_path = models_dir / f"{args.classifier}_classifier_config.json"
    pipeline_path = models_dir / f"{args.classifier}_classifier_pipeline.joblib"
    registry_path = models_dir / "registry.json"

    # Validate config exists
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        print("   Run the notebook's deployment cell first to create the config.", file=sys.stderr)
        sys.exit(1)

    # Validate the pipeline exists before anything is written: the version stamp hashes it,
    # and a registry entry written without it would describe no artifact.
    if not pipeline_path.exists():
        print(f"❌ Pipeline not found: {pipeline_path}", file=sys.stderr)
        print("   Run the notebook's deployment cell first to create the pipeline.", file=sys.stderr)
        sys.exit(1)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load registry for version detection
    with open(registry_path) as f:
        registry = json.load(f)

    # Determine version
    if args.version:
        version = args.version
    else:
        version = next_version(registry, args.classifier, args.bump)
        print(f"Auto-detected version: {version}")

    # A registered version name is never rewritten (D033). With --update-registry, refuse
    # here, before the dry-run summary and before the MLflow run.
    existing_versions = registry.get(args.classifier, {}).get("versions", {})
    if args.update_registry:
        try:
            require_new_version(registry, args.classifier, version, args.bump)
        except ValueError as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
    elif version in existing_versions:
        free = next_version(registry, args.classifier, args.bump)
        print(f"⚠️  Version {version} is already in the registry; the registry is not written.")
        print(f"   --update-registry refuses an existing name (D033); the next free version is {free}.")
        print(f"   The MLflow run is tagged {version} over the bytes now in {models_dir}/, which may")
        print(f"   not be {version}'s. The registry, not MLflow, records which bytes {version} names.")

    if args.dry_run:
        print(f"\n🔍 Dry run - would register:")
        print(f"   Classifier: {args.classifier}")
        print(f"   Version: {version}")
        print(f"   Model: {config.get('model_name')}")
        print(f"   Threshold: {config.get('threshold'):.4f}")
        print(f"   Test F2: {config.get('test_f2', 'N/A')}")
        if args.notes:
            print(f"   Notes: {args.notes}")
        print(f"\n   MLflow: Would register run (sqlite:///mlruns.db)")
        if args.register_model:
            print(f"   MLflow Model Registry: Would register {args.classifier}-classifier")
        if args.update_registry:
            print(f"   Registry: Would update {registry_path}")
        if args.set_production:
            print(f"   Production: Would set {version} as production")
        return

    # Register in MLflow
    print(f"\n📊 Registering {args.classifier} classifier {version} in MLflow...")
    run_id, model_version = register_mlflow(
        classifier=args.classifier,
        config=config,
        version=version,
        notes=args.notes,
        pipeline_path=pipeline_path,
        config_path=config_path,
        register_model=args.register_model,
    )
    if run_id:
        print(f"   ✅ MLflow run: {run_id[:8]}...")
    if model_version:
        print(f"   ✅ Model Registry: {args.classifier}-classifier version {model_version}")

    # Update registry if requested. write_version checks the name again; that check can
    # refuse only if another writer registered the name since the check above, and by then
    # the MLflow run exists.
    if args.update_registry:
        print(f"\n📝 Updating {registry_path}...")
        update_registry(
            registry_path=registry_path,
            classifier=args.classifier,
            version=version,
            config=config,
            config_path=config_path,
            pipeline_path=pipeline_path,
            notes=args.notes,
            set_production=args.set_production,
            bump=args.bump,
        )
        print(f"   ✅ Added version {version}")
        print(f"   ✅ Stamped {version} into {config_path}")
        if args.set_production:
            print(f"   ✅ Set as production")

    print(f"\n✅ Registration complete!")
    print(f"   Classifier: {args.classifier}")
    print(f"   Version: {version}")
    print(f"   Model: {config.get('model_name')}")
    print(f"   Test F2: {config.get('test_f2', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
