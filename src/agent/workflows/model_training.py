"""Model training workflow for FP and EP classifiers.

This workflow automates the model training pipeline:
1. Export training data from the database
2. Check data quality (record counts, class balance)
3. Notify user and pause for manual notebook execution
4. (User runs training notebooks)
5. Compare new model metrics to production
6. Prompt for promotion approval
7. Promote model to production

Usage:
    # Start the workflow
    uv run python -m src.agent run model_training

    # After running notebooks, resume
    uv run python -m src.agent continue model_training

    # Run for specific classifier only
    uv run python -m src.agent run model_training --classifier fp
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import agent_settings
from ..notifications import NotificationManager
from ..runner import run_export_training_data, run_script
from .base import StepDefinition, Workflow, WorkflowRegistry

logger = logging.getLogger(__name__)

# Minimum records required for training
MIN_RECORDS_FP = 500
MIN_RECORDS_EP = 200

# Maximum class imbalance ratio (majority/minority)
MAX_IMBALANCE_RATIO = 10.0


def export_training_data(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Export training data for FP and EP classifiers.

    Exports JSONL files to data/ directory for notebook consumption.
    """
    results = {
        "export_success": True,
        "datasets": {},
    }

    # Determine which classifiers to export
    # Map classifier types to dataset names
    classifier_to_dataset = {
        "fp": "fp",
        "ep": "esg-prefilter",  # EP classifier uses esg-prefilter dataset
    }
    classifiers = context.get("classifiers", ["fp", "ep"])
    if isinstance(classifiers, str):
        classifiers = [classifiers]

    for classifier in classifiers:
        dataset_name = classifier_to_dataset.get(classifier, classifier)
        logger.info(f"Exporting {classifier} training data (dataset: {dataset_name})...")

        result = run_export_training_data(dataset=dataset_name)

        dataset_result = {
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
        }

        if result.success:
            # Parse record count from output
            for line in result.stdout.split("\n"):
                if "records" in line.lower() or "exported" in line.lower():
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            dataset_result["record_count"] = int(numbers[0])
                    except (ValueError, IndexError):
                        pass
            logger.info(f"Exported {classifier} data: {dataset_result.get('record_count', 'unknown')} records")
        else:
            results["export_success"] = False
            dataset_result["error"] = result.stderr[:500]
            logger.error(f"Failed to export {classifier} data: {result.stderr[:200]}")

        results["datasets"][classifier] = dataset_result

    return results


def check_data_quality(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Check quality of exported training data.

    Validates:
    - Minimum record count
    - Class balance (not too imbalanced)
    - Data freshness
    """
    if not context.get("export_success", False):
        return {"quality_check_skipped": True, "reason": "export_failed"}

    quality_results = {
        "quality_passed": True,
        "issues": [],
        "warnings": [],
        "datasets": {},
    }

    data_dir = Path("data")

    # Map classifier types to training data file names
    classifier_to_file = {
        "fp": "fp_training_data.jsonl",
        "ep": "esg-prefilter_training_data.jsonl",
    }

    for classifier, dataset_info in context.get("datasets", {}).items():
        if not dataset_info.get("success"):
            continue

        filename = classifier_to_file.get(classifier, f"{classifier}_training_data.jsonl")
        data_file = data_dir / filename
        classifier_result = {"file": str(data_file)}

        if not data_file.exists():
            quality_results["quality_passed"] = False
            quality_results["issues"].append(f"{classifier}: Training data file not found")
            quality_results["datasets"][classifier] = classifier_result
            continue

        # Count records and check class distribution
        # Different datasets use different label fields
        label_fields = {
            "fp": "is_sportswear",
            "ep": "has_esg",
        }
        label_field = label_fields.get(classifier, "label")

        try:
            positive_count = 0
            negative_count = 0

            with open(data_file) as f:
                for line in f:
                    record = json.loads(line)
                    label = record.get(label_field)
                    if label is True or label == 1:
                        positive_count += 1
                    elif label is False or label == 0:
                        negative_count += 1

            total_count = positive_count + negative_count
            classifier_result["total_records"] = total_count
            classifier_result["positive_count"] = positive_count
            classifier_result["negative_count"] = negative_count

            # Check minimum records
            min_required = MIN_RECORDS_FP if classifier == "fp" else MIN_RECORDS_EP
            if total_count < min_required:
                quality_results["quality_passed"] = False
                quality_results["issues"].append(
                    f"{classifier}: Only {total_count} records (min: {min_required})"
                )

            # Check class balance
            if positive_count > 0 and negative_count > 0:
                imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
                classifier_result["imbalance_ratio"] = round(imbalance_ratio, 2)

                if imbalance_ratio > MAX_IMBALANCE_RATIO:
                    quality_results["warnings"].append(
                        f"{classifier}: High class imbalance ({imbalance_ratio:.1f}:1)"
                    )

            # Check file freshness
            mtime = datetime.fromtimestamp(data_file.stat().st_mtime, tz=timezone.utc)
            classifier_result["last_modified"] = mtime.isoformat()

            logger.info(
                f"{classifier} data quality: {total_count} records, "
                f"{positive_count} positive, {negative_count} negative"
            )

        except Exception as e:
            quality_results["quality_passed"] = False
            quality_results["issues"].append(f"{classifier}: Error reading data - {e}")
            logger.error(f"Error checking {classifier} data quality: {e}")

        quality_results["datasets"][classifier] = classifier_result

    return quality_results


def notify_and_pause(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send notification and pause for manual notebook execution.

    The user needs to run the training notebooks:
    - notebooks/fp1_EDA_FE.ipynb → fp2_model_selection_tuning.ipynb → fp3_model_evaluation_deployment.ipynb
    - notebooks/ep1_EDA_FE.ipynb → ep2_model_selection_tuning.ipynb → ep3_model_evaluation_deployment.ipynb
    """
    # Build notification content
    datasets = context.get("datasets", {})
    quality = context.get("quality_passed", True)
    issues = context.get("issues", [])
    warnings = context.get("warnings", [])

    # Format dataset summary
    dataset_lines = []
    for classifier, info in datasets.items():
        if isinstance(info, dict):
            records = info.get("total_records", info.get("record_count", "?"))
            dataset_lines.append(f"  {classifier.upper()}: {records} records")

    summary = {
        "title": "Model Training Data Ready",
        "datasets": "\n".join(dataset_lines) if dataset_lines else "No datasets exported",
        "quality_passed": quality,
        "issues": issues,
        "warnings": warnings,
        "instructions": [
            "1. Run the appropriate training notebooks:",
            "   - FP: fp1_EDA_FE.ipynb → fp2_model_selection_tuning.ipynb → fp3_model_evaluation_deployment.ipynb",
            "   - EP: ep1_EDA_FE.ipynb → ep2_model_selection_tuning.ipynb → ep3_model_evaluation_deployment.ipynb",
            "",
            "2. Review and tune hyperparameters as needed",
            "",
            "3. When done, resume the workflow:",
            "   uv run python -m src.agent continue model_training",
        ],
    }

    # Send email notification
    if agent_settings.email_enabled:
        try:
            subject = "Model Training Data Ready - Action Required"

            body_lines = [
                "Training data has been exported and is ready for notebook execution.",
                "",
                "Dataset Summary:",
                summary["datasets"],
                "",
            ]

            if issues:
                body_lines.extend(["Issues:", *[f"  - {i}" for i in issues], ""])

            if warnings:
                body_lines.extend(["Warnings:", *[f"  - {w}" for w in warnings], ""])

            body_lines.extend([
                "Next Steps:",
                *summary["instructions"],
            ])

            notifier = NotificationManager()
            notifier.send_email(
                subject=subject,
                body="\n".join(body_lines),
            )
            logger.info("Sent training notification email")
            summary["email_sent"] = True
        except Exception as e:
            logger.warning(f"Failed to send email notification: {e}")
            summary["email_sent"] = False
            summary["email_error"] = str(e)
    else:
        summary["email_sent"] = False
        summary["email_skipped"] = "Email notifications disabled"

    # Print instructions to console
    print("\n" + "=" * 60)
    print("MODEL TRAINING - ACTION REQUIRED")
    print("=" * 60)
    print(f"\n{summary['datasets']}")
    if issues:
        print(f"\nIssues: {', '.join(issues)}")
    if warnings:
        print(f"\nWarnings: {', '.join(warnings)}")
    print("\nNext Steps:")
    for instruction in summary["instructions"]:
        print(instruction)
    print("=" * 60 + "\n")

    return summary


def compare_models(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Compare newly trained models with production.

    Reads model configs and registry to compare F2 scores.
    """
    models_dir = Path("models")
    registry_path = models_dir / "registry.json"

    comparison_results = {
        "comparison_complete": True,
        "classifiers": {},
    }

    # Load registry
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        return {
            "comparison_complete": False,
            "error": f"Failed to load registry: {e}",
        }

    # Compare each classifier
    classifiers = context.get("classifiers", ["fp", "ep"])
    if isinstance(classifiers, str):
        classifiers = [classifiers]

    for classifier in classifiers:
        classifier_result = {}

        # Get production version info
        prod_version = registry.get(classifier, {}).get("production")
        if prod_version:
            prod_info = registry.get(classifier, {}).get("versions", {}).get(prod_version, {})
            classifier_result["production"] = {
                "version": prod_version,
                "cv_f2": prod_info.get("metrics", {}).get("cv_f2"),
                "test_f2": prod_info.get("metrics", {}).get("test_f2"),
                "test_recall": prod_info.get("metrics", {}).get("test_recall"),
                "test_precision": prod_info.get("metrics", {}).get("test_precision"),
            }
        else:
            classifier_result["production"] = None

        # Get newly trained model info from config files
        config_path = models_dir / f"{classifier}_classifier_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                classifier_result["new_model"] = {
                    "cv_f2": config.get("cv_f2"),
                    "test_f2": config.get("test_f2"),
                    "test_recall": config.get("test_recall"),
                    "test_precision": config.get("test_precision"),
                    "threshold": config.get("threshold"),
                    "model_name": config.get("model_name"),
                }

                # Calculate improvement
                if classifier_result.get("production") and classifier_result["new_model"]["test_f2"]:
                    prod_f2 = classifier_result["production"]["test_f2"] or 0
                    new_f2 = classifier_result["new_model"]["test_f2"]
                    improvement = new_f2 - prod_f2
                    # Consider "better" only if improvement is meaningful (> 0.001)
                    is_better = improvement > 0.001
                    classifier_result["improvement"] = {
                        "test_f2_delta": round(improvement, 4),
                        "test_f2_pct": round(improvement / prod_f2 * 100, 2) if prod_f2 > 0 else None,
                        "is_better": is_better,
                    }

                    logger.info(
                        f"{classifier}: Test F2 {prod_f2:.4f} → {new_f2:.4f} "
                        f"({improvement:+.4f}, {classifier_result['improvement']['test_f2_pct']:+.2f}%)"
                    )

            except Exception as e:
                logger.warning(f"Failed to read {classifier} config: {e}")
                classifier_result["new_model"] = None
                classifier_result["error"] = str(e)
        else:
            classifier_result["new_model"] = None
            logger.info(f"No new model config found for {classifier}")

        comparison_results["classifiers"][classifier] = classifier_result

    # Print comparison summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)

    for classifier, result in comparison_results["classifiers"].items():
        print(f"\n{classifier.upper()} Classifier:")

        if result.get("production"):
            prod = result["production"]
            print(f"  Production ({prod['version']}): F2={prod['test_f2']:.4f}")

        if result.get("new_model"):
            new = result["new_model"]
            print(f"  New Model: F2={new['test_f2']:.4f}, Recall={new['test_recall']:.4f}")

            if result.get("improvement"):
                imp = result["improvement"]
                if imp["is_better"]:
                    status = "✓ BETTER"
                elif abs(imp["test_f2_delta"]) < 0.001:
                    status = "= SAME"
                else:
                    status = "✗ WORSE"
                print(f"  Improvement: {imp['test_f2_delta']:+.4f} ({imp['test_f2_pct']:+.2f}%) {status}")
        else:
            print("  No new model found - notebook may not have been run")

    print("=" * 60 + "\n")

    return comparison_results


def prompt_promotion(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Prompt user for model promotion decision.

    This step pauses the workflow and waits for user approval.
    The approval is handled by the resume mechanism.
    """
    comparison = context.get("classifiers", {})

    # Check if any models improved
    models_to_promote = []
    for classifier, result in comparison.items():
        if result.get("improvement", {}).get("is_better"):
            models_to_promote.append(classifier)

    if not models_to_promote:
        logger.info("No models showed improvement - skipping promotion prompt")
        return {
            "promotion_prompted": False,
            "reason": "no_improvement",
            "models_to_promote": [],
        }

    # Store which models should be promoted when resumed
    return {
        "promotion_prompted": True,
        "models_to_promote": models_to_promote,
        "awaiting_approval": True,
    }


def promote_model(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Promote approved models to production.

    Uses the retrain.py script with --auto-promote to update registry.
    """
    models_to_promote = context.get("models_to_promote", [])

    if not models_to_promote:
        return {"promotion_skipped": True, "reason": "no_models_to_promote"}

    promotion_results = {
        "promoted": [],
        "failed": [],
    }

    for classifier in models_to_promote:
        logger.info(f"Promoting {classifier} model...")

        # Use retrain.py with --auto-promote to update registry
        # The notebooks have already trained the model, we just need to register it
        result = run_script(
            [
                "uv", "run", "python", "scripts/register_model.py",
                "--classifier", classifier,
                "--bump", "minor",  # Default to minor version bump
                "--update-registry",
            ],
            retries=0,
        )

        if result.success:
            promotion_results["promoted"].append(classifier)
            logger.info(f"Successfully promoted {classifier}")
        else:
            promotion_results["failed"].append({
                "classifier": classifier,
                "error": result.stderr[:500],
            })
            logger.error(f"Failed to promote {classifier}: {result.stderr[:200]}")

    # Send notification about promotion results
    if agent_settings.email_enabled and (promotion_results["promoted"] or promotion_results["failed"]):
        try:
            subject = "Model Promotion Complete"
            body_lines = ["Model promotion workflow completed.", ""]

            if promotion_results["promoted"]:
                body_lines.append(f"Promoted: {', '.join(promotion_results['promoted'])}")

            if promotion_results["failed"]:
                body_lines.append(f"Failed: {', '.join(f['classifier'] for f in promotion_results['failed'])}")

            notifier = NotificationManager()
            notifier.send_email(
                subject=subject,
                body="\n".join(body_lines),
            )
        except Exception as e:
            logger.warning(f"Failed to send promotion notification: {e}")

    return promotion_results


@WorkflowRegistry.register
class ModelTrainingWorkflow(Workflow):
    """Model training workflow with manual notebook execution.

    This workflow exports training data, pauses for manual notebook
    execution, then compares and optionally promotes new models.

    Steps:
    1. Export training data (fp + ep datasets)
    2. Check data quality (min records, class balance)
    3. Notify user and pause for notebook execution
    4. [User runs notebooks manually]
    5. Compare new models to production
    6. Prompt for promotion approval
    7. Promote approved models
    """

    name = "model_training"
    description = "Export training data, run notebooks, compare and promote models"

    steps = [
        StepDefinition(
            name="export_training_data",
            description="Export FP and EP training datasets from database",
            handler=export_training_data,
        ),
        StepDefinition(
            name="check_data_quality",
            description="Validate training data quality and class balance",
            handler=check_data_quality,
        ),
        StepDefinition(
            name="notify_and_pause",
            description="Send notification and pause for manual notebook execution",
            handler=notify_and_pause,
            requires_approval=True,  # This pauses the workflow
        ),
        StepDefinition(
            name="compare_models",
            description="Compare newly trained models with production",
            handler=compare_models,
        ),
        StepDefinition(
            name="prompt_promotion",
            description="Prompt user for model promotion decision",
            handler=prompt_promotion,
            requires_approval=True,  # This pauses for promotion approval
        ),
        StepDefinition(
            name="promote_model",
            description="Promote approved models to production",
            handler=promote_model,
            skip_on_dry_run=True,
        ),
    ]
