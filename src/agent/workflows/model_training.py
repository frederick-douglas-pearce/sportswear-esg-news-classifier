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
from ..notifications import Notification, NotificationManager, NotificationType
from ..runner import run_export_training_data, run_script
from .base import StepDefinition, Workflow, WorkflowRegistry

# Lazy imports for experiment tracking (avoid import errors if not needed)
_ExperimentTracker = None
_ExperimentReflector = None


def _get_tracker_class():
    global _ExperimentTracker
    if _ExperimentTracker is None:
        from src.experiment_log.tracker import ExperimentTracker
        _ExperimentTracker = ExperimentTracker
    return _ExperimentTracker


def _get_reflector_class():
    global _ExperimentReflector
    if _ExperimentReflector is None:
        from src.experiment_log.reflection import ExperimentReflector
        _ExperimentReflector = ExperimentReflector
    return _ExperimentReflector

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
    # Map classifier types to dataset names and standard output files
    classifier_to_dataset = {
        "fp": "fp",
        "ep": "esg-prefilter",  # EP classifier uses esg-prefilter dataset
    }
    # Standard output files that notebooks expect
    classifier_to_standard_file = {
        "fp": "data/fp_training_data.jsonl",
        "ep": "data/ep_training_data.jsonl",
    }
    classifiers = context.get("classifiers", ["fp", "ep"])
    if isinstance(classifiers, str):
        classifiers = [classifiers]

    data_dir = Path("data")

    for classifier in classifiers:
        dataset_name = classifier_to_dataset.get(classifier, classifier)
        standard_file = classifier_to_standard_file.get(classifier)
        logger.info(f"Exporting {classifier} training data (dataset: {dataset_name})...")

        # Export to timestamped file (default behavior)
        result = run_export_training_data(dataset=dataset_name)

        if result.success:
            # Find the latest timestamped file and copy to standard name
            pattern = f"{dataset_name}_*.jsonl"
            timestamped_files = sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if timestamped_files and standard_file:
                import shutil
                latest_file = timestamped_files[0]
                shutil.copy2(latest_file, standard_file)
                logger.info(f"Copied {latest_file.name} to {standard_file}")

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
        "ep": "ep_training_data.jsonl",
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

    # --- Experiment tracking: create experiments for successful exports ---
    experiment_ids = {}
    for classifier, dataset_info in context.get("datasets", {}).items():
        if not dataset_info.get("success"):
            continue
        try:
            TrackerClass = _get_tracker_class()
            tracker = TrackerClass(classifier=classifier)
            # Merge export context with quality results for richer state
            combined = {**context, **quality_results}
            exp_id = tracker.create_experiment(combined)
            experiment_ids[classifier] = exp_id
            logger.info(f"Created experiment {exp_id} for {classifier}")
        except Exception as e:
            logger.warning(f"Failed to create experiment for {classifier}: {e}")

    if experiment_ids:
        quality_results["experiment_ids"] = experiment_ids

    return quality_results


def notify_and_pause(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send notification and pause for manual notebook execution.

    The user needs to run the training notebooks:
    - notebooks/fp1_EDA_FE.ipynb → fp2_model_selection_tuning.ipynb → fp3_model_evaluation_deployment.ipynb
    - notebooks/ep1_EDA_FE.ipynb → ep2_model_selection_tuning.ipynb → ep3_model_evaluation_deployment.ipynb

    This step explicitly pauses the workflow after sending the notification,
    rather than using requires_approval=True which pauses before execution.
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

            notification = Notification(
                notification_type=NotificationType.WORKFLOW_COMPLETE,
                subject="Model Training Data Ready - Action Required",
                message="\n".join(body_lines),
                details={
                    "workflow": "model_training",
                    "datasets": summary["datasets"],
                    "quality_passed": quality,
                },
                severity="info",
            )

            notifier = NotificationManager()
            result = notifier.send(notification, channels=["email"])
            logger.info("Sent training notification email")
            summary["email_sent"] = result.get("email", False)
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

    # Pause workflow after sending notification
    # This is done explicitly rather than via requires_approval=True
    # so the notification is sent BEFORE pausing
    workflow._workflow_state.current_step = "notify_and_pause"
    workflow.state.pause_workflow(
        workflow.name,
        reason="Waiting for manual notebook execution",
    )
    logger.info("Workflow paused - waiting for notebook execution")

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

    # --- Experiment tracking: record observations ---
    experiment_ids = context.get("experiment_ids", {})
    for classifier in classifiers:
        exp_id = experiment_ids.get(classifier)
        if not exp_id:
            continue
        try:
            TrackerClass = _get_tracker_class()
            tracker = TrackerClass(classifier=classifier)
            tracker.resume(exp_id)
            tracker.record_observation(comparison_results)
            logger.info(f"Recorded observation for experiment {exp_id}")
        except Exception as e:
            logger.warning(f"Failed to record observation for {classifier}: {e}")

    return comparison_results


def prompt_promotion(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Prompt user for model promotion decision.

    This step determines which models improved, displays results,
    then pauses explicitly for user approval before promotion.
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

    # Display models that would be promoted
    print("\n" + "=" * 60)
    print("MODELS READY FOR PROMOTION")
    print("=" * 60)
    for classifier in models_to_promote:
        result = comparison.get(classifier, {})
        imp = result.get("improvement", {})
        new_model = result.get("new_model", {})
        print(f"\n{classifier.upper()}:")
        print(f"  New Test F2: {new_model.get('test_f2', 'N/A'):.4f}")
        print(f"  Improvement: {imp.get('test_f2_delta', 0):+.4f} ({imp.get('test_f2_pct', 0):+.2f}%)")
    print("\nTo promote these models, resume the workflow:")
    print("  uv run python -m src.agent continue model_training")
    print("=" * 60 + "\n")

    result = {
        "promotion_prompted": True,
        "models_to_promote": models_to_promote,
    }

    # Pause workflow explicitly for user approval
    # This is done after the handler runs so models_to_promote is in the context
    workflow._workflow_state.current_step = "prompt_promotion"
    workflow.state.pause_workflow(
        workflow.name,
        reason="Approval required for model promotion",
    )
    logger.info(f"Workflow paused - approval required to promote: {models_to_promote}")

    return result


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
            body_lines = ["Model promotion workflow completed.", ""]

            if promotion_results["promoted"]:
                body_lines.append(f"Promoted: {', '.join(promotion_results['promoted'])}")

            if promotion_results["failed"]:
                body_lines.append(f"Failed: {', '.join(f['classifier'] for f in promotion_results['failed'])}")

            notification = Notification(
                notification_type=NotificationType.WORKFLOW_COMPLETE,
                subject="Model Promotion Complete",
                message="\n".join(body_lines),
                details={
                    "workflow": "model_training",
                    "promoted": promotion_results["promoted"],
                    "failed": [f["classifier"] for f in promotion_results["failed"]],
                },
                severity="info" if not promotion_results["failed"] else "warning",
            )

            notifier = NotificationManager()
            notifier.send(notification, channels=["email"])
        except Exception as e:
            logger.warning(f"Failed to send promotion notification: {e}")

    return promotion_results


def trigger_deployment(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Trigger GitHub Actions deployment for promoted models.

    Uses `gh workflow run` to trigger the deploy.yml workflow for each
    promoted classifier. Only triggers for minor/major versions.

    Requires:
    - GitHub CLI (`gh`) installed and authenticated
    - deploy.yml workflow in .github/workflows/
    """
    promoted = context.get("promoted", [])

    if not promoted:
        return {"deployment_skipped": True, "reason": "no_promoted_models"}

    deployment_results = {
        "triggered": [],
        "skipped": [],
        "failed": [],
    }

    # Read registry to get version info
    registry_path = Path("models/registry.json")
    registry = {}
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read registry: {e}")

    for classifier in promoted:
        # Get version from registry
        classifier_info = registry.get(classifier, {})
        version = classifier_info.get("version", "unknown")

        # Determine bump type based on version change
        # Default to minor since that's what promote_model uses
        bump_type = "minor"

        logger.info(f"Triggering deployment for {classifier} ({version}, {bump_type})")

        # Trigger GitHub Actions workflow
        try:
            result = run_script(
                [
                    "gh", "workflow", "run", "deploy.yml",
                    "-f", f"classifier={classifier}",
                    "-f", f"version={version}",
                    "-f", f"bump_type={bump_type}",
                ],
                retries=0,
                timeout=30,
            )

            if result.success:
                deployment_results["triggered"].append({
                    "classifier": classifier,
                    "version": version,
                    "bump_type": bump_type,
                })
                logger.info(f"Deployment triggered for {classifier} {version}")
            else:
                # Check if gh is not installed or not authenticated
                if "gh: command not found" in result.stderr or "not found" in result.stderr.lower():
                    logger.warning("GitHub CLI (gh) not installed - skipping deployment trigger")
                    deployment_results["skipped"].append({
                        "classifier": classifier,
                        "reason": "gh_not_installed",
                    })
                elif "authentication" in result.stderr.lower() or "login" in result.stderr.lower():
                    logger.warning("GitHub CLI not authenticated - skipping deployment trigger")
                    deployment_results["skipped"].append({
                        "classifier": classifier,
                        "reason": "gh_not_authenticated",
                    })
                else:
                    deployment_results["failed"].append({
                        "classifier": classifier,
                        "error": result.stderr[:200],
                    })
                    logger.error(f"Failed to trigger deployment for {classifier}: {result.stderr[:200]}")

        except Exception as e:
            logger.error(f"Exception triggering deployment for {classifier}: {e}")
            deployment_results["failed"].append({
                "classifier": classifier,
                "error": str(e),
            })

    # Log summary
    if deployment_results["triggered"]:
        print("\n" + "=" * 60)
        print("DEPLOYMENT TRIGGERED")
        print("=" * 60)
        for item in deployment_results["triggered"]:
            print(f"  {item['classifier']}: {item['version']} ({item['bump_type']})")
        print("Monitor progress at: https://github.com/<owner>/<repo>/actions")
        print("=" * 60 + "\n")

    return deployment_results


def finalize_experiments(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Finalize experiment entries with reward, reflection, and completion.

    For each classifier with an experiment_id in context:
    - Record reward (promoted or not)
    - Optional LLM reflection (gated by agent settings)
    - Complete the experiment
    """
    experiment_ids = context.get("experiment_ids", {})
    promoted = context.get("promoted", [])
    finalized = []

    for classifier, exp_id in experiment_ids.items():
        try:
            TrackerClass = _get_tracker_class()
            tracker = TrackerClass(classifier=classifier)
            tracker.resume(exp_id)

            # Record reward
            is_promoted = classifier in promoted
            reason = "improvement" if is_promoted else "no_improvement"
            tracker.record_reward(
                promoted=is_promoted,
                promoted_as=context.get("classifiers", {}).get(classifier, {}).get(
                    "production", {}
                ).get("version")
                if is_promoted
                else None,
                reason=reason,
            )

            # Optional LLM reflection
            if (
                agent_settings.llm_analysis_enabled
                and agent_settings.anthropic_api_key
                and tracker.experiment is not None
            ):
                try:
                    ReflectorClass = _get_reflector_class()
                    reflector = ReflectorClass(
                        api_key=agent_settings.anthropic_api_key,
                        model=agent_settings.llm_analysis_model,
                    )
                    reflection = reflector.reflect(tracker.experiment)
                    tracker.record_reflection(reflection)
                    logger.info(f"Recorded reflection for {exp_id}")
                except Exception as e:
                    logger.warning(f"Failed to reflect on {exp_id}: {e}")

            tracker.complete()
            finalized.append(exp_id)
            logger.info(f"Finalized experiment {exp_id}")

            # Update heuristic counters from decisions recorded during experiment
            try:
                from src.experiment_log.cli import update_heuristic_outcome
                decisions = tracker.store.load_decisions(exp_id)
                for dec in decisions:
                    if dec.outcome_success is not None:
                        update_heuristic_outcome(
                            classifier,
                            dec.trigger_description,
                            dec.outcome_success,
                            store=tracker.store,
                        )
            except Exception as e:
                logger.warning(f"Failed to update heuristics for {exp_id}: {e}")

        except Exception as e:
            logger.warning(f"Failed to finalize experiment {exp_id}: {e}")

    return {"experiments_finalized": finalized}


@WorkflowRegistry.register
class ModelTrainingWorkflow(Workflow):
    """Model training workflow with manual notebook execution.

    This workflow exports training data, pauses for manual notebook
    execution, then compares and optionally promotes new models.

    Steps:
    1. Export training data (fp + ep datasets)
    2. Check data quality (min records, class balance) + create experiments
    3. Notify user and pause for notebook execution
    4. [User runs notebooks manually]
    5. Compare new models to production + record observations
    6. Prompt for promotion approval
    7. Promote approved models
    8. Trigger Cloud Run deployment (via GitHub Actions)
    9. Finalize experiment entries (reward, reflection, completion)
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
            # Note: This step pauses explicitly after execution (to send email first)
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
            # Note: This step pauses explicitly after determining models_to_promote
        ),
        StepDefinition(
            name="promote_model",
            description="Promote approved models to production",
            handler=promote_model,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="trigger_deployment",
            description="Trigger Cloud Run deployment via GitHub Actions",
            handler=trigger_deployment,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="finalize_experiments",
            description="Finalize experiment log entries with reward and reflection",
            handler=finalize_experiments,
        ),
    ]
