"""Website feed export workflow."""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import agent_settings
from ..notifications import Notification, NotificationManager, NotificationType
from ..runner import run_export_website_feed, run_script
from .base import StepDefinition, Workflow, WorkflowRegistry

logger = logging.getLogger(__name__)


def export_feeds(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Export JSON and Atom feeds for the website."""
    website_repo = agent_settings.website_repo_path

    if not website_repo:
        logger.warning("AGENT_WEBSITE_REPO_PATH not configured")
        return {
            "export_skipped": True,
            "reason": "website_repo_not_configured",
        }

    json_output = str(website_repo / "_data" / "esg_news.json")
    atom_output = str(website_repo / "assets" / "feeds" / "esg_news.atom")

    logger.info(f"Exporting feeds to {website_repo}")

    result = run_export_website_feed(
        json_output=json_output,
        atom_output=atom_output,
        format="both",
    )

    export_result = {
        "export_success": result.success,
        "export_exit_code": result.exit_code,
        "export_duration_seconds": result.duration_seconds,
        "json_output": json_output,
        "atom_output": atom_output,
    }

    if not result.success:
        export_result["export_error"] = result.stderr[:1000]
        logger.error(f"Export failed: {result.stderr[:500]}")

    # Parse article count from output
    if result.stdout:
        for line in result.stdout.split("\n"):
            if "articles exported" in line.lower() or "exported" in line.lower():
                try:
                    # Look for number in line
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            export_result["articles_exported"] = int(word)
                            break
                except (ValueError, IndexError):
                    pass

    return export_result


def save_scorecard_snapshot(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Save scorecard data to the database for historical tracking.

    This step reads the scorecard from the exported JSON file and saves it
    to the scorecard_snapshots and scorecard_brand_scores tables.
    """
    if context.get("export_skipped"):
        return {"scorecard_skipped": True, "reason": "export_skipped"}

    if not context.get("export_success"):
        return {"scorecard_skipped": True, "reason": "export_failed"}

    json_path = context.get("json_output")
    if not json_path:
        return {"scorecard_skipped": True, "reason": "no_json_output"}

    dry_run = context.get("dry_run", False)
    if dry_run:
        logger.info("Dry run - skipping scorecard database save")
        return {"scorecard_skipped": True, "reason": "dry_run"}

    # Read the exported JSON to get scorecard data
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to read JSON for scorecard: {e}")
        return {"scorecard_saved": False, "error": str(e)}

    scorecard_data = data.get("scorecard")
    if not scorecard_data:
        logger.warning("No scorecard data found in exported JSON")
        return {"scorecard_skipped": True, "reason": "no_scorecard_in_json"}

    # Save to database
    try:
        from src.data_collection.database import db
        from src.scorecard.database import scorecard_db

        db.init_db()

        with db.get_session() as session:
            snapshot = scorecard_db.save_scorecard_snapshot(session, scorecard_data)
            brand_count = len(scorecard_data.get("all_brand_scores", []))

            logger.info(
                f"Saved scorecard snapshot: {brand_count} brands, "
                f"period={scorecard_data['period_start']} to {scorecard_data['period_end']}"
            )

            return {
                "scorecard_saved": True,
                "snapshot_id": str(snapshot.id),
                "brand_count": brand_count,
                "period_days": scorecard_data["period_days"],
                "period_start": scorecard_data["period_start"],
                "period_end": scorecard_data["period_end"],
            }

    except Exception as e:
        logger.error(f"Failed to save scorecard snapshot: {e}")
        return {"scorecard_saved": False, "error": str(e)}


def validate_export(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Validate that exported files are valid JSON/XML."""
    import xml.etree.ElementTree as ET

    if context.get("export_skipped"):
        return {"validation_skipped": True}

    json_path = context.get("json_output")
    atom_path = context.get("atom_output")
    validation_result = {"validation_passed": True, "errors": []}

    # Validate JSON
    if json_path:
        try:
            with open(json_path) as f:
                data = json.load(f)
            validation_result["json_valid"] = True
            validation_result["json_article_count"] = len(data) if isinstance(data, list) else 0
            logger.info(f"JSON valid: {validation_result['json_article_count']} articles")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            validation_result["json_valid"] = False
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"JSON error: {e}")
            logger.error(f"JSON validation failed: {e}")

    # Validate Atom XML
    if atom_path:
        try:
            ET.parse(atom_path)
            validation_result["atom_valid"] = True
            logger.info("Atom feed valid")
        except (ET.ParseError, FileNotFoundError) as e:
            validation_result["atom_valid"] = False
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"Atom error: {e}")
            logger.error(f"Atom validation failed: {e}")

    return validation_result


def commit_and_push(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Commit and push changes to the website repository."""
    if context.get("export_skipped") or context.get("validation_skipped"):
        return {"git_skipped": True, "reason": "export_skipped"}

    if not context.get("validation_passed", False):
        return {"git_skipped": True, "reason": "validation_failed"}

    website_repo = agent_settings.website_repo_path
    if not website_repo:
        return {"git_skipped": True, "reason": "website_repo_not_configured"}

    dry_run = context.get("dry_run", False)
    if dry_run:
        logger.info("Dry run - skipping git commit and push")
        return {"git_skipped": True, "reason": "dry_run"}

    # Check if there are changes to commit
    status_result = run_script(
        ["git", "status", "--porcelain"],
        cwd=website_repo,
        retries=0,
    )

    if not status_result.stdout.strip():
        logger.info("No changes to commit")
        return {"git_skipped": True, "reason": "no_changes"}

    # Run prettier to format JSON file before committing
    prettier_result = run_script(
        ["npx", "prettier", "--write", "_data/esg_news.json"],
        cwd=website_repo,
        retries=0,
        timeout=60,
    )

    if not prettier_result.success:
        logger.warning(f"Prettier formatting failed: {prettier_result.stderr}")
        # Continue anyway - prettier failure shouldn't block the export

    # Add files
    add_result = run_script(
        ["git", "add", "_data/esg_news.json", "assets/feeds/esg_news.atom"],
        cwd=website_repo,
        retries=0,
    )

    if not add_result.success:
        logger.error(f"Git add failed: {add_result.stderr}")
        return {"git_success": False, "error": "git_add_failed"}

    # Commit
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit_result = run_script(
        ["git", "commit", "-m", f"Update ESG news feed - {today}"],
        cwd=website_repo,
        retries=0,
    )

    if not commit_result.success:
        logger.error(f"Git commit failed: {commit_result.stderr}")
        return {"git_success": False, "error": "git_commit_failed"}

    # Push
    push_result = run_script(
        ["git", "push"],
        cwd=website_repo,
        retries=1,
        timeout=60,
    )

    if not push_result.success:
        logger.error(f"Git push failed: {push_result.stderr}")
        return {"git_success": False, "error": "git_push_failed"}

    logger.info("Successfully committed and pushed to website repo")
    return {
        "git_success": True,
        "commit_message": f"Update ESG news feed - {today}",
    }


def send_error_notification(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send email notification only if there was an error during export.

    This step checks for failures in previous steps and sends an alert email
    if any issues were detected. Successful exports are silent.
    """
    errors = []

    # Check export step
    if not context.get("export_success", True) and not context.get("export_skipped"):
        errors.append(f"Export failed: {context.get('export_error', 'Unknown error')}")

    # Check validation step
    if not context.get("validation_passed", True) and not context.get("validation_skipped"):
        validation_errors = context.get("errors", [])
        errors.append(f"Validation failed: {', '.join(validation_errors)}")

    # Check git step
    if context.get("git_success") is False:
        errors.append(f"Git operation failed: {context.get('error', 'Unknown error')}")

    # No errors - silent success
    if not errors:
        logger.info("Website export completed successfully - no notification needed")
        return {"notification_sent": False, "reason": "no_errors"}

    # Build error notification
    error_message = "\n".join(f"• {e}" for e in errors)
    details = {
        "export_success": context.get("export_success", "N/A"),
        "validation_passed": context.get("validation_passed", "N/A"),
        "git_success": context.get("git_success", "N/A"),
    }

    if context.get("json_output"):
        details["json_output"] = context.get("json_output")
    if context.get("atom_output"):
        details["atom_output"] = context.get("atom_output")

    notification = Notification(
        notification_type=NotificationType.WORKFLOW_FAILED,
        subject="Website Export Failed",
        message=f"The website export workflow encountered errors:\n\n{error_message}",
        details=details,
        severity="error",
    )

    # Send notification
    manager = NotificationManager()
    result = manager.send(notification)

    channels_used = [k for k, v in result.items() if v]
    logger.warning(f"Sent error notification via: {channels_used}")

    return {
        "notification_sent": True,
        "errors": errors,
        "channels": channels_used,
    }


@WorkflowRegistry.register
class WebsiteExportWorkflow(Workflow):
    """Website feed export workflow.

    Steps:
    1. Export JSON and Atom feeds
    2. Save scorecard snapshot to database
    3. Validate exported files
    4. Commit and push to website repository
    5. Send error notification (only if there was a failure)
    """

    name = "website_export"
    description = "Export ESG news feeds to website repository"

    steps = [
        StepDefinition(
            name="export_feeds",
            description="Export JSON and Atom feeds to website repository",
            handler=export_feeds,
        ),
        StepDefinition(
            name="save_scorecard_snapshot",
            description="Save scorecard data to database for historical tracking",
            handler=save_scorecard_snapshot,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="validate_export",
            description="Validate exported JSON and Atom files",
            handler=validate_export,
        ),
        StepDefinition(
            name="commit_and_push",
            description="Commit and push changes to website repository",
            handler=commit_and_push,
            skip_on_dry_run=True,
        ),
        StepDefinition(
            name="send_error_notification",
            description="Send email notification if export failed",
            handler=send_error_notification,
            skip_on_dry_run=True,
        ),
    ]
