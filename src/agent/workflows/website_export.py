"""Website feed export workflow."""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import agent_settings
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


def validate_export(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Validate that exported files are valid JSON/XML."""
    import json
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


@WorkflowRegistry.register
class WebsiteExportWorkflow(Workflow):
    """Website feed export workflow.

    Steps:
    1. Export JSON and Atom feeds
    2. Validate exported files
    3. Commit and push to website repository
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
    ]
