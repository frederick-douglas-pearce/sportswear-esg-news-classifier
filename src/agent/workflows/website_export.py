"""Website feed export workflow."""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.data_collection.text_normalize import find_illegal_chars

from ..config import agent_settings
from ..notifications import Notification, NotificationManager, NotificationType
from ..runner import run_export_website_feed, run_script
from .base import StepDefinition, Workflow, WorkflowRegistry

logger = logging.getLogger(__name__)


def export_feeds(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Export JSON and Atom feeds for the website."""
    if context.get("prepare_ready") is False:
        return {"export_skipped": True, "reason": "prepare_failed"}

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
        raw = None
        try:
            with open(json_path, encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            validation_result["json_valid"] = True
            # The feed's top level is a dict ({"articles": [...], ...}); count the
            # articles list. (Previously this counted len(data) only for lists, so
            # the dict feed always reported 0 articles.)
            articles = data.get("articles", []) if isinstance(data, dict) else data
            validation_result["json_article_count"] = len(articles) if isinstance(articles, list) else 0
            logger.info(f"JSON valid: {validation_result['json_article_count']} articles")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            validation_result["json_valid"] = False
            validation_result["validation_passed"] = False
            validation_result["errors"].append(f"JSON error: {e}")
            logger.error(f"JSON validation failed: {e}")
            raw = None  # skip the YAML check below; the JSON error is the root cause

        # Jekyll parses _data/*.json with its YAML parser (Ruby Psych), which
        # rejects control characters that JSON itself permits (notably the C1
        # range U+0080-U+009F from Windows-1252 mojibake). A feed that is valid
        # JSON but invalid YAML breaks the site build, so mirror Jekyll's parser
        # here and fail validation before the push. PyYAML reproduces Psych's
        # rejection of these characters; the explicit scan adds a precise message.
        if raw is not None:
            illegal = sorted(find_illegal_chars(raw))
            yaml_error = None
            try:
                yaml.safe_load(raw)
            except yaml.YAMLError as e:
                yaml_error = str(e).replace("\n", " ")
            if illegal or yaml_error:
                if illegal:
                    codes = ", ".join(f"U+{ord(c):04X}" for c in illegal)
                    msg = f"feed contains control characters Jekyll's YAML parser rejects: {codes}"
                else:
                    msg = f"feed is valid JSON but not YAML-parseable (Jekyll will fail): {yaml_error}"
                validation_result["yaml_valid"] = False
                validation_result["validation_passed"] = False
                validation_result["errors"].append(f"YAML error: {msg}")
                logger.error(f"YAML validation failed: {msg}")
            else:
                validation_result["yaml_valid"] = True

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


# Files written by export_feeds and committed by commit_and_push.
# Kept as a module-level constant so prepare_worktree and commit_and_push
# can both reason about the same paths.
_FEED_FILES = ("_data/esg_news.json", "assets/feeds/esg_news.atom")


class WorkflowError(RuntimeError):
    """Raised by the final notification step so the workflow ends in FAILED state."""


def _check_branch(
    website_repo: Path, expected: str
) -> tuple[bool, str | None, str | None, str]:
    """Run `git symbolic-ref --short HEAD` and return (ok, branch, error_code, stderr).

    - ok=True, branch=<name>, error_code=None, stderr=""  -> on expected branch
    - ok=False, branch=<name>, error_code="wrong_branch"   -> on a different branch
    - ok=False, branch=None,  error_code="git_branch_check_failed" -> command itself failed
    """
    result = run_script(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=website_repo,
        retries=0,
        dry_run=False,
    )
    if not result.success:
        return False, None, "git_branch_check_failed", result.stderr
    current = result.stdout.strip()
    if current != expected:
        return False, current, "wrong_branch", ""
    return True, current, None, ""


def prepare_worktree(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Verify the worktree is on the expected branch and fast-forward it.

    Runs BEFORE export_feeds so the FF pull never has to contend with the
    files export_feeds is about to overwrite. Failures set
    `prepare_ready: False` and an `error` code; downstream steps short-circuit
    on that flag, and `send_error_notification` raises so the workflow ends
    in FAILED state.
    """
    website_repo = agent_settings.website_repo_path
    if not website_repo:
        return {"prepare_skipped": True, "reason": "website_repo_not_configured"}

    dry_run = context.get("dry_run", False)
    if dry_run:
        logger.info("Dry run - skipping worktree preparation")
        return {"prepare_skipped": True, "reason": "dry_run"}

    expected = agent_settings.website_expected_branch

    ok, current_branch, error_code, branch_stderr = _check_branch(website_repo, expected)
    if not ok:
        if error_code == "wrong_branch":
            logger.error(
                f"Refusing to proceed: website repo is on '{current_branch}', "
                f"expected '{expected}'. The export workflow must run in a "
                f"worktree pinned to '{expected}' (see CLAUDE.md)."
            )
        else:
            logger.error(f"Could not determine current branch: {branch_stderr}")
        return {
            "prepare_ready": False,
            "error": error_code,
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": branch_stderr[:500] if branch_stderr else "",
        }

    pull_result = run_script(
        ["git", "pull", "--ff-only", "origin", expected],
        cwd=website_repo,
        retries=1,
        dry_run=False,
        timeout=60,
    )
    if not pull_result.success:
        logger.error(
            f"Fast-forward pull of origin/{expected} failed: {pull_result.stderr}"
        )
        return {
            "prepare_ready": False,
            "error": "git_pull_ff_failed",
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": pull_result.stderr[:500],
        }

    return {
        "prepare_ready": True,
        "current_branch": current_branch,
        "expected_branch": expected,
    }


def commit_and_push(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Stage, commit, and push the exported feed files.

    Assumes `prepare_worktree` has already verified the branch and pulled.
    Performs a defense-in-depth branch check in case anything raced between
    the two steps, then stages only the feed files, commits, and pushes with
    an explicit `origin HEAD:<branch>` refspec. Every error return includes
    `current_branch`, `expected_branch`, and a `git_stderr` snippet for the
    notification step to render.
    """
    if context.get("export_skipped") or context.get("validation_skipped"):
        return {"git_skipped": True, "reason": "export_skipped"}

    if not context.get("validation_passed", False):
        return {"git_skipped": True, "reason": "validation_failed"}

    if context.get("prepare_ready") is False:
        return {"git_skipped": True, "reason": "prepare_failed"}

    website_repo = agent_settings.website_repo_path
    if not website_repo:
        return {"git_skipped": True, "reason": "website_repo_not_configured"}

    dry_run = context.get("dry_run", False)
    if dry_run:
        logger.info("Dry run - skipping git commit and push")
        return {"git_skipped": True, "reason": "dry_run"}

    expected = agent_settings.website_expected_branch

    # Defense-in-depth branch check in case state changed between
    # prepare_worktree and this step (e.g., an operator touched the worktree).
    ok, current_branch, error_code, branch_stderr = _check_branch(website_repo, expected)
    if not ok:
        return {
            "git_success": False,
            "error": error_code,
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": branch_stderr[:500] if branch_stderr else "",
        }

    # Scope status to the feed files we actually care about. Any other dirty
    # state (operator poking around) is intentionally ignored here.
    status_result = run_script(
        ["git", "status", "--porcelain", "--", *_FEED_FILES],
        cwd=website_repo,
        retries=0,
        dry_run=False,
    )
    if not status_result.success:
        logger.error(f"git status failed: {status_result.stderr}")
        return {
            "git_success": False,
            "error": "git_status_failed",
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": status_result.stderr[:500],
        }

    if not status_result.stdout.strip():
        logger.info("No changes to commit")
        return {
            "git_skipped": True,
            "reason": "no_changes",
            "current_branch": current_branch,
            "expected_branch": expected,
        }

    # The feed JSON is emitted in Prettier-compatible formatting by
    # export_website_feed.write_json, so no `prettier --write` pass is needed
    # here (it silently no-op'd in the worktree anyway, lacking node_modules).
    add_result = run_script(
        ["git", "add", *_FEED_FILES],
        cwd=website_repo,
        retries=0,
        dry_run=False,
    )
    if not add_result.success:
        logger.error(f"Git add failed: {add_result.stderr}")
        return {
            "git_success": False,
            "error": "git_add_failed",
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": add_result.stderr[:500],
        }

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit_result = run_script(
        ["git", "commit", "-m", f"Update ESG news feed - {today}"],
        cwd=website_repo,
        retries=0,
        dry_run=False,
    )
    if not commit_result.success:
        logger.error(f"Git commit failed: {commit_result.stderr}")
        return {
            "git_success": False,
            "error": "git_commit_failed",
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": commit_result.stderr[:500],
        }

    # Push with an explicit refspec, no upstream tracking dependency.
    # No retry: the commit stays local, the next scheduled run handles
    # any transient or non-FF condition cleanly via prepare_worktree.
    push_result = run_script(
        ["git", "push", "origin", f"HEAD:{expected}"],
        cwd=website_repo,
        retries=0,
        dry_run=False,
        timeout=60,
    )
    if not push_result.success:
        logger.error(f"Git push failed: {push_result.stderr}")
        return {
            "git_success": False,
            "error": "git_push_failed",
            "current_branch": current_branch,
            "expected_branch": expected,
            "git_stderr": push_result.stderr[:500],
        }

    logger.info("Successfully committed and pushed to website repo")
    return {
        "git_success": True,
        "commit_message": f"Update ESG news feed - {today}",
        "branch": expected,
        "current_branch": current_branch,
        "expected_branch": expected,
    }


def _describe_git_error(context: dict[str, Any], git_error: str) -> str:
    """Build a human-readable description of a git failure code.

    Includes a stderr snippet when present so transient/auth/dirty-tree
    failures don't all look like 'branch diverged'.
    """
    current = context.get("current_branch")
    expected = context.get("expected_branch")
    stderr = (context.get("git_stderr") or "").strip()
    stderr_suffix = f" git stderr: {stderr}" if stderr else ""

    if git_error == "wrong_branch":
        return (
            f"website repo is on '{current}', expected '{expected}'. "
            f"Workflow must run in a worktree pinned to the expected "
            f"branch (see CLAUDE.md)."
        )
    if git_error == "git_branch_check_failed":
        return (
            f"could not determine current branch of website repo "
            f"(expected '{expected}'). This usually means the worktree is "
            f"in detached HEAD, mid-rebase, or `.git` is corrupted.{stderr_suffix}"
        )
    if git_error == "git_pull_ff_failed":
        return (
            f"fast-forward pull of origin/{expected} failed. Causes include "
            f"divergence from origin, uncommitted local changes blocking the "
            f"merge, network failure, or authentication failure. The next "
            f"scheduled run will retry if the cause was transient.{stderr_suffix}"
        )
    if git_error == "git_status_failed":
        return (
            f"`git status` failed in the website repo. A stale `.git/index.lock` "
            f"from an interrupted git process is the most common cause; "
            f"inspect and remove it if no git process is running.{stderr_suffix}"
        )
    if git_error == "git_add_failed":
        return f"`git add` of the feed files failed.{stderr_suffix}"
    if git_error == "git_commit_failed":
        return (
            f"`git commit` failed — usually because the staged set was empty "
            f"or commit signing/hooks rejected the change.{stderr_suffix}"
        )
    if git_error == "git_push_failed":
        return (
            f"`git push origin HEAD:{expected}` failed. If the remote moved "
            f"between the FF pull and the push (TOCTOU), the next scheduled "
            f"run will rebase and push.{stderr_suffix}"
        )
    return f"unknown git error code '{git_error}'.{stderr_suffix}"


def send_error_notification(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Aggregate failures across prior steps, notify, and raise on failure.

    Raises WorkflowError when any prior step recorded a failure so the
    workflow base class marks the run as FAILED — without this, returning a
    notification dict makes the step COMPLETED and the whole workflow status
    incorrectly reports clean success.
    """
    errors = []
    wrong_branch_only = True  # toggled off when any other failure is also present

    # prepare_worktree
    if context.get("prepare_ready") is False:
        prep_error = context.get("error", "Unknown prepare error")
        errors.append(f"Worktree preparation failed: {_describe_git_error(context, prep_error)}")
        if prep_error != "wrong_branch":
            wrong_branch_only = False

    # export
    if not context.get("export_success", True) and not context.get("export_skipped"):
        errors.append(f"Export failed: {context.get('export_error', 'Unknown error')}")
        wrong_branch_only = False

    # validation (default False: a missing flag is treated as failure, not success)
    if not context.get("validation_passed", False) and not context.get("validation_skipped"):
        validation_errors = context.get("errors", [])
        errors.append(f"Validation failed: {', '.join(validation_errors)}")
        wrong_branch_only = False

    # scorecard snapshot
    if context.get("scorecard_saved") is False:
        scorecard_error = context.get("error", "Unknown error")
        errors.append(f"Scorecard snapshot save failed: {scorecard_error}")
        wrong_branch_only = False

    # commit/push
    if context.get("git_success") is False:
        git_error = context.get("error", "Unknown error")
        errors.append(f"Git operation failed: {_describe_git_error(context, git_error)}")
        if git_error != "wrong_branch":
            wrong_branch_only = False

    if not errors:
        logger.info("Website export completed successfully - no notification needed")
        return {"notification_sent": False, "reason": "no_errors"}

    error_message = "\n".join(f"• {e}" for e in errors)
    details: dict[str, Any] = {
        "prepare_ready": context.get("prepare_ready", "N/A"),
        "export_success": context.get("export_success", "N/A"),
        "validation_passed": context.get("validation_passed", "N/A"),
        "scorecard_saved": context.get("scorecard_saved", "N/A"),
        "git_success": context.get("git_success", "N/A"),
    }

    # Only surface feed paths in details when push actually attempted/succeeded.
    # When the only failure is wrong_branch, the feed files were written locally
    # but never pushed — including the paths reads as misleadingly successful.
    if not wrong_branch_only:
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

    manager = NotificationManager()
    result = manager.send(notification)
    channels_used = [k for k, v in result.items() if v]
    logger.warning(f"Sent error notification via: {channels_used}")

    # Raise so Workflow.run records the run as FAILED. The notification has
    # already been sent; the exception only changes workflow-level status.
    raise WorkflowError(
        f"Website export workflow failed with {len(errors)} error(s); "
        f"notification dispatched via {channels_used or 'no channels'}."
    )


@WorkflowRegistry.register
class WebsiteExportWorkflow(Workflow):
    """Website feed export workflow.

    Steps:
    1. Prepare worktree (assert branch, fast-forward pull) - BEFORE any writes
    2. Export JSON and Atom feeds
    3. Save scorecard snapshot to database
    4. Validate exported files
    5. Commit and push to website repository
    6. Send error notification (raises so workflow status reflects failure)
    """

    name = "website_export"
    description = "Export ESG news feeds to website repository"

    steps = [
        StepDefinition(
            name="prepare_worktree",
            description="Assert worktree branch and fast-forward pull before any writes",
            handler=prepare_worktree,
            skip_on_dry_run=True,
        ),
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
            description="Aggregate errors, notify, and raise so workflow status reflects failure",
            handler=send_error_notification,
            skip_on_dry_run=True,
        ),
    ]
