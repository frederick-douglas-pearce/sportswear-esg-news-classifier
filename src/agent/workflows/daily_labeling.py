"""Daily data collection and labeling workflow."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from src.data_collection.database import db

from ..config import agent_settings
from ..runner import run_label_articles
from .base import StepDefinition, Workflow, WorkflowRegistry

logger = logging.getLogger(__name__)


def check_collection_status(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Check collection status from last 24 hours.

    Queries the database for recent collection runs and pending articles.
    """
    logger.info("Checking collection status from last 24 hours")

    with db.get_session() as session:
        # Get collection stats from last 24 hours
        result = session.execute(
            text("""
                SELECT
                    COUNT(*) as runs,
                    COALESCE(SUM(articles_fetched), 0) as fetched,
                    COALESCE(SUM(articles_scraped), 0) as scraped,
                    COALESCE(SUM(articles_scrape_failed), 0) as failed
                FROM collection_runs
                WHERE started_at >= NOW() - INTERVAL '24 hours'
            """)
        )
        row = result.fetchone()
        collection_stats = {
            "collection_runs_24h": row.runs,
            "articles_fetched_24h": row.fetched,
            "articles_scraped_24h": row.scraped,
            "articles_failed_24h": row.failed,
        }

        # Get pending article count
        pending_count = session.execute(
            text("SELECT COUNT(*) FROM articles WHERE labeling_status = 'pending'")
        ).scalar()
        collection_stats["articles_pending"] = pending_count

        # Get labeling status breakdown
        result = session.execute(
            text("""
                SELECT labeling_status, COUNT(*) as count
                FROM articles
                GROUP BY labeling_status
            """)
        )
        status_counts = {row.labeling_status: row.count for row in result}
        collection_stats["labeling_status_breakdown"] = status_counts

    logger.info(
        f"Collection stats: {collection_stats['collection_runs_24h']} runs, "
        f"{collection_stats['articles_fetched_24h']} fetched, "
        f"{collection_stats['articles_pending']} pending"
    )

    return collection_stats


def run_labeling(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run labeling pipeline on all pending articles."""
    pending_count = context.get("articles_pending", 0)

    if pending_count == 0:
        logger.info("No pending articles to label")
        return {"labeling_skipped": True, "reason": "no_pending_articles"}

    logger.info(f"Running labeling on {pending_count} pending articles")

    # Run labeling without batch limit (process all pending)
    result = run_label_articles(
        batch_size=None,  # Process all pending
        dry_run=context.get("dry_run", False),
    )

    labeling_result = {
        "labeling_success": result.success,
        "labeling_exit_code": result.exit_code,
        "labeling_duration_seconds": result.duration_seconds,
    }

    # Parse labeling output for stats
    if result.stdout:
        labeling_result["labeling_output"] = _parse_labeling_output(result.stdout)

    if not result.success:
        labeling_result["labeling_error"] = result.stderr[:1000]
        logger.error(f"Labeling failed: {result.stderr[:500]}")

    return labeling_result


def _parse_labeling_output(output: str) -> dict[str, Any]:
    """Parse labeling script output for statistics."""
    stats = {}
    lines = output.strip().split("\n")

    for line in lines:
        # Look for key metrics in output
        if "Articles processed:" in line:
            stats["articles_processed"] = _extract_number(line)
        elif "Articles labeled:" in line:
            stats["articles_labeled"] = _extract_number(line)
        elif "Articles skipped:" in line:
            stats["articles_skipped"] = _extract_number(line)
        elif "False positives:" in line:
            stats["false_positives"] = _extract_number(line)
        elif "Articles failed:" in line:
            stats["articles_failed"] = _extract_number(line)
        elif "LLM API calls:" in line:
            stats["llm_calls"] = _extract_number(line)
        elif "Estimated cost:" in line:
            # Extract cost like "$0.8032"
            try:
                cost_str = line.split("$")[1].strip()
                stats["estimated_cost_usd"] = float(cost_str)
            except (IndexError, ValueError):
                pass
        elif "FP classifier calls:" in line:
            stats["fp_classifier_calls"] = _extract_number(line)
        elif "Skipped LLM:" in line:
            stats["fp_skipped_llm"] = _extract_number(line)

    return stats


def _extract_number(line: str) -> int:
    """Extract integer from a line like 'Key: 123'."""
    try:
        parts = line.split(":")
        if len(parts) >= 2:
            return int(parts[-1].strip())
    except ValueError:
        pass
    return 0


def check_labeling_quality(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Check labeling quality metrics and detect anomalies."""
    labeling_output = context.get("labeling_output", {})

    if context.get("labeling_skipped"):
        return {"quality_check_skipped": True}

    articles_processed = labeling_output.get("articles_processed", 0)
    articles_failed = labeling_output.get("articles_failed", 0)
    articles_labeled = labeling_output.get("articles_labeled", 0)
    false_positives = labeling_output.get("false_positives", 0)

    quality_metrics = {
        "articles_processed": articles_processed,
        "articles_failed": articles_failed,
        "articles_labeled": articles_labeled,
        "false_positives": false_positives,
    }

    # Calculate rates
    if articles_processed > 0:
        error_rate = articles_failed / articles_processed
        fp_rate = false_positives / articles_processed
        label_rate = articles_labeled / articles_processed

        quality_metrics["error_rate"] = error_rate
        quality_metrics["fp_rate"] = fp_rate
        quality_metrics["label_rate"] = label_rate

        # Flag anomalies
        quality_metrics["high_error_rate"] = error_rate > 0.10
        quality_metrics["high_fp_rate"] = fp_rate > 0.50  # More than 50% FP is unusual

        if quality_metrics["high_error_rate"]:
            logger.warning(f"High error rate detected: {error_rate:.1%}")
        if quality_metrics["high_fp_rate"]:
            logger.warning(f"High false positive rate detected: {fp_rate:.1%}")

    return quality_metrics


def run_llm_analysis(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Run LLM analysis on labeling results.

    This step is configurable:
    - If AGENT_LLM_ANALYSIS=true and AGENT_LLM_ERROR_THRESHOLD=0.0: always run
    - If AGENT_LLM_ERROR_THRESHOLD > 0: only run if error_rate exceeds threshold
    """
    if context.get("labeling_skipped"):
        return {"llm_analysis_skipped": True, "reason": "labeling_skipped"}

    # Check if LLM analysis is enabled
    if not agent_settings.llm_analysis_enabled:
        return {"llm_analysis_skipped": True, "reason": "disabled"}

    # Check error threshold
    error_rate = context.get("error_rate", 0)
    threshold = agent_settings.llm_error_threshold

    if threshold > 0 and error_rate < threshold:
        logger.info(
            f"Skipping LLM analysis: error_rate {error_rate:.1%} < threshold {threshold:.1%}"
        )
        return {
            "llm_analysis_skipped": True,
            "reason": "below_threshold",
            "error_rate": error_rate,
            "threshold": threshold,
        }

    # For now, prepare context for LLM analysis but don't call LLM
    # This will be implemented in Phase 4
    logger.info("LLM analysis would run here (placeholder)")

    analysis_context = {
        "llm_analysis_pending": True,
        "articles_processed": context.get("articles_processed", 0),
        "articles_labeled": context.get("articles_labeled", 0),
        "articles_failed": context.get("articles_failed", 0),
        "error_rate": context.get("error_rate", 0),
        "fp_rate": context.get("fp_rate", 0),
    }

    # TODO: Phase 4 - Implement actual LLM analysis
    # - Query recent labeling results from database
    # - Sample false positives and errors
    # - Send to Claude for analysis
    # - Parse and store recommendations

    return {
        "llm_analysis_completed": False,
        "llm_analysis_placeholder": True,
        "analysis_context": analysis_context,
    }


def generate_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Generate daily summary report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_name": workflow.name,
        "dry_run": context.get("dry_run", False),
    }

    # Collection stats
    report["collection"] = {
        "runs_24h": context.get("collection_runs_24h", 0),
        "articles_fetched": context.get("articles_fetched_24h", 0),
        "articles_scraped": context.get("articles_scraped_24h", 0),
    }

    # Labeling stats
    if context.get("labeling_skipped"):
        report["labeling"] = {"skipped": True, "reason": context.get("reason")}
    else:
        labeling_output = context.get("labeling_output", {})
        report["labeling"] = {
            "articles_processed": labeling_output.get("articles_processed", 0),
            "articles_labeled": labeling_output.get("articles_labeled", 0),
            "articles_skipped": labeling_output.get("articles_skipped", 0),
            "false_positives": labeling_output.get("false_positives", 0),
            "articles_failed": labeling_output.get("articles_failed", 0),
            "estimated_cost_usd": labeling_output.get("estimated_cost_usd", 0),
            "fp_classifier_calls": labeling_output.get("fp_classifier_calls", 0),
            "fp_skipped_llm": labeling_output.get("fp_skipped_llm", 0),
        }

    # Quality metrics
    report["quality"] = {
        "error_rate": context.get("error_rate", 0),
        "fp_rate": context.get("fp_rate", 0),
        "high_error_rate": context.get("high_error_rate", False),
        "high_fp_rate": context.get("high_fp_rate", False),
    }

    # LLM analysis
    report["llm_analysis"] = {
        "enabled": agent_settings.llm_analysis_enabled,
        "completed": context.get("llm_analysis_completed", False),
        "skipped": context.get("llm_analysis_skipped", False),
    }

    logger.info(f"Generated report: {report}")
    return {"report": report}


def save_report(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Save daily report to file."""
    import json
    from pathlib import Path

    report = context.get("report", {})

    # Save to reports directory
    reports_dir = agent_settings.project_root / "reports" / "daily_labeling"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Use date from report or current date
    report_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    report_file = reports_dir / f"report_{report_date}.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {report_file}")

    return {"report_saved": True, "report_file": str(report_file)}


def send_notification(workflow: Workflow, context: dict[str, Any]) -> dict[str, Any]:
    """Send notification with summary via configured channels."""
    from ..notifications import send_labeling_summary

    report = context.get("report", {})

    # Always log summary to console
    _log_summary(report)

    # Extract labeling stats for notification
    labeling = report.get("labeling", {})
    if labeling.get("skipped"):
        logger.info("Labeling was skipped - sending minimal notification")
        # Send minimal notification for skipped runs
        result = send_labeling_summary(
            articles_processed=0,
            articles_labeled=0,
            false_positives=0,
            articles_failed=0,
            additional_details={"skipped": True, "reason": labeling.get("reason")},
        )
    else:
        result = send_labeling_summary(
            articles_processed=labeling.get("articles_processed", 0),
            articles_labeled=labeling.get("articles_labeled", 0),
            false_positives=labeling.get("false_positives", 0),
            articles_failed=labeling.get("articles_failed", 0),
            estimated_cost=labeling.get("estimated_cost_usd"),
            additional_details={
                "collection_runs_24h": report.get("collection", {}).get("runs_24h", 0),
                "articles_fetched_24h": report.get("collection", {}).get("articles_fetched", 0),
                "error_rate": report.get("quality", {}).get("error_rate", 0),
                "fp_rate": report.get("quality", {}).get("fp_rate", 0),
            },
        )

    # Determine what channels were used
    channels_used = [k for k, v in result.items() if v]

    return {
        "notification_sent": len(channels_used) > 0,
        "channels": channels_used,
        "notification_result": result,
    }


def _log_summary(report: dict[str, Any]) -> None:
    """Log a human-readable summary."""
    print("\n" + "=" * 60)
    print("DAILY LABELING WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Generated: {report.get('generated_at', 'N/A')}")
    print(f"Dry Run: {report.get('dry_run', False)}")

    collection = report.get("collection", {})
    print(f"\nCollection (24h):")
    print(f"  Runs: {collection.get('runs_24h', 0)}")
    print(f"  Fetched: {collection.get('articles_fetched', 0)}")
    print(f"  Scraped: {collection.get('articles_scraped', 0)}")

    labeling = report.get("labeling", {})
    if labeling.get("skipped"):
        print(f"\nLabeling: Skipped ({labeling.get('reason', 'unknown')})")
    else:
        print(f"\nLabeling:")
        print(f"  Processed: {labeling.get('articles_processed', 0)}")
        print(f"  Labeled: {labeling.get('articles_labeled', 0)}")
        print(f"  Skipped: {labeling.get('articles_skipped', 0)}")
        print(f"  False Positives: {labeling.get('false_positives', 0)}")
        print(f"  Failed: {labeling.get('articles_failed', 0)}")
        print(f"  Cost: ${labeling.get('estimated_cost_usd', 0):.4f}")
        if labeling.get("fp_classifier_calls"):
            print(f"  FP Classifier Calls: {labeling.get('fp_classifier_calls', 0)}")
            print(f"  FP Skipped LLM: {labeling.get('fp_skipped_llm', 0)}")

    quality = report.get("quality", {})
    print(f"\nQuality:")
    print(f"  Error Rate: {quality.get('error_rate', 0):.1%}")
    print(f"  FP Rate: {quality.get('fp_rate', 0):.1%}")
    if quality.get("high_error_rate"):
        print("  ⚠️  HIGH ERROR RATE DETECTED")
    if quality.get("high_fp_rate"):
        print("  ⚠️  HIGH FALSE POSITIVE RATE DETECTED")

    print("=" * 60 + "\n")


@WorkflowRegistry.register
class DailyLabelingWorkflow(Workflow):
    """Daily data collection check and labeling workflow.

    Steps:
    1. Check collection status from last 24 hours
    2. Run labeling on all pending articles
    3. Check labeling quality metrics
    4. Run LLM analysis (if enabled)
    5. Generate daily summary report
    6. Save report to file
    7. Send notification
    """

    name = "daily_labeling"
    description = "Check collection status, label pending articles, and generate report"

    steps = [
        StepDefinition(
            name="check_collection_status",
            description="Query collection runs and pending articles from last 24 hours",
            handler=check_collection_status,
        ),
        StepDefinition(
            name="run_labeling",
            description="Run labeling pipeline on all pending articles",
            handler=run_labeling,
        ),
        StepDefinition(
            name="check_labeling_quality",
            description="Check labeling quality metrics and detect anomalies",
            handler=check_labeling_quality,
        ),
        StepDefinition(
            name="run_llm_analysis",
            description="Run LLM analysis on labeling results (if enabled)",
            handler=run_llm_analysis,
        ),
        StepDefinition(
            name="generate_report",
            description="Generate daily summary report",
            handler=generate_report,
        ),
        StepDefinition(
            name="save_report",
            description="Save report to file",
            handler=save_report,
        ),
        StepDefinition(
            name="send_notification",
            description="Send notification via configured channels",
            handler=send_notification,
            skip_on_dry_run=True,
        ),
    ]
