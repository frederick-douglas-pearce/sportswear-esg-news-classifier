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

    # Run labeling with explicit batch size to process all pending
    # Note: batch_size=None would use the script's default (10), so we
    # explicitly pass the pending count to ensure all articles are processed
    result = run_label_articles(
        batch_size=pending_count,  # Process all pending articles
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
    in_error_breakdown = False
    error_types = {}

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
        elif "Continued to LLM:" in line:
            stats["fp_continued_llm"] = _extract_number(line)
        elif "Est. LLM cost saved:" in line:
            # Extract cost like "$0.1234"
            try:
                cost_str = line.split("$")[1].strip()
                stats["fp_cost_saved_usd"] = float(cost_str)
            except (IndexError, ValueError):
                pass
        elif "Error Type Breakdown:" in line:
            in_error_breakdown = True
        elif in_error_breakdown:
            # Parse error type lines like "    connection: 5"
            line_stripped = line.strip()
            if line_stripped and ":" in line_stripped and not line_stripped.startswith("-"):
                try:
                    error_type, count = line_stripped.split(":", 1)
                    error_type = error_type.strip()
                    count = int(count.strip())
                    if error_type and count > 0:
                        error_types[error_type] = count
                except (ValueError, IndexError):
                    # End of error type section
                    in_error_breakdown = False
            elif line_stripped.startswith("-") or "Individual Errors:" in line:
                # End of error type section
                in_error_breakdown = False

    if error_types:
        stats["error_types"] = error_types

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

    Uses Claude Sonnet to analyze recent labeling results and identify:
    - Potential labeling errors
    - Patterns in false positives
    - Improvement suggestions
    """
    if context.get("labeling_skipped"):
        return {"llm_analysis_skipped": True, "reason": "labeling_skipped"}

    # Check if LLM analysis is enabled
    if not agent_settings.llm_analysis_enabled:
        return {"llm_analysis_skipped": True, "reason": "disabled"}

    # Check if API key is available
    if not agent_settings.anthropic_api_key:
        logger.warning("LLM analysis skipped: ANTHROPIC_API_KEY not set")
        return {"llm_analysis_skipped": True, "reason": "no_api_key"}

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

    # Import LLM analysis module
    from ..llm import LabelingAnalyzer, get_recent_labeling_samples

    # Get labeling output stats
    labeling_output = context.get("labeling_output", {})
    stats = {
        "articles_processed": labeling_output.get("articles_processed", 0),
        "articles_labeled": labeling_output.get("articles_labeled", 0),
        "articles_skipped": labeling_output.get("articles_skipped", 0),
        "false_positives": labeling_output.get("false_positives", 0),
        "articles_failed": labeling_output.get("articles_failed", 0),
        "error_rate": context.get("error_rate", 0),
        "fp_rate": context.get("fp_rate", 0),
    }

    # Get recent labeling samples from database
    logger.info("Fetching recent labeling samples for LLM analysis...")
    labeled_sample, fp_sample, skipped_sample = get_recent_labeling_samples(
        days=1, sample_size=10
    )

    logger.info(
        f"Samples: {len(labeled_sample)} labeled, {len(fp_sample)} FP, {len(skipped_sample)} skipped"
    )

    # Skip if no samples to analyze
    if not labeled_sample and not fp_sample and not skipped_sample:
        logger.info("No recent samples to analyze, skipping LLM analysis")
        return {
            "llm_analysis_skipped": True,
            "reason": "no_samples",
        }

    # Run LLM analysis
    logger.info("Running Claude analysis on labeling results...")
    try:
        analyzer = LabelingAnalyzer(
            api_key=agent_settings.anthropic_api_key,
            model=agent_settings.llm_analysis_model,
        )
        result = analyzer.analyze_labeling_results(
            stats=stats,
            labeled_sample=labeled_sample,
            fp_sample=fp_sample,
            skipped_sample=skipped_sample,
        )

        if result.success:
            logger.info(
                f"LLM analysis completed: {result.input_tokens} input, "
                f"{result.output_tokens} output tokens"
            )
            return {
                "llm_analysis_completed": True,
                "llm_analysis": result.analysis,
                "llm_tokens": {
                    "input": result.input_tokens,
                    "output": result.output_tokens,
                },
                "llm_model": result.model,
            }
        else:
            logger.error(f"LLM analysis failed: {result.error}")
            return {
                "llm_analysis_completed": False,
                "llm_analysis_error": result.error,
                "llm_analysis_truncated": result.truncated,
            }

    except Exception as e:
        logger.error(f"LLM analysis exception: {e}")
        return {
            "llm_analysis_completed": False,
            "llm_analysis_error": str(e),
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
            "fp_cost_saved_usd": labeling_output.get("fp_cost_saved_usd", 0),
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

    # Surface failure detail so a truncated/errored analysis is visible in the
    # report rather than rendering identically to "not run" (see issue #42).
    if not context.get("llm_analysis_completed") and context.get("llm_analysis_error"):
        report["llm_analysis"]["error"] = context.get("llm_analysis_error")
        report["llm_analysis"]["truncated"] = context.get("llm_analysis_truncated", False)

    # Include LLM analysis details if available
    if context.get("llm_analysis_completed"):
        llm_analysis = context.get("llm_analysis", {})
        report["llm_analysis"]["summary"] = llm_analysis.get("summary")
        report["llm_analysis"]["overall_assessment"] = llm_analysis.get("overall_assessment")
        report["llm_analysis"]["potential_errors_count"] = len(llm_analysis.get("potential_errors", []))
        report["llm_analysis"]["patterns_detected_count"] = len(llm_analysis.get("patterns_detected", []))
        report["llm_analysis"]["improvement_suggestions_count"] = len(llm_analysis.get("improvement_suggestions", []))
        report["llm_analysis"]["tokens"] = context.get("llm_tokens")
        # Store full analysis for detailed review
        report["llm_analysis"]["details"] = llm_analysis

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
        # Build additional details
        additional_details = {
            "articles_skipped": labeling.get("articles_skipped", 0),
            "articles_pending_at_start": context.get("articles_pending", 0),
            "collection_runs_24h": report.get("collection", {}).get("runs_24h", 0),
            "articles_fetched_24h": report.get("collection", {}).get("articles_fetched", 0),
            "articles_scraped_24h": report.get("collection", {}).get("articles_scraped", 0),
            "error_rate": report.get("quality", {}).get("error_rate", 0),
            "fp_rate": report.get("quality", {}).get("fp_rate", 0),
        }

        # Add FP classifier stats if available
        if labeling.get("fp_classifier_calls", 0) > 0:
            additional_details["fp_classifier_calls"] = labeling.get("fp_classifier_calls", 0)
            additional_details["fp_skipped_llm"] = labeling.get("fp_skipped_llm", 0)
            if labeling.get("fp_cost_saved_usd", 0) > 0:
                additional_details["fp_cost_saved"] = f"${labeling.get('fp_cost_saved_usd', 0):.4f}"

        result = send_labeling_summary(
            articles_processed=labeling.get("articles_processed", 0),
            articles_labeled=labeling.get("articles_labeled", 0),
            false_positives=labeling.get("false_positives", 0),
            articles_failed=labeling.get("articles_failed", 0),
            estimated_cost=labeling.get("estimated_cost_usd"),
            additional_details=additional_details,
        )

    # Determine what channels were used
    channels_used = [k for k, v in result.items() if v]

    # The whole purpose of this run is to deliver the morning report, so a run
    # that could not deliver it must not report success. "console" is the
    # no-channel-configured fallback (always succeeds) and is not a delivery
    # failure. If real channels (email/webhook) were attempted and every one
    # failed -- typically DNS/network down at cron time -- raise so the
    # workflow is marked FAILED and surfaces in `agent status` and the cron
    # exit code, instead of silently archiving as "completed". See issue #51.
    attempted_channels = [k for k in result if k != "console"]
    if attempted_channels and not channels_used:
        raise RuntimeError(
            "Notification delivery failed on all configured channels "
            f"({', '.join(sorted(attempted_channels))}); the report was generated "
            "and saved but not delivered. This is usually DNS/network being "
            "unavailable at cron time."
        )

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
            if labeling.get("fp_cost_saved_usd"):
                print(f"  FP Cost Saved: ${labeling.get('fp_cost_saved_usd', 0):.4f}")

    quality = report.get("quality", {})
    print(f"\nQuality:")
    print(f"  Error Rate: {quality.get('error_rate', 0):.1%}")
    print(f"  FP Rate: {quality.get('fp_rate', 0):.1%}")
    if quality.get("high_error_rate"):
        print("  ⚠️  HIGH ERROR RATE DETECTED")
    if quality.get("high_fp_rate"):
        print("  ⚠️  HIGH FALSE POSITIVE RATE DETECTED")

    # LLM Analysis summary
    llm = report.get("llm_analysis", {})
    print(f"\nLLM Analysis:")
    if llm.get("skipped"):
        print(f"  Skipped: {llm.get('reason', 'unknown')}")
    elif llm.get("completed"):
        print(f"  Status: Completed")
        if llm.get("summary"):
            print(f"  Summary: {llm.get('summary')}")
        print(f"  Potential Errors: {llm.get('potential_errors_count', 0)}")
        print(f"  Patterns Detected: {llm.get('patterns_detected_count', 0)}")
        print(f"  Improvement Suggestions: {llm.get('improvement_suggestions_count', 0)}")
        tokens = llm.get("tokens", {})
        if tokens:
            print(f"  Tokens: {tokens.get('input', 0)} in, {tokens.get('output', 0)} out")
    else:
        print(f"  Status: Not run")

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
