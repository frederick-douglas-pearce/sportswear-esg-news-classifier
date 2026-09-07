#!/usr/bin/env python3
"""Monitor model predictions for drift.

This script analyzes prediction logs to detect:
- Probability distribution drift
- Prediction rate shifts
- Input distribution changes
- Data quality issues

Uses Evidently AI when enabled, falls back to legacy KS test otherwise.

Usage:
    # Basic drift check from database (recommended for production)
    uv run python scripts/monitor_drift.py --classifier fp --from-db

    # Extended analysis period
    uv run python scripts/monitor_drift.py --classifier fp --from-db --days 30

    # Generate HTML report (requires EVIDENTLY_ENABLED=true)
    uv run python scripts/monitor_drift.py --classifier fp --from-db --html-report

    # Create reference dataset from database predictions
    uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 90

    # Send alert if drift detected
    uv run python scripts/monitor_drift.py --classifier fp --from-db --alert

    # Legacy: Load from log files (for local API testing)
    uv run python scripts/monitor_drift.py --classifier fp --logs-dir logs/predictions
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops import (
    EXIT_DRIFT_DETECTED,
    EXIT_INDETERMINATE,
    EXIT_NO_DRIFT,
    DriftMonitor,
    create_reference_dataset,
    get_reference_stats,
    mlops_settings,
    run_drift_analysis,
    send_drift_alert,
)

# Label for the machine-readable summary block. The agent workflow reads the
# summary via ScriptResult.parsed_output rather than scraping the
# human-readable report above it -- a scraper finds nothing when a check never
# ran, and absence read as healthy for 231 days (issue #71).
#
# The label sits on its OWN line and the JSON on the next, because the runner's
# `_parse_json_from_output` takes whole lines: a `label: {...}` line fails to
# parse, the workflow sees no summary, and every verdict degrades to `unknown`.
SUMMARY_LABEL = "--- drift summary (machine-readable) ---"


def print_report(report, verbose: bool = False) -> None:
    """Print drift report to console."""
    print("=" * 60)
    print(f"DRIFT MONITORING REPORT - {report.classifier_type.upper()}")
    print("=" * 60)
    print(f"\nTimestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Drift Detected: {'YES' if report.drift_detected else 'NO'}")
    print(f"Drift Score: {report.drift_score:.4f} (threshold: {report.threshold:.4f})")

    if report.report_path:
        print(f"\nHTML Report: {report.report_path}")

    if verbose and report.details:
        print("\nDetails:")
        print("-" * 40)
        for key, value in report.details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # Status indicator
    if report.indeterminate:
        reason = report.details.get("error", "no verdict produced")
        print(f"❓ Status: INDETERMINATE - drift could not be assessed ({reason})")
    elif report.drift_detected:
        print("⚠️  ACTION REQUIRED: Drift detected - consider retraining")
    else:
        print("✅ Status: Healthy - no significant drift detected")


def exit_code_for(report) -> int:
    """Map a DriftReport to this script's exit-code contract.

    An indeterminate report is checked FIRST: it carries drift_detected=False
    and drift_score=0.0 because nothing was measured, so testing drift first
    would report "no drift" for a check that never ran (issue #71).
    """
    if report.indeterminate:
        return EXIT_INDETERMINATE
    return EXIT_DRIFT_DETECTED if report.drift_detected else EXIT_NO_DRIFT


def print_summary_json(report, exit_code: int) -> None:
    """Emit the machine-readable summary the agent workflow consumes.

    Printed last, and as a bare single-line JSON object, so
    `_parse_json_from_output` (which scans stdout backwards for the last
    balanced brace block, line by line) finds this and nothing else.
    """
    summary = {
        "classifier": report.classifier_type,
        "exit_code": exit_code,
        "indeterminate": report.indeterminate,
        "drift_detected": report.drift_detected,
        "drift_score": report.drift_score,
        "threshold": report.threshold,
        "error": report.details.get("error") if report.details else None,
    }
    print(SUMMARY_LABEL)
    print(json.dumps(summary))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor model predictions for drift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--classifier", "-c",
        required=True,
        choices=["fp", "ep", "esg"],
        help="Classifier to analyze",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days of recent data to analyze (default: 7)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/predictions"),
        help="Directory containing prediction logs (ignored if --from-db is used)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load predictions from database instead of log files (recommended for production)",
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report (requires EVIDENTLY_ENABLED=true)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for JSON report (default: stdout)",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Send webhook alert if drift detected",
    )
    parser.add_argument(
        "--create-reference",
        action="store_true",
        help="Create reference dataset from historical data",
    )
    parser.add_argument(
        "--reference-stats",
        action="store_true",
        help="Show reference dataset statistics",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # Handle reference dataset operations
    if args.create_reference:
        source = "database" if args.from_db else f"logs in {args.logs_dir}"
        print(f"Creating reference dataset for {args.classifier} from {source}...")
        try:
            path = create_reference_dataset(
                classifier_type=args.classifier,
                logs_dir=args.logs_dir,
                days=args.days,
                from_database=args.from_db,
            )
            print(f"Reference dataset created: {path}")
            return EXIT_NO_DRIFT
        except Exception as e:
            # EXIT_INDETERMINATE, not 1: under this script's contract 1 means
            # "drift detected", and it is non-retryable -- so a failed
            # reference build would be logged by scripts/cron_monitor.sh as
            # "WARNING: Drift detected". stderr for the same reason the
            # analysis path uses it (issue #71). `Exception`, not `ValueError`:
            # a database error or an OSError writing the parquet would
            # otherwise escape and exit 1 by default.
            print(f"Error creating reference dataset: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return EXIT_INDETERMINATE

    if args.reference_stats:
        stats = get_reference_stats(args.classifier)
        if stats:
            print(f"\nReference Dataset Stats ({args.classifier}):")
            print("-" * 40)
            print(json.dumps(stats, indent=2, default=str))
            return EXIT_NO_DRIFT
        # "The thing you asked about does not exist" is not success.
        print(f"No reference dataset found for {args.classifier}", file=sys.stderr)
        return EXIT_INDETERMINATE

    # Check if Evidently is enabled for HTML reports
    if args.html_report and not mlops_settings.evidently_enabled:
        print("Warning: HTML reports require EVIDENTLY_ENABLED=true")
        print("Continuing with JSON output only...")

    # Run drift analysis
    if args.verbose:
        source = "database" if args.from_db else f"log files in {args.logs_dir}"
        print(f"Analyzing {args.classifier} predictions from last {args.days} days...")
        print(f"Data source: {source}")
        print(f"Evidently enabled: {mlops_settings.evidently_enabled}")

    try:
        report = run_drift_analysis(
            classifier_type=args.classifier,
            days=args.days,
            save_report=args.html_report,
            send_alert=False,  # Handle alert separately
            from_database=args.from_db,
        )
    except Exception as e:
        # stderr, not stdout: the agent workflow logs the command's stderr, so
        # printing here to stdout produced 221 log lines reading
        # "FP drift check failed: " with the diagnosis nowhere (issue #71).
        # traceback goes with it -- the message alone ("'novelty_score'") does
        # not say which frame lacked the column.
        print(f"Error running drift analysis: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return EXIT_INDETERMINATE

    # Print report
    print_report(report, verbose=args.verbose)

    # Output JSON if requested
    if args.output:
        report_dict = {
            "classifier_type": report.classifier_type,
            "timestamp": report.timestamp.isoformat(),
            "drift_detected": report.drift_detected,
            "drift_score": report.drift_score,
            "threshold": report.threshold,
            "indeterminate": report.indeterminate,
            "details": report.details,
            "report_path": str(report.report_path) if report.report_path else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nJSON report written to: {args.output}")

    # Send alert if requested and drift detected. Derived from the same
    # exit-code mapping the process returns, so there is exactly one definition
    # of "drift detected" in this script; the agent workflow raises its own
    # alert for an indeterminate verdict.
    if args.alert and exit_code_for(report) == EXIT_DRIFT_DETECTED:
        if mlops_settings.alert_webhook_url:
            success = send_drift_alert(
                classifier_type=args.classifier,
                drift_score=report.drift_score,
                threshold=report.threshold,
                details=report.details,
            )
            if success:
                print("Alert sent successfully")
            else:
                print("Failed to send alert")
        else:
            print("Warning: No ALERT_WEBHOOK_URL configured")

    # Return exit code per the contract in src/mlops/exit_codes.py
    exit_code = exit_code_for(report)
    print_summary_json(report, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
