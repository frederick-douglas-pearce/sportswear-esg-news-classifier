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
    uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30

    # Send alert if drift detected
    uv run python scripts/monitor_drift.py --classifier fp --from-db --alert

    # Legacy: Load from log files (for local API testing)
    uv run python scripts/monitor_drift.py --classifier fp --logs-dir logs/predictions
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops import (
    DriftMonitor,
    create_reference_dataset,
    get_reference_stats,
    mlops_settings,
    run_drift_analysis,
    send_drift_alert,
)


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
    if report.drift_detected:
        print("⚠️  ACTION REQUIRED: Drift detected - consider retraining")
    else:
        print("✅ Status: Healthy - no significant drift detected")


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
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    if args.reference_stats:
        stats = get_reference_stats(args.classifier)
        if stats:
            print(f"\nReference Dataset Stats ({args.classifier}):")
            print("-" * 40)
            print(json.dumps(stats, indent=2, default=str))
        else:
            print(f"No reference dataset found for {args.classifier}")
        return 0

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
        print(f"Error running drift analysis: {e}")
        return 1

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
            "details": report.details,
            "report_path": str(report.report_path) if report.report_path else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"\nJSON report written to: {args.output}")

    # Send alert if requested and drift detected
    if args.alert and report.drift_detected:
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

    # Return exit code based on drift status
    return 1 if report.drift_detected else 0


if __name__ == "__main__":
    sys.exit(main())
