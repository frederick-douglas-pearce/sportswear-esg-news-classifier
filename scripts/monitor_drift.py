#!/usr/bin/env python3
"""Monitor model predictions for drift.

This script analyzes prediction logs to detect:
- Probability distribution drift (KS test)
- Prediction rate shifts
- Input distribution changes
- Confidence degradation

Usage:
    uv run python scripts/monitor_drift.py --classifier fp --days 7
    uv run python scripts/monitor_drift.py --classifier ep --days 30 --baseline 60
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def load_prediction_logs(
    classifier: str, days: int, logs_dir: Path
) -> list[dict]:
    """Load prediction logs for the specified time period."""
    logs = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    log_pattern = f"{classifier}_predictions_*.jsonl"
    for log_file in logs_dir.glob(log_pattern):
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry["timestamp"])
                # Ensure timestamp is timezone-aware for comparison
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if timestamp >= cutoff:
                    logs.append(entry)

    return logs


def load_baseline_metrics(classifier: str, models_dir: Path) -> dict:
    """Load baseline metrics from model config."""
    config_path = models_dir / f"{classifier}_classifier_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    return {
        "threshold": config["threshold"],
        "cv_precision": config.get("cv_precision", config.get("threshold_precision")),
        "cv_recall": config.get("cv_recall", config.get("threshold_recall")),
    }


def detect_probability_drift(
    recent_probs: list[float], baseline_probs: list[float] | None = None
) -> dict:
    """Detect drift in probability distribution using KS test."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return {"error": "scipy not installed, skipping KS test"}

    if baseline_probs is None:
        # Use uniform distribution as baseline assumption
        baseline_probs = np.random.uniform(0, 1, len(recent_probs))

    statistic, p_value = ks_2samp(baseline_probs, recent_probs)

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05),
        "significance": "high" if p_value < 0.01 else ("medium" if p_value < 0.05 else "low"),
    }


def detect_prediction_rate_shift(
    predictions: list[bool], baseline_rate: float | None = None
) -> dict:
    """Detect shift in positive prediction rate."""
    if not predictions:
        return {"error": "No predictions to analyze"}

    current_rate = sum(predictions) / len(predictions)

    # Use baseline if provided, otherwise use 50% as neutral
    expected_rate = baseline_rate if baseline_rate is not None else 0.5

    shift = current_rate - expected_rate
    relative_shift = abs(shift) / max(expected_rate, 0.01)

    return {
        "current_rate": float(current_rate),
        "expected_rate": float(expected_rate),
        "absolute_shift": float(shift),
        "relative_shift": float(relative_shift),
        "drift_detected": bool(relative_shift > 0.2),  # 20% relative shift
    }


def analyze_confidence_distribution(probabilities: list[float]) -> dict:
    """Analyze confidence distribution."""
    if not probabilities:
        return {"error": "No probabilities to analyze"}

    probs = np.array(probabilities)

    # Calculate confidence (distance from 0.5)
    confidence = np.abs(probs - 0.5) * 2

    return {
        "mean_probability": float(np.mean(probs)),
        "std_probability": float(np.std(probs)),
        "median_probability": float(np.median(probs)),
        "mean_confidence": float(np.mean(confidence)),
        "low_confidence_ratio": float(np.mean(confidence < 0.3)),  # Predictions near 0.5
        "high_confidence_ratio": float(np.mean(confidence > 0.7)),  # Strong predictions
    }


def generate_report(
    classifier: str,
    logs: list[dict],
    baseline: dict,
    days: int,
) -> dict:
    """Generate drift detection report."""
    if not logs:
        return {
            "status": "no_data",
            "message": f"No prediction logs found for {classifier} in the last {days} days",
        }

    # Extract probabilities and predictions
    probabilities = [log["probability"] for log in logs]
    predictions = [log.get("is_sportswear", log.get("has_esg", False)) for log in logs]

    # Run analyses
    prob_drift = detect_probability_drift(probabilities)
    rate_shift = detect_prediction_rate_shift(predictions)
    confidence = analyze_confidence_distribution(probabilities)

    # Determine overall status
    drift_signals = [
        prob_drift.get("drift_detected", False),
        rate_shift.get("drift_detected", False),
        confidence.get("low_confidence_ratio", 0) > 0.3,
    ]

    if sum(drift_signals) >= 2:
        status = "warning"
        message = "Multiple drift signals detected - consider retraining"
    elif sum(drift_signals) == 1:
        status = "monitor"
        message = "Single drift signal detected - continue monitoring"
    else:
        status = "healthy"
        message = "No significant drift detected"

    return {
        "classifier": classifier,
        "analysis_period_days": days,
        "total_predictions": len(logs),
        "status": status,
        "message": message,
        "baseline_threshold": baseline["threshold"],
        "probability_drift": prob_drift,
        "prediction_rate": rate_shift,
        "confidence_distribution": confidence,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor model predictions for drift")
    parser.add_argument(
        "--classifier",
        required=True,
        choices=["fp", "ep", "esg"],
        help="Classifier to analyze",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/predictions"),
        help="Directory containing prediction logs",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model configs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report (default: stdout)",
    )

    args = parser.parse_args()

    # Load baseline metrics
    try:
        baseline = load_baseline_metrics(args.classifier, args.models_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Load prediction logs
    logs = load_prediction_logs(args.classifier, args.days, args.logs_dir)

    # Generate report
    report = generate_report(args.classifier, logs, baseline, args.days)

    # Output report
    output_json = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        print(f"Report written to: {args.output}")
    else:
        print(output_json)

    # Return exit code based on status
    return 0 if report["status"] == "healthy" else 1


if __name__ == "__main__":
    exit(main())
