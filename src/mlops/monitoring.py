"""Evidently-based drift detection and monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import mlops_settings
from .reference_data import load_prediction_logs, load_reference_dataset

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Results of a drift analysis."""

    classifier_type: str
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    threshold: float
    details: dict[str, Any]
    report_path: Path | None = None


class DriftMonitor:
    """Monitor for prediction drift using Evidently.

    Gracefully degrades to legacy KS test when Evidently is disabled.
    """

    def __init__(self, classifier_type: str):
        """Initialize drift monitor.

        Args:
            classifier_type: Type of classifier (fp, ep, esg)
        """
        self.classifier_type = classifier_type
        self.enabled = mlops_settings.evidently_enabled
        self.threshold = mlops_settings.drift_threshold
        self._evidently = None

        if self.enabled:
            self._setup_evidently()

    def _setup_evidently(self) -> None:
        """Initialize Evidently components."""
        try:
            from evidently import ColumnMapping
            from evidently.metrics import (
                ColumnDriftMetric,
                DatasetDriftMetric,
                DatasetMissingValuesMetric,
            )
            from evidently.report import Report

            self._evidently = {
                "Report": Report,
                "ColumnMapping": ColumnMapping,
                "ColumnDriftMetric": ColumnDriftMetric,
                "DatasetDriftMetric": DatasetDriftMetric,
                "DatasetMissingValuesMetric": DatasetMissingValuesMetric,
            }
            logger.info("Evidently initialized successfully")
        except ImportError:
            logger.warning("Evidently not installed, using legacy drift detection")
            self.enabled = False

    def check_drift(
        self,
        current_data: pd.DataFrame | None = None,
        reference_data: pd.DataFrame | None = None,
        days: int = 7,
        save_report: bool = True,
        from_database: bool = False,
    ) -> DriftReport:
        """Check for prediction drift.

        Args:
            current_data: Current prediction data (loads from logs if None)
            reference_data: Reference data (loads from file if None)
            days: Days of current data to analyze
            save_report: Whether to save HTML report
            from_database: If True, load predictions from database instead of files

        Returns:
            DriftReport with results
        """
        # Load data if not provided
        if current_data is None:
            current_data = load_prediction_logs(
                self.classifier_type,
                days=days,
                from_database=from_database,
            )

        if reference_data is None:
            try:
                reference_data = load_reference_dataset(self.classifier_type)
            except FileNotFoundError:
                logger.warning("No reference data found, using first half of current data")
                midpoint = len(current_data) // 2
                reference_data = current_data.iloc[:midpoint]
                current_data = current_data.iloc[midpoint:]

        if current_data.empty or reference_data.empty:
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details={"error": "Insufficient data for drift analysis"},
            )

        # Run drift analysis
        if self.enabled:
            return self._evidently_drift_check(
                current_data, reference_data, save_report
            )
        else:
            return self._legacy_drift_check(current_data, reference_data)

    def _evidently_drift_check(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        save_report: bool,
    ) -> DriftReport:
        """Run Evidently-based drift detection.

        Args:
            current_data: Current prediction data
            reference_data: Reference data
            save_report: Whether to save HTML report

        Returns:
            DriftReport with results
        """
        Report = self._evidently["Report"]
        ColumnDriftMetric = self._evidently["ColumnDriftMetric"]
        DatasetDriftMetric = self._evidently["DatasetDriftMetric"]
        DatasetMissingValuesMetric = self._evidently["DatasetMissingValuesMetric"]

        # Determine columns to analyze
        numeric_columns = []
        for col in ["probability", "text_length"]:
            if col in current_data.columns and col in reference_data.columns:
                numeric_columns.append(col)

        if not numeric_columns:
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details={"error": "No numeric columns available for drift detection"},
            )

        # Build report with metrics
        metrics = [DatasetDriftMetric(), DatasetMissingValuesMetric()]
        for col in numeric_columns:
            metrics.append(ColumnDriftMetric(column_name=col))

        report = Report(metrics=metrics)
        report.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        # Extract results
        report_dict = report.as_dict()
        details = {}
        drift_scores = []

        for metric_result in report_dict.get("metrics", []):
            metric_id = metric_result.get("metric", "")
            result = metric_result.get("result", {})

            if "DatasetDriftMetric" in metric_id:
                details["dataset_drift"] = result.get("dataset_drift", False)
                details["drift_share"] = result.get("drift_share", 0)
                drift_scores.append(result.get("drift_share", 0))

            elif "ColumnDriftMetric" in metric_id:
                col_name = result.get("column_name", "unknown")
                details[f"{col_name}_drift"] = result.get("drift_detected", False)
                details[f"{col_name}_drift_score"] = result.get("drift_score", 0)
                if result.get("drift_detected"):
                    drift_scores.append(result.get("drift_score", 0))

        # Overall drift score (max of individual scores)
        overall_drift = max(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift > self.threshold

        # Save report
        report_path = None
        if save_report:
            reports_dir = mlops_settings.get_reports_dir(self.classifier_type)
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            logger.info(f"Saved drift report to {report_path}")

        # Add data stats
        details["reference_size"] = len(reference_data)
        details["current_size"] = len(current_data)
        if "probability" in current_data.columns:
            details["current_prob_mean"] = float(current_data["probability"].mean())
            details["reference_prob_mean"] = float(reference_data["probability"].mean())

        return DriftReport(
            classifier_type=self.classifier_type,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=overall_drift,
            threshold=self.threshold,
            details=details,
            report_path=report_path,
        )

    def _legacy_drift_check(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
    ) -> DriftReport:
        """Legacy drift detection using scipy KS test.

        Args:
            current_data: Current prediction data
            reference_data: Reference data

        Returns:
            DriftReport with results
        """
        from scipy import stats

        details = {}
        drift_scores = []

        # Check probability distribution
        if "probability" in current_data.columns and "probability" in reference_data.columns:
            ks_stat, p_value = stats.ks_2samp(
                reference_data["probability"],
                current_data["probability"],
            )
            details["probability_ks_statistic"] = float(ks_stat)
            details["probability_p_value"] = float(p_value)
            drift_scores.append(ks_stat)

        # Check prediction rate
        if "prediction" in current_data.columns and "prediction" in reference_data.columns:
            ref_rate = reference_data["prediction"].mean()
            curr_rate = current_data["prediction"].mean()
            rate_diff = abs(curr_rate - ref_rate)
            details["reference_prediction_rate"] = float(ref_rate)
            details["current_prediction_rate"] = float(curr_rate)
            details["prediction_rate_diff"] = float(rate_diff)
            drift_scores.append(rate_diff)

        # Overall drift score
        overall_drift = max(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift > self.threshold

        details["reference_size"] = len(reference_data)
        details["current_size"] = len(current_data)

        return DriftReport(
            classifier_type=self.classifier_type,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=overall_drift,
            threshold=self.threshold,
            details=details,
        )


def run_drift_analysis(
    classifier_type: str,
    days: int = 7,
    save_report: bool = True,
    send_alert: bool = True,
    from_database: bool = False,
) -> DriftReport:
    """Run drift analysis for a classifier.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        days: Days of current data to analyze
        save_report: Whether to save HTML report
        send_alert: Whether to send alert if drift detected
        from_database: If True, load predictions from database instead of files

    Returns:
        DriftReport with results
    """
    monitor = DriftMonitor(classifier_type)
    report = monitor.check_drift(days=days, save_report=save_report, from_database=from_database)

    if report.drift_detected and send_alert:
        from .alerts import send_drift_alert

        send_drift_alert(
            classifier_type=classifier_type,
            drift_score=report.drift_score,
            threshold=report.threshold,
            details=report.details,
        )

    return report
