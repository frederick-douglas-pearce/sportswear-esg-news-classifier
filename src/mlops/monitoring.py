"""Evidently-based drift detection and monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import mlops_settings
from .reference_data import TRACKED_BRANDS, load_prediction_logs, load_reference_dataset

logger = logging.getLogger(__name__)

# Drift detection configuration by feature type
# Keys are column name patterns, values are (method, p_value_threshold)
DRIFT_CONFIG = {
    # Core prediction metrics - standard threshold
    "probability": ("ks", 0.05),
    "prediction": ("chisquare", 0.05),
    # Novelty score - important for detecting novel topics
    "novelty_score": ("ks", 0.05),
    # Brand columns - looser threshold (brands can vary naturally)
    "brand_": ("chisquare", 0.01),
}


# Columns whose absence from the reference is reported. Deliberately the core
# metrics only: `brand_*` columns come and go with the TRACKED_BRANDS list, so
# listing every one of them would bury the signal. Callers describing this
# behaviour must say "a core column", not "a column" (issue #71 review).
CORE_DRIFT_COLUMNS = ["probability", "prediction", "novelty_score"]


def _missing_from_reference(
    current_data: pd.DataFrame, reference_data: pd.DataFrame
) -> list[str]:
    """Core columns the current data has and the reference cannot answer for.

    A reference written against an older schema silently compares whichever
    columns it happens to share and returns a verdict that looks complete.
    Recording the gap is what stops a partial check reading as a whole one.
    """
    return [
        c
        for c in CORE_DRIFT_COLUMNS
        if c in current_data.columns and c not in reference_data.columns
    ]


@dataclass
class DriftReport:
    """Results of a drift analysis.

    `indeterminate` is a first-class field rather than a key in `details`
    because it drives control flow: `scripts/monitor_drift.py` maps it to
    `EXIT_INDETERMINATE`, and the run archives it produces are read
    programmatically downstream. `details` is a free-form grab-bag -- it also
    carries `columns_checked`, `reference_size` and per-column p-values -- so
    inferring "was this a verdict at all?" from the presence of an `error` key
    in it puts a load-bearing signal somewhere a reader has to know to look.

    When `indeterminate` is True, `drift_detected` and `drift_score` carry
    their zero values because nothing was measured; they are NOT evidence of
    health and must not be read as such (issue #71).
    """

    classifier_type: str
    timestamp: datetime
    drift_detected: bool
    drift_score: float
    threshold: float
    details: dict[str, Any]
    report_path: Path | None = None
    indeterminate: bool = False

    def __post_init__(self) -> None:
        """Coerce numpy scalars to built-ins so this report can be serialized.

        `drift_detected` is computed as `overall_drift > self.threshold`, and
        when `overall_drift` came out of scipy/pandas that comparison yields a
        `numpy.bool_`. **`numpy.bool_` does not subclass `bool`** -- unlike
        `numpy.float64`, which does subclass `float` and so passes unnoticed --
        and `json.dumps` refuses it with
        `TypeError: Object of type bool is not JSON serializable`. The type's
        `__name__` is even "bool", so it does not look wrong when printed.

        That crashed `print_summary_json` on every successful run of the legacy
        path, and it reaches `json.dump` on the `--output` path too. Coercing
        here rather than at each call site means a construction site added
        later cannot reintroduce it.
        """
        self.drift_detected = bool(self.drift_detected)
        self.indeterminate = bool(self.indeterminate)
        self.drift_score = float(self.drift_score)
        self.threshold = float(self.threshold)


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
        """Initialize Evidently components using v0.7+ API."""
        try:
            from evidently import Report
            from evidently.metrics import ValueDrift

            self._evidently = {
                "Report": Report,
                "ValueDrift": ValueDrift,
            }
            logger.info("Evidently v0.7+ initialized successfully")
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
            # This branch returns before either checker runs, so the
            # `drift_detected=False` below is fabricated rather than measured.
            # Before #71 that was indistinguishable from a real clean result:
            # the script reported exit 0 and the workflow read it as healthy,
            # which is how the EP classifier -- which has never made a single
            # prediction -- passed on every run.
            logger.warning(
                f"{self.classifier_type}: insufficient data for drift analysis "
                f"(reference={len(reference_data)} rows, current={len(current_data)} rows)"
            )
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details={
                    "error": "Insufficient data for drift analysis",
                    "reference_size": len(reference_data),
                    "current_size": len(current_data),
                },
                indeterminate=True,
            )

        # Run drift analysis
        if self.enabled:
            return self._evidently_drift_check(
                current_data, reference_data, save_report
            )
        else:
            return self._legacy_drift_check(current_data, reference_data)

    def _get_drift_config(self, column_name: str) -> tuple[str, float]:
        """Get drift detection method and threshold for a column.

        Args:
            column_name: Name of the column

        Returns:
            Tuple of (method, p_value_threshold)
        """
        for pattern, config in DRIFT_CONFIG.items():
            if column_name.startswith(pattern) or column_name == pattern:
                return config
        # Default config
        return ("ks", 0.05)

    def _evidently_drift_check(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        save_report: bool,
    ) -> DriftReport:
        """Run Evidently-based drift detection using v0.7+ API.

        Args:
            current_data: Current prediction data
            reference_data: Reference data
            save_report: Whether to save HTML report

        Returns:
            DriftReport with results
        """
        Report = self._evidently["Report"]
        ValueDrift = self._evidently["ValueDrift"]

        # Determine columns to analyze: probability, prediction, novelty, and brand columns
        columns_to_check = []

        # Core metrics
        for col in ["probability", "prediction", "novelty_score"]:
            if col in current_data.columns and col in reference_data.columns:
                columns_to_check.append(col)

        # Brand columns
        brand_cols = [c for c in current_data.columns if c.startswith("brand_")]
        for col in brand_cols:
            if col in reference_data.columns:
                columns_to_check.append(col)

        if not columns_to_check:
            # Reference and current share no comparable column -- typically a
            # reference written against an older schema. Nothing was measured,
            # so this is indeterminate, not healthy (issue #71).
            logger.warning(
                f"{self.classifier_type}: no columns in common between reference "
                f"and current data; drift cannot be assessed"
            )
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details={
                    "error": "No columns available for drift detection",
                    "reference_columns": sorted(reference_data.columns),
                    "current_columns": sorted(current_data.columns),
                },
                indeterminate=True,
            )

        # Say so when the reference cannot answer for a column the current data
        # has. Fixing the novelty_score KeyError (issue #71) removed a crash
        # that had been accidentally surfacing exactly this: a reference written
        # against an older schema now compares the columns it happens to share
        # and returns a verdict that looks complete. The comparison is still
        # useful, so this does not make the report indeterminate -- but a
        # partial check reported as a whole one is the defect this epic is
        # about, so it is recorded and logged rather than left silent.
        missing_from_reference = _missing_from_reference(current_data, reference_data)
        if missing_from_reference:
            logger.warning(
                f"{self.classifier_type}: reference dataset lacks "
                f"{', '.join(missing_from_reference)} - drift for those columns "
                f"was NOT assessed. Regenerate the reference with "
                f"--create-reference to compare them."
            )

        # Filter to only the columns we want to analyze
        ref_filtered = reference_data[columns_to_check].copy()
        curr_filtered = current_data[columns_to_check].copy()

        # Build metrics with per-column configuration
        metrics = []
        for col in columns_to_check:
            method, threshold = self._get_drift_config(col)
            metrics.append(ValueDrift(column=col, method=method, threshold=threshold))

        # Build and run report
        report = Report(metrics=metrics)
        snapshot = report.run(
            reference_data=ref_filtered,
            current_data=curr_filtered,
        )

        # Extract results from the new API structure
        report_dict = snapshot.dict()
        details = {
            "columns_checked": columns_to_check,
            "columns_missing_from_reference": missing_from_reference,
            "core_metrics_drifted": [],
            "brand_metrics_drifted": [],
        }
        core_drifted = 0
        brand_drifted = 0
        total_core = 0
        total_brand = 0

        for metric in report_dict.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            config = metric.get("config", {})
            value = metric.get("value")

            if "ValueDrift" in metric_name:
                col_name = config.get("column", "unknown")
                p_value_threshold = config.get("threshold", 0.05)
                p_value = value if isinstance(value, (int, float)) else 1.0

                col_drift = p_value < p_value_threshold
                details[f"{col_name}_drift"] = col_drift
                details[f"{col_name}_p_value"] = float(p_value)

                if col_name.startswith("brand_"):
                    total_brand += 1
                    if col_drift:
                        brand_drifted += 1
                        details["brand_metrics_drifted"].append(col_name)
                else:
                    total_core += 1
                    if col_drift:
                        core_drifted += 1
                        details["core_metrics_drifted"].append(col_name)

        # No CORE metric was assessed, so the drift score below -- which is
        # `core_drift_score`, and only that -- would be a fabricated 0.0.
        #
        # Keyed on `total_core`, not on `total_core == 0 and total_brand == 0`.
        # The stricter form let a reference sharing only `brand_*` columns
        # through: `columns_to_check` was non-empty so the guard above did not
        # fire, one brand metric came back so this one did not either, and the
        # report said HEALTHY with a 0.0 score while the log said probability
        # and prediction had not been assessed. `_legacy_drift_check` calls that
        # same input indeterminate; the two paths must agree.
        #
        # It also covers the case this guard was added for: a snapshot whose
        # shape or `metric_name` spelling changed, yielding no readable metric
        # at all (this code targets "v0.7+", so that has happened once).
        if total_core == 0:
            logger.warning(
                f"{self.classifier_type}: no core drift metric could be read "
                f"(columns offered: {columns_to_check}, brand metrics read: "
                f"{total_brand}); drift cannot be assessed"
            )
            details["error"] = "No core drift metrics could be read from the Evidently report"
            details["reference_size"] = len(reference_data)
            details["current_size"] = len(current_data)
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details=details,
                indeterminate=True,
            )

        # Calculate drift scores
        core_drift_score = core_drifted / total_core if total_core > 0 else 0.0
        brand_drift_score = brand_drifted / total_brand if total_brand > 0 else 0.0

        # Overall drift: triggered if any core metric drifts OR significant brand drift
        # Core metrics (probability, prediction) are more important
        drift_detected = core_drifted > 0 or brand_drift_score > self.threshold

        details["core_drift_score"] = core_drift_score
        details["brand_drift_score"] = brand_drift_score
        details["core_drifted_count"] = core_drifted
        details["brand_drifted_count"] = brand_drifted

        # Save report
        report_path = None
        if save_report:
            reports_dir = mlops_settings.get_reports_dir(self.classifier_type)
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            snapshot.save_html(str(report_path))
            logger.info(f"Saved drift report to {report_path}")

        # Add data stats.
        #
        # Every read below guards ITS OWN frame. Reading `reference_data[col]`
        # under `if col in current_data.columns` is what raised
        # `KeyError: 'novelty_score'` once `novelty_score` was added to
        # `classifier_predictions` (2026-01-25) while the reference parquet,
        # written 2026-01-17, lacked the column (issue #71).
        #
        # THREE sites had that shape, not one: novelty_score was merely the one
        # that fired, because a reference sharing only `brand_*` columns would
        # have died on `reference_prob_mean` first. An earlier revision of this
        # fix corrected novelty_score alone and claimed the others were already
        # symmetric; they were not.
        details["reference_size"] = len(reference_data)
        details["current_size"] = len(current_data)
        if "probability" in current_data.columns:
            details["current_prob_mean"] = float(current_data["probability"].mean())
        if "probability" in reference_data.columns:
            details["reference_prob_mean"] = float(reference_data["probability"].mean())
        if "prediction" in current_data.columns:
            details["current_prediction_rate"] = float(current_data["prediction"].mean())
        if "prediction" in reference_data.columns:
            details["reference_prediction_rate"] = float(reference_data["prediction"].mean())
        if "novelty_score" in current_data.columns:
            # Filter out NaN values for novelty stats
            current_novelty = current_data["novelty_score"].dropna()
            if len(current_novelty) > 0:
                details["current_novelty_mean"] = float(current_novelty.mean())
                details["current_novelty_p90"] = float(current_novelty.quantile(0.9))
                details["current_high_novelty_pct"] = float((current_novelty > 0.7).mean())
        if "novelty_score" in reference_data.columns:
            reference_novelty = reference_data["novelty_score"].dropna()
            if len(reference_novelty) > 0:
                details["reference_novelty_mean"] = float(reference_novelty.mean())
                details["reference_novelty_p90"] = float(reference_novelty.quantile(0.9))

        return DriftReport(
            classifier_type=self.classifier_type,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=core_drift_score,  # Use core drift as primary score
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

        details["reference_size"] = len(reference_data)
        details["current_size"] = len(current_data)
        details["columns_missing_from_reference"] = _missing_from_reference(
            current_data, reference_data
        )

        # Nothing comparable was found, so nothing was measured. Without this,
        # `max(drift_scores) if drift_scores else 0.0` yields 0.0 -> no drift ->
        # exit 0 -> "healthy", which is issue #71's exact shape on the path this
        # module takes BY DEFAULT (`EVIDENTLY_ENABLED` defaults to false, and
        # `_setup_evidently` also falls back here on ImportError). The Evidently
        # path got its guard first; this one had to have it too.
        if not drift_scores:
            logger.warning(
                f"{self.classifier_type}: reference and current data share no "
                f"comparable column; drift cannot be assessed"
            )
            details["error"] = "No columns available for drift detection"
            return DriftReport(
                classifier_type=self.classifier_type,
                timestamp=datetime.now(),
                drift_detected=False,
                drift_score=0.0,
                threshold=self.threshold,
                details=details,
                indeterminate=True,
            )

        if details["columns_missing_from_reference"]:
            logger.warning(
                f"{self.classifier_type}: reference dataset lacks "
                f"{', '.join(details['columns_missing_from_reference'])} - drift "
                f"for those columns was NOT assessed. Regenerate the reference "
                f"with --create-reference to compare them."
            )

        # Overall drift score
        overall_drift = max(drift_scores)
        drift_detected = overall_drift > self.threshold

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
