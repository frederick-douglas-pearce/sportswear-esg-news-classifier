"""Evidently-based drift detection and monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to Python built-ins, recursively.

    `numpy.generic.item()` handles every numpy scalar in one branch --
    `np.bool_` -> `bool`, `np.float64` -> `float`, `np.int64` -> `int` --
    which matters because `np.bool_` is the one that does NOT subclass its
    Python counterpart and so is the one `json.dumps` refuses.
    """
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _categorical_p_value(
    reference_col: pd.Series,
    current_col: pd.Series,
    min_size: int = 1,
) -> tuple[float | None, str | None]:
    """Chi-square p-value for a categorical column, with the reason where none.

    Returns `(p_value, None)` when the test ran and `(None, reason)` when it did
    not; exactly one of the two is ever set. The caller records the reason and
    treats a None p-value as "not assessed" -- never as "no drift".

    Every rejection names its own cause. They used to reach the caller as one
    string, which left `columns_skipped` unable to answer the only question it
    exists to answer.

    What this deliberately does NOT reject is a column too rare for the
    chi-square approximation to be sound. A rare brand is power-ASYMMETRIC, not
    powerless: `brand_li-ning` (3 positives in the shipped 934-row reference)
    cannot evidence a decrease, but three positives in a 73-row window take it
    to p=0.0011 -- and a rare brand suddenly appearing is the drift this project
    most wants to hear about. A minimum-expected-cell floor removes the dead
    reading and that detection together, so it is not applied (D017).

    Counts are aligned by LABEL, via `reindex`. `value_counts()` sorts by count
    descending, so a positional read takes the most-frequent category rather
    than the one asked for: a reference of ints against a current frame of
    bools built a symmetric table out of a total flip and returned p=1.0.
    """
    from scipy import stats

    # The #94 row floor, per column and after the NaN drop, on this path too. It
    # was reachable only from the three core columns while five places said it
    # applied per column.
    reference_clean = reference_col.dropna()
    current_clean = current_col.dropna()
    if len(reference_clean) < min_size or len(current_clean) < min_size:
        return None, "below the sample floor"

    # Normalize bool to int before counting. `True` and `1` are equal and hash
    # alike, so a bool-indexed and an int-indexed `value_counts` dedupe into one
    # category set while neither can be looked up in the other's index -- which
    # is how a total flip came back symmetric.
    if pd.api.types.is_bool_dtype(reference_clean):
        reference_clean = reference_clean.astype("int64")
    if pd.api.types.is_bool_dtype(current_clean):
        current_clean = current_clean.astype("int64")

    reference_counts = reference_clean.value_counts()
    current_counts = current_clean.value_counts()

    # A column whose labels cannot be ordered together (mixed str/int, say) is
    # not comparable; `sorted` would raise and abort the whole drift check.
    try:
        categories = sorted(set(reference_counts.index) | set(current_counts.index))
    except TypeError:
        return None, "labels cannot be ordered together"

    table = np.array(
        [
            reference_counts.reindex(categories, fill_value=0).to_numpy(),
            current_counts.reindex(categories, fill_value=0).to_numpy(),
        ],
        dtype=float,
    )

    # One category across both frames means the value never varies. A zero row
    # or column marginal makes the expected frequencies degenerate, which
    # `chi2_contingency` raises on.
    if table.shape[1] < 2:
        return None, "one category in both frames"
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return None, "a category or a frame is empty"

    _, p_value, _, _ = stats.chi2_contingency(table)
    if not np.isfinite(p_value):
        return None, "p-value is not finite"
    return float(p_value), None


def _comparable_series(
    reference_col: pd.Series, current_col: pd.Series, min_size: int = 1
) -> tuple[pd.Series, pd.Series] | None:
    """Both columns with NaN dropped, or None where nothing is comparable.

    None means the KS test would be undefined, vacuous, or too small to mean
    anything. The caller records that as not-assessed and never as no-drift.

    Three rejections. Either side below `min_size` (issue #94): `ks_2samp` on
    one surviving row returns a statistic of 1.0 -- the loudest value the
    instrument can produce -- alongside a p-value saying it means nothing, and
    this path uses the statistic. Either side empty after the NaN drop. And the
    same constant in both frames, where there is nothing to compare.

    What this deliberately does NOT reject: the same column constant at
    DIFFERENT values in the two frames. That is a total distributional shift,
    not a degenerate comparison.
    """
    reference_clean = reference_col.dropna()
    current_clean = current_col.dropna()

    if len(reference_clean) < min_size or len(current_clean) < min_size:
        return None
    if (
        reference_clean.nunique() == 1
        and current_clean.nunique() == 1
        and reference_clean.iloc[0] == current_clean.iloc[0]
    ):
        return None
    return reference_clean, current_clean


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
        path. Coercing here rather than at each call site means a construction
        site added later cannot reintroduce it.

        **`details` is coerced too, and that is not belt-and-braces.** Evidently
        returns `numpy.float64` for a metric's value, so
        `col_drift = p_value < p_value_threshold` in `_evidently_drift_check` is
        a `numpy.bool_` -- and `scripts/monitor_drift.py` writes `report.details`
        straight into the `--output` JSON that
        `.github/workflows/monitoring.yml` reads with `jq`. Coercing only the
        four scalar fields left that path raising `TypeError` mid-write, exiting
        1 (which this project's contract reads as "drift detected") and leaving
        a truncated file on disk.
        """
        self.drift_detected = bool(self.drift_detected)
        self.indeterminate = bool(self.indeterminate)
        self.drift_score = float(self.drift_score)
        self.threshold = float(self.threshold)
        self.details = _to_builtin(self.details)


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
                # Refuse to compare the window against itself.
                #
                # This branch used to split `current_data` in half and call the
                # first half the reference. Two halves of one window have the
                # same distribution by construction, so every statistic below
                # returns "no drift" -- a HEALTHY verdict with
                # `indeterminate=False`, manufactured out of the ABSENCE of a
                # baseline. That is issue #71's class exactly.
                #
                # Reachable for any classifier that HAS predictions and LACKS a
                # reference. Today that is none of them -- `fp` is the only one
                # with rows and its reference is tracked in git -- so this is
                # latent, not live; an earlier revision of this comment claimed
                # otherwise. It goes live the moment EP resumes, or a third
                # classifier lands, ahead of `--create-reference`. A classifier
                # with no predictions at all stops at the empty-frame guard
                # below instead, which is why `esg` was already indeterminate.
                #
                # `--create-reference` is the deliberate way to establish a
                # baseline, which leaves the bootstrap nothing to justify it.
                reference_path = mlops_settings.get_reference_data_path(
                    self.classifier_type
                )
                logger.warning(
                    f"{self.classifier_type}: no reference dataset at "
                    f"{reference_path}; drift cannot be assessed "
                    f"(run --create-reference to establish one)"
                )
                return DriftReport(
                    classifier_type=self.classifier_type,
                    timestamp=datetime.now(),
                    drift_detected=False,
                    drift_score=0.0,
                    threshold=self.threshold,
                    details={
                        "error": "No reference dataset for this classifier",
                        "reference_path": str(reference_path),
                        "current_size": len(current_data),
                    },
                    indeterminate=True,
                )

        min_sample_size = mlops_settings.drift_min_sample_size
        if (
            len(current_data) < min_sample_size
            or len(reference_data) < min_sample_size
        ):
            # This branch returns before either checker runs, so the
            # `drift_detected=False` below is fabricated rather than measured.
            # Before #71 that was indistinguishable from a real clean result:
            # the script reported exit 0 and the workflow read it as healthy,
            # which is how the EP classifier -- which has never made a single
            # prediction -- passed on every run.
            #
            # The floor is a minimum, not emptiness (#94). Enough rows to
            # compute a statistic is not enough for it to mean anything, and
            # the two frames are tested independently: a large reference
            # against a handful of current rows is as untrustworthy as the
            # reverse.
            logger.warning(
                f"{self.classifier_type}: insufficient data for drift analysis "
                f"(reference={len(reference_data)} rows, current={len(current_data)} rows, "
                f"minimum={min_sample_size})"
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
        # `columns_assessed` is NOT `columns_checked`. The latter is what was
        # offered to Evidently; the former is what came back with a readable
        # metric, which is the narrower and more honest set -- a column whose
        # metric is unreadable was checked and not assessed.
        columns_assessed: list[str] = []
        columns_skipped: dict[str, str] = {}
        details = {
            "columns_checked": columns_to_check,
            "columns_assessed": columns_assessed,
            "columns_skipped": columns_skipped,
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
                # An unreadable value is NOT evidence of no drift. This
                # used to coerce it to `p_value = 1.0`, which made
                # `col_drift` False AND counted the metric toward
                # `total_core` -- so the `total_core == 0` guard below could
                # not fire, and the run reported a measured-looking 0.0 built
                # from a metric nobody could read. Skip it and say so.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    logger.warning(
                        f"{self.classifier_type}: metric for {col_name!r} had no "
                        f"readable value (got {type(value).__name__}); it is not "
                        f"counted as 'no drift'"
                    )
                    details.setdefault("metrics_unreadable", []).append(col_name)
                    columns_skipped[col_name] = (
                        f"metric had no readable value "
                        f"(got {type(value).__name__})"
                    )
                    continue

                p_value = float(value)
                col_drift = p_value < p_value_threshold
                details[f"{col_name}_drift"] = col_drift
                details[f"{col_name}_p_value"] = p_value
                columns_assessed.append(col_name)

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

        # No CORE metric was assessed, so the drift score below would be a
        # fabricated 0.0 -- `max()` of two scores neither of which measured
        # probability or prediction.
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
        # at all (this code targets "v0.7+", so that has happened once). That
        # is true only because the loop above `continue`s past an unreadable
        # value rather than counting it as p=1.0; an earlier revision of this
        # comment claimed the coverage while the coercion still defeated it.
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

        # Report the larger of the two, NOT `core_drift_score` alone.
        #
        # `drift_detected` reads both scores; the report used to carry only the
        # core one. Brand-only drift therefore emitted `drift_detected=True`
        # with `drift_score=0.0`, and the workflow logged "score 0.0 exceeds
        # 0.15" -- an alert whose own number contradicts it. Confirmed live on
        # 940 rows. Both components stay in `details` below.
        drift_score = max(core_drift_score, brand_drift_score)

        details["core_drift_score"] = core_drift_score
        details["brand_drift_score"] = brand_drift_score
        details["core_drifted_count"] = core_drifted
        details["brand_drifted_count"] = brand_drifted
        # The denominator, so a reader can tell 1-of-1 from 1-of-40. Named to
        # match the legacy path's key rather than `total_brand` (#102).
        details["brand_assessed_count"] = total_brand

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
            drift_score=drift_score,
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

        **This is the DEFAULT path, and it can also be reached by accident.**
        `EVIDENTLY_ENABLED` defaults to false, and `_setup_evidently` falls back
        here on ImportError -- so a deployment lands here either by not setting
        the variable or by a dependency problem that changes the instrument
        without announcing it. Which path a given deployment actually takes
        depends on its environment and is not recorded anywhere in the report
        (#136); do not infer it from this docstring.

        It used to compare `probability` and `prediction` and nothing else,
        while `columns_missing_from_reference` stayed empty because
        `novelty_score` and the `brand_*` columns ARE in the reference. They
        were simply never looked at, so a total distributional shift in
        `novelty_score` reported as healthy and a partial check was
        indistinguishable from a whole one (issue #102).

        The core score is an effect size: `probability`, `prediction` and
        `novelty_score` contribute magnitudes -- a KS statistic, a rate
        difference -- and the core score is the largest of them. Brand is a
        separate component, a fraction of significant chi-square tests, OR'd
        into the verdict. `details["drift_score_source"]` records which of the
        two produced the reported `drift_score`.

        Args:
            current_data: Current prediction data
            reference_data: Reference data

        Returns:
            DriftReport with results
        """
        from scipy import stats

        details: dict[str, Any] = {}
        drift_scores = []

        # What was actually measured, as opposed to what was offered. Both
        # fields carry the same TYPES on the Evidently path, but not the same
        # coverage: that path has one writer (an unreadable metric) and still
        # drops a brand column absent from the reference silently, so the two
        # are not like-for-like and #105's cross-check will have to say which
        # path it is reading. #104 lifts them into the summary (they do not
        # reach it today). Introduced here as a plain record -- it must NOT
        # drive indeterminacy, which is #105's call, not this one's (D017).
        columns_assessed: list[str] = []
        columns_skipped: dict[str, str] = {}
        details["columns_assessed"] = columns_assessed
        details["columns_skipped"] = columns_skipped

        min_size = mlops_settings.drift_min_sample_size

        def _skip(column: str, reason: str) -> None:
            columns_skipped[column] = reason

        # Every core column goes through the SAME guard. Applying it to one of
        # the three is what let a NaN reach `drift_scores`, where it poisons
        # `max` -- NaN comparisons are False, so `max` keeps whichever operand
        # it started with -- and `nan > threshold` is False, so the run reported
        # healthy over a total shift in another column while `columns_assessed`
        # named the NaN column as measured.

        # Check probability distribution
        if "probability" in current_data.columns and "probability" in reference_data.columns:
            comparable = _comparable_series(
                reference_data["probability"], current_data["probability"], min_size
            )
            if comparable is None:
                _skip("probability", "no comparable values")
            else:
                ks_stat, p_value = stats.ks_2samp(*comparable)
                details["probability_ks_statistic"] = float(ks_stat)
                details["probability_p_value"] = float(p_value)
                drift_scores.append(float(ks_stat))
                columns_assessed.append("probability")

        # Check prediction rate
        if "prediction" in current_data.columns and "prediction" in reference_data.columns:
            comparable = _comparable_series(
                reference_data["prediction"], current_data["prediction"], min_size
            )
            if comparable is None:
                _skip("prediction", "no comparable values")
            else:
                reference_clean, current_clean = comparable
                ref_rate = reference_clean.mean()
                curr_rate = current_clean.mean()
                rate_diff = abs(curr_rate - ref_rate)
                details["reference_prediction_rate"] = float(ref_rate)
                details["current_prediction_rate"] = float(curr_rate)
                details["prediction_rate_diff"] = float(rate_diff)
                drift_scores.append(float(rate_diff))
                columns_assessed.append("prediction")

        # Check novelty distribution (issue #102).
        #
        # A KS statistic is the same KIND of number as `probability`'s, so this
        # folds into `drift_scores` with no arithmetic change and no
        # comparability break -- which is what makes it the load-bearing half of
        # #102's fix rather than a redesign.
        if (
            "novelty_score" in current_data.columns
            and "novelty_score" in reference_data.columns
        ):
            comparable = _comparable_series(
                reference_data["novelty_score"],
                current_data["novelty_score"],
                min_size,
            )
            if comparable is None:
                _skip("novelty_score", "no comparable values")
            else:
                ks_stat, p_value = stats.ks_2samp(*comparable)
                # Spelled `novelty_score_*` to match the Evidently path's
                # `f"{col_name}_p_value"`. #104 lifts these keys into the
                # machine-readable summary, and two spellings for one column
                # would become its problem.
                details["novelty_score_ks_statistic"] = float(ks_stat)
                details["novelty_score_p_value"] = float(p_value)
                drift_scores.append(float(ks_stat))
                columns_assessed.append("novelty_score")

        # Check brand columns (issue #102).
        #
        # These stay OUT of `drift_scores`: a fraction of significant tests is a
        # different unit from an effect size, and blending them under one `max`
        # would change what the reported number means. Brand gets its own
        # component below, mirroring the Evidently path's core-OR-brand
        # structure without adopting its arithmetic (D017).
        brand_drifted = 0
        brand_assessed = 0
        brand_metrics_drifted: list[str] = []
        _, brand_p_threshold = self._get_drift_config("brand_")

        for col in sorted(c for c in current_data.columns if c.startswith("brand_")):
            if col not in reference_data.columns:
                # Recorded, not silently dropped. `_missing_from_reference`
                # covers CORE_DRIFT_COLUMNS only by design, so without this a
                # brand column the reference cannot answer for left no trace in
                # any field. A reference built from files rather than the
                # database carries no brand column at all -- `_add_brand_columns`
                # runs only in `load_predictions_from_database` -- so every one
                # of them took this branch.
                _skip(col, "not in reference")
                continue
            p_value, reason = _categorical_p_value(
                reference_data[col], current_data[col], min_size
            )
            if p_value is None:
                # Skipped and recorded, never counted toward the denominator.
                # Counting it would report an unmeasured column as evidence of
                # health and dilute the score.
                _skip(col, reason or "not comparable")
                continue
            brand_assessed += 1
            columns_assessed.append(col)
            details[f"{col}_p_value"] = p_value
            if p_value < brand_p_threshold:
                brand_drifted += 1
                brand_metrics_drifted.append(col)

        if columns_skipped:
            logger.warning(
                f"{self.classifier_type}: {len(columns_skipped)} column(s) were "
                f"present but could not be assessed "
                f"({', '.join(sorted(columns_skipped))}); they are NOT counted "
                f"as 'no drift'"
            )

        brand_drift_score = brand_drifted / brand_assessed if brand_assessed else 0.0

        # Written here rather than beside the verdict below so these keys are
        # present on the indeterminate return too.
        details["brand_metrics_drifted"] = brand_metrics_drifted
        details["brand_assessed_count"] = brand_assessed
        details["brand_drifted_count"] = brand_drifted
        details["brand_drift_score"] = brand_drift_score
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

        # Brand stays out of the core effect-size aggregation: `drift_scores`
        # holds magnitudes (KS statistics, a rate difference) and
        # `brand_drift_score` is a fraction of significant tests. The reported
        # `drift_score` is then `max(core, brand)`, mirroring the Evidently
        # path -- which is what stops brand-only drift alerting as
        # "score 0.0 exceeds 0.15". So the reported number can be either
        # quantity, and `drift_score_source` below says which.
        core_drift_score = max(drift_scores)

        drift_detected = (
            core_drift_score > self.threshold or brand_drift_score > self.threshold
        )
        drift_score = max(core_drift_score, brand_drift_score)

        details["core_drift_score"] = core_drift_score
        details["drift_score_source"] = (
            "core" if core_drift_score >= brand_drift_score else "brand"
        )

        return DriftReport(
            classifier_type=self.classifier_type,
            timestamp=datetime.now(),
            drift_detected=drift_detected,
            drift_score=drift_score,
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
