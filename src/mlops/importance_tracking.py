"""Feature importance history tracking for model stability analysis.

Tracks SHAP and permutation importance across training runs to identify:
- Importance variability (which features are stable vs unstable)
- Correlation patterns (features that co-vary in importance)
- Drift indicators (changes in importance over time)

Usage:
    from src.mlops.importance_tracking import ImportanceHistory

    # Create or load history
    history = ImportanceHistory.load("fp")  # Or ImportanceHistory("fp")

    # Add a new record after training
    history.add_record(
        shap_importance=shap_result.as_dict(),
        perm_importance=perm_result.as_dict(),
        model_params={"n_estimators": 120, "max_depth": 15},
        metrics={"f2": 0.985, "recall": 0.992},
        data_info={"n_train": 1104, "n_test": 369},
    )
    history.save()

    # Analyze variability
    analysis = history.analyze_stability()
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default directory for importance history files
DEFAULT_HISTORY_DIR = Path("data/importance_history")


@dataclass
class ImportanceRecord:
    """A single training run's feature importance results.

    Attributes:
        timestamp: When the training run occurred
        classifier_type: Type of classifier (fp, ep, esg)
        version: Model version if available
        shap_importance: SHAP feature group importance (from FeatureGroupImportance.as_dict())
        perm_importance: Permutation importance (from PermutationImportance.as_dict())
        model_params: Model hyperparameters
        metrics: Performance metrics (f2, recall, precision, etc.)
        data_info: Information about training data (n_train, n_test, etc.)
        feature_method: Feature engineering method used
        notes: Optional notes about this run
    """

    timestamp: str
    classifier_type: str
    shap_importance: dict[str, Any]
    perm_importance: dict[str, Any]
    metrics: dict[str, float]
    version: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    data_info: dict[str, Any] = field(default_factory=dict)
    feature_method: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImportanceRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class StabilityAnalysis:
    """Analysis of feature importance stability across training runs.

    Attributes:
        n_runs: Number of training runs analyzed
        shap_mean: Mean SHAP contribution by feature group
        shap_std: Standard deviation of SHAP contribution
        shap_cv: Coefficient of variation (std/mean) for SHAP
        perm_mean: Mean permutation contribution by feature group
        perm_std: Standard deviation of permutation contribution
        perm_cv: Coefficient of variation for permutation importance
        correlation_matrix: Correlation between feature groups (from permutation)
        most_stable_groups: Groups with lowest CV (most consistent)
        least_stable_groups: Groups with highest CV (most variable)
        performance_correlation: Correlation between importance and F2 score
    """

    n_runs: int
    shap_mean: dict[str, float]
    shap_std: dict[str, float]
    shap_cv: dict[str, float]
    perm_mean: dict[str, float]
    perm_std: dict[str, float]
    perm_cv: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]] | None
    most_stable_groups: list[tuple[str, float]]  # (group, cv)
    least_stable_groups: list[tuple[str, float]]
    performance_correlation: dict[str, float] | None  # group -> correlation with F2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ImportanceHistory:
    """Manages feature importance history across training runs.

    Stores history in JSON files for portability and easy inspection.
    One history file per classifier type.
    """

    def __init__(
        self,
        classifier_type: str,
        history_dir: Path | str | None = None,
    ):
        """Initialize importance history.

        Args:
            classifier_type: Type of classifier (fp, ep, esg)
            history_dir: Directory for history files (default: data/importance_history)
        """
        self.classifier_type = classifier_type
        self.history_dir = Path(history_dir) if history_dir else DEFAULT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[ImportanceRecord] = []

    @property
    def history_file(self) -> Path:
        """Path to the history JSON file."""
        return self.history_dir / f"{self.classifier_type}_importance_history.json"

    @property
    def records(self) -> list[ImportanceRecord]:
        """Get all records."""
        return self._records

    def add_record(
        self,
        shap_importance: dict[str, Any],
        perm_importance: dict[str, Any],
        metrics: dict[str, float],
        model_params: dict[str, Any] | None = None,
        data_info: dict[str, Any] | None = None,
        feature_method: str = "",
        version: str = "",
        notes: str = "",
    ) -> ImportanceRecord:
        """Add a new importance record.

        Args:
            shap_importance: SHAP importance from FeatureGroupImportance.as_dict()
            perm_importance: Permutation importance from PermutationImportance.as_dict()
            metrics: Performance metrics (f2, recall, precision, etc.)
            model_params: Model hyperparameters
            data_info: Information about training data
            feature_method: Feature engineering method used
            version: Model version
            notes: Optional notes

        Returns:
            The created ImportanceRecord
        """
        record = ImportanceRecord(
            timestamp=datetime.now().isoformat(),
            classifier_type=self.classifier_type,
            shap_importance=shap_importance,
            perm_importance=perm_importance,
            metrics=metrics,
            model_params=model_params or {},
            data_info=data_info or {},
            feature_method=feature_method,
            version=version,
            notes=notes,
        )
        self._records.append(record)
        logger.info(
            f"Added importance record for {self.classifier_type} "
            f"(total records: {len(self._records)})"
        )
        return record

    def save(self) -> Path:
        """Save history to JSON file.

        Returns:
            Path to the saved file
        """
        data = {
            "classifier_type": self.classifier_type,
            "n_records": len(self._records),
            "last_updated": datetime.now().isoformat(),
            "records": [r.to_dict() for r in self._records],
        }

        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved importance history to {self.history_file}")
        return self.history_file

    @classmethod
    def load(
        cls,
        classifier_type: str,
        history_dir: Path | str | None = None,
    ) -> "ImportanceHistory":
        """Load existing history or create new.

        Args:
            classifier_type: Type of classifier (fp, ep, esg)
            history_dir: Directory for history files

        Returns:
            ImportanceHistory instance (loaded or new)
        """
        history = cls(classifier_type, history_dir)
        if history.history_file.exists():
            with open(history.history_file) as f:
                data = json.load(f)
            history._records = [
                ImportanceRecord.from_dict(r) for r in data.get("records", [])
            ]
            logger.info(
                f"Loaded {len(history._records)} importance records for {classifier_type}"
            )
        else:
            logger.info(f"No existing history for {classifier_type}, starting fresh")
        return history

    def analyze_stability(self, min_runs: int = 3) -> StabilityAnalysis | None:
        """Analyze feature importance stability across runs.

        Args:
            min_runs: Minimum number of runs required for analysis

        Returns:
            StabilityAnalysis or None if insufficient data
        """
        if len(self._records) < min_runs:
            logger.warning(
                f"Insufficient runs for stability analysis "
                f"({len(self._records)} < {min_runs})"
            )
            return None

        # Collect SHAP contributions across runs
        shap_contributions: dict[str, list[float]] = {}
        perm_contributions: dict[str, list[float]] = {}
        f2_scores: list[float] = []

        for record in self._records:
            # SHAP contributions (percentages)
            shap_pct = record.shap_importance.get("group_contribution_pct", {})
            for group, pct in shap_pct.items():
                if group not in shap_contributions:
                    shap_contributions[group] = []
                shap_contributions[group].append(pct)

            # Permutation contributions (percentages)
            perm_pct = record.perm_importance.get("group_contribution_pct", {})
            for group, pct in perm_pct.items():
                if group not in perm_contributions:
                    perm_contributions[group] = []
                perm_contributions[group].append(pct)

            # F2 score
            if "f2" in record.metrics:
                f2_scores.append(record.metrics["f2"])

        # Compute statistics for SHAP
        shap_mean = {g: float(np.mean(vals)) for g, vals in shap_contributions.items()}
        shap_std = {g: float(np.std(vals)) for g, vals in shap_contributions.items()}
        shap_cv = {
            g: float(shap_std[g] / shap_mean[g]) if shap_mean[g] > 0 else 0.0
            for g in shap_mean
        }

        # Compute statistics for permutation
        perm_mean = {g: float(np.mean(vals)) for g, vals in perm_contributions.items()}
        perm_std = {g: float(np.std(vals)) for g, vals in perm_contributions.items()}
        perm_cv = {
            g: float(perm_std[g] / perm_mean[g]) if perm_mean[g] > 0 else 0.0
            for g in perm_mean
        }

        # Sort by CV for most/least stable
        perm_cv_sorted = sorted(perm_cv.items(), key=lambda x: x[1])
        most_stable = perm_cv_sorted[:3] if len(perm_cv_sorted) >= 3 else perm_cv_sorted
        least_stable = (
            perm_cv_sorted[-3:][::-1] if len(perm_cv_sorted) >= 3 else perm_cv_sorted[::-1]
        )

        # Compute correlation matrix between feature groups (from permutation)
        correlation_matrix = None
        if len(perm_contributions) >= 2:
            groups = list(perm_contributions.keys())
            correlation_matrix = {}
            for g1 in groups:
                correlation_matrix[g1] = {}
                for g2 in groups:
                    if len(perm_contributions[g1]) == len(perm_contributions[g2]):
                        corr = np.corrcoef(
                            perm_contributions[g1], perm_contributions[g2]
                        )[0, 1]
                        correlation_matrix[g1][g2] = (
                            float(corr) if not np.isnan(corr) else 0.0
                        )
                    else:
                        correlation_matrix[g1][g2] = 0.0

        # Compute correlation between importance and F2 score
        performance_correlation = None
        if len(f2_scores) >= min_runs:
            performance_correlation = {}
            for group, vals in perm_contributions.items():
                if len(vals) == len(f2_scores):
                    corr = np.corrcoef(vals, f2_scores)[0, 1]
                    performance_correlation[group] = (
                        float(corr) if not np.isnan(corr) else 0.0
                    )

        return StabilityAnalysis(
            n_runs=len(self._records),
            shap_mean=shap_mean,
            shap_std=shap_std,
            shap_cv=shap_cv,
            perm_mean=perm_mean,
            perm_std=perm_std,
            perm_cv=perm_cv,
            correlation_matrix=correlation_matrix,
            most_stable_groups=most_stable,
            least_stable_groups=least_stable,
            performance_correlation=performance_correlation,
        )

    def get_latest_record(self) -> ImportanceRecord | None:
        """Get the most recent record."""
        return self._records[-1] if self._records else None

    def get_records_by_version(self, version: str) -> list[ImportanceRecord]:
        """Get all records for a specific version."""
        return [r for r in self._records if r.version == version]

    def summary(self) -> str:
        """Get a summary of the history."""
        if not self._records:
            return f"No importance history for {self.classifier_type}"

        lines = [
            f"Importance History: {self.classifier_type}",
            f"  Records: {len(self._records)}",
            f"  Date range: {self._records[0].timestamp[:10]} to {self._records[-1].timestamp[:10]}",
        ]

        # Latest record summary
        latest = self._records[-1]
        lines.append(f"\n  Latest run ({latest.timestamp[:19]}):")
        lines.append(f"    F2: {latest.metrics.get('f2', 'N/A')}")

        # Top 3 feature groups by permutation importance
        perm_top = latest.perm_importance.get("top_groups", [])[:3]
        if perm_top:
            lines.append("    Top groups (permutation):")
            for group, mean, std in perm_top:
                lines.append(f"      - {group}: {mean:.4f} ± {std:.4f}")

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of records."""
        return len(self._records)

    def __repr__(self) -> str:
        """String representation."""
        return f"ImportanceHistory({self.classifier_type}, n_records={len(self._records)})"
