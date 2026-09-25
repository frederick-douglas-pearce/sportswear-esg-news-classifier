"""Reference data management for drift detection."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DRIFT_WINDOW_DAYS, mlops_settings

logger = logging.getLogger(__name__)

# Expected columns in prediction logs
PREDICTION_LOG_COLUMNS = [
    "timestamp",
    "probability",
    "prediction",
    "novelty_score",
]

# Brands to track for drift detection (subset of most common)
TRACKED_BRANDS = [
    "Nike",
    "Adidas",
    "Puma",
    "Under Armour",
    "Lululemon",
    "Patagonia",
    "New Balance",
    "ASICS",
    "Reebok",
    "Hoka",
    "Skechers",
    "The North Face",
    "Anta",
    "Li-Ning",
]


def _add_brand_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot encoded brand columns for drift tracking.

    Args:
        df: DataFrame with brands_mentioned column

    Returns:
        DataFrame with added brand_* columns
    """
    if "brands_mentioned" not in df.columns:
        return df

    # Create binary columns for each tracked brand
    for brand in TRACKED_BRANDS:
        col_name = f"brand_{brand.lower().replace(' ', '_')}"
        df[col_name] = df["brands_mentioned"].apply(
            lambda brands: 1 if brands and brand in brands else 0
        )

    # Also track "other" brands not in tracked list
    df["brand_other"] = df["brands_mentioned"].apply(
        lambda brands: 1 if brands and any(b not in TRACKED_BRANDS for b in brands) else 0
    )

    # Drop the original brands_mentioned column (not needed for drift detection)
    df = df.drop(columns=["brands_mentioned"])

    return df


def load_predictions_from_database(
    classifier_type: str,
    days: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Load prediction data from the classifier_predictions database table.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        days: Number of days to load (from today)
        start_date: Start date for loading predictions
        end_date: End date for loading predictions

    Returns:
        DataFrame with prediction data
    """
    from sqlalchemy import create_engine, text

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set, cannot load from database")
        return pd.DataFrame()

    # Determine date range
    if days is not None:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
    elif start_date is None and end_date is None:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

    try:
        engine = create_engine(database_url)
        query = text("""
            SELECT
                cp.created_at as timestamp,
                cp.probability,
                cp.prediction,
                cp.novelty_score,
                cp.threshold_used as threshold,
                cp.risk_level as confidence_level,
                cp.action_taken,
                cp.model_version,
                a.brands_mentioned
            FROM classifier_predictions cp
            JOIN articles a ON cp.article_id = a.id
            WHERE cp.classifier_type = :classifier_type
              AND cp.created_at >= :start_date
              AND cp.created_at <= :end_date
            ORDER BY cp.created_at
        """)

        with engine.connect() as conn:
            result = conn.execute(
                query,
                {
                    "classifier_type": classifier_type,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            rows = result.fetchall()
            columns = result.keys()

        if not rows:
            logger.warning(f"No predictions found for {classifier_type} in date range")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Create one-hot encoded brand columns for drift tracking
        df = _add_brand_columns(df)

        logger.info(f"Loaded {len(df)} predictions from database for {classifier_type}")
        return df

    except Exception as e:
        logger.error(f"Error loading predictions from database: {e}")
        return pd.DataFrame()


def load_prediction_logs(
    classifier_type: str,
    logs_dir: str | Path = "logs/predictions",
    days: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    from_database: bool = False,
) -> pd.DataFrame:
    """Load prediction logs for a classifier.

    Args:
        classifier_type: Type of classifier (fp, ep, esg)
        logs_dir: Directory containing prediction logs
        days: Number of days to load (from today)
        start_date: Start date for loading logs
        end_date: End date for loading logs
        from_database: If True, load from classifier_predictions table instead of files

    Returns:
        DataFrame with prediction data
    """
    # Load from database if requested
    if from_database:
        return load_predictions_from_database(
            classifier_type=classifier_type,
            days=days,
            start_date=start_date,
            end_date=end_date,
        )

    logs_path = Path(logs_dir)
    if not logs_path.exists():
        logger.warning(f"Logs directory not found: {logs_path}")
        return pd.DataFrame()

    # Determine date range
    if days is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    elif start_date is None and end_date is None:
        # Default to last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

    # Find matching log files
    pattern = f"{classifier_type}_predictions_*.jsonl"
    log_files = sorted(logs_path.glob(pattern))

    if not log_files:
        logger.warning(f"No log files found matching: {pattern}")
        return pd.DataFrame()

    # Filter by date
    selected_files = []
    for log_file in log_files:
        # Extract date from filename: {type}_predictions_{YYYYMMDD}.jsonl
        try:
            date_str = log_file.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if start_date <= file_date <= end_date:
                selected_files.append(log_file)
        except (ValueError, IndexError):
            continue

    if not selected_files:
        logger.warning(f"No log files found in date range {start_date} to {end_date}")
        return pd.DataFrame()

    # Load and concatenate
    dfs = []
    for log_file in selected_files:
        try:
            records = []
            with open(log_file) as f:
                for line in f:
                    record = json.loads(line)
                    records.append(record)
            if records:
                dfs.append(pd.DataFrame(records))
        except Exception as e:
            logger.warning(f"Error loading {log_file}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    logger.info(f"Loaded {len(df)} predictions from {len(selected_files)} files")
    return df


def resolve_reference_window(
    days: int,
    end_date: datetime | None = None,
    exclude_recent_days: int | None = None,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return the half-open `[start, end)` window a reference is built from.

    `end` is `end_date` when one is given. Otherwise it is `now` minus
    `exclude_recent_days`, which defaults to the drift check's default
    comparison window, so a reference built today does not contain that window
    (issue #97). Both bounds are timezone-aware UTC.
    """
    if end_date is not None and exclude_recent_days is not None:
        raise ValueError("pass end_date or exclude_recent_days, not both")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")

    now = now or datetime.now(timezone.utc)
    if end_date is None:
        if exclude_recent_days is None:
            exclude_recent_days = DEFAULT_DRIFT_WINDOW_DAYS
        if exclude_recent_days < 0:
            raise ValueError(
                f"exclude_recent_days must not be negative, got {exclude_recent_days}"
            )
        end = now - timedelta(days=exclude_recent_days)
    else:
        end = (
            end_date.astimezone(timezone.utc)
            if end_date.tzinfo
            else end_date.replace(tzinfo=timezone.utc)
        )
        if end > now:
            raise ValueError(f"reference end date {end.isoformat()} is in the future")

    return end - timedelta(days=days), end


def _within_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Keep only rows whose timestamp falls in `[start, end)`.

    The database loader bounds its query inclusively at both ends, and the
    file-log loader returns whole days by filename (read from the midnight
    before `start`). Filtering here is what makes the window exactly `[start,
    end)` for both sources, so a row at the boundary instant belongs to the
    reference or to the comparison window, never to both.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "prediction data has no timestamp column, so the reference window "
            "cannot be enforced"
        )
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df[(ts >= start) & (ts < end)].reset_index(drop=True)


def create_reference_dataset(
    classifier_type: str,
    logs_dir: str | Path = "logs/predictions",
    days: int | None = None,
    output_path: Path | None = None,
    from_database: bool = False,
    end_date: datetime | None = None,
    exclude_recent_days: int | None = None,
) -> Path:
    """Create a reference dataset from historical predictions.

    The window is `days` long and ends at `end_date`, or, by default, at the
    start of the drift check's comparison window (`DEFAULT_DRIFT_WINDOW_DAYS`
    ago), so the reference does not contain the default comparison window
    (issue #97). The window that was asked for is stored in the parquet itself
    as `attrs["reference_window"]`, so it cannot come apart from the data.

    Args:
        classifier_type: Type of classifier
        logs_dir: Directory containing prediction logs
        days: Length of the window in days (default from settings)
        output_path: Output path (default from settings)
        from_database: If True, load from database instead of log files
        end_date: Exclusive end of the window. Mutually exclusive with
            exclude_recent_days.
        exclude_recent_days: End the window this many days before now
            (default DEFAULT_DRIFT_WINDOW_DAYS)

    Returns:
        Path to saved reference dataset
    """
    days = days or mlops_settings.reference_window_days
    output_path = output_path or mlops_settings.get_reference_data_path(classifier_type)
    start, end = resolve_reference_window(days, end_date, exclude_recent_days)

    # `days=None` is load-bearing: both loaders discard start_date/end_date
    # whenever `days` is set, which would rebuild the trailing window ending
    # now -- the exact overlap this function exists to prevent.
    if from_database:
        df = load_prediction_logs(
            classifier_type, logs_dir, days=None,
            start_date=start, end_date=end, from_database=True,
        )
    else:
        # The file-log loader selects whole files by their naive (UTC) filename
        # date at midnight, keeping a file only when `start_date <= file_date`.
        # Floored to midnight so the window's first, partial day is read;
        # `_within_window` below then trims it to the exact start.
        df = load_prediction_logs(
            classifier_type, logs_dir, days=None,
            start_date=start.replace(
                tzinfo=None, hour=0, minute=0, second=0, microsecond=0
            ),
            end_date=end.replace(tzinfo=None),
        )

    if not df.empty:
        df = _within_window(df, start, end)

    if df.empty:
        raise ValueError(
            f"No prediction data found for {classifier_type} in "
            f"[{start.isoformat()}, {end.isoformat()})"
        )

    # JSON primitives only: pandas serialises attrs into the parquet footer
    # with json.dumps (pandas>=2.1; older versions drop them on read).
    df.attrs["reference_window"] = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "end_exclusive": True,
        "source": "database" if from_database else f"logs:{logs_dir}",
        "rows": len(df),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Create reference directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)
    logger.info(
        f"Created reference dataset: {output_path} ({len(df)} records, "
        f"window [{start.isoformat()}, {end.isoformat()}))"
    )

    return output_path


def reference_provenance(
    reference_data: pd.DataFrame, current_data: pd.DataFrame
) -> dict[str, Any]:
    """Say which baseline a drift comparison used, and whether it overlaps.

    `reference_window` is the window requested when the reference was built, or
    None for a reference written before that was recorded. It is never inferred
    from the data. `reference_observed` is what the timestamps actually span.

    `reference_overlaps_current` compares OBSERVED timestamps -- the reference's
    latest against the current window's earliest -- so it answers for a legacy
    reference too. It is None, not False, when either frame has no timestamps:
    an unmeasured overlap must not read as no overlap.
    """
    window = reference_data.attrs.get("reference_window")

    def _span(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if "timestamp" not in df.columns or df.empty:
            return None
        ts = pd.to_datetime(df["timestamp"], utc=True).dropna()
        if ts.empty:
            return None
        return ts.min(), ts.max()

    ref_span = _span(reference_data)
    cur_span = _span(current_data)
    overlaps = None
    if ref_span is not None and cur_span is not None:
        overlaps = bool(ref_span[1] >= cur_span[0])

    return {
        "reference_window": dict(window) if window else None,
        "reference_observed": (
            {"start": ref_span[0].isoformat(), "end": ref_span[1].isoformat()}
            if ref_span is not None
            else None
        ),
        "reference_overlaps_current": overlaps,
    }


COUNT_CONNECT_TIMEOUT_SECONDS = 10
COUNT_STATEMENT_TIMEOUT_MS = 30_000


def count_predictions(classifier_type: str, days: int) -> int:
    """Count one classifier's predictions in the last `days` days.

    **Raises rather than returning 0 on any failure** -- an unset DATABASE_URL,
    a connection error, a bad query. That deliberately departs from
    `load_predictions_from_database`, which logs and returns an empty frame:
    the caller (the EP drift skip, issue #96) reads 0 as "EP is not running",
    so a swallowed failure would report a clean skip over a count nobody took.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not set, cannot count predictions")

    since = datetime.now(timezone.utc) - timedelta(days=days)
    # Bounded, because this runs in-process in the agent rather than in a
    # subprocess with a timeout: a database that accepts the connection and
    # never answers would otherwise hang the drift workflow before any failure
    # path runs. A timeout raises, and the caller reads a raise as UNKNOWN.
    # NullPool closes the connection on exit instead of pooling it.
    engine = create_engine(
        database_url,
        poolclass=NullPool,
        connect_args={
            "connect_timeout": COUNT_CONNECT_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={COUNT_STATEMENT_TIMEOUT_MS}",
        },
    )
    with engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM classifier_predictions "
                "WHERE classifier_type = :classifier_type AND created_at >= :since"
            ),
            {"classifier_type": classifier_type, "since": since},
        ).scalar_one()
    return int(count)


def load_reference_dataset(
    classifier_type: str,
    reference_path: Path | None = None,
) -> pd.DataFrame:
    """Load a reference dataset.

    Args:
        classifier_type: Type of classifier
        reference_path: Path to reference dataset (default from settings)

    Returns:
        Reference DataFrame
    """
    reference_path = reference_path or mlops_settings.get_reference_data_path(classifier_type)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

    df = pd.read_parquet(reference_path)
    logger.info(f"Loaded reference dataset: {reference_path} ({len(df)} records)")
    return df


def get_reference_stats(classifier_type: str) -> dict[str, Any] | None:
    """Get statistics about the reference dataset.

    Args:
        classifier_type: Type of classifier

    Returns:
        Dict with reference stats or None if not found
    """
    try:
        df = load_reference_dataset(classifier_type)
    except FileNotFoundError:
        return None

    stats = {
        "n_records": len(df),
        "requested_window": df.attrs.get("reference_window"),
        "date_range": {
            "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns else None,
            "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns else None,
        },
    }

    if "probability" in df.columns:
        stats["probability"] = {
            "mean": float(df["probability"].mean()),
            "std": float(df["probability"].std()),
            "min": float(df["probability"].min()),
            "max": float(df["probability"].max()),
        }

    if "prediction" in df.columns:
        stats["prediction_rate"] = float(df["prediction"].mean())

    return stats
