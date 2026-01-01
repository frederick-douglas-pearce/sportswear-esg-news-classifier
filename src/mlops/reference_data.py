"""Reference data management for drift detection."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import mlops_settings

logger = logging.getLogger(__name__)

# Expected columns in prediction logs
PREDICTION_LOG_COLUMNS = [
    "timestamp",
    "probability",
    "prediction",
    "text_length",
    "has_brand_context",
]


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
                cp.threshold_used as threshold,
                cp.risk_level,
                cp.action_taken,
                cp.model_version,
                LENGTH(a.title || ' ' || COALESCE(a.full_content, '')) as text_length
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


def create_reference_dataset(
    classifier_type: str,
    logs_dir: str | Path = "logs/predictions",
    days: int | None = None,
    output_path: Path | None = None,
    from_database: bool = False,
) -> Path:
    """Create a reference dataset from historical predictions.

    Args:
        classifier_type: Type of classifier
        logs_dir: Directory containing prediction logs
        days: Number of days to use (default from settings)
        output_path: Output path (default from settings)
        from_database: If True, load from database instead of log files

    Returns:
        Path to saved reference dataset
    """
    days = days or mlops_settings.reference_window_days
    output_path = output_path or mlops_settings.get_reference_data_path(classifier_type)

    # Load predictions
    df = load_prediction_logs(classifier_type, logs_dir, days=days, from_database=from_database)

    if df.empty:
        raise ValueError(f"No prediction data found for {classifier_type}")

    # Create reference directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Created reference dataset: {output_path} ({len(df)} records)")

    return output_path


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
