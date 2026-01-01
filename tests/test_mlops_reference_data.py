"""Tests for the MLOps reference_data module."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.mlops.reference_data import (
    PREDICTION_LOG_COLUMNS,
    create_reference_dataset,
    get_reference_stats,
    load_prediction_logs,
    load_predictions_from_database,
    load_reference_dataset,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_prediction_data():
    """Create sample prediction data."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="h"),
        "probability": [0.3, 0.5, 0.7, 0.8, 0.4, 0.6, 0.9, 0.2, 0.5, 0.75],
        "prediction": [0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        "text_length": [100, 200, 150, 300, 250, 180, 220, 90, 310, 275],
        "has_brand_context": [True, False, True, True, False, True, True, False, True, True],
    })


@pytest.fixture
def temp_logs_dir(sample_prediction_data):
    """Create a temporary logs directory with prediction files."""
    with TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir)

        # Create log file for today
        today = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"fp_predictions_{today}.jsonl"

        with open(log_file, "w") as f:
            for _, row in sample_prediction_data.iterrows():
                record = row.to_dict()
                record["timestamp"] = record["timestamp"].isoformat()
                f.write(json.dumps(record) + "\n")

        yield logs_dir


@pytest.fixture
def temp_reference_dir(sample_prediction_data):
    """Create a temporary directory with reference data."""
    with TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "fp_reference.parquet"
        sample_prediction_data.to_parquet(ref_path, index=False)
        yield Path(tmpdir), ref_path


# ============================================================================
# load_predictions_from_database Tests
# ============================================================================


class TestLoadPredictionsFromDatabase:
    """Tests for load_predictions_from_database function."""

    def test_returns_empty_when_no_database_url(self):
        """Test returns empty DataFrame when DATABASE_URL not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure DATABASE_URL is not in environment
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]

            result = load_predictions_from_database("fp", days=7)

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_loads_with_days_parameter(self):
        """Test loading predictions with days parameter."""
        mock_rows = [
            (datetime.now(timezone.utc), 0.75, 1, 0.5, "low", "none", "v1.0.0", 500),
            (datetime.now(timezone.utc), 0.25, 0, 0.5, "low", "none", "v1.0.0", 300),
        ]
        mock_columns = ["timestamp", "probability", "prediction", "threshold",
                        "risk_level", "action_taken", "model_version", "text_length"]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_result.keys.return_value = mock_columns

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            with patch("sqlalchemy.create_engine", return_value=mock_engine):
                result = load_predictions_from_database("fp", days=7)

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 2
                assert "probability" in result.columns
                assert "prediction" in result.columns

    def test_loads_with_date_range(self):
        """Test loading predictions with explicit date range."""
        mock_rows = [
            (datetime.now(timezone.utc), 0.8, 1, 0.5, "medium", "review", "v2.0.0", 450),
        ]
        mock_columns = ["timestamp", "probability", "prediction", "threshold",
                        "risk_level", "action_taken", "model_version", "text_length"]

        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        mock_result.keys.return_value = mock_columns

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        start = datetime.now(timezone.utc) - timedelta(days=30)
        end = datetime.now(timezone.utc)

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            with patch("sqlalchemy.create_engine", return_value=mock_engine):
                result = load_predictions_from_database(
                    "fp",
                    start_date=start,
                    end_date=end
                )

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 1

    def test_returns_empty_on_no_results(self):
        """Test returns empty DataFrame when no predictions found."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            with patch("sqlalchemy.create_engine", return_value=mock_engine):
                result = load_predictions_from_database("fp", days=7)

                assert isinstance(result, pd.DataFrame)
                assert result.empty

    def test_returns_empty_on_database_error(self):
        """Test returns empty DataFrame on database error."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            with patch("sqlalchemy.create_engine", side_effect=Exception("Connection failed")):
                result = load_predictions_from_database("fp", days=7)

                assert isinstance(result, pd.DataFrame)
                assert result.empty

    def test_defaults_to_30_days_when_no_range_specified(self):
        """Test defaults to 30 days when no date range specified."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"}):
            with patch("sqlalchemy.create_engine", return_value=mock_engine):
                load_predictions_from_database("fp")

                # Verify execute was called (date range defaults are used)
                mock_conn.execute.assert_called_once()


# ============================================================================
# load_prediction_logs Tests
# ============================================================================


class TestLoadPredictionLogs:
    """Tests for load_prediction_logs function."""

    def test_delegates_to_database_when_from_database_true(self):
        """Test that from_database=True delegates to database loader."""
        with patch("src.mlops.reference_data.load_predictions_from_database") as mock_db:
            mock_db.return_value = pd.DataFrame({"probability": [0.5]})

            result = load_prediction_logs("fp", days=7, from_database=True)

            mock_db.assert_called_once_with(
                classifier_type="fp",
                days=7,
                start_date=None,
                end_date=None,
            )

    def test_loads_from_files_when_from_database_false(self, temp_logs_dir):
        """Test loading from log files when from_database=False."""
        result = load_prediction_logs(
            "fp",
            logs_dir=temp_logs_dir,
            days=1,
            from_database=False
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "probability" in result.columns

    def test_returns_empty_when_logs_dir_not_found(self):
        """Test returns empty DataFrame when logs directory doesn't exist."""
        result = load_prediction_logs(
            "fp",
            logs_dir="/nonexistent/path",
            days=7,
            from_database=False
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_when_no_matching_files(self, temp_logs_dir):
        """Test returns empty DataFrame when no files match pattern."""
        result = load_prediction_logs(
            "ep",  # Different classifier type
            logs_dir=temp_logs_dir,
            days=7,
            from_database=False
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_filters_by_date_range(self, temp_logs_dir, sample_prediction_data):
        """Test filtering log files by date range."""
        # Create a file for yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        old_log = temp_logs_dir / f"fp_predictions_{yesterday}.jsonl"

        with open(old_log, "w") as f:
            for _, row in sample_prediction_data.iterrows():
                record = row.to_dict()
                record["timestamp"] = record["timestamp"].isoformat()
                f.write(json.dumps(record) + "\n")

        result = load_prediction_logs(
            "fp",
            logs_dir=temp_logs_dir,
            days=2,
            from_database=False
        )

        # Should include both today and yesterday
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ============================================================================
# create_reference_dataset Tests
# ============================================================================


class TestCreateReferenceDataset:
    """Tests for create_reference_dataset function."""

    def test_creates_reference_from_logs(self, temp_logs_dir):
        """Test creating reference dataset from log files."""
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "reference.parquet"

            with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
                mock_settings.reference_window_days = 30
                mock_settings.get_reference_data_path.return_value = output_path

                result_path = create_reference_dataset(
                    "fp",
                    logs_dir=temp_logs_dir,
                    days=1,
                    output_path=output_path,
                    from_database=False
                )

                assert result_path == output_path
                assert output_path.exists()

                # Verify data was saved correctly
                saved_data = pd.read_parquet(output_path)
                assert not saved_data.empty

    def test_creates_reference_from_database(self):
        """Test creating reference dataset from database."""
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "reference.parquet"

            mock_data = pd.DataFrame({
                "timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
                "probability": [0.3, 0.5, 0.7, 0.8, 0.4],
                "prediction": [0, 1, 1, 1, 0],
            })

            with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
                mock_settings.reference_window_days = 30
                mock_settings.get_reference_data_path.return_value = output_path

                with patch("src.mlops.reference_data.load_prediction_logs", return_value=mock_data):
                    result_path = create_reference_dataset(
                        "fp",
                        days=30,
                        output_path=output_path,
                        from_database=True
                    )

                    assert result_path == output_path
                    assert output_path.exists()

    def test_raises_on_empty_data(self, temp_logs_dir):
        """Test raises ValueError when no prediction data found."""
        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "reference.parquet"

            with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
                mock_settings.reference_window_days = 30
                mock_settings.get_reference_data_path.return_value = output_path

                with pytest.raises(ValueError, match="No prediction data found"):
                    create_reference_dataset(
                        "nonexistent",  # No logs for this classifier
                        logs_dir=temp_logs_dir,
                        days=1,
                        output_path=output_path,
                        from_database=False
                    )


# ============================================================================
# load_reference_dataset Tests
# ============================================================================


class TestLoadReferenceDataset:
    """Tests for load_reference_dataset function."""

    def test_loads_reference_data(self, temp_reference_dir):
        """Test loading reference dataset from parquet file."""
        tmpdir, ref_path = temp_reference_dir

        with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
            mock_settings.get_reference_data_path.return_value = ref_path

            result = load_reference_dataset("fp")

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert "probability" in result.columns

    def test_raises_when_file_not_found(self):
        """Test raises FileNotFoundError when reference file doesn't exist."""
        with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
            mock_settings.get_reference_data_path.return_value = Path("/nonexistent/ref.parquet")

            with pytest.raises(FileNotFoundError):
                load_reference_dataset("fp")

    def test_uses_custom_reference_path(self, temp_reference_dir):
        """Test loading with custom reference path."""
        tmpdir, ref_path = temp_reference_dir

        result = load_reference_dataset("fp", reference_path=ref_path)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ============================================================================
# get_reference_stats Tests
# ============================================================================


class TestGetReferenceStats:
    """Tests for get_reference_stats function."""

    def test_returns_stats_for_existing_reference(self, temp_reference_dir):
        """Test returns stats for existing reference dataset."""
        tmpdir, ref_path = temp_reference_dir

        with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
            mock_settings.get_reference_data_path.return_value = ref_path

            stats = get_reference_stats("fp")

            assert stats is not None
            assert "n_records" in stats
            assert stats["n_records"] == 10  # From sample_prediction_data
            assert "date_range" in stats
            assert "probability" in stats
            assert "prediction_rate" in stats

    def test_returns_none_when_no_reference(self):
        """Test returns None when reference file doesn't exist."""
        with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
            mock_settings.get_reference_data_path.return_value = Path("/nonexistent/ref.parquet")

            stats = get_reference_stats("fp")

            assert stats is None

    def test_stats_include_probability_metrics(self, temp_reference_dir):
        """Test stats include probability mean, std, min, max."""
        tmpdir, ref_path = temp_reference_dir

        with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
            mock_settings.get_reference_data_path.return_value = ref_path

            stats = get_reference_stats("fp")

            assert "probability" in stats
            assert "mean" in stats["probability"]
            assert "std" in stats["probability"]
            assert "min" in stats["probability"]
            assert "max" in stats["probability"]


# ============================================================================
# PREDICTION_LOG_COLUMNS Tests
# ============================================================================


class TestPredictionLogColumns:
    """Tests for PREDICTION_LOG_COLUMNS constant."""

    def test_contains_required_columns(self):
        """Test that PREDICTION_LOG_COLUMNS contains expected columns."""
        assert "timestamp" in PREDICTION_LOG_COLUMNS
        assert "probability" in PREDICTION_LOG_COLUMNS
        assert "prediction" in PREDICTION_LOG_COLUMNS
        assert "text_length" in PREDICTION_LOG_COLUMNS
        assert "has_brand_context" in PREDICTION_LOG_COLUMNS

    def test_is_list(self):
        """Test that PREDICTION_LOG_COLUMNS is a list."""
        assert isinstance(PREDICTION_LOG_COLUMNS, list)
