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
                        "confidence_level", "action_taken", "model_version", "text_length"]

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
                        "confidence_level", "action_taken", "model_version", "text_length"]

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

    def test_creates_reference_from_logs(self, tmp_path):
        """Test creating reference dataset from log files.

        The log file and its records are dated inside the default window
        (the 30 days ending DEFAULT_DRIFT_WINDOW_DAYS ago), because the window
        is enforced on each record's timestamp (issue #97).
        """
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        day = datetime.now(timezone.utc) - timedelta(days=10)
        with open(logs_dir / f"fp_predictions_{day.strftime('%Y%m%d')}.jsonl", "w") as f:
            for i in range(5):
                ts = day.replace(hour=0, minute=0) + timedelta(hours=i)
                f.write(json.dumps({"timestamp": ts.isoformat(), "probability": 0.5}) + "\n")

        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "reference.parquet"

            with patch("src.mlops.reference_data.mlops_settings") as mock_settings:
                mock_settings.reference_window_days = 30
                mock_settings.get_reference_data_path.return_value = output_path

                result_path = create_reference_dataset(
                    "fp",
                    logs_dir=logs_dir,
                    days=30,
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
                # Inside the default window: 30 days ending 7 days ago.
                "timestamp": pd.date_range(
                    datetime.now(timezone.utc) - timedelta(days=10), periods=5, freq="h"
                ),
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
        assert "novelty_score" in PREDICTION_LOG_COLUMNS

    def test_is_list(self):
        """Test that PREDICTION_LOG_COLUMNS is a list."""
        assert isinstance(PREDICTION_LOG_COLUMNS, list)


# ============================================================================
# Reference window (issue #97)
# ============================================================================

from src.mlops.config import DEFAULT_DRIFT_WINDOW_DAYS  # noqa: E402
from src.mlops.reference_data import (  # noqa: E402
    count_predictions,
    reference_provenance,
    resolve_reference_window,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _rows(timestamps):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "probability": [0.5] * len(timestamps),
    })


class TestResolveReferenceWindow:
    def test_default_ends_where_the_comparison_window_starts(self):
        start, end = resolve_reference_window(90, now=NOW)

        assert end == NOW - timedelta(days=DEFAULT_DRIFT_WINDOW_DAYS)
        assert start == end - timedelta(days=90)

    def test_explicit_end_date(self):
        end_date = datetime(2026, 9, 1, tzinfo=timezone.utc)

        start, end = resolve_reference_window(30, end_date=end_date, now=NOW)

        assert (start, end) == (end_date - timedelta(days=30), end_date)

    def test_naive_end_date_is_read_as_utc(self):
        _, end = resolve_reference_window(30, end_date=datetime(2026, 9, 1), now=NOW)

        assert end == datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_end_date_and_exclude_are_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            resolve_reference_window(
                30, end_date=NOW - timedelta(days=10), exclude_recent_days=7, now=NOW
            )

    def test_future_end_date_is_rejected(self):
        with pytest.raises(ValueError, match="future"):
            resolve_reference_window(30, end_date=NOW + timedelta(days=1), now=NOW)

    @pytest.mark.parametrize("days", [90, 30])
    def test_documented_commands_do_not_overlap_the_comparison_window(self, days):
        """CLAUDE.md builds --days 90 and docs/MLOPS.md --days 30 (AC-4).

        The comparison window is the last DEFAULT_DRIFT_WINDOW_DAYS days, and
        the loader bounds it inclusively (`created_at >= now - N`), so the
        reference's exclusive end must not be after that start.
        """
        _, end = resolve_reference_window(days, now=NOW)
        comparison_start = NOW - timedelta(days=DEFAULT_DRIFT_WINDOW_DAYS)

        assert end <= comparison_start


class TestCreateReferenceWindow:
    """The window reaches the loader and is enforced on the rows (AC-1, AC-2)."""

    def test_loader_gets_dates_and_not_days(self, tmp_path):
        """`days` set on the loader discards the dates and rebuilds the overlap."""
        with patch(
            "src.mlops.reference_data.load_prediction_logs",
            return_value=_rows([datetime.now(timezone.utc) - timedelta(days=20)]),
        ) as mock_load, patch(
            "src.mlops.reference_data.datetime", wraps=datetime
        ) as mock_dt:
            mock_dt.now.return_value = NOW
            create_reference_dataset(
                "fp", days=30, output_path=tmp_path / "r.parquet", from_database=True
            )

        start, end = resolve_reference_window(30, now=NOW)
        mock_load.assert_called_once_with(
            "fp", "logs/predictions", days=None,
            start_date=start, end_date=end, from_database=True,
        )

    def test_rows_outside_the_window_are_dropped_and_the_end_is_exclusive(self, tmp_path):
        """A loader that returns everything still yields only [start, end).

        The row at exactly `end` belongs to the comparison window, whose
        loader bound is `>= now - N` -- so it must not be in the reference too.
        """
        now = datetime.now(timezone.utc)
        start, end = resolve_reference_window(30, now=now)
        inside = end - timedelta(microseconds=1)
        on_start = start
        at_end = end
        after = now - timedelta(days=1)
        before = start - timedelta(days=1)

        with patch(
            "src.mlops.reference_data.load_prediction_logs",
            return_value=_rows([before, on_start, inside, at_end, after]),
        ), patch("src.mlops.reference_data.resolve_reference_window", return_value=(start, end)):
            path = create_reference_dataset(
                "fp", days=30, output_path=tmp_path / "r.parquet", from_database=True
            )

        kept = set(pd.read_parquet(path)["timestamp"])
        assert kept == {pd.Timestamp(on_start), pd.Timestamp(inside)}

    def test_nothing_in_the_window_raises(self, tmp_path):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        with patch(
            "src.mlops.reference_data.load_prediction_logs", return_value=_rows([recent])
        ):
            with pytest.raises(ValueError, match="No prediction data found"):
                create_reference_dataset(
                    "fp", days=30, output_path=tmp_path / "r.parquet", from_database=True
                )

    def test_requested_window_round_trips_through_the_parquet_file(self, tmp_path):
        """A real write and read, so an attrs-dropping pandas fails here."""
        start, end = resolve_reference_window(30, now=datetime.now(timezone.utc))
        with patch(
            "src.mlops.reference_data.load_prediction_logs",
            return_value=_rows([start + timedelta(days=1), start + timedelta(days=2)]),
        ), patch("src.mlops.reference_data.resolve_reference_window", return_value=(start, end)):
            path = create_reference_dataset(
                "fp", days=30, output_path=tmp_path / "r.parquet", from_database=True
            )

        window = load_reference_dataset("fp", reference_path=path).attrs["reference_window"]

        assert window["requested_start"] == start.isoformat()
        assert window["requested_end"] == end.isoformat()
        assert window["end_exclusive"] is True
        assert window["source"] == "database"
        assert window["rows"] == 2


class TestReferenceProvenance:
    """What a drift report says about its baseline (AC-2, AC-3)."""

    def test_legacy_reference_reports_no_requested_window(self, tmp_path):
        """Never a window inferred from the data; observed span still given."""
        path = tmp_path / "legacy.parquet"
        _rows([NOW - timedelta(days=40), NOW - timedelta(days=20)]).to_parquet(path, index=False)
        legacy = load_reference_dataset("fp", reference_path=path)

        result = reference_provenance(legacy, _rows([NOW - timedelta(days=3)]))

        assert result["reference_window"] is None
        assert result["reference_observed"] == {
            "start": (NOW - timedelta(days=40)).isoformat(),
            "end": (NOW - timedelta(days=20)).isoformat(),
        }
        assert result["reference_overlaps_current"] is False

    def test_overlap_is_read_from_observed_timestamps(self):
        reference = _rows([NOW - timedelta(days=30), NOW - timedelta(days=2)])

        result = reference_provenance(reference, _rows([NOW - timedelta(days=5), NOW]))

        assert result["reference_overlaps_current"] is True

    def test_shared_boundary_instant_is_an_overlap(self):
        edge = NOW - timedelta(days=7)

        result = reference_provenance(_rows([edge]), _rows([edge, NOW]))

        assert result["reference_overlaps_current"] is True

    def test_unmeasurable_overlap_is_none_not_false(self):
        no_ts = pd.DataFrame({"probability": [0.5]})

        result = reference_provenance(no_ts, _rows([NOW]))

        assert result["reference_overlaps_current"] is None
        assert result["reference_observed"] is None


class TestCountPredictions:
    """A count nobody took must raise, never read as 0 (#96 AC-7)."""

    def test_missing_database_url_raises(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            count_predictions("ep", 7)

    def test_database_error_propagates(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
        with patch("sqlalchemy.create_engine", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                count_predictions("ep", 7)

    def test_returns_the_scalar_for_the_classifier_and_window(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
        engine = MagicMock()
        conn = engine.connect.return_value.__enter__.return_value
        conn.execute.return_value.scalar_one.return_value = 42

        with patch("sqlalchemy.create_engine", return_value=engine):
            assert count_predictions("ep", 7) == 42

        params = conn.execute.call_args.args[1]
        assert params["classifier_type"] == "ep"
        expected_since = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((params["since"] - expected_since).total_seconds()) < 60
