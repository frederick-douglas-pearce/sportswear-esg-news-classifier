"""Tests for fp1_nb data utilities."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.fp1_nb.data_utils import (
    analyze_target_stats,
    load_jsonl_data,
    split_train_val_test,
)


class TestLoadJsonlData:
    """Tests for load_jsonl_data function."""

    def test_load_valid_jsonl(self, tmp_path):
        """Test loading valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [
            {"id": 1, "text": "hello", "label": 0},
            {"id": 2, "text": "world", "label": 1},
            {"id": 3, "text": "test", "label": 0},
        ]
        with open(jsonl_file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        df = load_jsonl_data(str(jsonl_file), verbose=False)

        assert len(df) == 3
        assert list(df.columns) == ["id", "text", "label"]
        assert df["id"].tolist() == [1, 2, 3]

    def test_load_with_verbose(self, tmp_path, capsys):
        """Test verbose output."""
        jsonl_file = tmp_path / "test.jsonl"
        data = [{"id": 1, "text": "hello"}]
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(data[0]) + "\n")

        load_jsonl_data(str(jsonl_file), verbose=True)

        captured = capsys.readouterr()
        assert "Loaded 1 records" in captured.out
        assert "Columns:" in captured.out

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_jsonl_data("/nonexistent/path/file.jsonl")


class TestAnalyzeTargetStats:
    """Tests for analyze_target_stats function."""

    def test_balanced_dataset(self, capsys):
        """Test with balanced dataset."""
        df = pd.DataFrame({
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })

        results = analyze_target_stats(df, "label", plot=False)

        assert results["total_samples"] == 10
        assert results["imbalance_ratio"] == 1.0
        assert results["is_imbalanced"] == False  # Use == for numpy bool
        assert results["class_counts"] == {0: 5, 1: 5}

        captured = capsys.readouterr()
        assert "[OK] Dataset is reasonably balanced" in captured.out

    def test_imbalanced_dataset(self, capsys):
        """Test with imbalanced dataset."""
        df = pd.DataFrame({
            "label": [0] * 95 + [1] * 5
        })

        results = analyze_target_stats(
            df, "label", imbalance_threshold=10.0, plot=False
        )

        assert results["total_samples"] == 100
        assert results["imbalance_ratio"] == 19.0
        assert results["is_imbalanced"] == True  # Use == for numpy bool
        assert results["majority_class"] == 0
        assert results["minority_class"] == 1

        captured = capsys.readouterr()
        assert "[WARNING] Dataset is imbalanced" in captured.out

    def test_with_label_names(self, capsys):
        """Test with custom label names."""
        df = pd.DataFrame({"label": [0, 0, 1]})

        analyze_target_stats(
            df, "label",
            label_names=["Negative", "Positive"],
            plot=False
        )

        captured = capsys.readouterr()
        assert "Negative" in captured.out or "Positive" in captured.out

    def test_custom_imbalance_threshold(self):
        """Test custom imbalance threshold."""
        df = pd.DataFrame({"label": [0] * 8 + [1] * 2})

        # With threshold 5, should be imbalanced (ratio = 4)
        results_high = analyze_target_stats(
            df, "label", imbalance_threshold=3.0, plot=False
        )
        assert results_high["is_imbalanced"] == True  # Use == for numpy bool

        # With threshold 10, should not be imbalanced
        results_low = analyze_target_stats(
            df, "label", imbalance_threshold=10.0, plot=False
        )
        assert results_low["is_imbalanced"] == False  # Use == for numpy bool


class TestSplitTrainValTest:
    """Tests for split_train_val_test function."""

    def test_default_split_ratios(self):
        """Test default 60/20/20 split."""
        df = pd.DataFrame({"x": range(100), "label": [0] * 50 + [1] * 50})

        train, val, test = split_train_val_test(
            df, target_col="label", verbose=False
        )

        # Allow small variations due to stratification
        assert len(train) == pytest.approx(60, abs=2)
        assert len(val) == pytest.approx(20, abs=2)
        assert len(test) == pytest.approx(20, abs=2)
        assert len(train) + len(val) + len(test) == 100

    def test_custom_split_ratios(self):
        """Test custom split ratios."""
        df = pd.DataFrame({"x": range(100), "label": [0] * 50 + [1] * 50})

        train, val, test = split_train_val_test(
            df,
            target_col="label",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            verbose=False
        )

        assert len(train) == pytest.approx(70, abs=2)
        assert len(val) == pytest.approx(15, abs=2)
        assert len(test) == pytest.approx(15, abs=2)

    def test_stratification_preserved(self):
        """Test that stratification is preserved in splits."""
        # Create imbalanced dataset
        df = pd.DataFrame({
            "x": range(100),
            "label": [0] * 80 + [1] * 20
        })

        train, val, test = split_train_val_test(
            df, target_col="label", verbose=False
        )

        # Each split should have approximately 80/20 class balance
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            class_ratio = split["label"].mean()
            assert class_ratio == pytest.approx(0.2, abs=0.1), \
                f"{name} split has incorrect class balance"

    def test_invalid_ratios_raise_error(self):
        """Test that invalid ratios raise ValueError."""
        df = pd.DataFrame({"x": range(10), "label": [0, 1] * 5})

        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_train_val_test(
                df,
                target_col="label",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
                verbose=False
            )

    def test_reproducibility(self):
        """Test that same random_state gives same splits."""
        df = pd.DataFrame({"x": range(100), "label": [0] * 50 + [1] * 50})

        train1, val1, test1 = split_train_val_test(
            df, target_col="label", random_state=42, verbose=False
        )
        train2, val2, test2 = split_train_val_test(
            df, target_col="label", random_state=42, verbose=False
        )

        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                       train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(val1.reset_index(drop=True),
                                       val2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True),
                                       test2.reset_index(drop=True))

    def test_no_stratification(self):
        """Test split without stratification."""
        df = pd.DataFrame({"x": range(100)})

        train, val, test = split_train_val_test(
            df, target_col=None, verbose=False
        )

        assert len(train) + len(val) + len(test) == 100

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        df = pd.DataFrame({"x": range(100), "label": [0] * 50 + [1] * 50})

        split_train_val_test(df, target_col="label", verbose=True)

        captured = capsys.readouterr()
        assert "TRAIN/VALIDATION/TEST SPLIT" in captured.out
        assert "Total samples: 100" in captured.out
        assert "Train:" in captured.out
        assert "Validation:" in captured.out
        assert "Test:" in captured.out
