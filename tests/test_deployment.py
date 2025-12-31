"""Tests for the deployment modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.deployment.config import (
    ClassifierType,
    CLASSIFIER_CONFIG,
    get_classifier_config,
    get_classifier_paths,
    get_risk_level,
    load_config,
    save_config,
)
from src.deployment.data import (
    create_text_features,
    load_training_data,
    split_data,
)
from src.deployment.preprocessing import (
    clean_text,
    prepare_input,
    truncate_text,
)


# ============================================================================
# Config Module Tests
# ============================================================================

class TestClassifierType:
    """Tests for ClassifierType enum."""

    def test_classifier_type_values(self):
        """Test that ClassifierType has expected values."""
        assert ClassifierType.FP.value == "fp"
        assert ClassifierType.EP.value == "ep"
        assert ClassifierType.ESG.value == "esg"

    def test_all_types_have_config(self):
        """Test that all classifier types have configurations."""
        for clf_type in ClassifierType:
            assert clf_type in CLASSIFIER_CONFIG


class TestGetClassifierConfig:
    """Tests for get_classifier_config function."""

    def test_get_fp_config(self):
        """Test getting FP classifier config."""
        config = get_classifier_config(ClassifierType.FP)

        assert "pipeline_path" in config
        assert "config_path" in config
        assert "default_threshold" in config
        assert "target_recall" in config
        assert config["model_name"] == "FP_Classifier"

    def test_get_ep_config(self):
        """Test getting EP classifier config."""
        config = get_classifier_config(ClassifierType.EP)

        assert config["model_name"] == "EP_Classifier"
        assert config["target_recall"] == 0.99

    def test_get_esg_config(self):
        """Test getting ESG classifier config."""
        config = get_classifier_config(ClassifierType.ESG)

        assert config["model_name"] == "ESG_Classifier"


class TestGetClassifierPaths:
    """Tests for get_classifier_paths function."""

    def test_get_fp_paths(self):
        """Test getting FP classifier paths."""
        pipeline_path, config_path = get_classifier_paths(ClassifierType.FP)

        assert "fp_classifier" in pipeline_path
        assert pipeline_path.endswith(".joblib")
        assert config_path.endswith(".json")

    def test_get_ep_paths(self):
        """Test getting EP classifier paths."""
        pipeline_path, config_path = get_classifier_paths(ClassifierType.EP)

        assert "ep_classifier" in pipeline_path


class TestGetRiskLevel:
    """Tests for get_risk_level function."""

    def test_low_risk(self):
        """Test low risk level (likely false positive)."""
        assert get_risk_level(0.0) == "low"
        assert get_risk_level(0.1) == "low"
        assert get_risk_level(0.29) == "low"

    def test_medium_risk(self):
        """Test medium risk level (uncertain)."""
        assert get_risk_level(0.3) == "medium"
        assert get_risk_level(0.5) == "medium"
        assert get_risk_level(0.59) == "medium"

    def test_high_risk(self):
        """Test high risk level (likely sportswear)."""
        assert get_risk_level(0.6) == "high"
        assert get_risk_level(0.8) == "high"
        assert get_risk_level(1.0) == "high"

    def test_invalid_probability_below_zero(self):
        """Test that negative probability raises error."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            get_risk_level(-0.1)

    def test_invalid_probability_above_one(self):
        """Test that probability > 1 raises error."""
        with pytest.raises(ValueError, match="Probability must be between 0 and 1"):
            get_risk_level(1.5)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_data = {
            "threshold": 0.5,
            "model_name": "TestModel",
            "cv_f2": 0.95,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        loaded = load_config(str(config_path))

        assert loaded["threshold"] == 0.5
        assert loaded["model_name"] == "TestModel"

    def test_load_missing_config(self):
        """Test loading non-existent config raises error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.json")

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, "w") as f:
            f.write("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_config(str(config_path))


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_basic_config(self, tmp_path):
        """Test saving basic config."""
        config_path = tmp_path / "config.json"
        metrics = {"cv_f2": 0.95, "cv_recall": 0.98}

        save_config(str(config_path), threshold=0.5, metrics=metrics)

        # Verify file was created and contents are correct
        with open(config_path) as f:
            saved = json.load(f)

        assert saved["threshold"] == 0.5
        assert saved["cv_f2"] == 0.95
        assert saved["cv_recall"] == 0.98

    def test_save_config_with_all_options(self, tmp_path):
        """Test saving config with all options."""
        config_path = tmp_path / "subdir/config.json"
        metrics = {"test_f2": 0.90}
        best_params = {"n_estimators": 100, "max_depth": 5}

        save_config(
            str(config_path),
            threshold=0.6,
            metrics=metrics,
            model_name="CustomModel",
            transformer_method="tfidf",
            best_params=best_params,
            target_recall=0.99,
        )

        with open(config_path) as f:
            saved = json.load(f)

        assert saved["model_name"] == "CustomModel"
        assert saved["transformer_method"] == "tfidf"
        assert saved["best_params"] == best_params
        assert saved["target_recall"] == 0.99


# ============================================================================
# Data Module Tests
# ============================================================================

class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_load_valid_jsonl(self, tmp_path):
        """Test loading valid JSONL file."""
        data = [
            {"title": "Test 1", "content": "Content 1", "is_sportswear": 1},
            {"title": "Test 2", "content": "Content 2", "is_sportswear": 0},
        ]
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        df = load_training_data(str(jsonl_path), verbose=False)

        assert len(df) == 2
        assert "title" in df.columns
        assert "content" in df.columns
        assert "is_sportswear" in df.columns

    def test_load_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            load_training_data("/nonexistent/data.jsonl", verbose=False)

    def test_load_with_verbose(self, tmp_path, capsys):
        """Test that verbose mode prints info."""
        data = [{"title": "Test", "content": "Content", "is_sportswear": 1}]
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        load_training_data(str(jsonl_path), verbose=True)

        captured = capsys.readouterr()
        assert "Loaded" in captured.out
        assert "1 records" in captured.out


class TestCreateTextFeatures:
    """Tests for create_text_features function."""

    def test_basic_text_combination(self):
        """Test basic title + content + brands combination."""
        df = pd.DataFrame({
            "title": ["Nike News"],
            "content": ["Article about Nike"],
            "brands": [["Nike"]],
            "source_name": ["ESPN"],
            "category": [["sports"]],
        })

        result = create_text_features(df, include_metadata=False)

        assert "Nike News" in result.iloc[0]
        assert "Article about Nike" in result.iloc[0]
        assert "Nike" in result.iloc[0]

    def test_with_metadata(self):
        """Test text features include metadata when enabled."""
        df = pd.DataFrame({
            "title": ["Test Title"],
            "content": ["Test Content"],
            "brands": [["Brand"]],
            "source_name": ["CNN"],
            "category": [["business"]],
        })

        result = create_text_features(df, include_metadata=True)

        # Metadata should be included as prefix
        assert "CNN" in result.iloc[0]
        assert "business" in result.iloc[0]

    def test_handles_missing_values(self):
        """Test handling of None/NaN values."""
        df = pd.DataFrame({
            "title": ["Title", None],
            "content": [None, "Content"],
            "brands": [[], ["Brand"]],
            "source_name": [None, "Source"],
            "category": [None, ["cat"]],
        })

        result = create_text_features(df, include_metadata=True)

        assert len(result) == 2
        # Should not contain "None" as string
        assert "None" not in result.iloc[0]

    def test_brands_as_string(self):
        """Test handling brands as string instead of list."""
        df = pd.DataFrame({
            "title": ["Title"],
            "content": ["Content"],
            "brands": ["Nike Adidas"],  # String instead of list
        })

        result = create_text_features(
            df, include_metadata=False, source_name_col=None, category_col=None
        )

        assert "Nike Adidas" in result.iloc[0]


class TestSplitData:
    """Tests for split_data function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for splitting."""
        np.random.seed(42)
        return pd.DataFrame({
            "title": [f"Title {i}" for i in range(100)],
            "content": [f"Content {i}" for i in range(100)],
            "is_sportswear": [1] * 70 + [0] * 30,  # 70% positive
        })

    def test_default_split_ratios(self, sample_df):
        """Test default 60/20/20 split."""
        train, val, test = split_data(sample_df, verbose=False)

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_custom_split_ratios(self, sample_df):
        """Test custom split ratios."""
        train, val, test = split_data(
            sample_df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            verbose=False,
        )

        # Allow for small rounding variations in split
        assert 68 <= len(train) <= 72
        assert 13 <= len(val) <= 17
        assert 13 <= len(test) <= 17
        # Total should still be 100
        assert len(train) + len(val) + len(test) == 100

    def test_stratification(self, sample_df):
        """Test that splits are stratified."""
        train, val, test = split_data(sample_df, verbose=False)

        # Each split should have approximately 70% positive
        for split_df in [train, val, test]:
            pos_ratio = split_df["is_sportswear"].mean()
            assert 0.65 <= pos_ratio <= 0.75

    def test_invalid_ratios(self, sample_df):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_data(
                sample_df,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1
                verbose=False,
            )

    def test_reproducibility(self, sample_df):
        """Test that same random_state gives same results."""
        train1, val1, test1 = split_data(sample_df, random_state=42, verbose=False)
        train2, val2, test2 = split_data(sample_df, random_state=42, verbose=False)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)


# ============================================================================
# Preprocessing Module Tests
# ============================================================================

class TestCleanText:
    """Tests for clean_text function."""

    def test_lowercase_conversion(self):
        """Test text is converted to lowercase."""
        result = clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_url_removal(self):
        """Test URLs are removed."""
        result = clean_text("Visit https://example.com for more")
        assert "https://" not in result
        assert "example.com" not in result

    def test_email_removal(self):
        """Test email addresses are removed."""
        result = clean_text("Contact us at test@example.com")
        assert "@" not in result

    def test_whitespace_normalization(self):
        """Test excessive whitespace is normalized."""
        result = clean_text("Hello    World  \n\t Test")
        assert "  " not in result
        assert result == "hello world test"

    def test_special_character_removal(self):
        """Test special characters are removed."""
        result = clean_text("Hello™ World® #test")
        assert "™" not in result
        assert "®" not in result
        assert "#" not in result

    def test_keeps_basic_punctuation(self):
        """Test basic punctuation is preserved."""
        result = clean_text("Hello, world! How are you?")
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_empty_input(self):
        """Test empty input returns empty string."""
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_non_string_input(self):
        """Test non-string input returns empty string."""
        assert clean_text(123) == ""
        assert clean_text([]) == ""


class TestPrepareInput:
    """Tests for prepare_input function."""

    def test_basic_combination(self):
        """Test basic title + content combination."""
        result = prepare_input(
            title="Test Title",
            content="Test content here",
            include_metadata=False,
        )

        assert "Test Title" in result
        assert "Test content here" in result

    def test_with_brands(self):
        """Test brands are included."""
        result = prepare_input(
            title="News",
            content="Content",
            brands=["Nike", "Adidas"],
            include_metadata=False,
        )

        assert "Nike" in result
        assert "Adidas" in result

    def test_with_metadata(self):
        """Test metadata is included when enabled."""
        result = prepare_input(
            title="News",
            content="Content",
            source_name="ESPN",
            category=["sports", "business"],
            include_metadata=True,
        )

        assert "Source ESPN" in result
        assert "Category sports business" in result

    def test_without_metadata(self):
        """Test metadata is excluded when disabled."""
        result = prepare_input(
            title="News",
            content="Content",
            source_name="ESPN",
            category=["sports"],
            include_metadata=False,
        )

        assert "Source" not in result
        assert "ESPN" not in result

    def test_handles_none_values(self):
        """Test None values are handled gracefully."""
        result = prepare_input(
            title=None,
            content=None,
            brands=None,
            source_name=None,
            category=None,
        )

        assert result == ""

    def test_brands_as_string(self):
        """Test brands can be passed as string."""
        result = prepare_input(
            title="News",
            content="Content",
            brands="Nike Adidas",  # String instead of list
            include_metadata=False,
        )

        assert "Nike Adidas" in result


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Test short text is not modified."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_long_text_truncated(self):
        """Test long text is truncated."""
        text = "A" * 100
        result = truncate_text(text, max_length=50)
        assert len(result) <= 50

    def test_truncates_at_word_boundary(self):
        """Test truncation happens at word boundary."""
        text = "Hello world this is a test"
        result = truncate_text(text, max_length=15)

        # Should truncate at a space, not mid-word
        assert not result.endswith("worl")

    def test_default_max_length(self):
        """Test default max length is 10000."""
        text = "A" * 9999
        result = truncate_text(text)
        assert len(result) == 9999  # Under limit, unchanged

        text = "A" * 11000
        result = truncate_text(text)
        assert len(result) <= 10000


# ============================================================================
# Base Classifier Tests (with mocking)
# ============================================================================

class TestBaseClassifier:
    """Tests for BaseClassifier functionality via concrete implementations."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock sklearn pipeline."""
        pipeline = MagicMock()
        pipeline.predict_proba = MagicMock(
            return_value=np.array([[0.3, 0.7]])
        )
        return pipeline

    @pytest.fixture
    def mock_config(self):
        """Create mock config data."""
        return {
            "threshold": 0.5,
            "model_name": "TestModel",
            "target_recall": 0.98,
            "transformer_method": "test_method",
            "best_params": {"n_estimators": 100},
            "cv_f2": 0.95,
            "cv_recall": 0.98,
            "cv_precision": 0.90,
        }

    def test_truncate_text_method(self):
        """Test _truncate_text method from base class."""
        from src.deployment.base import BaseClassifier

        # Create a minimal concrete subclass for testing
        class TestClassifier(BaseClassifier):
            CLASSIFIER_TYPE = "test"

            def _get_default_paths(self):
                return "pipeline.joblib", "config.json"

            def _build_result(self, probability, is_positive):
                return {"result": is_positive}

            def _prepare_input(self, title, content, brands, source_name, category):
                return f"{title} {content}"

        # Test via instance method after mocking init
        with patch.object(BaseClassifier, '__init__', return_value=None):
            classifier = TestClassifier()
            classifier.pipeline = MagicMock()
            classifier.config = {}
            classifier.threshold = 0.5
            classifier.model_name = "Test"

            # Test truncation
            short = classifier._truncate_text("Short text")
            assert short == "Short text"

            long = classifier._truncate_text("A" * 20000)
            assert len(long) <= 10000


# ============================================================================
# FP Classifier Tests
# ============================================================================

class TestFPClassifier:
    """Tests for FPClassifier."""

    @pytest.fixture
    def mock_fp_classifier(self, tmp_path):
        """Create an FPClassifier with mocked pipeline and config."""
        # Create a real minimal pipeline file (will be overwritten by mock)
        pipeline_path = tmp_path / "pipeline.joblib"
        pipeline_path.touch()

        # Create config
        config_data = {
            "threshold": 0.5,
            "model_name": "FP_Test",
            "target_recall": 0.98,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock joblib.load to return our mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba = MagicMock(
            return_value=np.array([[0.3, 0.7]])
        )

        from src.deployment.fp.classifier import FPClassifier

        with patch('joblib.load', return_value=mock_pipeline):
            classifier = FPClassifier(
                pipeline_path=str(pipeline_path),
                config_path=str(config_path),
            )
        # Attach the mock pipeline so predict works
        classifier.pipeline = mock_pipeline
        return classifier

    def test_fp_classifier_type(self, mock_fp_classifier):
        """Test FPClassifier has correct type."""
        assert mock_fp_classifier.CLASSIFIER_TYPE == "fp"

    def test_fp_predict_returns_is_sportswear(self, mock_fp_classifier):
        """Test FP prediction includes is_sportswear field."""
        result = mock_fp_classifier.predict("Test article about Nike")

        assert "is_sportswear" in result
        assert "probability" in result
        assert "risk_level" in result
        assert "threshold" in result

    def test_fp_predict_risk_level(self, mock_fp_classifier):
        """Test FP prediction includes correct risk level."""
        # Mock returns 0.7 probability
        result = mock_fp_classifier.predict("Test")
        assert result["risk_level"] == "high"

    def test_fp_predict_from_fields(self, mock_fp_classifier):
        """Test prediction from article fields."""
        result = mock_fp_classifier.predict_from_fields(
            title="Nike launches new shoe",
            content="The athletic giant announced...",
            brands=["Nike"],
            source_name="ESPN",
            category=["sports"],
        )

        assert "is_sportswear" in result


# ============================================================================
# EP Classifier Tests
# ============================================================================

class TestEPClassifier:
    """Tests for EPClassifier."""

    @pytest.fixture
    def mock_ep_classifier(self, tmp_path):
        """Create an EPClassifier with mocked pipeline and config."""
        # Create a real minimal pipeline file (will be overwritten by mock)
        pipeline_path = tmp_path / "pipeline.joblib"
        pipeline_path.touch()

        # Create config
        config_data = {
            "threshold": 0.5,
            "model_name": "EP_Test",
            "target_recall": 0.99,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Mock joblib.load to return our mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba = MagicMock(
            return_value=np.array([[0.2, 0.8]])
        )

        from src.deployment.ep.classifier import EPClassifier

        with patch('joblib.load', return_value=mock_pipeline):
            classifier = EPClassifier(
                pipeline_path=str(pipeline_path),
                config_path=str(config_path),
            )
        classifier.pipeline = mock_pipeline
        return classifier

    def test_ep_classifier_type(self, mock_ep_classifier):
        """Test EPClassifier has correct type."""
        assert mock_ep_classifier.CLASSIFIER_TYPE == "ep"

    def test_ep_predict_returns_has_esg(self, mock_ep_classifier):
        """Test EP prediction includes has_esg field."""
        result = mock_ep_classifier.predict("Article about carbon emissions")

        assert "has_esg" in result
        assert "probability" in result
        assert "threshold" in result
        # EP classifier should NOT have risk_level
        assert "risk_level" not in result

    def test_ep_predict_from_fields(self, mock_ep_classifier):
        """Test prediction from article fields."""
        result = mock_ep_classifier.predict_from_fields(
            title="Nike announces carbon neutrality",
            content="The company pledged to reduce emissions...",
            brands=["Nike"],
            source_name="Reuters",
        )

        assert "has_esg" in result
        assert result["has_esg"] is True  # 0.8 > 0.5 threshold


# ============================================================================
# Batch Prediction Tests
# ============================================================================

class TestBatchPrediction:
    """Tests for batch prediction functionality."""

    @pytest.fixture
    def mock_classifier(self, tmp_path):
        """Create a classifier for batch testing."""
        pipeline_path = tmp_path / "pipeline.joblib"
        pipeline_path.touch()

        config_data = {"threshold": 0.5, "model_name": "Test"}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba = MagicMock(
            return_value=np.array([
                [0.3, 0.7],
                [0.8, 0.2],
                [0.4, 0.6],
            ])
        )

        from src.deployment.fp.classifier import FPClassifier

        with patch('joblib.load', return_value=mock_pipeline):
            classifier = FPClassifier(
                pipeline_path=str(pipeline_path),
                config_path=str(config_path),
            )
        classifier.pipeline = mock_pipeline
        return classifier

    def test_batch_predict_returns_list(self, mock_classifier):
        """Test batch prediction returns list of results."""
        texts = ["Text 1", "Text 2", "Text 3"]
        results = mock_classifier.predict_batch(texts)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_predict_empty_list(self, mock_classifier):
        """Test batch prediction with empty list."""
        results = mock_classifier.predict_batch([])
        assert results == []

    def test_batch_predict_from_fields(self, mock_classifier):
        """Test batch prediction from article dictionaries."""
        articles = [
            {"title": "Title 1", "content": "Content 1", "brands": ["Nike"]},
            {"title": "Title 2", "content": "Content 2"},
            {"title": "Title 3", "content": "Content 3", "source_name": "ESPN"},
        ]

        results = mock_classifier.predict_batch_from_fields(articles)

        assert len(results) == 3
        assert all("is_sportswear" in r for r in results)

    def test_batch_predict_probability_toggle(self, mock_classifier):
        """Test probability can be excluded from results."""
        texts = ["Text 1", "Text 2"]
        results = mock_classifier.predict_batch(texts, return_proba=False)

        assert all("probability" not in r for r in results)


# ============================================================================
# Model Info Tests
# ============================================================================

class TestGetModelInfo:
    """Tests for get_model_info method."""

    @pytest.fixture
    def configured_classifier(self, tmp_path):
        """Create a classifier with full config."""
        pipeline_path = tmp_path / "pipeline.joblib"
        pipeline_path.touch()

        config_data = {
            "threshold": 0.6,
            "model_name": "TestModel",
            "target_recall": 0.98,
            "transformer_method": "sentence_transformer",
            "best_params": {"n_estimators": 100},
            "cv_f2": 0.95,
            "cv_recall": 0.98,
            "cv_precision": 0.90,
            "test_f2": 0.93,
            "test_recall": 0.97,
            "test_precision": 0.88,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba = MagicMock(return_value=np.array([[0.5, 0.5]]))

        from src.deployment.fp.classifier import FPClassifier

        with patch('joblib.load', return_value=mock_pipeline):
            classifier = FPClassifier(
                pipeline_path=str(pipeline_path),
                config_path=str(config_path),
            )
        classifier.pipeline = mock_pipeline
        return classifier

    def test_model_info_structure(self, configured_classifier):
        """Test model info contains expected fields."""
        info = configured_classifier.get_model_info()

        assert "classifier_type" in info
        assert "model_name" in info
        assert "threshold" in info
        assert "target_recall" in info
        assert "metrics" in info

    def test_model_info_metrics(self, configured_classifier):
        """Test model info contains metrics."""
        info = configured_classifier.get_model_info()
        metrics = info["metrics"]

        assert metrics["cv_f2"] == 0.95
        assert metrics["cv_recall"] == 0.98
        assert metrics["test_f2"] == 0.93

    def test_classifier_repr(self, configured_classifier):
        """Test classifier string representation."""
        repr_str = repr(configured_classifier)

        assert "FPClassifier" in repr_str
        assert "TestModel" in repr_str
        assert "0.600" in repr_str  # threshold


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in deployment modules."""

    def test_missing_pipeline_file(self, tmp_path):
        """Test error when pipeline file doesn't exist."""
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"threshold": 0.5}, f)

        from src.deployment.fp.classifier import FPClassifier

        with pytest.raises(FileNotFoundError, match="Pipeline not found"):
            FPClassifier(
                pipeline_path="/nonexistent/pipeline.joblib",
                config_path=str(config_path),
            )

    def test_missing_config_file(self, tmp_path):
        """Test error when config file doesn't exist."""
        pipeline_path = tmp_path / "pipeline.joblib"
        pipeline_path.touch()

        from src.deployment.fp.classifier import FPClassifier

        with pytest.raises(FileNotFoundError, match="Config not found"):
            with patch('joblib.load', return_value=MagicMock()):
                FPClassifier(
                    pipeline_path=str(pipeline_path),
                    config_path="/nonexistent/config.json",
                )
