"""Integration tests for ESG News Classifier pipelines.

These tests verify end-to-end functionality of classifier pipelines,
from data loading through prediction.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_fp_articles():
    """Sample articles for FP classifier testing."""
    return [
        {
            "title": "Nike Announces Carbon Neutrality Goals",
            "content": "Nike Inc. unveiled ambitious sustainability goals...",
            "brands": ["Nike"],
            "source_name": "ESPN",
            "category": ["sports", "business"],
            "is_sportswear": 1,
        },
        {
            "title": "Wild Puma Spotted in Montana",
            "content": "Wildlife researchers captured footage of a puma...",
            "brands": ["Puma"],
            "source_name": "National Geographic",
            "category": ["wildlife", "science"],
            "is_sportswear": 0,
        },
        {
            "title": "Adidas ESG Report Shows Progress",
            "content": "Adidas released their annual ESG report...",
            "brands": ["Adidas"],
            "source_name": "Reuters",
            "category": ["business"],
            "is_sportswear": 1,
        },
    ]


@pytest.fixture
def sample_ep_articles():
    """Sample articles for EP classifier testing."""
    return [
        {
            "title": "Nike Announces Carbon Neutrality Goals",
            "content": "Nike Inc. unveiled ambitious sustainability goals for carbon emissions reduction...",
            "brands": ["Nike"],
            "source_name": "Reuters",
            "category": ["business", "environment"],
            "has_esg": 1,
        },
        {
            "title": "Nike Releases New Air Jordan Collection",
            "content": "Nike announced the release of a new Air Jordan Retro collection...",
            "brands": ["Nike"],
            "source_name": "ESPN",
            "category": ["sports"],
            "has_esg": 0,
        },
        {
            "title": "Adidas Worker Rights Investigation",
            "content": "Reports surface about labor conditions at Adidas supplier factories...",
            "brands": ["Adidas"],
            "source_name": "BBC",
            "category": ["business"],
            "has_esg": 1,
        },
    ]


# ============================================================================
# FP Classifier Pipeline Tests
# ============================================================================

class TestFPClassifierPipeline:
    """Integration tests for False Positive classifier pipeline."""

    def test_fp_classifier_training_flow(self, sample_fp_articles):
        """Test the complete FP classifier training flow."""
        # Convert to DataFrame
        df = pd.DataFrame(sample_fp_articles)

        # Create text features
        df['text_features'] = df.apply(
            lambda row: f"{row['title']} {row['content']}", axis=1
        )

        # Simple feature extraction for testing
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['text_features'])
        y = df['is_sportswear'].values

        # Train model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X.toarray(), y)

        # Verify predictions
        predictions = clf.predict(X.toarray())
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)

    def test_fp_classifier_threshold_logic(self, sample_fp_articles):
        """Test threshold-based classification logic."""
        # Simulate probability predictions
        probabilities = np.array([
            [0.2, 0.8],  # High sportswear probability
            [0.7, 0.3],  # Low sportswear probability
            [0.4, 0.6],  # Medium sportswear probability
        ])

        # Test different thresholds
        threshold_high = 0.7
        threshold_low = 0.4

        predictions_high = (probabilities[:, 1] >= threshold_high).astype(int)
        predictions_low = (probabilities[:, 1] >= threshold_low).astype(int)

        # High threshold: only first article passes
        assert predictions_high.tolist() == [1, 0, 0]

        # Low threshold: first and third pass
        assert predictions_low.tolist() == [1, 0, 1]

    def test_fp_pipeline_serialization(self, sample_fp_articles, tmp_path):
        """Test that FP pipeline can be serialized and loaded."""
        import joblib

        # Create and train a simple pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = pd.DataFrame(sample_fp_articles)
        df['text_features'] = df.apply(
            lambda row: f"{row['title']} {row['content']}", axis=1
        )

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100)),
            ('clf', RandomForestClassifier(n_estimators=5, random_state=42)),
        ])

        pipeline.fit(df['text_features'], df['is_sportswear'])

        # Save pipeline
        pipeline_path = tmp_path / "test_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Load and verify
        loaded_pipeline = joblib.load(pipeline_path)
        predictions = loaded_pipeline.predict(df['text_features'])

        assert len(predictions) == 3


# ============================================================================
# EP Classifier Pipeline Tests
# ============================================================================

class TestEPClassifierPipeline:
    """Integration tests for ESG Pre-filter classifier pipeline."""

    def test_ep_classifier_training_flow(self, sample_ep_articles):
        """Test the complete EP classifier training flow."""
        df = pd.DataFrame(sample_ep_articles)
        df['text_features'] = df.apply(
            lambda row: f"{row['title']} {row['content']}", axis=1
        )

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['text_features'])
        y = df['has_esg'].values

        clf = LogisticRegression(random_state=42)
        clf.fit(X.toarray(), y)

        predictions = clf.predict(X.toarray())
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)

    def test_ep_classifier_probability_calibration(self, sample_ep_articles):
        """Test that EP classifier probabilities are well-calibrated."""
        df = pd.DataFrame(sample_ep_articles)
        df['text_features'] = df.apply(
            lambda row: f"{row['title']} {row['content']}", axis=1
        )

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['text_features'])
        y = df['has_esg'].values

        clf = LogisticRegression(random_state=42)
        clf.fit(X.toarray(), y)

        # Get probabilities
        probas = clf.predict_proba(X.toarray())

        # Verify probability properties
        assert probas.shape == (3, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestDeploymentConfiguration:
    """Tests for deployment configuration handling."""

    def test_config_file_format(self, tmp_path):
        """Test that configuration files are valid JSON."""
        config = {
            "threshold": 0.5,
            "target_recall": 0.99,
            "model_name": "RandomForest",
            "cv_f2": 0.95,
            "cv_recall": 0.98,
            "cv_precision": 0.92,
        }

        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Load and verify
        with open(config_path) as f:
            loaded_config = json.load(f)

        assert loaded_config == config

    def test_config_threshold_validation(self):
        """Test threshold validation logic."""
        def validate_threshold(threshold):
            if not isinstance(threshold, (int, float)):
                raise ValueError("Threshold must be numeric")
            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between 0 and 1")
            return True

        assert validate_threshold(0.5)
        assert validate_threshold(0.0)
        assert validate_threshold(1.0)

        with pytest.raises(ValueError):
            validate_threshold(-0.1)

        with pytest.raises(ValueError):
            validate_threshold(1.5)


# ============================================================================
# Cross-Component Integration Tests
# ============================================================================

class TestCrossComponentIntegration:
    """Tests for integration between different system components."""

    def test_training_to_prediction_flow(self, sample_fp_articles):
        """Test complete flow from training data to predictions."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Prepare data
        df = pd.DataFrame(sample_fp_articles)
        df['text_features'] = df.apply(
            lambda row: f"{row['title']} {row['content']}", axis=1
        )

        # Train
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['text_features'])
        y = df['is_sportswear'].values

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X.toarray(), y)

        # New article for prediction
        new_article = "Under Armour announces new running shoes with recycled materials"
        X_new = vectorizer.transform([new_article])

        # Predict
        prediction = clf.predict(X_new.toarray())[0]
        probability = clf.predict_proba(X_new.toarray())[0]

        assert prediction in [0, 1]
        assert len(probability) == 2
        assert abs(sum(probability) - 1.0) < 0.01

    def test_batch_prediction_consistency(self, sample_fp_articles):
        """Test that batch and single predictions are consistent."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = pd.DataFrame(sample_fp_articles)
        texts = [f"{row['title']} {row['content']}" for _, row in df.iterrows()]

        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        y = df['is_sportswear'].values

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X.toarray(), y)

        # Batch prediction
        batch_predictions = clf.predict(X.toarray())

        # Single predictions
        single_predictions = []
        for text in texts:
            X_single = vectorizer.transform([text])
            pred = clf.predict(X_single.toarray())[0]
            single_predictions.append(pred)

        # Should be identical
        assert list(batch_predictions) == single_predictions


# ============================================================================
# Data Pipeline Tests
# ============================================================================

class TestDataPipeline:
    """Tests for data processing pipeline."""

    def test_text_feature_creation(self, sample_fp_articles):
        """Test text feature creation from article components."""
        df = pd.DataFrame(sample_fp_articles)

        # Test combining title and content
        df['text_features'] = df['title'] + ' ' + df['content']

        for idx, row in df.iterrows():
            assert row['title'] in row['text_features']
            assert row['content'] in row['text_features']

    def test_category_handling(self, sample_fp_articles):
        """Test handling of category lists."""
        df = pd.DataFrame(sample_fp_articles)

        # Categories should be lists
        for idx, row in df.iterrows():
            assert isinstance(row['category'], list)
            assert len(row['category']) > 0

    def test_brand_extraction(self, sample_fp_articles):
        """Test brand extraction from articles."""
        df = pd.DataFrame(sample_fp_articles)

        brands_mentioned = set()
        for idx, row in df.iterrows():
            brands_mentioned.update(row['brands'])

        assert 'Nike' in brands_mentioned
        assert 'Puma' in brands_mentioned
        assert 'Adidas' in brands_mentioned
