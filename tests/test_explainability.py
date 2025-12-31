"""Tests for the explainability module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.fp3_nb.explainability import (
    TextExplainer,
    LIMEExplanation,
    FeatureGroupImportance,
    PrototypeExplanation,
    get_fp_feature_groups,
    get_ep_feature_groups,
    explain_prediction,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_feature_groups():
    """Create sample feature groups for testing."""
    return {
        'lsa_features': range(0, 10),
        'ner_features': range(10, 15),
        'brand_features': range(15, 20),
    }


@pytest.fixture
def mock_classifier():
    """Create a mock classifier for testing."""
    classifier = MagicMock(spec=RandomForestClassifier)
    classifier.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    classifier.predict = MagicMock(return_value=np.array([1]))
    classifier.classes_ = np.array([0, 1])
    return classifier


@pytest.fixture
def mock_pipeline(mock_classifier):
    """Create a mock sklearn pipeline for testing."""
    pipeline = MagicMock(spec=Pipeline)
    pipeline.named_steps = {'classifier': mock_classifier}
    pipeline.predict_proba = mock_classifier.predict_proba
    pipeline.predict = mock_classifier.predict
    return pipeline


@pytest.fixture
def text_explainer(mock_pipeline, sample_feature_groups):
    """Create a TextExplainer instance for testing."""
    return TextExplainer(
        pipeline=mock_pipeline,
        feature_groups=sample_feature_groups,
        class_names=['false_positive', 'sportswear'],
        threshold=0.5,
    )


@pytest.fixture
def mock_transformer():
    """Create a mock feature transformer for testing."""
    transformer = MagicMock()
    transformer.transform = MagicMock(return_value=np.random.rand(1, 20))
    return transformer


@pytest.fixture
def sample_train_df():
    """Create a sample training DataFrame for prototype tests."""
    return pd.DataFrame({
        'title': ['Nike Sustainability Report', 'Wild Puma Spotted', 'Adidas ESG Goals'],
        'text_features': ['nike sustainability carbon', 'puma wildlife animal', 'adidas esg environment'],
        'is_sportswear': [1, 0, 1],
        'source_name': ['ESPN', 'NatGeo', 'Reuters'],
        'category': [['sports'], ['wildlife'], ['business']],
    })


# ============================================================================
# Feature Group Tests
# ============================================================================

class TestGetFeatureGroups:
    """Tests for get_fp_feature_groups and get_ep_feature_groups functions."""

    def test_get_fp_feature_groups_tfidf_lsa_ner_proximity_brands(self):
        """Test FP feature groups for tfidf_lsa_ner_proximity_brands method."""
        groups = get_fp_feature_groups('tfidf_lsa_ner_proximity_brands')

        assert 'lsa_features' in groups
        assert 'ner_features' in groups
        assert 'proximity_features' in groups
        assert 'negative_context' in groups
        assert 'brand_indicators' in groups
        assert 'brand_summary' in groups

        # Check that indices are ranges
        assert isinstance(groups['lsa_features'], range)
        assert len(list(groups['lsa_features'])) == 100

    def test_get_fp_feature_groups_unknown_method(self):
        """Test FP feature groups returns empty dict for unknown method."""
        groups = get_fp_feature_groups('unknown_method')
        assert groups == {}

    def test_get_ep_feature_groups_tfidf_lsa(self):
        """Test EP feature groups for tfidf_lsa method."""
        groups = get_ep_feature_groups('tfidf_lsa')

        assert 'lsa_features' in groups
        assert 'metadata_features' in groups

        # Check LSA has 100 components
        assert len(list(groups['lsa_features'])) == 100

    def test_get_ep_feature_groups_tfidf_lsa_ner(self):
        """Test EP feature groups for tfidf_lsa_ner method."""
        groups = get_ep_feature_groups('tfidf_lsa_ner')

        assert 'lsa_features' in groups
        assert 'ner_features' in groups

    def test_get_ep_feature_groups_unknown_method(self):
        """Test EP feature groups returns empty dict for unknown method."""
        groups = get_ep_feature_groups('unknown_method')
        assert groups == {}


# ============================================================================
# TextExplainer Initialization Tests
# ============================================================================

class TestTextExplainerInit:
    """Tests for TextExplainer initialization."""

    def test_init_with_pipeline(self, mock_pipeline, sample_feature_groups):
        """Test TextExplainer initializes correctly with a pipeline."""
        explainer = TextExplainer(
            pipeline=mock_pipeline,
            feature_groups=sample_feature_groups,
            class_names=['neg', 'pos'],
            threshold=0.5,
        )

        assert explainer.pipeline == mock_pipeline
        assert explainer.feature_groups == sample_feature_groups
        assert explainer.class_names == ['neg', 'pos']
        assert explainer.threshold == 0.5

    def test_init_with_classifier_directly(self, mock_classifier, sample_feature_groups):
        """Test TextExplainer works when given a classifier directly."""
        explainer = TextExplainer(
            pipeline=mock_classifier,
            feature_groups=sample_feature_groups,
            class_names=['neg', 'pos'],
            threshold=0.5,
        )

        assert explainer.pipeline == mock_classifier

    def test_init_with_empty_feature_groups(self, mock_pipeline):
        """Test TextExplainer works with empty feature groups."""
        explainer = TextExplainer(
            pipeline=mock_pipeline,
            feature_groups={},
            class_names=['neg', 'pos'],
            threshold=0.5,
        )

        assert explainer.feature_groups == {}


# ============================================================================
# SHAP Feature Group Importance Tests
# ============================================================================

class TestExplainFeatureGroups:
    """Tests for SHAP feature group importance."""

    def test_explain_feature_groups_with_tree_explainer(self, sample_feature_groups):
        """Test explain_feature_groups with TreeExplainer (mocked)."""
        # Create a real RandomForest for this test
        from sklearn.ensemble import RandomForestClassifier

        X_train = np.random.rand(50, 20)
        y_train = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        clf.fit(X_train, y_train)

        explainer = TextExplainer(
            pipeline=clf,
            feature_groups=sample_feature_groups,
            class_names=['neg', 'pos'],
            threshold=0.5,
        )

        X_test = np.random.rand(10, 20)

        # This should work with real SHAP
        result = explainer.explain_feature_groups(
            X_test,
            sample_size=5,
            use_tree_explainer=True,
        )

        assert isinstance(result, FeatureGroupImportance)
        assert 'lsa_features' in result.group_importance
        assert len(result.top_groups) == 3

    def test_explain_feature_groups_contributions_sum_to_one(self, sample_feature_groups):
        """Test that feature group contributions sum to approximately 1."""
        from sklearn.ensemble import RandomForestClassifier

        X_train = np.random.rand(50, 20)
        y_train = np.random.randint(0, 2, 50)

        clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
        clf.fit(X_train, y_train)

        explainer = TextExplainer(
            pipeline=clf,
            feature_groups=sample_feature_groups,
            class_names=['neg', 'pos'],
            threshold=0.5,
        )

        X_test = np.random.rand(10, 20)

        result = explainer.explain_feature_groups(
            X_test,
            sample_size=5,
            use_tree_explainer=True,
        )

        total_contribution = sum(result.group_contribution.values())
        assert abs(total_contribution - 1.0) < 0.01


# ============================================================================
# LIME Explanation Tests
# ============================================================================

class TestExplainLime:
    """Tests for LIME word-level explanations."""

    def test_explain_lime_returns_explanation(self, text_explainer, mock_transformer):
        """Test explain_lime returns LIMEExplanation."""
        with patch('lime.lime_text.LimeTextExplainer') as mock_lime:
            mock_exp = MagicMock()
            mock_exp.as_list = MagicMock(return_value=[('nike', 0.5), ('shoe', 0.3)])
            mock_lime_instance = MagicMock()
            mock_lime_instance.explain_instance = MagicMock(return_value=mock_exp)
            mock_lime.return_value = mock_lime_instance

            result = text_explainer.explain_lime(
                text="Nike announces new sustainability initiative",
                transformer=mock_transformer,
                source_name="ESPN",
                categories=['sports'],
                num_features=5,
            )

            assert isinstance(result, LIMEExplanation)
            assert result.text == "Nike announces new sustainability initiative"
            assert len(result.top_words) == 2

    def test_explain_lime_prediction_uses_threshold(self, sample_feature_groups):
        """Test LIME explanation uses threshold for prediction."""
        # Create a mock classifier without named_steps (not a Pipeline)
        # Use spec to prevent MagicMock from auto-creating attributes
        from sklearn.ensemble import RandomForestClassifier

        def mock_predict_proba(X):
            if isinstance(X, MagicMock):
                n_samples = 1
            elif hasattr(X, 'shape'):
                n_samples = X.shape[0]
            else:
                n_samples = 1
            return np.array([[0.6, 0.4]] * n_samples)  # Below 0.5 threshold

        mock_clf = MagicMock(spec=RandomForestClassifier)
        mock_clf.predict_proba = mock_predict_proba

        # Create mock transformer that returns numpy arrays
        def mock_transform(*args, **kwargs):
            return np.random.rand(1, 20)

        mock_transformer = MagicMock()
        mock_transformer.transform = mock_transform

        explainer = TextExplainer(
            pipeline=mock_clf,
            feature_groups=sample_feature_groups,
            class_names=['false_positive', 'sportswear'],
            threshold=0.5,
        )

        with patch('lime.lime_text.LimeTextExplainer') as mock_lime:
            mock_exp = MagicMock()
            mock_exp.as_list = MagicMock(return_value=[('nike', 0.5)])
            mock_lime_instance = MagicMock()
            mock_lime_instance.explain_instance = MagicMock(return_value=mock_exp)
            mock_lime.return_value = mock_lime_instance

            result = explainer.explain_lime(
                text="Nike sustainability",
                transformer=mock_transformer,
                source_name="ESPN",
                categories=['sports'],
            )

            # 0.4 < 0.5 threshold, so prediction should be 'false_positive'
            assert result.prediction == 'false_positive'


# ============================================================================
# Prototype Explanation Tests
# ============================================================================

class TestPrototypeExplainer:
    """Tests for prototype-based explanations."""

    def test_fit_prototype_explainer(self, text_explainer):
        """Test fitting the prototype explainer."""
        X_train = np.random.rand(100, 20)

        text_explainer.fit_prototype_explainer(X_train, metric='cosine')

        assert text_explainer._nn_model is not None

    def test_explain_prototype_returns_explanation(
        self, text_explainer, mock_transformer, sample_train_df
    ):
        """Test explain_prototype returns PrototypeExplanation."""
        X_train = np.random.rand(3, 20)
        text_explainer.fit_prototype_explainer(X_train, metric='cosine')

        result = text_explainer.explain_prototype(
            text="Nike sustainability goals",
            train_df=sample_train_df,
            transformer=mock_transformer,
            source_name="ESPN",
            categories=['sports'],
            n_neighbors=2,
            title_col='title',
            label_col='is_sportswear',
            text_col='text_features',
        )

        assert isinstance(result, PrototypeExplanation)
        assert result.query_text == "Nike sustainability goals"
        assert len(result.similar_examples) <= 2

    def test_explain_prototype_without_fitting_raises_error(
        self, text_explainer, mock_transformer, sample_train_df
    ):
        """Test explain_prototype raises error if not fitted."""
        with pytest.raises(ValueError, match="fit_prototype_explainer"):
            text_explainer.explain_prototype(
                text="Nike sustainability",
                train_df=sample_train_df,
                transformer=mock_transformer,
                source_name="ESPN",
                categories=['sports'],
            )

    def test_fit_prototype_with_different_metrics(self, text_explainer):
        """Test fitting prototype explainer with different distance metrics."""
        X_train = np.random.rand(50, 20)

        text_explainer.fit_prototype_explainer(X_train, metric='euclidean')
        assert text_explainer._nn_model is not None

        text_explainer.fit_prototype_explainer(X_train, metric='cosine')
        assert text_explainer._nn_model is not None


# ============================================================================
# LIMEExplanation Dataclass Tests
# ============================================================================

class TestLIMEExplanation:
    """Tests for LIMEExplanation dataclass."""

    def test_generate_text_explanation_positive(self):
        """Test text explanation generation for positive prediction."""
        exp = LIMEExplanation(
            text="Nike sustainability report",
            prediction='sportswear',
            probability=0.85,
            top_words=[('nike', 0.5), ('sustainability', 0.3), ('report', -0.1)],
        )

        explanation = exp._generate_text_explanation()

        assert 'sportswear' in explanation
        assert '85' in explanation
        assert 'nike' in explanation.lower()
        assert 'sustainability' in explanation.lower()

    def test_generate_text_explanation_negative(self):
        """Test text explanation generation for negative prediction."""
        exp = LIMEExplanation(
            text="Wild puma spotted in forest",
            prediction='false_positive',
            probability=0.75,
            top_words=[('puma', -0.4), ('wild', -0.3), ('forest', -0.2)],
        )

        explanation = exp._generate_text_explanation()

        assert 'false_positive' in explanation
        assert '75' in explanation

    def test_generate_text_explanation_empty_words(self):
        """Test text explanation with no top words."""
        exp = LIMEExplanation(
            text="Some text",
            prediction='sportswear',
            probability=0.9,
            top_words=[],
        )

        explanation = exp._generate_text_explanation()
        assert 'sportswear' in explanation
        assert '90' in explanation

    def test_as_dict(self):
        """Test LIMEExplanation as_dict method."""
        exp = LIMEExplanation(
            text="Nike report",
            prediction='sportswear',
            probability=0.85,
            top_words=[('nike', 0.5)],
        )

        result = exp.as_dict()

        assert 'prediction' in result
        assert 'probability' in result
        assert 'top_words' in result
        assert 'explanation' in result
        assert result['prediction'] == 'sportswear'


# ============================================================================
# PrototypeExplanation Dataclass Tests
# ============================================================================

class TestPrototypeExplanation:
    """Tests for PrototypeExplanation dataclass."""

    def test_generate_text_explanation_unanimous(self):
        """Test explanation when all neighbors agree."""
        exp = PrototypeExplanation(
            query_text="Wild puma in forest",
            prediction='false_positive',
            probability=0.8,
            similar_examples=[
                {'title': 'Puma wildlife', 'label': 'false_positive', 'similarity': 0.9},
                {'title': 'Zoo puma', 'label': 'false_positive', 'similarity': 0.85},
            ],
        )

        explanation = exp._generate_text_explanation()

        assert len(explanation) > 0

    def test_generate_text_explanation_mixed(self):
        """Test explanation when neighbors disagree."""
        exp = PrototypeExplanation(
            query_text="Puma sneakers review",
            prediction='sportswear',
            probability=0.6,
            similar_examples=[
                {'title': 'Nike shoes', 'label': 'sportswear', 'similarity': 0.8},
                {'title': 'Wild puma', 'label': 'false_positive', 'similarity': 0.75},
            ],
        )

        explanation = exp._generate_text_explanation()
        assert len(explanation) > 0

    def test_generate_text_explanation_no_examples(self):
        """Test explanation with no similar examples."""
        exp = PrototypeExplanation(
            query_text="Some query",
            prediction='sportswear',
            probability=0.7,
            similar_examples=[],
        )

        explanation = exp._generate_text_explanation()
        assert 'No similar examples' in explanation

    def test_as_dict(self):
        """Test PrototypeExplanation as_dict method."""
        exp = PrototypeExplanation(
            query_text="Nike query",
            prediction='sportswear',
            probability=0.85,
            similar_examples=[{'title': 'Example', 'label': 'sportswear'}],
        )

        result = exp.as_dict()

        assert 'prediction' in result
        assert 'probability' in result
        assert 'similar_examples' in result
        assert 'explanation' in result


# ============================================================================
# FeatureGroupImportance Dataclass Tests
# ============================================================================

class TestFeatureGroupImportance:
    """Tests for FeatureGroupImportance dataclass."""

    def test_top_groups_ordering(self):
        """Test that top_groups maintains order."""
        importance = FeatureGroupImportance(
            group_importance={'a': 0.5, 'b': 0.3, 'c': 0.2},
            group_contribution={'a': 0.5, 'b': 0.3, 'c': 0.2},
            top_groups=[('a', 0.5), ('b', 0.3), ('c', 0.2)],
        )

        assert importance.top_groups[0][0] == 'a'
        assert importance.top_groups[0][1] == 0.5

    def test_as_dict(self):
        """Test FeatureGroupImportance as_dict method."""
        importance = FeatureGroupImportance(
            group_importance={'a': 0.5, 'b': 0.3},
            group_contribution={'a': 0.6, 'b': 0.4},
            top_groups=[('a', 0.5), ('b', 0.3)],
        )

        result = importance.as_dict()

        assert 'group_importance' in result
        assert 'group_contribution_pct' in result
        assert 'top_groups' in result


# ============================================================================
# explain_prediction Integration Tests
# ============================================================================

class TestExplainPrediction:
    """Tests for the explain_prediction convenience function."""

    def test_explain_prediction_returns_all_explanations(self, sample_feature_groups, sample_train_df):
        """Test explain_prediction returns dict with all explanation types."""
        from sklearn.ensemble import RandomForestClassifier

        # Create mock classifier with proper predict_proba function
        def mock_predict_proba(X):
            if isinstance(X, MagicMock):
                n_samples = 1
            elif hasattr(X, 'shape'):
                n_samples = X.shape[0]
            else:
                n_samples = 1
            return np.array([[0.3, 0.7]] * n_samples)

        mock_clf = MagicMock(spec=RandomForestClassifier)
        mock_clf.predict_proba = mock_predict_proba

        def mock_transform(*args, **kwargs):
            return np.random.rand(1, 20)

        mock_transformer = MagicMock()
        mock_transformer.transform = mock_transform

        X_train = np.random.rand(3, 20)

        with patch('lime.lime_text.LimeTextExplainer') as mock_lime:
            mock_exp = MagicMock()
            mock_exp.as_list = MagicMock(return_value=[('nike', 0.5)])
            mock_lime_instance = MagicMock()
            mock_lime_instance.explain_instance = MagicMock(return_value=mock_exp)
            mock_lime.return_value = mock_lime_instance

            result = explain_prediction(
                pipeline=mock_clf,
                transformer=mock_transformer,
                text="Nike sustainability",
                train_df=sample_train_df,
                X_train=X_train,
                source_name="ESPN",
                categories=['sports'],
                feature_groups=sample_feature_groups,
                class_names=['false_positive', 'sportswear'],
                threshold=0.5,
            )

            assert 'prediction' in result
            assert 'probability' in result
            assert 'lime' in result
            assert 'prototypes' in result

    def test_explain_prediction_with_defaults(self, sample_train_df):
        """Test explain_prediction works with default parameters."""
        from sklearn.ensemble import RandomForestClassifier

        # Create mock classifier with proper predict_proba function
        def mock_predict_proba(X):
            if isinstance(X, MagicMock):
                n_samples = 1
            elif hasattr(X, 'shape'):
                n_samples = X.shape[0]
            else:
                n_samples = 1
            return np.array([[0.3, 0.7]] * n_samples)

        mock_clf = MagicMock(spec=RandomForestClassifier)
        mock_clf.predict_proba = mock_predict_proba

        def mock_transform(*args, **kwargs):
            return np.random.rand(1, 20)

        mock_transformer = MagicMock()
        mock_transformer.transform = mock_transform

        X_train = np.random.rand(3, 20)

        with patch('lime.lime_text.LimeTextExplainer') as mock_lime:
            mock_exp = MagicMock()
            mock_exp.as_list = MagicMock(return_value=[('nike', 0.5)])
            mock_lime_instance = MagicMock()
            mock_lime_instance.explain_instance = MagicMock(return_value=mock_exp)
            mock_lime.return_value = mock_lime_instance

            result = explain_prediction(
                pipeline=mock_clf,
                transformer=mock_transformer,
                text="Nike sustainability",
                train_df=sample_train_df,
                X_train=X_train,
            )

            assert 'prediction' in result
            assert 'lime' in result
            assert 'prototypes' in result
