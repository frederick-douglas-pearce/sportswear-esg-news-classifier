"""Model explainability utilities for text classifiers.

Provides three complementary explanation approaches:
1. LIME: Word-level importance for individual predictions
2. SHAP Feature Groups: Aggregate importance by feature type
3. Prototype Explanations: Similar training examples

These methods address the challenge of explaining models that use
abstract features like embeddings, LSA components, or NER features.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass
class LIMEExplanation:
    """LIME explanation for a single prediction."""

    text: str
    prediction: str
    probability: float
    top_words: List[Tuple[str, float]]  # (word, weight) pairs

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "top_words": [{"word": w, "weight": round(s, 4)} for w, s in self.top_words],
            "explanation": self._generate_text_explanation(),
        }

    def _generate_text_explanation(self) -> str:
        """Generate human-readable explanation."""
        if not self.top_words:
            return f"Classified as {self.prediction} with {self.probability:.1%} confidence."

        positive_words = [w for w, s in self.top_words if s > 0][:3]
        negative_words = [w for w, s in self.top_words if s < 0][:3]

        parts = [f"Classified as {self.prediction} ({self.probability:.1%} confidence)."]

        if positive_words:
            parts.append(f"Key words supporting this: {', '.join(positive_words)}.")
        if negative_words:
            parts.append(f"Words suggesting otherwise: {', '.join(negative_words)}.")

        return " ".join(parts)


@dataclass
class FeatureGroupImportance:
    """SHAP-based feature group importance."""

    group_importance: Dict[str, float]  # Group name -> mean |SHAP|
    group_contribution: Dict[str, float]  # Group name -> percentage of total
    top_groups: List[Tuple[str, float]]  # Sorted by importance

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "group_importance": {k: round(v, 4) for k, v in self.group_importance.items()},
            "group_contribution_pct": {k: round(v * 100, 1) for k, v in self.group_contribution.items()},
            "top_groups": [(g, round(i, 4)) for g, i in self.top_groups],
        }


@dataclass
class PrototypeExplanation:
    """Prototype-based explanation using similar training examples."""

    query_text: str
    prediction: str
    probability: float
    similar_examples: List[Dict[str, Any]]  # List of similar training examples

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "similar_examples": self.similar_examples,
            "explanation": self._generate_text_explanation(),
        }

    def _generate_text_explanation(self) -> str:
        """Generate human-readable explanation."""
        if not self.similar_examples:
            return f"Classified as {self.prediction}. No similar examples found."

        # Group by label
        same_label = [ex for ex in self.similar_examples if ex.get("label") == self.prediction]
        diff_label = [ex for ex in self.similar_examples if ex.get("label") != self.prediction]

        parts = [f"Classified as {self.prediction} ({self.probability:.1%} confidence)."]

        if same_label:
            titles = [ex.get("title", "")[:50] for ex in same_label[:2]]
            parts.append(f"Similar to: {'; '.join(titles)}...")

        if diff_label:
            parts.append(f"Note: {len(diff_label)} similar examples had different labels.")

        return " ".join(parts)


class TextExplainer:
    """Explainability wrapper for text classification models.

    Provides LIME, SHAP feature group analysis, and prototype explanations.

    Example usage:
        explainer = TextExplainer(
            pipeline=trained_pipeline,
            feature_groups={"embeddings": range(0, 384), "ner": range(384, 390)},
            class_names=["false_positive", "sportswear"],
        )

        # LIME explanation
        lime_exp = explainer.explain_lime(text, brands=["Nike"])

        # Feature group importance
        shap_exp = explainer.explain_feature_groups(X_test)

        # Prototype explanation
        proto_exp = explainer.explain_prototype(
            text, X_train, train_df, n_neighbors=3
        )
    """

    def __init__(
        self,
        pipeline: Any,
        feature_groups: Optional[Dict[str, range]] = None,
        class_names: List[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize the explainer.

        Args:
            pipeline: Trained sklearn pipeline with predict_proba method
            feature_groups: Dict mapping group names to feature index ranges
            class_names: List of class names (default: ["negative", "positive"])
            threshold: Classification threshold (default: 0.5)
        """
        self.pipeline = pipeline
        self.feature_groups = feature_groups or {}
        self.class_names = class_names or ["negative", "positive"]
        self.threshold = threshold

        # LIME explainer (lazy initialization)
        self._lime_explainer = None

        # SHAP explainer (lazy initialization)
        self._shap_explainer = None

        # Nearest neighbors model (lazy initialization)
        self._nn_model = None
        self._nn_embeddings = None

    def _get_lime_explainer(self):
        """Get or create LIME explainer."""
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer
            self._lime_explainer = LimeTextExplainer(
                class_names=self.class_names,
                split_expression=r'\W+',  # Split on non-word characters
            )
        return self._lime_explainer

    def _create_text_predictor(
        self,
        transformer: Any = None,
        source_name: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> Callable:
        """Create a text-to-probability function for LIME.

        Args:
            transformer: Feature transformer (if pipeline doesn't handle text directly)
            source_name: Article source name for metadata features
            categories: Article categories for metadata features

        Returns:
            Function that takes list of texts and returns probability array
        """
        def predict_fn(texts: List[str]) -> np.ndarray:
            """Predict probabilities for a list of texts."""
            n_samples = len(texts)

            if transformer is not None:
                # Use transformer to get features
                source_names = [source_name] * n_samples if source_name else None
                cats = [categories] * n_samples if categories else None
                features = transformer.transform(texts, source_names=source_names, categories=cats)

                # Get classifier from pipeline (last step)
                if hasattr(self.pipeline, 'named_steps'):
                    classifier = self.pipeline.named_steps.get('classifier', self.pipeline)
                else:
                    classifier = self.pipeline

                return classifier.predict_proba(features)
            else:
                # Pipeline handles text directly
                return self.pipeline.predict_proba(texts)

        return predict_fn

    def explain_lime(
        self,
        text: str,
        transformer: Any = None,
        source_name: Optional[str] = None,
        categories: Optional[List[str]] = None,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> LIMEExplanation:
        """Generate LIME explanation for a single text.

        LIME perturbs the input text (removing words) and observes how
        predictions change, identifying which words are most influential.

        Args:
            text: Input text to explain
            transformer: Feature transformer (if needed)
            source_name: Article source name for metadata features
            categories: Article categories for metadata features
            num_features: Number of top features to return
            num_samples: Number of perturbed samples for LIME

        Returns:
            LIMEExplanation with word importance
        """
        explainer = self._get_lime_explainer()
        predict_fn = self._create_text_predictor(transformer, source_name, categories)

        # Get base prediction
        proba = predict_fn([text])[0]
        pred_idx = 1 if proba[1] >= self.threshold else 0
        prediction = self.class_names[pred_idx]
        probability = proba[pred_idx]

        # Generate LIME explanation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

        # Extract word importance (for positive class)
        top_words = exp.as_list(label=1)

        return LIMEExplanation(
            text=text,
            prediction=prediction,
            probability=probability,
            top_words=top_words,
        )

    def explain_feature_groups(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
        sample_size: Optional[int] = None,
        use_tree_explainer: bool = True,
    ) -> FeatureGroupImportance:
        """Compute SHAP-based feature group importance.

        Aggregates SHAP values by feature group to provide interpretable
        importance metrics for abstract features like embeddings.

        Args:
            X: Feature matrix (n_samples, n_features)
            sample_size: Number of samples to use (default: all)
            use_tree_explainer: Use TreeExplainer for tree models (faster)

        Returns:
            FeatureGroupImportance with per-group metrics
        """
        import shap

        if not self.feature_groups:
            raise ValueError("feature_groups must be set to use this method")

        # Convert sparse to dense if needed
        if sparse.issparse(X):
            X = X.toarray()

        # Sample if requested
        if sample_size and sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X[indices]

        # Get classifier from pipeline
        if hasattr(self.pipeline, 'named_steps'):
            classifier = self.pipeline.named_steps.get('classifier', self.pipeline)
        else:
            classifier = self.pipeline

        # Create SHAP explainer
        model_type = type(classifier).__name__

        if use_tree_explainer and model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X)
            # For binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use KernelExplainer for other models (slower)
            # Sample background data
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        # Aggregate by feature group
        group_importance = {}
        for group_name, indices in self.feature_groups.items():
            # Handle range objects
            if isinstance(indices, range):
                indices = list(indices)
            # Ensure indices are within bounds
            valid_indices = [i for i in indices if i < shap_values.shape[1]]
            if valid_indices:
                group_shap = shap_values[:, valid_indices]
                group_importance[group_name] = np.abs(group_shap).mean()
            else:
                group_importance[group_name] = 0.0

        # Compute relative contribution
        total_importance = sum(group_importance.values())
        if total_importance > 0:
            group_contribution = {k: v / total_importance for k, v in group_importance.items()}
        else:
            group_contribution = {k: 0.0 for k in group_importance}

        # Sort by importance
        top_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

        return FeatureGroupImportance(
            group_importance=group_importance,
            group_contribution=group_contribution,
            top_groups=top_groups,
        )

    def fit_prototype_explainer(
        self,
        X_train: Union[np.ndarray, sparse.spmatrix],
        metric: str = 'cosine',
    ) -> None:
        """Fit the nearest neighbors model for prototype explanations.

        Args:
            X_train: Training feature matrix
            metric: Distance metric (default: 'cosine')
        """
        # Convert sparse to dense if needed
        if sparse.issparse(X_train):
            X_train = X_train.toarray()

        self._nn_embeddings = X_train
        self._nn_model = NearestNeighbors(n_neighbors=10, metric=metric)
        self._nn_model.fit(X_train)

    def explain_prototype(
        self,
        text: str,
        train_df: pd.DataFrame,
        transformer: Any,
        source_name: Optional[str] = None,
        categories: Optional[List[str]] = None,
        n_neighbors: int = 3,
        title_col: str = 'title',
        label_col: str = 'is_sportswear',
        text_col: str = 'text',
    ) -> PrototypeExplanation:
        """Generate prototype-based explanation using similar training examples.

        Finds training examples most similar to the query in feature space
        and shows their labels, helping explain by example.

        Args:
            text: Input text to explain
            train_df: Training DataFrame with title, label, and text columns
            transformer: Feature transformer to encode the query
            source_name: Article source name for metadata features
            categories: Article categories for metadata features
            n_neighbors: Number of similar examples to return
            title_col: Column name for article titles
            label_col: Column name for labels
            text_col: Column name for text content

        Returns:
            PrototypeExplanation with similar training examples
        """
        if self._nn_model is None:
            raise ValueError("Call fit_prototype_explainer() first")

        # Get features for query
        source_names = [source_name] if source_name else None
        cats = [categories] if categories else None
        X_query = transformer.transform([text], source_names=source_names, categories=cats)

        if sparse.issparse(X_query):
            X_query = X_query.toarray()

        # Get prediction
        if hasattr(self.pipeline, 'named_steps'):
            classifier = self.pipeline.named_steps.get('classifier', self.pipeline)
        else:
            classifier = self.pipeline

        proba = classifier.predict_proba(X_query)[0]
        pred_idx = 1 if proba[1] >= self.threshold else 0
        prediction = self.class_names[pred_idx]
        probability = proba[pred_idx]

        # Find nearest neighbors
        distances, indices = self._nn_model.kneighbors(X_query, n_neighbors=n_neighbors)

        # Build similar examples list
        similar_examples = []
        for dist, idx in zip(distances[0], indices[0]):
            row = train_df.iloc[idx]
            label_value = row[label_col]
            # Convert to class name
            if isinstance(label_value, (int, np.integer)):
                label = self.class_names[int(label_value)]
            else:
                label = str(label_value)

            similar_examples.append({
                "title": str(row.get(title_col, ""))[:100],
                "label": label,
                "similarity": round(1 - dist, 4),  # Convert distance to similarity
                "text_preview": str(row.get(text_col, ""))[:200] + "...",
            })

        return PrototypeExplanation(
            query_text=text,
            prediction=prediction,
            probability=probability,
            similar_examples=similar_examples,
        )


def get_fp_feature_groups(
    method: str = 'sentence_transformer_ner_brands',
    lsa_n_components: int = 100,
    include_fp_indicators: bool = True,
) -> Dict[str, range]:
    """Get feature group definitions for FP classifier.

    Args:
        method: Feature engineering method used
        lsa_n_components: Number of LSA components (for LSA-based methods)
        include_fp_indicators: Whether FP indicator features are included

    Returns:
        Dictionary mapping group names to feature index ranges
    """
    if method == 'sentence_transformer_ner_brands':
        # sentence_transformer (384) + NER (6) + brand_indicators (50) + brand_summary (3)
        return {
            'sentence_embeddings': range(0, 384),
            'ner_features': range(384, 390),
            'brand_indicators': range(390, 440),
            'brand_summary': range(440, 443),
        }
    elif method == 'sentence_transformer_ner':
        return {
            'sentence_embeddings': range(0, 384),
            'ner_features': range(384, 390),
        }
    elif method == 'sentence_transformer_ner_proximity':
        # sentence_transformer (384) + NER (6) + proximity (4) + neg_context (4)
        return {
            'sentence_embeddings': range(0, 384),
            'ner_features': range(384, 390),
            'proximity_features': range(390, 394),
            'negative_context': range(394, 398),
        }
    elif method == 'tfidf_lsa_ner_proximity':
        # LSA (n) + NER (6) + brand_ner (8) + proximity (4) + neg_context (4) + optional FP indicators (8)
        idx = 0
        groups = {}

        groups['lsa_features'] = range(idx, idx + lsa_n_components)
        idx += lsa_n_components

        groups['ner_features'] = range(idx, idx + 6)
        idx += 6

        groups['brand_ner_features'] = range(idx, idx + 8)
        idx += 8

        groups['proximity_features'] = range(idx, idx + 4)
        idx += 4

        groups['negative_context'] = range(idx, idx + 4)
        idx += 4

        if include_fp_indicators:
            groups['fp_indicators'] = range(idx, idx + 13)
            idx += 13

        return groups
    elif method == 'tfidf_lsa_ner_proximity_brands':
        # LSA (n) + NER (6) + brand_ner (8) + proximity (4) + neg_context (4) + FP indicators (13) + brand_indicators (50) + brand_summary (3)
        idx = 0
        groups = {}

        groups['lsa_features'] = range(idx, idx + lsa_n_components)
        idx += lsa_n_components

        groups['ner_features'] = range(idx, idx + 6)
        idx += 6

        groups['brand_ner_features'] = range(idx, idx + 8)
        idx += 8

        groups['proximity_features'] = range(idx, idx + 4)
        idx += 4

        groups['negative_context'] = range(idx, idx + 4)
        idx += 4

        if include_fp_indicators:
            groups['fp_indicators'] = range(idx, idx + 13)
            idx += 13

        groups['brand_indicators'] = range(idx, idx + 50)
        idx += 50

        groups['brand_summary'] = range(idx, idx + 3)
        idx += 3

        return groups
    else:
        # Generic fallback - no grouping
        return {}


def get_ep_feature_groups(method: str = 'tfidf_lsa') -> Dict[str, range]:
    """Get feature group definitions for EP classifier.

    Args:
        method: Feature engineering method used

    Returns:
        Dictionary mapping group names to feature index ranges
    """
    if method == 'tfidf_lsa':
        # LSA components (100) + optional metadata (8)
        return {
            'lsa_features': range(0, 100),
            'metadata_features': range(100, 108),
        }
    elif method.startswith('tfidf_lsa_ner'):
        # LSA (100) + NER (varies)
        return {
            'lsa_features': range(0, 100),
            'ner_features': range(100, 106),
        }
    else:
        return {}


def explain_prediction(
    pipeline: Any,
    transformer: Any,
    text: str,
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    source_name: Optional[str] = None,
    categories: Optional[List[str]] = None,
    feature_groups: Optional[Dict[str, range]] = None,
    class_names: List[str] = None,
    threshold: float = 0.5,
    n_neighbors: int = 3,
) -> Dict[str, Any]:
    """Generate comprehensive explanation for a single prediction.

    Convenience function that runs all three explanation methods.

    Args:
        pipeline: Trained sklearn pipeline
        transformer: Feature transformer
        text: Input text to explain
        train_df: Training DataFrame
        X_train: Training feature matrix
        source_name: Article source name
        categories: Article categories
        feature_groups: Feature group definitions
        class_names: Class names (default: ["false_positive", "sportswear"])
        threshold: Classification threshold
        n_neighbors: Number of prototype examples

    Returns:
        Dictionary with all explanation types
    """
    class_names = class_names or ["false_positive", "sportswear"]

    explainer = TextExplainer(
        pipeline=pipeline,
        feature_groups=feature_groups or {},
        class_names=class_names,
        threshold=threshold,
    )

    # LIME explanation
    lime_exp = explainer.explain_lime(
        text=text,
        transformer=transformer,
        source_name=source_name,
        categories=categories,
    )

    # Prototype explanation
    explainer.fit_prototype_explainer(X_train)
    proto_exp = explainer.explain_prototype(
        text=text,
        train_df=train_df,
        transformer=transformer,
        source_name=source_name,
        categories=categories,
        n_neighbors=n_neighbors,
    )

    return {
        "prediction": lime_exp.prediction,
        "probability": lime_exp.probability,
        "lime": lime_exp.as_dict(),
        "prototypes": proto_exp.as_dict(),
    }
