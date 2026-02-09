"""Embedding methods for article deduplication analysis."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "BaseEmbedder":
        """Fit the embedder on training texts.

        Args:
            texts: List of text strings to fit on

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Embedding matrix (n_texts x embedding_dim)
        """
        pass

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            texts: List of text strings

        Returns:
            Embedding matrix (n_texts x embedding_dim)
        """
        return self.fit(texts).transform(texts)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for the embedder."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformer based embedder.

    Uses pre-trained models from sentence-transformers library
    for high-quality semantic embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        """Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformer model.
                       Options include:
                       - 'all-MiniLM-L6-v2' (384-dim, fast, good quality)
                       - 'all-mpnet-base-v2' (768-dim, best quality)
                       - 'paraphrase-MiniLM-L6-v2' (384-dim, optimized for paraphrase)
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._embedding_dim = None

    def _ensure_model(self):
        """Lazy load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Get embedding dimension from model
            self._embedding_dim = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        """Sentence transformers are pre-trained, so fit is a no-op.

        Args:
            texts: List of text strings (ignored)

        Returns:
            self for method chaining
        """
        self._ensure_model()
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Embedding matrix (n_texts x embedding_dim)
        """
        self._ensure_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        self._ensure_model()
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return a descriptive name for the embedder."""
        return f"SentenceTransformer({self.model_name})"


class TfidfLsaEmbedder(BaseEmbedder):
    """TF-IDF + LSA embedder.

    Uses TF-IDF vectorization followed by Truncated SVD (LSA)
    for dimensionality reduction.
    """

    def __init__(
        self,
        max_features: int = 10000,
        n_components: int = 100,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        normalize: bool = True,
    ):
        """Initialize the embedder.

        Args:
            max_features: Maximum vocabulary size for TF-IDF
            n_components: Number of LSA components (embedding dimension)
            ngram_range: Range of n-grams to include
            sublinear_tf: Apply sublinear TF scaling (1 + log(tf))
            normalize: Whether to L2-normalize embeddings
        """
        self.max_features = max_features
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.normalize = normalize

        self._tfidf = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfLsaEmbedder":
        """Fit the TF-IDF vectorizer and LSA model.

        Args:
            texts: List of text strings to fit on

        Returns:
            self for method chaining
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
        )

        self._svd = TruncatedSVD(n_components=self.n_components, random_state=42)

        # Fit TF-IDF and transform
        tfidf_matrix = self._tfidf.fit_transform(texts)

        # Fit LSA
        self._svd.fit(tfidf_matrix)
        self._fitted = True

        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Embedding matrix (n_texts x n_components)
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        tfidf_matrix = self._tfidf.transform(texts)
        embeddings = self._svd.transform(tfidf_matrix)

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.n_components

    @property
    def name(self) -> str:
        """Return a descriptive name for the embedder."""
        return f"TF-IDF+LSA(dim={self.n_components})"


def get_embedder(
    method: str = "sentence-transformer",
    **kwargs,
) -> BaseEmbedder:
    """Factory function to create an embedder by name.

    Args:
        method: Embedder type ('sentence-transformer', 'tfidf-lsa')
        **kwargs: Additional arguments passed to the embedder

    Returns:
        Configured embedder instance
    """
    embedders = {
        "sentence-transformer": SentenceTransformerEmbedder,
        "all-MiniLM-L6-v2": lambda **kw: SentenceTransformerEmbedder(
            model_name="all-MiniLM-L6-v2", **kw
        ),
        "all-mpnet-base-v2": lambda **kw: SentenceTransformerEmbedder(
            model_name="all-mpnet-base-v2", **kw
        ),
        "paraphrase-MiniLM-L6-v2": lambda **kw: SentenceTransformerEmbedder(
            model_name="paraphrase-MiniLM-L6-v2", **kw
        ),
        "tfidf-lsa": TfidfLsaEmbedder,
    }

    if method not in embedders:
        raise ValueError(f"Unknown embedder: {method}. Options: {list(embedders.keys())}")

    return embedders[method](**kwargs)
