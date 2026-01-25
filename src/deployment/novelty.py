"""Novelty scoring for detecting out-of-distribution articles.

This module provides topic novelty detection by comparing article embeddings
against cluster centroids learned from training data. High novelty scores
indicate articles with topics the model hasn't seen, which may cause
performance degradation.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) for embeddings:
- Lower dimensionality helps with curse of dimensionality in clustering
- Fast, local, free (no API costs)
- Can score ALL articles before FP classification

Example:
    # During training/setup
    scorer = NoveltyScorer()
    scorer.fit_from_texts(training_texts, n_clusters=25)
    scorer.save(model_dir / "novelty_centroids.pkl")

    # During prediction
    scorer = NoveltyScorer(model_dir / "novelty_centroids.pkl")
    result = scorer.score_text("Nike announces new sustainability initiative...")
    print(f"Novelty: {result.score:.3f}")
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default sentence-transformer model
# all-MiniLM-L6-v2: 384 dimensions, fast, good quality
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class NoveltyScore:
    """Result of novelty scoring for a single article.

    Attributes:
        score: Novelty score from 0-1. Higher = more novel/unfamiliar.
               0 = identical to a training cluster centroid
               1 = orthogonal to all training clusters
        nearest_cluster: Index of the nearest cluster centroid
        distance: Raw cosine distance to nearest centroid (1 - similarity)
    """

    score: float
    nearest_cluster: int
    distance: float


class NoveltyScorer:
    """Scores articles for topic novelty relative to training distribution.

    Uses K-Means clustering on sentence-transformer embeddings from training
    data. At inference time, computes cosine distance to nearest cluster
    centroid as a novelty score.

    The sentence-transformer model (all-MiniLM-L6-v2) produces 384-dimensional
    embeddings, which helps avoid the curse of dimensionality that affects
    higher-dimensional embeddings like OpenAI's 1536-dim vectors.

    Attributes:
        n_clusters: Number of K-Means clusters
        centroids: Cluster centroid vectors (n_clusters × embedding_dim)
        enabled: Whether scoring is available (centroids loaded)
        embedding_model: Name of the sentence-transformer model
    """

    DEFAULT_N_CLUSTERS = 25

    def __init__(
        self,
        centroids_path: Path | str | None = None,
        n_clusters: int | None = None,
        embedding_model: str | None = None,
    ):
        """Initialize NoveltyScorer.

        Args:
            centroids_path: Path to saved centroids file. If provided and exists,
                           centroids are loaded automatically.
            n_clusters: Number of clusters for K-Means. Only used during fit().
                       Defaults to NOVELTY_N_CLUSTERS env var or 25.
            embedding_model: Sentence-transformer model name. Defaults to
                            NOVELTY_EMBEDDING_MODEL env var or all-MiniLM-L6-v2.
        """
        self.n_clusters = n_clusters or int(
            os.getenv("NOVELTY_N_CLUSTERS", str(self.DEFAULT_N_CLUSTERS))
        )
        self.embedding_model = embedding_model or os.getenv(
            "NOVELTY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.centroids: np.ndarray | None = None
        self._kmeans: KMeans | None = None
        self._sentence_model = None  # Lazy loaded
        self._fitted = False
        self._metadata: dict[str, Any] = {}

        if centroids_path:
            path = Path(centroids_path)
            if path.exists():
                self.load(path)

    def _ensure_sentence_model(self):
        """Lazy load the sentence-transformer model."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded sentence-transformer model: {self.embedding_model}")
        return self._sentence_model

    @property
    def enabled(self) -> bool:
        """Whether novelty scoring is available."""
        return self._fitted and self.centroids is not None

    @property
    def embedding_dim(self) -> int | None:
        """Dimension of embeddings, if fitted."""
        if self.centroids is not None:
            return self.centroids.shape[1]
        return None

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute sentence-transformer embeddings for texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array (n_texts × embedding_dim)
        """
        model = self._ensure_sentence_model()
        embeddings = model.encode(texts, show_progress_bar=show_progress)
        return np.array(embeddings)

    def embed_text(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (embedding_dim,)
        """
        return self.embed_texts([text], show_progress=False)[0]

    def fit(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
        random_state: int = 42,
    ) -> "NoveltyScorer":
        """Fit K-Means on pre-computed embeddings to create reference centroids.

        Args:
            embeddings: Training embeddings array (n_samples × embedding_dim)
            n_clusters: Number of clusters. Defaults to instance setting.
            random_state: Random state for reproducibility

        Returns:
            self for method chaining

        Raises:
            ValueError: If embeddings array is invalid
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")

        n_samples, embedding_dim = embeddings.shape
        n_clusters = n_clusters or self.n_clusters

        # Ensure we don't have more clusters than samples
        # Also need at least 2 samples per cluster on average for silhouette_score
        max_clusters = max(2, n_samples - 1)
        if n_clusters > max_clusters:
            logger.warning(
                f"n_clusters ({n_clusters}) > max allowable ({max_clusters}), "
                f"reducing to {max_clusters}"
            )
            n_clusters = max_clusters

        logger.info(
            f"Fitting NoveltyScorer with {n_clusters} clusters on "
            f"{n_samples} samples ({embedding_dim}-dim embeddings)"
        )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Fit K-Means
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self._kmeans.fit(normalized)

        # Store normalized centroids
        self.centroids = self._kmeans.cluster_centers_
        self.n_clusters = n_clusters
        self._fitted = True

        # Compute and store metadata
        labels = self._kmeans.labels_
        silhouette = silhouette_score(normalized, labels) if n_clusters > 1 else 0.0

        self._metadata = {
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "embedding_dim": embedding_dim,
            "embedding_model": self.embedding_model,
            "silhouette_score": float(silhouette),
            "inertia": float(self._kmeans.inertia_),
            "cluster_sizes": np.bincount(labels).tolist(),
        }

        logger.info(
            f"Fitted {n_clusters} clusters. "
            f"Silhouette score: {silhouette:.3f}, "
            f"Inertia: {self._kmeans.inertia_:.2f}"
        )

        return self

    def fit_from_texts(
        self,
        texts: list[str],
        n_clusters: int | None = None,
        random_state: int = 42,
        show_progress: bool = True,
    ) -> "NoveltyScorer":
        """Fit K-Means on texts by computing embeddings first.

        Convenience method that handles embedding computation.

        Args:
            texts: Training texts to cluster
            n_clusters: Number of clusters. Defaults to instance setting.
            random_state: Random state for reproducibility
            show_progress: Whether to show embedding progress bar

        Returns:
            self for method chaining
        """
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        return self.fit(embeddings, n_clusters=n_clusters, random_state=random_state)

    def score(self, embedding: np.ndarray) -> NoveltyScore:
        """Score a single article embedding for novelty.

        Args:
            embedding: Article embedding vector (embedding_dim,) or (1, embedding_dim)

        Returns:
            NoveltyScore with score, nearest cluster, and distance

        Raises:
            RuntimeError: If scorer hasn't been fitted
            ValueError: If embedding dimension doesn't match
        """
        if not self.enabled:
            raise RuntimeError("NoveltyScorer not fitted. Call fit() or load() first.")

        # Handle 1D or 2D input
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embedding.shape[1]} doesn't match "
                f"expected {self.embedding_dim}"
            )

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Compute cosine similarity to all centroids
        similarities = cosine_similarity(embedding, self.centroids)[0]

        # Find nearest cluster
        nearest_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[nearest_idx])

        # Convert to distance/novelty score
        # similarity ranges from -1 to 1, but normalized embeddings typically 0 to 1
        # distance = 1 - similarity gives 0 (identical) to 2 (opposite)
        # We cap at 1 for the novelty score
        distance = 1.0 - max_similarity
        novelty_score = min(1.0, max(0.0, distance))

        return NoveltyScore(
            score=novelty_score,
            nearest_cluster=nearest_idx,
            distance=distance,
        )

    def score_text(self, text: str) -> NoveltyScore:
        """Score a single text for novelty.

        Convenience method that handles embedding computation.

        Args:
            text: Article text to score

        Returns:
            NoveltyScore with score, nearest cluster, and distance
        """
        embedding = self.embed_text(text)
        return self.score(embedding)

    def score_batch(self, embeddings: np.ndarray) -> list[NoveltyScore]:
        """Score multiple embeddings for novelty.

        Args:
            embeddings: Array of embeddings (n_samples × embedding_dim)

        Returns:
            List of NoveltyScore objects
        """
        if not self.enabled:
            raise RuntimeError("NoveltyScorer not fitted. Call fit() or load() first.")

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")

        # Normalize all embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        # Compute similarities to all centroids
        similarities = cosine_similarity(normalized, self.centroids)

        # Find nearest cluster for each
        nearest_indices = np.argmax(similarities, axis=1)
        max_similarities = similarities[np.arange(len(similarities)), nearest_indices]

        # Convert to novelty scores
        distances = 1.0 - max_similarities
        scores = np.clip(distances, 0.0, 1.0)

        return [
            NoveltyScore(
                score=float(scores[i]),
                nearest_cluster=int(nearest_indices[i]),
                distance=float(distances[i]),
            )
            for i in range(len(embeddings))
        ]

    def score_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[NoveltyScore]:
        """Score multiple texts for novelty.

        Convenience method that handles embedding computation.

        Args:
            texts: List of texts to score
            show_progress: Whether to show progress bar

        Returns:
            List of NoveltyScore objects
        """
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        return self.score_batch(embeddings)

    def save(self, path: Path | str) -> None:
        """Save centroids and metadata to disk.

        Args:
            path: Path to save centroids (will use joblib format)
        """
        if not self.enabled:
            raise RuntimeError("Cannot save: NoveltyScorer not fitted")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "centroids": self.centroids,
            "n_clusters": self.n_clusters,
            "embedding_model": self.embedding_model,
            "metadata": self._metadata,
        }

        joblib.dump(data, path)
        logger.info(f"Saved novelty centroids to {path}")

    def load(self, path: Path | str) -> None:
        """Load centroids from disk.

        Args:
            path: Path to saved centroids file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Centroids file not found: {path}")

        data = joblib.load(path)

        self.centroids = data["centroids"]
        self.n_clusters = data["n_clusters"]
        self.embedding_model = data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self._metadata = data.get("metadata", {})
        self._fitted = True

        logger.info(
            f"Loaded novelty centroids from {path}: "
            f"{self.n_clusters} clusters, {self.embedding_dim}-dim "
            f"(model: {self.embedding_model})"
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about the fitted model.

        Returns:
            Dictionary with fitting statistics
        """
        return {
            "enabled": self.enabled,
            "n_clusters": self.n_clusters,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            **self._metadata,
        }

    def get_training_novelty_stats(self, embeddings: np.ndarray) -> dict[str, float]:
        """Compute novelty statistics on training data (for validation).

        Training data should have low novelty scores. This method helps
        validate that the scorer is working correctly.

        Args:
            embeddings: Training embeddings to score

        Returns:
            Dictionary with mean, std, min, max, p90, p99 novelty scores
        """
        scores = self.score_batch(embeddings)
        novelty_values = [s.score for s in scores]

        return {
            "mean": float(np.mean(novelty_values)),
            "std": float(np.std(novelty_values)),
            "min": float(np.min(novelty_values)),
            "max": float(np.max(novelty_values)),
            "p90": float(np.percentile(novelty_values, 90)),
            "p99": float(np.percentile(novelty_values, 99)),
        }
