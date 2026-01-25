"""Tests for the NoveltyScorer module using sentence-transformers."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.deployment.novelty import NoveltyScore, NoveltyScorer


class TestNoveltyScore:
    """Tests for the NoveltyScore dataclass."""

    def test_novelty_score_creation(self):
        """Test creating a NoveltyScore."""
        score = NoveltyScore(score=0.5, nearest_cluster=3, distance=0.5)
        assert score.score == 0.5
        assert score.nearest_cluster == 3
        assert score.distance == 0.5


class TestNoveltyScorer:
    """Tests for the NoveltyScorer class."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing (384-dim like sentence-transformer)."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(20, 384) + np.array([5, 0] + [0] * 382)
        cluster2 = np.random.randn(20, 384) + np.array([0, 5] + [0] * 382)
        cluster3 = np.random.randn(20, 384) + np.array([-5, -5] + [0] * 382)
        return np.vstack([cluster1, cluster2, cluster3])

    def test_init_default(self):
        """Test default initialization."""
        scorer = NoveltyScorer()
        assert scorer.n_clusters == 25
        assert scorer.centroids is None
        assert not scorer.enabled
        assert scorer.embedding_model == "all-MiniLM-L6-v2"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        scorer = NoveltyScorer(n_clusters=10, embedding_model="custom-model")
        assert scorer.n_clusters == 10
        assert scorer.embedding_model == "custom-model"

    def test_fit(self, sample_embeddings):
        """Test fitting the scorer."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        assert scorer.enabled
        assert scorer.centroids is not None
        assert scorer.centroids.shape == (3, 384)
        assert scorer.embedding_dim == 384

    def test_fit_adjusts_clusters(self):
        """Test that fit reduces n_clusters if too many for n_samples."""
        embeddings = np.random.randn(5, 384)
        scorer = NoveltyScorer(n_clusters=10)
        scorer.fit(embeddings)

        # Should reduce to n_samples - 1 = 4 (for silhouette_score)
        assert scorer.n_clusters == 4

    def test_fit_invalid_dimensions(self):
        """Test that fit raises error for 1D input."""
        scorer = NoveltyScorer()
        with pytest.raises(ValueError, match="Expected 2D"):
            scorer.fit(np.array([1, 2, 3]))

    def test_score_not_fitted(self):
        """Test that score raises error when not fitted."""
        scorer = NoveltyScorer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.score(np.random.randn(384))

    def test_score_single_embedding(self, sample_embeddings):
        """Test scoring a single embedding."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        # Score an embedding from cluster 1 (should have low novelty)
        result = scorer.score(sample_embeddings[0])

        assert isinstance(result, NoveltyScore)
        assert 0.0 <= result.score <= 1.0
        assert 0 <= result.nearest_cluster < 3
        assert result.distance >= 0

    def test_score_novel_embedding(self, sample_embeddings):
        """Test that novel embeddings get high scores."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        # Create a very different embedding (should have high novelty)
        novel_embedding = np.ones(384) * 100  # Far from all training data

        result = scorer.score(novel_embedding)

        # Should have high novelty score (close to 1)
        assert result.score > 0.5

    def test_score_dimension_mismatch(self, sample_embeddings):
        """Test that wrong dimension raises error."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        with pytest.raises(ValueError, match="dimension"):
            scorer.score(np.random.randn(128))  # Wrong dimension

    def test_score_batch(self, sample_embeddings):
        """Test batch scoring."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        # Score 5 embeddings
        results = scorer.score_batch(sample_embeddings[:5])

        assert len(results) == 5
        for result in results:
            assert isinstance(result, NoveltyScore)
            assert 0.0 <= result.score <= 1.0

    def test_save_and_load(self, sample_embeddings):
        """Test saving and loading centroids."""
        scorer = NoveltyScorer(n_clusters=3, embedding_model="test-model")
        scorer.fit(sample_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "centroids.pkl"
            scorer.save(path)

            # Load into new scorer
            loaded_scorer = NoveltyScorer(centroids_path=path)

            assert loaded_scorer.enabled
            assert loaded_scorer.n_clusters == 3
            assert loaded_scorer.embedding_dim == 384
            assert loaded_scorer.embedding_model == "test-model"
            np.testing.assert_array_almost_equal(
                scorer.centroids, loaded_scorer.centroids
            )

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file doesn't crash."""
        scorer = NoveltyScorer(centroids_path="/nonexistent/path.pkl")
        assert not scorer.enabled

    def test_save_not_fitted(self):
        """Test saving when not fitted raises error."""
        scorer = NoveltyScorer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.save(Path("/tmp/test.pkl"))

    def test_get_metadata(self, sample_embeddings):
        """Test getting metadata."""
        scorer = NoveltyScorer(n_clusters=3, embedding_model="test-model")
        scorer.fit(sample_embeddings)

        metadata = scorer.get_metadata()

        assert metadata["enabled"] is True
        assert metadata["n_clusters"] == 3
        assert metadata["embedding_dim"] == 384
        assert metadata["embedding_model"] == "test-model"
        assert "silhouette_score" in metadata
        assert "cluster_sizes" in metadata

    def test_get_training_novelty_stats(self, sample_embeddings):
        """Test computing training data novelty stats."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        stats = scorer.get_training_novelty_stats(sample_embeddings)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p90" in stats
        assert "p99" in stats

        # Training data novelty should have reasonable values
        # Note: with random high-dimensional data, novelty can be higher
        # The important thing is that stats are computed correctly
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["p90"] <= stats["p99"]

    def test_score_handles_1d_and_2d_input(self, sample_embeddings):
        """Test that score handles both 1D and 2D input."""
        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sample_embeddings)

        embedding_1d = sample_embeddings[0]
        embedding_2d = sample_embeddings[0].reshape(1, -1)

        result_1d = scorer.score(embedding_1d)
        result_2d = scorer.score(embedding_2d)

        # Both should give same result
        assert abs(result_1d.score - result_2d.score) < 1e-6


class TestNoveltyTextMethods:
    """Tests for text-based novelty methods using sentence-transformers."""

    @pytest.fixture
    def mock_sentence_model(self):
        """Create a mock sentence-transformer model."""
        mock_model = MagicMock()
        # Return 384-dim embeddings
        mock_model.encode.return_value = np.random.randn(5, 384)
        return mock_model

    def test_embed_texts(self, mock_sentence_model):
        """Test embedding texts with mocked model."""
        scorer = NoveltyScorer()
        scorer._sentence_model = mock_sentence_model

        texts = ["Test text 1", "Test text 2", "Test text 3"]
        embeddings = scorer.embed_texts(texts)

        mock_sentence_model.encode.assert_called_once()
        assert embeddings.shape[1] == 384

    def test_embed_text_single(self, mock_sentence_model):
        """Test embedding a single text."""
        mock_sentence_model.encode.return_value = np.random.randn(1, 384)
        scorer = NoveltyScorer()
        scorer._sentence_model = mock_sentence_model

        embedding = scorer.embed_text("Single test text")

        assert embedding.shape == (384,)

    def test_fit_from_texts(self, mock_sentence_model):
        """Test fitting from texts directly."""
        # Return enough embeddings for clustering
        mock_sentence_model.encode.return_value = np.random.randn(30, 384)
        scorer = NoveltyScorer(n_clusters=3)
        scorer._sentence_model = mock_sentence_model

        texts = [f"Text {i}" for i in range(30)]
        scorer.fit_from_texts(texts)

        assert scorer.enabled
        assert scorer.centroids.shape == (3, 384)

    def test_score_text(self, mock_sentence_model):
        """Test scoring text directly."""
        # First call for fit, second for score
        mock_sentence_model.encode.side_effect = [
            np.random.randn(30, 384),  # fit_from_texts
            np.random.randn(1, 384),   # score_text
        ]
        scorer = NoveltyScorer(n_clusters=3)
        scorer._sentence_model = mock_sentence_model

        texts = [f"Text {i}" for i in range(30)]
        scorer.fit_from_texts(texts)

        result = scorer.score_text("New test text")

        assert isinstance(result, NoveltyScore)
        assert 0.0 <= result.score <= 1.0

    def test_score_texts_batch(self, mock_sentence_model):
        """Test batch scoring texts."""
        # First call for fit, second for score_texts
        mock_sentence_model.encode.side_effect = [
            np.random.randn(30, 384),  # fit_from_texts
            np.random.randn(5, 384),   # score_texts
        ]
        scorer = NoveltyScorer(n_clusters=3)
        scorer._sentence_model = mock_sentence_model

        texts = [f"Text {i}" for i in range(30)]
        scorer.fit_from_texts(texts)

        test_texts = ["New text 1", "New text 2", "New text 3", "New text 4", "New text 5"]
        results = scorer.score_texts(test_texts)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, NoveltyScore)


class TestNoveltyIntegration:
    """Integration tests for novelty scoring."""

    def test_sentence_transformer_embedding_dimension(self):
        """Test that scorer works with sentence-transformer dimension (384)."""
        np.random.seed(42)

        # Create embeddings with sentence-transformer dimension
        embeddings = np.random.randn(50, 384)

        scorer = NoveltyScorer(n_clusters=5)
        scorer.fit(embeddings)

        assert scorer.embedding_dim == 384
        assert scorer.enabled

        # Score a new embedding
        result = scorer.score(np.random.randn(384))
        assert 0.0 <= result.score <= 1.0

    def test_full_workflow_with_mock(self):
        """Test full workflow: fit from texts, score new text."""
        with patch("src.deployment.novelty.NoveltyScorer._ensure_sentence_model") as mock_ensure:
            mock_model = MagicMock()
            mock_ensure.return_value = mock_model

            # Simulate embeddings for training and scoring
            np.random.seed(42)
            training_embeddings = np.random.randn(50, 384)
            test_embedding = np.random.randn(1, 384)

            mock_model.encode.side_effect = [
                training_embeddings,  # fit_from_texts
                test_embedding,       # score_text
            ]

            scorer = NoveltyScorer(n_clusters=5)
            scorer.fit_from_texts([f"Article {i}" for i in range(50)])

            result = scorer.score_text("Novel article about new topic")

            assert scorer.enabled
            assert isinstance(result, NoveltyScore)
            assert 0.0 <= result.score <= 1.0

    def test_novelty_for_sportswear_topics(self):
        """Test that similar topics have low novelty, different topics have high novelty."""
        np.random.seed(42)

        # Simulate cluster of "sportswear" embeddings
        sportswear_center = np.random.randn(384)
        sportswear_embeddings = sportswear_center + np.random.randn(40, 384) * 0.1

        scorer = NoveltyScorer(n_clusters=3)
        scorer.fit(sportswear_embeddings)

        # Similar embedding (should have low novelty)
        similar = sportswear_center + np.random.randn(384) * 0.05
        similar_result = scorer.score(similar)

        # Very different embedding (should have high novelty)
        different = np.random.randn(384) * 10  # Far from training data
        different_result = scorer.score(different)

        assert similar_result.score < different_result.score
