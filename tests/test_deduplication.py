"""Tests for article deduplication logic."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add scripts and src directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from export_website_feed import deduplicate_articles


class TestGreedyCosineClustering:
    """Tests for the greedy cosine clustering algorithm."""

    def test_empty_input(self):
        """Should handle empty embedding array."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        embeddings = np.array([]).reshape(0, 384)
        keep_indices, meta = greedy_cosine_clustering(embeddings, threshold=0.75)

        assert keep_indices == []
        assert meta["n_clusters"] == 0
        assert meta["duplicates_removed"] == 0

    def test_single_article(self):
        """Should keep single article."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        embeddings = np.random.randn(1, 384)
        keep_indices, meta = greedy_cosine_clustering(embeddings, threshold=0.75)

        assert keep_indices == [0]
        assert meta["n_clusters"] == 1
        assert meta["duplicates_removed"] == 0

    def test_identical_articles(self):
        """Should remove exact duplicates."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        # Two identical embeddings
        base_embedding = np.random.randn(1, 384)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize
        embeddings = np.vstack([base_embedding, base_embedding])

        keep_indices, meta = greedy_cosine_clustering(embeddings, threshold=0.75)

        assert len(keep_indices) == 1
        assert keep_indices == [0]  # First article kept
        assert meta["duplicates_removed"] == 1

    def test_distinct_articles(self):
        """Should keep all distinct articles."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        # Create orthogonal embeddings (similarity = 0)
        embeddings = np.eye(5, 384)

        keep_indices, meta = greedy_cosine_clustering(embeddings, threshold=0.75)

        assert len(keep_indices) == 5
        assert meta["duplicates_removed"] == 0

    def test_threshold_sensitivity(self):
        """Higher threshold should keep more articles."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        # Create embeddings with varying similarity
        np.random.seed(42)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)

        # Create embeddings at different similarity levels
        embeddings = []
        for noise_scale in [0, 0.1, 0.3, 0.5, 0.8]:
            noise = np.random.randn(384) * noise_scale
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        # Lower threshold should merge more
        _, meta_low = greedy_cosine_clustering(embeddings, threshold=0.5)
        _, meta_high = greedy_cosine_clustering(embeddings, threshold=0.95)

        assert meta_low["duplicates_removed"] >= meta_high["duplicates_removed"]

    def test_cluster_labels_assigned_correctly(self):
        """Each article should have a valid cluster label."""
        from src.dedup1_nb.clustering import greedy_cosine_clustering

        np.random.seed(42)
        embeddings = np.random.randn(10, 384)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        keep_indices, meta = greedy_cosine_clustering(embeddings, threshold=0.5)

        # All labels should be valid cluster IDs
        labels = meta["cluster_labels"]
        assert len(labels) == 10
        assert all(isinstance(l, int) for l in labels)
        assert all(l >= 0 for l in labels)


def _hdbscan_available():
    """Check if hdbscan is available."""
    try:
        import hdbscan
        return True
    except ImportError:
        return False


class TestHDBSCANClustering:
    """Tests for HDBSCAN clustering."""

    @pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
    def test_basic_clustering(self):
        """Should perform basic clustering."""
        from src.dedup1_nb.clustering import hdbscan_clustering

        # Create two distinct clusters
        np.random.seed(42)
        cluster1 = np.random.randn(5, 384) * 0.1
        cluster2 = np.random.randn(5, 384) * 0.1 + 10  # Shifted

        embeddings = np.vstack([cluster1, cluster2])

        keep_indices, meta = hdbscan_clustering(
            embeddings, min_cluster_size=2, min_samples=1, metric="euclidean"
        )

        # Should find some structure
        assert len(keep_indices) <= 10
        assert "n_clusters" in meta

    @pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
    def test_handles_small_dataset(self):
        """Should handle dataset smaller than min_cluster_size."""
        from src.dedup1_nb.clustering import hdbscan_clustering

        embeddings = np.random.randn(1, 384)

        keep_indices, meta = hdbscan_clustering(
            embeddings, min_cluster_size=5, min_samples=2
        )

        # Should keep all articles when can't form clusters
        assert len(keep_indices) == 1


class TestAgglomerativeClustering:
    """Tests for agglomerative clustering."""

    def test_basic_clustering(self):
        """Should perform hierarchical clustering."""
        from src.dedup1_nb.clustering import agglomerative_clustering

        np.random.seed(42)
        embeddings = np.random.randn(10, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        keep_indices, meta = agglomerative_clustering(
            embeddings, distance_threshold=0.5, linkage="average", metric="cosine"
        )

        assert len(keep_indices) > 0
        assert len(keep_indices) <= 10
        assert meta["method"] == "agglomerative"

    def test_distance_threshold_effect(self):
        """Lower distance threshold should create more clusters."""
        from src.dedup1_nb.clustering import agglomerative_clustering

        np.random.seed(42)
        embeddings = np.random.randn(20, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        _, meta_low = agglomerative_clustering(embeddings, distance_threshold=0.1)
        _, meta_high = agglomerative_clustering(embeddings, distance_threshold=0.5)

        # Lower threshold = stricter = more clusters = fewer duplicates removed
        assert meta_low["n_clusters"] >= meta_high["n_clusters"]


class TestClusterMetrics:
    """Tests for cluster quality metrics."""

    def test_silhouette_score_valid_range(self):
        """Silhouette score should be between -1 and 1."""
        from src.dedup1_nb.evaluation import calculate_cluster_metrics

        np.random.seed(42)
        embeddings = np.random.randn(20, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create some clusters
        labels = [i % 3 for i in range(20)]

        metrics = calculate_cluster_metrics(embeddings, labels)

        if metrics["silhouette_score"] is not None:
            assert -1 <= metrics["silhouette_score"] <= 1

    def test_single_cluster_returns_none(self):
        """Should return None for silhouette when only 1 cluster."""
        from src.dedup1_nb.evaluation import calculate_cluster_metrics

        embeddings = np.random.randn(10, 384)
        labels = [0] * 10  # All in same cluster

        metrics = calculate_cluster_metrics(embeddings, labels)

        assert metrics["silhouette_score"] is None
        assert metrics["n_clusters"] == 1

    def test_cluster_size_statistics(self):
        """Should compute correct cluster size statistics."""
        from src.dedup1_nb.evaluation import calculate_cluster_metrics

        embeddings = np.random.randn(10, 384)
        labels = [0, 0, 0, 1, 1, 2, 3, 3, 3, 3]  # Sizes: 3, 2, 1, 4

        metrics = calculate_cluster_metrics(embeddings, labels)

        assert metrics["n_clusters"] == 4
        assert metrics["max_cluster_size"] == 4
        assert metrics["min_cluster_size"] == 1
        assert metrics["singleton_clusters"] == 1


class TestLabeledPairEvaluation:
    """Tests for evaluation on labeled pairs."""

    def test_perfect_clustering(self):
        """Perfect clustering should have precision and recall of 1."""
        from src.dedup1_nb.evaluation import evaluate_on_labeled_pairs

        embeddings = np.random.randn(4, 384)
        article_ids = ["a1", "a2", "a3", "a4"]

        # Cluster labels: 0 and 1 are together, 2 and 3 are together
        cluster_labels = [0, 0, 1, 1]

        # Labeled pairs match the clustering
        labeled_pairs = [
            {"article_id_1": "a1", "article_id_2": "a2", "is_duplicate": True},
            {"article_id_1": "a3", "article_id_2": "a4", "is_duplicate": True},
            {"article_id_1": "a1", "article_id_2": "a3", "is_duplicate": False},
        ]

        metrics = evaluate_on_labeled_pairs(
            embeddings, article_ids, labeled_pairs, cluster_labels
        )

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_false_positives(self):
        """Clustering non-duplicates together should have zero precision."""
        from src.dedup1_nb.evaluation import evaluate_on_labeled_pairs

        embeddings = np.random.randn(4, 384)
        article_ids = ["a1", "a2", "a3", "a4"]

        # All in same cluster
        cluster_labels = [0, 0, 0, 0]

        # But none are actually duplicates
        labeled_pairs = [
            {"article_id_1": "a1", "article_id_2": "a2", "is_duplicate": False},
            {"article_id_1": "a1", "article_id_2": "a3", "is_duplicate": False},
        ]

        metrics = evaluate_on_labeled_pairs(
            embeddings, article_ids, labeled_pairs, cluster_labels
        )

        assert metrics["precision"] == 0.0
        assert metrics["false_positives"] == 2

    def test_skips_unlabeled_pairs(self):
        """Should skip pairs without labels."""
        from src.dedup1_nb.evaluation import evaluate_on_labeled_pairs

        embeddings = np.random.randn(4, 384)
        article_ids = ["a1", "a2", "a3", "a4"]
        cluster_labels = [0, 0, 1, 1]

        labeled_pairs = [
            {"article_id_1": "a1", "article_id_2": "a2", "is_duplicate": True},
            {"article_id_1": "a1", "article_id_2": "a3", "is_duplicate": None},  # Unlabeled
        ]

        metrics = evaluate_on_labeled_pairs(
            embeddings, article_ids, labeled_pairs, cluster_labels
        )

        assert metrics["evaluated_pairs"] == 1
        assert metrics["skipped_pairs"] == 1


class TestSentenceTransformerEmbedder:
    """Tests for the sentence transformer embedder."""

    def test_embedding_dimension(self):
        """Should produce correct embedding dimension."""
        from src.dedup1_nb.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        texts = ["This is a test sentence.", "Another test sentence."]

        embeddings = embedder.fit_transform(texts)

        assert embeddings.shape == (2, 384)
        assert embedder.embedding_dim == 384

    def test_normalized_embeddings(self):
        """Normalized embeddings should have unit norm."""
        from src.dedup1_nb.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2", normalize=True)
        texts = ["Test sentence one.", "Test sentence two."]

        embeddings = embedder.fit_transform(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0], decimal=5)


class TestTfidfLsaEmbedder:
    """Tests for the TF-IDF + LSA embedder."""

    def test_embedding_dimension(self):
        """Should produce correct embedding dimension."""
        from src.dedup1_nb.embedders import TfidfLsaEmbedder

        # Use enough texts with diverse vocabulary
        embedder = TfidfLsaEmbedder(n_components=10, max_features=5000)
        texts = [
            "Nike announces new sustainability initiative for running shoes and athletic apparel designed for professional athletes.",
            "Adidas launches eco-friendly athletic apparel collection with recycled materials from ocean plastic waste.",
            "Puma invests in renewable energy for manufacturing facilities worldwide to reduce carbon footprint.",
            "Under Armour commits to carbon neutrality goals and sustainable practices in all global operations.",
            "Lululemon expands yoga wear line with organic cotton materials sourced from certified farms.",
            "Patagonia leads outdoor industry in environmental responsibility and ethical labor practices.",
            "New Balance focuses on domestic manufacturing and quality control with American workers.",
            "ASICS develops innovative gel technology for marathon runners seeking performance and comfort.",
            "Reebok partners with environmental groups to create biodegradable sneakers for consumers.",
            "Skechers invests in research for sustainable foam materials in comfortable walking shoes.",
            "Columbia Sportswear introduces waterproof jackets made from recycled polyester materials.",
            "The North Face commits to using only recycled materials in outdoor gear by 2025.",
        ]

        embeddings = embedder.fit_transform(texts)

        assert embeddings.shape == (12, 10)
        assert embedder.embedding_dim == 10

    def test_requires_fit_before_transform(self):
        """Should raise error if transform called before fit."""
        from src.dedup1_nb.embedders import TfidfLsaEmbedder

        embedder = TfidfLsaEmbedder(n_components=50)

        with pytest.raises(RuntimeError, match="not fitted"):
            embedder.transform(["Test text"])


class TestDeduplicateArticles:
    """Tests for the production deduplicate_articles function."""

    def test_returns_tuple(self):
        """Should return (articles, count) tuple."""
        # Create mock articles
        articles = []
        for i in range(5):
            article = MagicMock()
            article.title = f"Article {i} about Nike"
            article.full_content = f"Content about Nike sustainability initiative {i}."
            article.description = None
            articles.append(article)

        result = deduplicate_articles(articles, similarity_threshold=0.99)

        assert isinstance(result, tuple)
        assert len(result) == 2
        deduplicated, removed_count = result
        assert isinstance(deduplicated, list)
        assert isinstance(removed_count, int)

    def test_empty_list(self):
        """Should handle empty list."""
        result = deduplicate_articles([], similarity_threshold=0.75)

        assert result == ([], 0)

    def test_single_article(self):
        """Should keep single article."""
        article = MagicMock()
        article.title = "Nike announces sustainability goals"
        article.full_content = "Full article content here."
        article.description = None

        result = deduplicate_articles([article], similarity_threshold=0.75)

        assert result == ([article], 0)

    def test_removes_duplicates(self):
        """Should remove duplicate articles."""
        # Create two articles with identical content
        article1 = MagicMock()
        article1.title = "Nike announces new sustainability initiative"
        article1.full_content = "Nike today announced a major new sustainability initiative focused on reducing carbon emissions."
        article1.description = None

        article2 = MagicMock()
        article2.title = "Nike announces new sustainability initiative"  # Same title
        article2.full_content = "Nike today announced a major new sustainability initiative focused on reducing carbon emissions."  # Same content
        article2.description = None

        deduplicated, removed = deduplicate_articles(
            [article1, article2], similarity_threshold=0.75
        )

        assert len(deduplicated) == 1
        assert removed == 1

    def test_keeps_distinct_articles(self):
        """Should keep distinct articles."""
        article1 = MagicMock()
        article1.title = "Nike launches new running shoe"
        article1.full_content = "Nike unveiled its latest running shoe technology with improved cushioning and support."
        article1.description = None

        article2 = MagicMock()
        article2.title = "Adidas announces sustainability goals"
        article2.full_content = "Adidas today committed to using 100% recycled materials in all products by 2030."
        article2.description = None

        deduplicated, removed = deduplicate_articles(
            [article1, article2], similarity_threshold=0.75
        )

        # Different topics should be kept
        assert len(deduplicated) == 2
        assert removed == 0


class TestThresholdSweep:
    """Tests for threshold sweep analysis."""

    def test_returns_results_for_all_thresholds(self):
        """Should return results for each threshold."""
        from src.dedup1_nb.evaluation import threshold_sweep

        # Create embeddings with some similarity structure
        np.random.seed(42)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)

        # Create 20 embeddings with varying similarity to base
        embeddings = []
        for i in range(20):
            noise = np.random.randn(384) * (0.1 + i * 0.05)
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        thresholds = [0.5, 0.7, 0.9]
        results = threshold_sweep(embeddings, thresholds=thresholds)

        assert len(results) == 3
        assert all(r["threshold"] in thresholds for r in results)

    def test_reduction_rate_increases_with_lower_threshold(self):
        """Lower threshold should remove more duplicates."""
        from src.dedup1_nb.evaluation import threshold_sweep

        # Create embeddings with clear similarity structure
        np.random.seed(42)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)

        # Create 20 embeddings with varying similarity
        embeddings = []
        for i in range(20):
            noise = np.random.randn(384) * (0.1 + i * 0.05)
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        results = threshold_sweep(embeddings, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])

        # Reduction rate should generally decrease with higher threshold
        reduction_rates = [r["reduction_rate"] for r in results]
        # Allow some noise but overall trend should be decreasing
        assert reduction_rates[0] >= reduction_rates[-1]


class TestValidationPairs:
    """Tests for validation pair creation and loading."""

    def test_create_validation_pairs(self):
        """Should create pairs across similarity ranges."""
        from src.dedup1_nb.evaluation import create_validation_pairs
        import pandas as pd

        np.random.seed(42)
        embeddings = np.random.randn(50, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create proper DataFrame
        df = pd.DataFrame({
            "article_id": [f"a{i}" for i in range(50)],
            "title": [f"Title {i}" for i in range(50)],
            "source_name": [f"source{i}" for i in range(50)],
        })

        pairs = create_validation_pairs(
            embeddings,
            df,
            similarity_ranges=[(0.0, 0.3), (0.3, 0.6)],  # Use low similarity ranges for random data
            n_pairs_per_range=5,
        )

        # Should create pairs (may be less if not enough pairs in range)
        assert len(pairs) <= 10
        assert all("is_duplicate" in p for p in pairs)
        assert all(p["is_duplicate"] is None for p in pairs)  # Not labeled yet

    def test_save_and_load_validation_pairs(self, tmp_path):
        """Should save and load validation pairs correctly."""
        from src.dedup1_nb.data_utils import save_validation_pairs, load_validation_pairs

        pairs = [
            {"article_id_1": "a1", "article_id_2": "a2", "is_duplicate": True},
            {"article_id_1": "a3", "article_id_2": "a4", "is_duplicate": False},
        ]

        file_path = tmp_path / "test_pairs.jsonl"
        save_validation_pairs(pairs, str(file_path))

        loaded = load_validation_pairs(str(file_path))

        assert len(loaded) == 2
        assert loaded[0]["is_duplicate"] is True
        assert loaded[1]["is_duplicate"] is False


class TestEmbedderFactory:
    """Tests for the embedder factory function."""

    def test_get_sentence_transformer(self):
        """Should return sentence transformer embedder."""
        from src.dedup1_nb.embedders import get_embedder

        embedder = get_embedder("sentence-transformer", model_name="all-MiniLM-L6-v2")

        assert embedder.name == "SentenceTransformer(all-MiniLM-L6-v2)"

    def test_get_tfidf_lsa(self):
        """Should return TF-IDF + LSA embedder."""
        from src.dedup1_nb.embedders import get_embedder

        embedder = get_embedder("tfidf-lsa", n_components=100)

        assert embedder.name == "TF-IDF+LSA(dim=100)"

    def test_unknown_embedder_raises(self):
        """Should raise for unknown embedder type."""
        from src.dedup1_nb.embedders import get_embedder

        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("unknown-embedder")


class TestClusteringFactory:
    """Tests for the clustering method factory function."""

    def test_get_greedy(self):
        """Should return greedy clustering function."""
        from src.dedup1_nb.clustering import get_clustering_method, greedy_cosine_clustering

        func = get_clustering_method("greedy")
        assert func == greedy_cosine_clustering

    def test_get_hdbscan(self):
        """Should return HDBSCAN clustering function."""
        from src.dedup1_nb.clustering import get_clustering_method, hdbscan_clustering

        func = get_clustering_method("hdbscan")
        assert func == hdbscan_clustering

    def test_unknown_method_raises(self):
        """Should raise for unknown clustering method."""
        from src.dedup1_nb.clustering import get_clustering_method

        with pytest.raises(ValueError, match="Unknown clustering method"):
            get_clustering_method("unknown-method")
