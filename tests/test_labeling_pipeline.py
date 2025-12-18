"""Tests for the labeling pipeline orchestration."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse
from src.labeling.pipeline import LabelingPipeline, LabelingStats


class TestLabelingStats:
    """Tests for LabelingStats dataclass."""

    def test_default_values(self):
        """Should have zero defaults."""
        stats = LabelingStats()
        assert stats.articles_processed == 0
        assert stats.articles_labeled == 0
        assert stats.articles_skipped == 0
        assert stats.articles_false_positive == 0
        assert stats.articles_failed == 0
        assert stats.brands_labeled == 0
        assert stats.false_positive_brands == 0
        assert stats.chunks_created == 0
        assert stats.embeddings_generated == 0
        assert stats.llm_calls == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.errors == []

    def test_errors_list_independent(self):
        """Errors list should be independent per instance."""
        stats1 = LabelingStats()
        stats2 = LabelingStats()
        stats1.errors.append("Error 1")

        assert len(stats1.errors) == 1
        assert len(stats2.errors) == 0


class TestLabelingPipelineInit:
    """Tests for LabelingPipeline initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default components."""
        with patch("src.labeling.pipeline.db"):
            pipeline = LabelingPipeline()
            assert pipeline.chunker is not None
            assert pipeline._embedder_initialized is False
            assert pipeline._labeler_initialized is False

    def test_init_with_custom_components(self):
        """Should use provided components."""
        mock_database = MagicMock()
        mock_chunker = MagicMock()
        mock_embedder = MagicMock()
        mock_labeler = MagicMock()

        pipeline = LabelingPipeline(
            database=mock_database,
            chunker=mock_chunker,
            embedder=mock_embedder,
            labeler=mock_labeler,
        )

        assert pipeline.database == mock_database
        assert pipeline.chunker == mock_chunker
        assert pipeline.embedder == mock_embedder
        assert pipeline.labeler == mock_labeler
        assert pipeline._embedder_initialized is True
        assert pipeline._labeler_initialized is True


class TestLabelingPipelineLazyInit:
    """Tests for lazy initialization of API clients."""

    def test_ensure_embedder_lazy_init(self):
        """Should lazily initialize embedder."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.OpenAIEmbedder") as mock_embedder_class:
                mock_embedder = MagicMock()
                mock_embedder_class.return_value = mock_embedder

                pipeline = LabelingPipeline()
                assert pipeline._embedder_initialized is False

                # Trigger lazy init
                embedder = pipeline._ensure_embedder()

                assert pipeline._embedder_initialized is True
                assert embedder == mock_embedder

    def test_ensure_labeler_lazy_init(self):
        """Should lazily initialize labeler."""
        with patch("src.labeling.pipeline.db"):
            with patch("src.labeling.pipeline.ArticleLabeler") as mock_labeler_class:
                mock_labeler = MagicMock()
                mock_labeler_class.return_value = mock_labeler

                pipeline = LabelingPipeline()
                assert pipeline._labeler_initialized is False

                # Trigger lazy init
                labeler = pipeline._ensure_labeler()

                assert pipeline._labeler_initialized is True
                assert labeler == mock_labeler


class TestLabelingPipelineProcessArticle:
    """Tests for _process_article method."""

    def create_mock_article(self, **kwargs):
        """Helper to create mock article dict."""
        return {
            "id": kwargs.get("id", uuid4()),
            "title": kwargs.get("title", "Test Article"),
            "full_content": kwargs.get("full_content", "Test content " * 50),
            "description": kwargs.get("description", "Test description"),
            "brands_mentioned": kwargs.get("brands_mentioned", ["Nike"]),
            "published_at": kwargs.get("published_at", datetime.now()),
            "source_name": kwargs.get("source_name", "Test Source"),
        }

    def create_mock_labeling_response(self, brands=None, is_sportswear=True):
        """Helper to create mock labeling response."""
        if brands is None:
            brands = ["Nike"]

        brand_analyses = []
        for brand in brands:
            brand_analyses.append(
                BrandAnalysis(
                    brand=brand,
                    is_sportswear_brand=is_sportswear,
                    not_sportswear_reason=None if is_sportswear else "Not sportswear",
                    categories={
                        "environmental": CategoryLabel(
                            applies=True, sentiment=1, evidence=["Test evidence"]
                        ),
                        "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                        "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                        "digital_transformation": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                    },
                    confidence=0.9,
                    reasoning="Test reasoning",
                )
            )

        return LabelingResponse(
            brand_analyses=brand_analyses,
            article_summary="Test summary",
        )

    def test_process_article_insufficient_content(self):
        """Should skip article with insufficient content."""
        mock_database = MagicMock()
        mock_database.db.get_session.return_value.__enter__ = MagicMock()
        mock_database.db.get_session.return_value.__exit__ = MagicMock()

        pipeline = LabelingPipeline(database=mock_database)

        article = self.create_mock_article(full_content="Short", description=None)
        result = pipeline._process_article(article, dry_run=True)

        assert result["skipped"] is True
        assert result["labeled"] is False

    def test_process_article_no_brands(self):
        """Should skip article with no brands mentioned."""
        mock_database = MagicMock()
        mock_database.db.get_session.return_value.__enter__ = MagicMock()
        mock_database.db.get_session.return_value.__exit__ = MagicMock()

        pipeline = LabelingPipeline(database=mock_database)

        article = self.create_mock_article(brands_mentioned=[])
        result = pipeline._process_article(article, dry_run=True)

        assert result["skipped"] is True
        assert result["labeled"] is False

    def test_process_article_dry_run_no_save(self):
        """Should not save to database in dry run mode."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        # Mock successful labeling
        mock_label_result = MagicMock()
        mock_label_result.success = True
        mock_label_result.response = self.create_mock_labeling_response()
        mock_label_result.input_tokens = 100
        mock_label_result.output_tokens = 50
        mock_label_result.model = "test-model"
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=True, skip_chunking=True, skip_embedding=True
        )

        assert result["labeled"] is True
        assert result["llm_calls"] == 1
        # In dry run, database save methods should not be called
        mock_database.save_brand_labels.assert_not_called()

    def test_process_article_false_positive_detection(self):
        """Should detect false positive brands."""
        mock_database = MagicMock()
        mock_database.db.get_session.return_value.__enter__ = MagicMock()
        mock_database.db.get_session.return_value.__exit__ = MagicMock()
        mock_labeler = MagicMock()

        # Mock labeling response with non-sportswear brand
        mock_label_result = MagicMock()
        mock_label_result.success = True
        mock_label_result.response = self.create_mock_labeling_response(
            brands=["Puma"], is_sportswear=False
        )
        mock_label_result.input_tokens = 100
        mock_label_result.output_tokens = 50
        mock_label_result.model = "test-model"
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article(brands_mentioned=["Puma"])
        result = pipeline._process_article(
            article, dry_run=True, skip_chunking=True, skip_embedding=True
        )

        assert result["false_positive"] is True
        assert result["false_positive_brands"] == 1
        assert result["labeled"] is False


class TestLabelingPipelineGetStats:
    """Tests for pipeline statistics."""

    def test_get_stats_empty(self):
        """Should return stats from components."""
        mock_database = MagicMock()
        mock_database.get_labeling_stats.return_value = {
            "total_articles": 100,
            "labeled": 50,
        }
        mock_database.db.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_database.db.get_session.return_value.__exit__ = MagicMock()

        pipeline = LabelingPipeline(database=mock_database)
        stats = pipeline.get_stats()

        assert "database" in stats
        assert stats["database"]["total_articles"] == 100

    def test_get_stats_with_labeler(self):
        """Should include labeler stats when initialized."""
        mock_database = MagicMock()
        mock_database.get_labeling_stats.return_value = {}
        mock_database.db.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_database.db.get_session.return_value.__exit__ = MagicMock()

        mock_labeler = MagicMock()
        mock_labeler.get_stats.return_value = {
            "total_api_calls": 10,
            "total_input_tokens": 1000,
        }

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )
        stats = pipeline.get_stats()

        assert "labeler" in stats
        assert stats["labeler"]["total_api_calls"] == 10

    def test_get_stats_with_embedder(self):
        """Should include embedder stats when initialized."""
        mock_database = MagicMock()
        mock_database.get_labeling_stats.return_value = {}
        mock_database.db.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_database.db.get_session.return_value.__exit__ = MagicMock()

        mock_embedder = MagicMock()
        mock_embedder.get_stats.return_value = {
            "total_tokens": 500,
        }

        pipeline = LabelingPipeline(
            database=mock_database, embedder=mock_embedder
        )
        stats = pipeline.get_stats()

        assert "embedder" in stats
        assert stats["embedder"]["total_tokens"] == 500
