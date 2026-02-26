"""Tests for the labeling pipeline orchestration."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.labeling.labeler import LabelingResult
from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse
from src.labeling.pipeline import (
    TRANSIENT_ERROR_TYPES,
    LabelingPipeline,
    LabelingStats,
)


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


class TestTransientErrorHandling:
    """Tests that transient API errors keep articles pending instead of marking failed."""

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

    def test_transient_error_types_defined(self):
        """TRANSIENT_ERROR_TYPES should include all API/network error categories."""
        assert "server_error" in TRANSIENT_ERROR_TYPES
        assert "timeout" in TRANSIENT_ERROR_TYPES
        assert "connection" in TRANSIENT_ERROR_TYPES
        assert "rate_limit" in TRANSIENT_ERROR_TYPES

    def test_permanent_errors_not_transient(self):
        """Permanent errors should not be in TRANSIENT_ERROR_TYPES."""
        assert "authentication" not in TRANSIENT_ERROR_TYPES
        assert "api_error" not in TRANSIENT_ERROR_TYPES
        assert "unknown" not in TRANSIENT_ERROR_TYPES

    @pytest.mark.parametrize(
        "error_type",
        ["server_error", "timeout", "connection", "rate_limit"],
    )
    def test_transient_error_does_not_mark_failed(self, error_type):
        """Articles should stay pending when LLM returns a transient error."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        mock_label_result = LabelingResult(
            success=False,
            error=f"Simulated {error_type} error",
            error_type=error_type,
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=False, skip_chunking=True, skip_embedding=True
        )

        assert result["error_type"] == error_type
        assert result["labeled"] is False
        # The key assertion: no status update call with 'failed'
        for call in mock_database.update_article_labeling_status.call_args_list:
            assert call[0][2] != "failed", (
                f"Article should not be marked 'failed' for transient error {error_type}"
            )

    @pytest.mark.parametrize(
        "error_type",
        ["authentication", "api_error", "unknown"],
    )
    def test_permanent_error_marks_failed(self, error_type):
        """Articles should be marked failed for permanent errors."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        mock_label_result = LabelingResult(
            success=False,
            error=f"Simulated {error_type} error",
            error_type=error_type,
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=False, skip_chunking=True, skip_embedding=True
        )

        assert result["error_type"] == error_type
        assert result["labeled"] is False
        # Should have a call that sets status to 'failed'
        failed_calls = [
            call for call in mock_database.update_article_labeling_status.call_args_list
            if call[0][2] == "failed"
        ]
        assert len(failed_calls) == 1, (
            f"Expected exactly one 'failed' status update for {error_type}"
        )

    def test_server_error_500_keeps_pending(self):
        """Simulates the Feb 25 outage scenario: Claude API returns 500s."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        mock_label_result = LabelingResult(
            success=False,
            error="Anthropic server error (500): Internal Server Error",
            error_type="server_error",
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=False, skip_chunking=True, skip_embedding=True
        )

        assert result["error"] == "Anthropic server error (500): Internal Server Error"
        assert result["error_type"] == "server_error"
        # Article stays pending — no 'failed' status update
        for call in mock_database.update_article_labeling_status.call_args_list:
            assert call[0][2] != "failed", "Article should not be marked failed for server error"

    def test_transient_error_dry_run(self):
        """Transient errors in dry run should also not touch database."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        mock_label_result = LabelingResult(
            success=False,
            error="API timeout",
            error_type="timeout",
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(article, dry_run=True)

        assert result["error_type"] == "timeout"
        mock_database.update_article_labeling_status.assert_not_called()

    def test_permanent_error_dry_run_no_db_update(self):
        """Even permanent errors should not update DB in dry run."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        mock_label_result = LabelingResult(
            success=False,
            error="Auth failed",
            error_type="authentication",
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(article, dry_run=True)

        assert result["error_type"] == "authentication"
        mock_database.update_article_labeling_status.assert_not_called()

    def test_multiple_transient_errors_all_stay_pending(self):
        """When all articles hit transient errors, none should be marked failed."""
        mock_database = MagicMock()
        mock_labeler = MagicMock()

        # Simulate API outage - every call returns server_error
        mock_label_result = LabelingResult(
            success=False,
            error="Anthropic server error (500): Internal Server Error",
            error_type="server_error",
        )
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        # Process 5 articles (simulating a batch during an outage)
        for _ in range(5):
            article = self.create_mock_article()
            result = pipeline._process_article(
                article, dry_run=False, skip_chunking=True, skip_embedding=True
            )
            assert result["error_type"] == "server_error"

        # None should have been marked failed
        for call in mock_database.update_article_labeling_status.call_args_list:
            assert call[0][2] != "failed", (
                "No articles should be marked failed during an API outage"
            )


class TestEmptyBrandLabelFiltering:
    """Tests for filtering brands with no applicable ESG categories."""

    def create_mock_article(self, **kwargs):
        """Helper to create mock article dict."""
        return {
            "id": kwargs.get("id", uuid4()),
            "title": kwargs.get("title", "Test Article"),
            "full_content": kwargs.get("full_content", "Test content " * 50),
            "description": kwargs.get("description", "Test description"),
            "brands_mentioned": kwargs.get("brands_mentioned", ["Nike", "Anta"]),
            "published_at": kwargs.get("published_at", datetime.now()),
            "source_name": kwargs.get("source_name", "Test Source"),
        }

    def _make_brand_analysis(self, brand, has_esg=True, is_sportswear=True):
        """Create a BrandAnalysis with or without applicable ESG categories."""
        if has_esg:
            categories = {
                "environmental": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "social": CategoryLabel(applies=True, sentiment=1, evidence=["Evidence"]),
                "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "digital_transformation": CategoryLabel(
                    applies=False, sentiment=None, evidence=[]
                ),
            }
        else:
            categories = {
                "environmental": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "digital_transformation": CategoryLabel(
                    applies=False, sentiment=None, evidence=[]
                ),
            }
        return BrandAnalysis(
            brand=brand,
            is_sportswear_brand=is_sportswear,
            not_sportswear_reason=None if is_sportswear else "Not sportswear",
            categories=categories if is_sportswear else {},
            confidence=0.9,
            reasoning=f"Test reasoning for {brand}",
        )

    def test_multi_brand_filters_empty_labels_in_pipeline(self):
        """Multi-brand article should only save brands with ESG content."""
        mock_database = MagicMock()
        mock_database.db.get_session.return_value.__enter__ = MagicMock()
        mock_database.db.get_session.return_value.__exit__ = MagicMock()
        mock_database.save_brand_labels.return_value = [MagicMock()]
        mock_labeler = MagicMock()

        # Anta has ESG content, Nike does not (all categories false)
        response = LabelingResponse(
            brand_analyses=[
                self._make_brand_analysis("Anta", has_esg=True),
                self._make_brand_analysis("Nike", has_esg=False),
            ],
            article_summary="Test summary",
        )

        mock_label_result = MagicMock()
        mock_label_result.success = True
        mock_label_result.response = response
        mock_label_result.input_tokens = 100
        mock_label_result.output_tokens = 50
        mock_label_result.model = "test-model"
        mock_label_result.prompt_version = "v1.8.0"
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=True, skip_chunking=True, skip_embedding=True
        )

        # Article should still be labeled (Anta has ESG content)
        assert result["labeled"] is True

    def test_all_brands_empty_labels_skips_article(self):
        """Article where all sportswear brands have no ESG content should be skipped."""
        mock_database = MagicMock()
        mock_database.db.get_session.return_value.__enter__ = MagicMock()
        mock_database.db.get_session.return_value.__exit__ = MagicMock()
        mock_labeler = MagicMock()

        # Both brands have all-false categories
        response = LabelingResponse(
            brand_analyses=[
                self._make_brand_analysis("Anta", has_esg=False),
                self._make_brand_analysis("Nike", has_esg=False),
            ],
            article_summary="Test summary",
        )

        mock_label_result = MagicMock()
        mock_label_result.success = True
        mock_label_result.response = response
        mock_label_result.input_tokens = 100
        mock_label_result.output_tokens = 50
        mock_label_result.model = "test-model"
        mock_labeler.label_article.return_value = mock_label_result

        pipeline = LabelingPipeline(
            database=mock_database, labeler=mock_labeler
        )

        article = self.create_mock_article()
        result = pipeline._process_article(
            article, dry_run=False, skip_chunking=True, skip_embedding=True
        )

        # Should be skipped since no brand has ESG content
        assert result["skipped"] is True
        assert result["labeled"] is False

    def test_save_brand_labels_skips_empty_categories(self):
        """save_brand_labels should not save brands with all categories false."""
        from src.labeling.database import LabelingDatabase

        mock_db = MagicMock()
        mock_session = MagicMock()

        labeling_db = LabelingDatabase(database=mock_db)

        brand_analyses = [
            self._make_brand_analysis("Anta", has_esg=True),
            self._make_brand_analysis("Nike", has_esg=False),
        ]

        labels = labeling_db.save_brand_labels(
            mock_session,
            uuid4(),
            brand_analyses,
            model_version="test",
            prompt_version="v1.8.0",
        )

        # Only Anta should be saved (1 label), Nike should be skipped
        assert len(labels) == 1
        assert labels[0].brand == "Anta"

    def test_save_brand_labels_saves_all_with_esg(self):
        """save_brand_labels should save all brands that have ESG content."""
        from src.labeling.database import LabelingDatabase

        mock_db = MagicMock()
        mock_session = MagicMock()

        labeling_db = LabelingDatabase(database=mock_db)

        brand_analyses = [
            self._make_brand_analysis("Anta", has_esg=True),
            self._make_brand_analysis("Nike", has_esg=True),
        ]

        labels = labeling_db.save_brand_labels(
            mock_session,
            uuid4(),
            brand_analyses,
            model_version="test",
            prompt_version="v1.8.0",
        )

        # Both should be saved
        assert len(labels) == 2
        brands_saved = {l.brand for l in labels}
        assert brands_saved == {"Anta", "Nike"}

    def test_save_brand_labels_skips_non_sportswear_and_empty(self):
        """save_brand_labels should skip both non-sportswear and empty-category brands."""
        from src.labeling.database import LabelingDatabase

        mock_db = MagicMock()
        mock_session = MagicMock()

        labeling_db = LabelingDatabase(database=mock_db)

        brand_analyses = [
            self._make_brand_analysis("Anta", has_esg=True),
            self._make_brand_analysis("Nike", has_esg=False),  # empty categories
            self._make_brand_analysis("Puma", is_sportswear=False),  # non-sportswear
        ]

        labels = labeling_db.save_brand_labels(
            mock_session,
            uuid4(),
            brand_analyses,
            model_version="test",
            prompt_version="v1.8.0",
        )

        # Only Anta should be saved
        assert len(labels) == 1
        assert labels[0].brand == "Anta"
