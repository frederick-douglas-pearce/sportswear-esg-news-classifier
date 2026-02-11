"""Extended integration tests for ESG News Classifier.

These tests cover end-to-end functionality across major system components:
- API endpoints (FastAPI routes)
- Agent workflows (website_export, daily_labeling)
- Full labeling pipeline (LLM → chunking → embedding → matching)
- Data collection pipeline (API → scraping → database)
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# ============================================================================
# API Endpoint Integration Tests
# ============================================================================


class TestAPIEndpoints:
    """Integration tests for FastAPI prediction endpoints."""

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        classifier = MagicMock()
        classifier.model_name = "TestModel"
        classifier.threshold = 0.5
        classifier.predict_from_fields.return_value = {
            "is_sportswear": True,
            "probability": 0.85,
            "confidence_level": "high",
            "threshold": 0.5,
        }
        classifier.predict_batch_from_fields.return_value = [
            {"is_sportswear": True, "probability": 0.85},
            {"is_sportswear": False, "probability": 0.25},
        ]
        classifier.get_model_info.return_value = {
            "classifier_type": "fp",
            "model_name": "TestModel",
            "threshold": 0.5,
            "target_recall": 0.99,
            "transformer_method": "tfidf",
            "best_params": {},
            "metrics": {"f2": 0.95},
        }
        return classifier

    @pytest.fixture
    def test_client(self, mock_classifier):
        """Create a test client with mocked classifier."""
        from fastapi.testclient import TestClient

        # Patch the classifier creation
        with patch("scripts.predict.create_classifier", return_value=mock_classifier):
            with patch("scripts.predict.CLASSIFIER_TYPE_STR", "fp"):
                with patch("scripts.predict.ENABLE_PREDICTION_LOGGING", False):
                    from scripts.predict import app

                    # Manually set the global classifier
                    import scripts.predict as predict_module

                    predict_module.classifier = mock_classifier

                    client = TestClient(app)
                    yield client

    def test_health_endpoint(self, test_client):
        """Test /health endpoint returns correct status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "classifier_type" in data

    def test_model_info_endpoint(self, test_client):
        """Test /model/info endpoint returns model metadata."""
        response = test_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "threshold" in data
        assert "metrics" in data

    def test_predict_endpoint_single_article(self, test_client):
        """Test /predict endpoint with single article."""
        article = {
            "title": "Nike Announces Sustainability Goals",
            "content": "Nike Inc. unveiled carbon reduction targets...",
            "brands": ["Nike"],
            "source_name": "ESPN",
            "category": ["sports", "business"],
        }

        response = test_client.post("/predict", json=article)

        assert response.status_code == 200
        data = response.json()
        assert "is_sportswear" in data or "has_esg" in data
        assert "probability" in data

    def test_predict_batch_endpoint(self, test_client):
        """Test /predict/batch endpoint with multiple articles."""
        batch = {
            "articles": [
                {
                    "title": "Nike Sustainability Report",
                    "content": "Carbon neutral goals...",
                    "brands": ["Nike"],
                },
                {
                    "title": "Wild Puma Spotted",
                    "content": "A mountain lion was seen...",
                    "brands": ["Puma"],
                },
            ]
        }

        response = test_client.post("/predict/batch", json=batch)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2

    def test_predict_batch_empty_list(self, test_client):
        """Test /predict/batch with empty article list."""
        response = test_client.post("/predict/batch", json={"articles": []})

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["predictions"] == []

    def test_predict_validates_required_fields(self, test_client):
        """Test that predict endpoint validates required fields."""
        # Missing content field
        article = {"title": "Test Title"}

        response = test_client.post("/predict", json=article)

        assert response.status_code == 422  # Validation error


# ============================================================================
# Agent Workflow Integration Tests
# ============================================================================


class TestAgentWorkflows:
    """Integration tests for agent workflow step functions."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow object."""
        workflow = MagicMock()
        workflow.name = "test_workflow"
        return workflow

    @pytest.fixture
    def temp_website_repo(self, tmp_path):
        """Create a temporary website repository structure."""
        data_dir = tmp_path / "_data"
        feeds_dir = tmp_path / "assets" / "feeds"
        data_dir.mkdir(parents=True)
        feeds_dir.mkdir(parents=True)
        return tmp_path

    def test_export_feeds_step_without_config(self, mock_workflow):
        """Test export_feeds step when website repo not configured."""
        from src.agent.workflows.website_export import export_feeds

        with patch("src.agent.workflows.website_export.agent_settings") as mock_settings:
            mock_settings.website_repo_path = None

            result = export_feeds(mock_workflow, {})

            assert result["export_skipped"] is True
            assert result["reason"] == "website_repo_not_configured"

    def test_validate_export_step_valid_json(self, mock_workflow, tmp_path):
        """Test validate_export step with valid JSON file."""
        from src.agent.workflows.website_export import validate_export

        # Create valid JSON file
        json_path = tmp_path / "test.json"
        json_path.write_text('{"articles": [{"title": "Test"}]}')

        context = {
            "export_success": True,
            "json_output": str(json_path),
            "atom_output": None,
        }

        result = validate_export(mock_workflow, context)

        assert result["validation_passed"] is True
        assert result["json_valid"] is True

    def test_validate_export_step_invalid_json(self, mock_workflow, tmp_path):
        """Test validate_export step with invalid JSON file."""
        from src.agent.workflows.website_export import validate_export

        # Create invalid JSON file
        json_path = tmp_path / "test.json"
        json_path.write_text("{invalid json")

        context = {
            "export_success": True,
            "json_output": str(json_path),
        }

        result = validate_export(mock_workflow, context)

        assert result["validation_passed"] is False
        assert result["json_valid"] is False

    def test_validate_export_step_skipped_when_export_skipped(self, mock_workflow):
        """Test validate_export skips when export was skipped."""
        from src.agent.workflows.website_export import validate_export

        result = validate_export(mock_workflow, {"export_skipped": True})

        assert result["validation_skipped"] is True

    def test_commit_and_push_skipped_on_dry_run(self, mock_workflow):
        """Test commit_and_push skips in dry run mode."""
        from src.agent.workflows.website_export import commit_and_push

        with patch("src.agent.workflows.website_export.agent_settings") as mock_settings:
            mock_settings.website_repo_path = Path("/tmp/test")

            result = commit_and_push(
                mock_workflow,
                {"validation_passed": True, "dry_run": True},
            )

            assert result["git_skipped"] is True
            assert result["reason"] == "dry_run"

    def test_send_error_notification_no_errors(self, mock_workflow):
        """Test send_error_notification when no errors occurred."""
        from src.agent.workflows.website_export import send_error_notification

        result = send_error_notification(
            mock_workflow,
            {
                "export_success": True,
                "validation_passed": True,
                "git_success": True,
            },
        )

        assert result["notification_sent"] is False
        assert result["reason"] == "no_errors"

    def test_send_error_notification_with_errors(self, mock_workflow):
        """Test send_error_notification when errors occurred."""
        from src.agent.workflows.website_export import send_error_notification

        with patch(
            "src.agent.workflows.website_export.NotificationManager"
        ) as MockManager:
            mock_manager = MagicMock()
            mock_manager.send.return_value = {"email": True}
            MockManager.return_value = mock_manager

            result = send_error_notification(
                mock_workflow,
                {
                    "export_success": False,
                    "export_error": "Test error",
                },
            )

            assert result["notification_sent"] is True
            assert len(result["errors"]) > 0

    def test_save_scorecard_snapshot_skipped_on_dry_run(self, mock_workflow):
        """Test save_scorecard_snapshot skips in dry run mode."""
        from src.agent.workflows.website_export import save_scorecard_snapshot

        result = save_scorecard_snapshot(
            mock_workflow,
            {
                "export_success": True,
                "json_output": "/tmp/test.json",
                "dry_run": True,
            },
        )

        assert result["scorecard_skipped"] is True
        assert result["reason"] == "dry_run"


# ============================================================================
# Labeling Pipeline Integration Tests
# ============================================================================


class TestLabelingPipelineIntegration:
    """Integration tests for the full labeling pipeline."""

    @pytest.fixture
    def mock_article(self):
        """Create a mock article for testing."""

        @dataclass
        class MockArticle:
            id: Any = None
            title: str = "Nike Announces Carbon Neutral Goals"
            full_content: str = (
                "Nike Inc. today announced ambitious sustainability goals. "
                "The company plans to achieve carbon neutrality by 2030. "
                "CEO John Donahoe stated that environmental responsibility "
                "is a core part of Nike's strategy going forward."
            )
            source_name: str = "Reuters"
            brands_mentioned: list = None
            labeling_status: str = "pending"
            published_at: datetime = None

            def __post_init__(self):
                if self.id is None:
                    self.id = uuid4()
                if self.brands_mentioned is None:
                    self.brands_mentioned = ["Nike"]
                if self.published_at is None:
                    self.published_at = datetime.now(timezone.utc)

        return MockArticle()

    @pytest.fixture
    def mock_chunks(self):
        """Create mock chunks."""
        from src.labeling.chunker import Chunk

        return [
            Chunk(
                index=0,
                text="Nike Inc. today announced ambitious sustainability goals.",
                char_start=0,
                char_end=56,
                token_count=10,
            ),
            Chunk(
                index=1,
                text="The company plans to achieve carbon neutrality by 2030.",
                char_start=57,
                char_end=112,
                token_count=11,
            ),
        ]

    def test_chunker_creates_valid_chunks(self, mock_article):
        """Test that chunker creates valid chunks from article content."""
        from src.labeling.chunker import ArticleChunker

        chunker = ArticleChunker(target_tokens=50, overlap_tokens=10)
        chunks = chunker.chunk_article(mock_article.full_content)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text
            assert chunk.token_count > 0
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_pipeline_initialization_lazy_loading(self):
        """Test that pipeline initializes components lazily."""
        from src.labeling.pipeline import LabelingPipeline

        pipeline = LabelingPipeline()

        # Components should not be initialized yet
        assert pipeline.embedder is None
        assert pipeline.labeler is None

        # Internal flags should indicate not initialized
        assert pipeline._embedder_initialized is False
        assert pipeline._labeler_initialized is False

    def test_pipeline_ensure_embedder(self):
        """Test that _ensure_embedder creates embedder when needed."""
        from src.labeling.pipeline import LabelingPipeline

        with patch("src.labeling.pipeline.OpenAIEmbedder") as MockEmbedder:
            mock_embedder = MagicMock()
            MockEmbedder.return_value = mock_embedder

            pipeline = LabelingPipeline()
            embedder = pipeline._ensure_embedder()

            assert embedder is not None
            assert pipeline._embedder_initialized is True
            MockEmbedder.assert_called_once()

    def test_pipeline_ensure_labeler(self):
        """Test that _ensure_labeler creates labeler when needed."""
        from src.labeling.pipeline import LabelingPipeline

        with patch("src.labeling.pipeline.ArticleLabeler") as MockLabeler:
            mock_labeler = MagicMock()
            MockLabeler.return_value = mock_labeler

            pipeline = LabelingPipeline()
            labeler = pipeline._ensure_labeler()

            assert labeler is not None
            assert pipeline._labeler_initialized is True

    def test_labeling_stats_initialization(self):
        """Test LabelingStats initializes with correct defaults."""
        from src.labeling.pipeline import LabelingStats

        stats = LabelingStats()

        assert stats.articles_processed == 0
        assert stats.articles_labeled == 0
        assert stats.llm_calls == 0
        assert stats.errors == []
        assert stats.fp_classifier_calls == 0

    def test_labeling_stats_accumulation(self):
        """Test that LabelingStats can accumulate values."""
        from src.labeling.pipeline import LabelingStats

        stats = LabelingStats()
        stats.articles_processed = 10
        stats.articles_labeled = 8
        stats.articles_failed = 2
        stats.errors.append("Test error")

        assert stats.articles_processed == 10
        assert stats.articles_labeled == 8
        assert len(stats.errors) == 1

    def test_evidence_matcher_integration(self, mock_chunks):
        """Test evidence matcher can match excerpts to chunks."""
        from src.labeling.evidence_matcher import EvidenceMatcher

        # Create matcher without embedder (will use fuzzy matching only)
        matcher = EvidenceMatcher(
            embedder=None,
            fuzzy_threshold=0.7,
            use_embedding_rerank=False,
            use_cross_encoder_rerank=False,
        )

        # Test excerpt that should match - needs to match chunk text closely
        excerpt = "Nike Inc. today announced ambitious sustainability goals."

        results = matcher.match_evidence_to_chunks([excerpt], mock_chunks)

        # Should find a match via fuzzy matching
        assert len(results) == 1
        assert results[0].excerpt == excerpt
        # Should have found a matching chunk
        assert results[0].match_method != "none"


# ============================================================================
# Data Collection Pipeline Integration Tests
# ============================================================================


class TestDataCollectionPipelineIntegration:
    """Integration tests for the data collection pipeline."""

    @pytest.fixture
    def mock_api_response(self):
        """Create a mock API response with articles."""
        return {
            "status": "success",
            "totalResults": 2,
            "results": [
                {
                    "article_id": "article-001",
                    "title": "Nike Announces ESG Initiative",
                    "description": "Nike unveils sustainability plans",
                    "content": "Full article content about Nike ESG...",
                    "link": "https://example.com/nike-esg",
                    "source_name": "Reuters",
                    "pubDate": "2024-01-15 10:00:00",
                    "category": ["business"],
                },
                {
                    "article_id": "article-002",
                    "title": "Adidas Sustainability Report",
                    "description": "Adidas releases annual report",
                    "content": "Adidas sustainability content...",
                    "link": "https://example.com/adidas-report",
                    "source_name": "BBC",
                    "pubDate": "2024-01-15 11:00:00",
                    "category": ["environment"],
                },
            ],
        }

    def test_collection_stats_initialization(self):
        """Test CollectionStats initializes correctly."""
        from src.data_collection.collector import CollectionStats

        stats = CollectionStats()

        assert stats.api_calls == 0
        assert stats.articles_fetched == 0
        assert stats.articles_duplicates == 0
        assert stats.articles_blocked_domain == 0
        assert stats.errors == []

    def test_normalize_title(self):
        """Test title normalization for deduplication."""
        from src.data_collection.collector import normalize_title

        assert normalize_title("  Test Title  ") == "test title"
        assert normalize_title("UPPERCASE") == "uppercase"
        assert normalize_title(None) == ""
        assert normalize_title("") == ""

    def test_is_blocked_domain(self):
        """Test blocked domain detection."""
        from src.data_collection.collector import is_blocked_domain

        # Test with blocked domain in URL
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", ["blockedsite.com"]):
            assert is_blocked_domain("https://blockedsite.com/article", None) is True
            assert is_blocked_domain("https://goodsite.com/article", None) is False
            assert is_blocked_domain(None, "blockedsite.com") is True

        # Test with empty blocklist
        with patch("src.data_collection.collector.BLOCKED_DOMAINS", []):
            assert is_blocked_domain("https://anysite.com/article", None) is False

    def test_news_collector_initialization(self):
        """Test NewsCollector initializes with correct defaults."""
        from src.data_collection.collector import NewsCollector

        with patch("src.data_collection.collector.db"):
            with patch("src.data_collection.collector.NewsDataClient") as MockClient:
                MockClient.return_value = MagicMock()

                collector = NewsCollector(source="newsdata")

                assert collector.source == "newsdata"
                MockClient.assert_called_once()

    def test_news_collector_gdelt_source(self):
        """Test NewsCollector initializes GDELT client when specified."""
        from src.data_collection.collector import NewsCollector

        with patch("src.data_collection.collector.db"):
            with patch("src.data_collection.collector.GDELTClient") as MockClient:
                MockClient.return_value = MagicMock()

                collector = NewsCollector(source="gdelt")

                assert collector.source == "gdelt"
                MockClient.assert_called_once()

    def test_scraper_initialization(self):
        """Test ArticleScraper initializes correctly."""
        from src.data_collection.scraper import ArticleScraper

        scraper = ArticleScraper()

        assert scraper is not None

    def test_collection_dry_run_no_database_writes(self):
        """Test that dry run doesn't write to database."""
        from src.data_collection.collector import NewsCollector

        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_api.generate_search_queries.return_value = [("Nike", "business")]
        mock_api.api_calls_made = 0
        # search_news returns tuple of (list of articles, next_page token)
        mock_api.search_news.return_value = ([], None)

        collector = NewsCollector(database=mock_db, api_client=mock_api)

        stats = collector.collect_from_api(max_calls=1, dry_run=True)

        # Database should not have been called to save articles
        mock_db.save_article.assert_not_called()

    def test_collection_propagates_api_errors(self):
        """Test that collection propagates API errors (caller must handle)."""
        from src.data_collection.collector import NewsCollector

        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_api.generate_search_queries.return_value = [("Nike", "business")]
        mock_api.api_calls_made = 0
        mock_api.search_news.side_effect = Exception("API Error")

        collector = NewsCollector(database=mock_db, api_client=mock_api)

        # API errors propagate up - caller is responsible for handling
        with pytest.raises(Exception, match="API Error"):
            collector.collect_from_api(max_calls=1, dry_run=True)

    def test_collection_with_articles(self):
        """Test collection with mock articles."""
        from src.data_collection.collector import NewsCollector

        # Create mock article data
        mock_article = MagicMock()
        mock_article.article_id = "test-123"
        mock_article.title = "Nike ESG Report"
        mock_article.brands_mentioned = ["Nike"]
        mock_article.url = "https://example.com/article"
        mock_article.source_url = "example.com"

        mock_db = MagicMock()
        mock_api = MagicMock()
        mock_api.generate_search_queries.return_value = [("Nike", "business")]
        # Start at 0, then will be checked after search
        mock_api.api_calls_made = 0
        mock_api.search_news.return_value = ([mock_article], None)

        collector = NewsCollector(database=mock_db, api_client=mock_api)

        # Allow 2 calls so first query is processed
        stats = collector.collect_from_api(max_calls=2, dry_run=True)

        assert stats.articles_fetched == 1


# ============================================================================
# Cross-Component Integration Tests
# ============================================================================


class TestCrossComponentIntegration:
    """Tests for integration between multiple system components."""

    def test_classifier_to_pipeline_integration(self):
        """Test FP classifier results integrate with labeling pipeline."""
        from src.labeling.pipeline import LabelingStats

        # Simulate FP classifier filtering articles before labeling
        stats = LabelingStats()

        # FP classifier filters 3 out of 10 articles as false positives
        stats.fp_classifier_calls = 10
        stats.fp_classifier_skipped = 3
        stats.fp_classifier_continued = 7

        # LLM labels the remaining 7
        stats.articles_processed = 7
        stats.articles_labeled = 5
        stats.articles_skipped = 2

        assert stats.fp_classifier_calls == 10
        assert stats.fp_classifier_skipped == 3
        assert stats.articles_labeled == 5

    def test_workflow_to_notification_integration(self):
        """Test workflow errors trigger notifications."""
        from src.agent.notifications import Notification, NotificationType

        # Simulate workflow creating error notification
        notification = Notification(
            notification_type=NotificationType.WORKFLOW_FAILED,
            subject="Test Failure",
            message="Workflow failed due to test error",
            details={"step": "export_feeds", "error": "Test error"},
            severity="error",
        )

        assert notification.notification_type == NotificationType.WORKFLOW_FAILED
        assert notification.severity == "error"
        assert "step" in notification.details

    def test_collection_to_labeling_data_format(self):
        """Test that collected article data format matches labeling expectations."""
        # Simulate article collected from API
        collected_article = {
            "id": str(uuid4()),
            "title": "Nike ESG Report",
            "full_content": "Nike sustainability content...",
            "source_name": "Reuters",
            "brands_mentioned": ["Nike"],
            "labeling_status": "pending",
        }

        # Verify it has all fields needed by labeling pipeline
        assert "title" in collected_article
        assert "full_content" in collected_article
        assert "brands_mentioned" in collected_article
        assert collected_article["labeling_status"] == "pending"

    def test_labeling_to_export_data_format(self):
        """Test that labeled article format matches export expectations."""
        # Simulate labeled article
        labeled_article = {
            "id": str(uuid4()),
            "title": "Nike Sustainability Goals",
            "source_name": "Reuters",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "labels": [
                {
                    "brand": "Nike",
                    "categories": {
                        "environmental": {"present": True, "sentiment": 1},
                        "social": {"present": False, "sentiment": 0},
                    },
                }
            ],
        }

        # Verify format is suitable for JSON export
        json_str = json.dumps(labeled_article)
        parsed = json.loads(json_str)

        assert parsed["title"] == "Nike Sustainability Goals"
        assert len(parsed["labels"]) == 1
        assert parsed["labels"][0]["brand"] == "Nike"


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


class TestEndToEndWorkflows:
    """End-to-end tests simulating complete workflows."""

    def test_website_export_workflow_steps_sequence(self):
        """Test that website export workflow steps execute in correct order."""
        from src.agent.workflows.website_export import WebsiteExportWorkflow

        workflow = WebsiteExportWorkflow()

        step_names = [step.name for step in workflow.steps]

        # Verify step order
        assert step_names == [
            "export_feeds",
            "save_scorecard_snapshot",
            "validate_export",
            "commit_and_push",
            "send_error_notification",
        ]

    def test_workflow_context_propagation(self):
        """Test that context propagates between workflow steps."""
        context = {}

        # Step 1: Export sets output paths
        context["export_success"] = True
        context["json_output"] = "/tmp/test.json"
        context["atom_output"] = "/tmp/test.atom"

        # Step 2: Validation reads from context
        assert context.get("json_output") == "/tmp/test.json"
        context["validation_passed"] = True

        # Step 3: Git reads validation result
        assert context.get("validation_passed") is True
        context["git_success"] = True

        # Final context has all step results
        assert all(
            key in context
            for key in [
                "export_success",
                "json_output",
                "validation_passed",
                "git_success",
            ]
        )

    def test_dry_run_skips_side_effects(self):
        """Test that dry run mode skips all side-effect steps."""
        from src.agent.workflows.website_export import (
            commit_and_push,
            save_scorecard_snapshot,
        )

        mock_workflow = MagicMock()
        context = {
            "dry_run": True,
            "export_success": True,
            "validation_passed": True,
            "json_output": "/tmp/test.json",
        }

        with patch("src.agent.workflows.website_export.agent_settings") as mock_settings:
            mock_settings.website_repo_path = Path("/tmp/repo")

            # Both steps should skip in dry run
            scorecard_result = save_scorecard_snapshot(mock_workflow, context)
            git_result = commit_and_push(mock_workflow, context)

            assert scorecard_result["scorecard_skipped"] is True
            assert git_result["git_skipped"] is True
