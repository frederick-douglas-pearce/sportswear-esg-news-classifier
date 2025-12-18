"""Tests for OpenAI embedding generation."""

from unittest.mock import MagicMock, patch

import pytest

from src.labeling.embedder import EmbeddingResult, OpenAIEmbedder


class TestEmbeddingResultDataclass:
    """Tests for EmbeddingResult dataclass."""

    def test_embedding_result_creation(self):
        """Should create embedding result with all fields."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            total_tokens=100,
            model="text-embedding-3-small",
        )
        assert len(result.embeddings) == 2
        assert result.total_tokens == 100
        assert result.model == "text-embedding-3-small"

    def test_embedding_result_empty(self):
        """Should create empty embedding result."""
        result = EmbeddingResult(
            embeddings=[],
            total_tokens=0,
            model="text-embedding-3-small",
        )
        assert len(result.embeddings) == 0
        assert result.total_tokens == 0


class TestOpenAIEmbedderInit:
    """Tests for OpenAIEmbedder initialization."""

    def test_missing_api_key_raises_error(self):
        """Should raise error when API key is missing."""
        with patch("src.labeling.embedder.labeling_settings") as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.embedding_batch_size = 100

            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIEmbedder()

    def test_custom_parameters(self):
        """Should use custom parameters when provided."""
        with patch("src.labeling.embedder.OpenAI"):
            embedder = OpenAIEmbedder(
                api_key="test-key",
                model="custom-model",
                batch_size=50,
                max_retries=5,
                retry_delay=2.0,
            )
            assert embedder.api_key == "test-key"
            assert embedder.model == "custom-model"
            assert embedder.batch_size == 50
            assert embedder.max_retries == 5
            assert embedder.retry_delay == 2.0


class TestOpenAIEmbedderEmbedBatch:
    """Tests for batch embedding generation."""

    def test_embed_batch_empty_list(self):
        """Should return empty result for empty input."""
        with patch("src.labeling.embedder.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
            result = embedder.embed_batch([])

            assert result.embeddings == []
            assert result.total_tokens == 0

    def test_embed_batch_single_text(self):
        """Should embed single text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            result = embedder.embed_batch(["test text"])

            assert len(result.embeddings) == 1
            assert result.embeddings[0] == [0.1, 0.2, 0.3]
            assert result.total_tokens == 10

    def test_embed_batch_multiple_texts(self):
        """Should embed multiple texts."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_response.usage.total_tokens = 20
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            result = embedder.embed_batch(["text 1", "text 2"])

            assert len(result.embeddings) == 2
            assert result.total_tokens == 20

    def test_embed_batch_tracks_usage(self):
        """Should track cumulative token usage."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")

            # First batch
            embedder.embed_batch(["text 1"])
            assert embedder.total_tokens_used == 10
            assert embedder.total_api_calls == 1

            # Second batch
            embedder.embed_batch(["text 2"])
            assert embedder.total_tokens_used == 20
            assert embedder.total_api_calls == 2


class TestOpenAIEmbedderEmbedText:
    """Tests for single text embedding."""

    def test_embed_text_returns_vector(self):
        """Should return embedding vector for single text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])]
        mock_response.usage.total_tokens = 5
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            embedding = embedder.embed_text("single text")

            assert embedding == [0.1, 0.2, 0.3, 0.4]


class TestOpenAIEmbedderBatching:
    """Tests for batch processing with size limits."""

    def test_embed_batch_respects_batch_size(self):
        """Should split large batches into smaller ones."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 5
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key", batch_size=2)

            # 5 texts with batch_size=2 should result in 3 API calls
            texts = ["text1", "text2", "text3", "text4", "text5"]
            embedder.embed_batch(texts)

            assert mock_client.embeddings.create.call_count == 3


class TestOpenAIEmbedderRetry:
    """Tests for retry logic on rate limits."""

    def test_retry_on_rate_limit(self):
        """Should retry on rate limit error."""
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 5

        # First call raises RateLimitError, second succeeds
        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit", response=MagicMock(), body=None),
            mock_response,
        ]

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            with patch("src.labeling.embedder.time.sleep"):
                embedder = OpenAIEmbedder(api_key="test-key", max_retries=3, retry_delay=0.1)
                result = embedder.embed_batch(["test"])

                assert len(result.embeddings) == 1
                assert mock_client.embeddings.create.call_count == 2

    def test_raises_after_max_retries(self):
        """Should raise after max retries exceeded."""
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RateLimitError(
            "Rate limit", response=MagicMock(), body=None
        )

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            with patch("src.labeling.embedder.time.sleep"):
                embedder = OpenAIEmbedder(api_key="test-key", max_retries=2, retry_delay=0.1)

                with pytest.raises(RateLimitError):
                    embedder.embed_batch(["test"])


class TestOpenAIEmbedderStats:
    """Tests for statistics tracking."""

    def test_get_stats(self):
        """Should return usage statistics."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 100
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            embedder.embed_batch(["test"])

            stats = embedder.get_stats()

            assert stats["total_tokens"] == 100
            assert stats["total_api_calls"] == 1
            assert "estimated_cost_usd" in stats
            assert stats["estimated_cost_usd"] > 0

    def test_reset_stats(self):
        """Should reset statistics."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 100
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            embedder.embed_batch(["test"])

            assert embedder.total_tokens_used == 100

            embedder.reset_stats()

            assert embedder.total_tokens_used == 0
            assert embedder.total_api_calls == 0

    def test_cost_estimation(self):
        """Should estimate cost correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage.total_tokens = 1000  # 1K tokens
        mock_client.embeddings.create.return_value = mock_response

        with patch("src.labeling.embedder.OpenAI", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="test-key")
            embedder.embed_batch(["test"])

            stats = embedder.get_stats()

            # text-embedding-3-small: $0.00002 per 1K tokens
            expected_cost = (1000 / 1000) * 0.00002
            assert abs(stats["estimated_cost_usd"] - expected_cost) < 0.00001
