"""OpenAI embedding generation for article chunks."""

import logging
import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError

from .config import labeling_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: list[list[float]]
    total_tokens: int
    model: str


class OpenAIEmbedder:
    """Generates embeddings using OpenAI's text-embedding models.

    Features:
    - Batch processing for efficiency
    - Automatic retry with exponential backoff for rate limits
    - Token usage tracking for cost estimation
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key (default: from settings)
            model: Embedding model to use (default: from settings)
            batch_size: Number of texts per API call (default: from settings)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial delay between retries in seconds
        """
        self.api_key = api_key or labeling_settings.openai_api_key
        self.model = model or labeling_settings.embedding_model
        self.batch_size = batch_size or labeling_settings.embedding_batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

        # Track usage
        self.total_tokens_used = 0
        self.total_api_calls = 0

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (1536 dimensions for text-embedding-3-small)
        """
        result = self.embed_batch([text])
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings and usage info
        """
        if not texts:
            return EmbeddingResult(embeddings=[], total_tokens=0, model=self.model)

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings, tokens = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
            total_tokens += tokens

        self.total_tokens_used += total_tokens
        self.total_api_calls += (len(texts) + self.batch_size - 1) // self.batch_size

        logger.debug(
            f"Generated {len(texts)} embeddings using {total_tokens} tokens "
            f"(total session: {self.total_tokens_used} tokens)"
        )

        return EmbeddingResult(
            embeddings=all_embeddings,
            total_tokens=total_tokens,
            model=self.model,
        )

    def _embed_with_retry(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Embed texts with automatic retry for rate limits.

        Args:
            texts: Batch of texts to embed

        Returns:
            Tuple of (embeddings list, tokens used)
        """
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                embeddings = [item.embedding for item in response.data]
                tokens = response.usage.total_tokens

                return embeddings, tokens

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    raise

            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise

        # Should not reach here, but just in case
        raise RuntimeError("Embedding failed after all retries")

    def get_stats(self) -> dict[str, int | float]:
        """Get embedding statistics.

        Returns:
            Dictionary with usage statistics
        """
        # Cost estimate for text-embedding-3-small: $0.00002 per 1K tokens
        estimated_cost = (self.total_tokens_used / 1000) * 0.00002

        return {
            "total_tokens": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_api_calls = 0
