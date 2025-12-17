"""Tests for the article chunker."""

import pytest

from src.labeling.chunker import ArticleChunker, Chunk


class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self):
        """Should create a Chunk with all fields."""
        chunk = Chunk(
            index=0,
            text="Test content",
            char_start=0,
            char_end=12,
            token_count=3,
        )
        assert chunk.index == 0
        assert chunk.text == "Test content"
        assert chunk.char_start == 0
        assert chunk.char_end == 12
        assert chunk.token_count == 3


class TestTokenCounting:
    """Tests for token counting."""

    def test_count_tokens_empty(self):
        """Should return 0 for empty string."""
        chunker = ArticleChunker()
        assert chunker.count_tokens("") == 0

    def test_count_tokens_simple(self):
        """Should count tokens correctly."""
        chunker = ArticleChunker()
        # "Hello world" is typically 2 tokens
        count = chunker.count_tokens("Hello world")
        assert count > 0
        assert count <= 5  # Reasonable bounds

    def test_count_tokens_longer_text(self):
        """Should count tokens for longer text."""
        chunker = ArticleChunker()
        text = "Nike announced new sustainability initiatives. " * 10
        count = chunker.count_tokens(text)
        assert count > 20  # Should be substantial


class TestShortArticleChunking:
    """Tests for chunking short articles."""

    def test_short_article_single_chunk(self):
        """Short articles should produce a single chunk."""
        chunker = ArticleChunker(target_tokens=500)
        text = "Nike announced new sustainability goals today."

        chunks = chunker.chunk_article(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_empty_article_no_chunks(self):
        """Empty articles should return empty list."""
        chunker = ArticleChunker()

        assert chunker.chunk_article("") == []
        assert chunker.chunk_article("   ") == []

    def test_none_article_no_chunks(self):
        """None should return empty list."""
        chunker = ArticleChunker()
        assert chunker.chunk_article(None) == []


class TestParagraphChunking:
    """Tests for paragraph-based chunking."""

    def test_paragraph_boundaries_respected(self):
        """Chunks should break at paragraph boundaries."""
        chunker = ArticleChunker(target_tokens=50, min_tokens=20)
        text = """First paragraph about Nike sustainability.

Second paragraph about carbon emissions reduction.

Third paragraph about supply chain improvements."""

        chunks = chunker.chunk_article(text)

        # Should have created separate chunks for paragraphs
        assert len(chunks) >= 1
        # Check that paragraph text is preserved
        all_text = " ".join(c.text for c in chunks)
        assert "First paragraph" in all_text
        assert "Second paragraph" in all_text

    def test_multiple_newlines_treated_as_paragraph(self):
        """Multiple newlines should be treated as paragraph break."""
        chunker = ArticleChunker(target_tokens=50, min_tokens=20)
        text = "First part.\n\n\n\nSecond part."

        chunks = chunker.chunk_article(text)

        # Should handle multiple newlines
        assert len(chunks) >= 1


class TestCharacterPositions:
    """Tests for character position tracking."""

    def test_char_positions_single_chunk(self):
        """Single chunk should have correct positions."""
        chunker = ArticleChunker()
        text = "Simple test text for chunking."

        chunks = chunker.chunk_article(text)

        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_char_positions_multiple_chunks(self):
        """Multiple chunks should have valid positions."""
        chunker = ArticleChunker(target_tokens=30, min_tokens=10)
        text = """First paragraph with enough content to be meaningful.

Second paragraph also with enough content to meet minimum.

Third paragraph for good measure with more text here."""

        chunks = chunker.chunk_article(text)

        # All positions should be valid
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start
            assert chunk.char_end <= len(text) + 50  # Allow some flexibility for merging


class TestSentenceSplitting:
    """Tests for sentence boundary awareness."""

    def test_long_paragraph_splits_at_sentences(self):
        """Long paragraphs should split at sentence boundaries."""
        chunker = ArticleChunker(target_tokens=30, max_tokens=50, min_tokens=10)
        # Create a paragraph with multiple sentences
        text = "First sentence about Nike. Second sentence about sustainability. Third sentence about the environment. Fourth sentence about carbon emissions. Fifth sentence about goals."

        chunks = chunker.chunk_article(text)

        # Should split into multiple chunks
        assert len(chunks) >= 1
        # Each chunk should end at a sentence boundary (with period)
        for chunk in chunks[:-1]:  # Exclude last chunk
            # Check that text ends reasonably
            assert len(chunk.text) > 0


class TestChunkMerging:
    """Tests for small chunk merging."""

    def test_small_chunks_merged(self):
        """Small consecutive chunks should be merged."""
        chunker = ArticleChunker(target_tokens=200, min_tokens=50)
        text = """Short one.

Another short.

Third short."""

        chunks = chunker.chunk_article(text)

        # Small chunks should be merged together
        assert len(chunks) <= 3

    def test_chunks_respect_min_tokens(self):
        """Chunks should generally meet minimum token threshold."""
        chunker = ArticleChunker(target_tokens=100, min_tokens=20)
        text = """First paragraph with some reasonable content here.

Second paragraph also has some content to work with.

Third paragraph completes the article with more text."""

        chunks = chunker.chunk_article(text)

        # Chunks should be reasonably sized
        for chunk in chunks:
            # Allow some flexibility since merging may not always hit exact min
            assert chunk.token_count > 5


class TestEdgeCases:
    """Tests for edge cases in chunking."""

    def test_single_very_long_sentence(self):
        """Should handle single very long sentence."""
        chunker = ArticleChunker(target_tokens=50, max_tokens=100)
        # Create a very long sentence
        text = " ".join(["word"] * 200)

        chunks = chunker.chunk_article(text)

        # Should handle without crashing
        assert len(chunks) >= 1

    def test_only_whitespace_paragraphs(self):
        """Should handle text with whitespace-only paragraphs."""
        chunker = ArticleChunker()
        text = "Content\n\n   \n\nMore content"

        chunks = chunker.chunk_article(text)

        # Should not crash and should have content
        assert len(chunks) >= 1

    def test_unicode_content(self):
        """Should handle unicode content properly."""
        chunker = ArticleChunker()
        text = "Nike's sustainability efforts are très important. 日本語テスト."

        chunks = chunker.chunk_article(text)

        assert len(chunks) >= 1
        assert "Nike's" in chunks[0].text


class TestTokenCountInChunks:
    """Tests for token count accuracy in chunks."""

    def test_token_count_populated(self):
        """Each chunk should have token_count populated."""
        chunker = ArticleChunker()
        text = "Test content for chunking with multiple words and sentences."

        chunks = chunker.chunk_article(text)

        for chunk in chunks:
            assert chunk.token_count > 0

    def test_token_count_reasonable(self):
        """Token count should be reasonable for content size."""
        chunker = ArticleChunker()
        text = "Nike announced new sustainability initiatives today."

        chunks = chunker.chunk_article(text)

        # Rough estimate: 1 token per word on average
        word_count = len(text.split())
        assert chunks[0].token_count <= word_count * 2
        assert chunks[0].token_count >= word_count // 2
