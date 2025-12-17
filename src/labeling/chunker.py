"""Article chunking for embeddings and evidence retrieval."""

import logging
import re
from dataclasses import dataclass

import tiktoken

from .config import labeling_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of article text with position tracking."""

    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int


class ArticleChunker:
    """Chunks articles into semantic units for embedding and evidence retrieval.

    Uses paragraph-based chunking with sentence boundary awareness:
    - Splits on paragraph boundaries (double newlines)
    - Merges small paragraphs to meet minimum token threshold
    - Splits large paragraphs at sentence boundaries
    - Tracks character positions for evidence linking
    """

    def __init__(
        self,
        target_tokens: int | None = None,
        max_tokens: int | None = None,
        min_tokens: int | None = None,
        overlap_tokens: int | None = None,
        model: str = "cl100k_base",
    ):
        """Initialize the chunker.

        Args:
            target_tokens: Target chunk size in tokens (default: from settings)
            max_tokens: Maximum chunk size in tokens (default: from settings)
            min_tokens: Minimum chunk size in tokens (default: from settings)
            overlap_tokens: Overlap between chunks in tokens (default: from settings)
            model: Tiktoken encoding model for token counting
        """
        self.target_tokens = target_tokens or labeling_settings.target_chunk_tokens
        self.max_tokens = max_tokens or labeling_settings.max_chunk_tokens
        self.min_tokens = min_tokens or labeling_settings.min_chunk_tokens
        self.overlap_tokens = overlap_tokens or labeling_settings.chunk_overlap_tokens

        # Initialize tiktoken encoder
        self.encoder = tiktoken.get_encoding(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving sentence endings."""
        # Pattern matches sentence-ending punctuation followed by space or end
        # Handles common abbreviations to avoid false splits
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list[tuple[str, int]]:
        """Split text into paragraphs with their starting positions.

        Returns:
            List of (paragraph_text, char_start) tuples
        """
        paragraphs = []
        current_pos = 0

        # Split on double newlines (paragraph breaks)
        parts = re.split(r'\n\s*\n', text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Find the actual position of this paragraph in the original text
            start_pos = text.find(part, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            paragraphs.append((part, start_pos))
            current_pos = start_pos + len(part)

        return paragraphs

    def _split_long_paragraph(self, text: str, char_start: int) -> list[tuple[str, int, int]]:
        """Split a long paragraph into smaller chunks at sentence boundaries.

        Args:
            text: The paragraph text to split
            char_start: Starting character position in the original document

        Returns:
            List of (chunk_text, char_start, char_end) tuples
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [(text, char_start, char_start + len(text))]

        chunks = []
        current_chunk_sentences = []
        current_chunk_start = char_start
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds max, add it as its own chunk
            if sentence_tokens > self.max_tokens:
                # First, save any accumulated sentences
                if current_chunk_sentences:
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunk_end = current_chunk_start + len(chunk_text)
                    chunks.append((chunk_text, current_chunk_start, chunk_end))
                    current_chunk_sentences = []
                    current_tokens = 0

                # Add the long sentence as its own chunk
                sentence_start = text.find(sentence)
                if sentence_start != -1:
                    abs_start = char_start + sentence_start
                else:
                    abs_start = current_chunk_start
                chunks.append((sentence, abs_start, abs_start + len(sentence)))
                current_chunk_start = abs_start + len(sentence) + 1
                continue

            # If adding this sentence would exceed target, start new chunk
            if current_tokens + sentence_tokens > self.target_tokens and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_end = current_chunk_start + len(chunk_text)
                chunks.append((chunk_text, current_chunk_start, chunk_end))

                current_chunk_start = chunk_end + 1
                current_chunk_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_end = current_chunk_start + len(chunk_text)
            chunks.append((chunk_text, current_chunk_start, chunk_end))

        return chunks

    def chunk_article(self, content: str) -> list[Chunk]:
        """Chunk an article into semantic units.

        Args:
            content: The full article content to chunk

        Returns:
            List of Chunk objects with text and position information
        """
        if not content or not content.strip():
            return []

        content = content.strip()
        total_tokens = self.count_tokens(content)

        # Short articles: single chunk
        if total_tokens <= self.target_tokens:
            return [
                Chunk(
                    index=0,
                    text=content,
                    char_start=0,
                    char_end=len(content),
                    token_count=total_tokens,
                )
            ]

        paragraphs = self._split_into_paragraphs(content)
        if not paragraphs:
            return [
                Chunk(
                    index=0,
                    text=content,
                    char_start=0,
                    char_end=len(content),
                    token_count=total_tokens,
                )
            ]

        # Process paragraphs into chunks
        raw_chunks: list[tuple[str, int, int]] = []  # (text, char_start, char_end)

        for para_text, para_start in paragraphs:
            para_tokens = self.count_tokens(para_text)

            if para_tokens > self.max_tokens:
                # Split long paragraph at sentence boundaries
                sub_chunks = self._split_long_paragraph(para_text, para_start)
                raw_chunks.extend(sub_chunks)
            else:
                para_end = para_start + len(para_text)
                raw_chunks.append((para_text, para_start, para_end))

        # Merge small chunks
        merged_chunks = self._merge_small_chunks(raw_chunks)

        # Convert to Chunk objects
        chunks = []
        for i, (text, char_start, char_end) in enumerate(merged_chunks):
            chunks.append(
                Chunk(
                    index=i,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=self.count_tokens(text),
                )
            )

        logger.debug(f"Chunked article into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks

    def _merge_small_chunks(
        self, chunks: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Merge consecutive small chunks that are below the minimum token threshold.

        Args:
            chunks: List of (text, char_start, char_end) tuples

        Returns:
            Merged list of chunks
        """
        if not chunks:
            return []

        merged = []
        current_text = ""
        current_start = 0
        current_end = 0
        current_tokens = 0

        for text, char_start, char_end in chunks:
            chunk_tokens = self.count_tokens(text)

            if not current_text:
                # Start new accumulator
                current_text = text
                current_start = char_start
                current_end = char_end
                current_tokens = chunk_tokens
            elif current_tokens + chunk_tokens <= self.target_tokens:
                # Merge with current chunk
                current_text = current_text + "\n\n" + text
                current_end = char_end
                current_tokens += chunk_tokens
            else:
                # Save current and start new
                if current_tokens >= self.min_tokens or len(merged) == 0:
                    merged.append((current_text, current_start, current_end))
                else:
                    # Too small, merge with previous if exists
                    if merged:
                        prev_text, prev_start, _ = merged.pop()
                        merged.append((prev_text + "\n\n" + current_text, prev_start, current_end))
                    else:
                        merged.append((current_text, current_start, current_end))

                current_text = text
                current_start = char_start
                current_end = char_end
                current_tokens = chunk_tokens

        # Don't forget the last chunk
        if current_text:
            if current_tokens >= self.min_tokens or len(merged) == 0:
                merged.append((current_text, current_start, current_end))
            elif merged:
                # Merge with previous
                prev_text, prev_start, _ = merged.pop()
                merged.append((prev_text + "\n\n" + current_text, prev_start, current_end))
            else:
                merged.append((current_text, current_start, current_end))

        return merged
