"""LLM-based labeling pipeline for ESG news classification."""

from .chunker import ArticleChunker, Chunk
from .config import ESG_CATEGORIES, LABELING_SYSTEM_PROMPT, labeling_settings
from .database import LabelingDatabase, labeling_db
from .embedder import OpenAIEmbedder
from .evidence_matcher import EvidenceMatch, EvidenceMatcher, match_all_evidence
from .labeler import ArticleLabeler, LabelingResult
from .models import BrandAnalysis, CategoryLabel, LabelingResponse
from .pipeline import LabelingPipeline, LabelingStats

__all__ = [
    "ArticleChunker",
    "ArticleLabeler",
    "BrandAnalysis",
    "CategoryLabel",
    "Chunk",
    "ESG_CATEGORIES",
    "EvidenceMatch",
    "EvidenceMatcher",
    "LABELING_SYSTEM_PROMPT",
    "LabelingDatabase",
    "LabelingPipeline",
    "LabelingResponse",
    "LabelingResult",
    "LabelingStats",
    "OpenAIEmbedder",
    "labeling_db",
    "labeling_settings",
    "match_all_evidence",
]
