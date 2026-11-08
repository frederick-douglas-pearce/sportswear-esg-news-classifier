"""LLM-based labeling pipeline for ESG news classification.

The package attributes below load their submodule on first access (PEP 562), not at import.
Agent processes import `src.labeling.exit_codes`, which runs this file; an eager import here
built `labeling_settings` in every agent process, so one unparseable labeling setting stopped
every agent workflow, `run_audit` included (#211). `tests/test_labeling_import_isolation.py`
pins that importing the agent does not import `src.labeling.config`.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# Public name -> the submodule that defines it.
_EXPORTS = {
    "ArticleChunker": "chunker",
    "Chunk": "chunker",
    "ESG_CATEGORIES": "config",
    "LABELING_SYSTEM_PROMPT": "config",
    "labeling_settings": "config",
    "LabelingDatabase": "database",
    "labeling_db": "database",
    "OpenAIEmbedder": "embedder",
    "EvidenceMatch": "evidence_matcher",
    "EvidenceMatcher": "evidence_matcher",
    "match_all_evidence": "evidence_matcher",
    "ArticleLabeler": "labeler",
    "LabelingResult": "labeler",
    "BrandAnalysis": "models",
    "CategoryLabel": "models",
    "LabelingResponse": "models",
    "LabelingPipeline": "pipeline",
    "LabelingStats": "pipeline",
    "CrossEncoderReranker": "reranker",
    "RerankCandidate": "reranker",
    "RerankResult": "reranker",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from .chunker import ArticleChunker, Chunk
    from .config import ESG_CATEGORIES, LABELING_SYSTEM_PROMPT, labeling_settings
    from .database import LabelingDatabase, labeling_db
    from .embedder import OpenAIEmbedder
    from .evidence_matcher import EvidenceMatch, EvidenceMatcher, match_all_evidence
    from .labeler import ArticleLabeler, LabelingResult
    from .models import BrandAnalysis, CategoryLabel, LabelingResponse
    from .pipeline import LabelingPipeline, LabelingStats
    from .reranker import CrossEncoderReranker, RerankCandidate, RerankResult
