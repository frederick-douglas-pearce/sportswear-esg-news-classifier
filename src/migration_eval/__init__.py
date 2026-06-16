"""Retrospective model-migration evaluation.

Compares a candidate model's labeling decisions against the labels already
stored in the database (produced by the outgoing model) to catch significant
regressions before promoting the candidate to production.

The comparison logic in :mod:`src.migration_eval.compare` is pure and
DB/API-free so it can be unit-tested. The orchestration (DB sampling, API
calls, reporting) lives in ``scripts/eval_model_migration.py``.
"""

from src.migration_eval.compare import (
    build_verdict,
    compare_article,
    decisions_from_response,
    effective_decision_from_analysis,
    outcome_from_decisions,
)
from src.migration_eval.models import (
    CATEGORIES,
    ArticleComparison,
    BrandComparison,
    BrandDecision,
    CategoryDiff,
    MigrationVerdict,
    Thresholds,
)

__all__ = [
    "CATEGORIES",
    "ArticleComparison",
    "BrandComparison",
    "BrandDecision",
    "CategoryDiff",
    "MigrationVerdict",
    "Thresholds",
    "build_verdict",
    "compare_article",
    "decisions_from_response",
    "effective_decision_from_analysis",
    "outcome_from_decisions",
]
