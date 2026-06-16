"""Dataclasses for model-migration evaluation.

These are pure data containers with no DB or API dependencies so the
comparison logic in :mod:`src.migration_eval.compare` stays unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ESG category keys, matching BrandAnalysis.categories / BrandLabel columns.
# Note the column/key asymmetry for the digital category: the response model
# and BrandLabel boolean use "digital_transformation", but the BrandLabel
# sentiment column is "digital_sentiment".
CATEGORIES: tuple[str, ...] = (
    "environmental",
    "social",
    "governance",
    "digital_transformation",
)


@dataclass(frozen=True)
class BrandDecision:
    """A single brand's effective ESG label for one article.

    This is the apples-to-apples unit of comparison: it represents a brand
    that *would be persisted* as a ``brand_labels`` row (sportswear brand with
    at least one applicable category). Brands judged false-positive or with no
    applicable category are represented by the *absence* of a BrandDecision.
    """

    brand: str
    # Per-category applies flags, keyed by CATEGORIES.
    applies: dict[str, bool]
    # Per-category sentiment (-1/0/1 or None), keyed by CATEGORIES.
    sentiment: dict[str, int | None]
    confidence: float = 0.0
    reasoning: str = ""

    @property
    def applicable_categories(self) -> list[str]:
        return [c for c in CATEGORIES if self.applies.get(c)]


@dataclass
class CategoryDiff:
    """Per-category disagreement detail for one (article, brand)."""

    category: str
    baseline_applies: bool
    candidate_applies: bool
    baseline_sentiment: int | None
    candidate_sentiment: int | None

    @property
    def applies_match(self) -> bool:
        return self.baseline_applies == self.candidate_applies

    @property
    def sentiment_match(self) -> bool:
        # Only meaningful when both mark the category as applicable.
        return self.baseline_sentiment == self.candidate_sentiment


@dataclass
class BrandComparison:
    """Comparison of one brand across baseline and candidate for one article."""

    brand: str
    in_baseline: bool
    in_candidate: bool
    category_diffs: list[CategoryDiff] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.in_baseline and self.in_candidate:
            return "shared"
        if self.in_baseline:
            return "dropped"  # candidate no longer labels this brand
        return "added"  # candidate newly labels this brand


@dataclass
class ArticleComparison:
    """Full comparison for a single article."""

    article_id: str
    title: str
    baseline_outcome: str  # "labeled" | "false_positive"
    candidate_outcome: str  # "labeled" | "false_positive"
    brand_comparisons: list[BrandComparison] = field(default_factory=list)
    # Provenance / triage aids.
    baseline_reasoning: str = ""
    candidate_reasoning: str = ""
    is_human_anchor: bool = False  # baseline includes a human_reviewed label
    parse_ok: bool = True  # candidate response parsed successfully
    error: str | None = None

    @property
    def outcome_match(self) -> bool:
        return self.baseline_outcome == self.candidate_outcome

    @property
    def has_disagreement(self) -> bool:
        if not self.parse_ok:
            return True
        if not self.outcome_match:
            return True
        for bc in self.brand_comparisons:
            if bc.status != "shared":
                return True
            if any(
                not d.applies_match or not d.sentiment_match for d in bc.category_diffs
            ):
                return True
        return False


@dataclass
class Thresholds:
    """Advisory pass/fail thresholds (a noise-screen, not the real gate).

    The real gate is human adjudication of the disagreement table. These only
    flag results worth a closer look.
    """

    # Any parse failure means the integration broke -> hard fail.
    max_parse_failures: int = 0
    # Fraction of compared articles whose labeled/false_positive outcome flips.
    max_outcome_disagreement: float = 0.10
    # Per-category boolean agreement F1 (candidate vs baseline as reference).
    min_category_f1: float = 0.85
    # Exact-match rate of sentiment where both mark a category applicable.
    min_sentiment_exact: float = 0.80
    # Fraction of baseline false-positive articles the candidate newly labels
    # (scorecard-inflation risk).
    max_newly_labeled_rate: float = 0.10


@dataclass
class MigrationVerdict:
    """Aggregate metrics + advisory verdict over a set of comparisons."""

    n_compared: int
    n_parse_failures: int
    outcome_confusion: dict[str, int]  # keys: ll, lf, fl, ff
    outcome_disagreement_rate: float
    newly_labeled_rate: float
    dropped_label_rate: float
    category_metrics: dict[str, dict[str, float]]  # category -> {precision,recall,f1}
    category_f1_macro: float
    sentiment_exact_rate: float
    sentiment_within1_rate: float
    n_human_anchors: int
    n_human_anchor_disagreements: int
    # Cost / performance.
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    mean_latency_s: float = 0.0
    # Advisory verdict.
    flags: list[str] = field(default_factory=list)

    @property
    def hard_fail(self) -> bool:
        return any(f.startswith("HARD_FAIL") for f in self.flags)

    @property
    def passed_advisory(self) -> bool:
        return len(self.flags) == 0
