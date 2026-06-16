"""Pure comparison logic for model-migration evaluation.

No DB or API access here — everything operates on plain data structures so it
can be unit-tested without a database or live model. The orchestration script
builds the baseline decisions from stored ``brand_labels`` and the candidate
decisions from a fresh labeling run, then calls into these functions.

The functions are deliberately generic ("baseline" vs "candidate" label sets)
rather than hardcoding "DB vs live 4.6", so the same logic can back the general
prompt/model contrastive harness (issue #4, #39).
"""

from __future__ import annotations

from src.labeling.models import BrandAnalysis, LabelingResponse
from src.migration_eval.models import (
    CATEGORIES,
    ArticleComparison,
    BrandComparison,
    BrandDecision,
    CategoryDiff,
    MigrationVerdict,
    Thresholds,
)


def effective_decision_from_analysis(analysis: BrandAnalysis) -> BrandDecision | None:
    """Derive the *persisted* decision for a brand, mirroring save_brand_labels.

    Returns None when the brand would NOT be saved as a ``brand_labels`` row:
    a non-sportswear brand (false positive), a brand with no category data, or a
    sportswear brand where no ESG category applies. This makes the candidate
    output apples-to-apples with the stored baseline, which only contains
    sportswear brands with at least one applicable category.
    """
    if not analysis.is_sportswear_brand:
        return None
    if not analysis.categories:
        return None
    applies = {c: bool(analysis.categories.get(c) and analysis.categories[c].applies) for c in CATEGORIES}
    if not any(applies.values()):
        return None
    sentiment = {
        c: (analysis.categories[c].sentiment if c in analysis.categories else None)
        for c in CATEGORIES
    }
    return BrandDecision(
        brand=analysis.brand,
        applies=applies,
        sentiment=sentiment,
        confidence=analysis.confidence,
        reasoning=analysis.reasoning,
    )


def decisions_from_response(response: LabelingResponse) -> dict[str, BrandDecision]:
    """Build {brand_lower: BrandDecision} for brands that would be persisted."""
    decisions: dict[str, BrandDecision] = {}
    for analysis in response.brand_analyses:
        decision = effective_decision_from_analysis(analysis)
        if decision is not None:
            decisions[decision.brand.lower()] = decision
    return decisions


def outcome_from_decisions(decisions: dict[str, BrandDecision]) -> str:
    """Article-level outcome implied by a set of effective decisions."""
    return "labeled" if decisions else "false_positive"


def compare_article(
    article_id: str,
    title: str,
    baseline: dict[str, BrandDecision],
    candidate: dict[str, BrandDecision],
    *,
    baseline_reasoning: str = "",
    candidate_reasoning: str = "",
    is_human_anchor: bool = False,
    parse_ok: bool = True,
    error: str | None = None,
) -> ArticleComparison:
    """Compare baseline vs candidate effective decisions for one article.

    Brand keys are lowercased brand names. The brand universe is the union of
    both sides, so brands dropped or newly added by the candidate are captured
    as disagreements (and contribute to the category confusion counts).
    """
    brand_keys = sorted(set(baseline) | set(candidate))
    brand_comparisons: list[BrandComparison] = []
    for key in brand_keys:
        b = baseline.get(key)
        c = candidate.get(key)
        diffs = [
            CategoryDiff(
                category=cat,
                baseline_applies=bool(b and b.applies.get(cat)),
                candidate_applies=bool(c and c.applies.get(cat)),
                baseline_sentiment=(b.sentiment.get(cat) if b else None),
                candidate_sentiment=(c.sentiment.get(cat) if c else None),
            )
            for cat in CATEGORIES
        ]
        brand_comparisons.append(
            BrandComparison(
                brand=(b.brand if b else c.brand),
                in_baseline=b is not None,
                in_candidate=c is not None,
                category_diffs=diffs,
            )
        )

    return ArticleComparison(
        article_id=article_id,
        title=title,
        baseline_outcome=outcome_from_decisions(baseline),
        candidate_outcome=outcome_from_decisions(candidate),
        brand_comparisons=brand_comparisons,
        baseline_reasoning=baseline_reasoning,
        candidate_reasoning=candidate_reasoning,
        is_human_anchor=is_human_anchor,
        parse_ok=parse_ok,
        error=error,
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_verdict(
    comparisons: list[ArticleComparison],
    thresholds: Thresholds | None = None,
    *,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    mean_latency_s: float = 0.0,
) -> MigrationVerdict:
    """Aggregate comparisons into metrics + an advisory verdict.

    Only articles that parsed successfully count toward the agreement metrics;
    parse failures are tracked separately (and trigger a hard fail).
    """
    thresholds = thresholds or Thresholds()

    n_parse_failures = sum(1 for c in comparisons if not c.parse_ok)
    scored = [c for c in comparisons if c.parse_ok]
    n = len(scored)

    # Article-outcome confusion: first letter = baseline, second = candidate.
    # l = labeled, f = false_positive.
    confusion = {"ll": 0, "lf": 0, "fl": 0, "ff": 0}
    for c in scored:
        b = "l" if c.baseline_outcome == "labeled" else "f"
        cand = "l" if c.candidate_outcome == "labeled" else "f"
        confusion[b + cand] += 1

    disagreements = confusion["lf"] + confusion["fl"]
    outcome_disagreement_rate = disagreements / n if n else 0.0
    baseline_fp = confusion["fl"] + confusion["ff"]
    baseline_labeled = confusion["ll"] + confusion["lf"]
    newly_labeled_rate = confusion["fl"] / baseline_fp if baseline_fp else 0.0
    dropped_label_rate = confusion["lf"] / baseline_labeled if baseline_labeled else 0.0

    # Per-category boolean confusion (baseline True = positive class).
    cat_metrics: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for cat in CATEGORIES:
        tp = fp = fn = 0
        for c in scored:
            for bc in c.brand_comparisons:
                d = next((x for x in bc.category_diffs if x.category == cat), None)
                if d is None:
                    continue
                if d.baseline_applies and d.candidate_applies:
                    tp += 1
                elif not d.baseline_applies and d.candidate_applies:
                    fp += 1
                elif d.baseline_applies and not d.candidate_applies:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = _f1(precision, recall)
        cat_metrics[cat] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    category_f1_macro = sum(f1s) / len(f1s) if f1s else 0.0

    # Sentiment agreement where both mark the category applicable.
    sent_total = sent_exact = sent_within1 = 0
    for c in scored:
        for bc in c.brand_comparisons:
            for d in bc.category_diffs:
                if d.baseline_applies and d.candidate_applies:
                    sent_total += 1
                    if d.baseline_sentiment == d.candidate_sentiment:
                        sent_exact += 1
                    if (
                        d.baseline_sentiment is not None
                        and d.candidate_sentiment is not None
                        and abs(d.baseline_sentiment - d.candidate_sentiment) <= 1
                    ):
                        sent_within1 += 1
    sentiment_exact_rate = sent_exact / sent_total if sent_total else 1.0
    sentiment_within1_rate = sent_within1 / sent_total if sent_total else 1.0

    human_anchors = [c for c in scored if c.is_human_anchor]
    n_human_anchor_disagreements = sum(1 for c in human_anchors if c.has_disagreement)

    # Advisory flags (noise-screen; human triage is the real gate).
    flags: list[str] = []
    if n_parse_failures > thresholds.max_parse_failures:
        flags.append(
            f"HARD_FAIL: {n_parse_failures} parse/integration failure(s) "
            f"(threshold {thresholds.max_parse_failures})"
        )
    if outcome_disagreement_rate > thresholds.max_outcome_disagreement:
        flags.append(
            f"INVESTIGATE: outcome disagreement {outcome_disagreement_rate:.1%} "
            f"> {thresholds.max_outcome_disagreement:.0%}"
        )
    if newly_labeled_rate > thresholds.max_newly_labeled_rate:
        flags.append(
            f"INVESTIGATE: candidate newly labels {newly_labeled_rate:.1%} of baseline "
            f"false-positives (scorecard-inflation risk) > {thresholds.max_newly_labeled_rate:.0%}"
        )
    if category_f1_macro < thresholds.min_category_f1:
        flags.append(
            f"INVESTIGATE: macro category F1 {category_f1_macro:.3f} "
            f"< {thresholds.min_category_f1}"
        )
    if sentiment_exact_rate < thresholds.min_sentiment_exact:
        flags.append(
            f"INVESTIGATE: sentiment exact-match {sentiment_exact_rate:.1%} "
            f"< {thresholds.min_sentiment_exact:.0%}"
        )

    return MigrationVerdict(
        n_compared=n,
        n_parse_failures=n_parse_failures,
        outcome_confusion=confusion,
        outcome_disagreement_rate=outcome_disagreement_rate,
        newly_labeled_rate=newly_labeled_rate,
        dropped_label_rate=dropped_label_rate,
        category_metrics=cat_metrics,
        category_f1_macro=category_f1_macro,
        sentiment_exact_rate=sentiment_exact_rate,
        sentiment_within1_rate=sentiment_within1_rate,
        n_human_anchors=len(human_anchors),
        n_human_anchor_disagreements=n_human_anchor_disagreements,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        mean_latency_s=mean_latency_s,
        flags=flags,
    )
