"""Unit tests for the pure model-migration comparison logic."""

from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse
from src.migration_eval.compare import (
    build_verdict,
    compare_article,
    decisions_from_response,
    effective_decision_from_analysis,
    outcome_from_decisions,
)
from src.migration_eval.models import CATEGORIES, BrandDecision, Thresholds


def _cat(applies: bool, sentiment: int | None = None) -> CategoryLabel:
    return CategoryLabel(applies=applies, sentiment=sentiment, evidence=[])


def _analysis(
    brand: str,
    *,
    is_sportswear: bool = True,
    env: tuple[bool, int | None] = (False, None),
    soc: tuple[bool, int | None] = (False, None),
    gov: tuple[bool, int | None] = (False, None),
    dig: tuple[bool, int | None] = (False, None),
    confidence: float = 0.9,
    reasoning: str = "",
) -> BrandAnalysis:
    return BrandAnalysis(
        brand=brand,
        is_sportswear_brand=is_sportswear,
        categories={
            "environmental": _cat(*env),
            "social": _cat(*soc),
            "governance": _cat(*gov),
            "digital_transformation": _cat(*dig),
        },
        confidence=confidence,
        reasoning=reasoning,
    )


def _decision(
    brand: str,
    *,
    env: tuple[bool, int | None] = (False, None),
    soc: tuple[bool, int | None] = (False, None),
    gov: tuple[bool, int | None] = (False, None),
    dig: tuple[bool, int | None] = (False, None),
) -> BrandDecision:
    pairs = {"environmental": env, "social": soc, "governance": gov, "digital_transformation": dig}
    return BrandDecision(
        brand=brand,
        applies={c: pairs[c][0] for c in CATEGORIES},
        sentiment={c: pairs[c][1] for c in CATEGORIES},
    )


# --- effective_decision_from_analysis (mirrors save_brand_labels filter) ---


def test_non_sportswear_brand_is_filtered_out():
    a = _analysis("Puma", is_sportswear=False, env=(True, 1))
    assert effective_decision_from_analysis(a) is None


def test_brand_with_no_applicable_category_is_filtered_out():
    a = _analysis("Nike")  # all categories False
    assert effective_decision_from_analysis(a) is None


def test_brand_with_empty_categories_is_filtered_out():
    a = BrandAnalysis(brand="Nike", is_sportswear_brand=True, categories={})
    assert effective_decision_from_analysis(a) is None


def test_valid_sportswear_brand_yields_decision():
    a = _analysis("Nike", env=(True, -1), gov=(True, 1))
    d = effective_decision_from_analysis(a)
    assert d is not None
    assert d.applies["environmental"] is True
    assert d.applies["governance"] is True
    assert d.applies["social"] is False
    assert d.sentiment["environmental"] == -1
    assert d.sentiment["governance"] == 1
    assert d.applicable_categories == ["environmental", "governance"]


def test_decisions_from_response_keys_lowercased_and_filtered():
    resp = LabelingResponse(
        brand_analyses=[
            _analysis("Nike", env=(True, 1)),
            _analysis("Adidas", is_sportswear=False, env=(True, 1)),  # FP, dropped
            _analysis("Puma"),  # no categories apply, dropped
        ],
        article_summary="x",
    )
    decisions = decisions_from_response(resp)
    assert set(decisions) == {"nike"}


def test_outcome_from_decisions():
    assert outcome_from_decisions({}) == "false_positive"
    assert outcome_from_decisions({"nike": _decision("Nike", env=(True, 1))}) == "labeled"


# --- compare_article ---


def test_compare_shared_brand_full_agreement_has_no_disagreement():
    base = {"nike": _decision("Nike", env=(True, 1))}
    cand = {"nike": _decision("Nike", env=(True, 1))}
    comp = compare_article("a1", "t", base, cand)
    assert comp.outcome_match
    assert not comp.has_disagreement
    assert comp.brand_comparisons[0].status == "shared"


def test_compare_detects_dropped_and_added_brands():
    base = {"nike": _decision("Nike", env=(True, 1))}
    cand = {"adidas": _decision("Adidas", soc=(True, 0))}
    comp = compare_article("a1", "t", base, cand)
    statuses = {bc.brand: bc.status for bc in comp.brand_comparisons}
    assert statuses == {"Nike": "dropped", "Adidas": "added"}
    assert comp.has_disagreement


def test_compare_detects_outcome_flip():
    base: dict[str, BrandDecision] = {}  # baseline false_positive
    cand = {"nike": _decision("Nike", env=(True, 1))}  # candidate labeled
    comp = compare_article("a1", "t", base, cand)
    assert comp.baseline_outcome == "false_positive"
    assert comp.candidate_outcome == "labeled"
    assert not comp.outcome_match


def test_compare_detects_sentiment_disagreement():
    base = {"nike": _decision("Nike", env=(True, 1))}
    cand = {"nike": _decision("Nike", env=(True, -1))}
    comp = compare_article("a1", "t", base, cand)
    assert comp.outcome_match  # both labeled
    assert comp.has_disagreement  # sentiment differs
    env_diff = next(
        d for d in comp.brand_comparisons[0].category_diffs if d.category == "environmental"
    )
    assert env_diff.applies_match
    assert not env_diff.sentiment_match


# --- build_verdict ---


def test_verdict_hard_fails_on_parse_failure():
    comps = [
        compare_article("a1", "t", {}, {}, parse_ok=False, error="bad json"),
        compare_article("a2", "t", {"nike": _decision("Nike", env=(True, 1))}, {"nike": _decision("Nike", env=(True, 1))}),
    ]
    v = build_verdict(comps)
    assert v.n_parse_failures == 1
    assert v.n_compared == 1  # parse failures excluded from scored set
    assert v.hard_fail


def test_verdict_clean_run_passes_advisory():
    comps = [
        compare_article(f"a{i}", "t", {"nike": _decision("Nike", env=(True, 1))}, {"nike": _decision("Nike", env=(True, 1))})
        for i in range(10)
    ]
    v = build_verdict(comps)
    assert v.passed_advisory
    assert v.outcome_disagreement_rate == 0.0
    assert v.category_f1_macro == 1.0
    assert v.sentiment_exact_rate == 1.0


def test_verdict_flags_outcome_disagreement_and_inflation():
    # 8 agree (false_positive), 2 flip fp->labeled (newly labeled).
    comps = [compare_article(f"f{i}", "t", {}, {}) for i in range(8)]
    comps += [
        compare_article(f"x{i}", "t", {}, {"nike": _decision("Nike", env=(True, 1))})
        for i in range(2)
    ]
    v = build_verdict(comps, Thresholds(max_outcome_disagreement=0.10, max_newly_labeled_rate=0.10))
    assert v.outcome_confusion["fl"] == 2
    assert v.outcome_confusion["ff"] == 8
    assert v.outcome_disagreement_rate == 0.2
    assert v.newly_labeled_rate == 0.2
    assert any("outcome disagreement" in f for f in v.flags)
    assert any("scorecard-inflation" in f for f in v.flags)
    assert not v.hard_fail


def test_verdict_category_confusion_counts():
    # baseline labels env, candidate labels social -> 1 FN env, 1 FP social.
    base = {"nike": _decision("Nike", env=(True, 1))}
    cand = {"nike": _decision("Nike", soc=(True, 1))}
    v = build_verdict([compare_article("a1", "t", base, cand)])
    assert v.category_metrics["environmental"]["fn"] == 1
    assert v.category_metrics["social"]["fp"] == 1


def test_verdict_tracks_human_anchor_disagreements():
    comps = [
        compare_article("a1", "t", {"nike": _decision("Nike", env=(True, 1))}, {"nike": _decision("Nike", env=(True, -1))}, is_human_anchor=True),
        compare_article("a2", "t", {"nike": _decision("Nike", env=(True, 1))}, {"nike": _decision("Nike", env=(True, 1))}, is_human_anchor=True),
    ]
    v = build_verdict(comps)
    assert v.n_human_anchors == 2
    assert v.n_human_anchor_disagreements == 1
