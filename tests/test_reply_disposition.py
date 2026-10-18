"""Tests for the article-level disposition of a labeling reply (#192, D032)."""

from src.data_collection import models
from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED, dispose_reply
from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse, normalize_brand

CATEGORIES = ("environmental", "social", "governance", "digital_transformation")


def _analysis(brand, sportswear=True, esg=False, reason=None):
    if sportswear:
        categories = {
            c: CategoryLabel(
                applies=esg and c == "environmental",
                sentiment=1 if esg and c == "environmental" else None,
                evidence=["quote"] if esg and c == "environmental" else [],
            )
            for c in CATEGORIES
        }
    else:
        categories = {}
    return BrandAnalysis(
        brand=brand,
        is_sportswear_brand=sportswear,
        not_sportswear_reason=None if sportswear else (reason or "not sportswear"),
        categories=categories,
        confidence=0.9,
        reasoning="r",
    )


class TestDisposeReply:
    def test_empty_reply_is_unjudged(self):
        d = dispose_reply(["Nike", "Puma"], [])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_NO_ESG_CONTENT)
        assert d.omitted == ["Nike", "Puma"]

    def test_sportswear_brand_with_esg_is_labeled(self):
        d = dispose_reply(["Nike"], [_analysis("Nike", esg=True)])
        assert (d.status, d.reason) == (LABELED, None)
        assert [a.brand for a in d.sportswear] == ["Nike"]

    def test_labeled_even_when_another_brand_is_left_out(self):
        d = dispose_reply(["Nike", "Puma"], [_analysis("Nike", esg=True)])
        assert d.status == LABELED
        assert d.omitted == ["Puma"]

    def test_sportswear_without_esg_is_a_judged_skip(self):
        d = dispose_reply(["Nike"], [_analysis("Nike")])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_NO_ESG_CATEGORIES)
        assert models.SKIP_NO_ESG_CATEGORIES in models.JUDGED_SKIP_REASONS

    def test_sportswear_without_esg_and_a_brand_left_out_is_still_judged(self):
        """One sportswear judgment makes the article a positive, whatever was left out."""
        d = dispose_reply(["Nike", "Puma"], [_analysis("Nike")])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_NO_ESG_CATEGORIES)

    def test_all_non_sportswear_covering_every_brand_is_false_positive(self):
        d = dispose_reply(
            ["Puma", "Vans"],
            [_analysis("Puma", sportswear=False, reason="the animal"),
             _analysis("Vans", sportswear=False, reason="vehicles")],
        )
        assert d.status == FALSE_POSITIVE
        assert d.reason == "Puma: the animal; Vans: vehicles"
        assert d.omitted == []

    def test_all_non_sportswear_with_a_brand_left_out_is_unjudged(self):
        """The reply does not say what the left-out brand was (D032): under v1.10.0 it may be a
        sportswear brand with no ESG content, which that version tells the model to omit."""
        d = dispose_reply(["Puma", "Nike"], [_analysis("Puma", sportswear=False)])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_INCOMPLETE_JUDGMENT)
        assert d.omitted == ["Nike"]
        assert models.SKIP_INCOMPLETE_JUDGMENT in models.UNJUDGED_SKIP_REASONS

    def test_names_match_ignoring_case_and_surrounding_space(self):
        d = dispose_reply(["Under Armour"], [_analysis(" under armour ", sportswear=False)])
        assert d.status == FALSE_POSITIVE
        assert d.omitted == [] and d.unsent == []

    def test_a_differently_spelled_name_counts_as_left_out(self):
        """No alias table: a name miss fails safe to unjudged."""
        d = dispose_reply(["lululemon"], [_analysis("Lululemon Athletica", sportswear=False)])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_INCOMPLETE_JUDGMENT)
        assert d.omitted == ["lululemon"]
        assert d.unsent == ["Lululemon Athletica"]

    def test_unsent_brand_verdict_still_counts(self):
        d = dispose_reply(["Puma"], [_analysis("Puma", sportswear=False), _analysis("Nike", esg=True)])
        assert d.status == LABELED
        assert d.unsent == ["Nike"]

    def test_conflicting_verdicts_sportswear_wins(self):
        d = dispose_reply(["Puma"], [_analysis("Puma", sportswear=False), _analysis("PUMA", esg=True)])
        assert d.status == LABELED
        assert d.conflicting == ["Puma"]

    def test_false_positive_reason_is_truncated(self):
        d = dispose_reply(["Puma"], [_analysis("Puma", sportswear=False, reason="x" * 600)])
        assert d.status == FALSE_POSITIVE
        assert len(d.reason) == 500


    def test_one_sportswear_brand_with_esg_labels_the_article(self):
        """v1.11.0 returns every sportswear brand, most with no ESG category."""
        d = dispose_reply(["Adidas", "Nike"], [_analysis("Adidas"), _analysis("Nike", esg=True)])
        assert (d.status, d.reason) == (LABELED, None)
        assert d.omitted == []

    def test_sportswear_with_empty_categories_is_a_judged_skip(self):
        """The shape v1.11.0 asks for when no ESG category applies."""
        nike = BrandAnalysis(
            brand="Nike", is_sportswear_brand=True, categories={}, confidence=0.9, reasoning="r"
        )
        d = dispose_reply(["Nike"], [nike])
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_NO_ESG_CATEGORIES)

    def test_analysis_without_a_verdict_counts_as_left_out(self):
        """is_sportswear_brand defaults to True; a parsed entry that never set it is no judgment."""
        response = LabelingResponse.model_validate({
            "brand_analyses": [{"brand": "Nike", "categories": {}, "confidence": 0.9, "reasoning": "r"}],
            "article_summary": "s",
        })
        assert response.brand_analyses[0].is_sportswear_brand is True
        d = dispose_reply(["Nike"], response.brand_analyses)
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_NO_ESG_CONTENT)
        assert d.omitted == ["Nike"] and d.unset == ["Nike"]

    def test_analysis_without_a_verdict_beside_a_non_sportswear_one_is_unjudged(self):
        response = LabelingResponse.model_validate({
            "brand_analyses": [
                {"brand": "Puma", "is_sportswear_brand": False, "not_sportswear_reason": "the animal",
                 "categories": {}, "confidence": 0.9, "reasoning": "r"},
                {"brand": "Nike", "categories": {}, "confidence": 0.9, "reasoning": "r"},
            ],
            "article_summary": "s",
        })
        d = dispose_reply(["Puma", "Nike"], response.brand_analyses)
        assert (d.status, d.reason) == (SKIPPED, models.SKIP_INCOMPLETE_JUDGMENT)
        assert d.omitted == ["Nike"]

    def test_parsed_analysis_with_a_category_but_no_verdict_still_labels(self):
        """An applicable category is a verdict, as before #192."""
        category = {"applies": True, "sentiment": 1, "evidence": ["q"]}
        empty = {"applies": False, "sentiment": None, "evidence": []}
        response = LabelingResponse.model_validate({
            "brand_analyses": [{
                "brand": "Nike",
                "categories": {"environmental": category, "social": empty,
                               "governance": empty, "digital_transformation": empty},
                "confidence": 0.9, "reasoning": "r",
            }],
            "article_summary": "s",
        })
        assert dispose_reply(["Nike"], response.brand_analyses).status == LABELED


class TestNormalizeBrand:
    def test_strips_and_casefolds(self):
        assert normalize_brand("  ASICS ") == "asics"
        # casefold, not lower: "ß" folds to "ss"
        assert normalize_brand("STRASSE") == normalize_brand("straße")

    def test_get_analysis_for_brand_uses_it(self):
        response = LabelingResponse(brand_analyses=[_analysis("New Balance ")], article_summary="s")
        assert response.get_analysis_for_brand("new balance") is not None
