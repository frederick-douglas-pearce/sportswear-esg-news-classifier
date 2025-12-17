"""Tests for the article labeler and response parsing."""

import json

import pytest

from src.labeling.models import BrandAnalysis, CategoryLabel, LabelingResponse


class TestCategoryLabel:
    """Tests for CategoryLabel model."""

    def test_category_label_applies_true(self):
        """Should create category label with applies=True."""
        label = CategoryLabel(
            applies=True,
            sentiment=1,
            evidence=["Quote from article"],
        )
        assert label.applies is True
        assert label.sentiment == 1
        assert len(label.evidence) == 1

    def test_category_label_applies_false(self):
        """Should create category label with applies=False."""
        label = CategoryLabel(
            applies=False,
            sentiment=None,
            evidence=[],
        )
        assert label.applies is False
        assert label.sentiment is None
        assert len(label.evidence) == 0

    def test_category_label_negative_sentiment(self):
        """Should accept negative sentiment."""
        label = CategoryLabel(applies=True, sentiment=-1, evidence=[])
        assert label.sentiment == -1

    def test_category_label_neutral_sentiment(self):
        """Should accept neutral sentiment."""
        label = CategoryLabel(applies=True, sentiment=0, evidence=[])
        assert label.sentiment == 0

    def test_category_label_multiple_evidence(self):
        """Should accept multiple evidence quotes."""
        evidence = ["Quote 1", "Quote 2", "Quote 3"]
        label = CategoryLabel(applies=True, sentiment=1, evidence=evidence)
        assert len(label.evidence) == 3


class TestBrandAnalysis:
    """Tests for BrandAnalysis model."""

    def test_brand_analysis_complete(self):
        """Should create complete brand analysis."""
        analysis = BrandAnalysis(
            brand="Nike",
            categories={
                "environmental": CategoryLabel(
                    applies=True, sentiment=1, evidence=["Nike leads in sustainability"]
                ),
                "social": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "digital_transformation": CategoryLabel(
                    applies=False, sentiment=None, evidence=[]
                ),
            },
            confidence=0.85,
            reasoning="Article focuses on Nike's environmental initiatives",
        )

        assert analysis.brand == "Nike"
        assert analysis.confidence == 0.85
        assert analysis.categories["environmental"].applies is True

    def test_brand_analysis_get_applicable_categories(self):
        """Should return list of applicable categories."""
        analysis = BrandAnalysis(
            brand="Nike",
            categories={
                "environmental": CategoryLabel(
                    applies=True, sentiment=1, evidence=["Quote"]
                ),
                "social": CategoryLabel(applies=True, sentiment=0, evidence=["Quote"]),
                "governance": CategoryLabel(applies=False, sentiment=None, evidence=[]),
                "digital_transformation": CategoryLabel(
                    applies=False, sentiment=None, evidence=[]
                ),
            },
            confidence=0.9,
            reasoning="Test",
        )

        applicable = analysis.get_applicable_categories()
        assert "environmental" in applicable
        assert "social" in applicable
        assert "governance" not in applicable
        assert len(applicable) == 2

    def test_brand_analysis_missing_category_raises_error(self):
        """Should raise error when required category is missing."""
        with pytest.raises(ValueError, match="Missing required categories"):
            BrandAnalysis(
                brand="Nike",
                categories={
                    "environmental": CategoryLabel(
                        applies=True, sentiment=1, evidence=[]
                    ),
                    # Missing social, governance, digital_transformation
                },
                confidence=0.8,
                reasoning="Test",
            )


class TestLabelingResponse:
    """Tests for LabelingResponse model."""

    def test_labeling_response_complete(self):
        """Should create complete labeling response."""
        response = LabelingResponse(
            brand_analyses=[
                BrandAnalysis(
                    brand="Nike",
                    categories={
                        "environmental": CategoryLabel(
                            applies=True, sentiment=1, evidence=["Quote"]
                        ),
                        "social": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "governance": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "digital_transformation": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                    },
                    confidence=0.9,
                    reasoning="Test",
                )
            ],
            article_summary="Article about Nike sustainability",
        )

        assert len(response.brand_analyses) == 1
        assert response.article_summary == "Article about Nike sustainability"

    def test_labeling_response_get_brands(self):
        """Should return list of brands."""
        response = LabelingResponse(
            brand_analyses=[
                BrandAnalysis(
                    brand="Nike",
                    categories={
                        "environmental": CategoryLabel(
                            applies=True, sentiment=1, evidence=[]
                        ),
                        "social": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "governance": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "digital_transformation": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                    },
                    confidence=0.9,
                    reasoning="Test",
                ),
                BrandAnalysis(
                    brand="Adidas",
                    categories={
                        "environmental": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "social": CategoryLabel(
                            applies=True, sentiment=-1, evidence=[]
                        ),
                        "governance": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "digital_transformation": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                    },
                    confidence=0.8,
                    reasoning="Test",
                ),
            ],
            article_summary="Test",
        )

        brands = response.get_brands()
        assert "Nike" in brands
        assert "Adidas" in brands
        assert len(brands) == 2

    def test_labeling_response_get_analysis_for_brand(self):
        """Should return analysis for specific brand."""
        response = LabelingResponse(
            brand_analyses=[
                BrandAnalysis(
                    brand="Nike",
                    categories={
                        "environmental": CategoryLabel(
                            applies=True, sentiment=1, evidence=[]
                        ),
                        "social": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "governance": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                        "digital_transformation": CategoryLabel(
                            applies=False, sentiment=None, evidence=[]
                        ),
                    },
                    confidence=0.9,
                    reasoning="Test",
                )
            ],
            article_summary="Test",
        )

        nike_analysis = response.get_analysis_for_brand("Nike")
        assert nike_analysis is not None
        assert nike_analysis.brand == "Nike"

        # Case insensitive
        nike_analysis_lower = response.get_analysis_for_brand("nike")
        assert nike_analysis_lower is not None

        # Non-existent brand
        adidas_analysis = response.get_analysis_for_brand("Adidas")
        assert adidas_analysis is None


class TestResponseParsing:
    """Tests for parsing JSON responses into models."""

    def test_parse_valid_json_response(self):
        """Should parse valid JSON response."""
        json_data = {
            "brand_analyses": [
                {
                    "brand": "Nike",
                    "categories": {
                        "environmental": {
                            "applies": True,
                            "sentiment": 1,
                            "evidence": ["Nike announced carbon neutrality goals"],
                        },
                        "social": {"applies": False, "sentiment": None, "evidence": []},
                        "governance": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                        "digital_transformation": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                    },
                    "confidence": 0.85,
                    "reasoning": "Article discusses Nike's sustainability initiatives",
                }
            ],
            "article_summary": "Nike announces new environmental goals",
        }

        response = LabelingResponse.model_validate(json_data)

        assert len(response.brand_analyses) == 1
        assert response.brand_analyses[0].brand == "Nike"
        assert response.brand_analyses[0].categories["environmental"].applies is True
        assert response.brand_analyses[0].categories["environmental"].sentiment == 1

    def test_parse_multiple_brands(self):
        """Should parse response with multiple brands."""
        json_data = {
            "brand_analyses": [
                {
                    "brand": "Nike",
                    "categories": {
                        "environmental": {
                            "applies": True,
                            "sentiment": 1,
                            "evidence": [],
                        },
                        "social": {"applies": False, "sentiment": None, "evidence": []},
                        "governance": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                        "digital_transformation": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                    },
                    "confidence": 0.9,
                    "reasoning": "Test",
                },
                {
                    "brand": "Adidas",
                    "categories": {
                        "environmental": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                        "social": {"applies": True, "sentiment": -1, "evidence": []},
                        "governance": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                        "digital_transformation": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                    },
                    "confidence": 0.75,
                    "reasoning": "Test",
                },
            ],
            "article_summary": "Article about Nike and Adidas",
        }

        response = LabelingResponse.model_validate(json_data)

        assert len(response.brand_analyses) == 2
        nike = response.get_analysis_for_brand("Nike")
        adidas = response.get_analysis_for_brand("Adidas")
        assert nike.categories["environmental"].applies is True
        assert adidas.categories["social"].applies is True
        assert adidas.categories["social"].sentiment == -1

    def test_parse_empty_brand_analyses(self):
        """Should handle empty brand analyses list."""
        json_data = {"brand_analyses": [], "article_summary": "No ESG content found"}

        response = LabelingResponse.model_validate(json_data)

        assert len(response.brand_analyses) == 0
        assert response.get_brands() == []


class TestEvidenceExtraction:
    """Tests for evidence extraction from parsed responses."""

    def test_evidence_preserved_in_parsing(self):
        """Evidence quotes should be preserved."""
        evidence = [
            "Nike has committed to reducing carbon emissions by 50%",
            "The company plans to use 100% renewable energy",
        ]
        json_data = {
            "brand_analyses": [
                {
                    "brand": "Nike",
                    "categories": {
                        "environmental": {
                            "applies": True,
                            "sentiment": 1,
                            "evidence": evidence,
                        },
                        "social": {"applies": False, "sentiment": None, "evidence": []},
                        "governance": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                        "digital_transformation": {
                            "applies": False,
                            "sentiment": None,
                            "evidence": [],
                        },
                    },
                    "confidence": 0.9,
                    "reasoning": "Test",
                }
            ],
            "article_summary": "Test",
        }

        response = LabelingResponse.model_validate(json_data)

        parsed_evidence = response.brand_analyses[0].categories["environmental"].evidence
        assert len(parsed_evidence) == 2
        assert "carbon emissions" in parsed_evidence[0]
        assert "renewable energy" in parsed_evidence[1]
