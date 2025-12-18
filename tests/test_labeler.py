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


# Tests for ArticleLabeler class
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.labeling.labeler import ArticleLabeler, LabelingResult


class TestLabelingResultDataclass:
    """Tests for LabelingResult dataclass."""

    def test_labeling_result_success(self):
        """Should create successful labeling result."""
        response = LabelingResponse(
            brand_analyses=[],
            article_summary="Test",
        )
        result = LabelingResult(
            success=True,
            response=response,
            input_tokens=100,
            output_tokens=50,
            model="claude-3-sonnet",
        )
        assert result.success is True
        assert result.response is not None
        assert result.error is None

    def test_labeling_result_failure(self):
        """Should create failed labeling result."""
        result = LabelingResult(
            success=False,
            error="API error",
            model="claude-3-sonnet",
        )
        assert result.success is False
        assert result.error == "API error"
        assert result.response is None


class TestArticleLabelerInit:
    """Tests for ArticleLabeler initialization."""

    def test_missing_api_key_raises_error(self):
        """Should raise error when API key is missing."""
        with patch("src.labeling.labeler.labeling_settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            mock_settings.labeling_model = "test-model"

            with pytest.raises(ValueError, match="Anthropic API key is required"):
                ArticleLabeler()

    def test_custom_parameters(self):
        """Should use custom parameters when provided."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(
                api_key="test-key",
                model="custom-model",
                max_retries=5,
                retry_delay=2.0,
                max_tokens=3000,
            )
            assert labeler.api_key == "test-key"
            assert labeler.model == "custom-model"
            assert labeler.max_retries == 5
            assert labeler.retry_delay == 2.0
            assert labeler.max_tokens == 3000


class TestArticleLabelerLabelArticle:
    """Tests for label_article method."""

    def test_label_article_no_content(self):
        """Should return error for empty content."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            result = labeler.label_article(
                title="Test",
                content="",
                brands=["Nike"],
            )
            assert result.success is False
            assert "No content" in result.error

    def test_label_article_no_brands(self):
        """Should return error for empty brands list."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            result = labeler.label_article(
                title="Test",
                content="Test content here",
                brands=[],
            )
            assert result.success is False
            assert "No brands" in result.error

    def test_label_article_success(self):
        """Should successfully label article."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "brand_analyses": [
                            {
                                "brand": "Nike",
                                "is_sportswear_brand": True,
                                "categories": {
                                    "environmental": {
                                        "applies": True,
                                        "sentiment": 1,
                                        "evidence": ["Test"],
                                    },
                                    "social": {
                                        "applies": False,
                                        "sentiment": None,
                                        "evidence": [],
                                    },
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
                        "article_summary": "Test summary",
                    }
                )
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        with patch("src.labeling.labeler.Anthropic", return_value=mock_client):
            labeler = ArticleLabeler(api_key="test-key")
            result = labeler.label_article(
                title="Test Article",
                content="Nike announced sustainability goals " * 20,
                brands=["Nike"],
                published_at=datetime.now(),
                source_name="Test Source",
            )

            assert result.success is True
            assert result.response is not None
            assert len(result.response.brand_analyses) == 1
            assert result.input_tokens == 100
            assert result.output_tokens == 50


class TestArticleLabelerTruncateContent:
    """Tests for content truncation."""

    def test_truncate_short_content(self):
        """Should not truncate short content."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            content = "Short content."
            result = labeler._truncate_content(content, max_tokens=1000)
            assert result == content

    def test_truncate_long_content(self):
        """Should truncate long content."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            content = "A" * 10000  # Very long content
            result = labeler._truncate_content(content, max_tokens=100)

            # Should be truncated (100 tokens * 4 chars = 400 chars max)
            assert len(result) < len(content)
            assert "[Content truncated...]" in result

    def test_truncate_at_sentence_boundary(self):
        """Should truncate at sentence boundary when possible."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            # Create content with sentences
            content = "First sentence. " * 100
            result = labeler._truncate_content(content, max_tokens=50)

            # Should end with a period followed by truncation notice
            assert "." in result
            assert "[Content truncated...]" in result


class TestArticleLabelerExtractJson:
    """Tests for JSON extraction from responses."""

    def test_extract_json_from_code_block(self):
        """Should extract JSON from markdown code block."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            text = '```json\n{"key": "value"}\n```'
            result = labeler._extract_json(text)
            assert result == '{"key": "value"}'

    def test_extract_json_raw(self):
        """Should extract raw JSON object."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            text = 'Here is the result: {"key": "value"}'
            result = labeler._extract_json(text)
            assert result == '{"key": "value"}'

    def test_extract_json_no_json(self):
        """Should return None when no JSON found."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            text = "No JSON here, just plain text"
            result = labeler._extract_json(text)
            assert result is None


class TestArticleLabelerFixJson:
    """Tests for JSON fixing."""

    def test_fix_trailing_comma(self):
        """Should remove trailing commas."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            json_str = '{"key": "value",}'
            result = labeler._fix_json(json_str)
            assert result == '{"key": "value"}'

    def test_fix_trailing_comma_in_array(self):
        """Should remove trailing commas in arrays."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            json_str = '{"arr": [1, 2, 3,]}'
            result = labeler._fix_json(json_str)
            assert result == '{"arr": [1, 2, 3]}'


class TestArticleLabelerStats:
    """Tests for statistics tracking."""

    def test_get_stats(self):
        """Should return usage statistics."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            labeler.total_input_tokens = 1000
            labeler.total_output_tokens = 500
            labeler.total_api_calls = 5

            stats = labeler.get_stats()

            assert stats["total_input_tokens"] == 1000
            assert stats["total_output_tokens"] == 500
            assert stats["total_api_calls"] == 5
            assert "estimated_cost_usd" in stats

    def test_reset_stats(self):
        """Should reset statistics."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            labeler.total_input_tokens = 1000
            labeler.total_output_tokens = 500
            labeler.total_api_calls = 5

            labeler.reset_stats()

            assert labeler.total_input_tokens == 0
            assert labeler.total_output_tokens == 0
            assert labeler.total_api_calls == 0

    def test_cost_estimation(self):
        """Should estimate cost correctly."""
        with patch("src.labeling.labeler.Anthropic"):
            labeler = ArticleLabeler(api_key="test-key")
            labeler.total_input_tokens = 1_000_000  # 1M tokens
            labeler.total_output_tokens = 1_000_000  # 1M tokens

            stats = labeler.get_stats()

            # Input: $3.00 per 1M, Output: $15.00 per 1M
            expected_cost = 3.00 + 15.00
            assert abs(stats["estimated_cost_usd"] - expected_cost) < 0.01
