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


class TestArticleLabelerEscapeInteriorQuotes:
    """Tests for repairing unescaped quotes inside string values (issue #82)."""

    @pytest.fixture
    def labeler(self):
        with patch("src.labeling.labeler.Anthropic"):
            return ArticleLabeler(api_key="test-key")

    def test_escapes_quote_followed_by_text(self, labeler):
        """A quote mid-string should be escaped, not treated as a terminator."""
        broken = '{"evidence": ["I think it will slow," Morningstar said."]}'
        data = json.loads(labeler._escape_interior_quotes(broken))
        assert data["evidence"] == ['I think it will slow," Morningstar said.']

    def test_real_response_from_20260905(self, labeler):
        """Regression: the exact excerpt that failed article e6ae677f."""
        broken = (
            '{"evidence": ['
            '"Lululemon shares tumbled 18% to an eight-year low.",'
            '"I think the store expansion will slow down," Morningstar analyst '
            "David Swartz said, adding that cost cuts, management changes and a "
            "possible operational 'realignment' can be expected.\"]}"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(broken)

        data = json.loads(labeler._escape_interior_quotes(broken))
        assert len(data["evidence"]) == 2
        assert data["evidence"][1].startswith('I think the store expansion will slow down,"')

    def test_valid_json_is_unchanged(self, labeler):
        """Well-formed JSON must survive the scanner byte-for-byte."""
        valid = '{"a": "x", "b": ["y", "z"], "c": {"d": 1}, "e": null}'
        assert labeler._escape_interior_quotes(valid) == valid

    def test_already_escaped_quotes_are_preserved(self, labeler):
        """An existing \\" must not be double-escaped."""
        valid = '{"quote": "He said \\"hello\\" once."}'
        result = labeler._escape_interior_quotes(valid)
        assert json.loads(result)["quote"] == 'He said "hello" once.'

    def test_backslash_escapes_preserved(self, labeler):
        """Newline and backslash escapes pass through intact."""
        valid = '{"path": "C:\\\\tmp", "text": "line1\\nline2"}'
        data = json.loads(labeler._escape_interior_quotes(valid))
        assert data["path"] == "C:\\tmp"
        assert data["text"] == "line1\nline2"

    def test_closing_quote_before_each_structural_char(self, labeler):
        """Quotes followed by , } ] : all read as terminators."""
        valid = '{"k": "v", "arr": ["a"], "obj": {"n": "m"}}'
        assert labeler._escape_interior_quotes(valid) == valid

    def test_quote_at_end_of_input(self, labeler):
        """A trailing quote with nothing after it closes the string."""
        assert labeler._escape_interior_quotes('"tail"') == '"tail"'

    def test_known_limit_interior_quote_before_comma(self, labeler):
        """Documents the heuristic's blind spot: still unparseable, never wrong.

        `"yes",` reads as a terminator, so this stays broken and the caller
        returns None — the same outcome as before the repair existed.
        """
        broken = '{"evidence": ["He said "yes", loudly."]}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(labeler._escape_interior_quotes(broken))

    def test_known_limit_interior_quote_before_colon(self, labeler):
        """The colon blind spot fails closed the same way a comma does."""
        broken = '{"evidence": ["the report titled "Impact 2030": a review"]}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(labeler._escape_interior_quotes(broken))

    def test_dropped_comma_between_elements_fails_closed(self, labeler):
        """A missing comma must not merge two excerpts into one corrupted string.

        Regression for the fail-open case: with `"` absent from
        STRUCTURAL_AFTER_STRING both inner quotes read as interior text, the two
        array elements collapse into a single string, and the result *validates*
        — sending a garbled excerpt to the public feed. It must stay unparseable.
        """
        broken = (
            '{"evidence": ["Nike cut emissions by 30%." '
            '"The company sourced 96% recycled polyester."]}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(labeler._escape_interior_quotes(broken))
        assert labeler._recover_json(broken) is None

    def test_dropped_comma_across_a_newline_fails_closed(self, labeler):
        """Same case with the elements on separate lines, as a model would emit."""
        broken = '{"evidence": [\n  "first excerpt."\n  "second excerpt."\n]}'
        assert labeler._recover_json(broken) is None

    def test_quote_terminator_does_not_break_the_motivating_case(self, labeler):
        """Adding `"` as a terminator must not cost us the bug we set out to fix."""
        broken = (
            '{"evidence": ["Lululemon shares tumbled 18%.",'
            '"I think the store expansion will slow down," Morningstar analyst '
            'David Swartz said."]}'
        )
        data = labeler._recover_json(broken)
        assert data is not None
        assert len(data["evidence"]) == 2
        assert data["evidence"][1].startswith('I think the store expansion will slow down,"')


class TestRecoverJson:
    """Tests for the staged repair ladder."""

    @pytest.fixture
    def labeler(self):
        with patch("src.labeling.labeler.Anthropic"):
            return ArticleLabeler(api_key="test-key")

    def test_prefers_the_least_invasive_repair(self, labeler):
        """Quote escaping alone should win before the non-string-aware regexes run.

        `_fix_json`'s unquoted-key pattern matches `, <identifier>:` anywhere,
        including inside a string value, and rewrites it to `,"<identifier>":`.
        On this input that mangling makes the document unparseable, so the
        ladder has to stop at the escape-only candidate to recover it at all.
        """
        broken = '{"evidence": ["Q1," he said, revenue: up 5 percent."]}'

        # Escaping alone succeeds; the full repair destroys the same document.
        assert json.loads(labeler._escape_interior_quotes(broken)) is not None
        with pytest.raises(json.JSONDecodeError):
            json.loads(labeler._fix_json(broken))

        assert labeler._recover_json(broken) == {
            "evidence": ['Q1," he said, revenue: up 5 percent.']
        }

    def test_falls_through_to_full_repair(self, labeler):
        """A trailing comma needs the regex pass, which escaping alone won't fix."""
        broken = '{"evidence": ["a", "b",]}'
        assert labeler._recover_json(broken) == {"evidence": ["a", "b"]}

    def test_returns_none_when_nothing_helps(self, labeler):
        assert labeler._recover_json('{"evidence": [') is None

    def test_logs_original_context_on_first_failure(self, labeler, caplog):
        """The first diagnostic must describe the model's output, not the repair.

        Repairs shift every offset after the edit, so a window taken from the
        repaired text can point at parser-introduced damage (issue #82).
        """
        broken = '{"evidence": ["I think it will slow," an analyst said."]}'

        with caplog.at_level("WARNING", logger="src.labeling.labeler"):
            labeler._parse_response(broken)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("JSON parse error" in r.message for r in warnings)
        assert any("Context:" in r.message for r in warnings)
        # The window is cut from the unrepaired text, so it has no backslashes.
        context_line = next(r.message for r in warnings if "Context:" in r.message)
        assert "\\\\\"" not in context_line

    def test_fix_json_applies_the_repair(self, labeler):
        """_fix_json wires the scanner in ahead of its regex passes."""
        broken = '{"evidence": ["a," b."],}'
        assert json.loads(labeler._fix_json(broken))["evidence"] == ['a," b.']

    def test_parse_response_recovers_fenced_payload(self, labeler):
        """End to end: fenced JSON with a bare quote parses into the model."""
        response = (
            "```json\n"
            "{\n"
            '  "brand_analyses": [{\n'
            '    "brand": "Lululemon",\n'
            '    "is_sportswear_brand": true,\n'
            '    "categories": {\n'
            '      "environmental": {"applies": false, "sentiment": null, "evidence": []},\n'
            '      "social": {"applies": false, "sentiment": null, "evidence": []},\n'
            '      "governance": {"applies": true, "sentiment": -1, "evidence": [\n'
            '        "I think it will slow," an analyst said."\n'
            "      ]},\n"
            '      "digital_transformation": '
            '{"applies": false, "sentiment": null, "evidence": []}\n'
            "    },\n"
            '    "confidence": 0.9,\n'
            '    "reasoning": "Leadership transition."\n'
            "  }],\n"
            '  "article_summary": "Stock fell."\n'
            "}\n"
            "```\n\n"
            "**Note on Nike:** mentioned only as a comparison."
        )
        parsed = labeler._parse_response(response)
        assert parsed is not None
        governance = parsed.brand_analyses[0].categories["governance"]
        assert governance.evidence == ['I think it will slow," an analyst said.']


class TestJsonErrorContext:
    """Tests for the parse-failure diagnostic helper."""

    def test_returns_window_around_position(self):
        from src.labeling.labeler import _json_error_context

        result = _json_error_context("abcdefghij", pos=5, window=2)
        assert result == repr("defg")

    def test_clamps_at_boundaries(self):
        from src.labeling.labeler import _json_error_context

        assert _json_error_context("abc", pos=0, window=50) == repr("abc")


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

            # Claude Haiku 4.5 — Input: $1.00 per 1M, Output: $5.00 per 1M
            expected_cost = 1.00 + 5.00
            assert abs(stats["estimated_cost_usd"] - expected_cost) < 0.01



class TestClassifyApiError:
    """Tests for API error classification."""

    def test_authentication_error(self):
        """Should classify authentication errors."""
        from anthropic import AuthenticationError
        from src.labeling.labeler import classify_api_error

        # Create a mock error that passes isinstance check
        class MockAuthError(AuthenticationError):
            def __init__(self):
                pass  # Skip parent __init__

            def __str__(self):
                return "Invalid API key"

        error = MockAuthError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "authentication"
        assert "authentication" in error_msg.lower()

    def test_rate_limit_error(self):
        """Should classify rate limit errors."""
        from anthropic import RateLimitError
        from src.labeling.labeler import classify_api_error

        # Create a mock error that passes isinstance check
        class MockRateLimitError(RateLimitError):
            def __init__(self):
                pass  # Skip parent __init__

            def __str__(self):
                return "Rate limit exceeded"

        error = MockRateLimitError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "rate_limit"
        assert "rate limit" in error_msg.lower()

    def test_timeout_error(self):
        """Should classify timeout errors."""
        from anthropic import APITimeoutError
        from src.labeling.labeler import classify_api_error

        # Create a mock error that passes isinstance check
        class MockTimeoutError(APITimeoutError):
            def __init__(self):
                pass  # Skip parent __init__

            def __str__(self):
                return "Request timed out"

        error = MockTimeoutError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "timeout"
        assert "timed out" in error_msg.lower()

    def test_connection_error(self):
        """Should classify connection errors."""
        from anthropic import APIConnectionError
        from src.labeling.labeler import classify_api_error

        # Create a mock error that passes isinstance check
        class MockConnectionError(APIConnectionError):
            def __init__(self):
                pass  # Skip parent __init__

            def __str__(self):
                return "Connection failed"

        error = MockConnectionError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "connection"
        assert "connection" in error_msg.lower()

    def test_connection_error_dns_failure(self):
        """Should detect DNS resolution failures."""
        from anthropic import APIConnectionError
        from src.labeling.labeler import classify_api_error

        # Create a mock error with DNS-related message
        class MockConnectionError(APIConnectionError):
            def __init__(self):
                pass

            def __str__(self):
                return "Name or service not known"

        error = MockConnectionError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "connection"
        assert "dns" in error_msg.lower()

    def test_connection_error_network_unreachable(self):
        """Should detect network unreachable errors."""
        from anthropic import APIConnectionError
        from src.labeling.labeler import classify_api_error

        class MockConnectionError(APIConnectionError):
            def __init__(self):
                pass

            def __str__(self):
                return "Network is unreachable"

        error = MockConnectionError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "connection"
        assert "internet outage" in error_msg.lower()

    def test_server_error(self):
        """Should classify internal server errors."""
        from anthropic import InternalServerError
        from src.labeling.labeler import classify_api_error

        # Create a mock error that passes isinstance check
        class MockServerError(InternalServerError):
            def __init__(self):
                pass  # Skip parent __init__

            def __str__(self):
                return "Internal server error"

        error = MockServerError()
        error_type, error_msg = classify_api_error(error)

        assert error_type == "server_error"
        assert "500" in error_msg or "server" in error_msg.lower()

    def test_unknown_error(self):
        """Should classify unknown errors."""
        from src.labeling.labeler import classify_api_error

        error = ValueError("Some unknown error")
        error_type, error_msg = classify_api_error(error)

        assert error_type == "unknown"
        assert "ValueError" in error_msg

    def test_network_keyword_in_generic_error(self):
        """Should detect network keywords in generic errors."""
        from src.labeling.labeler import classify_api_error

        error = Exception("Socket connection timeout occurred")
        error_type, error_msg = classify_api_error(error)

        assert error_type == "connection"
        assert "network" in error_msg.lower()
