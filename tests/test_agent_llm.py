"""Tests for agent LLM analysis module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.llm import (
    AnalysisResult,
    LABELING_ANALYSIS_SYSTEM_PROMPT,
    LABELING_ANALYSIS_USER_PROMPT,
    LabelingAnalyzer,
)


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_successful_result(self):
        """Should create successful result with analysis."""
        result = AnalysisResult(
            success=True,
            analysis={"summary": "Good quality labeling"},
            input_tokens=100,
            output_tokens=50,
            model="claude-sonnet-4-20250514",
        )

        assert result.success is True
        assert result.analysis == {"summary": "Good quality labeling"}
        assert result.error is None
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_failed_result(self):
        """Should create failed result with error."""
        result = AnalysisResult(
            success=False,
            error="API error occurred",
            model="claude-sonnet-4-20250514",
        )

        assert result.success is False
        assert result.analysis is None
        assert result.error == "API error occurred"

    def test_default_values(self):
        """Should have sensible defaults."""
        result = AnalysisResult(success=True)

        assert result.analysis is None
        assert result.error is None
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.model == ""


class TestLabelingAnalyzerInit:
    """Tests for LabelingAnalyzer initialization."""

    def test_requires_api_key(self):
        """Should raise error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.agent.llm.agent_settings") as mock_settings:
                mock_settings.anthropic_api_key = None
                with pytest.raises(ValueError, match="API key required"):
                    LabelingAnalyzer(api_key=None)

    def test_uses_provided_api_key(self):
        """Should use provided API key."""
        with patch("src.agent.llm.Anthropic") as mock_anthropic:
            analyzer = LabelingAnalyzer(api_key="test-key")
            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert analyzer.api_key == "test-key"

    def test_uses_custom_model(self):
        """Should use custom model when provided."""
        with patch("src.agent.llm.Anthropic"):
            analyzer = LabelingAnalyzer(api_key="test-key", model="custom-model")
            assert analyzer.model == "custom-model"

    def test_default_retry_settings(self):
        """Should have default retry settings."""
        with patch("src.agent.llm.Anthropic"):
            analyzer = LabelingAnalyzer(api_key="test-key")
            assert analyzer.max_retries == 3
            assert analyzer.retry_delay == 1.0


class TestFormatArticleSample:
    """Tests for _format_article_sample method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked client."""
        with patch("src.agent.llm.Anthropic"):
            return LabelingAnalyzer(api_key="test-key")

    def test_empty_list(self, analyzer):
        """Should return empty string for empty list."""
        result = analyzer._format_article_sample([], "labeled")
        assert result == ""

    def test_formats_labeled_article(self, analyzer):
        """Should format labeled article correctly."""
        articles = [
            {
                "id": "uuid-123",
                "title": "Nike Sustainability Report",
                "source_name": "Reuters",
                "brands": ["Nike"],
                "content": "Nike announces carbon goals...",
                "categories": ["environmental", "governance"],
            }
        ]

        result = analyzer._format_article_sample(articles, "labeled")

        assert "Article 1" in result
        assert "uuid-123" in result
        assert "Nike Sustainability Report" in result
        assert "Reuters" in result
        assert "Nike" in result
        assert "environmental" in result
        assert "governance" in result
        assert "carbon goals" in result

    def test_formats_false_positive_article(self, analyzer):
        """Should format false positive article correctly."""
        articles = [
            {
                "id": "uuid-456",
                "title": "Puma (Animal) Documentary",
                "source_name": "Nature",
                "brands": ["Puma"],
                "content": "A documentary about mountain lions...",
            }
        ]

        result = analyzer._format_article_sample(articles, "false_positive")

        assert "Article 1" in result
        assert "uuid-456" in result
        assert "Puma (Animal)" in result
        # Should not include categories for non-labeled articles

    def test_truncates_long_titles(self, analyzer):
        """Should truncate titles longer than 100 chars."""
        long_title = "A" * 150
        articles = [
            {
                "id": "uuid-789",
                "title": long_title,
                "source_name": "Source",
                "brands": [],
            }
        ]

        result = analyzer._format_article_sample(articles, "labeled")

        # Title should be truncated to 100 chars
        assert len(long_title) > 100
        assert long_title[:100] in result
        assert long_title not in result  # Full title should not appear

    def test_truncates_long_content(self, analyzer):
        """Should truncate content to 300 chars."""
        long_content = "B" * 500
        articles = [
            {
                "id": "uuid-abc",
                "title": "Title",
                "source_name": "Source",
                "brands": [],
                "content": long_content,
            }
        ]

        result = analyzer._format_article_sample(articles, "labeled")

        # Content snippet should be ~300 chars plus "..."
        assert "..." in result
        assert long_content not in result

    def test_limits_to_10_articles(self, analyzer):
        """Should limit output to 10 articles."""
        articles = [
            {
                "id": f"uuid-{i}",
                "title": f"Article {i}",
                "source_name": "Source",
                "brands": [],
            }
            for i in range(15)
        ]

        result = analyzer._format_article_sample(articles, "labeled")

        # Should only have articles 1-10
        assert "Article 1" in result
        assert "Article 10" in result
        assert "Article 11" not in result

    def test_handles_missing_fields(self, analyzer):
        """Should handle articles with missing optional fields."""
        articles = [
            {"id": "uuid-minimal"}  # Missing most fields
        ]

        result = analyzer._format_article_sample(articles, "labeled")

        assert "Article 1" in result
        assert "uuid-minimal" in result
        assert "No title" in result
        assert "Unknown" in result


class TestParseJSONResponse:
    """Tests for _parse_json_response method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked client."""
        with patch("src.agent.llm.Anthropic"):
            return LabelingAnalyzer(api_key="test-key")

    def test_parses_json_in_code_block(self, analyzer):
        """Should extract JSON from markdown code block."""
        response = '''Here is my analysis:

```json
{"summary": "Good quality", "score": 85}
```

Let me know if you need more details.'''

        result = analyzer._parse_json_response(response)

        assert result == {"summary": "Good quality", "score": 85}

    def test_parses_json_without_language_tag(self, analyzer):
        """Should extract JSON from code block without json tag."""
        response = '''Analysis:

```
{"summary": "Analysis complete"}
```'''

        result = analyzer._parse_json_response(response)

        assert result == {"summary": "Analysis complete"}

    def test_parses_raw_json(self, analyzer):
        """Should extract JSON object without code block."""
        response = '{"summary": "Direct JSON"}'

        result = analyzer._parse_json_response(response)

        assert result == {"summary": "Direct JSON"}

    def test_parses_json_with_surrounding_text(self, analyzer):
        """Should extract JSON even with surrounding text."""
        response = 'Here is the result: {"value": 42} - analysis complete.'

        result = analyzer._parse_json_response(response)

        assert result == {"value": 42}

    def test_returns_raw_for_no_json(self, analyzer):
        """Should return raw response when no JSON found."""
        response = "This is just plain text without any JSON."

        result = analyzer._parse_json_response(response)

        assert result == {"raw_response": response}

    def test_returns_raw_for_invalid_json(self, analyzer):
        """Should return raw response for invalid JSON that matches regex but fails parsing."""
        # JSON that looks valid to regex but has invalid syntax (trailing comma)
        response = '{"key": "value",}'

        result = analyzer._parse_json_response(response)

        assert "raw_response" in result
        assert "parse_error" in result

    def test_returns_raw_for_incomplete_json(self, analyzer):
        """Should return raw response when no JSON object found."""
        response = '{"incomplete": true'  # Missing closing brace - won't match regex

        result = analyzer._parse_json_response(response)

        assert result == {"raw_response": response}

    def test_parses_complex_json(self, analyzer):
        """Should parse complex nested JSON."""
        response = '''```json
{
    "overall_assessment": "Good quality",
    "potential_errors": [
        {"article_id": "123", "issue": "May be mislabeled"}
    ],
    "patterns_detected": [],
    "summary": "Analysis complete"
}
```'''

        result = analyzer._parse_json_response(response)

        assert result["overall_assessment"] == "Good quality"
        assert len(result["potential_errors"]) == 1
        assert result["patterns_detected"] == []


class TestAnalyzeLabelingResults:
    """Tests for analyze_labeling_results method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with mocked client."""
        with patch("src.agent.llm.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            analyzer = LabelingAnalyzer(api_key="test-key")
            analyzer.client = mock_client
            return analyzer

    def test_successful_analysis(self, analyzer):
        """Should return successful analysis."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"summary": "Analysis complete", "score": 85}')
        ]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 200
        analyzer.client.messages.create.return_value = mock_response

        stats = {
            "articles_processed": 100,
            "articles_labeled": 80,
            "articles_skipped": 15,
            "false_positives": 5,
            "articles_failed": 0,
            "error_rate": 0.0,
            "fp_rate": 0.05,
        }

        result = analyzer.analyze_labeling_results(
            stats=stats,
            labeled_sample=[],
            fp_sample=[],
            skipped_sample=[],
        )

        assert result.success is True
        assert result.analysis == {"summary": "Analysis complete", "score": 85}
        assert result.input_tokens == 1000
        assert result.output_tokens == 200

    def test_handles_api_error(self, analyzer):
        """Should handle API errors gracefully."""
        analyzer.client.messages.create.side_effect = Exception("API Error")

        result = analyzer.analyze_labeling_results(
            stats={},
            labeled_sample=[],
            fp_sample=[],
            skipped_sample=[],
        )

        assert result.success is False
        assert "API Error" in result.error

    def test_formats_samples_in_prompt(self, analyzer):
        """Should include formatted samples in prompt."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "OK"}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        analyzer.client.messages.create.return_value = mock_response

        labeled_sample = [
            {
                "id": "uuid-1",
                "title": "ESG Article",
                "source_name": "Source",
                "brands": ["Nike"],
                "content": "Content here",
            }
        ]

        analyzer.analyze_labeling_results(
            stats={"articles_processed": 1, "articles_labeled": 1, "articles_skipped": 0,
                   "false_positives": 0, "articles_failed": 0, "error_rate": 0, "fp_rate": 0},
            labeled_sample=labeled_sample,
            fp_sample=[],
            skipped_sample=[],
        )

        # Check that the API was called with the formatted content
        call_args = analyzer.client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[0]["content"]

        assert "ESG Article" in user_content
        assert "Nike" in user_content


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_system_prompt_contains_key_info(self):
        """System prompt should contain key ESG context."""
        assert "ESG" in LABELING_ANALYSIS_SYSTEM_PROMPT
        assert "sportswear" in LABELING_ANALYSIS_SYSTEM_PROMPT
        assert "Nike" in LABELING_ANALYSIS_SYSTEM_PROMPT
        assert "false_positive" in LABELING_ANALYSIS_SYSTEM_PROMPT
        assert "labeled" in LABELING_ANALYSIS_SYSTEM_PROMPT

    def test_user_prompt_has_placeholders(self):
        """User prompt should have all required placeholders."""
        required_placeholders = [
            "{articles_processed}",
            "{articles_labeled}",
            "{articles_skipped}",
            "{false_positives}",
            "{articles_failed}",
            "{error_rate",
            "{fp_rate",
            "{labeled_articles_sample}",
            "{false_positives_sample}",
            "{skipped_articles_sample}",
        ]

        for placeholder in required_placeholders:
            assert placeholder in LABELING_ANALYSIS_USER_PROMPT, f"Missing {placeholder}"

    def test_user_prompt_specifies_json_format(self):
        """User prompt should specify expected JSON response format."""
        assert "JSON" in LABELING_ANALYSIS_USER_PROMPT
        assert "overall_assessment" in LABELING_ANALYSIS_USER_PROMPT
        assert "potential_errors" in LABELING_ANALYSIS_USER_PROMPT
        assert "patterns_detected" in LABELING_ANALYSIS_USER_PROMPT
        assert "improvement_suggestions" in LABELING_ANALYSIS_USER_PROMPT


class TestRateLimitRetry:
    """Tests for rate limit retry logic."""

    def test_retries_on_rate_limit(self):
        """Should retry on rate limit error."""
        from anthropic import RateLimitError

        with patch("src.agent.llm.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # First call raises rate limit, second succeeds
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"summary": "OK"}')]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_client.messages.create.side_effect = [
                RateLimitError("Rate limited", response=MagicMock(), body={}),
                mock_response,
            ]

            with patch("time.sleep"):  # Don't actually sleep
                analyzer = LabelingAnalyzer(api_key="test-key", retry_delay=0.01)
                result = analyzer.analyze_labeling_results(
                    stats={
                        "articles_processed": 0,
                        "articles_labeled": 0,
                        "articles_skipped": 0,
                        "false_positives": 0,
                        "articles_failed": 0,
                        "error_rate": 0,
                        "fp_rate": 0,
                    },
                    labeled_sample=[],
                    fp_sample=[],
                    skipped_sample=[],
                )

            assert result.success is True
            assert mock_client.messages.create.call_count == 2

    def test_fails_after_max_retries(self):
        """Should fail after max retries exceeded."""
        from anthropic import RateLimitError

        with patch("src.agent.llm.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # All calls raise rate limit
            mock_client.messages.create.side_effect = RateLimitError(
                "Rate limited", response=MagicMock(), body={}
            )

            with patch("time.sleep"):  # Don't actually sleep
                analyzer = LabelingAnalyzer(
                    api_key="test-key", max_retries=2, retry_delay=0.01
                )
                result = analyzer.analyze_labeling_results(
                    stats={
                        "articles_processed": 0,
                        "articles_labeled": 0,
                        "articles_skipped": 0,
                        "false_positives": 0,
                        "articles_failed": 0,
                        "error_rate": 0,
                        "fp_rate": 0,
                    },
                    labeled_sample=[],
                    fp_sample=[],
                    skipped_sample=[],
                )

            assert result.success is False
            assert "Rate limit exceeded" in result.error
            assert mock_client.messages.create.call_count == 2
