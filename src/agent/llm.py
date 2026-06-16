"""LLM-based analysis for agent workflows.

Uses Claude Sonnet via the Anthropic API to analyze labeling results,
detect patterns in errors, and suggest improvements.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from anthropic import Anthropic, RateLimitError

from .config import agent_settings

logger = logging.getLogger(__name__)

# Analysis model - use Sonnet for cost efficiency
DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class AnalysisResult:
    """Result of LLM analysis."""

    success: bool
    analysis: dict[str, Any] | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


LABELING_ANALYSIS_SYSTEM_PROMPT = """You are an expert data quality analyst reviewing article labeling results for an ESG (Environmental, Social, Governance) news classification system focused on sportswear brands.

Your task is to analyze recent labeling outcomes and identify:
1. Potential labeling errors or inconsistencies
2. Patterns in false positives (articles incorrectly matched to brands)
3. Articles that may have been mislabeled
4. Suggestions for improving the labeling process

Target sportswear brands: Nike, Adidas, Puma, Under Armour, Lululemon, Patagonia, Columbia Sportswear, New Balance, ASICS, Reebok, Anta, Li-Ning, Xtep, 361 Degrees

Labeling status definitions:
- labeled: Successfully processed with ESG categories assigned
- false_positive: Brand name matched but article is not about sportswear brand
- skipped: Deliberately skipped (insufficient content, no ESG relevance)
- unlabelable: Cannot be labeled (scrape failed, paywall, non-English)

Respond with a JSON object containing your analysis."""

LABELING_ANALYSIS_USER_PROMPT = """Analyze the following labeling results from the past 24 hours:

## Summary Statistics
- Articles processed: {articles_processed}
- Articles labeled (ESG content found): {articles_labeled}
- Articles skipped (no ESG content): {articles_skipped}
- False positives detected: {false_positives}
- Articles failed: {articles_failed}
- Error rate: {error_rate:.1%}
- False positive rate: {fp_rate:.1%}

## Recent Labeled Articles (sample)
{labeled_articles_sample}

## Recent False Positives (sample)
{false_positives_sample}

## Recent Skipped Articles (sample)
{skipped_articles_sample}

Please analyze these results and respond with a JSON object in this format:
{{
    "overall_assessment": "brief assessment of labeling quality",
    "potential_errors": [
        {{
            "article_id": "uuid if specific, or null",
            "issue": "description of potential error",
            "severity": "low|medium|high",
            "recommendation": "suggested action"
        }}
    ],
    "patterns_detected": [
        {{
            "pattern": "description of pattern",
            "affected_count": "estimated number affected",
            "recommendation": "suggested action"
        }}
    ],
    "false_positive_analysis": {{
        "common_causes": ["list of common FP causes"],
        "brands_most_affected": ["brands with most FPs"],
        "recommendations": ["suggestions to reduce FPs"]
    }},
    "improvement_suggestions": [
        {{
            "area": "labeling|prompts|pre-filtering|data-collection",
            "suggestion": "specific improvement",
            "priority": "low|medium|high",
            "effort": "low|medium|high"
        }}
    ],
    "summary": "2-3 sentence executive summary"
}}"""


class LabelingAnalyzer:
    """Analyzes labeling results using Claude."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the analyzer.

        Args:
            api_key: Anthropic API key (default: from environment)
            model: Model to use (default: claude-sonnet-4-6)
            max_retries: Maximum retry attempts for rate limits
            retry_delay: Initial delay between retries in seconds
        """
        self.api_key = api_key or agent_settings.anthropic_api_key
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)

    def analyze_labeling_results(
        self,
        stats: dict[str, Any],
        labeled_sample: list[dict[str, Any]],
        fp_sample: list[dict[str, Any]],
        skipped_sample: list[dict[str, Any]],
    ) -> AnalysisResult:
        """Analyze labeling results and return findings.

        Args:
            stats: Summary statistics (articles_processed, error_rate, etc.)
            labeled_sample: Sample of recently labeled articles
            fp_sample: Sample of false positive articles
            skipped_sample: Sample of skipped articles

        Returns:
            AnalysisResult with findings
        """
        # Format samples for the prompt
        labeled_text = self._format_article_sample(labeled_sample, "labeled")
        fp_text = self._format_article_sample(fp_sample, "false_positive")
        skipped_text = self._format_article_sample(skipped_sample, "skipped")

        user_prompt = LABELING_ANALYSIS_USER_PROMPT.format(
            articles_processed=stats.get("articles_processed", 0),
            articles_labeled=stats.get("articles_labeled", 0),
            articles_skipped=stats.get("articles_skipped", 0),
            false_positives=stats.get("false_positives", 0),
            articles_failed=stats.get("articles_failed", 0),
            error_rate=stats.get("error_rate", 0),
            fp_rate=stats.get("fp_rate", 0),
            labeled_articles_sample=labeled_text or "No labeled articles in sample",
            false_positives_sample=fp_text or "No false positives in sample",
            skipped_articles_sample=skipped_text or "No skipped articles in sample",
        )

        return self._call_api(user_prompt)

    def _format_article_sample(
        self, articles: list[dict[str, Any]], status: str
    ) -> str:
        """Format article sample for the prompt."""
        if not articles:
            return ""

        lines = []
        for i, article in enumerate(articles[:10], 1):  # Limit to 10
            lines.append(f"\n### Article {i}")
            lines.append(f"- ID: {article.get('id', 'unknown')}")
            lines.append(f"- Title: {article.get('title', 'No title')[:100]}")
            lines.append(f"- Source: {article.get('source_name', 'Unknown')}")
            lines.append(f"- Brands: {', '.join(article.get('brands', []))}")

            if status == "labeled":
                # Include ESG categories if available
                categories = article.get("categories", [])
                if categories:
                    lines.append(f"- ESG Categories: {', '.join(categories)}")

            # Include snippet of content
            content = article.get("content", article.get("scraped_content", ""))
            if content:
                snippet = content[:300].replace("\n", " ")
                lines.append(f"- Content snippet: {snippet}...")

        return "\n".join(lines)

    def _call_api(self, user_prompt: str) -> AnalysisResult:
        """Call Claude API with retry logic."""
        import time

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Claude API for analysis (attempt {attempt + 1})")

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.0,
                    system=LABELING_ANALYSIS_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                # Extract response text
                response_text = response.content[0].text

                # Parse JSON from response
                analysis = self._parse_json_response(response_text)

                return AnalysisResult(
                    success=True,
                    analysis=analysis,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=self.model,
                )

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limited, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    return AnalysisResult(
                        success=False,
                        error=f"Rate limit exceeded: {e}",
                        model=self.model,
                    )

            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                return AnalysisResult(
                    success=False,
                    error=str(e),
                    model=self.model,
                )

        return AnalysisResult(
            success=False,
            error="Max retries exceeded",
            model=self.model,
        )

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from response, handling markdown code blocks."""
        import re

        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Return raw text if no JSON found
                return {"raw_response": response_text}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {"raw_response": response_text, "parse_error": str(e)}


def get_recent_labeling_samples(
    days: int = 1,
    sample_size: int = 10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Get samples of recent labeling results from database.

    Args:
        days: Number of days to look back
        sample_size: Max articles per category

    Returns:
        Tuple of (labeled_sample, fp_sample, skipped_sample)
    """
    from sqlalchemy import create_engine, text
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set, returning empty samples")
        return [], [], []

    engine = create_engine(database_url)

    labeled_sample = []
    fp_sample = []
    skipped_sample = []

    try:
        with engine.connect() as conn:
            # Get recently labeled articles
            result = conn.execute(
                text("""
                    SELECT a.id, a.title, a.source_name, a.brands_mentioned,
                           LEFT(a.full_content, 500) as content
                    FROM articles a
                    WHERE a.labeling_status = 'labeled'
                      AND a.labeled_at >= NOW() - INTERVAL '1 day' * :days
                    ORDER BY a.labeled_at DESC
                    LIMIT :limit
                """),
                {"days": days, "limit": sample_size},
            )
            for row in result:
                labeled_sample.append({
                    "id": str(row.id),
                    "title": row.title,
                    "source_name": row.source_name,
                    "brands": row.brands_mentioned or [],
                    "content": row.content,
                })

            # Get recent false positives
            result = conn.execute(
                text("""
                    SELECT a.id, a.title, a.source_name, a.brands_mentioned,
                           LEFT(a.full_content, 500) as content
                    FROM articles a
                    WHERE a.labeling_status = 'false_positive'
                      AND a.skipped_at >= NOW() - INTERVAL '1 day' * :days
                    ORDER BY a.skipped_at DESC
                    LIMIT :limit
                """),
                {"days": days, "limit": sample_size},
            )
            for row in result:
                fp_sample.append({
                    "id": str(row.id),
                    "title": row.title,
                    "source_name": row.source_name,
                    "brands": row.brands_mentioned or [],
                    "content": row.content,
                })

            # Get recent skipped articles
            result = conn.execute(
                text("""
                    SELECT a.id, a.title, a.source_name, a.brands_mentioned,
                           LEFT(a.full_content, 500) as content
                    FROM articles a
                    WHERE a.labeling_status = 'skipped'
                      AND a.skipped_at >= NOW() - INTERVAL '1 day' * :days
                    ORDER BY a.skipped_at DESC
                    LIMIT :limit
                """),
                {"days": days, "limit": sample_size},
            )
            for row in result:
                skipped_sample.append({
                    "id": str(row.id),
                    "title": row.title,
                    "source_name": row.source_name,
                    "brands": row.brands_mentioned or [],
                    "content": row.content,
                })

    except Exception as e:
        logger.error(f"Failed to get labeling samples: {e}")

    return labeled_sample, fp_sample, skipped_sample
