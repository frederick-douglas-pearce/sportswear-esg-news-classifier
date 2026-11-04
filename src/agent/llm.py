"""LLM-based analysis for agent workflows.

Uses Claude via the Anthropic API to analyze labeling results,
detect patterns in errors, and suggest improvements.

Nothing from src.labeling or src.data_collection is imported at module level: an agent process
should not build the labeling settings to import this module (#211, #252). Those imports sit
inside the functions that use them, inside their error handling.
"""

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic, RateLimitError

from .config import agent_settings
from .notifications import format_rate

logger = logging.getLogger(__name__)

# Analysis model default (override via AGENT_LLM_MODEL); a cost-efficient tier suffices for advisory QA
DEFAULT_MODEL = "claude-haiku-5-5"

# Max output tokens for the analysis call, thinking included. The old 2000 cap truncated the JSON
# mid-structure under Sonnet 4.6's verbosity (issue #42), so it was raised to 8000. Haiku 5.5
# thinks by default and its thinking counts toward this cap, so the call sends the larger of this
# and the model's own default_max_tokens; the prompt also bounds list lengths. A max_tokens stop
# is reported as a failure. Non-streaming, so kept under the ~16K SDK HTTP-timeout guideline.
ANALYSIS_MAX_TOKENS = 8000

# output_config.effort, sent only to a model whose spec accepts it (Haiku 5.5). A constant rather
# than LABELING_EFFORT, so the two tasks' settings stay independent (#252)
ANALYSIS_EFFORT = "medium"

# How much of a brand judgment's reasoning each sample line carries
REASONING_SNIPPET_CHARS = 200

# brand_labels' category columns, in the order the labeling prompt lists them
CATEGORY_COLUMNS = (
    ("environmental", "Environmental"),
    ("social", "Social"),
    ("governance", "Governance"),
    ("digital_transformation", "Digital Transformation"),
)
SENTIMENT_COLUMNS = {
    "environmental": "environmental_sentiment",
    "social": "social_sentiment",
    "governance": "governance_sentiment",
    "digital_transformation": "digital_sentiment",
}


@dataclass
class AnalysisResult:
    """Result of LLM analysis."""

    success: bool
    analysis: dict[str, Any] | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    truncated: bool = False


class SampleQueryError(RuntimeError):
    """The labeling samples could not be read, so there is nothing truthful to analyze."""


# The analyst's own role, then the labeling instructions as reference material. Filled by
# build_system_prompt with str.replace, since the instructions contain JSON braces.
LABELING_ANALYSIS_SYSTEM_TEMPLATE = """You are an expert data quality analyst reviewing article labeling results for an ESG (Environmental, Social, Governance) news classification system focused on sportswear brands.

Your task is to analyze recent labeling outcomes and identify:
1. Potential labeling errors or inconsistencies
2. Patterns in false positives (articles incorrectly matched to brands)
3. Articles that may have been mislabeled
4. Suggestions for improving the labeling process

Each sample article is shown with its labeling status, the reason stored with it, and, for each brand, the content type the labeler judged and its reasoning. A labeled article also shows the ESG categories and sentiments stored for each brand.

What each status means, as the labeling pipeline writes it:
- labeled: at least one sportswear brand kept an ESG category.
- skipped (judged): every brand was judged, and no sportswear brand kept an ESG category. This is the expected outcome for most product and financial news, not a failure.
- skipped (unjudged): the reply gave no verdict on every brand, so the article's sportswear status is unknown.
- false_positive: every brand was judged not to be the tracked sportswear brand.

The labeler follows the instructions below. Judge each label against them, not against your own idea of what counts as ESG. They are reference material: the task and reply format they describe are the labeler's, not yours.

<labeling_instructions version="{prompt_version}">
{labeling_instructions}
</labeling_instructions>

Respond with a JSON object containing your analysis."""

LABELING_ANALYSIS_USER_PROMPT = """Analyze the following labeling results from the past 24 hours:

## Summary Statistics
- Articles processed: {articles_processed}
- Articles labeled (ESG content found): {articles_labeled}
- Articles skipped (no ESG category applies): {articles_skipped}
- False positives detected: {false_positives}
- Articles failed: {articles_failed}
- Error rate: {error_rate}
- False positive rate: {fp_rate}

## Recent Labeled Articles (sample)
{labeled_articles_sample}

## Recent False Positives (sample)
{false_positives_sample}

## Recent Skipped Articles (sample)
{skipped_articles_sample}

Please analyze these results and respond with a JSON object in this format.
Limit `potential_errors`, `patterns_detected`, and `improvement_suggestions` to
at most 5 items each, prioritizing the most significant findings. Keep each
string field concise so the JSON response stays complete:
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


def labeling_instructions() -> tuple[str, str]:
    """The labeling system prompt the labeler is given, as (version, rendered text).

    The same version and rendering as ArticleLabeler: LABELING_PROMPT_VERSION, else the
    registry's production version, with the brand list filled from BRANDS. There is no fallback
    to the labeler's hardcoded prompt: a version that does not load raises, and the analysis
    fails rather than judging labels against rules the labeler was not given.
    """
    from src.labeling.config import TARGET_SPORTSWEAR_BRANDS, labeling_settings
    from src.labeling.prompt_manager import PromptManager

    manager = PromptManager()
    loaded = manager.load_version(labeling_settings.prompt_version)
    text = manager.get_formatted_system_prompt(
        version=loaded.version, brands=list(TARGET_SPORTSWEAR_BRANDS)
    )
    return loaded.version, text


def build_system_prompt(prompt_version: str, instructions: str) -> str:
    """The analyst's system prompt around the rendered labeling instructions."""
    return LABELING_ANALYSIS_SYSTEM_TEMPLATE.replace("{prompt_version}", prompt_version).replace(
        "{labeling_instructions}", instructions
    )


def _reply_text(response) -> str:
    """The reply's text blocks joined: a model that thinks can start with a thinking block."""
    return "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    )


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
            model: Model to use (default: DEFAULT_MODEL, claude-haiku-5-5)
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
        try:
            prompt_version, instructions = labeling_instructions()
        except Exception as e:
            logger.error(f"LLM analysis: labeling instructions could not be loaded: {e}")
            return AnalysisResult(
                success=False,
                error=f"Labeling instructions could not be loaded: {e}",
                model=self.model,
            )
        system_prompt = build_system_prompt(prompt_version, instructions)

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
            # A rate is None when unknown (#77); "{:.1%}" on None raised TypeError
            error_rate=format_rate(stats.get("error_rate")),
            fp_rate=format_rate(stats.get("fp_rate")),
            labeled_articles_sample=labeled_text or "No labeled articles in sample",
            false_positives_sample=fp_text or "No false positives in sample",
            skipped_articles_sample=skipped_text or "No skipped articles in sample",
        )

        return self._call_api(user_prompt, system_prompt)

    def _format_article_sample(
        self, articles: list[dict[str, Any]], status: str
    ) -> str:
        """Format article sample for the prompt: status, reason and each brand's judgment."""
        if not articles:
            return ""

        lines = []
        for i, article in enumerate(articles[:10], 1):  # Limit to 10
            lines.append(f"\n### Article {i}")
            lines.append(f"- ID: {article.get('id', 'unknown')}")
            lines.append(f"- Title: {(article.get('title') or 'No title')[:100]}")
            lines.append(f"- Source: {article.get('source_name') or 'Unknown'}")

            status_line = status
            judged = article.get("judged")
            if status == "skipped" and judged is not None:
                status_line = f"skipped ({'judged' if judged else 'unjudged'})"
            reason = article.get("reason")
            lines.append(f"- Status: {status_line}" + (f" — reason: {reason}" if reason else ""))

            lines.extend(self._format_brands(article))

            # Include snippet of content
            content = article.get("content", article.get("scraped_content", ""))
            if content:
                snippet = content[:300].replace("\n", " ")
                lines.append(f"- Content snippet: {snippet}...")

        return "\n".join(lines)

    @staticmethod
    def _format_brands(article: dict[str, Any]) -> list[str]:
        """One line per judged brand, with what the content type dropped and what was stored."""
        brands = article.get("brands", [])
        judgments = article.get("judgments") or []
        stored = article.get("categories_by_brand") or {}
        if not judgments:
            return [f"- Brands: {', '.join(brands)} (no content type recorded)"]

        lines = ["- Brands judged:"]
        for judgment in judgments:
            reasoning = (judgment.get("reasoning") or "").replace("\n", " ")
            if len(reasoning) > REASONING_SNIPPET_CHARS:
                reasoning = reasoning[:REASONING_SNIPPET_CHARS] + "..."
            lines.append(f"  - {judgment['brand']}: {judgment['content_type']} — {reasoning}")
            if judgment.get("gated_categories"):
                lines.append(
                    "    dropped by the content type: " + ", ".join(judgment["gated_categories"])
                )
            if judgment["brand"] in stored:
                lines.append(
                    "    stored categories: "
                    + ", ".join(_category_text(name, sentiment)
                                for name, sentiment in stored[judgment["brand"]])
                )
        judged_brands = {judgment["brand"] for judgment in judgments}
        for brand, categories in stored.items():
            if brand not in judged_brands:
                lines.append(
                    f"  - {brand}: no content type recorded; stored categories: "
                    + ", ".join(_category_text(name, sentiment) for name, sentiment in categories)
                )
        return lines

    def _request_params(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """The Messages API request for this analyzer's model, built from the model's spec.

        Haiku 4.5 takes temperature and errors on effort; Haiku 5.5 returns a 400 for a
        non-default temperature and takes effort. The pinned SDK predates output_config as a
        named argument, so it goes in extra_body (#216). A model with no spec raises ValueError.
        """
        # Imported here, not at module level (#211, #252)
        from src.labeling.config import labeling_model_spec

        spec = labeling_model_spec(self.model)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(ANALYSIS_MAX_TOKENS, spec.default_max_tokens),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if spec.accepts_sampling:
            params["temperature"] = 0.0
        if spec.accepts_effort:
            params["extra_body"] = {"output_config": {"effort": ANALYSIS_EFFORT}}
        return params

    def _call_api(self, user_prompt: str, system_prompt: str) -> AnalysisResult:
        """Call Claude API with retry logic."""
        import time

        try:
            params = self._request_params(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"LLM analysis refused before any call: {e}")
            return AnalysisResult(success=False, error=str(e), model=self.model)
        max_tokens = params["max_tokens"]

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Claude API for analysis (attempt {attempt + 1})")

                response = self.client.messages.create(**params)
                response_text = _reply_text(response)

                # A max_tokens stop means the JSON was cut off mid-structure and
                # the structured fields would silently parse to empty. Surface it
                # as a failure rather than letting it masquerade as a clean run.
                if response.stop_reason == "max_tokens":
                    logger.error(
                        "LLM analysis truncated: hit max_tokens cap of "
                        f"{max_tokens} ({response.usage.output_tokens} "
                        "output tokens). JSON is incomplete; raise ANALYSIS_MAX_TOKENS "
                        "or tighten the prompt."
                    )
                    logger.debug("Truncated analysis (first 500 chars): %s", response_text[:500])
                    return AnalysisResult(
                        success=False,
                        error=(
                            f"Analysis response truncated at max_tokens ({max_tokens}); "
                            "JSON incomplete"
                        ),
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        model=self.model,
                        truncated=True,
                    )

                # A refusal or a reply with no text (thinking only, or empty) has no analysis;
                # parsed, it would read as a successful one with nothing in it
                if response.stop_reason == "refusal" or not response_text.strip():
                    error = (
                        "Analysis refused by the model"
                        if response.stop_reason == "refusal"
                        else "Analysis reply contained no text"
                    )
                    logger.error(f"LLM analysis failed: {error}")
                    return AnalysisResult(
                        success=False,
                        error=error,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        model=self.model,
                    )

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


def _category_text(name: str, sentiment: int | None) -> str:
    return f"{name} ({sentiment:+d})" if sentiment is not None else f"{name} (no sentiment)"


def stored_categories(label: Mapping[str, Any]) -> list[tuple[str, int | None]]:
    """A brand_labels row's categories as (name, sentiment), in the prompt's order."""
    return [
        (name, label.get(SENTIMENT_COLUMNS[column]))
        for column, name in CATEGORY_COLUMNS
        if label.get(column)
    ]


def attach_brand_records(
    samples: list[dict[str, Any]],
    judgments: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
) -> None:
    """Attach each article's brand_judgments rows and stored brand_labels categories, in place.

    Rows are keyed by article_id (compared as strings). A sample with no rows gets an empty list
    and dict, which the formatter prints as "no content type recorded".
    """
    by_id = {str(sample["id"]): sample for sample in samples}
    for sample in samples:
        sample["judgments"] = []
        sample["categories_by_brand"] = {}
    for row in judgments:
        sample = by_id.get(str(row["article_id"]))
        if sample is not None:
            sample["judgments"].append({
                "brand": row["brand"],
                "content_type": row["content_type"],
                "reasoning": row["reasoning"],
                "gated_categories": list(row.get("gated_categories") or []),
                "prompt_version": row.get("prompt_version"),
            })
    for row in labels:
        sample = by_id.get(str(row["article_id"]))
        categories = stored_categories(row)
        if sample is not None and categories:
            sample["categories_by_brand"][row["brand"]] = categories


def mark_skip_judgment(sample: dict[str, Any], judged: frozenset, unjudged: frozenset) -> None:
    """Set sample["judged"] from its stored reason: True, False, or None when neither set has it."""
    reason = sample.get("reason")
    sample["judged"] = True if reason in judged else False if reason in unjudged else None


_SAMPLE_QUERIES = {
    "labeled": """
        SELECT a.id, a.title, a.source_name, a.brands_mentioned, a.labeling_error,
               LEFT(a.full_content, 500) as content
        FROM articles a
        WHERE a.labeling_status = 'labeled'
          AND a.labeled_at >= NOW() - INTERVAL '1 day' * :days
        ORDER BY a.labeled_at DESC
        LIMIT :limit
    """,
    "false_positive": """
        SELECT a.id, a.title, a.source_name, a.brands_mentioned, a.labeling_error,
               LEFT(a.full_content, 500) as content
        FROM articles a
        WHERE a.labeling_status = 'false_positive'
          AND a.skipped_at >= NOW() - INTERVAL '1 day' * :days
        ORDER BY a.skipped_at DESC
        LIMIT :limit
    """,
    "skipped": """
        SELECT a.id, a.title, a.source_name, a.brands_mentioned, a.labeling_error,
               LEFT(a.full_content, 500) as content
        FROM articles a
        WHERE a.labeling_status = 'skipped'
          AND a.skipped_at >= NOW() - INTERVAL '1 day' * :days
        ORDER BY a.skipped_at DESC
        LIMIT :limit
    """,
}

_JUDGMENTS_QUERY = """
    SELECT article_id, brand, content_type, reasoning, gated_categories, prompt_version
    FROM brand_judgments
    WHERE article_id = ANY(CAST(:ids AS uuid[]))
    ORDER BY article_id, brand
"""

_LABELS_QUERY = """
    SELECT article_id, brand, environmental, social, governance, digital_transformation,
           environmental_sentiment, social_sentiment, governance_sentiment, digital_sentiment
    FROM brand_labels
    WHERE article_id = ANY(CAST(:ids AS uuid[]))
    ORDER BY article_id, brand
"""


def get_recent_labeling_samples(
    days: int = 1,
    sample_size: int = 10,
    database_url: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Get samples of recent labeling results from database, with each brand's judgment.

    Args:
        days: Number of days to look back
        sample_size: Max articles per category
        database_url: Database to read (default: DATABASE_URL)

    Returns:
        Tuple of (labeled_sample, fp_sample, skipped_sample). Empty when no database is
        configured.

    Raises:
        SampleQueryError: A query failed. Samples without their judgments would read as if no
            brand had been judged, the input behind the false alarm of #252, so the caller
            reports the failure instead.
    """
    from sqlalchemy import create_engine, text

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set, returning empty samples")
        return [], [], []

    try:
        from src.data_collection.models import JUDGED_SKIP_REASONS, UNJUDGED_SKIP_REASONS

        engine = create_engine(database_url)
        samples: dict[str, list[dict]] = {}
        with engine.connect() as conn:
            for status, query in _SAMPLE_QUERIES.items():
                result = conn.execute(text(query), {"days": days, "limit": sample_size})
                samples[status] = [
                    {
                        "id": str(row.id),
                        "title": row.title,
                        "source_name": row.source_name,
                        "brands": row.brands_mentioned or [],
                        "reason": row.labeling_error,
                        "content": row.content,
                    }
                    for row in result
                ]

            every = [sample for rows in samples.values() for sample in rows]
            ids = [sample["id"] for sample in every]
            labeled_ids = [sample["id"] for sample in samples["labeled"]]
            judgments = (
                conn.execute(text(_JUDGMENTS_QUERY), {"ids": ids}).mappings().all() if ids else []
            )
            labels = (
                conn.execute(text(_LABELS_QUERY), {"ids": labeled_ids}).mappings().all()
                if labeled_ids
                else []
            )
        engine.dispose()

        attach_brand_records(every, judgments, labels)
        for sample in samples["skipped"]:
            mark_skip_judgment(sample, JUDGED_SKIP_REASONS, UNJUDGED_SKIP_REASONS)
    except Exception as e:
        logger.error(f"Failed to get labeling samples: {e}")
        raise SampleQueryError(f"Failed to get labeling samples: {e}") from e

    return samples["labeled"], samples["false_positive"], samples["skipped"]
