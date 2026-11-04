"""LLM-based analysis for agent workflows.

Uses Claude via the Anthropic API to analyze labeling results,
detect patterns in errors, and suggest improvements.

This module imports nothing from src.labeling or src.data_collection at module level (#252), so
it adds no import path to the labeling package; the agent package still imports src.labeling
through its runner until #211. Those imports sit inside the functions that use them, and their
callers catch what they raise.
"""

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

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
2. Patterns in false_positive outcomes (the name meant another entity, or the article gave the brand no substantive content)
3. Articles that may have been mislabeled
4. Suggestions for improving the labeling process

Each sample article is shown with its labeling status, who decided it, and the reason stored with it. Then, for each brand sent to the labeler, the content type it judged and the start of its reasoning, or "not judged"; and for a labeled article, the ESG categories and sentiments stored for each brand.

What each status means, by who decided it:
- labeled (labeler): at least one brand was judged sportswear and kept an ESG category.
- skipped (labeler): at least one brand was judged sportswear and none kept an ESG category; other sent brands may be "not judged". This is the expected outcome for most product, financial and business news, not a failure.
- skipped (unjudged): the article's sportswear status is unknown. Either no LLM call was made (insufficient content, no brands), or the reply stated no sportswear verdict (no analyses, or only non-sportswear brands with a sent brand left out).
- skipped (unattributed): the stored reason is not one the pipeline writes; read the reason.
- false_positive (labeler): every sent brand was judged other_entity, mention_only or retail_only. That includes the tracked company itself when the article gives it no substantive content.
- false_positive (FP classifier): the FP classifier scored the article below its threshold, so no LLM call was made and no content type is recorded.
- false_positive (keyword rule): a cleanup sweep wrote it; no content type is recorded.
Human rulings and human-written labels are left out of the samples.

What you see is partial: the first 300 characters of each article, the first 200 characters of each brand's reasoning, and no evidence excerpts. The labeler read the full article. Report a label as a likely error only where the judgment's own reasoning, the stored categories and the instructions disagree, and mark a doubt that depends on text you cannot see as "verify against the full article". A brand line marked "judged under <version>" was judged under a different prompt version from the instructions below.

The labeler follows the instructions below. Judge each label against them, not against your own idea of what counts as ESG. They are reference material: the task and reply format they describe are the labeler's, not yours.

<labeling_instructions version="{prompt_version}">
{labeling_instructions}
</labeling_instructions>

Respond with a JSON object containing your analysis."""

LABELING_ANALYSIS_USER_PROMPT = """Analyze the following labeling results from the past 24 hours:

## Summary Statistics
- Articles processed: {articles_processed}
- Articles labeled (ESG content found): {articles_labeled}
- Articles skipped (all skip reasons): {articles_skipped}
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
        labeled_text = self._format_article_sample(labeled_sample, "labeled", prompt_version)
        fp_text = self._format_article_sample(fp_sample, "false_positive", prompt_version)
        skipped_text = self._format_article_sample(skipped_sample, "skipped", prompt_version)

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
        self, articles: list[dict[str, Any]], status: str, prompt_version: str | None = None
    ) -> str:
        """Format article sample for the prompt: status, decider, reason and each brand's line.

        ``prompt_version`` is the version of the instructions the analyst is given; a judgment
        made under another version says so.
        """
        if not articles:
            return ""

        lines = []
        for i, article in enumerate(articles[:10], 1):  # Limit to 10
            lines.append(f"\n### Article {i}")
            lines.append(f"- ID: {article.get('id', 'unknown')}")
            lines.append(f"- Title: {(article.get('title') or 'No title')[:100]}")
            lines.append(f"- Source: {article.get('source_name') or 'Unknown'}")

            decider = article.get("decider")
            status_line = f"{status} ({decider})" if decider else status
            reason = article.get("reason")
            lines.append(f"- Status: {status_line}" + (f" — reason: {reason}" if reason else ""))

            lines.extend(self._format_brands(article, prompt_version))

            # Include snippet of content
            content = article.get("content", article.get("scraped_content", ""))
            if content:
                snippet = content[:300].replace("\n", " ")
                lines.append(f"- Content snippet: {snippet}...")

        return "\n".join(lines)

    @staticmethod
    def _format_brands(article: dict[str, Any], prompt_version: str | None = None) -> list[str]:
        """One line per brand: its judgment or "not judged", what the content type dropped, and
        the categories stored for it."""
        brand_lines = article.get("brand_lines")
        if brand_lines is None:
            return [f"- Brands: {', '.join(article.get('brands', []))}"]
        if not brand_lines:
            return ["- Brands: none recorded"]

        lines = ["- Brands:"]
        for line in brand_lines:
            judgment = line["judgment"]
            if judgment:
                reasoning = (judgment.get("reasoning") or "").replace("\n", " ")
                if len(reasoning) > REASONING_SNIPPET_CHARS:
                    reasoning = reasoning[:REASONING_SNIPPET_CHARS] + "..."
                version = judgment.get("prompt_version")
                judged_under = (
                    f" (judged under {version})"
                    if prompt_version and version and version != prompt_version
                    else ""
                )
                lines.append(
                    f"  - {line['brand']}: {judgment['content_type']}{judged_under} — {reasoning}"
                )
                if judgment.get("gated_categories"):
                    lines.append(
                        "    dropped by the content type: "
                        + ", ".join(_category_name(c) for c in judgment["gated_categories"])
                    )
            else:
                lines.append(f"  - {line['brand']}: not judged (no content type recorded)")
            if line["stored"]:
                lines.append(
                    "    stored categories: "
                    + ", ".join(_category_text(name, sentiment) for name, sentiment in line["stored"])
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


def _category_name(key: str) -> str:
    """A category key as the prompt names it (gated_categories stores keys)."""
    return dict(CATEGORY_COLUMNS).get(key, key)


def _category_text(name: str, sentiment: int | None) -> str:
    if sentiment is None:
        return f"{name} (no sentiment)"
    return f"{name} ({sentiment:+d})" if sentiment else f"{name} (0)"


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
    key: Callable[[str], str] | None = None,
) -> None:
    """Set each sample's ``brand_lines``: one per brand, sent brands first, in place.

    A line carries the brand, its brand_judgments row (or None: not judged) and its stored
    brand_labels categories. Rows are matched to a sample by article id (compared as strings) and
    to a brand by ``key``, ``normalize_brand`` by default: brand_judgments stores the brand as
    sent, brand_labels as the reply spelled it, and the pipeline matches the two the same way.
    """
    if key is None:
        # Imported here, not at module level (#211, #252)
        from src.labeling.models import normalize_brand as key

    by_id = {str(sample["id"]): sample for sample in samples}
    judged: dict[str, dict[str, dict[str, Any]]] = {}
    for row in judgments:
        article_id = str(row["article_id"])
        if article_id in by_id:
            judged.setdefault(article_id, {})[key(row["brand"])] = {
                "brand": row["brand"],
                "content_type": row["content_type"],
                "reasoning": row["reasoning"],
                "gated_categories": list(row.get("gated_categories") or []),
                "prompt_version": row.get("prompt_version"),
            }
    stored: dict[str, dict[str, tuple[str, list]]] = {}
    for row in labels:
        article_id = str(row["article_id"])
        categories = stored_categories(row)
        if article_id in by_id and categories:
            stored.setdefault(article_id, {})[key(row["brand"])] = (row["brand"], categories)

    for article_id, sample in by_id.items():
        article_judged = judged.get(article_id, {})
        article_stored = stored.get(article_id, {})
        names = (
            list(sample.get("brands") or [])
            + [judgment["brand"] for judgment in article_judged.values()]
            + [brand for brand, _ in article_stored.values()]
        )
        lines, seen = [], set()
        for name in names:
            brand_key = key(name)
            if brand_key in seen:
                continue
            seen.add(brand_key)
            judgment = article_judged.get(brand_key)
            lines.append({
                "brand": judgment["brand"] if judgment else name,
                "judgment": judgment,
                "stored": article_stored.get(brand_key, (name, []))[1],
            })
        sample["brand_lines"] = lines


# The sample buckets, in the order get_recent_labeling_samples returns them
SAMPLE_STATUSES = ("labeled", "false_positive", "skipped")


def _decider_names() -> dict[str, str]:
    """provenance's label sources as the status line names them (the LLM's two eras are one)."""
    from src.data_collection import provenance

    return {
        provenance.LLM: "labeler",
        provenance.PRE_CLASSIFIER: "labeler",
        provenance.FP_CLASSIFIER: "FP classifier",
        provenance.UNJUDGED: "unjudged",
        provenance.UNATTRIBUTED: "unattributed",
        provenance.RULE: "keyword rule",
    }


def build_samples(
    article_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    judgments: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Samples from article rows keyed by status, with each one's decider and brand lines.

    Who decided each row comes from ``provenance.derive_label_source``, as the FP export reads it.
    A row a human decided is left out: human rulings are ground truth, not output to audit (#252).
    The queries already leave such rows out; this is the same rule applied to what they return.
    """
    from src.data_collection import provenance

    names = _decider_names()
    samples: dict[str, list[dict]] = {}
    for status in SAMPLE_STATUSES:
        rows = list(article_rows.get(status, []))
        human = provenance.human_verified_ids((str(row["id"]), status) for row in rows)
        bucket = []
        for row in rows:
            article_id = str(row["id"])
            try:
                source = provenance.derive_label_source(
                    status,
                    row["labeling_error"],
                    row.get("fp_action"),
                    row.get("fp_skip_reason"),
                    human_verified=article_id in human,
                    article_id=article_id,
                )
            except provenance.UnknownFPActionError:
                source = None
            if source == provenance.HUMAN:
                continue
            bucket.append({
                "id": article_id,
                "title": row["title"],
                "source_name": row["source_name"],
                "brands": list(row["brands_mentioned"] or []),
                "reason": row["labeling_error"],
                "decider": names.get(source, "unknown"),
                "content": row["content"],
            })
        samples[status] = bucket

    attach_brand_records([s for bucket in samples.values() for s in bucket], judgments, labels)
    return samples["labeled"], samples["false_positive"], samples["skipped"]


def _article_rows(conn, status: str, since: datetime, limit: int) -> list:
    """The most recent rows at a status decided since ``since``, with their latest FP prediction,
    leaving out what a human decided: a human ruling at this status, a human marker as the reason,
    or a human brand label."""
    from sqlalchemy import func, or_, select

    from src.data_collection import provenance
    from src.data_collection.models import (
        FP_REASON_MANUAL,
        MANUAL_SKIP_REASON,
        Article,
        BrandLabel,
    )

    latest = provenance.latest_prediction_subquery("fp")
    # A false_positive's skipped_at is not written, so this bucket is empty until #256
    decided_at = Article.labeled_at if status == "labeled" else Article.skipped_at
    human_labeled = select(BrandLabel.article_id).where(provenance.human_label_criterion())
    query = (
        select(
            Article.id,
            Article.title,
            Article.source_name,
            Article.brands_mentioned,
            Article.labeling_error,
            func.left(Article.full_content, 500).label("content"),
            latest.c.action_taken.label("fp_action"),
            latest.c.skip_reason.label("fp_skip_reason"),
        )
        .select_from(Article)
        .outerjoin(latest, latest.c.article_id == Article.id)
        .where(
            Article.labeling_status == status,
            decided_at >= since,
            Article.id.not_in(human_labeled),
            or_(
                Article.labeling_error.is_(None),
                Article.labeling_error.not_in([FP_REASON_MANUAL, MANUAL_SKIP_REASON]),
            ),
        )
        .order_by(decided_at.desc())
        .limit(limit)
    )
    ruled = [UUID(i) for i, ruled_status in provenance.HUMAN_RULINGS.items() if ruled_status == status]
    if ruled:
        query = query.where(Article.id.not_in(ruled))
    return conn.execute(query).mappings().all()


def _judgment_rows(conn, article_ids: list) -> list:
    from sqlalchemy import select

    from src.data_collection.models import BrandJudgment

    query = select(
        BrandJudgment.article_id,
        BrandJudgment.brand,
        BrandJudgment.content_type,
        BrandJudgment.reasoning,
        BrandJudgment.gated_categories,
        BrandJudgment.prompt_version,
    ).where(BrandJudgment.article_id.in_(article_ids)).order_by(
        BrandJudgment.article_id, BrandJudgment.brand
    )
    return conn.execute(query).mappings().all()


def _label_rows(conn, article_ids: list) -> list:
    from sqlalchemy import select

    from src.data_collection.models import BrandLabel

    query = select(
        BrandLabel.article_id,
        BrandLabel.brand,
        *(getattr(BrandLabel, column) for column, _ in CATEGORY_COLUMNS),
        *(getattr(BrandLabel, column) for column in SENTIMENT_COLUMNS.values()),
    ).where(BrandLabel.article_id.in_(article_ids)).order_by(
        BrandLabel.article_id, BrandLabel.brand
    )
    return conn.execute(query).mappings().all()


def get_recent_labeling_samples(
    days: int = 1,
    sample_size: int = 10,
    database_url: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Get samples of recent labeling results from database, with who decided each and each
    brand's judgment. Rows a human decided are left out.

    Args:
        days: Number of days to look back
        sample_size: Max articles per category
        database_url: Database to read (default: DATABASE_URL)

    Returns:
        Tuple of (labeled_sample, fp_sample, skipped_sample). Empty when no database is
        configured. The false_positive sample is empty until #256.

    Raises:
        SampleQueryError: A query failed. Samples without their judgments would read as if no
            brand had been judged, the input #252's false alarm was given, so the caller reports
            the failure instead.
    """
    from sqlalchemy import create_engine

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL not set, returning empty samples")
        return [], [], []

    engine = None
    try:
        engine = create_engine(database_url)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        with engine.connect() as conn:
            rows = {status: _article_rows(conn, status, since, sample_size)
                    for status in SAMPLE_STATUSES}
            every = [row["id"] for status_rows in rows.values() for row in status_rows]
            labeled = [row["id"] for row in rows["labeled"]]
            judgments = _judgment_rows(conn, every) if every else []
            labels = _label_rows(conn, labeled) if labeled else []
        return build_samples(rows, judgments, labels)
    except Exception as e:
        logger.error(f"Failed to get labeling samples: {e}")
        raise SampleQueryError(f"Failed to get labeling samples: {e}") from e
    finally:
        if engine is not None:
            engine.dispose()
