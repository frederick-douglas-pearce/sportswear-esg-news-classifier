"""Run two labeling prompt versions, or two models under one version, on the same articles and
summarize the differences.

This module makes no DB or API call of its own. `scripts/compare_prompts.py` reads the articles
and builds the two labelers; this module calls them, records each reply, and computes the report's
figures.

Each reply's article outcome comes from `dispose_reply`, the function the pipeline uses (D032).
The per-brand category and sentiment comparison reuses `src/migration_eval/compare.py`. Its
article outcome (labeled or false_positive) is not used: it cannot tell a judged skip from an
unjudged one, which is what prompt v1.11.0 changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Callable, Protocol

from src.data_collection.models import SKIP_NO_ESG_CATEGORIES, UNJUDGED_SKIP_REASONS
from src.labeling.disposition import FALSE_POSITIVE, LABELED, SKIPPED, dispose_reply
from src.labeling.labeler import REFUSAL_ERROR_TYPE, LabelingResult
from src.labeling.models import normalize_brand
from src.labeling.config import llm_cost_usd, worst_case_call_cost_usd
from src.migration_eval.compare import compare_article, decisions_from_response
from src.migration_eval.models import BrandDecision

# Article outcome classes. A skip is split by whether the reply judged sportswear status.
SKIPPED_JUDGED = "skipped_judged"
SKIPPED_UNJUDGED = "skipped_unjudged"
FAILED = "failed"
CLASSES = (LABELED, SKIPPED_JUDGED, SKIPPED_UNJUDGED, FALSE_POSITIVE, FAILED)

# Strata the report breaks out
SAMPLE = "sample"
MOST_BRANDS = "most_brands"
# Case strata whose expected brand (every brand, for a case that names none) should get no ESG
# category (D037's gate)
NO_CATEGORY_STRATA = frozenset(
    {"product_true", "product_false", "product_gray", "financial", "no_category"}
)
# The owner-ruling strata D040's roll-up scores (#230); ruling 9 is gray and has no expected outcome
RULING_PREFIX = "ruling_"
SCORED_RULINGS = frozenset(f"{RULING_PREFIX}{n}" for n in range(1, 9))

# Shortest quoted prompt text treated as an example an article could be the source of
_MIN_EXAMPLE_CHARS = 15
_EXAMPLE_MATCH_RATIO = 0.8


class Labeler(Protocol):
    """What the evaluation needs from `ArticleLabeler`."""

    model: str

    def label_article(
        self,
        title: str,
        content: str,
        brands: list[str],
        published_at: datetime | None = None,
        source_name: str | None = None,
    ) -> LabelingResult: ...


def disposition_class(status: str, reason: str | None) -> str:
    """The report's outcome class for a disposition's status and skip reason."""
    if status == SKIPPED:
        if reason == SKIP_NO_ESG_CATEGORIES:
            return SKIPPED_JUDGED
        if reason in UNJUDGED_SKIP_REASONS:
            return SKIPPED_UNJUDGED
        raise ValueError(f"Unrecognized skip reason {reason!r}")
    if status in (LABELED, FALSE_POSITIVE):
        return status
    raise ValueError(f"Unrecognized status {status!r}")


@dataclass(frozen=True)
class Expectation:
    """The verdict a case is expected to get for one brand under the candidate's rule."""

    brand: str
    is_sportswear_brand: bool
    # The applicable ESG categories the brand should get, or None when the case rules none
    categories: tuple[str, ...] | None = None
    # (category, sentiment) pairs for ruled categories whose sentiment the case also rules (#230)
    sentiments: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class SideSettings:
    """The model, output cap and effort one side of the comparison calls with."""

    model: str
    max_tokens: int
    # output_config.effort, or None for a model that does not take it
    effort: str | None = None


@dataclass
class EvalArticle:
    """One article to label, with what the report needs to know about it."""

    article_id: str
    title: str
    content: str
    brands: list[str]
    published_at: datetime | None = None
    source_name: str | None = None
    strata: tuple[str, ...] = ()
    repeats: int = 1
    expected: Expectation | None = None
    note: str = ""
    # The article's stored labeling status, and whether it is stored as labeled with brand labels
    # written under the baseline prompt version (the noise floor's population)
    stored_status: str | None = None
    stored_under_baseline: bool = False
    # A quoted example in the candidate prompt that this article looks like the source of
    prompt_example: str | None = None


@dataclass
class ReplyRecord:
    """What one reply decided, and what it cost."""

    outcome: str
    status: str | None = None
    reason: str | None = None
    error: str | None = None
    parse_failure: bool = False
    # The model declined the request (stop_reason "refusal"); not a parse failure
    refusal: bool = False
    stop_reason: str | None = None
    # normalized brand -> is_sportswear_brand, for brands the reply judged
    verdicts: dict[str, bool] = field(default_factory=dict)
    # normalized brand -> the reply's reasoning (and not_sportswear_reason) for it
    explanations: dict[str, str] = field(default_factory=dict)
    # normalized brand -> the content type the reply stated (v1.12.0 and later, D037)
    content_types: dict[str, str] = field(default_factory=dict)
    # normalized brand -> applicable categories kept, and those the content type dropped
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # normalized brand -> {applicable category kept: its sentiment}
    sentiments: dict[str, dict[str, int | None]] = field(default_factory=dict)
    gated: dict[str, tuple[str, ...]] = field(default_factory=dict)
    omitted: tuple[str, ...] = ()
    unset: tuple[str, ...] = ()
    unsent: tuple[str, ...] = ()
    conflicting: tuple[str, ...] = ()
    # Extra entries for a brand returned more than once (#201)
    duplicates: int = 0
    decisions: dict[str, BrandDecision] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def complete(self) -> bool:
        """The reply judged every brand it was sent: the pipeline's definition of a complete reply,
        the one `labeling_runs.incomplete_replies` counts. An entry with no verdict for a sent
        brand already puts that brand in `omitted`; `unset` entries are reported separately."""
        return self.outcome != FAILED and not self.omitted


@dataclass
class ArticleRun:
    """Both versions' replies for one article, one pair per repeat."""

    article: EvalArticle
    baseline: list[ReplyRecord] = field(default_factory=list)
    candidate: list[ReplyRecord] = field(default_factory=list)


def record_reply(sent_brands: list[str], result: LabelingResult) -> ReplyRecord:
    """Record one labeling result through `dispose_reply`.

    A failed result with output tokens is a parse failure: the API answered and the reply could
    not be parsed, unless the model declined, which is a refusal. A failed result without output
    tokens is an API or input error.
    """
    if not result.success or result.response is None:
        refusal = result.error_type == REFUSAL_ERROR_TYPE
        return ReplyRecord(
            outcome=FAILED,
            error=result.error,
            parse_failure=result.output_tokens > 0 and not refusal,
            refusal=refusal,
            stop_reason=result.stop_reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    analyses = result.response.brand_analyses
    disposition = dispose_reply(sent_brands, analyses)
    verdicts: dict[str, bool] = {}
    for analysis in disposition.non_sportswear:
        verdicts[normalize_brand(analysis.brand)] = False
    # Sportswear wins over a conflicting entry, as in dispose_reply
    for analysis in disposition.sportswear:
        verdicts[normalize_brand(analysis.brand)] = True
    explanations, content_types, categories, sentiments, gated = {}, {}, {}, {}, {}
    for analysis in analyses:
        key = normalize_brand(analysis.brand)
        text = analysis.reasoning
        if analysis.not_sportswear_reason:
            text = f"{text} [not sportswear: {analysis.not_sportswear_reason}]".strip()
        explanations.setdefault(key, text)
        if analysis.content_type is not None:
            content_types.setdefault(key, analysis.content_type.value)
        applicable = analysis.get_applicable_categories()
        categories.setdefault(key, tuple(applicable))
        sentiments.setdefault(key, {c: analysis.categories[c].sentiment for c in applicable})
        if analysis.gated_categories:
            gated.setdefault(key, tuple(analysis.gated_categories))
    counts = Counter(normalize_brand(a.brand) for a in analyses)

    return ReplyRecord(
        outcome=disposition_class(disposition.status, disposition.reason),
        status=disposition.status,
        reason=disposition.reason,
        verdicts=verdicts,
        explanations=explanations,
        content_types=content_types,
        categories=categories,
        sentiments=sentiments,
        gated=gated,
        omitted=tuple(disposition.omitted),
        unset=tuple(disposition.unset),
        unsent=tuple(disposition.unsent),
        conflicting=tuple(disposition.conflicting),
        duplicates=sum(n - 1 for n in counts.values()),
        decisions=decisions_from_response(result.response),
        stop_reason=result.stop_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def prompt_examples(prompt_text: str) -> list[str]:
    """Quoted strings in a prompt long enough to be an article title used as an example."""
    quoted = re.findall(r'"([^"\n]+)"', prompt_text)
    return [q for q in dict.fromkeys(quoted) if len(q) >= _MIN_EXAMPLE_CHARS]


def matching_prompt_example(title: str, examples: list[str]) -> str | None:
    """The prompt example an article title looks like the source of, if any."""
    key = title.casefold().strip()
    for example in examples:
        ex = example.casefold().strip()
        if ex in key or (len(key) >= _MIN_EXAMPLE_CHARS and key in ex):
            return example
        if SequenceMatcher(None, key, ex).ratio() >= _EXAMPLE_MATCH_RATIO:
            return example
    return None


def worst_case_call_cost(article: EvalArticle, prompt_chars: int, side: SideSettings) -> float:
    """An upper bound on one call's cost (`worst_case_call_cost_usd`) at the side's model and cap."""
    return worst_case_call_cost_usd(prompt_chars, len(article.content), side.max_tokens, side.model)


def worst_case_cost(
    articles: list[EvalArticle], calls: list[tuple[int, SideSettings]]
) -> float:
    """Worst-case cost of every call the run would make: one call per (prompt chars, side)."""
    return sum(
        article.repeats * worst_case_call_cost(article, chars, side)
        for article in articles
        for chars, side in calls
    )


def _label(labeler: Labeler, article: EvalArticle) -> ReplyRecord:
    result = labeler.label_article(
        title=article.title,
        content=article.content,
        brands=article.brands,
        published_at=article.published_at,
        source_name=article.source_name,
    )
    return record_reply(article.brands, result)


def run_articles(
    articles: list[EvalArticle],
    baseline: Labeler,
    candidate: Labeler,
    *,
    max_cost_usd: float,
    pair_bound: Callable[[EvalArticle], float],
    on_pair: Callable[[ArticleRun], None] | None = None,
) -> tuple[list[ArticleRun], bool]:
    """Label each article under both versions, back to back, once per repeat.

    Before each pair of calls, stops when the spend so far plus `pair_bound(article)` would
    exceed `max_cost_usd`. Returns the runs and whether the cap stopped the run early.
    """
    runs: list[ArticleRun] = []
    spent = 0.0
    for article in articles:
        run = ArticleRun(article)
        for _ in range(article.repeats):
            if spent + pair_bound(article) > max_cost_usd:
                if run.baseline:
                    runs.append(run)
                return runs, True
            for labeler, replies in ((baseline, run.baseline), (candidate, run.candidate)):
                reply = _label(labeler, article)
                replies.append(reply)
                spent += llm_cost_usd(reply.input_tokens, reply.output_tokens, labeler.model)
            if on_pair:
                on_pair(run)
        runs.append(run)
    return runs, False


# --- summary -------------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def _token_stats(replies: list[ReplyRecord], side: SideSettings) -> dict:
    """Token figures for one side. A reply is truncated when the API stopped it at max_tokens;
    output tokens include thinking, so a count at the cap is not the test."""
    outputs = [r.output_tokens for r in replies if r.output_tokens]
    near_floor = math.ceil(0.95 * side.max_tokens)
    truncated = [r for r in replies if r.stop_reason == "max_tokens"]
    return {
        "calls": len(replies),
        "input_tokens": sum(r.input_tokens for r in replies),
        "output_tokens": sum(r.output_tokens for r in replies),
        "output_mean": (sum(outputs) / len(outputs)) if outputs else None,
        "output_p95": _percentile(outputs, 0.95) if outputs else None,
        "output_max": max(outputs) if outputs else None,
        "truncated": len(truncated),
        "near_limit": sum(
            1 for r in replies if r.output_tokens >= near_floor and r.stop_reason != "max_tokens"
        ),
        "cost_usd": llm_cost_usd(
            sum(r.input_tokens for r in replies), sum(r.output_tokens for r in replies), side.model
        ),
    }


def _brief(run: ArticleRun) -> dict:
    return {"article_id": run.article.article_id, "title": run.article.title}


def _transition_detail(run: ArticleRun) -> dict:
    b, c = run.baseline[0], run.candidate[0]
    return {
        **_brief(run),
        "baseline": b.outcome,
        "candidate": c.outcome,
        "candidate_verdicts": c.verdicts,
        "candidate_explanations": c.explanations,
        "baseline_explanations": b.explanations,
    }


def _newly_labeled(runs: list[ArticleRun]) -> dict:
    """Non-labeled -> labeled, over the baseline's non-labeled replies, by baseline class."""
    by_class = {}
    for cls in (FALSE_POSITIVE, SKIPPED_JUDGED, SKIPPED_UNJUDGED):
        base = [r for r in runs if r.baseline[0].outcome == cls and r.candidate[0].outcome != FAILED]
        newly = sum(1 for r in base if r.candidate[0].outcome == LABELED)
        by_class[cls] = {"denominator": len(base), "newly_labeled": newly}
    denominator = sum(v["denominator"] for v in by_class.values())
    numerator = sum(v["newly_labeled"] for v in by_class.values())
    labeled = [r for r in runs if r.baseline[0].outcome == LABELED and r.candidate[0].outcome != FAILED]
    dropped = sum(1 for r in labeled if r.candidate[0].outcome != LABELED)
    return {
        "denominator": denominator,
        "newly_labeled": numerator,
        "rate": _rate(numerator, denominator),
        "by_baseline_class": by_class,
        "baseline_labeled": len(labeled),
        "dropped_labeled": dropped,
        "dropped_rate": _rate(dropped, len(labeled)),
    }


def summarize(runs: list[ArticleRun], *, baseline: SideSettings, candidate: SideSettings) -> dict:
    """The report's figures. Transitions use each article's first repeat."""
    sides = {"baseline": baseline, "candidate": candidate}
    paired = [r for r in runs if r.baseline and r.candidate]

    transitions = Counter(f"{r.baseline[0].outcome} -> {r.candidate[0].outcome}" for r in paired)
    to_labeled = [
        _transition_detail(r)
        for r in paired
        if r.baseline[0].outcome not in (LABELED, FAILED) and r.candidate[0].outcome == LABELED
    ]
    from_labeled = [
        _transition_detail(r)
        for r in paired
        if r.baseline[0].outcome == LABELED and r.candidate[0].outcome not in (LABELED, FAILED)
    ]

    unjudged_split = Counter(
        r.candidate[0].outcome for r in paired if r.baseline[0].outcome == SKIPPED_UNJUDGED
    )

    fp_flips, newly_judged = [], 0
    category_flips, one_sided_brands, sentiment_pairs, sentiment_equal = 0, 0, 0, 0
    for r in paired:
        b, c = r.baseline[0], r.candidate[0]
        if b.outcome == FAILED or c.outcome == FAILED:
            continue
        sent = {normalize_brand(name) for name in r.article.brands}
        for brand, verdict in c.verdicts.items():
            if brand not in sent:
                continue
            if brand not in b.verdicts:
                newly_judged += 1
            elif b.verdicts[brand] != verdict:
                fp_flips.append({
                    **_brief(r),
                    "brand": brand,
                    "baseline": b.verdicts[brand],
                    "candidate": verdict,
                    "baseline_explanation": b.explanations.get(brand, ""),
                    "candidate_explanation": c.explanations.get(brand, ""),
                })
        comparison = compare_article(r.article.article_id, r.article.title, b.decisions, c.decisions)
        for brand in comparison.brand_comparisons:
            # A brand only one side labeled is an outcome change, counted apart from category flips
            if not (brand.in_baseline and brand.in_candidate):
                one_sided_brands += 1
                continue
            for diff in brand.category_diffs:
                if diff.baseline_applies != diff.candidate_applies:
                    category_flips += 1
                elif diff.baseline_applies:
                    sentiment_pairs += 1
                    sentiment_equal += diff.baseline_sentiment == diff.candidate_sentiment

    def per_version(side: str) -> dict:
        replies = [x for r in paired for x in getattr(r, side)]
        most_brands = [x for r in paired if MOST_BRANDS in r.article.strata for x in getattr(r, side)]
        answered = [x for x in replies if x.outcome != FAILED]
        return {
            "outcomes": {cls: sum(1 for x in replies if x.outcome == cls) for cls in CLASSES},
            "parse_failures": sum(1 for x in replies if x.parse_failure),
            "refusals": sum(1 for x in replies if x.refusal),
            "unjudged_rate": _rate(
                sum(1 for x in answered if x.outcome == SKIPPED_UNJUDGED), len(answered)
            ),
            "compliance": _rate(sum(1 for x in answered if x.complete), len(answered)),
            "incomplete_replies": sum(1 for x in answered if not x.complete),
            "replies_with_unset_entries": sum(1 for x in answered if x.unset),
            "replies_with_duplicates": sum(1 for x in answered if x.duplicates),
            "duplicate_entries": sum(x.duplicates for x in answered),
            "replies_with_unsent_brands": sum(1 for x in answered if x.unsent),
            "tokens": _token_stats(replies, sides[side]),
            "tokens_most_brands": _token_stats(most_brands, sides[side]),
            "content_types": dict(
                sorted(Counter(t for x in answered for t in x.content_types.values()).items())
            ),
        }

    def gated(side: str) -> list[dict]:
        """Every applicable category a content type dropped, for a human read (D037 §3)."""
        return [
            {
                **_brief(r),
                "brand": brand,
                "content_type": x.content_types.get(brand),
                "categories": list(cats),
                "explanation": x.explanations.get(brand, ""),
            }
            for r in paired
            for x in getattr(r, side)
            for brand, cats in x.gated.items()
        ]

    def case_categories(side: str) -> list[dict]:
        """Categories kept on the expected brand of a Product, financial or no-category case, or
        on any brand of one that names none; there should be none."""
        found = []
        for r in paired:
            if not NO_CATEGORY_STRATA & set(r.article.strata):
                continue
            exp = r.article.expected
            for x in getattr(r, side):
                keys = [normalize_brand(exp.brand)] if exp else list(x.categories)
                for key in keys:
                    if x.categories.get(key):
                        found.append({
                            **_brief(r),
                            "brand": exp.brand if exp else key,
                            "categories": list(x.categories[key]),
                            "explanation": x.explanations.get(key, ""),
                        })
        return found

    def expected_categories(side: str) -> dict:
        """Per case that rules the brand's categories: matched if every answered reply gives the
        brand exactly those categories, unanswered if every reply failed. A reply that leaves the
        brand out or calls it not sportswear gives it none."""
        matched, misses, unanswered = 0, [], []
        for r in paired:
            exp = r.article.expected
            if exp is None or exp.categories is None:
                continue
            key = normalize_brand(exp.brand)
            want = set(exp.categories)
            answered = [x for x in getattr(r, side) if x.outcome != FAILED]
            if not answered:
                unanswered.append({**_brief(r), "brand": exp.brand})
                continue
            wrong = [x for x in answered if set(x.categories.get(key, ())) != want]
            if not wrong:
                matched += 1
                continue
            for x in wrong:
                got = set(x.categories.get(key, ()))
                misses.append({
                    **_brief(r),
                    "brand": exp.brand,
                    "content_type": x.content_types.get(key),
                    "expected": sorted(want),
                    "missing": sorted(want - got),
                    "extra": sorted(got - want),
                    "explanation": x.explanations.get(key, ""),
                })
        cases = matched + len({m["article_id"] for m in misses}) + len(unanswered)
        return {
            "cases": cases,
            "matched": matched,
            "unanswered": len(unanswered),
            "misses": misses,
            "unanswered_cases": unanswered,
        }

    def expected_sentiments(side: str) -> dict:
        """Per case that rules sentiments: matched if every answered reply that kept a ruled
        category gives it the ruled sentiment, mismatched if any gives another, not assessed if an
        answered reply exists and none kept one (that case is a miss under expected categories),
        and unanswered if every reply failed."""
        matched, mismatched, not_assessed, unanswered = [], [], [], []
        for r in paired:
            exp = r.article.expected
            if exp is None or not exp.sentiments:
                continue
            key = normalize_brand(exp.brand)
            entry = {**_brief(r), "brand": exp.brand, "strata": sorted(r.article.strata)}
            answered = [x for x in getattr(r, side) if x.outcome != FAILED]
            if not answered:
                unanswered.append(entry)
                continue
            assessed, wrong = 0, []
            for x in answered:
                kept = x.sentiments.get(key, {})
                for category, want in exp.sentiments:
                    if category not in kept:
                        continue
                    assessed += 1
                    if kept[category] != want:
                        wrong.append({"category": category, "expected": want, "got": kept[category]})
            if wrong:
                mismatched.append({
                    **entry,
                    "mismatches": wrong,
                    "explanation": next(
                        (x.explanations.get(key, "") for x in getattr(r, side) if key in x.explanations),
                        "",
                    ),
                })
            elif assessed:
                matched.append(entry)
            else:
                not_assessed.append(entry)
        return {
            "cases": len(matched) + len(mismatched) + len(not_assessed) + len(unanswered),
            "matched": len(matched),
            "mismatched": len(mismatched),
            "not_assessed": len(not_assessed),
            "unanswered": len(unanswered),
            "misses": mismatched,
            "not_assessed_cases": not_assessed,
            "unanswered_cases": unanswered,
        }

    def rulings(side: str) -> dict:
        """D040's per-ruling roll-up over the `ruling_1` to `ruling_8` cases with an expected
        outcome (gray and prompt-example cases excluded). A case passes if it has an answered
        reply and every answered reply judges the expected brand on the expected side, gives it
        exactly the ruled categories (where ruled) and each ruled sentiment (where ruled). A case
        whose replies all failed is unanswered, counted apart and outside the rate."""
        by_ruling: dict[str, dict[str, int]] = {}
        passed, failed, unanswered = [], [], []
        for r in paired:
            exp = r.article.expected
            ruled = sorted(set(r.article.strata) & SCORED_RULINGS)
            if exp is None or r.article.prompt_example or not ruled:
                continue
            key = normalize_brand(exp.brand)
            answered = [x for x in getattr(r, side) if x.outcome != FAILED]

            def passes(x: ReplyRecord) -> bool:
                if x.verdicts.get(key) != exp.is_sportswear_brand:
                    return False
                if exp.categories is not None and set(x.categories.get(key, ())) != set(exp.categories):
                    return False
                kept = x.sentiments.get(key, {})
                return all(kept.get(c) == want for c, want in exp.sentiments or ())

            entry = {**_brief(r), "brand": exp.brand, "rulings": ruled}
            if not answered:
                outcome, bucket = "unanswered", unanswered
            elif all(passes(x) for x in answered):
                outcome, bucket = "passed", passed
            else:
                outcome, bucket = "failed", failed
            bucket.append(entry)
            for ruling in ruled:
                counts = by_ruling.setdefault(ruling, {"passed": 0, "failed": 0, "unanswered": 0})
                counts[outcome] += 1
        return {
            "by_ruling": dict(sorted(by_ruling.items(), key=lambda kv: int(kv[0].removeprefix(RULING_PREFIX)))),
            "cases": len(passed) + len(failed) + len(unanswered),
            "passed": len(passed),
            "failed": len(failed),
            "unanswered": len(unanswered),
            "rate": _rate(len(passed), len(passed) + len(failed)),
            "failed_cases": failed,
            "unanswered_cases": unanswered,
        }

    stability = []
    for r in paired:
        if r.article.repeats < 2:
            continue
        b_outcomes = [x.outcome for x in r.baseline]
        c_outcomes = [x.outcome for x in r.candidate]
        # A case the spend cap cut short has fewer replies than repeats; it is never consistent
        complete = len(b_outcomes) == len(c_outcomes) == r.article.repeats
        stability.append({
            **_brief(r),
            "baseline": b_outcomes,
            "candidate": c_outcomes,
            "complete": complete,
            "baseline_consistent": complete and len(set(b_outcomes)) == 1,
            "candidate_consistent": complete and len(set(c_outcomes)) == 1,
        })

    def accuracy(side: str) -> dict:
        """Per case: correct if every reply that judged the expected brand is on the expected
        side, wrong if any is not, not judged if none judged it (v1.10.0 omits a sportswear brand
        with no ESG content by instruction, so its omissions are not counted as misses)."""
        correct, wrong, not_judged = [], [], []
        for r in paired:
            exp = r.article.expected
            if exp is None or r.article.prompt_example:
                continue
            key = normalize_brand(exp.brand)
            replies = getattr(r, side)
            verdicts = [x.verdicts[key] for x in replies if key in x.verdicts]
            entry = {
                **_brief(r),
                "brand": exp.brand,
                "strata": sorted(r.article.strata),
                "expected": exp.is_sportswear_brand,
                "got": verdicts,
                "explanation": next(
                    (x.explanations.get(key, "") for x in replies if key in x.verdicts), ""
                ),
            }
            if not verdicts:
                not_judged.append(entry)
            elif all(v == exp.is_sportswear_brand for v in verdicts):
                correct.append(entry)
            else:
                wrong.append(entry)
        judged = len(correct) + len(wrong)
        # What answering `true` for every judged case would score: the bar a rule has to beat
        always_true = sum(1 for e in correct + wrong if e["expected"])
        by_stratum: dict[str, dict[str, int]] = {}
        for outcome, entries in (("correct", correct), ("wrong", wrong), ("not_judged", not_judged)):
            for e in entries:
                for stratum in e["strata"]:
                    counts = by_stratum.setdefault(stratum, {"correct": 0, "wrong": 0, "not_judged": 0})
                    counts[outcome] += 1
        return {
            "by_stratum": dict(sorted(by_stratum.items())),
            "always_true": always_true,
            "always_true_rate": _rate(always_true, judged),
            "cases": judged + len(not_judged),
            "correct": len(correct),
            "wrong": len(wrong),
            "not_judged": len(not_judged),
            "rate": _rate(len(correct), judged),
            "misses": wrong,
            "not_judged_cases": not_judged,
        }

    sample = [r for r in paired if SAMPLE in r.article.strata]
    # Noise floor: random-sample articles stored as labeled under the baseline prompt whose fresh
    # baseline call answered, and how many of those it still labels. An article only in the named
    # cases is left out (they were chosen as hard); a named case the sample also drew is counted.
    floor = [r for r in sample if r.article.stored_under_baseline and r.baseline[0].outcome != FAILED]
    floor_agree = sum(1 for r in floor if r.baseline[0].outcome == LABELED)

    return {
        "articles": len(paired),
        "baseline": per_version("baseline"),
        "candidate": per_version("candidate"),
        "transitions": dict(sorted(transitions.items())),
        "newly_labeled_sample": _newly_labeled(sample),
        "newly_labeled_all": _newly_labeled(paired),
        "non_labeled_to_labeled": to_labeled,
        "labeled_to_non_labeled": from_labeled,
        "baseline_unjudged_split": dict(unjudged_split),
        "fp_flips": fp_flips,
        "brands_newly_judged": newly_judged,
        "category_flips": category_flips,
        "brands_labeled_by_one_side": one_sided_brands,
        "sentiment_agreement": _rate(sentiment_equal, sentiment_pairs),
        "stability": stability,
        "product_accuracy": {"baseline": accuracy("baseline"), "candidate": accuracy("candidate")},
        "case_categories": {"baseline": case_categories("baseline"), "candidate": case_categories("candidate")},
        "expected_categories": {
            "baseline": expected_categories("baseline"),
            "candidate": expected_categories("candidate"),
        },
        "expected_sentiments": {
            "baseline": expected_sentiments("baseline"),
            "candidate": expected_sentiments("candidate"),
        },
        "rulings": {"baseline": rulings("baseline"), "candidate": rulings("candidate")},
        "gated_categories": {"baseline": gated("baseline"), "candidate": gated("candidate")},
        "prompt_example_cases": [
            {**_brief(r), "example": r.article.prompt_example}
            for r in paired
            if r.article.prompt_example
        ],
        "noise_floor": {
            "articles": len(floor),
            "still_labeled": floor_agree,
            "rate": _rate(floor_agree, len(floor)),
        },
    }

