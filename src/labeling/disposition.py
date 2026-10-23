"""Article-level disposition of a labeling reply (#192, D032).

One pure function decides the status an article gets from the LLM's reply, under every
prompt version. The pipeline calls it, and #192 PR2's prompt evaluation is to call it too;
src/migration_eval/compare.py and scripts/eval_model_migration.py do not use it.
"""

from dataclasses import dataclass, field

from src.data_collection.models import (
    SKIP_INCOMPLETE_JUDGMENT,
    SKIP_NO_ESG_CATEGORIES,
    SKIP_NO_ESG_CONTENT,
)

from .models import BrandAnalysis, normalize_brand

LABELED = "labeled"
SKIPPED = "skipped"
FALSE_POSITIVE = "false_positive"

# Longest false_positive reason stored in articles.labeling_error
_FP_REASON_MAX_CHARS = 500


@dataclass(frozen=True)
class ReplyDisposition:
    """What a reply decides for its article, and which brands it left out."""

    status: str
    reason: str | None
    sportswear: list[BrandAnalysis] = field(default_factory=list)
    non_sportswear: list[BrandAnalysis] = field(default_factory=list)
    # Brands sent in the prompt that no analysis in the reply covers
    omitted: list[str] = field(default_factory=list)
    # Brands in the reply that were not sent; they cover no sent brand
    unsent: list[str] = field(default_factory=list)
    # Brands returned both as sportswear and as not sportswear; sportswear wins
    conflicting: list[str] = field(default_factory=list)
    # Brands returned with no is_sportswear_brand and no applicable category; they count
    # as left out
    unset: list[str] = field(default_factory=list)


def _states_a_verdict(analysis: BrandAnalysis) -> bool:
    """Whether the reply judged this brand, rather than relying on the field default. A content
    type (v1.12.0 and later, D037) is a verdict."""
    return (
        analysis.content_type is not None
        or "is_sportswear_brand" in analysis.model_fields_set
        or bool(analysis.get_applicable_categories())
    )


def dispose_reply(sent_brands: list[str], analyses: list[BrandAnalysis]) -> ReplyDisposition:
    """Decide an article's status from the brands sent and the analyses returned.

    - No analyses, or none that states a verdict: `skipped` / SKIP_NO_ESG_CONTENT (unjudged).
    - A sportswear brand with an applicable category: `labeled`.
    - Sportswear brands, none with an applicable category: `skipped` / SKIP_NO_ESG_CATEGORIES.
    - Only non-sportswear brands, and a sent brand left out: `skipped` /
      SKIP_INCOMPLETE_JUDGMENT (unjudged), since the brand left out may be the sportswear one.
    - Only non-sportswear brands, covering every sent brand: `false_positive`.

    Brand names are matched by `normalize_brand`; a name the reply spells differently
    counts as left out, which fails safe to unjudged. An analysis that neither sets
    `is_sportswear_brand` (the model defaults it to True) nor has an applicable category nor a
    content type stated no verdict, so it counts as left out too.
    """
    unset = [a.brand for a in analyses if not _states_a_verdict(a)]
    analyses = [a for a in analyses if _states_a_verdict(a)]
    sent_keys = {normalize_brand(b): b for b in sent_brands}
    returned_keys = {normalize_brand(a.brand) for a in analyses}
    omitted = [name for key, name in sent_keys.items() if key not in returned_keys]
    unsent = [a.brand for a in analyses if normalize_brand(a.brand) not in sent_keys]

    if not analyses:
        return ReplyDisposition(
            SKIPPED, SKIP_NO_ESG_CONTENT, omitted=omitted, unsent=unsent, unset=unset
        )

    sportswear = [a for a in analyses if a.is_sportswear_brand]
    non_sportswear = [a for a in analyses if not a.is_sportswear_brand]
    sportswear_keys = {normalize_brand(a.brand) for a in sportswear}
    conflicting = sorted(
        {a.brand for a in non_sportswear if normalize_brand(a.brand) in sportswear_keys}
    )
    lists = {
        "sportswear": sportswear,
        "non_sportswear": non_sportswear,
        "omitted": omitted,
        "unsent": unsent,
        "conflicting": conflicting,
        "unset": unset,
    }

    if sportswear:
        if any(a.get_applicable_categories() for a in sportswear):
            return ReplyDisposition(LABELED, None, **lists)
        return ReplyDisposition(SKIPPED, SKIP_NO_ESG_CATEGORIES, **lists)

    if omitted:
        return ReplyDisposition(SKIPPED, SKIP_INCOMPLETE_JUDGMENT, **lists)

    reason = "; ".join(f"{a.brand}: {a.not_sportswear_reason}" for a in non_sportswear)
    return ReplyDisposition(FALSE_POSITIVE, reason[:_FP_REASON_MAX_CHARS], **lists)
