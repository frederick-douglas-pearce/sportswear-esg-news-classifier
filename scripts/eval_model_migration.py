#!/usr/bin/env python3
"""Retrospective model-migration evaluation harness.

Re-labels a stratified sample of already-labeled articles with a candidate
model and diffs the result against the stored baseline labels, to gate a model
migration (Sonnet 4 -> 4.6) before flipping production defaults.

Why retrospective: the outgoing model has been retired, so a live A/B is
impossible. The stored labels are the only record of the old model's behavior.

Methodology guards (see issue #38):
  * Scope to ONE production regime -- baseline labels produced by the
    --baseline-model under --prompt-version -- and re-label with that same
    prompt, so the diff isolates the MODEL change (not prompt drift) and
    matches what production will actually run post-flip.
  * The baseline must reflect a genuine LLM decision. We exclude articles whose
    false_positive status came from the FP classifier pre-filter
    (classifier_predictions.action_taken == 'skipped_llm') or from
    deduplication, and we route human-reviewed labels into a separate anchor
    stratum.
  * A 'pending' (never-labeled) stratum has no baseline; it is re-labeled and
    emitted for human-only adjudication, to catch the candidate newly ACCEPTING
    junk/financial articles that never entered the labeled pool.

The numeric verdict is an advisory noise-screen. The real gate is human
adjudication of the disagreement and pending tables in the markdown report.

Usage:
    # Validate sampling + estimate cost without spending API tokens:
    uv run python scripts/eval_model_migration.py --dry-run

    # Run the eval against Sonnet 4.6:
    uv run python scripts/eval_model_migration.py \
        --candidate-model claude-sonnet-4-6 --sample-size 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

# Add project root to path (matches the other scripts/ entry points).
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.database import db
from src.data_collection.models import Article, BrandLabel, ClassifierPrediction
from src.labeling.labeler import ArticleLabeler
from src.migration_eval.compare import build_verdict, compare_article, decisions_from_response
from src.migration_eval.models import CATEGORIES, ArticleComparison, BrandDecision, Thresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_model_migration")

# Default outgoing/baseline regime.
DEFAULT_BASELINE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CANDIDATE_MODEL = "claude-sonnet-4-6"
DEFAULT_PROMPT_VERSION = "v1.8.0"

OUTPUT_DIR = Path("data/evaluation/migration")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _content_for(article: Article) -> str | None:
    return article.full_content or article.description


def _sample_ids(ids: list[UUID], n: int, rng: random.Random) -> list[UUID]:
    if n <= 0:
        return []
    if len(ids) <= n:
        return list(ids)
    return rng.sample(ids, n)


def sample_strata(
    session,
    *,
    baseline_model: str,
    prompt_version: str,
    n_labeled: int,
    n_false_positive: int,
    n_pending: int,
    n_anchor_cap: int,
    rng: random.Random,
) -> dict[str, list[UUID]]:
    """Return article ids per stratum, applying the methodology guards."""
    # Articles excluded from the false-positive baseline: status set by the FP
    # classifier rather than the LLM.
    fp_skipped = {
        row[0]
        for row in session.query(ClassifierPrediction.article_id)
        .filter(
            ClassifierPrediction.classifier_type == "fp",
            ClassifierPrediction.action_taken == "skipped_llm",
        )
        .all()
    }

    # Articles whose labels include a human review -> anchor stratum.
    human_article_ids = {
        row[0]
        for row in session.query(BrandLabel.article_id)
        .filter(BrandLabel.human_reviewed.is_(True))
        .all()
    }

    # labeled stratum: model+prompt match, no human review.
    labeled_ids = [
        row[0]
        for row in session.query(BrandLabel.article_id)
        .join(Article, Article.id == BrandLabel.article_id)
        .filter(
            Article.labeling_status == "labeled",
            BrandLabel.model_version == baseline_model,
            BrandLabel.prompt_version == prompt_version,
        )
        .distinct()
        .all()
        if row[0] not in human_article_ids
    ]

    # false_positive stratum: genuine LLM-decided false positives.
    # We exclude FP-classifier pre-filter skips (no LLM call) so the baseline is
    # a real model decision. We do NOT date-scope these: false_positive status
    # records no timestamp, and most in-window FPs are now caught by the FP
    # classifier, leaving too few LLM FPs in the v1.8.0 window. The baseline FP
    # set therefore spans prompt versions, but the confound is conservative —
    # older prompts were *looser* on false positives, so a baseline FP is, if
    # anything, even more likely to be FP under the stricter v1.8.0; a flip to
    # "labeled" remains a genuine signal worth human review.
    fp_ids = [
        row[0]
        for row in session.query(Article.id)
        .filter(Article.labeling_status == "false_positive")
        .all()
        if row[0] not in fp_skipped
    ]

    # pending stratum: never labeled, has content + brands to analyze.
    pending_ids = [
        a.id
        for a in session.query(Article)
        .filter(Article.labeling_status == "pending")
        .all()
        if _content_for(a) and a.brands_mentioned
    ]

    anchor_ids = list(human_article_ids)

    return {
        "labeled": _sample_ids(labeled_ids, n_labeled, rng),
        "false_positive": _sample_ids(fp_ids, n_false_positive, rng),
        "anchor": _sample_ids(anchor_ids, n_anchor_cap, rng),
        "pending": _sample_ids(pending_ids, n_pending, rng),
    }


def baseline_decisions_for(session, article_id: UUID, baseline_model: str) -> dict[str, BrandDecision]:
    """Reconstruct effective baseline decisions from stored brand_labels rows."""
    rows = (
        session.query(BrandLabel)
        .filter(BrandLabel.article_id == article_id)
        .all()
    )
    decisions: dict[str, BrandDecision] = {}
    for bl in rows:
        applies = {
            "environmental": bool(bl.environmental),
            "social": bool(bl.social),
            "governance": bool(bl.governance),
            "digital_transformation": bool(bl.digital_transformation),
        }
        sentiment = {
            "environmental": bl.environmental_sentiment,
            "social": bl.social_sentiment,
            "governance": bl.governance_sentiment,
            "digital_transformation": bl.digital_sentiment,
        }
        # Mirror save filter: a stored row always has >=1 applicable category,
        # but guard anyway for safety.
        if any(applies.values()):
            decisions[bl.brand.lower()] = BrandDecision(
                brand=bl.brand,
                applies=applies,
                sentiment=sentiment,
                confidence=bl.confidence_score or 0.0,
                reasoning=bl.reasoning or "",
            )
    return decisions


# --------------------------------------------------------------------------- #
# Re-labeling
# --------------------------------------------------------------------------- #


def relabel_article(labeler: ArticleLabeler, article: dict) -> tuple[ArticleComparison | None, dict]:
    """Run the candidate model on one article; return (comparison-inputs, meta).

    Returns the candidate decisions + parse status + reasoning + timing so the
    caller can build the ArticleComparison (which also needs the baseline).
    """
    content = article["content"]
    t0 = time.monotonic()
    result = labeler.label_article(
        title=article["title"],
        content=content,
        brands=article["brands_mentioned"],
        published_at=article["published_at"],
        source_name=article["source_name"],
    )
    latency = time.monotonic() - t0

    if not result.success or result.response is None:
        return None, {
            "parse_ok": False,
            "error": result.error or "labeling failed",
            "latency": latency,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "candidate_decisions": {},
            "candidate_reasoning": "",
        }

    decisions = decisions_from_response(result.response)
    return None, {
        "parse_ok": True,
        "error": None,
        "latency": latency,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "candidate_decisions": decisions,
        "candidate_reasoning": result.response.article_summary,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _comparison_to_dict(c: ArticleComparison) -> dict:
    return dataclasses.asdict(c)


def write_reports(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    verdict,
    comparisons: list[ArticleComparison],
    pending: list[dict],
    timestamp: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"migration_eval_{timestamp}.json"
    md_path = output_dir / f"migration_eval_{timestamp}.md"

    payload = {
        "generated_at": timestamp,
        "config": {
            "baseline_model": args.baseline_model,
            "candidate_model": args.candidate_model,
            "prompt_version": args.prompt_version,
            "seed": args.seed,
        },
        "verdict": dataclasses.asdict(verdict),
        "comparisons": [_comparison_to_dict(c) for c in comparisons],
        "pending_review": pending,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md_path.write_text(_render_markdown(args, verdict, comparisons, pending, timestamp))
    return json_path, md_path


def _render_markdown(args, verdict, comparisons, pending, timestamp) -> str:
    v = verdict
    lines: list[str] = []
    lines.append(f"# Model Migration Eval — {args.candidate_model}")
    lines.append("")
    lines.append(f"_Generated {timestamp}_")
    lines.append("")
    lines.append(
        f"Baseline `{args.baseline_model}` vs candidate `{args.candidate_model}`, "
        f"both on prompt `{args.prompt_version}`."
    )
    lines.append("")

    # Verdict banner.
    if v.hard_fail:
        banner = "🛑 **HARD FAIL** — integration broke; do not promote."
    elif v.passed_advisory:
        banner = "✅ **Advisory thresholds passed** — still requires human sign-off below."
    else:
        banner = "⚠️ **Flags raised** — investigate before promoting."
    lines.append(f"## Verdict: {banner}")
    lines.append("")
    lines.append(
        "> The numeric verdict is an advisory noise-screen. A clean diff only proves the "
        "candidate *mimics* the old model (including its known failure modes). **The gate is "
        "human review of the Disagreements and Pending tables below.**"
    )
    lines.append("")
    if v.flags:
        lines.append("**Flags:**")
        for f in v.flags:
            lines.append(f"- {f}")
        lines.append("")

    # Summary metrics.
    c = v.outcome_confusion
    lines.append("## Summary metrics")
    lines.append("")
    lines.append(f"- Articles compared: **{v.n_compared}** (parse failures: {v.n_parse_failures})")
    lines.append(f"- Outcome disagreement rate: **{v.outcome_disagreement_rate:.1%}**")
    lines.append(
        f"- Newly-labeled rate (baseline FP → candidate labeled, *scorecard-inflation risk*): "
        f"**{v.newly_labeled_rate:.1%}**"
    )
    lines.append(f"- Dropped-label rate (baseline labeled → candidate FP): **{v.dropped_label_rate:.1%}**")
    lines.append(f"- Category macro-F1 (candidate vs baseline reference): **{v.category_f1_macro:.3f}**")
    lines.append(
        f"- Sentiment agreement: exact **{v.sentiment_exact_rate:.1%}**, "
        f"within ±1 **{v.sentiment_within1_rate:.1%}**"
    )
    lines.append(
        f"- Human-anchor disagreements: **{v.n_human_anchor_disagreements}/{v.n_human_anchors}** "
        f"(closer-to-truth subset)"
    )
    lines.append(
        f"- Tokens: {v.total_input_tokens:,} in / {v.total_output_tokens:,} out · "
        f"est. cost **${v.estimated_cost_usd:.2f}** · mean latency {v.mean_latency_s:.2f}s"
    )
    lines.append("")
    lines.append("### Outcome confusion")
    lines.append("")
    lines.append("| baseline ↓ / candidate → | labeled | false_positive |")
    lines.append("|---|---|---|")
    lines.append(f"| **labeled** | {c['ll']} | {c['lf']} |")
    lines.append(f"| **false_positive** | {c['fl']} | {c['ff']} |")
    lines.append("")
    lines.append("### Per-category agreement")
    lines.append("")
    lines.append("| category | precision | recall | F1 | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat in CATEGORIES:
        m = v.category_metrics[cat]
        lines.append(
            f"| {cat} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{int(m['tp'])} | {int(m['fp'])} | {int(m['fn'])} |"
        )
    lines.append("")

    # Disagreement table.
    disagreements = [c for c in comparisons if c.has_disagreement]
    lines.append(f"## Disagreements to review ({len(disagreements)})")
    lines.append("")
    if not disagreements:
        lines.append("_None._")
    for comp in disagreements:
        anchor = " · 🔬 human-anchor" if comp.is_human_anchor else ""
        if not comp.parse_ok:
            lines.append(f"### ❌ {comp.title}{anchor}")
            lines.append(f"- **Parse failure:** {comp.error}")
            lines.append("")
            continue
        flip = "" if comp.outcome_match else f" — **OUTCOME FLIP: {comp.baseline_outcome} → {comp.candidate_outcome}**"
        lines.append(f"### {comp.title}{anchor}{flip}")
        for bc in comp.brand_comparisons:
            if bc.status != "shared":
                lines.append(f"- **{bc.brand}**: {bc.status}")
                continue
            cat_changes = []
            for d in bc.category_diffs:
                if not d.applies_match:
                    cat_changes.append(
                        f"{d.category}: applies {d.baseline_applies}→{d.candidate_applies}"
                    )
                elif d.baseline_applies and not d.sentiment_match:
                    cat_changes.append(
                        f"{d.category}: sentiment {d.baseline_sentiment}→{d.candidate_sentiment}"
                    )
            if cat_changes:
                lines.append(f"- **{bc.brand}**: " + "; ".join(cat_changes))
        if comp.candidate_reasoning:
            lines.append(f"  - _candidate summary:_ {comp.candidate_reasoning}")
        lines.append("")

    # Pending (human-only) stratum.
    lines.append(f"## Pending articles — candidate raw output ({len(pending)})")
    lines.append("")
    lines.append(
        "_No baseline exists for these (never labeled). Review whether the candidate "
        "correctly REJECTS junk/financial articles vs newly accepting them._"
    )
    lines.append("")
    for p in pending:
        outcome = p["candidate_outcome"]
        brands = ", ".join(p["candidate_brands"]) if p["candidate_brands"] else "—"
        flag = "⚠️ ACCEPTED" if outcome == "labeled" else "rejected"
        lines.append(f"- [{flag}] **{p['title']}** → brands: {brands}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-model", default=DEFAULT_CANDIDATE_MODEL)
    p.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    p.add_argument("--sample-size", type=int, default=300, help="Total comparable budget (labeled+FP)")
    p.add_argument("--labeled-frac", type=float, default=0.55, help="Fraction of comparable budget for labeled stratum")
    p.add_argument("--pending", type=int, default=40, help="Pending (human-only) stratum size")
    p.add_argument("--anchor-cap", type=int, default=40, help="Max human-reviewed anchor articles")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--dry-run", action="store_true", help="Sample + estimate only; no API calls")
    # Threshold overrides.
    p.add_argument("--max-outcome-disagreement", type=float, default=0.10)
    p.add_argument("--max-newly-labeled-rate", type=float, default=0.10)
    p.add_argument("--min-category-f1", type=float, default=0.85)
    p.add_argument("--min-sentiment-exact", type=float, default=0.80)
    return p.parse_args()


def _load_article_payloads(session, ids: list[UUID]) -> list[dict]:
    payloads = []
    for a in session.query(Article).filter(Article.id.in_(ids)).all():
        content = _content_for(a)
        if not content or not a.brands_mentioned:
            continue
        payloads.append(
            {
                "id": a.id,
                "title": a.title,
                "content": content,
                "brands_mentioned": list(a.brands_mentioned),
                "published_at": a.published_at,
                "source_name": a.source_name,
            }
        )
    return payloads


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    db.init_db()

    n_labeled = int(args.sample_size * args.labeled_frac)
    n_fp = args.sample_size - n_labeled

    with db.get_session() as session:
        strata = sample_strata(
            session,
            baseline_model=args.baseline_model,
            prompt_version=args.prompt_version,
            n_labeled=n_labeled,
            n_false_positive=n_fp,
            n_pending=args.pending,
            n_anchor_cap=args.anchor_cap,
            rng=rng,
        )

        logger.info("Sampled strata:")
        for name, ids in strata.items():
            logger.info("  %-15s %d articles", name, len(ids))

        # Build payloads + baselines while the session is open.
        comparable_ids = strata["labeled"] + strata["false_positive"] + strata["anchor"]
        payloads = {p["id"]: p for p in _load_article_payloads(session, comparable_ids)}
        pending_payloads = _load_article_payloads(session, strata["pending"])
        anchor_set = set(strata["anchor"])
        baselines = {
            aid: baseline_decisions_for(session, aid, args.baseline_model)
            for aid in payloads
        }

    n_comparable = len(payloads)
    n_pending = len(pending_payloads)
    if args.dry_run:
        # ~ rough token estimate: 3k in / 0.8k out per article.
        est_articles = n_comparable + n_pending
        est_cost = est_articles * (3000 / 1e6 * 3.0 + 800 / 1e6 * 15.0)
        logger.info(
            "DRY RUN: would label %d comparable + %d pending = %d articles (~$%.2f). No API calls made.",
            n_comparable, n_pending, est_articles, est_cost,
        )
        return 0

    labeler = ArticleLabeler(model=args.candidate_model, prompt_version=args.prompt_version)
    logger.info("Re-labeling %d comparable articles with %s …", n_comparable, args.candidate_model)

    comparisons: list[ArticleComparison] = []
    latencies: list[float] = []
    for i, (aid, payload) in enumerate(payloads.items(), 1):
        _, meta = relabel_article(labeler, payload)
        latencies.append(meta["latency"])
        comparisons.append(
            compare_article(
                str(aid),
                payload["title"],
                baselines[aid],
                meta["candidate_decisions"],
                baseline_reasoning="",
                candidate_reasoning=meta["candidate_reasoning"],
                is_human_anchor=aid in anchor_set,
                parse_ok=meta["parse_ok"],
                error=meta["error"],
            )
        )
        if i % 25 == 0:
            logger.info("  %d/%d", i, n_comparable)

    # Pending (human-only) stratum.
    pending_results: list[dict] = []
    logger.info("Re-labeling %d pending articles (human-only review) …", n_pending)
    for payload in pending_payloads:
        _, meta = relabel_article(labeler, payload)
        latencies.append(meta["latency"])
        decisions = meta["candidate_decisions"]
        pending_results.append(
            {
                "article_id": str(payload["id"]),
                "title": payload["title"],
                "candidate_outcome": "labeled" if decisions else "false_positive",
                "candidate_brands": [d.brand for d in decisions.values()],
                "parse_ok": meta["parse_ok"],
            }
        )

    stats = labeler.get_stats()
    verdict = build_verdict(
        comparisons,
        Thresholds(
            max_outcome_disagreement=args.max_outcome_disagreement,
            max_newly_labeled_rate=args.max_newly_labeled_rate,
            min_category_f1=args.min_category_f1,
            min_sentiment_exact=args.min_sentiment_exact,
        ),
        total_input_tokens=int(stats["total_input_tokens"]),
        total_output_tokens=int(stats["total_output_tokens"]),
        estimated_cost_usd=float(stats["estimated_cost_usd"]),
        mean_latency_s=(sum(latencies) / len(latencies) if latencies else 0.0),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = write_reports(
        Path(args.output_dir),
        args=args,
        verdict=verdict,
        comparisons=comparisons,
        pending=pending_results,
        timestamp=timestamp,
    )

    logger.info("=" * 60)
    logger.info("Verdict: %s", "HARD FAIL" if verdict.hard_fail else ("PASS (advisory)" if verdict.passed_advisory else "FLAGS RAISED"))
    for f in verdict.flags:
        logger.info("  %s", f)
    logger.info(
        "Outcome disagreement: %.1f%%  newly-labeled: %.1f%%  cat-F1: %.3f",
        verdict.outcome_disagreement_rate * 100,
        verdict.newly_labeled_rate * 100,
        verdict.category_f1_macro,
    )
    logger.info("Reports:\n  %s\n  %s", json_path, md_path)
    logger.info("=" * 60)
    return 1 if verdict.hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
