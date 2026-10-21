#!/usr/bin/env python3
"""Contrastive evaluation of two labeling prompt versions (#4, #192 PR2, D036).

Labels the same articles fresh under a baseline and a candidate prompt version, with the same
model, and reports what would change if the candidate were promoted. Nothing is written to the
database. A promotion is judged against D036's criteria by a human reading the report.

Articles come from any combination of:
  --case-file      named cases (ids, repeats, an expected verdict for one brand)
  --article-ids    ad-hoc ids
  --sample-size    a seeded random sample of recent LLM-judged articles
  --most-brands    the LLM-judged articles with the most brands mentioned
  --human-rulings  the articles a human ruled on in #182 (src/data_collection/provenance.py)

Usage:
    # Selection and worst-case cost, no API call:
    uv run python scripts/compare_prompts.py --candidate v1.11.0 \\
        --case-file prompts/labeling/eval_cases/product_and_stability.json \\
        --sample-size 100 --most-brands 10 --human-rulings --dry-run

    # The run (stops before exceeding --max-cost):
    uv run python scripts/compare_prompts.py --candidate v1.11.0 \\
        --case-file prompts/labeling/eval_cases/product_and_stability.json \\
        --sample-size 100 --most-brands 10 --human-rulings --max-cost 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection import provenance  # noqa: E402
from src.data_collection.database import db  # noqa: E402
from src.data_collection.models import Article, BrandLabel  # noqa: E402
from src.evaluation.comparator import (  # noqa: E402
    MOST_BRANDS,
    SAMPLE,
    EvalArticle,
    Expectation,
    matching_prompt_example,
    prompt_examples,
    run_articles,
    summarize,
    worst_case_call_cost,
    worst_case_cost,
)
from src.evaluation.diff_report import render_markdown  # noqa: E402
from src.labeling.article_input import article_content, has_sufficient_content  # noqa: E402
from src.labeling.config import TARGET_SPORTSWEAR_BRANDS, labeling_settings  # noqa: E402
from src.labeling.labeler import ArticleLabeler  # noqa: E402
from src.labeling.prompt_manager import PromptManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_prompts")

OUTPUT_DIR = Path("data/evaluation/comparisons")
CASES = "cases"
HUMAN_RULINGS = "human_rulings"
LLM_JUDGED_SOURCES = frozenset({provenance.LLM, provenance.UNJUDGED})


class EvalSetupError(RuntimeError):
    """The comparison cannot run as asked; nothing has been sent to the API."""


@dataclasses.dataclass
class Spec:
    """Why an article is in the run, before its content is loaded."""

    strata: set[str] = dataclasses.field(default_factory=set)
    repeats: int = 1
    expected: Expectation | None = None
    note: str = ""
    # A prompt example the case file says this article is the source of
    prompt_example: str | None = None


def parse_case_file(data: dict) -> dict[str, Spec]:
    """Case-file JSON -> {article_id: Spec}. Each case needs an article_id."""
    specs: dict[str, Spec] = {}
    for case in data.get("cases", []):
        article_id = case["article_id"]
        expected = case.get("expected")
        specs[article_id] = Spec(
            strata={CASES, *case.get("strata", [])},
            repeats=int(case.get("repeats", 1)),
            expected=(
                Expectation(expected["brand"], bool(expected["is_sportswear_brand"]))
                if expected
                else None
            ),
            note=case.get("note", ""),
            prompt_example=case.get("prompt_example"),
        )
    return specs


def merge_spec(specs: dict[str, Spec], article_id: str, stratum: str) -> None:
    """Add an article to a stratum, keeping what a case file already says about it."""
    specs.setdefault(article_id, Spec()).strata.add(stratum)


def check_prompt_versions(baseline: str, candidate: str, manager: PromptManager) -> dict[str, object]:
    """Load both versions, refusing equal or unloadable ones (D036 §4)."""
    if baseline == candidate:
        raise EvalSetupError(f"Baseline and candidate are the same version ({baseline})")
    loaded = {}
    for version in (baseline, candidate):
        try:
            loaded[version] = manager.load_version(version)
        except (FileNotFoundError, ValueError) as e:
            raise EvalSetupError(f"Prompt version {version} does not load: {e}") from e
        if loaded[version].version != version:
            raise EvalSetupError(f"Asked for {version}, loaded {loaded[version].version}")
    return loaded


def check_labeler(labeler: ArticleLabeler, version: str) -> None:
    """Refuse a labeler that fell back to the hardcoded prompt or loaded another version.

    `ArticleLabeler` logs a warning and carries on with the hardcoded prompt when a version
    fails to load, so a mistyped version would otherwise compare production with itself.
    """
    if labeler.loaded_prompt is None or labeler.prompt_version != version:
        raise EvalSetupError(
            f"Labeler for {version} loaded {labeler.prompt_version!r}; refusing to run"
        )


def _llm_judged_candidates(session, since: datetime):
    """(article_id, brand count) for recent articles whose status the LLM decided."""
    latest = provenance.latest_prediction_subquery("fp")
    rows = (
        session.query(
            Article.id,
            Article.labeling_status,
            Article.labeling_error,
            Article.brands_mentioned,
            latest.c.action_taken,
            latest.c.skip_reason,
        )
        .outerjoin(latest, latest.c.article_id == Article.id)
        .filter(Article.scrape_status == "success")
        .filter(
            Article.labeling_status.in_(
                [*provenance.FP_POSITIVE_STATUSES, provenance.FP_NEGATIVE_STATUS]
            )
        )
        .filter(Article.published_at >= since)
        .order_by(Article.id)
        .all()
    )
    out = []
    for article_id, status, error, brands, action, skip_reason in rows:
        if not brands:
            continue
        source = provenance.derive_label_source(
            status, error, action, skip_reason, article_id=article_id
        )
        if source in LLM_JUDGED_SOURCES:
            out.append((str(article_id), len(brands)))
    return out


def select_specs(session, args, specs: dict[str, Spec]) -> None:
    """Add the sampled, most-brands and human-ruling articles to `specs`."""
    if args.article_ids:
        for article_id in args.article_ids.split(","):
            merge_spec(specs, article_id.strip(), CASES)
    if args.human_rulings:
        for article_id in provenance.HUMAN_RULINGS:
            merge_spec(specs, article_id, HUMAN_RULINGS)
    if not (args.sample_size or args.most_brands):
        return
    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    candidates = _llm_judged_candidates(session, since)
    if args.most_brands:
        ranked = sorted(candidates, key=lambda c: (-c[1], c[0]))
        for article_id, _ in ranked[: args.most_brands]:
            merge_spec(specs, article_id, MOST_BRANDS)
    if args.sample_size:
        pool = sorted(c[0] for c in candidates)
        for article_id in random.Random(args.seed).sample(pool, min(args.sample_size, len(pool))):
            merge_spec(specs, article_id, SAMPLE)


def load_articles(session, specs: dict[str, Spec], baseline: str, examples: list[str]) -> list[EvalArticle]:
    """Read each article once and build what the comparison sends; skips what the pipeline would."""
    ids = list(specs)
    rows = session.query(Article).filter(Article.id.in_(ids)).all()
    by_id = {str(a.id): a for a in rows}
    under_baseline = {
        str(article_id)
        for (article_id,) in session.query(BrandLabel.article_id)
        .filter(BrandLabel.article_id.in_(ids), BrandLabel.prompt_version == baseline)
        .distinct()
    }
    articles = []
    for article_id in sorted(ids):
        spec = specs[article_id]
        a = by_id.get(article_id)
        if a is None:
            logger.warning(f"Article {article_id} not found; left out")
            continue
        content = article_content(a.full_content, a.description)
        if not has_sufficient_content(content) or not a.brands_mentioned:
            logger.warning(f"Article {article_id} has no content or brands to send; left out")
            continue
        articles.append(
            EvalArticle(
                article_id=article_id,
                title=a.title,
                content=content,
                brands=list(a.brands_mentioned),
                published_at=a.published_at,
                source_name=a.source_name,
                strata=tuple(sorted(spec.strata)),
                repeats=spec.repeats,
                expected=spec.expected,
                note=spec.note,
                stored_status=a.labeling_status,
                stored_under_baseline=(
                    article_id in under_baseline and a.labeling_status == "labeled"
                ),
                prompt_example=(
                    spec.prompt_example or matching_prompt_example(a.title, examples)
                ),
            )
        )
    return articles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default=None, help="Baseline prompt version (default: production)")
    p.add_argument("--candidate", required=True, help="Candidate prompt version")
    p.add_argument("--model", default=None, help="Model for both sides (default: LABELING_MODEL)")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--case-file", type=Path)
    p.add_argument("--article-ids", help="Comma-separated article ids")
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--most-brands", type=int, default=0)
    p.add_argument("--human-rulings", action="store_true")
    p.add_argument("--since-days", type=int, default=180, help="Sample window by published date")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cost", type=float, default=10.0, help="Spend cap in USD")
    p.add_argument("--dry-run", action="store_true", help="Select and estimate; no API call")
    p.add_argument("--output", type=Path, help="Output path stem (default: timestamped)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manager = PromptManager()
    baseline = args.baseline or manager.get_production_version()
    model = args.model or labeling_settings.labeling_model
    try:
        loaded = check_prompt_versions(baseline, args.candidate, manager)
    except EvalSetupError as e:
        logger.error(str(e))
        return 2

    prompt_chars = {
        v: len(manager.get_formatted_system_prompt(version=v, brands=list(TARGET_SPORTSWEAR_BRANDS)))
        + len(loaded[v].user_prompt)
        for v in (baseline, args.candidate)
    }
    examples = prompt_examples(loaded[args.candidate].system_prompt + loaded[args.candidate].user_prompt)

    specs = parse_case_file(json.loads(args.case_file.read_text())) if args.case_file else {}
    with db.get_session() as session:
        select_specs(session, args, specs)
        articles = load_articles(session, specs, baseline, examples)

    bound = worst_case_cost(articles, prompt_chars, args.max_tokens)
    calls = 2 * sum(a.repeats for a in articles)
    logger.info(f"{len(articles)} articles, {calls} calls; worst-case cost ${bound:.2f}")
    for stratum in sorted({s for a in articles for s in a.strata}):
        logger.info(f"  {stratum}: {sum(1 for a in articles if stratum in a.strata)}")
    for a in articles:
        if a.prompt_example:
            logger.info(f"  prompt-example case: {a.title!r} ~ {a.prompt_example!r}")
    if args.dry_run:
        return 0
    if not articles:
        logger.error("No articles to compare")
        return 2

    labelers = {}
    for version in (baseline, args.candidate):
        labelers[version] = ArticleLabeler(model=model, max_tokens=args.max_tokens, prompt_version=version)
        try:
            check_labeler(labelers[version], version)
        except EvalSetupError as e:
            logger.error(str(e))
            return 2

    def pair_bound(article: EvalArticle) -> float:
        return sum(worst_case_call_cost(article, c, args.max_tokens) for c in prompt_chars.values())

    runs, stopped = run_articles(
        articles,
        labelers[baseline],
        labelers[args.candidate],
        max_cost_usd=args.max_cost,
        pair_bound=pair_bound,
        on_pair=lambda r: logger.info(f"labeled {r.article.article_id}"),
    )
    if stopped:
        logger.warning("Stopped before the spend cap; the report covers the articles labeled so far")

    summary = summarize(runs, max_tokens=args.max_tokens)
    config = {
        "baseline": baseline,
        "candidate": args.candidate,
        "model": model,
        "max_tokens": args.max_tokens,
        "baseline_hashes": f"{loaded[baseline].system_prompt_hash}/{loaded[baseline].user_prompt_hash}",
        "candidate_hashes": (
            f"{loaded[args.candidate].system_prompt_hash}/{loaded[args.candidate].user_prompt_hash}"
        ),
        "stopped_early": stopped,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "most_brands": args.most_brands,
        "human_rulings": args.human_rulings,
        "case_file": str(args.case_file) if args.case_file else None,
        "max_cost_usd": args.max_cost,
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    stem = args.output or OUTPUT_DIR / (
        f"{baseline}_vs_{args.candidate}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Version names contain dots, so append the extension rather than use with_suffix
    json_path, md_path = Path(f"{stem}.json"), Path(f"{stem}.md")
    json_path.write_text(
        json.dumps(
            {"config": config, "summary": summary, "runs": [dataclasses.asdict(r) for r in runs]},
            indent=2,
            default=str,
        )
    )
    md_path.write_text(render_markdown(summary, config))
    logger.info(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
