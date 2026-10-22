#!/usr/bin/env python3
"""Contrastive evaluation of two labeling prompt versions, or two models (#4, #192 PR2, #212, D036).

Labels the same articles fresh under a baseline and a candidate side, and reports what would
change if the candidate were promoted. A side is a prompt version and a model; the two sides
differ in exactly one of them, so the report attributes every change to that one. Nothing is
written to the database. A promotion is judged against D036's criteria by a human reading the
report.

Articles come from any combination of:
  --case-file      named cases (ids, repeats, an expected verdict for one brand)
  --article-ids    ad-hoc ids
  --sample-size    a seeded random sample of recent articles the labeling LLM replied on
                   (label source llm or unjudged, pre-LLM skips left out), within --since-days
  --most-brands    the articles the labeling LLM replied on with the most brands mentioned,
                   from the whole database
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

    # Two models under one prompt version (#212):
    uv run python scripts/compare_prompts.py --baseline v1.11.0 \\
        --baseline-model claude-haiku-4-5-20251001 --candidate-model claude-haiku-5-5 \\
        --case-file prompts/labeling/eval_cases/product_and_stability.json \\
        --sample-size 100 --most-brands 10 --human-rulings --dry-run
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
from src.data_collection.models import (  # noqa: E402
    SKIP_INSUFFICIENT_CONTENT,
    SKIP_NO_BRANDS,
    Article,
    BrandLabel,
)
from src.evaluation.comparator import (  # noqa: E402
    MOST_BRANDS,
    SAMPLE,
    EvalArticle,
    Expectation,
    SideSettings,
    matching_prompt_example,
    prompt_examples,
    run_articles,
    summarize,
    worst_case_call_cost,
    worst_case_cost,
)
from src.evaluation.diff_report import render_markdown  # noqa: E402
from src.labeling.article_input import article_content, has_sufficient_content  # noqa: E402
from src.labeling.config import (  # noqa: E402
    TARGET_SPORTSWEAR_BRANDS,
    labeling_model_spec,
    labeling_settings,
)
from src.labeling.models import normalize_brand  # noqa: E402
from src.labeling.labeler import ArticleLabeler  # noqa: E402
from src.labeling.prompt_manager import PromptManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_prompts")

OUTPUT_DIR = Path("data/evaluation/comparisons")
CASES = "cases"
HUMAN_RULINGS = "human_rulings"
# Label sources whose status came from a reply of the labeling LLM: `llm`, and `unjudged` (a reply
# that judged no brand). The unjudged skips the pipeline writes before any LLM call are left out.
LLM_REPLY_SOURCES = frozenset({provenance.LLM, provenance.UNJUDGED})
PRE_LLM_SKIP_REASONS = frozenset({SKIP_INSUFFICIENT_CONTENT, SKIP_NO_BRANDS})


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


def check_contrast(baseline: str, candidate: str, baseline_model: str, candidate_model: str) -> None:
    """Refuse two sides that do not differ in exactly one of prompt version and model (D036 §4).

    Equal sides compare a setup with itself. Sides that differ in both leave no way to tell which
    change a difference in the report came from. A model with no spec is refused here, before
    any call, rather than when its labeler is built.
    """
    same_prompt, same_model = baseline == candidate, baseline_model == candidate_model
    if same_prompt and same_model:
        raise EvalSetupError(
            f"Baseline and candidate are the same version ({baseline}) and model ({baseline_model})"
        )
    if not same_prompt and not same_model:
        raise EvalSetupError(
            f"Baseline and candidate differ in both prompt ({baseline} vs {candidate}) and model "
            f"({baseline_model} vs {candidate_model}); vary one at a time"
        )
    for model in (baseline_model, candidate_model):
        try:
            labeling_model_spec(model)
        except ValueError as e:
            raise EvalSetupError(str(e)) from e


def check_prompt_versions(baseline: str, candidate: str, manager: PromptManager) -> dict[str, object]:
    """Load both versions, refusing unloadable ones (D036 §4)."""
    loaded = {}
    for version in (baseline, candidate):
        try:
            loaded[version] = manager.load_version(version)
        except (FileNotFoundError, ValueError) as e:
            raise EvalSetupError(f"Prompt version {version} does not load: {e}") from e
        if loaded[version].version != version:
            raise EvalSetupError(f"Asked for {version}, loaded {loaded[version].version}")
    return loaded


def check_labeler(labeler: ArticleLabeler, version: str, model: str) -> None:
    """Refuse a labeler that fell back to the hardcoded prompt, loaded another version, or runs
    another model.

    `ArticleLabeler` logs a warning and carries on with the hardcoded prompt when a version
    fails to load, so a mistyped version would otherwise compare production with itself.
    """
    if labeler.loaded_prompt is None or labeler.prompt_version != version:
        raise EvalSetupError(
            f"Labeler for {version} loaded {labeler.prompt_version!r}; refusing to run"
        )
    if labeler.model != model:
        raise EvalSetupError(f"Labeler for {model} runs {labeler.model!r}; refusing to run")


@dataclasses.dataclass(frozen=True)
class Candidate:
    """An article the sample and most-brands selection can draw from."""

    article_id: str
    brand_count: int
    # Published date, else the date it was collected
    dated: datetime | None


def replied_on_by_llm(source: str, labeling_status: str, labeling_error: str | None) -> bool:
    """Whether an article's stored status came from a reply of the labeling LLM."""
    if source not in LLM_REPLY_SOURCES:
        return False
    return not (labeling_status == "skipped" and labeling_error in PRE_LLM_SKIP_REASONS)


def pick_most_brands(candidates: list[Candidate], n: int) -> list[str]:
    """The n candidates with the most brands mentioned; ties broken by id."""
    ranked = sorted(candidates, key=lambda c: (-c.brand_count, c.article_id))
    return [c.article_id for c in ranked[:n]]


def pick_sample(candidates: list[Candidate], n: int, seed: int, since: datetime) -> list[str]:
    """A seeded random sample of the candidates dated on or after `since`."""
    pool = sorted(c.article_id for c in candidates if c.dated is not None and c.dated >= since)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def candidates_from_rows(rows) -> list[Candidate]:
    """Candidates from (id, status, error, brands, published, created, fp action, fp skip reason)
    rows. A status a human ruled on (provenance's human-verified ids) derives as `human`, as in the
    FP export, so it is left out of the pool; `--human-rulings` selects those articles."""
    verified = provenance.human_verified_ids((str(r[0]), r[1]) for r in rows)
    out = []
    for article_id, status, error, brands, published, created, action, skip_reason in rows:
        if not brands:
            continue
        source = provenance.derive_label_source(
            status,
            error,
            action,
            skip_reason,
            human_verified=str(article_id) in verified,
            article_id=article_id,
        )
        if replied_on_by_llm(source, status, error):
            dated = published or created
            if dated is not None and dated.tzinfo is None:
                dated = dated.replace(tzinfo=timezone.utc)
            out.append(Candidate(str(article_id), len(brands), dated))
    return out


def _replied_on_candidates(session) -> list[Candidate]:
    """Every article whose stored status came from a reply of the labeling LLM, with brands."""
    latest = provenance.latest_prediction_subquery("fp")
    rows = (
        session.query(
            Article.id,
            Article.labeling_status,
            Article.labeling_error,
            Article.brands_mentioned,
            Article.published_at,
            Article.created_at,
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
        .order_by(Article.id)
        .all()
    )
    return candidates_from_rows(rows)


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
    candidates = _replied_on_candidates(session)
    for article_id in pick_most_brands(candidates, args.most_brands):
        merge_spec(specs, article_id, MOST_BRANDS)
    since = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    for article_id in pick_sample(candidates, args.sample_size, args.seed, since):
        merge_spec(specs, article_id, SAMPLE)


def build_eval_article(
    article, spec: Spec, stored_under_baseline: bool, examples: list[str]
) -> EvalArticle | None:
    """What the comparison sends for one article, or None when the pipeline would not send it.

    Raises EvalSetupError when a case expects a verdict for a brand the article does not list:
    that case could never be scored.
    """
    article_id = str(article.id)
    content = article_content(article.full_content, article.description)
    if not has_sufficient_content(content) or not article.brands_mentioned:
        return None
    brands = list(article.brands_mentioned)
    if spec.expected and normalize_brand(spec.expected.brand) not in {
        normalize_brand(b) for b in brands
    }:
        raise EvalSetupError(
            f"Case {article_id} expects a verdict for {spec.expected.brand!r}, "
            f"which is not among its brands {brands}"
        )
    return EvalArticle(
        article_id=article_id,
        title=article.title,
        content=content,
        brands=brands,
        published_at=article.published_at,
        source_name=article.source_name,
        strata=tuple(sorted(spec.strata)),
        repeats=spec.repeats,
        expected=spec.expected,
        note=spec.note,
        stored_status=article.labeling_status,
        stored_under_baseline=stored_under_baseline and article.labeling_status == "labeled",
        prompt_example=spec.prompt_example or matching_prompt_example(article.title, examples),
    )


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
        a = by_id.get(article_id)
        if a is None:
            logger.warning(f"Article {article_id} not found; left out")
            continue
        built = build_eval_article(a, specs[article_id], article_id in under_baseline, examples)
        if built is None:
            logger.warning(f"Article {article_id} has no content or brands to send; left out")
            continue
        articles.append(built)
    return articles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--baseline",
        default=None,
        help="Baseline prompt version (default: the runtime prompt, LABELING_PROMPT_VERSION else production)",
    )
    p.add_argument(
        "--candidate", default=None, help="Candidate prompt version (default: the baseline's)"
    )
    p.add_argument("--model", default=None, help="Model for both sides (default: LABELING_MODEL)")
    p.add_argument("--baseline-model", default=None, help="Baseline model (default: --model)")
    p.add_argument("--candidate-model", default=None, help="Candidate model (default: --model)")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Output cap for both sides, thinking included (default: each model's own)",
    )
    p.add_argument(
        "--effort",
        default=None,
        help="output_config.effort for a side whose model takes it (default: LABELING_EFFORT)",
    )
    p.add_argument("--case-file", type=Path)
    p.add_argument("--article-ids", help="Comma-separated article ids")
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--most-brands", type=int, default=0)
    p.add_argument("--human-rulings", action="store_true")
    p.add_argument(
        "--since-days",
        type=int,
        default=180,
        help="Window for --sample-size, by published date (else collected); --most-brands is not windowed",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cost", type=float, default=10.0, help="Spend cap in USD")
    p.add_argument("--dry-run", action="store_true", help="Select and estimate; no API call")
    p.add_argument("--output", type=Path, help="Output path stem (default: timestamped)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manager = PromptManager()
    # The prompt the pipeline runs: LABELING_PROMPT_VERSION, else the registry's production
    baseline = args.baseline or labeling_settings.prompt_version or manager.get_production_version()
    if baseline != manager.get_production_version():
        logger.warning(
            f"Baseline {baseline} is not the registry's production version "
            f"({manager.get_production_version()})"
        )
    candidate = args.candidate or baseline
    model = args.model or labeling_settings.labeling_model
    models = {"baseline": args.baseline_model or model, "candidate": args.candidate_model or model}
    versions = {"baseline": baseline, "candidate": candidate}
    effort = args.effort or labeling_settings.labeling_effort
    try:
        check_contrast(baseline, candidate, models["baseline"], models["candidate"])
        loaded = check_prompt_versions(baseline, candidate, manager)
    except EvalSetupError as e:
        logger.error(str(e))
        return 2

    sides = {
        side: SideSettings(
            model=models[side],
            max_tokens=args.max_tokens or labeling_model_spec(models[side]).default_max_tokens,
        )
        for side in versions
    }
    prompt_chars = {
        side: len(manager.get_formatted_system_prompt(version=v, brands=list(TARGET_SPORTSWEAR_BRANDS)))
        + len(loaded[v].user_prompt)
        for side, v in versions.items()
    }
    examples = prompt_examples(loaded[candidate].system_prompt + loaded[candidate].user_prompt)

    specs = parse_case_file(json.loads(args.case_file.read_text())) if args.case_file else {}
    try:
        with db.get_session() as session:
            select_specs(session, args, specs)
            articles = load_articles(session, specs, baseline, examples)
    except EvalSetupError as e:
        logger.error(str(e))
        return 2

    calls_per_article = [(prompt_chars[side], sides[side]) for side in versions]
    bound = worst_case_cost(articles, calls_per_article)
    calls = 2 * sum(a.repeats for a in articles)
    logger.info(f"{len(articles)} articles, {calls} calls; worst-case cost ${bound:.2f}")
    for side, version in versions.items():
        logger.info(
            f"  {side}: {version} on {sides[side].model}, max_tokens {sides[side].max_tokens}"
        )
    for stratum in sorted({s for a in articles for s in a.strata}):
        logger.info(f"  {stratum}: {sum(1 for a in articles if stratum in a.strata)}")
    for a in articles:
        if a.prompt_example:
            logger.info(f"  prompt-example case: {a.title!r} ~ {a.prompt_example!r}")
    over_cap = (
        f"Worst-case cost ${bound:.2f} exceeds --max-cost ${args.max_cost:.2f}: raise the cap "
        "or narrow the selection"
    )
    if args.dry_run:
        if bound > args.max_cost:
            logger.warning(f"{over_cap} (a run would be refused)")
        return 0
    if bound > args.max_cost:
        logger.error(over_cap)
        return 2
    if not articles:
        logger.error("No articles to compare")
        return 2

    labelers = {}
    for side, version in versions.items():
        labelers[side] = ArticleLabeler(
            model=sides[side].model,
            max_tokens=sides[side].max_tokens,
            prompt_version=version,
            effort=effort,
        )
        try:
            check_labeler(labelers[side], version, sides[side].model)
        except EvalSetupError as e:
            logger.error(str(e))
            return 2

    def pair_bound(article: EvalArticle) -> float:
        return sum(worst_case_call_cost(article, chars, side) for chars, side in calls_per_article)

    runs, stopped = run_articles(
        articles,
        labelers["baseline"],
        labelers["candidate"],
        max_cost_usd=args.max_cost,
        pair_bound=pair_bound,
        on_pair=lambda r: logger.info(f"labeled {r.article.article_id}"),
    )
    if stopped:
        logger.warning("Stopped before the spend cap; the report covers the articles labeled so far")

    summary = summarize(runs, baseline=sides["baseline"], candidate=sides["candidate"])
    config = {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_model": sides["baseline"].model,
        "candidate_model": sides["candidate"].model,
        "baseline_max_tokens": sides["baseline"].max_tokens,
        "candidate_max_tokens": sides["candidate"].max_tokens,
        # Sent only to a side whose model takes it
        "effort": {
            side: effort if labeling_model_spec(sides[side].model).accepts_effort else None
            for side in versions
        },
        "baseline_hashes": f"{loaded[baseline].system_prompt_hash}/{loaded[baseline].user_prompt_hash}",
        "candidate_hashes": (
            f"{loaded[candidate].system_prompt_hash}/{loaded[candidate].user_prompt_hash}"
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
    contrast = (
        f"{baseline}_vs_{candidate}"
        if baseline != candidate
        else f"{baseline}_{sides['baseline'].model}_vs_{sides['candidate'].model}"
    )
    stem = args.output or OUTPUT_DIR / f"{contrast}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
