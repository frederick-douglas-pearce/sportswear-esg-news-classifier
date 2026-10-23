"""Markdown rendering of a contrastive prompt evaluation (see `comparator.summarize`)."""

from __future__ import annotations


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | int | None, fmt: str = "{:,.0f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def _side(config: dict, side: str) -> str:
    """One side's prompt version, model and effort, as the report names it."""
    effort = config["effort"][side]
    return f"{config[side]} on {config[f'{side}_model']}" + (f" (effort {effort})" if effort else "")


def _version_rows(summary: dict) -> list[str]:
    b, c = summary["baseline"], summary["candidate"]
    rows = [
        ("Parse failures", b["parse_failures"], c["parse_failures"]),
        ("Refusals", b["refusals"], c["refusals"]),
        ("Unjudged rate", _pct(b["unjudged_rate"]), _pct(c["unjudged_rate"])),
        ("Every-brand compliance", _pct(b["compliance"]), _pct(c["compliance"])),
        ("Incomplete replies (a sent brand left out)", b["incomplete_replies"], c["incomplete_replies"]),
        ("Replies with an entry stating no verdict", b["replies_with_unset_entries"], c["replies_with_unset_entries"]),
        ("Replies with duplicate brands", b["replies_with_duplicates"], c["replies_with_duplicates"]),
        ("Replies naming unsent brands", b["replies_with_unsent_brands"], c["replies_with_unsent_brands"]),
        ("Output tokens, mean", _num(b["tokens"]["output_mean"]), _num(c["tokens"]["output_mean"])),
        ("Output tokens, p95", _num(b["tokens"]["output_p95"]), _num(c["tokens"]["output_p95"])),
        ("Output tokens, max", _num(b["tokens"]["output_max"]), _num(c["tokens"]["output_max"])),
        ("Truncated (stopped at max_tokens)", b["tokens"]["truncated"], c["tokens"]["truncated"]),
        ("Near limit (>= 95%)", b["tokens"]["near_limit"], c["tokens"]["near_limit"]),
        (
            "Output tokens, max (most-brands stratum)",
            _num(b["tokens_most_brands"]["output_max"]),
            _num(c["tokens_most_brands"]["output_max"]),
        ),
        (
            "Output tokens, mean (most-brands stratum)",
            _num(b["tokens_most_brands"]["output_mean"]),
            _num(c["tokens_most_brands"]["output_mean"]),
        ),
        ("Cost (USD)", f"${b['tokens']['cost_usd']:.2f}", f"${c['tokens']['cost_usd']:.2f}"),
    ]
    lines = ["| Measure | Baseline | Candidate |", "|---|---|---|"]
    lines += [f"| {name} | {bv} | {cv} |" for name, bv, cv in rows]
    return lines


def _noise_floor(floor: dict) -> str:
    if not floor["articles"]:
        return (
            "Noise floor: empty. No random-sample article is stored as labeled under the baseline "
            "prompt and model, so transition counts have no baseline self-disagreement to read "
            "against."
        )
    return (
        f"Noise floor: of {floor['articles']} random-sample articles stored as labeled under the "
        f"baseline prompt and model whose fresh baseline call answered, it still labels "
        f"{floor['still_labeled']} ({_pct(floor['rate'])})."
    )


def _newly_labeled(title: str, block: dict) -> list[str]:
    lines = [
        f"**{title}:** {block['newly_labeled']} of {block['denominator']} baseline non-labeled "
        f"replies labeled by the candidate ({_pct(block['rate'])}); "
        f"{block['dropped_labeled']} of {block['baseline_labeled']} baseline-labeled replies "
        f"not labeled ({_pct(block['dropped_rate'])}).",
        "",
        "| Baseline class | Denominator | Newly labeled |",
        "|---|---|---|",
    ]
    for cls, v in block["by_baseline_class"].items():
        lines.append(f"| {cls} | {v['denominator']} | {v['newly_labeled']} |")
    return lines


def _transitions(title: str, items: list[dict]) -> list[str]:
    lines = [f"### {title} ({len(items)})", ""]
    for item in items:
        lines.append(f"- **{item['title']}** (`{item['article_id']}`): {item['baseline']} -> {item['candidate']}")
        for brand, verdict in item["candidate_verdicts"].items():
            explanation = item["candidate_explanations"].get(brand, "")
            lines.append(f"  - {brand}: sportswear={verdict}. {explanation}")
    lines.append("")
    return lines


def render_markdown(summary: dict, config: dict) -> str:
    """The report a human reads before deciding a promotion (D036)."""
    lines = [
        f"# Contrastive evaluation: {_side(config, 'baseline')} (baseline) vs "
        f"{_side(config, 'candidate')} (candidate)",
        "",
        *[
            f"- {side.capitalize()}: {_side(config, side)}, max_tokens "
            f"{config[f'{side}_max_tokens']} (thinking included)"
            + ("" if config["effort"][side] else ", effort not sent")
            for side in ("baseline", "candidate")
        ],
        f"- Prompt hashes: baseline {config['baseline_hashes']}, candidate {config['candidate_hashes']}",
        f"- Articles: {summary['articles']}; stopped early by the spend cap: {config['stopped_early']}",
        "",
        "## By version",
        "",
        *_version_rows(summary),
        "",
        "## Outcome transitions (first repeat)",
        "",
        "| Baseline -> candidate | Articles |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary["transitions"].items()],
        "",
        "Baseline-unjudged replies under the candidate: "
        + (", ".join(f"{k} {v}" for k, v in summary["baseline_unjudged_split"].items()) or "none"),
        "",
        *_newly_labeled("Random sample", summary["newly_labeled_sample"]),
        "",
        *_newly_labeled("All articles", summary["newly_labeled_all"]),
        "",
        f"Sent brands judged by the candidate that the baseline left out: "
        f"{summary['brands_newly_judged']}. "
        f"Category flips on brands both versions labeled: {summary['category_flips']}; "
        f"brands only one version labeled: {summary['brands_labeled_by_one_side']}. "
        f"Sentiment agreement where both apply: {_pct(summary['sentiment_agreement'])}.",
        "",
        _noise_floor(summary["noise_floor"]),
        "",
        "## Product-article accuracy (cases with an expected side, prompt-example cases excluded)",
        "",
        "Per case: correct if every reply that judged the brand is on the expected side; not judged "
        "if no reply judged it (v1.10.0 leaves out a sportswear brand with no ESG content). The rate "
        "is over judged cases. Only the candidate's prompt examples are excluded, so baseline "
        "accuracy is context only.",
        "",
    ]
    for side in ("baseline", "candidate"):
        acc = summary["product_accuracy"][side]
        lines.append(
            f"- {side}: {acc['correct']} correct, {acc['wrong']} wrong, {acc['not_judged']} not "
            f"judged ({_pct(acc['rate'])} of judged)"
        )
        for miss in acc["misses"]:
            lines.append(
                f"  - wrong: {miss['title']} (`{miss['article_id']}`), {miss['brand']} expected "
                f"{miss['expected']}, got {miss['got']}. {miss['explanation']}"
            )
    lines += ["", "## Prompt-example cases (reported apart from accuracy)", ""]
    lines += [
        f"- {c['title']} (`{c['article_id']}`) looks like the source of: \"{c['example']}\""
        for c in summary["prompt_example_cases"]
    ] or ["- none"]
    lines += ["", "## Stability (cases with repeats)", ""]
    for s in summary["stability"]:
        flag = "" if s["complete"] else " (incomplete: the spend cap cut its repeats)"
        lines.append(
            f"- {s['title']} (`{s['article_id']}`): baseline {s['baseline']}, "
            f"candidate {s['candidate']}{flag}"
        )
    lines += ["", "## Transitions to read (D036)", ""]
    lines += _transitions("Non-labeled -> labeled", summary["non_labeled_to_labeled"])
    lines += _transitions("Labeled -> non-labeled", summary["labeled_to_non_labeled"])
    lines += [f"### Per-brand sportswear flips ({len(summary['fp_flips'])})", ""]
    for f in summary["fp_flips"]:
        lines.append(
            f"- **{f['title']}** (`{f['article_id']}`), {f['brand']}: {f['baseline']} -> "
            f"{f['candidate']}. Baseline: {f['baseline_explanation']} "
            f"Candidate: {f['candidate_explanation']}"
        )
    return "\n".join(lines) + "\n"
