"""Markdown rendering of a contrastive prompt evaluation (see `comparator.summarize`)."""

from __future__ import annotations

from src.evaluation.comparator import RULING_PREFIX

# D040's falsifier: a baseline that already passes this share of the ruling cases does not need
# the rulings in its prompt (#230)
RULING_FALSIFIER_RATE = 0.85


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
        ("Content types stated (brand entries)", _counts(b["content_types"]), _counts(c["content_types"])),
    ]
    lines = ["| Measure | Baseline | Candidate |", "|---|---|---|"]
    lines += [f"| {name} | {bv} | {cv} |" for name, bv, cv in rows]
    return lines


def _counts(counts: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def _category_list(title: str, by_side: dict, intro: str) -> list[str]:
    lines = [f"## {title}", "", intro, ""]
    for side in ("baseline", "candidate"):
        items = by_side[side]
        lines.append(f"- {side}: {len(items)}")
        for item in items:
            type_note = f" ({item['content_type']})" if item.get("content_type") else ""
            lines.append(
                f"  - {item['title']} (`{item['article_id']}`), {item['brand']}{type_note}: "
                f"{', '.join(item['categories'])}. {item['explanation']}"
            )
    lines.append("")
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
        f"- Case files: {', '.join(config.get('case_files') or []) or 'none'}",
        f"- Commit: {config.get('commit') or 'unknown'}"
        + (" (uncommitted changes in the tree)" if config.get("dirty") else ""),
        *[
            f"  - `{path}` sha256 {digest}"
            for path, digest in (config.get("case_file_sha256") or {}).items()
        ],
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
        "## Sportswear-verdict accuracy (cases with an expected side, prompt-example cases excluded)",
        "",
        "Per case: correct if every reply that judged the brand is on the expected side; not judged "
        "if no reply judged it (v1.10.0 leaves out a sportswear brand with no ESG content). The rate "
        "is over judged cases, from every case file. Only the candidate's prompt examples are "
        "excluded, so baseline accuracy is context only. The table splits it by stratum, so the "
        "Product-article cases (`product_*`) read apart from the others.",
        "",
    ]
    for side in ("baseline", "candidate"):
        acc = summary["product_accuracy"][side]
        lines.append(
            f"- {side}: {acc['correct']} correct, {acc['wrong']} wrong, {acc['not_judged']} not "
            f"judged ({_pct(acc['rate'])} of judged; answering `true` for every judged case "
            f"scores {acc['always_true']}, {_pct(acc['always_true_rate'])})"
        )
        for miss in acc["misses"]:
            lines.append(
                f"  - wrong: {miss['title']} (`{miss['article_id']}`) [{', '.join(miss['strata'])}], "
                f"{miss['brand']} expected {miss['expected']}, got {miss['got']}. {miss['explanation']}"
            )
        for item in acc["not_judged_cases"]:
            lines.append(
                f"  - not judged: {item['title']} (`{item['article_id']}`) "
                f"[{', '.join(item['strata'])}], {item['brand']} expected {item['expected']}"
            )
    lines += ["", "| Stratum | Baseline correct / wrong / not judged | Candidate correct / wrong / not judged |", "|---|---|---|"]
    strata = sorted(
        set(summary["product_accuracy"]["baseline"]["by_stratum"])
        | set(summary["product_accuracy"]["candidate"]["by_stratum"])
    )
    for stratum in strata:
        cells = []
        for side in ("baseline", "candidate"):
            n = summary["product_accuracy"][side]["by_stratum"].get(
                stratum, {"correct": 0, "wrong": 0, "not_judged": 0}
            )
            cells.append(f"{n['correct']} / {n['wrong']} / {n['not_judged']}")
        lines.append(f"| {stratum} | {cells[0]} | {cells[1]} |")
    lines += [""]
    lines += _category_list(
        "Categories on Product, financial and no-category cases",
        summary["case_categories"],
        "Applicable categories kept on the expected brand of a Product, financial or no-category "
        "case (on any brand, for a case that names none), per reply. There should be none.",
    )
    lines += [
        "## Expected categories (cases that rule them)",
        "",
        "Per case: matched if every answered reply gives the expected brand exactly the ruled "
        "categories; unanswered if every reply failed. A reply that leaves the brand out or calls "
        "it not sportswear gives it none.",
        "",
    ]
    for side in ("baseline", "candidate"):
        block = summary["expected_categories"][side]
        lines.append(
            f"- {side}: {block['matched']} of {block['cases']} cases matched, "
            f"{block['unanswered']} unanswered"
        )
        for miss in block["misses"]:
            type_note = f" ({miss['content_type']})" if miss.get("content_type") else ""
            lines.append(
                f"  - {miss['title']} (`{miss['article_id']}`), {miss['brand']}{type_note}: expected "
                f"{', '.join(miss['expected']) or 'none'}; missing {', '.join(miss['missing']) or 'none'}, "
                f"extra {', '.join(miss['extra']) or 'none'}. {miss['explanation']}"
            )
    lines += [
        "",
        "## Expected sentiments (cases that rule them)",
        "",
        "Per case: matched if every answered reply that kept a ruled category gives it the ruled "
        "sentiment; not assessed if no answered reply kept one (that case is a miss above); "
        "unanswered if every reply failed.",
        "",
    ]
    for side in ("baseline", "candidate"):
        block = summary["expected_sentiments"][side]
        lines.append(
            f"- {side}: {block['matched']} matched, {block['mismatched']} mismatched, "
            f"{block['not_assessed']} not assessed, {block['unanswered']} unanswered, of "
            f"{block['cases']} cases"
        )
        for miss in block["misses"]:
            got = "; ".join(
                f"{m['category']} expected {m['expected']}, got {m['got']}" for m in miss["mismatches"]
            )
            lines.append(
                f"  - {miss['title']} (`{miss['article_id']}`), {miss['brand']}: {got}. "
                f"{miss['explanation']}"
            )
        for item in block["not_assessed_cases"]:
            lines.append(f"  - not assessed: {item['title']} (`{item['article_id']}`), {item['brand']}")
    lines += [
        "",
        "## Owner rulings (D040)",
        "",
        "Per `ruling_1` to `ruling_8` case (gray and prompt-example cases excluded): passed if every "
        "answered reply judges the expected brand on the expected side with exactly the ruled "
        "categories and sentiments; unanswered if every reply failed, outside the rate. D040's "
        f"falsifier: a baseline rate of {_pct(RULING_FALSIFIER_RATE)} or more means the rulings "
        "are not needed in the prompt, and promotion is reconsidered.",
        "",
    ]
    for side in ("baseline", "candidate"):
        block = summary["rulings"][side]
        lines.append(
            f"- {side}: {block['passed']} passed, {block['failed']} failed, {block['unanswered']} "
            f"unanswered, of {block['cases']} cases ({_pct(block['rate'])} of answered)"
        )
        for item in block["failed_cases"]:
            lines.append(
                f"  - failed: {item['title']} (`{item['article_id']}`) [{', '.join(item['rulings'])}], "
                f"{item['brand']}"
            )
        for item in block["unanswered_cases"]:
            lines.append(f"  - unanswered: {item['title']} (`{item['article_id']}`), {item['brand']}")
    rate = summary["rulings"]["baseline"]["rate"]
    lines.append(
        f"- Falsifier: baseline {_pct(rate)} "
        + ("(no answered case)" if rate is None else
           ("meets the threshold" if rate >= RULING_FALSIFIER_RATE else "is below the threshold"))
    )
    lines += ["", "| Ruling | Baseline passed / failed / unanswered | Candidate passed / failed / unanswered |", "|---|---|---|"]
    rulings = sorted(
        set(summary["rulings"]["baseline"]["by_ruling"]) | set(summary["rulings"]["candidate"]["by_ruling"]),
        key=lambda name: int(name.removeprefix(RULING_PREFIX)),
    )
    for ruling in rulings:
        cells = []
        for side in ("baseline", "candidate"):
            n = summary["rulings"][side]["by_ruling"].get(ruling, {"passed": 0, "failed": 0, "unanswered": 0})
            cells.append(f"{n['passed']} / {n['failed']} / {n['unanswered']}")
        lines.append(f"| {ruling} | {cells[0]} | {cells[1]} |")
    lines.append("")
    lines += _category_list(
        "Categories the content type dropped (D037 gate)",
        summary["gated_categories"],
        "Applicable categories a reply gave a brand whose content type keeps none. Each is read: "
        "a mistyped `business` article loses real labels here.",
    )
    lines += ["## Prompt-example cases (reported apart from accuracy)", ""]
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
