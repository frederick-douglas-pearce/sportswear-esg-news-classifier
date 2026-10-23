# AI-Based Article Labeling Pipeline

This document provides detailed information about the LLM-based labeling pipeline used to generate training data for the ML classifiers.

> **Quick Start:** For a high-level overview, see the [main README](../README.md#ai-based-article-labeling).

## Overview

The project uses Claude to label articles with ESG categories, sentiment, and supporting evidence. This labeled data is then used to train ML classifiers that can perform routine classification at scale.

## LLM Labeling Workflow

The LLM labeling pipeline processes articles through these steps:

1. **Chunking**: Articles are split into paragraph-based chunks (~500 tokens each) with character position tracking
2. **Embedding**: Chunks are embedded using OpenAI's `text-embedding-3-small` model for semantic search
3. **LLM Labeling**: Claude analyzes each article and returns structured JSON with:
   - Per-brand ESG category labels (Environmental, Social, Governance, Digital Transformation)
   - Ternary sentiment for each category (+1 positive, 0 neutral, -1 negative)
   - Supporting evidence quotes from the article
   - Confidence score and reasoning
4. **Evidence Matching**: Evidence excerpts are linked back to article chunks via exact match, fuzzy match, or embedding similarity

The labeled data is then exported to train ML classifiers that can handle routine classification at scale.

## Data Labeling Scope

Articles are only labeled if they are **primarily about** the sportswear brand's activities. The labeling pipeline filters out two types of false positives:

**1. Brand Name Conflicts**: When a brand name refers to something other than the sportswear company:
- "Puma" (the animal), "Patagonia" (the region), "Columbia" (the country/university)
- "Vans" (vehicles), "Anta" (Indian political district), "Decathlon" (investment firms)

**2. Tangential Brand Mentions**: When the brand name correctly refers to the sportswear company, but the article is not actually about that brand:
- Former executives now working at other companies (e.g., "Ex-Nike VP joins Tech Startup")
- Biographical context in profiles about people who no longer work at the brand
- Articles about other companies that briefly mention a sportswear brand for comparison

**Key Test**: Is this article primarily about the sportswear brand's current activities, products, or ESG initiatives? If the brand is only mentioned as background context, historical reference, or biographical detail, it should be marked as a false positive.

## Running the Labeling Pipeline

```bash
# Check labeling statistics
uv run python scripts/label_articles.py --stats

# Test with dry run (doesn't save to database)
uv run python scripts/label_articles.py --dry-run --batch-size 5

# Label a batch of articles
uv run python scripts/label_articles.py --batch-size 10

# Label a specific article by UUID
uv run python scripts/label_articles.py --article-id 12345678-1234-1234-1234-123456789abc

# Skip embedding generation (faster but no semantic evidence matching)
uv run python scripts/label_articles.py --batch-size 10 --skip-embedding

# Verbose mode for debugging
uv run python scripts/label_articles.py --batch-size 5 -v
```

## Labeling Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size N` | Number of articles to process | 10 |
| `--dry-run` | Show what would be done without saving | False |
| `--article-id UUID` | Label a specific article | - |
| `--skip-chunking` | Skip chunking for articles that already have chunks | False |
| `--skip-embedding` | Skip embedding generation | False |
| `--stats` | Show labeling statistics and exit | False |
| `-v, --verbose` | Enable verbose/debug logging | False |

## Cost Estimation

LLM prices and each model's request surface (whether it takes `temperature` or `output_config.effort`, and its default `max_tokens`) are in `LABELING_MODELS` in `src/labeling/config.py`, matched on the exact id or the id plus a `-YYYYMMDD` snapshot date. A labeler for a model with no entry is refused before any call. Haiku 5.5 thinks by default: the labeler reads the reply's text blocks by type, sends no sampling parameters and sends `LABELING_EFFORT` (default `medium`), and records a refusal (`stop_reason: "refusal"`; Haiku 5.5 has no server-side fallback) as a failed article with error type `refusal`. Neither a refusal nor a reply that stopped at `max_tokens` and did not parse is retried; find them with `labeling_error LIKE 'Model declined the request%'` or `LIKE 'Reply stopped at max_tokens%'` (`FAILED_REFUSAL_PREFIX`, `FAILED_MAX_TOKENS_PREFIX` in `src/data_collection/models.py`). Each run's config JSON records the model and effort it ran (`labeling_runs` has no model column).

| Component | Approximate Cost |
|-----------|------------------|
| OpenAI embeddings (text-embedding-3-small) | ~$0.02 per 1000 articles |
| Claude labeling | ~$3-5 per 1000 articles (Haiku 4.5) |
| **Total** | **~$3-5 per 1000 articles** |

## Promoting a Prompt Version (Contrastive Evaluation)

A prompt version becomes production (`registry.json`'s `production`) only after a contrastive evaluation, judged against criteria fixed before the run (D036). `scripts/compare_prompts.py` labels the same articles fresh under the baseline (by default the prompt the pipeline runs) and the candidate, and writes nothing to the database. A prompt contrast holds the model fixed; a model contrast (below) holds the prompt fixed. Each reply's outcome comes from `dispose_reply`, as in the pipeline.

The same harness compares two models, or one model at two efforts, under one prompt version (#212): pass `--baseline-model` / `--candidate-model` (or `--baseline-effort` / `--candidate-effort`) and leave out `--candidate`, which defaults to the baseline's version. A side is a prompt version and a call (model and effort), and the two sides must differ in exactly one of them.

```bash
# Haiku 4.5 vs Haiku 5.5 under one prompt version; drop --dry-run to run
uv run python scripts/compare_prompts.py --baseline v1.11.0 \
  --baseline-model claude-haiku-4-5-20251001 --candidate-model claude-haiku-5-5 --effort medium \
  --case-file prompts/labeling/eval_cases/product_and_stability.json \
  --sample-size 100 --most-brands 10 --human-rulings --dry-run
```

Each side's `max_tokens` defaults to its model's (`LABELING_MODELS` in `src/labeling/config.py`: 2000 for Haiku 4.5, 8000 for Haiku 5.5, whose thinking counts toward the cap); `--max-tokens` sets both. `--effort` (default `LABELING_EFFORT`) sets both sides' effort, and is sent only to a model that takes it, so asking Haiku 4.5 for two efforts is refused as equal sides. Models are compared by table key, so an alias and its dated snapshot count as one model. The noise floor counts articles stored under the baseline's prompt version *and* model; under a prompt version that was promoted recently, few or none qualify, and the report says when the floor is empty.

```bash
# Selection and worst-case cost, no API call
uv run python scripts/compare_prompts.py --candidate v1.11.0 \
  --case-file prompts/labeling/eval_cases/product_and_stability.json \
  --sample-size 100 --most-brands 10 --human-rulings --dry-run

# The run; stops before a pair of calls could take spend past --max-cost (USD)
uv run python scripts/compare_prompts.py --candidate v1.11.0 \
  --case-file prompts/labeling/eval_cases/product_and_stability.json \
  --sample-size 100 --most-brands 10 --human-rulings --max-cost 10
```

- **Articles.** Named cases (`prompts/labeling/eval_cases/`, reusable for any candidate: #182's stability set with repeats, Product-article cases with the verdict v1.11.0's rule gives one brand, gray-zone cases reported but not scored); a seeded random sample, within `--since-days`, of articles whose stored status came from a reply of the labeling LLM (label source `llm` or `unjudged`, leaving out the skips written before any LLM call); the articles of that kind with the most brands, from the whole database; and the articles a human ruled on in #182. The content sent is the pipeline's (`src/labeling/article_input.py`). The baseline defaults to the prompt the pipeline runs (`LABELING_PROMPT_VERSION`, else the registry's production).
- **Guards.** Before any call, the script refuses two sides that are equal or that differ in both prompt and call, a model with no entry in `LABELING_MODELS`, an unknown effort level, a version that does not load, a labeler that runs another model or effort, a labeler that fell back to the hardcoded prompt (it reports `legacy`), a case expecting a verdict for a brand its article does not list, and a run whose worst-case cost exceeds `--max-cost` (a dry run only warns).
- **Report** (`data/evaluation/comparisons/`, not committed): outcome transitions over four classes (`labeled`, judged skip, unjudged skip, `false_positive`) plus `failed` for a call that errored or did not parse; the newly-labeled rate over the baseline's non-labeled replies, by class; per-brand sportswear flips; category flips on brands both versions labeled, and brands only one labeled; sentiment agreement; every-brand compliance (no sent brand left out, as `labeling_runs.incomplete_replies` counts) and duplicate entries per version; output tokens, including on the most-brands articles, with a reply counted as truncated when the API stopped it at `max_tokens` (output tokens include thinking, so the count is not the test); refusals, apart from parse failures; stability across repeats, with a case the spend cap cut short marked incomplete; Product-article accuracy per case (correct, wrong, or not judged when no reply judged the brand), leaving out cases whose article is the source of a quoted example in the candidate prompt; the score answering `true` for every judged Product case would get, the bar a rule has to beat; for a content-type candidate (v1.12.0, D037), the content types stated, the categories kept on the expected brand of a Product, financial or no-category case, or on any brand of one that names none (there should be none), and every category the content type dropped, each to be read; for a case that rules its brand's categories (`expected.categories` in the case file), whether every reply gave exactly those, with what was missing or extra (a type that keeps no category loses a ruled label silently, since the gate only lists categories a reply gave); a noise floor (random-sample articles stored as labeled under the baseline prompt and model whose fresh baseline call answered, and how many of those it still labels); cost.
- **Criteria (D036).** No parse failures; no refusals, or each one read and explained (a refused reply is `failed`, which leaves it out of compliance and the newly-labeled rates, so refusals would otherwise flatter the candidate); no reply at `max_tokens`; candidate every-brand compliance at least 95%; each stability case consistent across repeats or explained; each Product-article case on its expected side or the miss explained; a human reads every labeled to non-labeled and non-labeled to labeled transition and every per-brand sportswear flip. The prompt hashes promoted are the hashes evaluated, and so is the reply schema's hash for a version that has one (the report prints it as `system/user/schema <hash>`); a text or schema change after the run needs a new run.

## Exporting Training Data

Export labeled data for ML classifier training:

```bash
# Export false positive classifier data (sportswear vs non-sportswear brands)
uv run python scripts/export_training_data.py --dataset fp

# Export ESG pre-filter data (has ESG content vs no ESG)
uv run python scripts/export_training_data.py --dataset esg-prefilter

# Export full ESG multi-label classifier data
uv run python scripts/export_training_data.py --dataset esg-labels

# Export only new data since a date (for incremental updates)
uv run python scripts/export_training_data.py --dataset fp --since 2025-01-01

# Export to specific file
uv run python scripts/export_training_data.py --dataset fp -o data/fp_data.jsonl

# FP only: put back a cohort the export excludes by default (see below)
uv run python scripts/export_training_data.py --dataset fp --include-classifier-labels
uv run python scripts/export_training_data.py --dataset fp --include-unjudged-skips
uv run python scripts/export_training_data.py --dataset fp --include-rule-labels

# FP only: owner-attested tier (see below)
uv run python scripts/export_training_data.py --dataset fp --attested-before YYYY-MM-DD
```

**Export Formats (JSONL):**

| Dataset | Fields | Use Case |
|---------|--------|----------|
| `fp` | article_id, title, content, brands, source_name, category, is_sportswear, source, label_source (+ fp_reason on negatives, attested_cutoff on attested rows) | False positive brand classifier |
| `esg-prefilter` | article_id, title, content, brands, has_esg | ESG content pre-filter |
| `esg-labels` | article_id, title, content, brand, E/S/G/D flags + sentiment | Multi-label ESG classifier |

### FP label provenance

Each `fp` record carries `label_source`: who decided its label. It is derived at export time
from the article's status, its stored reason, its latest FP prediction and the human rulings
(`src/data_collection/provenance.py`, #154, D030); nothing is stored. Where the writer stores
evidence (a `skipped` row's reason, an FP skip's reason) the source rests on it; where it does not
(`false_positive` and `labeled` rows), the source is inferred from the FP prediction, so a status
moved by hand reads as `llm` until #143's audit log records such moves.

| `label_source` | Meaning | Default |
|---|---|---|
| `llm` | The labeling LLM decided it, with an FP prediction recorded (pre-filter running, failed, or disabled with a novelty score). Inferred for `false_positive`/`labeled` rows (see above) | kept |
| `pre_classifier` | As `llm`, but no FP prediction is recorded: labeled before the FP classifier existed, or with the pre-filter off and no novelty score | kept |
| `human` | A human ruled on it (today: the `HUMAN_RULINGS` list from #182, until #143's audit log; `cleanup_false_positives.py`'s manual marks) | kept |
| `unattributed` | A `skipped` row whose reason no pipeline path writes (no reason, or one left by a status move) | kept, counted |
| `fp_classifier` | A `false_positive` whose latest FP prediction is a skip and whose stored reason is that skip's own reason, so nothing has relabeled it since | excluded (`--include-classifier-labels`) |
| `unjudged` | A `skipped` row whose reply, or lack of one, does not establish whether the article has a sportswear brand ("No ESG content found", "Insufficient content", "No brands mentioned", "No sportswear brand ESG content", "Some brands not judged; the rest not sportswear") | excluded (`--include-unjudged-skips`) |
| `rule` | A `false_positive` from `cleanup_false_positives.py`'s keyword sweep (`Auto-detected:`) | excluded (`--include-rule-labels`) |

The fp notebooks and `scripts/ablation_study.py` read `data/fp_training_data.jsonl`,
`scripts/train.py` reads it unless `--data-path` names another file, and `scripts/retrain.py` reads
whatever `--data` names. The exclusions reach training and evaluation only when those read this
export's output: `-o data/fp_training_data.jsonl` (and `--data` set to it for `retrain.py`), or the
`model_training` workflow, which copies its export to that path. A file exported before #154 has no
`label_source` and still holds the excluded rows. The export's rows are ordered by article id, so
the same data gives the same splits. The export prints what it
excluded (classifier-only negatives by the month of the FP skip, unjudged skips by reason) and
both classes broken down by `label_source`. Unjudged skips are "no ESG" rows, so excluding them
shrinks the "sportswear, no ESG" part of the positive class: read the composition, not only the
class ratio, before retraining (#192 re-judges those rows).

**Owner-attested tier.** `--attested-before YYYY-MM-DD` also keeps classifier-only negatives whose
FP skip (UTC) predates the date, on the owner's attestation that articles before it were reviewed.
It is weaker than a per-article human ruling: those rows keep `label_source: fp_classifier` and gain
an `attested_cutoff` field. There is no default date; the owner has not confirmed one, and #155's
review tests the attestation. It cannot be combined with `--include-classifier-labels`.

## ML Classifier Opportunities

The project is designed to train three progressively complex classifiers that can reduce Claude API costs while maintaining accuracy:

**1. False Positive Brand Classifier** ✅ (Complete)
- **Purpose**: Filter out articles where brand names match non-sportswear entities (e.g., "Puma" the animal, "Patagonia" the region, "Black Diamond" the power company)
- **Input**: Article title + content + detected brand name
- **Output**: Binary classification (is_sportswear: 0 or 1)
- **Training Data**: 993 records from `--dataset fp` export (856 sportswear, 137 false positives)
- **Impact**: Prevents ~15% of articles from requiring expensive LLM labeling
- **Best Model**: Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 0.988)

**2. ESG Pre-filter Classifier** ✅ (Complete)
- **Purpose**: Quickly identify whether an article contains any ESG-relevant content before detailed classification
- **Input**: Article title + content + metadata
- **Output**: Binary classification (has_esg: 0 or 1)
- **Training Data**: 870 records from `--dataset esg-prefilter` export (635 has ESG, 235 no ESG)
- **Impact**: Skip detailed ESG labeling for articles with no ESG content
- **Best Model**: Logistic Regression with TF-IDF + LSA features (Test F2: 0.931, Recall: 100%)

**3. ESG Multi-label Classifier** (Planned)
- **Purpose**: Classify articles into specific ESG categories with sentiment, replacing Claude for routine classification
- **Input**: Article title + content + brand name
- **Output**: Multi-label (Environmental, Social, Governance, Digital Transformation) with ternary sentiment (-1, 0, +1)
- **Training Data**: 554 records from `--dataset esg-labels` export
- **Impact**: Replace Claude API calls entirely for high-confidence predictions

## Stock Article Classification

Guidelines for distinguishing between `false_positive` (pure metrics) and `skipped` (substantive content) for stock/finance articles. This matters because the FP classifier bypasses LLM - articles incorrectly marked `false_positive` are permanently excluded from future labeling.

### Substantive Content (→ `skipped`, send to LLM)

Articles with human-written analysis or commentary about the brand's business:

- Analyst reports **with reasoning** (e.g., "Needham downgrades Nike citing brand weakness in North America")
- Earnings coverage **with context** (e.g., "Deckers reports record revenue as global demand for Hoka continues")
- CEO/CFO quotes or business strategy discussion
- Investment thesis articles with company-specific analysis
- Competitive analysis or market position commentary

### Template / Raw Metrics (→ `false_positive`, skip LLM)

Auto-generated or template content with no human analysis:

- **Institutional holdings** (13F filings): "Midwest Trust Co Invests $1.64M in NIKE" — just SEC filing data
- **Stock price/volume**: "Adidas Sees Strong Trading Volume" — just price and volume numbers
- **Short interest**: "Short Interest in Adidas AG Rises By 88.3%" — just share counts
- **Boilerplate analyst roundups**: "Columbia Sportswear Receives Consensus Rating of Hold" — rating lists without reasoning
- **Stock comparisons**: "Oxford Industries vs. Under Armour Head to Head" — side-by-side financial metrics
- **Return calculators**: "$100 Invested In Deckers 20 Years Ago" — just math
- **Sale listings**: "Under Armour Tech 2.0 tee now $15" — `false_positive` only when the article gives every brand nothing but retail details; a product article is decided per brand by the Product-article rule below, not by its genre

**Telltale signs of template articles:** "Get Free Report", "according to its most recent 13F filing", "200-day moving average of $X", "Several other research analysts also recently issued reports", "shares of the footwear maker". Sources like MarketBeat, MarketScreener, HoldingsChannel, tickerreport.com, themarketsdaily.com.

### Classification Examples

| Article Type | Example | Has Substantive Content? | Label |
|-------------|---------|-------------------------|-------|
| Raw metrics | "NKE stock up 4% today, 50-day MA $94.67" | No | `false_positive` |
| Short interest | "ANTA short interest up 535%, ratio 0.8 days" | No | `false_positive` |
| Institutional 13F | "Vontobel Holding Boosts Position in Under Armour" | No | `false_positive` |
| Boilerplate ratings | "Columbia Sportswear Receives Consensus PT of $60.50" | No | `false_positive` |
| Stock comparison | "Reviewing Victoria's Secret & Allbirds" (just metrics) | No | `false_positive` |
| Sale listing | "Under Armour Tech 2.0 tee now $15" (name, price and a one-line blurb or a line of specs only) | No | `false_positive` |
| Sale listing with a description | "Brooks Adrenaline GTS 24 now $20 off" (describes cushioning, support, upper) | Yes | `skipped` |
| Release announcement | "Reebok Angel Reese 1 releases November 14" | Yes | `skipped` |
| Analyst with reasoning | "Needham downgrades Nike citing brand weakness" | Yes | `skipped` |
| Earnings + context | "EPS $1.50 missed estimates, CEO announces restructuring" | Yes | `skipped` |
| Investment thesis | "Sinking 48%, Is Lululemon a Buying Opportunity?" | Yes | `skipped` |
| Business analysis | "Deckers record revenue driven by Hoka global demand" | Yes | `skipped` |

### is_sportswear_brand Policy

`is_sportswear_brand` is about **substantive content**, not just identity:
- `true` → Article has substantive content (product descriptions, reviews or releases per the Product-article rule below, business news, strategy, analyst commentary **with reasoning**)
- `false` → Brand refers to something else OR pure stock metrics/template content only

**The critical distinction:** Does the article contain **human-written analysis or commentary** about the brand's business? Or is it **auto-generated/template content** that just plugs financial data into a standard format? Template content = false positive.

**Finance Category Test:** Would this be useful with a "Finance" category? YES → `is_sportswear_brand: true`. Just raw Yahoo Finance metrics or template 13F data → `false`.

**Product-article rule** (prompt v1.11.0, #192): one rule for every product article, built like the Finance Category Test above. Judge each brand by what the article gives that brand, not by the article's genre: would it be useful under a future "Product" category? Product articles never receive ESG categories either way.
- `is_sportswear_brand: true` → for this brand, the article describes or evaluates a specific product, or reports its release (what it is made of or does, what is new, how it performs, or that it is releasing or launching and when). A release date counts on its own, however it is phrased: "releases November 14", "available from November 14" or "goes on sale November 14" for a product not sold before
- `is_sportswear_brand: false` → for this brand, the article gives only retail details: product names, prices, discounts, links, stock or where to buy, or promo codes. A restock, the dates a discount or sale event runs ("20% off until Sunday"), or a one-line sales blurb ("perfect for cold months") is a retail detail, not a release or a description

The genre is a hint, not the test: reviews, release announcements and collaboration launches are usually `true`, and sale listings, price-drop alerts, deals roundups and gift guides usually `false`. A release roundup is `true` for each brand whose release it describes and `false` for a brand it only names with a price or a link; a buying guide that tested the products is `true` for each brand whose product it assesses; a review that also mentions a discount is `true`; a sale listing that also describes the product (what it is made of, how it is built, how it performs, or how buyers say it holds up) is `true`, whatever the sale framing, while a line of specs (capacity, sizes, a feature list) is a retail detail; a release announcement is `true` even when it gives only the product, the date and a price.

v1.7.0 through v1.10.0 state both "shopping/deals → `false`" and "product announcements, reviews, sales → `true`" (v1.0.0 to v1.6.0 state only the second); v1.11.0 replaces both with this rule. v1.11.0 also rewords the brand-specific notes for Mizuno, ANTA, Vans and Decathlon, which listed the article topics that count ("Only ANTA Sports articles about shoes, apparel, athletes, or sports sponsorships are valid"), so that they say only which company is the brand and leave substantive content to the rules above.

v1.11.0's text was revised in place on 2026-10-22, after the first contrastive eval (#215) and before any database row recorded it. The articles behind its two sale-listing `false` examples (Skechers slippers reduced to £19, Brooks Adrenaline GTS 24 $20 off) both describe the product, so under the rule they are `true`: the Brooks example moved to the `true` side, the Skechers one was dropped, and an invented name-price-blurb listing took the `false` side. The revision also states that judging every brand is not a reason to find a category for it (no Digital Transformation for marketing, app features or facilities, no Environmental for a product's materials), after the eval found replies whose reasoning said a category did not apply while assigning it.

### Every brand judged (reply disposition)

Prompt v1.11.0 asks for every brand in the prompt's brand list in `brand_analyses` exactly once, with `is_sportswear_brand` set, and for empty `categories` when no ESG category applies. Earlier versions (v1.10.0 was production until v1.12.0 replaced it on 2026-10-23, #219) tell the model to leave out a sportswear brand with no ESG content, and v1.8.0 to v1.10.0 also a brand mentioned only as a competitor, reference point or context, which is not sportswear by the brand-verification rules. So a reply that leaves a brand out does not say what the brand was.

The article's status is decided by `dispose_reply` (`src/labeling/disposition.py`, D032) under every prompt version. Brand names are matched ignoring case and surrounding space; a name the reply spells differently counts as left out, and so does an analysis that sets no `is_sportswear_brand` and no applicable category (the reply model defaults the field to true, which is not a verdict).

| Reply | Status / reason | Judged? |
|---|---|---|
| No analyses, or none that states a verdict | `skipped` / "No ESG content found" | no |
| A sportswear brand with an applicable category | `labeled` | yes |
| Sportswear brands, none with an applicable category | `skipped` / "No ESG categories apply to sportswear brands" | yes |
| Only non-sportswear brands, a sent brand left out | `skipped` / "Some brands not judged; the rest not sportswear" | no |
| Only non-sportswear brands, covering every sent brand | `false_positive` | yes |

The fourth row is a change for every prompt version: until #192 such a reply was stored as `false_positive`, although under v1.10.0 the brand left out may be a sportswear brand with no ESG content, which that version tells the model to omit. Rows already stored that way keep their status until they are relabeled. The rows the fourth row writes are `unjudged` in the FP export, so they are left out by default, and `--include-unjudged-skips` would put them back as positives. The `esg-prefilter` export reads every `skipped` row as `has_esg=0`, these included (EP is on hold). The run summary's `Incomplete replies` line counts replies that left out at least one brand sent, an empty reply included. Under v1.10.0 that includes every "No ESG content found" reply, so the count measures compliance only under v1.11.0. It is stored per run in `labeling_runs.incomplete_replies` (from #192 PR2; NULL on earlier runs), so read it with `labeling_runs.prompt_version`.

### Content type (prompt v1.12.0, D037)

From v1.12.0 a reply does not state `is_sportswear_brand`. Each brand analysis writes `reasoning` first and then a `content_type`, under structured outputs: the version's `config.json` names `reply_schema.json`, and the labeler sends it as `output_config.format` (beside `effort` on Haiku 5.5). A version without a schema is called exactly as before. `BrandAnalysis` derives the verdict at parse time from one mapping, `CONTENT_TYPE_RULES` in `src/labeling/models.py`:

| `content_type` | What the article gives the brand | `is_sportswear_brand` | Categories |
|---|---|---|---|
| `other_entity` | The name means something else (the animal, the region, vehicles) | false | dropped |
| `mention_only` | A reference point, a tangential mention, or template financial data | false | dropped |
| `retail_only` | Only retail details, per the Product-article rule | false | dropped |
| `product` | A description, evaluation or release of its product, per the Product-article rule | true | dropped |
| `financial` | Stock, earnings or analyst coverage with human-written analysis | true | dropped |
| `business` | Substantive news about the company: strategy, operations, ESG | true | kept |

`not_sportswear_reason` is `"<content_type>: <reasoning>"`, so `false_positive` reasons stored under v1.12.0 start with the type (the retail-only group is separable in the FP export, #154 PR2). The categories a type drops are recorded on the analysis (`gated_categories`) and listed in the eval report; they are not stored. `dispose_reply` is unchanged and counts a content type as a verdict. Replies under v1.0–v1.11 carry no content type and parse as they always have.

Three rules decide between types (revised in place after #219's P1 runs, before any row recorded v1.12.0). ESG content decides first: an article that reports the brand's ESG initiative, partnership, donation, policy, commitment or controversy is `business`, however much of it is product, campaign or financial content, unless the article only recounts the event as history or background. An owned brand that is not itself tracked (UGG or Teva for Deckers, Jordan Brand for Nike) is judged as its owner; a tracked one (Converse, Hoka) is judged on its own. A brand the article is not about is `mention_only` even when the line about it is financial: one or two lines in a market wrap, a place in a list of companies an event affects, the goods or store in an incident it is not a party to, or an automated valuation template (DCF, fair value).

### FP Classifier Limitation

The FP classifier may underweight less common brands (Anta, Puma, Xtep) compared to Nike/Adidas/Lululemon. Review FP classifier predictions for these brands during labeling audits.
