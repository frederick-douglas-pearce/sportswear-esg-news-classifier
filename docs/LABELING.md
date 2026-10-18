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

| Component | Approximate Cost |
|-----------|------------------|
| OpenAI embeddings (text-embedding-3-small) | ~$0.02 per 1000 articles |
| Claude labeling | ~$3-5 per 1000 articles (Haiku 4.5) |
| **Total** | **~$3-5 per 1000 articles** |

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
- **Sale listings**: "Skechers slippers reduced to £19" — `false_positive` only when the article gives every brand nothing but retail details; a product article is decided per brand by the Product-article rule below, not by its genre

**Telltale signs of template articles:** "Get Free Report", "according to its most recent 13F filing", "200-day moving average of $X", "Several other research analysts also recently issued reports", "shares of the footwear maker". Sources like MarketBeat, MarketScreener, HoldingsChannel, tickerreport.com, themarketsdaily.com.

### Classification Examples

| Article Type | Example | Has Substantive Content? | Label |
|-------------|---------|-------------------------|-------|
| Raw metrics | "NKE stock up 4% today, 50-day MA $94.67" | No | `false_positive` |
| Short interest | "ANTA short interest up 535%, ratio 0.8 days" | No | `false_positive` |
| Institutional 13F | "Vontobel Holding Boosts Position in Under Armour" | No | `false_positive` |
| Boilerplate ratings | "Columbia Sportswear Receives Consensus PT of $60.50" | No | `false_positive` |
| Stock comparison | "Reviewing Victoria's Secret & Allbirds" (just metrics) | No | `false_positive` |
| Sale listing | "Skechers slippers reduced to £19" (name, price and link only) | No | `false_positive` |
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

The genre is a hint, not the test: reviews, release announcements and collaboration launches are usually `true`, and sale listings, price-drop alerts, deals roundups and gift guides usually `false`. A release roundup is `true` for each brand whose release it describes and `false` for a brand it only names with a price or a link; a buying guide that tested the products is `true` for each brand whose product it assesses; a review that also mentions a discount is `true`; a release announcement is `true` even when it gives only the product, the date and a price.

v1.7.0 through v1.10.0 state both "shopping/deals → `false`" and "product announcements, reviews, sales → `true`" (v1.0.0 to v1.6.0 state only the second); v1.11.0 replaces both with this rule. v1.11.0 also rewords the brand-specific notes for Mizuno, ANTA, Vans and Decathlon, which listed the article topics that count ("Only ANTA Sports articles about shoes, apparel, athletes, or sports sponsorships are valid"), so that they say only which company is the brand and leave substantive content to the rules above.

### Every brand judged (reply disposition)

Prompt v1.11.0 asks for every brand in the prompt's brand list in `brand_analyses` exactly once, with `is_sportswear_brand` set, and for empty `categories` when no ESG category applies. Earlier versions (production is still v1.10.0) tell the model to leave out a sportswear brand with no ESG content, and v1.8.0 to v1.10.0 also a brand mentioned only as a competitor, reference point or context, which is not sportswear by the brand-verification rules. So a reply that leaves a brand out does not say what the brand was.

The article's status is decided by `dispose_reply` (`src/labeling/disposition.py`, D032) under every prompt version. Brand names are matched ignoring case and surrounding space; a name the reply spells differently counts as left out, and so does an analysis that sets no `is_sportswear_brand` and no applicable category (the reply model defaults the field to true, which is not a verdict).

| Reply | Status / reason | Judged? |
|---|---|---|
| No analyses, or none that states a verdict | `skipped` / "No ESG content found" | no |
| A sportswear brand with an applicable category | `labeled` | yes |
| Sportswear brands, none with an applicable category | `skipped` / "No ESG categories apply to sportswear brands" | yes |
| Only non-sportswear brands, a sent brand left out | `skipped` / "Some brands not judged; the rest not sportswear" | no |
| Only non-sportswear brands, covering every sent brand | `false_positive` | yes |

The fourth row is a change for every prompt version: until #192 such a reply was stored as `false_positive`, although under v1.10.0 the brand left out may be a sportswear brand with no ESG content, which that version tells the model to omit. Rows already stored that way keep their status until they are relabeled. The rows the fourth row writes are `unjudged` in the FP export, so they are left out by default, and `--include-unjudged-skips` would put them back as positives. The `esg-prefilter` export reads every `skipped` row as `has_esg=0`, these included (EP is on hold). The run summary's `Incomplete replies` line counts replies that left out at least one brand sent, an empty reply included. Under v1.10.0 that includes every "No ESG content found" reply, so the count measures compliance only under v1.11.0. It is printed only; it is not stored in `labeling_runs`.

### FP Classifier Limitation

The FP classifier may underweight less common brands (Anta, Puma, Xtep) compared to Nike/Adidas/Lululemon. Review FP classifier predictions for these brands during labeling audits.
