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
```

**Export Formats (JSONL):**

| Dataset | Fields | Use Case |
|---------|--------|----------|
| `fp` | article_id, title, content, brands, is_sportswear | False positive brand classifier |
| `esg-prefilter` | article_id, title, content, brands, has_esg | ESG content pre-filter |
| `esg-labels` | article_id, title, content, brand, E/S/G/D flags + sentiment | Multi-label ESG classifier |

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
- **Shopping/deals**: "Skechers slippers reduced to £19" — product promotion, not business news

**Telltale signs of template articles:** "Get Free Report", "according to its most recent 13F filing", "200-day moving average of $X", "Several other research analysts also recently issued reports", "shares of the footwear maker". Sources like MarketBeat, MarketScreener, HoldingsChannel, tickerreport.com, themarketsdaily.com.

### Classification Examples

| Article Type | Example | Has Substantive Content? | Label |
|-------------|---------|-------------------------|-------|
| Raw metrics | "NKE stock up 4% today, 50-day MA $94.67" | No | `false_positive` |
| Short interest | "ANTA short interest up 535%, ratio 0.8 days" | No | `false_positive` |
| Institutional 13F | "Vontobel Holding Boosts Position in Under Armour" | No | `false_positive` |
| Boilerplate ratings | "Columbia Sportswear Receives Consensus PT of $60.50" | No | `false_positive` |
| Stock comparison | "Reviewing Victoria's Secret & Allbirds" (just metrics) | No | `false_positive` |
| Shopping/deals | "Patagonia winter sale deals — 8 picks" | No | `false_positive` |
| Analyst with reasoning | "Needham downgrades Nike citing brand weakness" | Yes | `skipped` |
| Earnings + context | "EPS $1.50 missed estimates, CEO announces restructuring" | Yes | `skipped` |
| Investment thesis | "Sinking 48%, Is Lululemon a Buying Opportunity?" | Yes | `skipped` |
| Business analysis | "Deckers record revenue driven by Hoka global demand" | Yes | `skipped` |

### is_sportswear_brand Policy

`is_sportswear_brand` is about **substantive content**, not just identity:
- `true` → Article has substantive content (products, business news, strategy, analyst commentary **with reasoning**)
- `false` → Brand refers to something else OR pure stock metrics/template content only

**The critical distinction:** Does the article contain **human-written analysis or commentary** about the brand's business? Or is it **auto-generated/template content** that just plugs financial data into a standard format? Template content = false positive.

**Finance Category Test:** Would this be useful with a "Finance" category? YES → `is_sportswear_brand: true`. Just raw Yahoo Finance metrics or template 13F data → `false`.

### FP Classifier Limitation

The FP classifier may underweight less common brands (Anta, Puma, Xtep) compared to Nike/Adidas/Lululemon. Review FP classifier predictions for these brands during labeling audits.
