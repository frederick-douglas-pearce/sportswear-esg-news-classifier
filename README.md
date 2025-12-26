# ESG News Classifier for Sportswear Brands

A multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for major sportswear brands including Nike, Adidas, Puma, Under Armour, Lululemon, Patagonia, Columbia Sportswear, New Balance, ASICS, and Reebok.

## System Architecture

```mermaid
flowchart TB
    subgraph sources["Data Sources"]
        newsdata["NewsData.io API"]
        gdelt["GDELT DOC 2.0 API"]
    end

    subgraph collection["Phase 1: Data Collection"]
        api_client["API Clients<br/><i>api_client.py, gdelt_client.py</i>"]
        scraper["Article Scraper<br/><i>newspaper4k + langdetect</i>"]
    end

    subgraph storage["PostgreSQL + pgvector"]
        articles[("articles<br/><i>metadata, full_content</i>")]
        chunks[("article_chunks<br/><i>text, embeddings</i>")]
        labels[("brand_labels<br/><i>ESG categories, sentiment</i>")]
        evidence[("label_evidence<br/><i>excerpts, similarity</i>")]
    end

    subgraph labeling["Phase 2: LLM Labeling"]
        chunker["Article Chunker<br/><i>~500 tokens/chunk</i>"]
        embedder["OpenAI Embedder<br/><i>text-embedding-3-small</i>"]
        claude["Claude Sonnet<br/><i>ESG classification</i>"]
        matcher["Evidence Matcher<br/><i>exact/fuzzy/semantic</i>"]
    end

    subgraph output["Output"]
        training["Training Data<br/><i>Per-brand ESG labels</i>"]
    end

    newsdata --> api_client
    gdelt --> api_client
    api_client --> scraper
    scraper --> articles

    articles --> chunker
    chunker --> chunks
    chunks --> embedder
    embedder --> chunks

    articles --> claude
    claude --> labels
    claude --> matcher
    chunks --> matcher
    matcher --> evidence

    labels --> training
    evidence --> training
```

## Project Roadmap

### Phase 1: Data Collection ✅
- [x] NewsData.io API integration
- [x] GDELT DOC 2.0 API integration (free, 3 months history)
- [x] Article scraping with newspaper4k
- [x] PostgreSQL + pgvector storage
- [x] Automated cron scheduling (8x daily: 4 NewsData + 4 GDELT)
- [x] Historical backfill script for GDELT
- [x] Target: 1,000-2,000 articles over 10-14 days

### Phase 2: Data Labeling ✅
- [x] LLM-based labeling pipeline with Claude Sonnet
- [x] Per-brand ESG category labels with ternary sentiment
- [x] Article chunking for evidence extraction
- [x] OpenAI embeddings for semantic evidence matching
- [x] Evidence linking to source text chunks
- [x] Labeling CLI with dry-run and batch support

### Phase 3: Model Development (Current)
- [x] Export labeled data for training (JSONL format for 3 classifier types)
- [x] False positive brand detection and cleanup tools
- [x] False Positive Classifier - 3-notebook pipeline complete
  - fp1: EDA + sentence-transformer/NER feature engineering
  - fp2: Model selection + hyperparameter tuning (3-fold CV)
  - fp3: Test evaluation + threshold optimization + deployment export
  - Random Forest achieves Test F2: 0.974, Recall: 98.8%
  - Supporting modules in `src/fp1_nb/`, `src/fp2_nb/`, `src/fp3_nb/`
- [x] ESG Pre-filter Classifier - 3-notebook pipeline complete
  - ep1: EDA + TF-IDF/LSA feature engineering with ESG vocabularies
  - ep2: Model selection + hyperparameter tuning (3-fold CV)
  - ep3: Test evaluation + threshold optimization + deployment export
  - Logistic Regression achieves Test F2: 0.931, Recall: 100%
  - Supporting modules in `src/ep1_nb/`, `src/ep2_nb/`, `src/ep3_nb/`
- [ ] ESG Multi-label Classifier: Category classification with sentiment
- [ ] Advanced: Fine-tuned DistilBERT/RoBERTa

### Phase 4: Evaluation & Explainability
- [ ] Per-category Precision, Recall, F1-score
- [ ] Hamming Loss (multi-label specific)
- [ ] SHAP values for model explainability

### Phase 5: Deployment (Current)
- [x] Unified FastAPI REST API (`scripts/predict.py`)
- [x] Unified training script (`scripts/train.py`)
- [x] Deployment module (`src/deployment/`)
- [x] Multi-stage Dockerfile
- [x] Docker Compose integration
- [x] GitHub Actions CI/CD to Google Cloud Run
- [x] Model registry with version tracking
- [x] Prediction logging for drift monitoring
- [ ] Streamlit dashboard (optional)

## Table of Contents

- [System Architecture](#system-architecture)
- [Project Roadmap](#project-roadmap)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Installation](#2-installation)
  - [3. Start the Database](#3-start-the-database)
- [Running the News Collection Script](#running-the-news-collection-script)
  - [Testing (Dry Run Mode)](#testing-dry-run-mode)
  - [Production Collection](#production-collection)
  - [Scheduled Collection (Cron)](#scheduled-collection-cron)
  - [Scrape-Only Mode](#scrape-only-mode)
  - [Command-Line Options](#command-line-options)
- [LLM-Based Article Labeling](#llm-based-article-labeling)
  - [LLM Labeling Workflow](#llm-labeling-workflow)
  - [Running the Labeling Pipeline](#running-the-labeling-pipeline)
  - [Labeling Command-Line Options](#labeling-command-line-options)
  - [Cost Estimation](#cost-estimation)
  - [Exporting Training Data](#exporting-training-data)
  - [ML Classifier Notebooks](#ml-classifier-notebooks)
  - [ML Classifier Opportunities](#ml-classifier-opportunities)
- [FP Classifier Deployment](#fp-classifier-deployment)
  - [Local Deployment](#local-deployment-without-docker)
  - [Docker Deployment](#docker-deployment)
  - [API Endpoints](#api-endpoints)
  - [Retraining the Model](#retraining-the-model)
- [Environment Variables](#environment-variables)
- [Database Schema](#database-schema)
  - [Articles Table](#articles-table)
  - [Collection Runs Table](#collection-runs-table)
  - [Labeling Tables](#labeling-tables)
- [Querying the Database](#querying-the-database)
  - [Quick Stats](#quick-stats)
  - [Detailed Queries](#detailed-queries)
  - [Labeling Queries](#labeling-queries)
  - [Interactive Database Access](#interactive-database-access)
- [ESG Category Structure](#esg-category-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Project Structure

```
sportswear-esg-news-classifier/
├── docker-compose.yml          # PostgreSQL + FP Classifier API containers
├── Dockerfile                  # Multi-stage build for FP Classifier API
├── pyproject.toml              # Project dependencies and metadata (uv/pip)
├── .env.example                # Environment variable template
├── .env                        # Local environment variables (not committed)
├── logs/                       # Application logs
├── scripts/
│   ├── collect_news.py         # CLI script for data collection
│   ├── label_articles.py       # CLI script for LLM-based labeling
│   ├── export_training_data.py # Export labeled data for ML training
│   ├── gdelt_backfill.py       # Historical backfill script (3 months)
│   ├── cleanup_non_english.py  # Remove non-English articles from database
│   ├── cleanup_false_positives.py # Identify/remove false positive brand matches
│   ├── train.py                # Unified training script for FP/EP classifiers
│   ├── predict.py              # Unified FastAPI service for all classifiers
│   ├── retrain.py              # Retrain models with version management
│   ├── monitor_drift.py        # Monitor prediction drift for deployed models
│   ├── cron_collect.sh         # Cron wrapper for NewsData.io collection
│   ├── cron_scrape.sh          # Cron wrapper for GDELT collection
│   └── setup_cron.sh           # User-friendly cron management
├── notebooks/
│   ├── fp1_EDA_FE.ipynb              # FP: EDA & Feature Engineering
│   ├── fp2_model_selection_tuning.ipynb  # FP: Model selection & tuning
│   ├── fp3_model_evaluation_deployment.ipynb  # FP: Test evaluation & deployment
│   ├── ep1_EDA_FE.ipynb              # EP: EDA & Feature Engineering
│   ├── ep2_model_selection_tuning.ipynb  # EP: Model selection & tuning
│   └── ep3_model_evaluation_deployment.ipynb  # EP: Test evaluation & deployment
├── models/                     # Saved ML models (gitignored)
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings, brands, keywords, API configuration
│   │   ├── api_client.py       # NewsData.io API wrapper with query generation
│   │   ├── gdelt_client.py     # GDELT DOC 2.0 API wrapper (free, 3 months history)
│   │   ├── scraper.py          # Full article text extraction with language detection
│   │   ├── database.py         # PostgreSQL operations with SQLAlchemy
│   │   ├── models.py           # SQLAlchemy models (Article, CollectionRun, labeling tables)
│   │   └── collector.py        # Orchestrates API collection + scraping phases
│   ├── labeling/
│   │   ├── __init__.py
│   │   ├── config.py           # Labeling settings, prompts, category definitions
│   │   ├── models.py           # Pydantic models for LLM responses
│   │   ├── chunker.py          # Paragraph-based article chunking
│   │   ├── embedder.py         # OpenAI embedding wrapper
│   │   ├── labeler.py          # Claude labeling logic
│   │   ├── evidence_matcher.py # Match excerpts to chunks via similarity
│   │   ├── database.py         # Labeling-specific DB operations
│   │   └── pipeline.py         # Orchestrates full labeling flow
│   ├── fp1_nb/                 # FP classifier - EDA & feature engineering
│   │   ├── __init__.py
│   │   ├── data_utils.py       # Data loading, splitting, target analysis
│   │   ├── eda_utils.py        # Text analysis, brand distribution, word frequencies
│   │   ├── preprocessing.py    # Text cleaning, feature engineering
│   │   ├── feature_transformer.py  # Sentence transformer + NER features
│   │   ├── ner_analysis.py     # Named entity recognition utilities
│   │   └── modeling.py         # GridSearchCV, model evaluation, comparison
│   ├── fp2_nb/                 # FP classifier - model selection & tuning
│   │   ├── __init__.py
│   │   └── overfitting_analysis.py  # Train-val gap visualization
│   ├── fp3_nb/                 # FP classifier - evaluation & deployment
│   │   ├── __init__.py
│   │   ├── threshold_optimization.py  # Threshold tuning for target recall
│   │   └── deployment.py       # Pipeline export utilities
│   ├── ep1_nb/                 # EP classifier - EDA & feature engineering
│   │   ├── __init__.py
│   │   ├── data_utils.py       # Data loading, splitting, target analysis
│   │   ├── eda_utils.py        # Text analysis, brand distribution
│   │   ├── preprocessing.py    # Text cleaning, feature engineering
│   │   ├── feature_transformer.py  # EPFeatureTransformer with ESG vocabularies
│   │   └── modeling.py         # GridSearchCV, model evaluation
│   ├── ep2_nb/                 # EP classifier - model selection & tuning
│   │   ├── __init__.py
│   │   └── overfitting_analysis.py  # Train-val gap visualization
│   ├── ep3_nb/                 # EP classifier - evaluation & deployment
│   │   ├── __init__.py
│   │   ├── threshold_optimization.py  # Threshold tuning for target recall
│   │   └── deployment.py       # Pipeline export utilities
│   └── deployment/             # Production deployment module
│       ├── __init__.py
│       ├── config.py           # Configuration and risk level mapping
│       ├── data.py             # Data loading and splitting utilities
│       ├── preprocessing.py    # Text preprocessing for API
│       └── prediction.py       # FPClassifier wrapper class
└── tests/
    ├── conftest.py             # Shared pytest fixtures
    ├── test_api_client.py      # NewsData.io client unit tests
    ├── test_gdelt_client.py    # GDELT client unit tests
    ├── test_scraper.py         # Scraper and language detection tests
    ├── test_collector.py       # Collector unit tests
    ├── test_database.py        # Database integration tests
    ├── test_chunker.py         # Article chunker unit tests
    ├── test_labeler.py         # LLM labeling and response parsing tests
    ├── test_embedder.py        # OpenAI embedder unit tests
    ├── test_evidence_matcher.py # Evidence matching unit tests
    └── test_labeling_pipeline.py # Labeling pipeline unit tests
```

## Quick Start

### 1. Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- NewsData.io API key ([get free tier here](https://newsdata.io/register)) - *optional if using GDELT*

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sportswear-esg-news-classifier

# Install dependencies with uv
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev

# Create environment file from template
cp .env.example .env

# Edit .env and add your API keys, e.g.
# NEWSDATA_API_KEY=your_api_key_here, etc
```

### 3. Start the Database

```bash
# Start PostgreSQL with pgvector extension
docker compose up -d

# Verify it's running
docker ps
```

The database will be available at `localhost:5434`.

## Running the News Collection Script

All commands use `uv run` to execute within the project's virtual environment. Alternatively, you can activate the venv first with `source .venv/bin/activate` and omit `uv run`.

### Testing (Dry Run Mode)

Before running a full collection, test with a dry run to verify your setup:

```bash
# Basic dry run - shows what would be done without saving
uv run python scripts/collect_news.py --dry-run --max-calls 5

# Dry run with verbose output for debugging
uv run python scripts/collect_news.py --dry-run --max-calls 5 -v
```

**What dry run does:**
- Connects to the NewsData.io API and fetches articles
- Displays statistics about what would be saved
- Does NOT write anything to the database
- Useful for testing API key and connectivity

**Expected output:**
```
2024-12-14 10:00:00 - __main__ - INFO - Starting ESG News Collection
2024-12-14 10:00:01 - src.data_collection.collector - INFO - Starting API collection with 120 queries, max 5 calls
2024-12-14 10:00:05 - src.data_collection.collector - INFO - [DRY RUN] Would save 10 articles
2024-12-14 10:00:05 - __main__ - INFO - Collection complete:
2024-12-14 10:00:05 - __main__ - INFO -   API calls made: 5
2024-12-14 10:00:05 - __main__ - INFO -   New articles: 50
2024-12-14 10:00:05 - __main__ - INFO -   Duplicates skipped: 0
2024-12-14 10:00:05 - __main__ - INFO -   Articles scraped: 0
2024-12-14 10:00:05 - __main__ - INFO -   Scrape failures: 0
```

### Production Collection

```bash
# Run full daily collection using NewsData.io (requires API key)
uv run python scripts/collect_news.py

# Run collection using GDELT (free, no API key needed, 3 months history)
uv run python scripts/collect_news.py --source gdelt

# GDELT with shorter time window (for frequent collection)
uv run python scripts/collect_news.py --source gdelt --timespan 6h

# GDELT historical backfill for specific date range
uv run python scripts/collect_news.py --source gdelt --start-date 2025-10-01 --end-date 2025-10-07

# With custom limits
uv run python scripts/collect_news.py --max-calls 100 --scrape-limit 50

# Verbose mode for monitoring
uv run python scripts/collect_news.py -v
```

### GDELT Historical Backfill

To collect 3 months of historical data from GDELT in weekly batches:

```bash
# Run full 3-month backfill
uv run python scripts/gdelt_backfill.py

# Test first batch only (dry run)
uv run python scripts/gdelt_backfill.py --dry-run --max-calls 5

# Resume from a specific date
uv run python scripts/gdelt_backfill.py --start-from 2025-11-01

# Backfill only 1 month
uv run python scripts/gdelt_backfill.py --months 1
```

### Scheduled Collection (Cron)

Set up automatic collection with two cron jobs:
- **NewsData job**: Fetches from NewsData.io API + scrapes (4x daily, requires API key)
- **GDELT job**: Fetches from GDELT API + scrapes (4x daily, free, no key needed)

```bash
# Install both cron jobs
./scripts/setup_cron.sh install

# Check status
./scripts/setup_cron.sh status

# Remove both cron jobs
./scripts/setup_cron.sh remove

# Install/remove individual jobs
./scripts/setup_cron.sh install-collect   # NewsData only
./scripts/setup_cron.sh install-scrape    # GDELT only
./scripts/setup_cron.sh remove-collect
./scripts/setup_cron.sh remove-scrape

# View logs
tail -f logs/collection_$(date +%Y%m%d).log   # NewsData logs
tail -f logs/gdelt_$(date +%Y%m%d).log        # GDELT logs
```

**Schedule:**
| Time | Job | Description |
|------|-----|-------------|
| 12am, 6am, 12pm, 6pm | NewsData | NewsData.io API (50 calls) + scrape (100 articles) |
| 3am, 9am, 3pm, 9pm | GDELT | GDELT API (6h window) + scrape (100 articles) |

### Scrape-Only Mode

If you have articles already fetched but not yet scraped:

```bash
# Only scrape pending articles (skip API collection)
uv run python scripts/collect_news.py --scrape-only

# With custom limit
uv run python scripts/collect_news.py --scrape-only --scrape-limit 50
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source SOURCE` | API source: `newsdata` or `gdelt` | `newsdata` |
| `--dry-run` | Don't save to database, just show what would be done | False |
| `--max-calls N` | Maximum API calls to make | 200 |
| `--scrape-only` | Only scrape pending articles, skip API collection | False |
| `--scrape-limit N` | Maximum articles to scrape | 100 |
| `--timespan SPAN` | GDELT only: relative time window (e.g., `6h`, `1d`, `1w`, `3m`) | `3m` |
| `--start-date DATE` | GDELT only: start date for historical collection (YYYY-MM-DD) | - |
| `--end-date DATE` | GDELT only: end date for historical collection (YYYY-MM-DD) | - |
| `-v, --verbose` | Enable verbose/debug logging | False |

## AI-Based Article Labeling

The project uses a **hybrid LLM + ML approach** for article classification:

1. **LLM Labeling (Claude Sonnet)**: High-quality labeling of articles into ESG categories with per-brand sentiment analysis. This generates training data for ML classifiers.

2. **ML Classifiers**: Cost-efficient models trained on LLM-labeled data that can filter and classify articles at little to no cost (~$0 vs ~$15/1000 articles).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Article Classification Pipeline                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  New Article ──► FP Classifier ──► ESG Pre-filter ──► ESG Classifier        │
│                  (Is this about     (Has ESG          (Category +           │
│                   sportswear?)       content?)         Sentiment)           │
│                       │                  │                  │               │
│                       ▼                  ▼                  ▼               │
│               ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│               │ False Positive│  │ No ESG Content│  │ High-Confidence│      │
│               │   (Skip)      │  │   (Skip)      │  │  Prediction   │       │
│               └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                             │               │
│                                         Low Confidence ─────┘               │
│                                                │                            │
│                                                ▼                            │
│                                    ┌───────────────────┐                    │
│                                    │  Claude Sonnet    │                    │
│                                    │  (Fallback LLM)   │                    │
│                                    └───────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Cost Savings**: By filtering false positives (~15% of articles) and non-ESG content (~30% of articles) with ML classifiers before LLM labeling, the pipeline can reduce Claude API costs by 40-50%.

### LLM Labeling Workflow

The LLM labeling pipeline processes articles through these steps:

1. **Chunking**: Articles are split into paragraph-based chunks (~500 tokens each) with character position tracking
2. **Embedding**: Chunks are embedded using OpenAI's `text-embedding-3-small` model for semantic search
3. **LLM Labeling**: Claude Sonnet analyzes each article and returns structured JSON with:
   - Per-brand ESG category labels (Environmental, Social, Governance, Digital Transformation)
   - Ternary sentiment for each category (+1 positive, 0 neutral, -1 negative)
   - Supporting evidence quotes from the article
   - Confidence score and reasoning
4. **Evidence Matching**: Evidence excerpts are linked back to article chunks via exact match, fuzzy match, or embedding similarity

The labeled data is then exported to train ML classifiers that can handle routine classification at scale.

### Data Labeling Scope

Articles are only labeled if they are **primarily about** the sportswear brand's activities. The labeling pipeline filters out two types of false positives:

**1. Brand Name Conflicts**: When a brand name refers to something other than the sportswear company:
- "Puma" (the animal), "Patagonia" (the region), "Columbia" (the country/university)
- "Vans" (vehicles), "Anta" (Indian political district), "Decathlon" (investment firms)

**2. Tangential Brand Mentions**: When the brand name correctly refers to the sportswear company, but the article is not actually about that brand:
- Former executives now working at other companies (e.g., "Ex-Nike VP joins Tech Startup")
- Biographical context in profiles about people who no longer work at the brand
- Articles about other companies that briefly mention a sportswear brand for comparison

**Key Test**: Is this article primarily about the sportswear brand's current activities, products, or ESG initiatives? If the brand is only mentioned as background context, historical reference, or biographical detail, it should be marked as a false positive.

### Running the Labeling Pipeline

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

### Labeling Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size N` | Number of articles to process | 10 |
| `--dry-run` | Show what would be done without saving | False |
| `--article-id UUID` | Label a specific article | - |
| `--skip-chunking` | Skip chunking for articles that already have chunks | False |
| `--skip-embedding` | Skip embedding generation | False |
| `--stats` | Show labeling statistics and exit | False |
| `-v, --verbose` | Enable verbose/debug logging | False |

### Cost Estimation

| Component | Approximate Cost |
|-----------|------------------|
| OpenAI embeddings (text-embedding-3-small) | ~$0.02 per 1000 articles |
| Claude Sonnet labeling | ~$10-15 per 1000 articles |
| **Total** | **~$15 per 1000 articles** |

### Exporting Training Data

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

### ML Classifier Opportunities

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

**3. ESG Multi-label Classifier**
- **Purpose**: Classify articles into specific ESG categories with sentiment, replacing Claude for routine classification
- **Input**: Article title + content + brand name
- **Output**: Multi-label (Environmental, Social, Governance, Digital Transformation) with ternary sentiment (-1, 0, +1)
- **Training Data**: 554 records from `--dataset esg-labels` export
- **Impact**: Replace Claude API calls entirely for high-confidence predictions

### ML Classifier Notebooks

The project includes 3-notebook pipelines for developing ML classifiers. Each pipeline follows the same structure with supporting utility modules for consistent preprocessing and evaluation.

**Two complete classifier pipelines:**
1. **False Positive (FP) Classifier**: Filters non-sportswear brand mentions (modules: `src/fp1_nb/`, `src/fp2_nb/`, `src/fp3_nb/`)
2. **ESG Pre-filter (EP) Classifier**: Identifies articles with ESG content (modules: `src/ep1_nb/`, `src/ep2_nb/`, `src/ep3_nb/`)

#### Notebook Pipeline Overview

```
fp1_EDA_FE.ipynb          fp2_model_selection_tuning.ipynb          fp3_model_evaluation_deployment.ipynb
(EDA & Features)          (Model Selection & Tuning)                 (Test Evaluation & Deployment)
      │                              │                                          │
      ▼                              ▼                                          ▼
┌─────────────────┐        ┌─────────────────────┐              ┌────────────────────────────┐
│ • Data loading  │        │ • Baseline models   │              │ • Load artifacts from      │
│ • EDA           │        │ • GridSearchCV      │              │   fp1 and fp2              │
│ • Feature eng   │ ────►  │ • Train-val gap     │ ──────────►  │ • Final test evaluation    │
│ • Transformer   │        │ • Best model select │              │ • Threshold optimization   │
│   export        │        │ • Model export      │              │ • Pipeline export          │
└─────────────────┘        └─────────────────────┘              └────────────────────────────┘
      │                              │                                          │
      ▼                              ▼                                          ▼
fp_feature_transformer.joblib  fp_best_classifier.joblib          fp_classifier_pipeline.joblib
fp_feature_config.json         fp_cv_metrics.json                  fp_classifier_config.json
```

#### fp1_EDA_FE.ipynb - EDA & Feature Engineering

- **Data Loading**: 993 articles (856 sportswear, 137 false positives)
- **EDA**: Text length distributions, brand distribution, word frequencies
- **Feature Engineering**: Sentence-transformer embeddings + NER brand context features
- **Hyperparameter Tuning**: Optimizes `proximity_window_size` for NER features
- **Exports**: Feature transformer and configuration for fp2

#### fp2_model_selection_tuning.ipynb - Model Selection & Tuning

- **Baseline Models**: LR, RF, HGB with 3-fold stratified CV
- **Hyperparameter Tuning**: GridSearchCV optimizing F2 score
- **Overfitting Analysis**: Train-validation gap visualization
- **Best Model**: Random Forest with `balanced_subsample` class weights
- **Exports**: Best classifier and CV metrics for fp3

**Overfitting Analysis Note:**

The Random Forest model shows training F2 = 1.0 across all hyperparameter combinations, while validation F2 ≈ 0.973. This pattern (perfect training scores) is expected with small datasets and Random Forest's default behavior:

1. **Why train = 1.0**: With only ~530 samples per CV training fold and 390 features, individual trees can perfectly memorize training data when `min_samples_leaf=1`.

2. **Why this is acceptable**: Despite individual tree overfitting, ensemble averaging reduces variance. The key evidence:
   - Train-val gap is only 2.7% (below the 5% warning threshold)
   - CV-to-test gap is negligible (+0.02%), confirming excellent generalization
   - Validation F2 continues improving with more trees (ensemble benefit)

3. **Generalization confirmed**: Test set performance (F2: 0.974) matches CV performance (F2: 0.973), indicating the model generalizes well to unseen data despite the high training scores.

#### fp3_model_evaluation_deployment.ipynb - Test Evaluation & Deployment

- **Test Evaluation**: Final held-out test set evaluation (ONLY notebook using test data)
- **Threshold Optimization**: Find optimal threshold for 98% target recall
- **Pipeline Export**: Complete sklearn Pipeline for deployment

**Performance (Random Forest):**
- CV F2: 0.973, Test F2: 0.974
- Test Recall: 98.8%, Test Precision: 91.9%
- Optimized threshold: 0.605 (at 98% recall)

**Supporting Modules:**
- `src/fp1_nb/` - Data loading, EDA, feature transformer, NER analysis, modeling utilities
- `src/fp2_nb/` - Train-validation gap analysis, overfitting visualization
- `src/fp3_nb/` - Threshold optimization, deployment pipeline utilities

#### ESG Pre-filter (EP) Classifier Pipeline

The EP classifier follows the same 3-notebook structure as FP, with ESG-specific feature engineering.

**ep1_EDA_FE.ipynb - EDA & Feature Engineering**
- **Data Loading**: 870 articles (635 has ESG, 235 no ESG)
- **EDA**: Text length distributions, brand distribution, word frequencies
- **Feature Engineering**: TF-IDF + LSA with ESG-specific vocabulary features
- **Hyperparameter Tuning**: Optimizes `lsa_n_components` for dimensionality reduction
- **Exports**: EPFeatureTransformer and configuration for ep2

**ep2_model_selection_tuning.ipynb - Model Selection & Tuning**
- **Baseline Models**: LR, RF, HGB with 3-fold stratified CV
- **Hyperparameter Tuning**: GridSearchCV optimizing F2 score
- **Overfitting Analysis**: Train-validation gap visualization
- **Best Model**: Logistic Regression with class_weight=None
- **Exports**: Best classifier and CV metrics for ep3

**ep3_model_evaluation_deployment.ipynb - Test Evaluation & Deployment**
- **Test Evaluation**: Final held-out test set evaluation
- **Threshold Optimization**: Find optimal threshold for 99% target recall
- **Pipeline Export**: Complete sklearn Pipeline for deployment

**Performance (Logistic Regression):**
- CV F2: 0.931, Test F2: 0.931
- Test Recall: 100%, Test Precision: 73%
- Optimized threshold: 0.724 (at 99% recall)

**EPFeatureTransformer Key Features:**
- ESG-specific vocabularies: Environmental, Social, Governance, Digital keywords
- TF-IDF with LSA dimensionality reduction (200 components)
- Metadata features from source_name and category

**Supporting Modules:**
- `src/ep1_nb/` - Data loading, EDA, EPFeatureTransformer with ESG vocabularies
- `src/ep2_nb/` - Train-validation gap analysis, overfitting visualization
- `src/ep3_nb/` - Threshold optimization, deployment pipeline utilities

## FP Classifier Deployment

The False Positive Brand Classifier is deployed as a FastAPI REST API service.

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FP Classifier API                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Client ──► FastAPI ──► FPClassifier ──► sklearn Pipeline        │
│              │               │               │                   │
│              │               │               ├── FPFeatureTransformer │
│              │               │               │   (sentence-transformer + NER) │
│              │               │               │                   │
│              │               │               └── RandomForestClassifier │
│              │               │                                   │
│              ▼               ▼                                   │
│         /predict         Threshold                               │
│         /predict/batch   (0.605)                                 │
│         /health                                                  │
│         /model/info                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Local Deployment (without Docker)

```bash
# Install dependencies
uv sync

# Run training (optional - model already trained)
uv run python scripts/train.py --classifier fp --verbose

# Start FP API server
CLASSIFIER_TYPE=fp uv run python scripts/predict.py
# Or with uvicorn directly:
CLASSIFIER_TYPE=fp uv run uvicorn scripts.predict:app --host 0.0.0.0 --port 8000

# Start EP API server (different port)
CLASSIFIER_TYPE=ep uv run uvicorn scripts.predict:app --host 0.0.0.0 --port 8001

# Access API docs
open http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and start the API service
docker compose build fp-classifier-api
docker compose up -d fp-classifier-api

# Check health
curl http://localhost:8000/health

# View logs
docker logs fp-classifier-api

# Stop
docker compose down fp-classifier-api
```

**Note**: The Docker image is ~1-2GB due to sentence-transformers and spaCy models.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for container orchestration |
| `/model/info` | GET | Model metadata and performance metrics |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Classify multiple articles in one request |

### Example API Requests

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}

# Get model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Nike releases new sustainability initiative",
    "content": "The athletic footwear giant unveiled plans...",
    "brands": ["Nike"]
  }'
# {"is_sportswear":true,"probability":0.935,"risk_level":"high","threshold":0.605}

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"title": "Nike shoe sales soar", "content": "...", "brands": ["Nike"]},
      {"title": "Adidas sponsors tournament", "content": "...", "brands": ["Adidas"]}
    ]
  }'
```

### Response Fields

| Field | Description |
|-------|-------------|
| `is_sportswear` | Boolean - true if article is about sportswear brands |
| `probability` | Float (0-1) - classifier confidence |
| `risk_level` | "low" (<0.3), "medium" (0.3-0.6), "high" (>=0.6) |
| `threshold` | Classification threshold used (default: 0.605) |

### Retraining the Model

```bash
# Train FP classifier with default settings
uv run python scripts/train.py --classifier fp --verbose

# Train EP classifier
uv run python scripts/train.py --classifier ep --verbose

# Train with custom parameters
uv run python scripts/train.py \
  --classifier fp \
  --data-path data/fp_training_data.jsonl \
  --target-recall 0.98 \
  --output-dir models \
  --verbose

# Retrain and auto-promote if model improves
uv run python scripts/retrain.py --classifier fp --auto-promote
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEWSDATA_API_KEY` | Your NewsData.io API key | Required (for NewsData) |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5434/esg_news` |
| `POSTGRES_USER` | Database username | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `POSTGRES_DB` | Database name | `esg_news` |
| `MAX_API_CALLS_PER_DAY` | API rate limit | `200` |
| `SCRAPE_DELAY_SECONDS` | Delay between scrape requests | `2` |
| `GDELT_TIMESPAN` | Default GDELT time window | `3m` |
| `GDELT_MAX_RECORDS` | Max records per GDELT query | `250` |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude labeling | Required (for labeling) |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required (for labeling) |
| `LABELING_MODEL` | Claude model for labeling | `claude-sonnet-4-20250514` |
| `EMBEDDING_MODEL` | OpenAI model for embeddings | `text-embedding-3-small` |
| `LABELING_BATCH_SIZE` | Default articles per labeling batch | `10` |
| `TARGET_CHUNK_TOKENS` | Target tokens per chunk | `500` |
| `MAX_CHUNK_TOKENS` | Maximum tokens per chunk | `800` |
| `MIN_CHUNK_TOKENS` | Minimum tokens per chunk | `100` |

## Database Schema

### Articles Table

Stores article metadata from API + full scraped content + future embeddings:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `article_id` | String | Unique ID from NewsData.io |
| `title` | Text | Article title |
| `description` | Text | Short description/summary |
| `full_content` | Text | Full scraped article text |
| `url` | String | Article URL |
| `published_at` | DateTime | Publication date |
| `source_name` | String | News source name |
| `brands_mentioned` | Array | Detected brand names |
| `scrape_status` | String | pending/success/failed |
| `embedding` | Vector(1536) | For future semantic search |

### Collection Runs Table

Logs each daily collection run with statistics for monitoring.

### Labeling Tables

The labeling pipeline adds several new tables:

**`article_chunks`** - Chunked article text for embeddings and evidence linking:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `article_id` | UUID | Foreign key to articles |
| `chunk_index` | Integer | Order within article |
| `chunk_text` | Text | Chunk content |
| `char_start`, `char_end` | Integer | Position in full_content |
| `token_count` | Integer | Token count |
| `embedding` | Vector(1536) | OpenAI embedding |

**`brand_labels`** - Per-brand ESG labels with sentiment:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `article_id` | UUID | Foreign key to articles |
| `brand` | String | Brand name (Nike, Adidas, etc.) |
| `environmental`, `social`, `governance`, `digital_transformation` | Boolean | Category flags |
| `environmental_sentiment`, etc. | SmallInt | Sentiment (-1, 0, 1, or NULL) |
| `confidence_score` | Float | LLM confidence (0-1) |
| `labeled_by` | String | Source (claude-sonnet, human, classifier) |
| `model_version` | String | Model identifier |

**`label_evidence`** - Supporting text excerpts:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `brand_label_id` | UUID | Foreign key to brand_labels |
| `chunk_id` | UUID | Foreign key to article_chunks (nullable) |
| `category` | String | ESG category |
| `excerpt` | Text | Evidence quote |
| `relevance_score` | Float | Match confidence |

**`labeling_runs`** - Tracks labeling batches with statistics and cost estimates.

## Querying the Database

Use these commands to check collection progress and database statistics.

### Quick Stats

```bash
# Total articles collected
docker exec esg_news_db psql -U postgres -d esg_news -c "SELECT COUNT(*) as total_articles FROM articles;"

# Articles by scrape status
docker exec esg_news_db psql -U postgres -d esg_news -c "SELECT scrape_status, COUNT(*) FROM articles GROUP BY scrape_status;"

# Articles pending scrape
docker exec esg_news_db psql -U postgres -d esg_news -c "SELECT COUNT(*) as pending FROM articles WHERE scrape_status = 'pending';"

# Successfully scraped articles (have full text)
docker exec esg_news_db psql -U postgres -d esg_news -c "SELECT COUNT(*) as scraped FROM articles WHERE scrape_status = 'success';"
```

### Detailed Queries

```bash
# Recent collection runs with statistics
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT
    started_at::date as date,
    status,
    api_calls_made,
    articles_fetched,
    articles_duplicates,
    articles_scraped,
    articles_scrape_failed
FROM collection_runs
ORDER BY started_at DESC
LIMIT 10;"

# Articles per brand (approximate - checks brands_mentioned array)
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT unnest(brands_mentioned) as brand, COUNT(*)
FROM articles
WHERE brands_mentioned IS NOT NULL
GROUP BY brand
ORDER BY count DESC;"

# Articles collected per day
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT created_at::date as date, COUNT(*) as articles
FROM articles
GROUP BY created_at::date
ORDER BY date DESC;"

# Sample of recent articles
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT LEFT(title, 60) as title, source_name, scrape_status, created_at::date
FROM articles
ORDER BY created_at DESC
LIMIT 10;"
```

### Labeling Queries

```bash
# Labeling statistics
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT labeling_status, COUNT(*)
FROM articles
GROUP BY labeling_status;"

# Brand labels by category
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT brand,
       COUNT(*) as total_labels,
       SUM(CASE WHEN environmental THEN 1 ELSE 0 END) as environmental,
       SUM(CASE WHEN social THEN 1 ELSE 0 END) as social,
       SUM(CASE WHEN governance THEN 1 ELSE 0 END) as governance,
       SUM(CASE WHEN digital_transformation THEN 1 ELSE 0 END) as digital
FROM brand_labels
GROUP BY brand
ORDER BY total_labels DESC;"

# Recent labeling runs
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT started_at::date, status, articles_processed, brands_labeled,
       ROUND(estimated_cost_usd::numeric, 4) as cost
FROM labeling_runs
ORDER BY started_at DESC
LIMIT 5;"

# Evidence excerpts for a brand
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT le.category, LEFT(le.excerpt, 80) as evidence, le.relevance_score
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
WHERE bl.brand = 'Nike'
LIMIT 10;"
```

### Interactive Database Access

```bash
# Open psql shell for interactive queries
docker exec -it esg_news_db psql -U postgres -d esg_news
```

## ESG Category Structure

The classifier will categorize articles into these ESG categories:

**Environmental:**
- `carbon_emissions` - Climate change, greenhouse gases
- `waste_management` - Recycling, waste reduction
- `sustainable_materials` - Eco-friendly materials, renewable resources

**Social:**
- `worker_rights` - Labor practices, fair wages
- `diversity_inclusion` - DEI initiatives, representation
- `community_engagement` - Local community impact, philanthropy

**Governance:**
- `ethical_sourcing` - Supply chain ethics, transparency
- `transparency` - Corporate disclosure, reporting
- `board_structure` - Corporate governance, leadership

**Digital Transformation:**
- `technology_innovation` - AI/ML applications, smart products, wearable tech
- `digital_retail` - E-commerce platforms, omnichannel experiences, direct-to-consumer
- `supply_chain_tech` - Blockchain traceability, inventory optimization, logistics automation

## Testing

The project includes a comprehensive test suite with 190 tests covering data collection and labeling pipelines (72% code coverage).

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_api_client.py

# Run database tests (requires PostgreSQL running)
# Note: Tests use a separate 'esg_news_test' database to protect production data
RUN_DB_TESTS=1 uv run pytest tests/test_database.py
```

**Test Coverage:**

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_api_client.py` | 23 | NewsData.io brand extraction, article parsing, query generation |
| `test_gdelt_client.py` | 31 | GDELT article parsing, query generation, date handling |
| `test_scraper.py` | 19 | Language detection, paywall detection, scraping |
| `test_collector.py` | 13 | Deduplication, dry run mode, API limits |
| `test_database.py` | 12 | Upsert operations, queries (requires PostgreSQL) |
| `test_chunker.py` | 21 | Article chunking, token counting, paragraph boundaries |
| `test_labeler.py` | 33 | LLM response parsing, ArticleLabeler, JSON extraction |
| `test_embedder.py` | 15 | OpenAI embedder, batching, retry logic |
| `test_evidence_matcher.py` | 24 | Evidence matching, fuzzy/exact/embedding similarity |
| `test_labeling_pipeline.py` | 13 | Pipeline orchestration, statistics tracking |

## Troubleshooting

### Port Already in Use

If you see `address already in use` when starting Docker:

```bash
# Check what's using the port
lsof -i :5434

# Or change the port in docker-compose.yml and .env
```

### API Key Issues

```bash
# Test your API key with a minimal dry run
uv run python scripts/collect_news.py --dry-run --max-calls 1 -v
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker ps

# Check logs
docker logs esg_news_db

# Test connection
psql postgresql://postgres:postgres@localhost:5434/esg_news
```
