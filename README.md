# ESG News Classifier for Sportswear Brands

> **ML Zoomcamp Capstone Project** - A production-ready multi-label text classification system that monitors sustainability news in the sportswear and outdoor apparel industry.
>
> **🔗 Live Demo:** [View ESG News Feed](https://frederick-douglas-pearce.github.io/projects/esg_classifier/) - Browse curated ESG articles with interactive filtering by brand and category

---

## 🏆 Sustainability Scorecard

The live ESG News Feed now features a **Sustainability Scorecard** that ranks sportswear brands based on their recent ESG news coverage.

**📊 [View Live Scorecard](https://frederick-douglas-pearce.github.io/projects/esg_classifier/)**

### How It Works

The scorecard analyzes ESG news from the last 14 days and assigns points based on sentiment:

| Sentiment | Points | Example |
|-----------|--------|---------|
| **Positive** | +2 pts | "Nike launches carbon-neutral shoe line" |
| **Neutral** | +1 pt | "Adidas reports quarterly sustainability metrics" |
| **Negative** | -1 pt | "Factory labor violations discovered at supplier" |

### Scorecard Features

- **🥇🥈🥉 Top Performers**: Brands with the highest positive scores earn gold, silver, and bronze medals
- **Back of the Pack**: Brands with negative scores are highlighted for transparency
- **Category Breakdown**: See scores by Environmental (E), Social (S), Governance (G), and Digital Transformation (D)
- **Deduplication**: Similar articles are automatically detected and deduplicated using sentence embeddings (cosine similarity ≥ 0.85)
- **Rolling Window**: Scores update automatically based on the most recent 14-day period
- **Historical Tracking**: Daily snapshots stored in database for trend analysis (see `queries/scorecard_queries.sql`)

### Example Scorecard

```
🏆 Top Performers                    ⚠️ Back of the Pack
├─ 🥇 Patagonia    +12 (E:+6 S:+4 G:+2)    ├─ Brand X    -3 (S:-3)
├─ 🥈 Nike         +8  (E:+4 S:+2 D:+2)    ├─ Brand Y    -2 (G:-2)
└─ 🥉 Adidas       +5  (E:+3 S:+2)         └─ Brand Z    -1 (E:-1)
```

The scorecard provides an at-a-glance view of which brands are leading—or lagging—in sustainability coverage.

---

## Problem Description

### The Challenge

As someone interested in sustainable clothing and the outdoor wear industry, I want to stay informed about ESG (Environmental, Social, Governance) and Digital Transformation (DT) developments from major sportswear and outdoor apparel brands. However, manually tracking sustainability news across 50+ brands is impractical:

- **Information overload**: Thousands of articles mention these brands daily
- **Signal vs. noise**: Most articles are product announcements, not ESG-related
- **False positives**: Brand names like "Puma" (animal), "Patagonia" (region), and "Columbia" (university) appear in unrelated contexts
- **Cost of analysis**: Using LLMs like Claude to classify every article costs ~$15/1000 articles

### The Solution

This project builds an **automated ESG news monitoring pipeline** that:

1. **Collects news** from NewsData.io and GDELT APIs (8x daily automated collection)
2. **Filters false positives** using an ML classifier (Random Forest, F2: 0.974) - articles where brand names refer to non-sportswear entities
3. **Pre-filters ESG content** using a second ML classifier (Logistic Regression, F2: 0.931) - identifies articles with sustainability content
4. **Labels articles** with detailed ESG categories, sentiment, and extracted ESG content chunks using Claude Sonnet (only for articles that pass both filters)
5. **Reduces costs by 20-30%** by using ML classifiers with >99% Recall as pre-filters before expensive LLM calls

### Real-World Application

The classifier powers a **live ESG news feed** on my personal website that tracks sustainability developments in the sportswear/outdoorwear industry:

**📊 [ESG News Feed](https://frederick-douglas-pearce.github.io/projects/esg_classifier/)** - A curated, searchable collection of ESG-related news articles

**Features:**
- **🏆 Sustainability Scorecard**: At-a-glance brand rankings with medal awards for top performers (see [Sustainability Scorecard](#-sustainability-scorecard) section above)
- **Brand Filtering**: Filter articles by any of the 50 monitored sportswear brands
- **ESG Category Filtering**: View articles by Environmental, Social, Governance, or Digital Transformation categories
- **Date Range Filtering**: Quick presets (7/14/30 days) or custom date ranges
- **Evidence Excerpts**: Each article shows relevant ESG quotes extracted from the source text
- **Sentiment Indicators**: Color-coded badges show positive/neutral/negative sentiment for each category
- **Source Links**: Direct links to original news articles

**Export Options:**
- JSON data feed for programmatic access
- Atom/RSS feed for news aggregators

The ML classifiers enable cost-effective continuous monitoring that would be significantly more expensive with LLM-only approaches.

### Autonomous Operations

The entire system runs autonomously with minimal human intervention through a **custom agent orchestrator**:

- **Self-maintaining**: Daily workflows handle news collection, article labeling, quality monitoring, and website updates automatically
- **Intelligent oversight**: Claude Sonnet analyzes labeling results daily to detect errors, identify patterns, and suggest improvements
- **Proactive monitoring**: ML classifier drift detection triggers alerts before model degradation affects production
- **Time-saving**: Eliminates ~5-10 hours/week of manual maintenance (running scripts, reviewing results, updating website)
- **Hands-off deployment**: Model training workflow includes human-in-the-loop notebook review, then automatically promotes and deploys improved models

**Daily operation cost**: ~$0.50-1.00/day (labeling + monitoring), fully automated with email reports.

📖 See [Agent Orchestrator](#agent-orchestrator) section for workflow details.

### Dataset

The training data was collected and labeled specifically for this project:

- **Source**: NewsData.io API + GDELT DOC 2.0 API (free, 3 months history)
- **Collection period**: December 2025 - Present
- **Articles collected**: ~3,000 articles mentioning target brands
- **Labeled articles**: 993 for FP classifier, 870 for EP classifier
- **Labeling method**: Claude Sonnet with structured JSON output + manual review

Training data is exported to JSONL format in `data/` directory using `scripts/export_training_data.py`.

### Target Brands (50)

The system monitors news for the following sportswear and outdoor apparel brands:

| Brand | Brand | Brand | Brand | Brand |
|-------|-------|-------|-------|-------|
| Nike | Adidas | Puma | Under Armour | Lululemon |
| Patagonia | Columbia Sportswear | New Balance | ASICS | Reebok |
| Skechers | Fila | The North Face | Vans | Converse |
| Salomon | Mammut | Umbro | Anta | Li-Ning |
| Brooks Running | Decathlon | Deckers | Yonex | Mizuno |
| K-Swiss | Altra Running | Hoka | Saucony | Merrell |
| Timberland | Spyder | On Running | Allbirds | Gymshark |
| Everlast | Arc'teryx | Jack Wolfskin | Athleta | Vuori |
| Cotopaxi | Prana | Eddie Bauer | 361 Degrees | Xtep |
| Peak Sport | Mountain Hardwear | Black Diamond | Outdoor Voices | Diadora |

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
        predictions[("classifier_predictions<br/><i>audit trail</i>")]
    end

    subgraph prefilter["Phase 2: ML Pre-filters"]
        fp_api["FP Classifier API<br/><i>Cloud Run / Local</i>"]
        ep_api["EP Classifier API<br/><i>Local only (pending deploy)</i>"]
    end

    subgraph labeling["Phase 3: LLM Labeling"]
        chunker["Article Chunker<br/><i>~500 tokens/chunk</i>"]
        embedder["OpenAI Embedder<br/><i>text-embedding-3-small</i>"]
        claude["Claude Sonnet<br/><i>ESG classification</i>"]
    end

    subgraph training["Phase 4: Model Training"]
        export["Export Training Data<br/><i>JSONL format</i>"]
        notebooks["Jupyter Notebooks<br/><i>EDA, tuning, evaluation</i>"]
        train["train.py<br/><i>Pipeline training</i>"]
    end

    subgraph deployment["Phase 5: Deployment"]
        registry["Model Registry<br/><i>registry.json + MLflow</i>"]
        docker["Docker Build<br/><i>Auto-detect dependencies</i>"]
        cloudrun["Google Cloud Run<br/><i>FP API (EP pending)</i>"]
    end

    subgraph monitoring["Phase 6: MLOps Monitoring"]
        pred_logs["Prediction Logging<br/><i>Database + files</i>"]
        drift["Drift Monitor<br/><i>Evidently AI</i>"]
        alerts["Webhook Alerts<br/><i>Slack/Discord</i>"]
    end

    subgraph website["Phase 7: Website Integration"]
        feed_export["Feed Export<br/><i>export_website_feed.py</i>"]
        json_feed["JSON Feed<br/><i>_data/esg_news.json</i>"]
        atom_feed["Atom Feed<br/><i>assets/feeds/esg_news.atom</i>"]
        jekyll["Jekyll Site<br/><i>GitHub Pages</i>"]
    end

    %% Data Collection Flow
    newsdata --> api_client
    gdelt --> api_client
    api_client --> scraper
    scraper --> articles

    %% Pre-filter Flow
    articles --> fp_api
    fp_api -->|"sportswear"| ep_api
    fp_api -->|"false positive"| predictions
    ep_api -->|"has ESG"| chunker
    ep_api -->|"no ESG"| predictions

    %% Labeling Flow
    chunker --> chunks
    chunks --> embedder
    articles --> claude
    claude --> labels

    %% Training Flow
    labels --> export
    predictions --> export
    export --> notebooks
    notebooks --> train
    train --> registry

    %% Deployment Flow
    registry --> docker
    docker --> cloudrun
    cloudrun --> fp_api
    cloudrun --> ep_api

    %% Monitoring Flow
    fp_api --> pred_logs
    ep_api --> pred_logs
    pred_logs --> predictions
    predictions --> drift
    drift -->|"drift detected"| alerts
    drift -->|"retrain signal"| train

    %% Website Flow
    labels --> feed_export
    feed_export --> json_feed
    feed_export --> atom_feed
    json_feed --> jekyll
    atom_feed --> jekyll

    %% Styling for pending deployment
    style ep_api stroke-dasharray: 5 5
```

## Project Roadmap

### Phase 1: Data Collection ✅
- [x] NewsData.io API integration
- [x] GDELT DOC 2.0 API integration (free, 3 months history)
- [x] Article scraping with newspaper4k
- [x] PostgreSQL + pgvector storage
- [x] Automated cron scheduling (8x daily: 4 NewsData + 4 GDELT)
- [x] Historical backfill script for GDELT
- [x] Domain blacklist for filtering low-quality sources (press releases, spam)
- [x] Target: 1,000-2,000 articles over 10-14 days

### Phase 2: Data Labeling ✅
- [x] LLM-based labeling pipeline with Claude Sonnet
- [x] Per-brand ESG category labels with ternary sentiment
- [x] Article chunking for evidence extraction
- [x] OpenAI embeddings for semantic evidence matching
- [x] Cross-encoder reranking for improved evidence quality
- [x] Evidence linking to source text chunks
- [x] Labeling CLI with dry-run and batch support

### Phase 3: Model Development ✅
- [x] Export labeled data for training (JSONL format for 3 classifier types)
- [x] False positive brand detection and cleanup tools
- [x] False Positive Classifier - 3-notebook pipeline complete
  - fp1: EDA + TF-IDF+LSA/NER/Brand/Proximity feature engineering selection w/ hyperparameter tuning
  - fp2: Model selection + hyperparameter tuning (3-fold CV)
  - fp3: Test evaluation + threshold optimization + feature importance + deployment export
  - Random Forest achieves Test F2: 0.974, Recall: 98.8%
  - Supporting modules in `src/fp1_nb/`, `src/fp2_nb/`, `src/fp3_nb/`
- [x] ESG Pre-filter Classifier - 3-notebook pipeline complete
  - ep1: EDA + TF-IDF/LSA feature engineering with ESG vocabularies
  - ep2: Model selection + hyperparameter tuning (3-fold CV)
  - ep3: Test evaluation + threshold optimization + deployment export
  - Logistic Regression achieves Test F2: 0.931, Recall: 100%
  - Supporting modules in `src/ep1_nb/`, `src/ep2_nb/`, `src/ep3_nb/`
- [ ] ESG Multi-label Classifier: Category classification with sentiment (future)
- [ ] Advanced: Fine-tuned DistilBERT/RoBERTa (future)

### Phase 4: Evaluation & Explainability ✅
- [x] Per-classifier Precision, Recall, PR-AUC, F2 scores
- [x] Threshold optimization for target recall (99% FP, 99% EP)
- [x] SHAP feature group importance analysis
- [x] LIME local explanations for individual predictions
- [x] Prototype-based explanations (similar training examples)
- [x] Explainability module (`src/fp3_nb/explainability.py`)

### Phase 5: Deployment (Current)
- [x] Unified FastAPI REST API (`scripts/predict.py`)
- [x] Unified training script (`scripts/train.py`)
- [x] Deployment module (`src/deployment/`)
- [x] Multi-stage Dockerfile with auto-dependency detection
- [x] Docker Compose integration
- [x] GitHub Actions CI/CD to Google Cloud Run
- [x] Model registry with version tracking (`models/registry.json`)
- [x] Prediction logging to database for drift monitoring
- [x] FP classifier deployed and integrated into labeling pipeline
- [ ] EP classifier deployment and labeling pipeline integration (future)

### Phase 6: MLOps Monitoring ✅
- [x] MLflow experiment tracking (hyperparameters, metrics, artifacts)
- [x] Evidently AI drift detection with HTML reports
- [x] Reference dataset management for drift comparison
- [x] Automated daily drift monitoring (cron + GitHub Actions)
- [x] Webhook alerts for Slack/Discord
- [x] Retraining pipeline with semantic versioning (`scripts/retrain.py`)
- [x] Model promotion workflow with auto-promote option

### Phase 7: Website Integration ✅
- [x] Export script for JSON and Atom feeds (`scripts/export_website_feed.py`)
- [x] Live ESG news feed on personal website
- [x] Client-side filtering by brand, ESG category, sentiment, and date range
- [x] Evidence excerpts with sentiment indicators
- [x] RSS/Atom feed for news aggregators
- [x] **Sustainability Scorecard** with brand rankings and medal awards
- [x] Article deduplication using sentence embeddings (cosine similarity ≥ 0.85)

### Phase 8: Agent Orchestrator ✅
- [x] Hybrid orchestrator for automated maintenance workflows
- [x] Daily labeling workflow (collection check → labeling → quality metrics → LLM analysis → reports)
- [x] Drift monitoring workflow (FP/EP classifier drift detection with alerts)
- [x] Website export workflow (JSON/Atom feed generation with git integration)
- [x] Model training workflow (data export → quality check → notify → compare → promote → deploy)
- [x] LLM intelligence: Claude Sonnet analysis of labeling results for error detection
- [x] GitHub Actions deployment trigger after model promotion
- [x] Unified notification system (Resend email + Slack/Discord webhooks)
- [x] YAML-based state management with checkpointing
- [x] Cron scheduling (5:30am drift, 6:30am labeling, 7:00am export)
- [x] CLI: `uv run python -m src.agent run|continue|status|list|history`

## Table of Contents

- [Sustainability Scorecard](#-sustainability-scorecard)
- [System Architecture](#system-architecture)
- [Project Roadmap](#project-roadmap)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [News Collection](#news-collection)
- [AI-Based Article Labeling](#ai-based-article-labeling)
  - [ML Classifier Notebooks](#ml-classifier-notebooks)
- [Classifier Deployment](#classifier-deployment)
- [MLOps](#mlops)
- [Agent Orchestrator](#agent-orchestrator)
  - [Claude Code Skills](#claude-code-skills)
- [Database](#database)
- [ESG Category Structure](#esg-category-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Project Structure

The project follows a modular architecture:

| Directory | Purpose |
|-----------|---------|
| `src/` | Core modules: data_collection, labeling, mlops, agent, notebook utilities |
| `scripts/` | CLI tools: collection, labeling, training, deployment, monitoring |
| `notebooks/` | ML classifier development: 6 notebooks (EDA → Tuning → Deployment) |
| `models/` | Trained models, configs, and version registry |
| `tests/` | Comprehensive test suite (818 tests) |
| `docs/` | Detailed documentation for each subsystem |

📁 See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for complete file listing.

## Quick Start

### 1. Prerequisites

**Required for all users:**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

**For ML Zoomcamp Reviewers:**

See the [**ML Zoomcamp Reviewer Guide**](#ml-zoomcamp-reviewer-guide) for step-by-step evaluation instructions covering:
- Notebook review (fp1, fp2, fp3)
- CLI training with `scripts/train.py`
- Local and Docker deployment
- Cloud Run deployment verification

**Quick start for notebooks only:**
```bash
uv sync
uv run jupyter lab notebooks/
```

Training data is included: `data/fp_training_data.jsonl` (no API keys needed)

**API Keys by Feature:**

| Feature | API Key | Where to Get |
|---------|---------|--------------|
| News Collection (NewsData) | `NEWSDATA_API_KEY` | [newsdata.io/register](https://newsdata.io/register) (free tier) |
| News Collection (GDELT) | *None required* | Free, no registration |
| Article Labeling | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| Embeddings | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

**For Cloud Deployment (Optional):**

See [`.github/DEPLOYMENT_SETUP.md`](.github/DEPLOYMENT_SETUP.md) for Google Cloud Run deployment requirements including:
- `GCP_PROJECT_ID` - Google Cloud project ID
- `GCP_SA_KEY` - Service account JSON key with Cloud Run Admin role
- `GCP_REGION` - Deployment region (optional, defaults to us-central1)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier.git
cd sportswear-esg-news-classifier

# Install dependencies with uv
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev

# Create environment file from template
cp .env.example .env

# Edit .env and add your API keys
# Required keys depend on which features you want to use (see table above)
```

### 3. Start the Database

*This step is needed for the labeling pipeline. Skip it if you only want to train and deploy the FP classifier.*

```bash
# Start PostgreSQL with pgvector extension
docker compose up -d

# Verify it's running
docker ps
```

The database will be available at `localhost:5434`.

## News Collection

The pipeline collects ESG-related news articles from two sources:
- **NewsData.io** - Paid API with real-time news (requires API key)
- **GDELT DOC 2.0** - Free API with 3 months of historical data

**Data Quality Features:**
- **Domain Blacklist**: Low-quality domains (e.g., press release sites) are automatically filtered during collection
- **Language Detection**: Only English articles are processed
- **Deduplication**: Articles with identical URLs or similar content are deduplicated

```bash
# Quick test (dry run)
uv run python scripts/collect_news.py --source gdelt --dry-run --max-calls 5

# Production collection
uv run python scripts/collect_news.py --source gdelt

# Set up automated collection (8x daily)
./scripts/setup_cron.sh install
```

📖 See [docs/COLLECTION.md](docs/COLLECTION.md) for full CLI options, cron setup, and backfill procedures.

## AI-Based Article Labeling

The project uses a **hybrid LLM + ML approach** for article classification:

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

**How it works:**
1. **LLM Labeling (Claude Sonnet)**: High-quality labeling into ESG categories with sentiment. Generates training data for ML classifiers (~$15/1000 articles).
2. **ML Classifiers**: Cost-efficient models filter articles before LLM labeling, reducing costs by 20-30%.

**ML Classifiers Developed:**
| Classifier | Purpose | Performance |
|------------|---------|-------------|
| **FP (False Positive)** ✅ | Filter non-sportswear brand mentions | F2: 0.974, Recall: 98.8% |
| **EP (ESG Pre-filter)** ✅ | Identify articles with ESG content | F2: 0.931, Recall: 100% |
| **ESG Multi-label** | Full category + sentiment classification | Planned |

📖 See [docs/LABELING.md](docs/LABELING.md) for LLM pipeline details, CLI options, and training data export.

📖 See [docs/PROMPT_VERSIONING.md](docs/PROMPT_VERSIONING.md) for versioned prompt management (tracking prompt templates used for each labeling run).

### ML Classifier Notebooks

The project includes 3-notebook pipelines for developing ML classifiers. Each pipeline follows the same structure with supporting utility modules for consistent preprocessing and evaluation. Moving much of the python code to modules leads to cleaner, easy to follow notebooks, with much of the code covered by test cases.

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

- **Data Loading**: News articles with sportswear and false positives without sportswear
- **EDA**: Text length distributions, brand distribution, word frequencies
- **Feature Engineering**: Sentence-transformer embeddings + NER brand context features
- **Hyperparameter Tuning**: Optimizes `proximity_window_size` for NER features
- **Exports**: Feature transformer and configuration for fp2

#### fp2_model_selection_tuning.ipynb - Model Selection & Tuning

- **Baseline Models**: LR, RF, HGB with 3-fold stratified CV
- **Hyperparameter Tuning**: GridSearchCV optimizing F2 score
- **Overfitting Analysis**: Train-validation gap visualization
- **Best Model**: Random Forest with `balanced` class weights
- **Exports**: Best classifier and CV metrics for fp3

#### fp3_model_evaluation_deployment.ipynb - Test Evaluation & Deployment

- **Test Evaluation**: Final held-out test set evaluation (ONLY notebook using test data)
- **Threshold Optimization**: Find optimal threshold for 98% target recall
- **Pipeline Export**: Complete sklearn Pipeline for deployment

**Performance (Random Forest):**
- CV F2: 0.973, Test F2: 0.974
- Test Recall: 98.8%, Test Precision: 91.9%
- Optimized threshold: 0.605 (at 98% recall)

**Overfitting Analysis Note:**

The Random Forest model often shows perfect training F2 values of 1.0, while validation F2 ≈ 0.973. This pattern (perfect training scores) is expected with small datasets and Random Forest's default behavior:

1. **Why train = 1.0**: With only ~600 samples per CV training fold and 200+ features, individual trees can perfectly memorize training data when `min_samples_leaf=1`.

2. **Why this is acceptable**: Despite individual tree overfitting, ensemble averaging reduces variance. The key evidence:
   - Train-val gap is <2% (below the 5% warning threshold)
   - CV-to-test gap is negligible (<0.5%), confirming excellent generalization
   - Validation F2 continues improving with more trees (ensemble benefit)

3. **Generalization confirmed**: Test set performance is consistently less than 0.5% different than CV performance, indicating the model generalizes well to unseen data despite the high training scores (see the fp3 or ep3 notebooks for details)

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

📖 See [docs/TEXT_FEATURES.md](docs/TEXT_FEATURES.md) for detailed explanation of NLP feature extraction methods (TF-IDF, LSA, NER, sentence embeddings).

## Classifier Deployment

The FP and EP classifiers are deployed as FastAPI REST APIs, integrated with the labeling pipeline as optional pre-filters to reduce LLM costs.

**Quick Start:**

```bash
# Local deployment
CLASSIFIER_TYPE=fp uv run python scripts/predict.py

# Docker deployment
docker compose up -d fp-classifier-api

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Nike sustainability report", "content": "...", "brands": ["Nike"]}'
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata and metrics |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Batch classification |

**Training & Retraining:**

```bash
# Train classifier
uv run python scripts/train.py --classifier fp --verbose

# Retrain with auto-promote
uv run python scripts/retrain.py --classifier fp --auto-promote
```

📖 See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment guide including Docker, Cloud Run, model registry, semantic versioning, and daily retraining workflows.

## MLOps

The project includes optional MLOps features for experiment tracking and production monitoring. All features use **graceful degradation** - they work when disabled with no code changes required.

| Feature | Tool | Purpose |
|---------|------|---------|
| Experiment Tracking | MLflow | Log hyperparameters, metrics, and model artifacts |
| Drift Monitoring | Evidently AI | Detect prediction distribution shifts |
| Automated Alerts | Webhooks | Slack/Discord notifications for drift |

```bash
# Enable MLflow tracking
MLFLOW_ENABLED=true uv run python scripts/train.py --classifier fp

# View MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

# Monitor for drift (from production database)
uv run python scripts/monitor_drift.py --classifier fp --from-db

# Set up daily monitoring
./scripts/setup_cron.sh install-monitor
```

📖 See [docs/MLOPS.md](docs/MLOPS.md) for detailed setup, configuration options, and programmatic usage.

## Agent Orchestrator

The project includes a custom-built agent orchestrator that automates daily operations, reducing manual maintenance to near-zero while ensuring data quality and system health.

### Why Build a Custom Agent?

Off-the-shelf workflow tools (Airflow, Prefect, Dagster) are powerful but add significant operational overhead for a single-developer project. The custom agent provides:

- **Lightweight**: Single Python module, no external services required
- **YAML state management**: Human-readable workflow state and history
- **LLM intelligence**: Claude Sonnet analyzes labeling results for quality assurance
- **Unified notifications**: Email (Resend) + webhooks (Slack/Discord) in one interface
- **Checkpointing**: Workflows can pause for human review and resume

### Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `daily_labeling` | 6:30 AM | Process pending articles through ML + LLM pipeline, generate quality reports |
| `drift_monitoring` | 5:30 AM | Check FP/EP classifier drift, alert if degradation detected |
| `website_export` | 7:00 AM | Export labeled articles to Jekyll site, commit and push |
| `model_training` | Manual | Export data → run notebooks → compare → promote → deploy |

### Quick Start

```bash
# List available workflows
uv run python -m src.agent list

# Run a workflow manually
uv run python -m src.agent run daily_labeling

# Dry run (no side effects)
uv run python -m src.agent run daily_labeling --dry-run

# Check workflow status
uv run python -m src.agent status

# Install cron jobs for automated operation
./scripts/setup_cron.sh install-agent
```

### Daily Operation Flow

```
5:30 AM  drift_monitoring   → Check classifier health, alert on drift
6:30 AM  daily_labeling     → Label articles, quality check, email report
7:00 AM  website_export     → Update live feed, push to GitHub Pages
```

Each workflow generates detailed logs in `logs/agent/` and archives state to `~/.esg-agent/history/`.

📖 See [docs/AGENT.md](docs/AGENT.md) for workflow details, configuration options, and architecture.

### Claude Code Skills

The project includes custom [Claude Code](https://claude.ai/code) skills for streamlined development workflows. Skills are invoked with `/skillname` in a Claude Code session.

| Skill | Purpose |
|-------|---------|
| `/esg-status` | Quick project dashboard: collection stats, labeling progress, recent runs |
| `/review-labels` | Review recent labeling for errors: sentiment breakdown, negative articles, false positives |

**Usage:**

```bash
# Start Claude Code in the project directory
cd sportswear-esg-news-classifier
claude

# Then invoke skills
/esg-status       # Shows collection stats, labeling breakdown, recent runs with costs
/review-labels    # Reviews articles labeled since last review, flags potential errors
```

**`/esg-status` output example:**
```
COLLECTION (Last 7 days)
  Runs: 56, Fetched: 171, Scraped: 148, Failed: 23

LABELING STATUS
  labeled             293 (11.4%)
  false_positive      417 (16.3%)
  skipped            1248 (48.6%)
  pending              19 (0.7%)
  unlabelable         571 (22.3%)

RECENT LABELING RUNS
  2026-01-29: 1 runs, 22 articles, 6 brands, $0.53
```

**`/review-labels` features:**
- Tracks last run date to avoid reviewing the same articles twice
- Shows brand sentiment breakdown (E/S/G positive/negative counts)
- Lists articles with negative sentiment for accuracy review
- Spot-checks recent false positives for missed ESG content
- Highlights multi-brand articles for consistency checking

Skill definitions are in `.claude/skills/`.

## Database

PostgreSQL with pgvector stores articles, labels, embeddings, and classifier predictions.

| Table | Purpose |
|-------|---------|
| `articles` | News metadata, full content, labeling status |
| `brand_labels` | Per-brand ESG classifications with sentiment |
| `article_chunks` | Text chunks with embeddings for evidence matching |
| `classifier_predictions` | ML classifier audit trail |
| `scorecard_snapshots` | Daily scorecard metadata for historical tracking |
| `scorecard_brand_scores` | Per-brand scores per snapshot (category scores, rank, medals) |

### Quick Commands

```bash
# Check labeling status
docker exec esg_news_db psql -U postgres -d esg_news -c \
  "SELECT labeling_status, COUNT(*) FROM articles GROUP BY labeling_status;"

# Interactive access
docker exec -it esg_news_db psql -U postgres -d esg_news

# Create backup
./scripts/backup_db.sh backup

# Set up daily backups
./scripts/setup_cron.sh install-backup
```

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5434/esg_news` |
| `NEWSDATA_API_KEY` | NewsData.io API key | Required (for NewsData) |
| `ANTHROPIC_API_KEY` | Claude API key | Required (for labeling) |
| `OPENAI_API_KEY` | OpenAI API key | Required (for embeddings) |
| `FP_CLASSIFIER_ENABLED` | Enable FP pre-filter | `false` |

📖 See [docs/DATABASE.md](docs/DATABASE.md) for full schema, queries, backup procedures, and all environment variables.

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

The project includes a comprehensive test suite with **818 tests** covering data collection, labeling pipelines, ML deployment, retraining workflows, MLOps modules, and agent orchestration.

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

**Test Coverage by Module:**

| Module | Coverage | Description |
|--------|----------|-------------|
| `src/deployment/` | 83-100% | Classifier deployment, config, preprocessing |
| `src/mlops/` | 56-91% | MLflow tracking, Evidently monitoring, reference data |
| `src/fp3_nb/explainability.py` | 90% | LIME, SHAP, prototype explanations |
| `src/labeling/` | 69-100% | LLM labeling pipeline |
| `src/data_collection/` | 76-95% | API clients, scraper, collector |

**Test Categories:**

| Category | Description |
|----------|-------------|
| Data Collection | NewsData.io client, GDELT client, scraper, collector, database |
| Labeling Pipeline | Chunker, labeler, embedder, evidence matcher, pipeline orchestration |
| FP Classifier Pre-filter | Classifier client integration, batch processing |
| Notebook Utilities | FP/EP data utils, modeling, overfitting analysis, threshold tuning |
| Deployment | FP/EP classifiers, config, preprocessing, prediction |
| Explainability | SHAP feature groups, LIME local explanations, prototypes |
| MLOps | MLflow tracking, Evidently monitoring, reference data, alerts |
| Retraining | Version management, auto-promotion, deployment triggers |
| Agent Orchestrator | State management, workflows, notifications, drift monitoring |
| Integration | End-to-end classifier pipeline tests |

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
