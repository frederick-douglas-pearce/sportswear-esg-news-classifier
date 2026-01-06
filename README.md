# ESG News Classifier for Sportswear Brands

> **ML Zoomcamp Capstone Project** - A production-ready multi-label text classification system that monitors sustainability news in the sportswear and outdoor apparel industry.
>
> **🔗 Live Demo:** [View ESG News Feed](https://frederick-douglas-pearce.github.io/projects/esg_classifier/) - Browse curated ESG articles with interactive filtering by brand and category

---

**For ML Zoomcamp Reviewers:** This project includes extensive infrastructure for data collection, LLM-based labeling, and MLOps monitoring to support a production news classification system. **For course evaluation, please focus on the FP (False Positive) classifier** - the core ML component. Jump to the **[ML Zoomcamp Reviewer Guide](#ml-zoomcamp-reviewer-guide)** for a streamlined 5-step evaluation covering notebooks, training, and deployment.

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
- **Brand Filtering**: Filter articles by any of the 50 monitored sportswear brands
- **ESG Category Filtering**: View articles by Environmental, Social, Governance, or Digital Transformation categories
- **Evidence Excerpts**: Each article shows relevant ESG quotes extracted from the source text
- **Sentiment Indicators**: Color-coded badges show positive/neutral/negative sentiment for each category
- **Source Links**: Direct links to original news articles

**Export Options:**
- JSON data feed for programmatic access
- Atom/RSS feed for news aggregators

The ML classifiers enable cost-effective continuous monitoring that would be significantly more expensive with LLM-only approaches.

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
- [x] Target: 1,000-2,000 articles over 10-14 days

### Phase 2: Data Labeling ✅
- [x] LLM-based labeling pipeline with Claude Sonnet
- [x] Per-brand ESG category labels with ternary sentiment
- [x] Article chunking for evidence extraction
- [x] OpenAI embeddings for semantic evidence matching
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
- [x] Client-side filtering by brand and ESG category
- [x] Evidence excerpts with sentiment indicators
- [x] RSS/Atom feed for news aggregators

## Table of Contents

- [System Architecture](#system-architecture)
- [Project Roadmap](#project-roadmap)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [ML Zoomcamp Reviewer Guide](#ml-zoomcamp-reviewer-guide) ⭐
- [News Collection](#news-collection)
- [AI-Based Article Labeling](#ai-based-article-labeling)
  - [ML Classifier Notebooks](#ml-classifier-notebooks) ⭐
  - [Text Feature Extraction Methods](#text-feature-extraction-methods)
- [FP Classifier Deployment](#fp-classifier-deployment) ⭐
- [Model Deployment Workflow](#model-deployment-workflow) ⭐
- [MLOps](#mlops)
- [Database](#database)
- [ESG Category Structure](#esg-category-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

> ⭐ = Key sections for ML Zoomcamp evaluation

## Project Structure

The project follows a modular architecture:

| Directory | Purpose |
|-----------|---------|
| `src/` | Core modules: data_collection, labeling, mlops, notebook utilities |
| `scripts/` | CLI tools: collection, labeling, training, deployment, monitoring |
| `notebooks/` | ML classifier development: 6 notebooks (EDA → Tuning → Deployment) |
| `models/` | Trained models, configs, and version registry |
| `tests/` | Comprehensive test suite (548 tests) |
| `docs/` | Detailed documentation for each subsystem |

**Key Files for ML Zoomcamp:**
- `notebooks/fp*.ipynb` - False Positive classifier pipeline
- `notebooks/ep*.ipynb` - ESG Pre-filter classifier pipeline
- `scripts/train.py` - CLI training script
- `scripts/predict.py` - FastAPI prediction service
- `Dockerfile` - Container deployment

📁 See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for complete file listing.

## Quick Start

### 1. Prerequisites

**Required for all users:**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

**API Keys by Feature:**

| Feature | API Key | Where to Get |
|---------|---------|--------------|
| News Collection (NewsData) | `NEWSDATA_API_KEY` | [newsdata.io/register](https://newsdata.io/register) (free tier) |
| News Collection (GDELT) | *None required* | Free, no registration |
| Article Labeling | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |
| Embeddings | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

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

*Skip this step if you only want to run the ML notebooks.*

```bash
# Start PostgreSQL with pgvector extension
docker compose up -d

# Verify it's running
docker ps
```

The database will be available at `localhost:5434`.

## ML Zoomcamp Reviewer Guide

This section provides a step-by-step guide for ML Zoomcamp reviewers to evaluate the FP (False Positive) classifier, which is the primary ML component of this project.

**What you'll evaluate:**
1. Training data and exploratory analysis (notebooks)
2. Model training pipeline (CLI script)
3. Local API deployment (FastAPI)
4. Containerized deployment (Docker)
5. Cloud deployment (Google Cloud Run)

> **Note:** This project uses text-based feature engineering techniques (TF-IDF, sentence embeddings, named entity recognition) that were not covered in the ML Zoomcamp curriculum. For a high-level explanation of these NLP methods, see [Text Feature Extraction Methods](#text-feature-extraction-methods).

### Step 1: Run the Notebooks

The FP classifier is developed through a 3-notebook pipeline. Training data is included in the repository.

```bash
# Clone and setup
git clone https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier.git
cd sportswear-esg-news-classifier
uv sync

# Launch Jupyter, paste link in browser
uv run jupyter lab notebooks/
```

**Notebooks to review (in order):**

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `fp1_EDA_FE.ipynb` | EDA & Feature Engineering | Feature transformer, hyperparameter tuning |
| `fp2_model_selection_tuning.ipynb` | Model Selection & Tuning | Best model (Random Forest), CV metrics |
| `fp3_model_evaluation_deployment.ipynb` | Test Evaluation & Deployment | Threshold optimization, pipeline export |

**Training data:** `data/fp_training_data.jsonl` (1,340 articles: 1,089 sportswear, 251 false positives)

**Key metrics to look for:**
- CV F2 Score: ~0.98
- Test F2 Score: ~0.97
- Test Recall: ~99% (optimized for high recall to minimize false negatives)

### Step 2: Train with CLI

After reviewing the notebooks, train the model using the CLI script:

```bash
# Train FP classifier (uses config exported from fp2 notebook)
uv run python scripts/train.py --classifier fp --verbose

# Expected output:
# ============================================================
# FP CLASSIFIER TRAINING
# ============================================================
# [1/7] Loading training data...
# [2/7] Creating text features...
# [3/7] Splitting data...
# [4/7] Fitting feature transformer...
# [5/7] Training RandomForest classifier...
# [6/7] Evaluating on test set...
# [7/7] Optimizing threshold for 99% recall...
# ============================================================
# TRAINING COMPLETE
# ============================================================
```

**Outputs:**
- `models/fp_classifier_pipeline.joblib` - Trained sklearn pipeline
- `models/fp_classifier_config.json` - Model configuration and metrics

### Step 3: Local API Deployment

Deploy the trained model as a FastAPI service:

```bash
# Start the FP classifier API
CLASSIFIER_TYPE=fp uv run python scripts/predict.py

# Expected output:
# Starting FP classifier API on port 8000
# Loaded FP classifier
# Model: RF_tuned
# Threshold: 0.5000
```

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true,"classifier_type":"fp"}

# Model info
curl http://localhost:8000/model/info

# Test prediction (sportswear article - should return is_sportswear: true)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Nike announces new sustainability initiative",
    "content": "The athletic footwear giant unveiled plans to reduce carbon emissions.",
    "brands": ["Nike"]
  }'

# Test prediction (false positive - should return is_sportswear: false)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Puma spotted in California mountains",
    "content": "Wildlife officials confirmed a mountain lion sighting near hiking trails.",
    "brands": ["Puma"]
  }'
```

**API Documentation:** http://localhost:8000/docs (Swagger UI)

### Step 4: Docker Deployment

Deploy the classifier using Docker Compose:

```bash
# Build and start the FP classifier container
docker compose build fp-classifier-api
docker compose up -d fp-classifier-api

# Check container status
docker ps

# Test health endpoint
curl http://localhost:8000/health

# View logs
docker logs fp-classifier-api

# Stop the container
docker compose down fp-classifier-api
```

**Docker implementation details:**
- Multi-stage build for minimal image size
- Auto-detects dependencies from model config (sentence-transformers, spaCy, etc.)
- Health check endpoint for container orchestration
- See `Dockerfile` and `docker-compose.yml` for configuration

### Step 5: Cloud Deployment

The FP classifier is deployed to **Google Cloud Run** for production use.

**Deployment Architecture:**

```
GitHub Actions (CI/CD)
         │
         ▼
┌─────────────────────────────┐
│   Google Cloud Run          │
│   ├── fp-classifier-api    │
│   │   ├── /health           │
│   │   ├── /model/info       │
│   │   ├── /predict          │
│   │   └── /predict/batch    │
│   └── (2GB memory, 300s timeout)
└─────────────────────────────┘
```

**Live API:** The deployed API URL is available upon request (not committed to repo for security).

**Deployment Screenshots:**

*Screenshots demonstrating the deployed Cloud Run API:*

<details>
<summary>Click to view Cloud Run deployment screenshots</summary>

**Health Check Response:**

![Health Check](images/gcr_health_check.png)

**Swagger UI Documentation:**

![API Docs](images/gcr_swagger_ui.png)

**Example Prediction:**

![Prediction](images/gcr_prediction_example.png)

</details>

> **Note:** If screenshots are not visible, the live API URL can be provided upon request for direct verification.

**Deployment Documentation:**

For detailed CI/CD setup instructions, see [`.github/DEPLOYMENT_SETUP.md`](.github/DEPLOYMENT_SETUP.md), which covers:
- Required GitHub Secrets (`GCP_PROJECT_ID`, `GCP_SA_KEY`)
- Service account creation with Cloud Run Admin role
- Deployment triggers and manual workflow dispatch
- Verifying deployment with `gcloud` CLI

**GitHub Actions Workflow:**

The deployment is automated via `.github/workflows/deploy.yml`:
- Triggers on push to main when model files change
- Builds Docker image with classifier dependencies
- Deploys to Cloud Run with appropriate memory/timeout settings

### Reviewer Checklist

Use this checklist to verify all evaluation criteria:

- [ ] **Notebooks**: fp1, fp2, fp3 notebooks run without errors
- [ ] **Training data**: `data/fp_training_data.jsonl` loads correctly
- [ ] **CLI training**: `scripts/train.py` completes successfully
- [ ] **Local API**: FastAPI starts and responds to `/health`
- [ ] **Predictions**: API returns correct predictions for test cases
- [ ] **Docker**: Container builds and runs with `docker compose`
- [ ] **Cloud Run**: Screenshots show deployed API (or request live URL)

## News Collection

The pipeline collects ESG-related news articles from two sources:
- **NewsData.io** - Paid API with real-time news (requires API key)
- **GDELT DOC 2.0** - Free API with 3 months of historical data

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

### Text Feature Extraction Methods

This section provides a high-level overview of the natural language processing (NLP) techniques used to convert raw text into numerical features that machine learning models can process. These methods are fundamental to text classification but may not be covered in standard ML courses.

#### The Challenge: Text → Numbers

Machine learning models work with numerical data, but news articles are text. The feature extraction pipeline transforms text into meaningful numerical representations while preserving semantic information about brands, ESG topics, and context.

```
Raw Text ──► Preprocessing ──► Feature Extraction ──► Numerical Matrix ──► ML Model
              (cleaning)         (TF-IDF, NER,         (X matrix)          (classify)
                                  embeddings)
```

#### TF-IDF (Term Frequency-Inverse Document Frequency)

**What it does:** Converts text into a sparse matrix where each column represents a word/phrase and each value indicates how important that term is to the document.

**How it works:**
- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: Penalizes common words that appear everywhere (e.g., "the", "and")
- **TF-IDF = TF × IDF**: Words that are frequent in a document but rare overall get higher scores

**Example:**
```
Document: "Nike releases sustainability report on carbon emissions"

TF-IDF scores (simplified):
  "sustainability": 0.45  (rare, domain-specific → high score)
  "nike":           0.38  (brand name, moderately rare)
  "releases":       0.12  (common action verb → lower score)
  "the":            0.00  (stop word, filtered out)
```

**In this project:** Used in the EP (ESG Pre-filter) classifier with n-grams (1-2 word phrases) and vocabulary limited to top 10,000 terms.

#### LSA (Latent Semantic Analysis) / Truncated SVD

**What it does:** Reduces the high-dimensional TF-IDF matrix to a smaller set of "topics" that capture semantic relationships between words.

**Why it's needed:** A TF-IDF matrix with 10,000 words creates 10,000 features—too many for efficient training. LSA compresses this to ~100-200 components while preserving the most important patterns.

**How it works:**
- Uses Singular Value Decomposition (SVD) to find latent "concepts"
- Words that co-occur in similar contexts are grouped together
- Reduces dimensionality while capturing semantic similarity

**Example:**
```
Original TF-IDF: 10,000 features (one per word)
After LSA (200 components): 200 features (latent topics)

Component 47 might capture: "carbon" + "emissions" + "climate" + "footprint"
Component 103 might capture: "workers" + "factory" + "wages" + "labor"
```

**In this project:** The EP classifier uses LSA with 200 components to reduce TF-IDF features while retaining ESG-relevant semantic patterns.

#### NER (Named Entity Recognition)

**What it does:** Identifies and classifies named entities in text—people, organizations, locations, etc.—using a pre-trained language model (spaCy).

**Why it's useful for this project:** Helps distinguish between brand name contexts:
- "Puma launches new shoe" → ORG (organization) = sportswear brand ✓
- "A puma was spotted in the mountains" → No ORG = animal, not the brand ✗

**Features extracted:**
- **Brand-as-ORG**: Is the brand name recognized as an organization entity?
- **Proximity features**: How close is the brand mention to other organization entities?
- **Context features**: What other entities appear near the brand name?

**Example:**
```
Text: "Nike CEO John Donahoe announced partnerships with Adidas and Puma"

NER output:
  Nike     → ORG (organization)
  John Donahoe → PERSON
  Adidas   → ORG
  Puma     → ORG

Features: brand_as_org=True, nearby_orgs=2, has_person=True
```

**In this project:** The FP (False Positive) classifier uses NER features to detect whether brand names appear in sportswear-related contexts.

#### Sentence Embeddings (Sentence-Transformers)

**What it does:** Converts entire sentences or documents into dense numerical vectors (typically 384-768 dimensions) that capture semantic meaning.

**How it differs from TF-IDF:**
- TF-IDF: Sparse, high-dimensional, based on word counts
- Embeddings: Dense, lower-dimensional, based on meaning

**Key advantage:** Semantically similar texts have similar embeddings, even if they use different words.

**Example:**
```
Text A: "Nike reduces carbon emissions by 30%"
Text B: "Athletic footwear company cuts greenhouse gases"

TF-IDF similarity: LOW (few shared words)
Embedding similarity: HIGH (same meaning)
```

**Models used:**
- `all-MiniLM-L6-v2`: Fast, 384-dimensional embeddings (used in FP classifier)
- `text-embedding-3-small`: OpenAI model, 1536 dimensions (used for evidence matching)

**In this project:** The FP classifier concatenates sentence embeddings with NER features for richer text representation.

#### Feature Combination Strategy

The classifiers combine multiple feature types for robust performance:

**FP Classifier (False Positive Detection):**
```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Article title + content + brand name                     │
├─────────────────────────────────────────────────────────────────┤
│ Feature Groups:                                                  │
│ ├── Sentence Embeddings (384 dim) - semantic meaning            │
│ ├── NER Context (12 features) - entity recognition              │
│ ├── Proximity Features (6 features) - brand/entity relationships│
│ └── FP Indicators (13 features) - domain-specific patterns      │
│                                                                  │
│ Total: ~415 features → Random Forest → is_sportswear (0/1)     │
└─────────────────────────────────────────────────────────────────┘
```

**EP Classifier (ESG Pre-filter):**
```
┌─────────────────────────────────────────────────────────────────┐
│ Input: Article title + content + metadata                        │
├─────────────────────────────────────────────────────────────────┤
│ Feature Groups:                                                  │
│ ├── TF-IDF + LSA (200 dim) - document topics                    │
│ ├── ESG Vocabulary Counts (4 features) - category keywords      │
│ └── Metadata Features (varies) - source, category               │
│                                                                  │
│ Total: ~210 features → Logistic Regression → has_esg (0/1)     │
└─────────────────────────────────────────────────────────────────┘
```

#### Why These Methods?

| Method | Strength | Trade-off |
|--------|----------|-----------|
| TF-IDF + LSA | Fast, interpretable, captures topic patterns | Loses word order, limited semantic understanding |
| NER | Identifies entities, provides context | Requires pre-trained model, language-specific |
| Sentence Embeddings | Rich semantic representation | Slower, less interpretable, larger model size |

The project uses **different methods for different tasks**:
- **EP classifier**: TF-IDF + LSA (fast, topic-focused for ESG detection)
- **FP classifier**: Embeddings + NER (semantic understanding for brand disambiguation)

This hybrid approach balances performance, interpretability, and computational cost.

## FP Classifier Deployment

The False Positive Brand Classifier is deployed as a FastAPI REST API service. It integrates with the labeling pipeline as an optional pre-filter to reduce LLM costs by skipping high-confidence false positives before calling Claude.

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
│         /predict/batch   (set in .env)                                 │
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

**Note**: Docker image size depends on the transformer method (~150MB for TF-IDF/NER, ~1GB with sentence-transformers). The build auto-detects dependencies from the model config.

### Labeling Pipeline Integration

Enable the FP classifier as a pre-filter in the labeling pipeline to reduce LLM costs:

```bash
# 1. Start the FP classifier API
docker compose up -d fp-classifier-api

# 2. Enable pre-filter in .env
FP_CLASSIFIER_ENABLED=true
FP_CLASSIFIER_URL=http://localhost:8000
FP_SKIP_LLM_THRESHOLD=0.5  # Skip LLM for articles <50% sportswear probability

# 3. Run labeling - FP classifier automatically filters false positives
uv run python scripts/label_articles.py --batch-size 20
```

The pipeline uses batch API calls for efficiency (1 call per batch, not per article). Articles below the threshold are marked as `false_positive` and skip LLM labeling, saving ~$0.01-0.02 per article.

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

### Model Registry & Deployment Workflow

The model registry (`models/registry.json`) tracks all model versions and their metadata. When deploying, the Dockerfile automatically reads the transformer method from the model config to install only the required dependencies.

**Transformer Method → Dependencies:**
| Method Pattern | Dependencies | Image Size |
|----------------|--------------|------------|
| `tfidf_lsa` | scikit-learn only | ~150MB |
| `*_ner*` | spaCy (en_core_web_sm) | ~150MB |
| `sentence_transformer*` | spaCy + sentence-transformers | ~1GB |

**Promoting a new model version:**

```bash
# After training, promote to registry
uv run python scripts/promote_model.py --classifier fp --version v3 --production

# Dry run to preview changes
uv run python scripts/promote_model.py --classifier fp --version v3 --dry-run

# The script will:
# 1. Read transformer_method from models/fp_classifier_config.json
# 2. Create version entry with metrics in models/registry.json
# 3. Set as production version (if --production flag used)
```

**Build will auto-detect dependencies:**

```bash
# Build reads fp_classifier_config.json and installs only what's needed
docker build -t fp-classifier-api .

# Build log will show:
# "Detected transformer method: tfidf_lsa_ner_proximity_brands"
# "Installing spaCy for tfidf_lsa_ner_proximity_brands..."
```

**Manual workflow (if not using promote_model.py):**

1. Train model via notebooks or `scripts/train.py`
2. Verify artifacts exist: `models/{classifier}_classifier_config.json`, `models/{classifier}_classifier_pipeline.joblib`
3. Update `models/registry.json` with new version entry
4. Commit changes: `git add models/ && git commit -m "Promote model"`
5. Build Docker image - dependencies auto-detected from config

## Model Deployment Workflow

This section describes the complete workflow for developing, training, evaluating, and deploying model updates. Follow this process when you have new training data or want to improve model performance.

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DEVELOP: Run notebooks with new data                         │
│    fp1 → fp2 → fp3 (or ep1 → ep2 → ep3)                        │
│    Exports: feature transformer, training config, CV metrics    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAIN: Run train.py                                          │
│    → Trains model using notebook config                         │
│    → Registers in MLflow Model Registry (if enabled)            │
│    → Saves artifacts to models/                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. EVALUATE: Compare to production                              │
│    - Check F2, recall, precision vs current production          │
│    - Review MLflow UI for metrics comparison                    │
│    - Decide if improvement justifies promotion                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. PROMOTE: Move to production (retrain.py)                     │
│    → Archives old production model                              │
│    → Updates models/registry.json                               │
│    → Promotes in MLflow Model Registry                          │
│    → Redeploy API service                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Development (Notebooks)

When you have new training data or want to experiment with model improvements:

```bash
# Export new training data
uv run python scripts/export_training_data.py --dataset fp

# Run the 3-notebook pipeline
# 1. fp1_EDA_FE.ipynb - EDA and feature engineering
# 2. fp2_model_selection_tuning.ipynb - Model selection and tuning
# 3. fp3_model_evaluation_deployment.ipynb - Final evaluation
```

**Notebook outputs (saved to `models/`):**
- `fp_feature_transformer.joblib` - Fitted feature transformer
- `fp_training_config.json` - Model type, hyperparameters, feature method
- `fp_cv_metrics.json` - Cross-validation metrics
- `fp_best_classifier.joblib` - Best classifier from CV

### Phase 2: Training (train.py)

After running notebooks, use `train.py` to train the final model:

```bash
# Train using config exported from notebooks
uv run python scripts/train.py --classifier fp --verbose

# With custom data path
uv run python scripts/train.py --classifier fp --data-path data/fp_training_data.jsonl --verbose
```

**What train.py does:**
1. Loads training config from `models/fp_training_config.json`
2. Fits feature transformer on training data only (prevents leakage)
3. Trains classifier on train+val combined
4. Optimizes threshold for target recall (default: 99%)
5. Logs to MLflow (if `MLFLOW_ENABLED=true`)
6. Registers model in MLflow Model Registry
7. Saves pipeline and config to `models/`

**Output:**
```
============================================================
FP CLASSIFIER TRAINING
============================================================
...
Final metrics:
  Test F2: 0.9835
  Threshold: 0.5356
  MLflow run ID: e48416f28cb6407f956d71da09705526
  MLflow model version: 2
  Registered model: esg-classifier-fp
```

### Phase 3: Evaluation (Decide to Promote)

Before promoting, compare the new model to production:

**Promotion Criteria:**

| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Test F2 Score | ≥ production F2 | Compare train.py output to registry |
| Test Recall | ≥ target (99% FP, 98% EP) | train.py output |
| No regression | F2 decrease < 1% | Manual comparison |

**Quick comparison:**

```bash
# View current production metrics
cat models/registry.json | python -m json.tool

# Or use MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000 → Compare runs
```

**When NOT to promote:**
- F2 score decreased by more than 1%
- Recall dropped below target threshold
- Training data had quality issues
- Model shows overfitting (large train-test gap)

### Phase 4: Promotion (retrain.py)

Use `retrain.py` to promote models with semantic versioning:

```bash
# Daily retraining with new data (minor version bump, e.g., v1.0.0 → v1.1.0)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl

# Major version bump for new architecture (v1.x.x → v2.0.0)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --major

# Patch version bump for threshold fix (v1.0.0 → v1.0.1)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --patch

# Auto-promote if metrics improve
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --auto-promote

# Force promote regardless of metrics
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --force-promote
```

**What retrain.py does:**
1. Determines next version based on bump type (major/minor/patch)
2. Runs `train.py` to train new model
3. Compares metrics to current production
4. Prompts for confirmation (or auto-promotes with `--auto-promote`)
5. Copies model to production location
6. Updates `models/registry.json`
7. Promotes in MLflow Model Registry (if enabled)

**Output:**
```
============================================================
Retraining FP Classifier
============================================================
Current version: v1.0.0
New version:     v1.1.0 (minor bump)

Training completed successfully!
  Test F2: 0.9850
  MLflow model version: 3

============================================================
Version Comparison
============================================================
Production F2:  0.9835
New version F2: 0.9850
Difference:     +0.0015 (+0.15%)

Verdict: F2 change: +0.0015 (+0.15%)

Promote v1.1.0 to production? [y/N]: y

Promoting FP v1.1.0 to production...
  Copied pipeline to: models/fp_classifier_pipeline.joblib
  Copied config to: models/fp_classifier_config.json
  Updated registry: models/registry.json
  Promoted MLflow model version 3 to Production stage

FP v1.1.0 is now in production!
```

### Semantic Versioning

The project uses semantic versioning for model releases:

| Version Type | When to Use | Example |
|--------------|-------------|---------|
| **Major** (`--major`) | Breaking changes: new model architecture, different features, schema changes | RandomForest → XGBoost, new feature transformer |
| **Minor** (default) | Improvements: new training data, hyperparameter tuning, threshold adjustments | Daily retraining with fresh data |
| **Patch** (`--patch`) | Bug fixes: config corrections, minor threshold tweaks | Fix threshold from 0.53 to 0.54 |

**Version format:** `vMAJOR.MINOR.PATCH` (e.g., `v1.2.3`)

**Examples:**
```bash
# You collected more training data → minor bump
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl
# v1.0.0 → v1.1.0

# You switched from RandomForest to XGBoost → major bump
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --major
# v1.1.0 → v2.0.0

# You fixed a threshold bug → patch bump
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --patch
# v2.0.0 → v2.0.1
```

### Daily Retraining Workflow

For automated daily retraining as new articles are collected:

```bash
# 1. Export fresh training data (includes new labeled articles)
uv run python scripts/export_training_data.py --dataset fp

# 2. Retrain with auto-promote (promotes if F2 improves)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --auto-promote

# 3. Rebuild and redeploy Docker container (if promoted)
docker compose build fp-classifier-api
docker compose up -d fp-classifier-api
```

**Automation with cron:**
```bash
# Add to crontab for daily retraining at 2am
0 2 * * * cd /path/to/project && ./scripts/retrain_daily.sh fp >> logs/retrain.log 2>&1
```

### Post-Deployment Verification

After promoting a new model:

```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. Verify model info
curl http://localhost:8000/model/info

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Nike sustainability report", "content": "...", "brands": ["Nike"]}'

# 4. Monitor for drift (next day)
uv run python scripts/monitor_drift.py --classifier fp --days 1
```

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

## Database

PostgreSQL with pgvector stores articles, labels, embeddings, and classifier predictions.

| Table | Purpose |
|-------|---------|
| `articles` | News metadata, full content, labeling status |
| `brand_labels` | Per-brand ESG classifications with sentiment |
| `article_chunks` | Text chunks with embeddings for evidence matching |
| `classifier_predictions` | ML classifier audit trail |

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

The project includes a comprehensive test suite with **548 tests** covering data collection, labeling pipelines, ML deployment, retraining workflows, and MLOps modules.

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
