# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESG News Classifier for sportswear brands - a multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for brands including Nike, Adidas, Puma, Under Armour, Lululemon, Patagonia, Columbia Sportswear, New Balance, ASICS, and Reebok.

## Commands

```bash
# Setup
uv sync                                    # Install dependencies
uv sync --extra dev                        # Install dev dependencies (testing)
cp .env.example .env                       # Create environment file (add API keys)
docker compose up -d                       # Start PostgreSQL + pgvector

# Data Collection (use uv run to execute in venv)
uv run python scripts/collect_news.py                            # NewsData.io collection (requires API key)
uv run python scripts/collect_news.py --source gdelt             # GDELT collection (free, no key needed)
uv run python scripts/collect_news.py --source gdelt --timespan 6h  # GDELT with 6-hour window
uv run python scripts/collect_news.py --dry-run --max-calls 5    # Test without saving
uv run python scripts/collect_news.py --scrape-only              # Only scrape pending articles
uv run python scripts/gdelt_backfill.py                          # 3-month historical backfill

# Article Labeling (requires ANTHROPIC_API_KEY and OPENAI_API_KEY)
uv run python scripts/label_articles.py --stats                  # View labeling statistics
uv run python scripts/label_articles.py --dry-run --batch-size 5 # Test without saving
uv run python scripts/label_articles.py --batch-size 10          # Label batch of articles
uv run python scripts/label_articles.py --article-id UUID        # Label specific article

# Export Training Data
uv run python scripts/export_training_data.py --dataset fp       # False positive classifier data
uv run python scripts/export_training_data.py --dataset esg-prefilter  # ESG pre-filter data
uv run python scripts/export_training_data.py --dataset esg-labels     # Multi-label ESG data

# ML Classifier Training & API
uv run python scripts/train.py --classifier fp                 # Train FP classifier
uv run python scripts/train.py --classifier ep                 # Train EP classifier
CLASSIFIER_TYPE=fp uv run python scripts/predict.py            # Start FP API (port 8000)
CLASSIFIER_TYPE=ep uv run python scripts/predict.py            # Start EP API (port 8000)

# Testing
uv run pytest                              # Run all tests (664 tests)
uv run pytest -v                           # Run with verbose output
uv run pytest --cov=src                    # Run with coverage report
RUN_DB_TESTS=1 uv run pytest tests/test_database.py  # Run database tests (requires PostgreSQL)

# Scheduled Collection (cron)
./scripts/setup_cron.sh install            # Set up both cron jobs
./scripts/setup_cron.sh status             # Check cron status
./scripts/setup_cron.sh remove             # Remove both cron jobs

# Database Backup
./scripts/backup_db.sh backup              # Create a new backup
./scripts/backup_db.sh list                # List available backups
./scripts/backup_db.sh status              # Show backup status and disk usage

# MLOps - Drift Monitoring
uv run python scripts/monitor_drift.py --classifier fp --from-db              # Production drift check (7 days)
uv run python scripts/monitor_drift.py --classifier fp --from-db --html-report  # Generate Evidently HTML report
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30  # Create reference dataset

# MLOps - MLflow (when MLFLOW_ENABLED=true)
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # Start MLflow UI (http://localhost:5000)

# Model Registration
uv run python scripts/register_model.py --classifier fp --version v2.2.0  # Register in MLflow
uv run python scripts/register_model.py --classifier fp --bump minor --update-registry  # Auto-version

# Website Feed Export
uv run python scripts/export_website_feed.py --format both \
  --json-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_data/esg_news.json \
  --atom-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/assets/feeds/esg_news.atom

# Agent Orchestrator
uv run python -m src.agent list                    # List available workflows
uv run python -m src.agent run daily_labeling      # Run daily labeling workflow
uv run python -m src.agent run drift_monitoring    # Run drift monitoring workflow
uv run python -m src.agent run website_export      # Run website export workflow
uv run python -m src.agent run daily_labeling --dry-run  # Dry run (no side effects)
uv run python -m src.agent status                  # Show workflow status
uv run python -m src.agent history                 # Show workflow history
./scripts/setup_cron.sh install-agent              # Install all agent cron jobs
./scripts/setup_cron.sh status                     # Check cron status
```

## Data Collection Status Reporting

When the user asks about data collection progress, run:

```python
uv run python -c "
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    print('=' * 60)
    print('DATA COLLECTION STATUS REPORT')
    print('=' * 60)

    result = conn.execute(text('''
        SELECT COUNT(*) as runs, SUM(articles_fetched) as fetched,
               SUM(articles_scraped) as scraped, SUM(articles_scrape_failed) as failed
        FROM collection_runs WHERE started_at >= NOW() - INTERVAL '7 days'
    '''))
    row = result.fetchone()
    print(f'Last 7 days: {row.runs} runs, {row.fetched} fetched, {row.scraped} scraped, {row.failed} failed')

    print('\nLABELING STATUS')
    total = conn.execute(text('SELECT COUNT(*) FROM articles')).scalar()
    for status in ['labeled', 'false_positive', 'skipped', 'pending', 'unlabelable']:
        count = conn.execute(text(f\"SELECT COUNT(*) FROM articles WHERE labeling_status = '{status}'\")).scalar()
        print(f'  {status:<16} {count:>6} ({count/total*100:.1f}%)')
    print(f'  TOTAL           {total:>6}')
"
```

### Labeling Status Definitions
- **labeled**: Successfully processed by LLM with ESG categories assigned
- **false_positive**: Brand name matched but article is not about sportswear
- **skipped**: Deliberately skipped (insufficient content, duplicate, etc.)
- **pending**: Awaiting labeling (scraped successfully, ready to process)
- **unlabelable**: Cannot be labeled (scrape failed, paywall, anti-bot, non-English)

### SQL Queries
See `queries/` folder for comprehensive SQL queries:
- `collection_queries.sql` - Collection runs, scrape errors, diagnostics
- `labeling_queries.sql` - Labeling progress, brand labels, historical review
- `article_queries.sql` - Article search, brand analysis, content analysis
- `evidence_queries.sql` - Evidence matching and chunk analysis

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, and API configuration
- `api_client.py` - NewsData.io API wrapper with OR-grouped query generation
- `gdelt_client.py` - GDELT DOC 2.0 API wrapper (free, 3 months history)
- `scraper.py` - Full article text extraction with language detection
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun, ArticleChunk, BrandLabel, LabelEvidence, LabelingRun)
- `collector.py` - Orchestrates API collection + scraping with in-memory deduplication

### Labeling Pipeline (`src/labeling/`)
- `config.py` - Labeling settings, Claude prompts, ESG category definitions
- `models.py` - Pydantic models for LLM response parsing
- `chunker.py` - Paragraph-based article chunking with tiktoken token counting
- `embedder.py` - OpenAI embedding wrapper with batch processing
- `labeler.py` - Claude Sonnet wrapper for ESG classification
- `classifier_client.py` - HTTP client for FP/EP classifier APIs
- `evidence_matcher.py` - Links evidence excerpts to chunks via similarity matching
- `pipeline.py` - Orchestrates FP pre-filter → chunking → embedding → labeling

### Prompt Versioning (`prompts/labeling/`)

```
prompts/labeling/
├── registry.json          # Version registry with metadata
├── v1.0.0/, v1.1.0/, v1.2.0/
    ├── config.json, system_prompt.txt, user_prompt.txt
```

**To update prompts:** Create new version directory → Update files → Update registry.json → Update `src/labeling/config.py` (runtime prompt)

### Scripts (`scripts/`)
- `collect_news.py` - CLI for NewsData.io/GDELT data collection
- `label_articles.py` - CLI for LLM-based article labeling
- `export_training_data.py` - Export labeled data for ML training (JSONL format)
- `train.py` - Unified training script for FP/EP classifiers (with MLflow integration)
- `predict.py` - Unified FastAPI service for all classifiers
- `retrain.py` - Retrain models with version management
- `register_model.py` - Register models in MLflow without retraining
- `monitor_drift.py` - Monitor prediction drift with Evidently AI
- `export_website_feed.py` - Export labeled articles as JSON/Atom feeds

### MLOps Module (`src/mlops/`)
- `config.py` - MLOps settings from environment variables
- `tracking.py` - MLflow experiment tracking wrapper
- `monitoring.py` - Evidently-based drift detection
- `reference_data.py` - Reference dataset management
- `alerts.py` - Webhook notifications for Slack/Discord

### Agent Orchestrator (`src/agent/`)
- `config.py` - Agent settings (state dir, email, retries)
- `state.py` - YAML-based state management with checkpointing
- `runner.py` - Script execution wrapper with retry logic
- `notifications.py` - Unified notifications (Resend email + webhooks)
- `workflows/` - Workflow definitions:
  - `base.py` - Workflow base class and registry
  - `daily_labeling.py` - Collection check → labeling → quality metrics → reports
  - `drift_monitoring.py` - FP/EP classifier drift detection with alerts
  - `website_export.py` - JSON/Atom feed generation with git integration
- `__main__.py` - CLI entry point (run, status, list, history)

### ML Classifier Notebooks (`notebooks/`)

**False Positive Classifier (3 notebooks):** fp1_EDA_FE.ipynb → fp2_model_selection_tuning.ipynb → fp3_model_evaluation_deployment.ipynb
- **Best Model:** Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 98.8%)

**ESG Pre-filter Classifier (3 notebooks):** ep1_EDA_FE.ipynb → ep2_model_selection_tuning.ipynb → ep3_model_evaluation_deployment.ipynb
- **Best Model:** Logistic Regression with TF-IDF + LSA features (Test F2: 0.931, Recall: 100%)

**Notebook Standards:** All imports in Setup section, grouped: stdlib → third-party → project modules

### Notebook Utilities
- `src/fp1_nb/` - EDA & feature engineering: data_utils, eda_utils, preprocessing, feature_transformer, ner_analysis, modeling
- `src/fp2_nb/` - Model selection: overfitting_analysis
- `src/fp3_nb/` - Deployment: threshold_optimization, deployment
- `src/ep1_nb/`, `src/ep2_nb/`, `src/ep3_nb/` - Same structure for EP classifier

### Test Suite (`tests/`) - 664 tests
Core tests: test_api_client, test_gdelt_client, test_scraper, test_collector, test_database, test_chunker, test_labeler, test_embedder, test_evidence_matcher, test_labeling_pipeline, test_deployment, test_explainability, test_mlops_*, test_retrain, test_agent_*, test_integration

### Database Schema
- **articles**: Article metadata + scraped content + labeling_status
- **collection_runs**: Collection run statistics
- **article_chunks**: Chunked text with embeddings (pgvector)
- **brand_labels**: Per-brand ESG labels with sentiment
- **label_evidence**: Supporting excerpts linked to chunks
- **labeling_runs**: Labeling batch tracking
- **classifier_predictions**: ML classifier predictions audit trail

### Environment Variables
```
# Data Collection
NEWSDATA_API_KEY, DATABASE_URL, MAX_API_CALLS_PER_DAY=200, SCRAPE_DELAY_SECONDS=2
GDELT_TIMESPAN=3m, GDELT_MAX_RECORDS=250

# Labeling
ANTHROPIC_API_KEY, OPENAI_API_KEY, LABELING_MODEL=claude-sonnet-4-20250514
EMBEDDING_MODEL=text-embedding-3-small, LABELING_BATCH_SIZE=10

# FP Classifier Pre-filter
FP_CLASSIFIER_ENABLED=false, FP_CLASSIFIER_URL=http://localhost:8000
FP_SKIP_LLM_THRESHOLD=0.5, FP_CLASSIFIER_TIMEOUT=30.0

# MLOps
MLFLOW_ENABLED=false, MLFLOW_TRACKING_URI=sqlite:///mlruns.db
EVIDENTLY_ENABLED=false, DRIFT_THRESHOLD=0.1
REFERENCE_DATA_DIR=data/reference, REFERENCE_WINDOW_DAYS=30
ALERT_WEBHOOK_URL, ALERT_ON_DRIFT=true

# Agent Orchestrator
AGENT_EMAIL_ENABLED=false, AGENT_EMAIL_RECIPIENT=, AGENT_EMAIL_SENDER=
RESEND_API_KEY=  # Recommended for email (resend.com, 3000/month free)
AGENT_LLM_ANALYSIS=false, AGENT_LLM_ERROR_THRESHOLD=0.1
```

## ESG Category Structure

- **Environmental**: carbon_emissions, waste_management, sustainable_materials
- **Social**: worker_rights, diversity_inclusion, community_engagement
- **Governance**: ethical_sourcing, transparency, board_structure
- **Digital Transformation**: technology innovation, digital initiatives

Sentiment values: +1 (positive), 0 (neutral), -1 (negative)

## ML Classifier Opportunities

1. **False Positive Classifier** ✅ - Filter non-sportswear brand matches (Test F2: 0.974)
2. **ESG Pre-filter Classifier** ✅ - Identify ESG content before Claude (Test F2: 0.931, Recall: 100%)
3. **ESG Multi-label Classifier** - Planned

## Project Phases

1. **Data Collection** ✅ - NewsData.io/GDELT, scraping, PostgreSQL+pgvector
2. **LLM-Based Labeling** ✅ - Claude classification, evidence extraction, embeddings
3. **Model Development** (Current) - FP classifier ✅, EP classifier ✅, ESG multi-label planned
4. **Deployment & MLOps** ✅ - FastAPI, Docker, Cloud Run, drift monitoring

## Deployment

### Docker
```bash
docker build --build-arg CLASSIFIER_TYPE=fp -t fp-classifier-api .
docker run -p 8000:8000 -e CLASSIFIER_TYPE=fp fp-classifier-api
# Or: docker compose up fp-classifier-api ep-classifier-api
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Classify multiple articles |

### CI/CD
GitHub Actions → Google Cloud Run. Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `GCP_REGION`

## Website Feed Export

JSON/Atom feeds for Jekyll/al-folio site. Website repo: `/home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io`

## Labeling Pipeline Changelog

### 2026-01-16: Expanded Stock Article Classification Guidelines

Clarified criteria for distinguishing between `false_positive` (pure metrics) and `skipped` (substantive content) for stock/finance articles. This matters because FP classifier bypasses LLM - articles incorrectly marked `false_positive` are permanently excluded from future labeling.

**Substantive Content Indicators (→ `skipped`, send to LLM):**
- Named analyst firms with specific ratings (e.g., "Citigroup reiterates neutral", "BNP Paribas upgrades to hold")
- Consensus rating breakdowns (e.g., "1 Buy, 5 Hold, 1 Sell")
- Earnings results with context or CEO commentary
- Hedge fund/institutional investor activity with specifics (e.g., "GAMMA Investing grew position by 30%")
- Price targets from analysts
- Any editorial analysis or reasoning about the company

**Raw Metrics Only (→ `false_positive`, skip LLM):**
- Just stock price, PE ratio, moving averages
- Short interest numbers without analyst context
- Boilerplate company descriptions (template text)
- Pure ticker data (e.g., "NKE $78.50 +2.3%")

| Article Type | Example | Has Substantive Content? | Label |
|-------------|---------|-------------------------|-------|
| Raw metrics | "NKE stock up 4% today, 50-day MA $94.67" | ❌ No | `false_positive` |
| Short interest only | "ANTA short interest up 535%, ratio 0.8 days" | ❌ No | `false_positive` |
| Analyst ratings | "Citigroup neutral, BNP upgrades to hold, consensus: Hold" | ✅ Yes | `skipped` |
| Hedge fund activity | "GAMMA Investing grew position by 30.1% to $161K" | ✅ Yes | `skipped` |
| Earnings + context | "EPS $1.50 missed estimates, CEO announces restructuring" | ✅ Yes | `skipped` |
| Analyst with targets | "Deutsche Bank reiterates buy, $146 target" | ✅ Yes | `skipped` |

**FP Classifier Limitation:** The FP classifier may underweight less common brands (Anta, Puma, Xtep) compared to Nike/Adidas/Lululemon. Review FP classifier predictions for these brands during labeling audits.

### 2026-01-14: Clarified is_sportswear_brand Policy for Stock Articles

`is_sportswear_brand` is about **substantive content**, not just identity:
- `true` → Article has substantive content (products, business news, strategy, analyst commentary with reasoning)
- `false` → Brand refers to something else OR pure stock metrics only (no substantive content)

**Finance Category Test:** Would this be useful with a "Finance" category? YES → `is_sportswear_brand: true`. Just raw Yahoo Finance metrics → `false`.

| Article | Substantive? | Label |
|---------|-------------|-------|
| "NKE stock up 4% today" | No | `false_positive` |
| "Jim Cramer says Nike CEO is reinventing the company" | Yes | `skipped` |
| "Nike shares surge after CEO announces restructuring" | Yes | `skipped` |

### 2025-12-29: MLOps Improvements

Added `src/mlops/` module: MLflow tracking, Evidently drift detection, webhook alerts, daily monitoring cron job.

### 2025-12-29: FP Classifier Batch API

Optimized to batch API calls (N articles → 1 call). Fixed Docker deployment issues.

### 2025-12-28: FP Classifier Pre-filter Integration

FP classifier as optional pre-filter: articles with probability < threshold marked `false_positive`, skip LLM.
- `FP_CLASSIFIER_ENABLED=true`, `FP_SKIP_LLM_THRESHOLD=0.5`
- Migration: `psql $DATABASE_URL -f migrations/002_classifier_predictions.sql`

### 2025-12-26: Added skipped_at Timestamp & Tangential Brand Mention Guidance

Added `skipped_at` column for tracking. Updated prompts to identify false positives for tangential brand mentions (biographical, stock-only, incidental references).

Migration: `ALTER TABLE articles ADD COLUMN skipped_at TIMESTAMP WITH TIME ZONE;`
