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

# Cross-Encoder Reranking Backfill
uv run python scripts/backfill_rerank_scores.py --dry-run    # Preview backfill
uv run python scripts/backfill_rerank_scores.py              # Run backfill
uv run python scripts/backfill_rerank_scores.py --batch-size 100  # Custom batch size

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
uv run python -m src.agent run model_training      # Run model training workflow (pauses for notebooks)
uv run python -m src.agent run daily_labeling --dry-run  # Dry run (no side effects)
uv run python -m src.agent continue model_training # Resume paused workflow after notebooks
uv run python -m src.agent status                  # Show workflow status
uv run python -m src.agent history                 # Show workflow history
./scripts/setup_cron.sh install-agent              # Install all agent cron jobs
./scripts/setup_cron.sh status                     # Check cron status
```

## Status Reporting & SQL Queries

For collection/labeling status reporting, see [docs/COLLECTION.md](docs/COLLECTION.md#status-reporting).

SQL query examples in `queries/` folder: `collection_queries.sql`, `labeling_queries.sql`, `article_queries.sql`, `evidence_queries.sql`, `scorecard_queries.sql`

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, blocked domains, and API configuration
- `api_client.py` - NewsData.io API wrapper with OR-grouped query generation
- `gdelt_client.py` - GDELT DOC 2.0 API wrapper (free, 3 months history)
- `scraper.py` - Full article text extraction with language detection
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun, ArticleChunk, BrandLabel, LabelEvidence, LabelingRun)
- `collector.py` - Orchestrates API collection + scraping with in-memory deduplication and domain filtering

**Domain Blacklist:** Low-quality or AI-generated content sources are blocked via `BLOCKED_DOMAINS` in `config.py`. Articles from blocked domains are filtered during collection and tracked in `CollectionStats.articles_blocked_domain`.

### Labeling Pipeline (`src/labeling/`)
- `config.py` - Labeling settings, Claude prompts, ESG category definitions
- `models.py` - Pydantic models for LLM response parsing
- `chunker.py` - Paragraph-based article chunking with tiktoken token counting
- `embedder.py` - OpenAI embedding wrapper with batch processing
- `labeler.py` - Claude Sonnet wrapper for ESG classification
- `classifier_client.py` - HTTP client for FP/EP classifier APIs
- `evidence_matcher.py` - Links evidence excerpts to chunks via similarity matching
- `reranker.py` - Cross-encoder reranking for improved evidence quality
- `pipeline.py` - Orchestrates FP pre-filter → chunking → embedding → labeling → reranking

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
- `config.py` - Agent settings (state dir, email, retries, LLM)
- `state.py` - YAML-based state management with checkpointing
- `runner.py` - Script execution wrapper with retry logic
- `notifications.py` - Unified notifications (Resend email + webhooks)
- `llm.py` - Claude Sonnet integration for labeling analysis
- `workflows/` - Workflow definitions:
  - `base.py` - Workflow base class and registry
  - `daily_labeling.py` - Collection check → labeling → quality metrics → reports
  - `drift_monitoring.py` - FP/EP classifier drift detection with alerts
  - `website_export.py` - JSON/Atom feed generation + scorecard history storage
  - `model_training.py` - Data export → quality check → pause → comparison → promotion → deploy
- `__main__.py` - CLI entry point (run, continue, status, list, history)

### Scorecard History (`src/scorecard/`)
- `database.py` - ScorecardDatabase class for save/query operations
- Stores daily scorecard snapshots with brand scores for trend analysis
- Integrated into `website_export` workflow (saves after each export)

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

### Test Suite (`tests/`) - 921 tests
Core tests: test_api_client, test_gdelt_client, test_scraper, test_collector, test_database, test_chunker, test_labeler, test_embedder, test_evidence_matcher, test_labeling_pipeline, test_deployment, test_explainability, test_mlops_*, test_retrain, test_agent_*, test_scorecard_database, test_integration

### Database Schema
- **articles**: Article metadata + scraped content + labeling_status
- **collection_runs**: Collection run statistics
- **article_chunks**: Chunked text with embeddings (pgvector)
- **brand_labels**: Per-brand ESG labels with sentiment
- **label_evidence**: Supporting excerpts linked to chunks (rerank_score, match_method)
- **labeling_runs**: Labeling batch tracking
- **classifier_predictions**: ML classifier predictions audit trail
- **scorecard_snapshots**: Daily scorecard metadata (period, article counts, dedup settings)
- **scorecard_brand_scores**: Per-brand scores per snapshot (category scores, rank, medals)

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

# Cross-Encoder Reranking (evidence quality improvement)
RERANK_ENABLED=true, RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10, RERANK_WEIGHT=0.6

# MLOps
MLFLOW_ENABLED=false, MLFLOW_TRACKING_URI=sqlite:///mlruns.db
EVIDENTLY_ENABLED=false, DRIFT_THRESHOLD=0.1
REFERENCE_DATA_DIR=data/reference, REFERENCE_WINDOW_DAYS=30
ALERT_WEBHOOK_URL, ALERT_ON_DRIFT=true

# Agent Orchestrator
AGENT_EMAIL_ENABLED=false, AGENT_EMAIL_RECIPIENT=, AGENT_EMAIL_SENDER=
RESEND_API_KEY=  # Recommended for email (resend.com, 3000/month free)
AGENT_LLM_ANALYSIS=true  # Enable Claude analysis of labeling results
AGENT_LLM_ERROR_THRESHOLD=0.0  # 0.0 = always run, >0 = only if error_rate exceeds
AGENT_LLM_MODEL=claude-sonnet-4-20250514  # Model for LLM analysis
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

### Export Commands
```bash
# Full export with scorecard (default)
uv run python scripts/export_website_feed.py --format both \
  --json-output ~/website/_data/esg_news.json \
  --atom-output ~/website/assets/feeds/esg_news.atom

# Export without scorecard
uv run python scripts/export_website_feed.py --format json --no-scorecard

# Custom scorecard period (default: 14 days)
uv run python scripts/export_website_feed.py --format json --scorecard-period-days 30

# Disable article deduplication for scorecard
uv run python scripts/export_website_feed.py --format json --no-dedupe

# Custom similarity threshold for deduplication (default: 0.56)
uv run python scripts/export_website_feed.py --format json --similarity-threshold 0.70
```

### Sustainability Scorecard

The website displays a "Sportswear Sustainability Scorecard" ranking brands based on ESG news sentiment over a rolling 14-day window.

**Scoring System:**
| Sentiment | Points |
|-----------|--------|
| Positive (+1) | +2 points |
| Neutral (0) | +1 point |
| Negative (-1) | -1 point |

**Key Rules:**
- Each article contributes **one score per brand per category** (Environmental, Social, Governance, Digital)
- Multiple evidence excerpts per category are for documentation only - they don't add extra points
- **Top Performers**: Top 3 brands with positive total scores (with medals: 🥇🥈🥉)
- **Back of the Pack**: Bottom 3 brands with negative total scores
- A brand cannot appear in both lists

**Article Deduplication:**
Similar news stories from different sources are deduplicated before scoring using sentence embeddings (all-MiniLM-L6-v2, 384-dim). Articles with cosine similarity >= 0.56 are considered duplicates; only the first is counted. This threshold was tuned on 200 labeled pairs to achieve ~92% recall with 73% precision.

**Website Features:**
- Date range filter with presets (7, 14, 30 days, All) and custom date inputs
- Filter by brand, category, and sentiment
- Collapsible evidence sections showing supporting excerpts

**Files:**
- `scripts/export_website_feed.py` - Export script with scorecard calculation
- `_pages/esg-news.md` - Jekyll page template
- `assets/js/esg_news_filter.js` - Client-side filtering
- `_sass/_esg_news.scss` - Scorecard and filter styling

## Changelog

For full changelog, see [docs/CHANGELOG.md](docs/CHANGELOG.md).

**Recent changes:**
- **2026-02-11**: Scorecard history storage - daily snapshots saved to `scorecard_snapshots` and `scorecard_brand_scores` tables
- **2026-01-27**: Domain blacklist for data collection - block low-quality sources via `BLOCKED_DOMAINS`
- **2026-01-26**: Sustainability scorecard for website - brand ranking with deduplication
- **2026-01-20**: Cross-encoder reranking for evidence quality

**Stock article classification guidelines:** See [docs/LABELING.md#stock-article-classification](docs/LABELING.md#stock-article-classification)
