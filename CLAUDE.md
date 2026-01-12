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
uv run python scripts/label_articles.py --skip-embedding         # Skip embedding generation

# Export Training Data
uv run python scripts/export_training_data.py --dataset fp       # False positive classifier data
uv run python scripts/export_training_data.py --dataset esg-prefilter  # ESG pre-filter data
uv run python scripts/export_training_data.py --dataset esg-labels     # Multi-label ESG data
uv run python scripts/export_training_data.py --dataset fp --since 2025-01-01  # Incremental export

# ML Classifier Training
uv run python scripts/train.py --classifier fp                 # Train FP classifier
uv run python scripts/train.py --classifier ep                 # Train EP classifier

# ML Classifier API Service
CLASSIFIER_TYPE=fp uv run python scripts/predict.py            # Start FP API (port 8000)
CLASSIFIER_TYPE=ep uv run python scripts/predict.py            # Start EP API (port 8000)
CLASSIFIER_TYPE=fp uv run uvicorn scripts.predict:app --port 8000  # FP with uvicorn
CLASSIFIER_TYPE=ep uv run uvicorn scripts.predict:app --port 8001  # EP with uvicorn

# Testing
uv run pytest                              # Run all tests (528 tests)
uv run pytest -v                           # Run with verbose output
uv run pytest --cov=src                    # Run with coverage report
RUN_DB_TESTS=1 uv run pytest tests/test_database.py  # Run database tests (requires PostgreSQL)

# Scheduled Collection (cron)
./scripts/setup_cron.sh install            # Set up both cron jobs
./scripts/setup_cron.sh status             # Check cron status
./scripts/setup_cron.sh remove             # Remove both cron jobs
./scripts/setup_cron.sh install-scrape     # Add GDELT collection job
./scripts/setup_cron.sh install-monitor    # Add drift monitoring job (daily)
./scripts/setup_cron.sh install-backup     # Add database backup job (daily at 2am)
tail -f logs/collection_$(date +%Y%m%d).log  # View NewsData logs
tail -f logs/gdelt_$(date +%Y%m%d).log       # View GDELT logs

# Database Backup
./scripts/backup_db.sh backup              # Create a new backup
./scripts/backup_db.sh list                # List available backups
./scripts/backup_db.sh status              # Show backup status and disk usage
./scripts/backup_db.sh restore --file backups/daily/esg_news_YYYYMMDD_HHMMSS.sql.gz  # Restore from backup
./scripts/backup_db.sh rotate              # Clean up old backups (runs automatically after backup)

# MLOps - Drift Monitoring (use --from-db for production predictions stored in database)
uv run python scripts/monitor_drift.py --classifier fp --from-db              # Production drift check (7 days)
uv run python scripts/monitor_drift.py --classifier fp --from-db --days 30    # Extended analysis
uv run python scripts/monitor_drift.py --classifier fp --from-db --html-report  # Generate Evidently HTML report
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30  # Create reference dataset
uv run python scripts/monitor_drift.py --classifier fp --reference-stats  # View reference stats
uv run python scripts/monitor_drift.py --classifier fp --from-db --alert      # Send webhook alert if drift
uv run python scripts/monitor_drift.py --classifier fp --logs-dir logs/predictions  # Legacy: from local log files

# MLOps - MLflow Experiment Tracking (when MLFLOW_ENABLED=true)
# NOTE: MLflow uses SQLite backend (mlruns.db). File-based backend (mlruns/) is deprecated.
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # Start MLflow UI (http://localhost:5000)

# Model Registration (after notebook training, without retraining)
uv run python scripts/register_model.py --classifier fp --version v2.2.0  # Register in MLflow
uv run python scripts/register_model.py --classifier fp --bump minor --update-registry  # Auto-version + update registry.json
uv run python scripts/register_model.py --classifier fp --register-model  # Also add to MLflow Model Registry

# Website Feed Export (for Jekyll/al-folio GitHub Pages)
# Website repo location: /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io
uv run python scripts/export_website_feed.py --format both \
  --json-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_data/esg_news.json \
  --atom-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/assets/feeds/esg_news.atom
uv run python scripts/export_website_feed.py --format json --limit 100 --dry-run  # Preview export
```

## Data Collection Status Reporting

When the user asks about data collection progress (e.g., "how is data collection going?", "collection status"), run the following Python script to generate a comprehensive report:

```python
uv run python -c "
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    print('=' * 80)
    print('DATA COLLECTION STATUS REPORT')
    print('=' * 80)

    # Collection runs summary (last 7 days)
    print('\nCOLLECTION RUNS (Last 7 Days)')
    print('-' * 60)
    result = conn.execute(text('''
        SELECT
            COUNT(*) as runs,
            SUM(articles_fetched) as fetched,
            SUM(articles_duplicates) as duplicates,
            SUM(articles_scraped) as scraped,
            SUM(articles_scrape_failed) as failed
        FROM collection_runs
        WHERE started_at >= NOW() - INTERVAL '7 days'
    '''))
    row = result.fetchone()
    print(f'  Runs completed:       {row.runs}')
    print(f'  Articles fetched:     {row.fetched}')
    print(f'  Duplicates skipped:   {row.duplicates}')
    print(f'  Successfully scraped: {row.scraped}')
    print(f'  Scrape failures:      {row.failed}')

    print('\n' + '=' * 80)
    print('ARTICLE LABELING STATUS')
    print('=' * 80)

    total = conn.execute(text('SELECT COUNT(*) FROM articles')).scalar()
    labeled = conn.execute(text(\"SELECT COUNT(*) FROM articles WHERE labeling_status = 'labeled'\")).scalar()
    false_pos = conn.execute(text(\"SELECT COUNT(*) FROM articles WHERE labeling_status = 'false_positive'\")).scalar()
    skipped = conn.execute(text(\"SELECT COUNT(*) FROM articles WHERE labeling_status = 'skipped'\")).scalar()
    pending = conn.execute(text(\"SELECT COUNT(*) FROM articles WHERE labeling_status = 'pending'\")).scalar()
    unlabelable = conn.execute(text(\"SELECT COUNT(*) FROM articles WHERE labeling_status = 'unlabelable'\")).scalar()

    print(f'\n{\"Status\":<20} {\"Count\":>8} {\"Percent\":>10}')
    print('-' * 40)
    print(f'{\"labeled\":<20} {labeled:>8} {(labeled/total)*100:>9.1f}%')
    print(f'{\"false_positive\":<20} {false_pos:>8} {(false_pos/total)*100:>9.1f}%')
    print(f'{\"skipped\":<20} {skipped:>8} {(skipped/total)*100:>9.1f}%')
    print(f'{\"pending\":<20} {pending:>8} {(pending/total)*100:>9.1f}%')
    print(f'{\"unlabelable\":<20} {unlabelable:>8} {(unlabelable/total)*100:>9.1f}%')
    print('-' * 40)
    print(f'{\"TOTAL\":<20} {total:>8}')

    print('\n' + '=' * 80)
    print('RECENT DAILY COLLECTION')
    print('=' * 80)
    result = conn.execute(text('''
        SELECT
            created_at::date as date,
            COUNT(*) as total,
            SUM(CASE WHEN scrape_status = 'success' THEN 1 ELSE 0 END) as scraped,
            SUM(CASE WHEN scrape_status = 'failed' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN labeling_status = 'labeled' THEN 1 ELSE 0 END) as labeled
        FROM articles
        WHERE created_at >= NOW() - INTERVAL '7 days'
        GROUP BY 1
        ORDER BY 1 DESC
    '''))
    print(f'\n{\"Date\":<12} {\"Total\":>8} {\"Scraped\":>8} {\"Failed\":>8} {\"Labeled\":>8}')
    print('-' * 50)
    for row in result:
        print(f'{str(row.date):<12} {row.total:>8} {row.scraped:>8} {row.failed:>8} {row.labeled:>8}')

    print('\n' + '=' * 80)
"
```

### Labeling Status Definitions

- **labeled**: Successfully processed by LLM with ESG categories assigned
- **false_positive**: Brand name matched but article is not about sportswear (e.g., "Puma" the animal)
- **skipped**: Deliberately skipped (insufficient content, duplicate, etc.)
- **pending**: Awaiting labeling (scraped successfully, ready to process)
- **unlabelable**: Cannot be labeled (scrape failed, paywall, anti-bot, non-English)

### Follow-up Actions

1. **If pending > 0**: Run the labeling pipeline to process pending articles:
   ```bash
   uv run python scripts/label_articles.py --batch-size <pending_count>
   ```

2. **If scrape failures are high**: Check the scrape error patterns:
   ```sql
   SELECT scrape_error, COUNT(*) FROM articles
   WHERE scrape_status = 'failed'
   GROUP BY 1 ORDER BY 2 DESC LIMIT 10;
   ```

3. **To view recent collection run details**:
   ```sql
   SELECT started_at::date, started_at::time, status, articles_fetched, articles_scraped, articles_scrape_failed
   FROM collection_runs ORDER BY started_at DESC LIMIT 10;
   ```

### Important Notes

- Failed/skipped scrapes are automatically marked as `unlabelable` (not `pending`)
- Cron jobs run 4x daily for both NewsData.io and GDELT collection
- Check logs at `logs/collection_*.log` and `logs/gdelt_*.log` for errors

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, and API configuration
- `api_client.py` - NewsData.io API wrapper with OR-grouped query generation
- `gdelt_client.py` - GDELT DOC 2.0 API wrapper (free, 3 months history)
- `scraper.py` - Full article text extraction with language detection (filters non-English)
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun, ArticleChunk, BrandLabel, LabelEvidence, LabelingRun)
- `collector.py` - Orchestrates API collection + scraping with in-memory deduplication

### Labeling Pipeline (`src/labeling/`)
- `config.py` - Labeling settings, Claude prompts, ESG category definitions
- `models.py` - Pydantic models for LLM response parsing (CategoryLabel, BrandAnalysis, LabelingResponse)
- `chunker.py` - Paragraph-based article chunking with tiktoken token counting
- `embedder.py` - OpenAI embedding wrapper with batch processing
- `labeler.py` - Claude Sonnet wrapper for ESG classification
- `classifier_client.py` - HTTP client for FP/EP classifier APIs (pre-filter integration)
- `evidence_matcher.py` - Links evidence excerpts to chunks via exact/fuzzy/embedding similarity
- `database.py` - Labeling-specific DB operations
- `pipeline.py` - Orchestrates FP pre-filter → chunking → embedding → labeling → evidence matching

### Scripts (`scripts/`)
- `collect_news.py` - CLI for NewsData.io/GDELT data collection
- `label_articles.py` - CLI for LLM-based article labeling
- `export_training_data.py` - Export labeled data for ML training (JSONL format)
- `gdelt_backfill.py` - Historical backfill script (3 months in weekly batches)
- `cleanup_non_english.py` - Remove non-English articles from database
- `cleanup_false_positives.py` - Identify/remove false positive brand matches
- `cron_collect.sh` - NewsData.io collection + scraping (runs 4x daily at 12am, 6am, 12pm, 6pm)
- `cron_scrape.sh` - GDELT collection + scraping (runs 4x daily at 3am, 9am, 3pm, 9pm)
- `cron_monitor.sh` - Drift monitoring wrapper (runs daily at 6am UTC)
- `setup_cron.sh` - Install/remove/status commands for cron management
- `train.py` - Unified training script for FP/EP classifiers (with MLflow integration)
- `predict.py` - Unified FastAPI service for all classifiers
- `retrain.py` - Retrain models with version management
- `register_model.py` - Register models in MLflow without retraining (for notebook workflows)
- `monitor_drift.py` - Monitor prediction drift with Evidently AI
- `export_website_feed.py` - Export labeled articles as JSON/Atom feeds for website display

### MLOps Module (`src/mlops/`)
- `config.py` - MLOps settings (MLflow, Evidently, alerts) from environment variables
- `tracking.py` - MLflow experiment tracking wrapper with graceful degradation
- `monitoring.py` - Evidently-based drift detection (falls back to KS test when disabled)
- `reference_data.py` - Reference dataset management for drift comparison
- `alerts.py` - Webhook notifications for Slack/Discord

### ML Classifier Notebooks (`notebooks/`)

**False Positive Classifier Pipeline (3 notebooks):**
- `fp1_EDA_FE.ipynb` - EDA & Feature Engineering (exports transformer)
- `fp2_model_selection_tuning.ipynb` - Model selection & hyperparameter tuning
- `fp3_model_evaluation_deployment.ipynb` - Test evaluation & deployment export

**Best Model:** Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 98.8%)

**ESG Pre-filter Classifier Pipeline (3 notebooks):**
- `ep1_EDA_FE.ipynb` - EDA & Feature Engineering (exports transformer)
- `ep2_model_selection_tuning.ipynb` - Model selection & hyperparameter tuning
- `ep3_model_evaluation_deployment.ipynb` - Test evaluation & deployment export

**Best Model:** Logistic Regression with TF-IDF + LSA features (Test F2: 0.931, Recall: 100%)

**Notebook Standards:**
- All package imports MUST be placed in the Setup section at the beginning of the notebook
- Do not scatter imports throughout the notebook cells
- Group imports: standard library, third-party packages, then project modules

### Notebook Utilities

**`src/fp1_nb/`** - EDA & feature engineering utilities:
- `data_utils.py` - JSONL loading, target analysis, stratified train/val/test splitting
- `eda_utils.py` - Text length analysis, brand distribution, word frequency analysis
- `preprocessing.py` - Text cleaning, feature engineering
- `feature_transformer.py` - Sentence transformer + NER brand context features
- `ner_analysis.py` - Named entity recognition utilities
- `modeling.py` - GridSearchCV utilities, model evaluation metrics, comparison plots

**`src/fp2_nb/`** - Model selection & tuning utilities:
- `overfitting_analysis.py` - Train-validation gap visualization, iteration performance

**`src/fp3_nb/`** - Evaluation & deployment utilities:
- `threshold_optimization.py` - Threshold tuning for target recall
- `deployment.py` - Pipeline export utilities

**`src/ep1_nb/`** - ESG Pre-filter EDA & feature engineering utilities:
- `data_utils.py` - JSONL loading, target analysis, stratified train/val/test splitting
- `eda_utils.py` - Text length analysis, brand distribution, word frequency analysis
- `preprocessing.py` - Text cleaning, feature engineering
- `feature_transformer.py` - EPFeatureTransformer with ESG-specific vocabularies
- `modeling.py` - GridSearchCV utilities, model evaluation metrics, comparison plots

**`src/ep2_nb/`** - ESG Pre-filter model selection & tuning utilities:
- `overfitting_analysis.py` - Train-validation gap visualization, iteration performance

**`src/ep3_nb/`** - ESG Pre-filter evaluation & deployment utilities:
- `threshold_optimization.py` - Threshold tuning for target recall
- `deployment.py` - Pipeline export utilities

### Test Suite (`tests/`) - 528 tests
- `conftest.py` - Shared pytest fixtures
- `test_api_client.py` - NewsData.io brand extraction, article parsing, query generation (23 tests)
- `test_gdelt_client.py` - GDELT article parsing, query generation, date handling (31 tests)
- `test_scraper.py` - Language detection, paywall detection, scraping (19 tests)
- `test_collector.py` - Deduplication, dry run mode, API limits (13 tests)
- `test_database.py` - Upsert operations, queries (12 tests, requires PostgreSQL)
- `test_chunker.py` - Article chunking, token counting, paragraph boundaries (21 tests)
- `test_labeler.py` - LLM response parsing, ArticleLabeler, JSON extraction (33 tests)
- `test_embedder.py` - OpenAI embedder, batching, retry logic (15 tests)
- `test_evidence_matcher.py` - Evidence matching, fuzzy/exact/embedding similarity (24 tests)
- `test_labeling_pipeline.py` - Pipeline orchestration, statistics tracking (13 tests)
- `test_deployment.py` - FP/EP classifiers, config, data loading, preprocessing (64 tests)
- `test_explainability.py` - LIME, SHAP feature groups, prototype explanations (28 tests)
- `test_mlops_tracking.py` - MLflow experiment tracking, graceful degradation (27 tests)
- `test_mlops_monitoring.py` - Evidently drift detection, legacy KS tests (28 tests)
- `test_mlops_reference_data.py` - Reference data loading, database predictions (22 tests)
- `test_retrain.py` - Retraining pipeline, semantic versioning, deployment triggers (38 tests)
- `test_integration.py` - End-to-end classifier pipeline tests (12 tests)

### Database Schema
- **articles**: Stores article metadata from API + full scraped content + labeling_status
- **collection_runs**: Logs each daily collection run with statistics
- **article_chunks**: Chunked article text with embeddings (pgvector) for evidence linking
- **brand_labels**: Per-brand ESG labels with sentiment (-1/0/+1) and confidence
- **label_evidence**: Supporting excerpts linked to chunks via similarity matching
- **labeling_runs**: Tracks labeling batches with cost estimates
- **classifier_predictions**: Stores ML classifier predictions (FP/EP/ESG) for audit trail

### Environment Variables
- `NEWSDATA_API_KEY` - Required for NewsData.io collection (not needed for GDELT)
- `DATABASE_URL` - PostgreSQL connection string
- `MAX_API_CALLS_PER_DAY` - Rate limit (default: 200)
- `SCRAPE_DELAY_SECONDS` - Delay between scrape requests (default: 2)
- `GDELT_TIMESPAN` - Default GDELT time window (default: 3m)
- `GDELT_MAX_RECORDS` - Max records per GDELT query (default: 250)
- `ANTHROPIC_API_KEY` - Required for Claude labeling
- `OPENAI_API_KEY` - Required for embeddings
- `LABELING_MODEL` - Claude model for labeling (default: claude-sonnet-4-20250514)
- `EMBEDDING_MODEL` - OpenAI embedding model (default: text-embedding-3-small)
- `LABELING_BATCH_SIZE` - Articles per labeling batch (default: 10)
- `FP_CLASSIFIER_ENABLED` - Enable FP classifier pre-filter (default: false)
- `FP_CLASSIFIER_URL` - FP classifier API URL (default: http://localhost:8000)
- `FP_SKIP_LLM_THRESHOLD` - Probability threshold to skip LLM (default: 0.5, matches trained model threshold of 0.5356)
- `FP_CLASSIFIER_TIMEOUT` - FP classifier API timeout in seconds (default: 30.0)
- `MLFLOW_ENABLED` - Enable MLflow experiment tracking (default: false)
- `MLFLOW_TRACKING_URI` - MLflow tracking URI (default: sqlite:///mlruns.db). Note: File-based backend (file:./mlruns) is deprecated.
- `MLFLOW_EXPERIMENT_PREFIX` - Prefix for experiment names (default: esg-classifier)
- `EVIDENTLY_ENABLED` - Enable Evidently drift detection (default: false)
- `EVIDENTLY_REPORTS_DIR` - Directory for HTML reports (default: reports/monitoring)
- `DRIFT_THRESHOLD` - Drift score threshold for alerts (default: 0.1)
- `REFERENCE_DATA_DIR` - Directory for reference datasets (default: data/reference)
- `REFERENCE_WINDOW_DAYS` - Days of data for reference (default: 30)
- `ALERT_WEBHOOK_URL` - Slack/Discord webhook URL for alerts
- `ALERT_ON_DRIFT` - Send alert on drift detection (default: true)
- `ALERT_ON_TRAINING` - Send alert after training (default: false)

## Search Keywords

Keywords are grouped by category for data collection:

- **Environmental**: sustainability, climate, emissions, recycling, environment, carbon, green
- **Social**: labor, workers, factory, supply chain, diversity
- **Governance**: ESG, ethics, transparency
- **Digital Transformation**: digital, technology, innovation

## ESG Category Structure (for classification)

- **Environmental**: carbon_emissions, waste_management, sustainable_materials
- **Social**: worker_rights, diversity_inclusion, community_engagement
- **Governance**: ethical_sourcing, transparency, board_structure

## Query Optimization

The API client uses OR-grouped queries to maximize article yield per API call:
- Format: `(Brand1 OR Brand2 OR Brand3) keyword`
- Respects NewsData.io free tier 100-char query limit
- Generates ~54 optimized queries (vs 180+ individual queries)

## ESG Labeling Categories

The labeling pipeline classifies articles into 4 top-level ESG categories with ternary sentiment:
- **Environmental**: Carbon emissions, waste management, sustainable materials
- **Social**: Worker rights, diversity & inclusion, community engagement
- **Governance**: Ethical sourcing, transparency, board structure
- **Digital Transformation**: Technology innovation, digital initiatives

Sentiment values: +1 (positive), 0 (neutral), -1 (negative)

## ML Classifier Opportunities

Three classifiers to reduce Claude API costs while maintaining accuracy:

1. **False Positive Brand Classifier** ✅ - Filter articles where brand names match non-sportswear entities (e.g., "Puma" animal, "Patagonia" region, "Black Diamond" power company). Binary classification using `--dataset fp` export. **Complete**: Random Forest with Test F2: 0.974.

2. **ESG Pre-filter Classifier** ✅ - Quickly identify if an article contains ESG content before detailed classification. Binary classification using `--dataset esg-prefilter` export. **Complete**: Logistic Regression with Test F2: 0.931, Recall: 100%.

3. **ESG Multi-label Classifier** - Full ESG category classification with sentiment, replacing Claude for routine cases. Multi-label output using `--dataset esg-labels` export.

## Project Phases

### Phase 1: Data Collection ✅
- NewsData.io and GDELT API integration
- Article scraping with language detection
- PostgreSQL + pgvector storage

### Phase 2: LLM-Based Labeling ✅
- Claude Sonnet classification with structured JSON output
- Per-brand ESG labels with sentiment
- Evidence extraction and chunk linking
- OpenAI embeddings for semantic matching

### Phase 3: Model Development (Current)
- Export labeled data for training (993 FP, 870 ESG-prefilter, 554 ESG-labels records)
- False Positive Classifier: 3-notebook pipeline complete ✅
  - Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 98.8%)
  - Threshold optimized for 98% target recall
- ESG Pre-filter Classifier: 3-notebook pipeline complete ✅
  - Logistic Regression with TF-IDF + LSA features (Test F2: 0.931, Recall: 100%)
  - Threshold optimized for 99% target recall
- ESG Multi-label Classifier: Planned
- Advanced: Fine-tuned DistilBERT/RoBERTa (future)

### Key Metrics
- Per-category Precision, Recall, F1-score
- Hamming Loss (multi-label specific)
- SHAP values for model explainability

### Phase 4: Deployment & MLOps ✅
- Unified FastAPI service for all classifiers (`scripts/predict.py`)
- Docker multi-stage builds optimized per classifier type
- GitHub Actions CI/CD to Google Cloud Run
- Model registry with version tracking (`models/registry.json`)
- Prediction logging for drift monitoring
- Retraining pipeline with auto-promotion

## Deployment

### Running Locally with Docker

```bash
# Build and run FP classifier
docker build --build-arg CLASSIFIER_TYPE=fp -t fp-classifier-api .
docker run -p 8000:8000 -e CLASSIFIER_TYPE=fp fp-classifier-api

# Build and run EP classifier
docker build --build-arg CLASSIFIER_TYPE=ep -t ep-classifier-api .
docker run -p 8001:8001 -e CLASSIFIER_TYPE=ep -e PORT=8001 ep-classifier-api

# Or use docker-compose for both
docker compose up fp-classifier-api ep-classifier-api
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for container orchestration |
| `/model/info` | GET | Model metadata and performance metrics |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Classify multiple articles |

### CI/CD Deployment

Automatic deployment to Google Cloud Run via GitHub Actions.

**Required GitHub Secrets:**
- `GCP_PROJECT_ID` - Google Cloud project ID
- `GCP_SA_KEY` - Service account JSON key (Cloud Run Admin + Storage Admin roles)
- `GCP_REGION` - (optional) Cloud Run region, defaults to us-central1

See `.github/DEPLOYMENT_SETUP.md` for detailed setup instructions.

### Model Management

```bash
# Train new model version
uv run python scripts/train.py --classifier fp --data data/fp_training_data.jsonl

# Retrain and auto-promote if better
uv run python scripts/retrain.py --classifier fp --auto-promote

# Monitor for drift (--from-db for production, or --logs-dir for local)
uv run python scripts/monitor_drift.py --classifier fp --from-db --days 7
```

### Prediction Logging

Production predictions (from Cloud Run) are logged to the `classifier_predictions` database table.
Local API predictions are logged to `logs/predictions/{classifier}_predictions_{date}.jsonl`.

Environment variables:
- `ENABLE_PREDICTION_LOGGING` - Enable/disable logging (default: true)
- `PREDICTION_LOGS_DIR` - Log directory (default: logs/predictions)

## Website Feed Export

Export labeled ESG news articles for display on a Jekyll/al-folio static website.

### JSON Schema

```json
{
  "generated_at": "2025-01-01T12:00:00Z",
  "total_articles": 150,
  "brands": ["Nike", "Adidas", ...],
  "categories": ["environmental", "social", "governance", "digital_transformation"],
  "articles": [
    {
      "id": "uuid",
      "title": "Article headline",
      "url": "https://source.com/article",
      "source_name": "Reuters",
      "published_at": "2025-01-15T10:30:00Z",
      "published_date": "2025-01-15",
      "brands": ["Nike"],
      "categories": ["environmental", "governance"],
      "brand_details": [
        {
          "brand": "Nike",
          "categories": {
            "environmental": {"applies": true, "sentiment": 1, "sentiment_label": "positive"},
            "social": {"applies": false, "sentiment": null, "sentiment_label": null}
          },
          "evidence": [
            {"category": "environmental", "excerpt": "Quote from article...", "relevance_score": 0.95}
          ],
          "confidence": 0.92,
          "reasoning": "LLM explanation"
        }
      ]
    }
  ]
}
```

### Update Workflow

```bash
# Website repo: /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io

# 1. In ML project: Run export script
cd /home/fdpearce/Documents/Courses/DataTalksClub/projects/machine-learning-zoomcamp/sportswear-esg-news-classifier
uv run python scripts/export_website_feed.py --format both \
  --json-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_data/esg_news.json \
  --atom-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/assets/feeds/esg_news.atom

# 2. In website repo: Commit and push
cd /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io
git add _data/esg_news.json assets/feeds/esg_news.atom
git commit -m "Update ESG news - $(date +%Y-%m-%d)"
git push
```

### Website Integration Files

The export creates data for a Jekyll/al-folio site. Required website files (created separately in website repo):

| File | Purpose |
|------|---------|
| `_data/esg_news.json` | Generated news data |
| `_pages/esg-news.md` | News feed page with filters |
| `_projects/esg-classifier.md` | Project description |
| `_includes/esg_news_card.html` | Card template |
| `assets/js/esg_news_filter.js` | Client-side filtering |
| `assets/feeds/esg_news.atom` | Atom feed for subscribers |

## Labeling Pipeline Changelog

### 2025-12-29: MLOps Improvements - MLflow & Evidently AI

**Change**: Added MLOps module for experiment tracking and production monitoring.

**New features:**
- **MLflow integration**: Automatic logging of training hyperparameters, metrics, and artifacts
- **Evidently AI monitoring**: Drift detection with HTML reports and legacy KS test fallback
- **Webhook alerts**: Slack/Discord notifications when drift is detected
- **Automated monitoring**: Daily cron job and GitHub Actions workflow

**New module**: `src/mlops/`
- `config.py` - Configuration from environment variables
- `tracking.py` - MLflow experiment tracking wrapper
- `monitoring.py` - Evidently-based drift detection
- `reference_data.py` - Reference dataset management
- `alerts.py` - Webhook notifications

**New scripts:**
- `scripts/cron_monitor.sh` - Daily drift monitoring automation

**Files modified:**
- `scripts/train.py` - Added MLflow tracking integration
- `scripts/monitor_drift.py` - Rewritten with Evidently support
- `scripts/setup_cron.sh` - Added `install-monitor` / `remove-monitor` commands
- `.env.example` - Added MLOps environment variables
- `pyproject.toml` - Added mlflow>=2.10 and evidently>=0.4.15 dependencies

**Usage:**
```bash
# Enable features in .env
MLFLOW_ENABLED=true
EVIDENTLY_ENABLED=true
ALERT_WEBHOOK_URL=https://hooks.slack.com/...

# Train with MLflow tracking
uv run python scripts/train.py --classifier fp --verbose

# Run drift monitoring (from production database)
uv run python scripts/monitor_drift.py --classifier fp --from-db --html-report

# Set up daily monitoring
./scripts/setup_cron.sh install-monitor
```

**Graceful degradation**: All features work when disabled - no code changes required.

---

### 2025-12-29: FP Classifier Batch API & Docker Fixes

**Change 1**: Optimized FP classifier pre-filter to use batch API calls instead of per-article calls.

**Before**: N articles → N API calls to FP classifier
**After**: N articles → 1 API call to FP classifier

**Files modified:**
- `src/labeling/pipeline.py` - Added `_run_fp_prefilter_batch()` method, updated `label_articles()` to call batch upfront

**Change 2**: Fixed Docker deployment issues for classifier API.

**Fixes:**
- Removed redundant `HEALTHCHECK` from Dockerfile (docker-compose.yml handles healthchecks)
- Fixed `ModuleNotFoundError` by creating minimal `__init__.py` in container that doesn't import SQLAlchemy models

**Files modified:**
- `Dockerfile` - Removed HEALTHCHECK, changed `COPY __init__.py` to `RUN echo` for minimal init

**Change 3**: Updated default `FP_SKIP_LLM_THRESHOLD` from 0.3 to 0.5 to match the trained model's optimal threshold (0.5356 for 99% recall).

---

### 2025-12-28: FP Classifier Pre-filter Integration

**Change**: Integrated FP (False Positive) classifier as an optional pre-filter in the labeling pipeline to reduce LLM costs by skipping high-confidence false positives before calling Claude.

**New flow:**
```
Articles → FP Classifier → [if probability < threshold] → Mark false_positive, skip LLM
                         → [if probability >= threshold] → Continue to LLM labeling
```

**Configuration:**
- `FP_CLASSIFIER_ENABLED=true` - Enable pre-filter
- `FP_CLASSIFIER_URL=http://localhost:8000` - FP classifier API URL
- `FP_SKIP_LLM_THRESHOLD=0.5` - Skip LLM for articles with <50% sportswear probability (matches trained model threshold)

**Files created:**
- `migrations/002_classifier_predictions.sql` - New table for audit trail
- `src/labeling/classifier_client.py` - HTTP client for classifier APIs

**Files modified:**
- `src/data_collection/models.py` - Added `ClassifierPrediction` model
- `src/labeling/config.py` - Added FP classifier settings
- `src/labeling/pipeline.py` - Added FP pre-filter integration
- `scripts/label_articles.py` - Added FP classifier stats output

**Migration**: Run the SQL migration to create the classifier_predictions table:
```bash
psql $DATABASE_URL -f migrations/002_classifier_predictions.sql
```

**Graceful degradation**: If FP classifier is unavailable or returns an error, the pipeline continues to LLM labeling (no article is lost due to classifier failure).

---

### 2025-12-27: Clarified is_sportswear_brand Semantics

**Change**: Updated `src/labeling/config.py` prompt to clarify that `is_sportswear_brand` is about IDENTITY (does the brand name refer to the sportswear company?), not CONTENT (does the article have ESG content?).

**Why**: Articles about sportswear brand products (sales, reviews, announcements) were incorrectly marked as `false_positive` instead of `skipped`. This matters because:
- `skipped` articles remain available for future labeling (e.g., if we add a "product" category)
- `false_positive` articles are excluded from all future consideration

**Correct classification:**
- `is_sportswear_brand: false` → Brand name refers to something else (Puma animal, Patagonia region)
- `is_sportswear_brand: true` (or omit) → Article is about the sportswear brand, even without ESG content

**Database fixes**:
- Timberland boot sale article: `false_positive` → `skipped`
- Black Diamond headlamp review: `false_positive` → `skipped`

**Also fixed**: Black Diamond Equipment (climbing/outdoor gear) was incorrectly listed in BRAND_NAME_CONFLICTS as a false positive, but it IS the brand we're tracking. Updated to list only true false positives (power company, gemstone, ski run difficulty).

---

### 2025-12-26: Added `skipped_at` Timestamp

**Change**: Added `skipped_at` column to articles table to track when articles are marked as skipped.

**Why**: Previously, skipped articles had NULL `labeled_at` timestamps, making it impossible to determine when they were processed. The new `skipped_at` field enables:
- Tracking when articles were skipped for future relabeling
- Filtering skipped articles by date if labeling criteria change
- Better data lineage for ML training data

**Files changed**:
- `src/data_collection/models.py`: Added `skipped_at` column to Article model
- `src/labeling/database.py`: Updated `update_article_labeling_status` to set `skipped_at`

**Migration**: Run this SQL to add the column to existing databases:
```sql
ALTER TABLE articles ADD COLUMN skipped_at TIMESTAMP WITH TIME ZONE;
```

---

### 2025-12-26: Added Tangential Brand Mention Guidance

**Change**: Updated `src/labeling/config.py` to add guidance for identifying false positives where a brand name correctly refers to the sportswear company, but the article is not primarily about that brand.

**New false positive categories added:**
- Biographical/resume mentions (former executives at other companies)
- Stock/financial articles with no substantive ESG content
- Incidental references in articles about other companies

**Potentially affected historical data:**
- Articles with `labeled` status before 2025-12-26 00:13:57 UTC may include false positives under the new definition
- `labeled` articles: May have ESG labels for articles that aren't primarily about the brand
- `skipped` articles: All 244 have NULL `labeled_at` (timestamp not recorded for skips), so date filtering won't work - review all skipped articles if needed
- Particularly affected: stock price/trading articles for Puma, Anta, 361 Degrees, etc.

**Query to identify articles processed before this change:**
```sql
-- Count by status
SELECT COUNT(*), labeling_status
FROM articles
WHERE labeled_at < '2025-12-26 00:13:57+00'
GROUP BY labeling_status;

-- Find potentially mislabeled stock/financial articles (labeled)
SELECT a.title, a.source_name, bl.brand, a.labeled_at
FROM articles a
JOIN brand_labels bl ON bl.article_id = a.id
WHERE a.labeled_at < '2025-12-26 00:13:57+00'
AND (
    a.title ILIKE '%stock%' OR
    a.title ILIKE '%shares%' OR
    a.title ILIKE '%trading%' OR
    a.title ILIKE '%short interest%'
)
ORDER BY a.labeled_at DESC;

-- Find skipped articles that may be false positives (no date filter - labeled_at is NULL for skips)
SELECT a.title, a.source_name, a.brands_mentioned
FROM articles a
WHERE a.labeling_status = 'skipped'
AND (
    a.title ILIKE '%stock%' OR
    a.title ILIKE '%shares%' OR
    a.title ILIKE '%trading%' OR
    a.title ILIKE '%short interest%' OR
    a.title ILIKE '%former%' OR
    a.title ILIKE '%ex-%' OR
    a.title ILIKE '%appoints%' OR
    a.title ILIKE '%joins%'
)
ORDER BY a.created_at DESC;
```

**To relabel historical articles** (will incur API costs):
```bash
# Reset specific articles to pending and relabel
# First, update labeling_status to 'pending' for target articles
# Then run: uv run python scripts/label_articles.py --batch-size N
```
