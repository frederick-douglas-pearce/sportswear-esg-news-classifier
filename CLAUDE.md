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

# ML Classifier Notebooks (run from project root)
# Note: Requires jupyter notebook to view/run interactively
# The notebook can also be executed as Python scripts using the fp1_nb module

# Testing
uv run pytest                              # Run all tests (190 tests)
uv run pytest -v                           # Run with verbose output
uv run pytest --cov=src                    # Run with coverage report
RUN_DB_TESTS=1 uv run pytest tests/test_database.py  # Run database tests (requires PostgreSQL)

# Scheduled Collection (cron)
./scripts/setup_cron.sh install            # Set up both cron jobs
./scripts/setup_cron.sh status             # Check cron status
./scripts/setup_cron.sh remove             # Remove both cron jobs
./scripts/setup_cron.sh install-scrape     # Add GDELT collection job
tail -f logs/collection_$(date +%Y%m%d).log  # View NewsData logs
tail -f logs/gdelt_$(date +%Y%m%d).log       # View GDELT logs
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
- `evidence_matcher.py` - Links evidence excerpts to chunks via exact/fuzzy/embedding similarity
- `database.py` - Labeling-specific DB operations
- `pipeline.py` - Orchestrates chunking → embedding → labeling → evidence matching

### Scripts (`scripts/`)
- `collect_news.py` - CLI for NewsData.io/GDELT data collection
- `label_articles.py` - CLI for LLM-based article labeling
- `export_training_data.py` - Export labeled data for ML training (JSONL format)
- `gdelt_backfill.py` - Historical backfill script (3 months in weekly batches)
- `cleanup_non_english.py` - Remove non-English articles from database
- `cleanup_false_positives.py` - Identify/remove false positive brand matches
- `cron_collect.sh` - NewsData.io collection + scraping (runs 4x daily at 12am, 6am, 12pm, 6pm)
- `cron_scrape.sh` - GDELT collection + scraping (runs 4x daily at 3am, 9am, 3pm, 9pm)
- `setup_cron.sh` - Install/remove/status commands for cron management

### ML Classifier Notebooks (`notebooks/`)

**False Positive Classifier Pipeline (3 notebooks):**
- `fp1_EDA_FE.ipynb` - EDA & Feature Engineering (exports transformer)
- `fp2_model_selection_tuning.ipynb` - Model selection & hyperparameter tuning
- `fp3_model_evaluation_deployment.ipynb` - Test evaluation & deployment export

**Best Model:** Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 98.8%)

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

### Test Suite (`tests/`) - 190 tests, 72% coverage
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

### Database Schema
- **articles**: Stores article metadata from API + full scraped content + labeling_status
- **collection_runs**: Logs each daily collection run with statistics
- **article_chunks**: Chunked article text with embeddings (pgvector) for evidence linking
- **brand_labels**: Per-brand ESG labels with sentiment (-1/0/+1) and confidence
- **label_evidence**: Supporting excerpts linked to chunks via similarity matching
- **labeling_runs**: Tracks labeling batches with cost estimates

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

2. **ESG Pre-filter Classifier** - Quickly identify if an article contains ESG content before detailed classification. Binary classification using `--dataset esg-prefilter` export.

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
- Export labeled data for training (993 FP, 856 ESG-prefilter, 554 ESG-labels records)
- False Positive Classifier: 3-notebook pipeline complete ✅
  - Random Forest with sentence-transformer + NER features (Test F2: 0.974, Recall: 98.8%)
  - Threshold optimized for 98% target recall
- ESG Pre-filter Classifier: Planned
- ESG Multi-label Classifier: Planned
- Advanced: Fine-tuned DistilBERT/RoBERTa (future)

### Key Metrics
- Per-category Precision, Recall, F1-score
- Hamming Loss (multi-label specific)
- SHAP values for model explainability
