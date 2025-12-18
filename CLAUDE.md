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
- Export labeled data for training
- Traditional ML: TF-IDF + Logistic Regression/Random Forest/XGBoost
- Transformer-Based: Fine-tuned DistilBERT/RoBERTa

### Key Metrics
- Per-category Precision, Recall, F1-score
- Hamming Loss (multi-label specific)
- SHAP values for model explainability
