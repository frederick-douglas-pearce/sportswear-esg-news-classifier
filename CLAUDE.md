# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESG News Classifier for sportswear brands - a multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for brands including Nike, Adidas, Puma, Under Armour, Lululemon, Patagonia, Columbia Sportswear, New Balance, ASICS, and Reebok.

## Commands

```bash
# Setup
uv sync                                    # Install dependencies
uv sync --extra dev                        # Install dev dependencies (testing)
cp .env.example .env                       # Create environment file (add API key)
docker compose up -d                       # Start PostgreSQL + pgvector

# Data Collection (use uv run to execute in venv)
uv run python scripts/collect_news.py             # Run full daily collection
uv run python scripts/collect_news.py --dry-run --max-calls 5    # Test without saving
uv run python scripts/collect_news.py --scrape-only              # Only scrape pending articles
uv run python scripts/collect_news.py -v          # Verbose logging

# Testing
uv run pytest                              # Run all tests (43 tests)
uv run pytest -v                           # Run with verbose output
uv run pytest --cov=src                    # Run with coverage report
RUN_DB_TESTS=1 uv run pytest tests/test_database.py  # Run database tests (requires PostgreSQL)

# Scheduled Collection (cron)
./scripts/setup_cron.sh install            # Set up cron job (4x daily)
./scripts/setup_cron.sh status             # Check cron status
./scripts/setup_cron.sh remove             # Remove cron job
tail -f logs/collection_$(date +%Y%m%d).log  # View today's logs
```

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, and API configuration
- `api_client.py` - NewsData.io API wrapper with OR-grouped query generation
- `scraper.py` - Full article text extraction using newspaper4k
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun)
- `collector.py` - Orchestrates API collection + scraping with in-memory deduplication

### Cron Scripts (`scripts/`)
- `cron_collect.sh` - Wrapper script for cron with daily log rotation
- `setup_cron.sh` - Install/remove/status commands for cron management

### Test Suite (`tests/`)
- `conftest.py` - Shared pytest fixtures
- `test_api_client.py` - Brand extraction, article parsing, query generation (21 tests)
- `test_collector.py` - Deduplication, dry run mode, API limits (10 tests)
- `test_database.py` - Upsert operations, queries (12 tests, requires PostgreSQL)

### Database Schema
- **articles**: Stores article metadata from API + full scraped content + future embeddings (pgvector)
- **collection_runs**: Logs each daily collection run with statistics

### Environment Variables
- `NEWSDATA_API_KEY` - Required for API collection
- `DATABASE_URL` - PostgreSQL connection string
- `MAX_API_CALLS_PER_DAY` - Rate limit (default: 200)
- `SCRAPE_DELAY_SECONDS` - Delay between scrape requests (default: 2)

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

## Future Phases

### Model Development (Phase 3)
- Traditional ML: TF-IDF + Logistic Regression/Random Forest/XGBoost
- Transformer-Based: Fine-tuned DistilBERT/RoBERTa

### Key Metrics
- Per-category Precision, Recall, F1-score
- Hamming Loss (multi-label specific)
- SHAP values for model explainability
