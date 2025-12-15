# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESG News Classifier for sportswear brands - a multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for brands like Nike, Adidas, Puma, Under Armour, Lululemon, and Patagonia.

## Commands

```bash
# Setup
uv sync                                    # Install dependencies
cp .env.example .env                       # Create environment file (add API key)
docker compose up -d                       # Start PostgreSQL + pgvector

# Data Collection
python scripts/collect_news.py             # Run full daily collection
python scripts/collect_news.py --dry-run --max-calls 5    # Test without saving
python scripts/collect_news.py --scrape-only              # Only scrape pending articles
python scripts/collect_news.py -v          # Verbose logging
```

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, and API configuration
- `api_client.py` - NewsData.io API wrapper with query generation
- `scraper.py` - Full article text extraction using newspaper4k
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun)
- `collector.py` - Orchestrates API collection + scraping phases

### Database Schema
- **articles**: Stores article metadata from API + full scraped content + future embeddings (pgvector)
- **collection_runs**: Logs each daily collection run with statistics

### Environment Variables
- `NEWSDATA_API_KEY` - Required for API collection
- `DATABASE_URL` - PostgreSQL connection string
- `MAX_API_CALLS_PER_DAY` - Rate limit (default: 200)
- `SCRAPE_DELAY_SECONDS` - Delay between scrape requests (default: 2)

## ESG Category Structure

- **Environmental**: carbon_emissions, waste_management, sustainable_materials
- **Social**: worker_rights, diversity_inclusion, community_engagement
- **Governance**: ethical_sourcing, transparency, board_structure

## Future Phases

### Model Development (Phase 3)
- Traditional ML: TF-IDF + Logistic Regression/Random Forest/XGBoost
- Transformer-Based: Fine-tuned DistilBERT/RoBERTa

### Key Metrics
- Per-category Precision, Recall, F1-score
- Hamming Loss (multi-label specific)
- SHAP values for model explainability
