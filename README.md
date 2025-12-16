# ESG News Classifier for Sportswear Brands

A multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for major sportswear brands including Nike, Adidas, Puma, Under Armour, Lululemon, Patagonia, Columbia Sportswear, New Balance, ASICS, and Reebok.

## Table of Contents

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
- [Environment Variables](#environment-variables)
- [Database Schema](#database-schema)
  - [Articles Table](#articles-table)
  - [Collection Runs Table](#collection-runs-table)
- [Querying the Database](#querying-the-database)
  - [Quick Stats](#quick-stats)
  - [Detailed Queries](#detailed-queries)
  - [Interactive Database Access](#interactive-database-access)
- [ESG Category Structure](#esg-category-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project Roadmap](#project-roadmap)

## Project Structure

```
sportswear-esg-news-classifier/
├── docker-compose.yml          # PostgreSQL + pgvector container configuration
├── pyproject.toml              # Project dependencies and metadata (uv/pip)
├── .env.example                # Environment variable template
├── .env                        # Local environment variables (not committed)
├── logs/                       # Application logs
├── scripts/
│   ├── collect_news.py         # CLI script for data collection
│   ├── gdelt_backfill.py       # Historical backfill script (3 months)
│   ├── cron_collect.sh         # Cron wrapper for NewsData.io collection
│   ├── cron_scrape.sh          # Cron wrapper for GDELT collection
│   └── setup_cron.sh           # User-friendly cron management
├── src/
│   └── data_collection/
│       ├── __init__.py
│       ├── config.py           # Settings, brands, keywords, API configuration
│       ├── api_client.py       # NewsData.io API wrapper with query generation
│       ├── gdelt_client.py     # GDELT DOC 2.0 API wrapper (free, 3 months history)
│       ├── scraper.py          # Full article text extraction (newspaper4k)
│       ├── database.py         # PostgreSQL operations with SQLAlchemy
│       ├── models.py           # SQLAlchemy models (Article, CollectionRun)
│       └── collector.py        # Orchestrates API collection + scraping phases
└── tests/
    ├── conftest.py             # Shared pytest fixtures
    ├── test_api_client.py      # NewsData.io client unit tests
    ├── test_gdelt_client.py    # GDELT client unit tests
    ├── test_collector.py       # Collector unit tests
    └── test_database.py        # Database integration tests
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

# Edit .env and add your NewsData.io API key
# NEWSDATA_API_KEY=your_api_key_here
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

## Testing

The project includes a comprehensive test suite with 79 tests covering the data collection pipeline.

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
| `test_collector.py` | 13 | Deduplication, dry run mode, API limits |
| `test_database.py` | 12 | Upsert operations, queries (requires PostgreSQL) |

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

## Project Roadmap

### Phase 1: Data Collection (Current)
- [x] NewsData.io API integration
- [x] GDELT DOC 2.0 API integration (free, 3 months history)
- [x] Article scraping with newspaper4k
- [x] PostgreSQL + pgvector storage
- [x] Automated cron scheduling (8x daily: 4 NewsData + 4 GDELT)
- [x] Historical backfill script for GDELT
- [ ] Target: 1,000-2,000 articles over 10-14 days

### Phase 2: Data Labeling & Preprocessing
- [ ] Create labeling guidelines
- [ ] Manual labeling of 500-800 articles
- [ ] Text preprocessing pipeline
- [ ] Multi-label dataset preparation

### Phase 3: Model Development
- [ ] Baseline: TF-IDF + Logistic Regression/Random Forest/XGBoost
- [ ] Advanced: Fine-tuned DistilBERT/RoBERTa
- [ ] Multi-label classification evaluation

### Phase 4: Evaluation & Explainability
- [ ] Per-category Precision, Recall, F1-score
- [ ] Hamming Loss (multi-label specific)
- [ ] SHAP values for model explainability

### Phase 5: Deployment
- [ ] FastAPI/Flask backend
- [ ] Streamlit dashboard
- [ ] Docker containerization
- [ ] Cloud deployment (Render/Railway/HuggingFace Spaces)
