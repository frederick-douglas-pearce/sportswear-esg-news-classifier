# ESG News Classifier for Sportswear Brands

A multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for major sportswear brands including Nike, Adidas, Puma, Under Armour, Lululemon, and Patagonia.

## Project Structure

```
sportswear-esg-news-classifier/
├── docker-compose.yml          # PostgreSQL + pgvector container configuration
├── pyproject.toml              # Project dependencies and metadata (uv/pip)
├── .env.example                # Environment variable template
├── .env                        # Local environment variables (not committed)
├── logs/                       # Application logs
├── scripts/
│   └── collect_news.py         # CLI script for data collection
└── src/
    └── data_collection/
        ├── __init__.py
        ├── config.py           # Settings, brands, keywords, API configuration
        ├── api_client.py       # NewsData.io API wrapper with query generation
        ├── scraper.py          # Full article text extraction (newspaper4k)
        ├── database.py         # PostgreSQL operations with SQLAlchemy
        ├── models.py           # SQLAlchemy models (Article, CollectionRun)
        └── collector.py        # Orchestrates API collection + scraping phases
```

## Quick Start

### 1. Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose
- NewsData.io API key ([get free tier here](https://newsdata.io/register))

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sportswear-esg-news-classifier

# Install dependencies with uv
uv sync

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
# Run full daily collection (API fetch + scraping)
uv run python scripts/collect_news.py

# With custom limits
uv run python scripts/collect_news.py --max-calls 100 --scrape-limit 50

# Verbose mode for monitoring
uv run python scripts/collect_news.py -v
```

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
| `--dry-run` | Don't save to database, just show what would be done | False |
| `--max-calls N` | Maximum API calls to make | 200 |
| `--scrape-only` | Only scrape pending articles, skip API collection | False |
| `--scrape-limit N` | Maximum articles to scrape | 100 |
| `-v, --verbose` | Enable verbose/debug logging | False |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEWSDATA_API_KEY` | Your NewsData.io API key | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5434/esg_news` |
| `POSTGRES_USER` | Database username | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `POSTGRES_DB` | Database name | `esg_news` |
| `MAX_API_CALLS_PER_DAY` | API rate limit | `200` |
| `SCRAPE_DELAY_SECONDS` | Delay between scrape requests | `2` |

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
- [x] Article scraping with newspaper4k
- [x] PostgreSQL + pgvector storage
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
