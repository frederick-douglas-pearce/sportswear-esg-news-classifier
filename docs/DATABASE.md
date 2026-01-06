# Database Reference

This document provides detailed information about the PostgreSQL database schema, queries, and backup procedures.

> **Quick Start:** For a high-level overview, see the [main README](../README.md#database).

## Schema Overview

The database uses PostgreSQL with the pgvector extension for embedding storage.

## Articles Table

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
| `labeling_status` | String | pending/labeled/skipped/false_positive/unlabelable |
| `labeled_at` | DateTime | Timestamp when article was labeled |
| `skipped_at` | DateTime | Timestamp when article was skipped (for future relabeling) |
| `embedding` | Vector(1536) | For future semantic search |

## Collection Runs Table

Logs each daily collection run with statistics for monitoring.

## Labeling Tables

The labeling pipeline adds several new tables:

### article_chunks

Chunked article text for embeddings and evidence linking:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `article_id` | UUID | Foreign key to articles |
| `chunk_index` | Integer | Order within article |
| `chunk_text` | Text | Chunk content |
| `char_start`, `char_end` | Integer | Position in full_content |
| `token_count` | Integer | Token count |
| `embedding` | Vector(1536) | OpenAI embedding |

### brand_labels

Per-brand ESG labels with sentiment:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `article_id` | UUID | Foreign key to articles |
| `brand` | String | Brand name (Nike, Adidas, etc.) |
| `environmental`, `social`, `governance`, `digital_transformation` | Boolean | Category flags |
| `environmental_sentiment`, etc. | SmallInt | Sentiment (-1, 0, 1, or NULL) |
| `confidence_score` | Float | LLM confidence (0-1) |
| `labeled_by` | String | Source (claude-sonnet, human, classifier) |
| `model_version` | String | Model identifier |

### label_evidence

Supporting text excerpts:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `brand_label_id` | UUID | Foreign key to brand_labels |
| `chunk_id` | UUID | Foreign key to article_chunks (nullable) |
| `category` | String | ESG category |
| `excerpt` | Text | Evidence quote |
| `relevance_score` | Float | Match confidence |

### labeling_runs

Tracks labeling batches with statistics and cost estimates.

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

### Labeling Queries

```bash
# Labeling statistics
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT labeling_status, COUNT(*)
FROM articles
GROUP BY labeling_status;"

# Brand labels by category
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT brand,
       COUNT(*) as total_labels,
       SUM(CASE WHEN environmental THEN 1 ELSE 0 END) as environmental,
       SUM(CASE WHEN social THEN 1 ELSE 0 END) as social,
       SUM(CASE WHEN governance THEN 1 ELSE 0 END) as governance,
       SUM(CASE WHEN digital_transformation THEN 1 ELSE 0 END) as digital
FROM brand_labels
GROUP BY brand
ORDER BY total_labels DESC;"

# Recent labeling runs
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT started_at::date, status, articles_processed, brands_labeled,
       ROUND(estimated_cost_usd::numeric, 4) as cost
FROM labeling_runs
ORDER BY started_at DESC
LIMIT 5;"

# Evidence excerpts for a brand
docker exec esg_news_db psql -U postgres -d esg_news -c "
SELECT le.category, LEFT(le.excerpt, 80) as evidence, le.relevance_score
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
WHERE bl.brand = 'Nike'
LIMIT 10;"
```

### Interactive Database Access

```bash
# Open psql shell for interactive queries
docker exec -it esg_news_db psql -U postgres -d esg_news
```

## Database Backup

The project includes automated backup infrastructure to protect collected and labeled data.

### Backup Commands

```bash
# Create a new backup (compressed, ~25MB)
./scripts/backup_db.sh backup

# List all available backups
./scripts/backup_db.sh list

# Show backup status and disk usage
./scripts/backup_db.sh status

# Restore from a backup file
./scripts/backup_db.sh restore --file backups/daily/esg_news_YYYYMMDD_HHMMSS.sql.gz

# Manually rotate old backups
./scripts/backup_db.sh rotate
```

### Retention Policy

| Type | Retention | Created |
|------|-----------|---------|
| Daily | 7 days | Every backup |
| Weekly | 4 weeks | Sundays |
| Monthly | 3 months | 1st of month |

After 3 months: ~14 backups, ~350MB disk space.

### Automated Backups

```bash
# Enable daily backups at 2am
./scripts/setup_cron.sh install-backup

# Check backup cron status
./scripts/setup_cron.sh status

# Remove backup cron job
./scripts/setup_cron.sh remove-backup
```

### Restore Process

The restore command includes safety features:
1. Creates a pre-restore backup automatically
2. Requires explicit "yes" confirmation
3. Recreates the database with pgvector extension
4. Verifies record counts after restore

## Environment Variables

### Database Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `POSTGRES_DB` | Database name | `esg_news` |

### Data Collection

| Variable | Description | Default |
|----------|-------------|---------|
| `NEWSDATA_API_KEY` | NewsData.io API key | Required (for NewsData) |
| `MAX_API_CALLS_PER_DAY` | API rate limit | `200` |
| `SCRAPE_DELAY_SECONDS` | Delay between scrape requests | `2` |
| `GDELT_TIMESPAN` | Default GDELT time window | `3m` |
| `GDELT_MAX_RECORDS` | Max records per GDELT query | `250` |

### LLM Labeling

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude labeling | Required |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `LABELING_MODEL` | Claude model for labeling | `claude-sonnet-4-20250514` |
| `EMBEDDING_MODEL` | OpenAI model for embeddings | `text-embedding-3-small` |
| `LABELING_BATCH_SIZE` | Default articles per labeling batch | `10` |
| `TARGET_CHUNK_TOKENS` | Target tokens per chunk | `500` |
| `MAX_CHUNK_TOKENS` | Maximum tokens per chunk | `800` |
| `MIN_CHUNK_TOKENS` | Minimum tokens per chunk | `100` |

### FP Classifier Pre-filter

| Variable | Description | Default |
|----------|-------------|---------|
| `FP_CLASSIFIER_ENABLED` | Enable FP classifier pre-filter | `false` |
| `FP_CLASSIFIER_URL` | FP classifier API URL | `http://localhost:8000` |
| `FP_SKIP_LLM_THRESHOLD` | Skip LLM for articles below this probability | `0.5` |
| `FP_CLASSIFIER_TIMEOUT` | FP classifier API timeout (seconds) | `30.0` |
