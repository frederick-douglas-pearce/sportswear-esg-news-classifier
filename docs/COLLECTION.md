# News Collection

This document provides detailed instructions for running the news collection pipeline.

> **Quick Start:** For a quick overview, see the [main README](../README.md#news-collection).

## Overview

The collection pipeline gathers ESG-related news articles from two sources:
- **NewsData.io** - Paid API with real-time news (requires API key)
- **GDELT DOC 2.0** - Free API with 3 months of historical data

All commands use `uv run` to execute within the project's virtual environment. Alternatively, you can activate the venv first with `source .venv/bin/activate` and omit `uv run`.

## Testing (Dry Run Mode)

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

## Production Collection

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

## GDELT Historical Backfill

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

## Scheduled Collection (Cron)

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

## Scrape-Only Mode

If you have articles already fetched but not yet scraped:

```bash
# Only scrape pending articles (skip API collection)
uv run python scripts/collect_news.py --scrape-only

# With custom limit
uv run python scripts/collect_news.py --scrape-only --scrape-limit 50
```

## Command-Line Options

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
