#!/bin/bash
# Cron wrapper for scraping pending articles (no API calls)
# Run this script via cron to scrape articles that haven't been processed yet
#
# Example crontab entry (runs every 3 hours, offset from collection):
#   0 2,5,8,11,14,17,20,23 * * * /path/to/sportswear-esg-news-classifier/scripts/cron_scrape.sh

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Log file with date
LOG_FILE="$PROJECT_DIR/logs/scrape_$(date +%Y%m%d).log"

# Run the scrape-only script
echo "========================================" >> "$LOG_FILE"
echo "Scrape started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Use uv to run the script - scrape only, no API calls
~/.local/bin/uv run python scripts/collect_news.py \
    --scrape-only \
    --scrape-limit 150 \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Scrape finished: $(date), exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
