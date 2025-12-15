#!/bin/bash
# Cron wrapper for ESG news collection
# Run this script via cron to collect news articles on a schedule
#
# Example crontab entry (runs every 6 hours):
#   0 */6 * * * /path/to/sportswear-esg-news-classifier/scripts/cron_collect.sh
#
# To edit your crontab: crontab -e
# To view your crontab: crontab -l

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Log file with date
LOG_FILE="$PROJECT_DIR/logs/collection_$(date +%Y%m%d).log"

# Run the collection script
# Adjust --max-calls and --scrape-limit as needed
echo "========================================" >> "$LOG_FILE"
echo "Collection started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Use uv to run the script in the project's virtual environment
~/.local/bin/uv run python scripts/collect_news.py \
    --max-calls 50 \
    --scrape-limit 100 \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Collection finished: $(date), exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
