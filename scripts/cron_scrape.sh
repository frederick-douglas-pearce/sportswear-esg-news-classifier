#!/bin/bash
# Cron wrapper for GDELT news collection (free API, no key required)
# Run this script via cron to collect recent news from GDELT and scrape content
#
# Schedule: runs at 3am, 9am, 3pm, 9pm (4x daily)
# Uses 6-hour timespan to capture news since last run

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Log file with date
LOG_FILE="$PROJECT_DIR/logs/gdelt_$(date +%Y%m%d).log"

# Run GDELT collection with 6-hour window
echo "========================================" >> "$LOG_FILE"
echo "GDELT collection started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Use uv to run GDELT collection with 6-hour timespan
~/.local/bin/uv run python scripts/collect_news.py \
    --source gdelt \
    --timespan 6h \
    --max-calls 50 \
    --scrape-limit 100 \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "GDELT collection finished: $(date), exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
