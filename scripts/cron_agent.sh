#!/bin/bash
# Cron wrapper script for agent workflows
#
# Usage:
#   ./scripts/cron_agent.sh daily_labeling
#   ./scripts/cron_agent.sh website_export
#
# Logs output to logs/agent/cron_<workflow>_YYYYMMDD.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WORKFLOW="${1:-daily_labeling}"
LOG_DATE=$(date +%Y%m%d)
LOG_DIR="$PROJECT_DIR/logs/agent"
LOG_FILE="$LOG_DIR/cron_${WORKFLOW}_${LOG_DATE}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "========================================" >> "$LOG_FILE"
echo "Starting $WORKFLOW workflow: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$PROJECT_DIR"

# Run agent workflow
~/.local/bin/uv run python -m src.agent run "$WORKFLOW" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "Finished: $(date)" >> "$LOG_FILE"
echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

exit $EXIT_CODE
