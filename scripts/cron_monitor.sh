#!/bin/bash
# Daily monitoring script for ESG classifiers
#
# This script runs drift monitoring for FP and EP classifiers,
# generates reports, and sends alerts if drift is detected.
#
# Usage:
#   ./scripts/cron_monitor.sh                 # Run monitoring for all classifiers
#   ./scripts/cron_monitor.sh --classifier fp # Run for specific classifier only
#
# Environment:
#   EVIDENTLY_ENABLED=true     # Enable Evidently AI reports
#   ALERT_WEBHOOK_URL=...      # Slack/Discord webhook for alerts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs/monitoring"
REPORT_DIR="$PROJECT_DIR/reports/monitoring"
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

run_monitoring() {
    local classifier=$1
    local LOG_FILE="$LOG_DIR/${classifier}_monitoring_${DATE}.log"

    log "Starting drift monitoring for $classifier classifier"

    cd "$PROJECT_DIR"

    # Activate virtual environment if using uv
    if [ -d ".venv" ]; then
        source .venv/bin/activate 2>/dev/null || true
    fi

    # Run drift monitoring
    # --alert flag sends webhook notification if drift detected
    # --html-report generates Evidently report if enabled
    local ARGS="--classifier $classifier --days 7 --verbose"

    if [ "${EVIDENTLY_ENABLED:-false}" = "true" ]; then
        ARGS="$ARGS --html-report"
    fi

    if [ -n "$ALERT_WEBHOOK_URL" ]; then
        ARGS="$ARGS --alert"
    fi

    # Run monitoring script
    if uv run python scripts/monitor_drift.py $ARGS >> "$LOG_FILE" 2>&1; then
        log "Drift monitoring completed successfully for $classifier"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 1 ]; then
            log "WARNING: Drift detected for $classifier classifier"
        else
            log "ERROR: Drift monitoring failed for $classifier (exit code: $exit_code)"
        fi
        return $exit_code
    fi
}

# Parse arguments
CLASSIFIER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier|-c)
            CLASSIFIER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run monitoring
if [ -n "$CLASSIFIER" ]; then
    # Run for specific classifier
    run_monitoring "$CLASSIFIER"
else
    # Run for all classifiers
    echo "Running drift monitoring for all classifiers..."
    echo "Log files: $LOG_DIR/"
    echo ""

    # Track overall status
    overall_status=0

    for classifier in fp ep; do
        if ! run_monitoring "$classifier"; then
            overall_status=1
        fi
        echo ""
    done

    echo "Monitoring complete. Check logs in $LOG_DIR/"
    exit $overall_status
fi
