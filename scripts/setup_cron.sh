#!/bin/bash
# Helper script to set up or remove the cron job for ESG news collection
#
# Usage:
#   ./scripts/setup_cron.sh install   # Add cron job (runs every 6 hours)
#   ./scripts/setup_cron.sh remove    # Remove cron job
#   ./scripts/setup_cron.sh status    # Show current cron jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_SCRIPT="$SCRIPT_DIR/cron_collect.sh"

# Cron schedule: runs at midnight, 6am, noon, 6pm
CRON_SCHEDULE="0 0,6,12,18 * * *"
CRON_ENTRY="$CRON_SCHEDULE $CRON_SCRIPT"
CRON_COMMENT="# ESG News Collector - runs 4 times daily"

case "$1" in
    install)
        # Check if already installed
        if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
            echo "Cron job already installed."
            echo ""
            echo "Current schedule:"
            crontab -l | grep -A1 "ESG News Collector"
        else
            # Add to crontab
            (crontab -l 2>/dev/null || true; echo "$CRON_COMMENT"; echo "$CRON_ENTRY") | crontab -
            echo "✓ Cron job installed successfully!"
            echo ""
            echo "Schedule: Every 6 hours (midnight, 6am, noon, 6pm)"
            echo "Script: $CRON_SCRIPT"
            echo "Logs: $SCRIPT_DIR/../logs/collection_YYYYMMDD.log"
            echo ""
            echo "To view: crontab -l"
            echo "To remove: $0 remove"
        fi
        ;;
    remove)
        if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
            crontab -l | grep -v "$CRON_SCRIPT" | grep -v "ESG News Collector" | crontab -
            echo "✓ Cron job removed."
        else
            echo "No cron job found to remove."
        fi
        ;;
    status)
        echo "Current crontab entries:"
        echo ""
        if crontab -l 2>/dev/null | grep -q "$CRON_SCRIPT"; then
            crontab -l | grep -A1 "ESG News Collector"
            echo ""
            echo "Status: ACTIVE"
        else
            crontab -l 2>/dev/null || echo "(no crontab)"
            echo ""
            echo "Status: NOT INSTALLED"
            echo "Run '$0 install' to set up the cron job."
        fi
        ;;
    *)
        echo "ESG News Collector - Cron Setup"
        echo ""
        echo "Usage: $0 {install|remove|status}"
        echo ""
        echo "Commands:"
        echo "  install  - Add cron job (runs every 6 hours)"
        echo "  remove   - Remove cron job"
        echo "  status   - Show current cron status"
        exit 1
        ;;
esac
