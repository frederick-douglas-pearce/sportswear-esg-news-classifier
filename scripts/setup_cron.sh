#!/bin/bash
# Helper script to set up or remove cron jobs for ESG news collection
#
# Usage:
#   ./scripts/setup_cron.sh install      # Add both cron jobs
#   ./scripts/setup_cron.sh remove       # Remove both cron jobs
#   ./scripts/setup_cron.sh status       # Show current cron status
#   ./scripts/setup_cron.sh install-collect   # Add collection job only
#   ./scripts/setup_cron.sh install-scrape    # Add scrape job only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_SCRIPT="$SCRIPT_DIR/cron_collect.sh"
SCRAPE_SCRIPT="$SCRIPT_DIR/cron_scrape.sh"

# Collection: runs at midnight, 6am, noon, 6pm (4x daily)
COLLECT_SCHEDULE="0 0,6,12,18 * * *"
COLLECT_ENTRY="$COLLECT_SCHEDULE $COLLECT_SCRIPT"
COLLECT_COMMENT="# ESG News Collector - API fetch + scrape (4x daily)"

# GDELT collection: runs at 3am, 9am, 3pm, 9pm (4x daily, offset from NewsData collection)
SCRAPE_SCHEDULE="0 3,9,15,21 * * *"
SCRAPE_ENTRY="$SCRAPE_SCHEDULE $SCRAPE_SCRIPT"
SCRAPE_COMMENT="# ESG News GDELT - collect from GDELT + scrape (4x daily)"

install_collect() {
    if crontab -l 2>/dev/null | grep -q "$COLLECT_SCRIPT"; then
        echo "Collection job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$COLLECT_COMMENT"; echo "$COLLECT_ENTRY") | crontab -
        echo "✓ Collection job installed (midnight, 6am, noon, 6pm)"
    fi
}

install_scrape() {
    if crontab -l 2>/dev/null | grep -q "$SCRAPE_SCRIPT"; then
        echo "GDELT job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$SCRAPE_COMMENT"; echo "$SCRAPE_ENTRY") | crontab -
        echo "✓ GDELT job installed (3am, 9am, 3pm, 9pm)"
    fi
}

remove_collect() {
    if crontab -l 2>/dev/null | grep -q "$COLLECT_SCRIPT"; then
        crontab -l | grep -v "$COLLECT_SCRIPT" | grep -v "ESG News Collector" | crontab -
        echo "✓ Collection job removed."
    else
        echo "No collection job found."
    fi
}

remove_scrape() {
    if crontab -l 2>/dev/null | grep -q "$SCRAPE_SCRIPT"; then
        crontab -l | grep -v "$SCRAPE_SCRIPT" | grep -v "ESG News GDELT" | grep -v "ESG News Scraper" | crontab -
        echo "✓ GDELT job removed."
    else
        echo "No GDELT job found."
    fi
}

case "$1" in
    install)
        install_collect
        install_scrape
        echo ""
        echo "Logs:"
        echo "  NewsData:   logs/collection_YYYYMMDD.log"
        echo "  GDELT:      logs/gdelt_YYYYMMDD.log"
        ;;
    install-collect)
        install_collect
        ;;
    install-scrape)
        install_scrape
        ;;
    remove)
        remove_collect
        remove_scrape
        ;;
    remove-collect)
        remove_collect
        ;;
    remove-scrape)
        remove_scrape
        ;;
    status)
        echo "ESG News Cron Jobs Status"
        echo "========================="
        echo ""
        if crontab -l 2>/dev/null | grep -q "$COLLECT_SCRIPT"; then
            echo "Collection: ACTIVE (midnight, 6am, noon, 6pm)"
        else
            echo "Collection: NOT INSTALLED"
        fi
        if crontab -l 2>/dev/null | grep -q "$SCRAPE_SCRIPT"; then
            echo "GDELT:      ACTIVE (3am, 9am, 3pm, 9pm)"
        else
            echo "GDELT:      NOT INSTALLED"
        fi
        echo ""
        echo "Current crontab:"
        crontab -l 2>/dev/null | grep -E "(ESG|cron_)" || echo "(no ESG jobs)"
        ;;
    *)
        echo "ESG News Collector - Cron Setup"
        echo ""
        echo "Usage: $0 {install|remove|status}"
        echo ""
        echo "Commands:"
        echo "  install          - Add both cron jobs"
        echo "  remove           - Remove both cron jobs"
        echo "  status           - Show current cron status"
        echo "  install-collect  - Add collection job only"
        echo "  install-scrape   - Add scrape job only"
        echo "  remove-collect   - Remove collection job only"
        echo "  remove-scrape    - Remove scrape job only"
        echo ""
        echo "Schedule:"
        echo "  NewsData (API + scrape):   midnight, 6am, noon, 6pm"
        echo "  GDELT (free API + scrape): 3am, 9am, 3pm, 9pm"
        exit 1
        ;;
esac
