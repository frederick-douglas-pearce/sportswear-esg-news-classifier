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
MONITOR_SCRIPT="$SCRIPT_DIR/cron_monitor.sh"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_db.sh"

# Collection: runs at midnight, 6am, noon, 6pm (4x daily)
COLLECT_SCHEDULE="0 0,6,12,18 * * *"
COLLECT_ENTRY="$COLLECT_SCHEDULE $COLLECT_SCRIPT"
COLLECT_COMMENT="# ESG News Collector - API fetch + scrape (4x daily)"

# GDELT collection: runs at 3am, 9am, 3pm, 9pm (4x daily, offset from NewsData collection)
SCRAPE_SCHEDULE="0 3,9,15,21 * * *"
SCRAPE_ENTRY="$SCRAPE_SCHEDULE $SCRAPE_SCRIPT"
SCRAPE_COMMENT="# ESG News GDELT - collect from GDELT + scrape (4x daily)"

# Monitoring: runs daily at 6am UTC
MONITOR_SCHEDULE="0 6 * * *"
MONITOR_ENTRY="$MONITOR_SCHEDULE $MONITOR_SCRIPT"
MONITOR_COMMENT="# ESG Classifier Monitoring - drift detection (daily)"

# Database backup: runs daily at 2am (before monitoring)
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_ENTRY="$BACKUP_SCHEDULE $BACKUP_SCRIPT backup"
BACKUP_COMMENT="# ESG Database Backup - daily with rotation"

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

install_monitor() {
    if crontab -l 2>/dev/null | grep -q "$MONITOR_SCRIPT"; then
        echo "Monitoring job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$MONITOR_COMMENT"; echo "$MONITOR_ENTRY") | crontab -
        echo "✓ Monitoring job installed (daily at 6am UTC)"
    fi
}

remove_monitor() {
    if crontab -l 2>/dev/null | grep -q "$MONITOR_SCRIPT"; then
        crontab -l | grep -v "$MONITOR_SCRIPT" | grep -v "ESG Classifier Monitoring" | crontab -
        echo "✓ Monitoring job removed."
    else
        echo "No monitoring job found."
    fi
}

install_backup() {
    if crontab -l 2>/dev/null | grep -q "$BACKUP_SCRIPT"; then
        echo "Backup job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$BACKUP_COMMENT"; echo "$BACKUP_ENTRY") | crontab -
        echo "✓ Backup job installed (daily at 2am)"
    fi
}

remove_backup() {
    if crontab -l 2>/dev/null | grep -q "$BACKUP_SCRIPT"; then
        crontab -l | grep -v "$BACKUP_SCRIPT" | grep -v "ESG Database Backup" | crontab -
        echo "✓ Backup job removed."
    else
        echo "No backup job found."
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
    install-monitor)
        install_monitor
        ;;
    install-backup)
        install_backup
        ;;
    remove)
        remove_collect
        remove_scrape
        remove_monitor
        remove_backup
        ;;
    remove-collect)
        remove_collect
        ;;
    remove-scrape)
        remove_scrape
        ;;
    remove-monitor)
        remove_monitor
        ;;
    remove-backup)
        remove_backup
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
        if crontab -l 2>/dev/null | grep -q "$MONITOR_SCRIPT"; then
            echo "Monitoring: ACTIVE (daily at 6am UTC)"
        else
            echo "Monitoring: NOT INSTALLED"
        fi
        if crontab -l 2>/dev/null | grep -q "$BACKUP_SCRIPT"; then
            echo "Backup:     ACTIVE (daily at 2am)"
        else
            echo "Backup:     NOT INSTALLED"
        fi
        echo ""
        echo "Current crontab:"
        crontab -l 2>/dev/null | grep -E "(ESG|cron_|backup_)" || echo "(no ESG jobs)"
        ;;
    *)
        echo "ESG News Collector - Cron Setup"
        echo ""
        echo "Usage: $0 {install|remove|status}"
        echo ""
        echo "Commands:"
        echo "  install          - Add collection cron jobs"
        echo "  remove           - Remove all cron jobs"
        echo "  status           - Show current cron status"
        echo "  install-collect  - Add NewsData collection job"
        echo "  install-scrape   - Add GDELT collection job"
        echo "  install-monitor  - Add monitoring job"
        echo "  install-backup   - Add database backup job"
        echo "  remove-collect   - Remove collection job"
        echo "  remove-scrape    - Remove scrape job"
        echo "  remove-monitor   - Remove monitoring job"
        echo "  remove-backup    - Remove backup job"
        echo ""
        echo "Schedule:"
        echo "  NewsData (API + scrape):   midnight, 6am, noon, 6pm"
        echo "  GDELT (free API + scrape): 3am, 9am, 3pm, 9pm"
        echo "  Backup (daily rotation):   2am"
        echo "  Monitoring (drift check):  6am UTC daily"
        exit 1
        ;;
esac
