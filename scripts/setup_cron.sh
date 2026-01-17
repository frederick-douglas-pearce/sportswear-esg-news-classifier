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
AGENT_SCRIPT="$SCRIPT_DIR/cron_agent.sh"

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

# Agent daily labeling: runs at 6:30am (after collection completes)
AGENT_LABELING_SCHEDULE="30 6 * * *"
AGENT_LABELING_ENTRY="$AGENT_LABELING_SCHEDULE $AGENT_SCRIPT daily_labeling"
AGENT_LABELING_COMMENT="# ESG Agent - daily labeling workflow"

# Agent website export: runs at 7am (after labeling completes)
AGENT_EXPORT_SCHEDULE="0 7 * * *"
AGENT_EXPORT_ENTRY="$AGENT_EXPORT_SCHEDULE $AGENT_SCRIPT website_export"
AGENT_EXPORT_COMMENT="# ESG Agent - website feed export"

# Agent drift monitoring: runs at 5:30am (before labeling)
AGENT_DRIFT_SCHEDULE="30 5 * * *"
AGENT_DRIFT_ENTRY="$AGENT_DRIFT_SCHEDULE $AGENT_SCRIPT drift_monitoring"
AGENT_DRIFT_COMMENT="# ESG Agent - drift monitoring workflow"

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

install_agent_labeling() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT daily_labeling"; then
        echo "Agent labeling job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$AGENT_LABELING_COMMENT"; echo "$AGENT_LABELING_ENTRY") | crontab -
        echo "✓ Agent labeling job installed (daily at 6:30am)"
    fi
}

install_agent_export() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT website_export"; then
        echo "Agent export job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$AGENT_EXPORT_COMMENT"; echo "$AGENT_EXPORT_ENTRY") | crontab -
        echo "✓ Agent export job installed (daily at 7am)"
    fi
}

remove_agent_labeling() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT daily_labeling"; then
        crontab -l | grep -v "$AGENT_SCRIPT daily_labeling" | grep -v "ESG Agent - daily labeling" | crontab -
        echo "✓ Agent labeling job removed."
    else
        echo "No agent labeling job found."
    fi
}

remove_agent_export() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT website_export"; then
        crontab -l | grep -v "$AGENT_SCRIPT website_export" | grep -v "ESG Agent - website" | crontab -
        echo "✓ Agent export job removed."
    else
        echo "No agent export job found."
    fi
}

install_agent_drift() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT drift_monitoring"; then
        echo "Agent drift job already installed."
    else
        (crontab -l 2>/dev/null || true; echo "$AGENT_DRIFT_COMMENT"; echo "$AGENT_DRIFT_ENTRY") | crontab -
        echo "✓ Agent drift job installed (daily at 5:30am)"
    fi
}

remove_agent_drift() {
    if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT drift_monitoring"; then
        crontab -l | grep -v "$AGENT_SCRIPT drift_monitoring" | grep -v "ESG Agent - drift" | crontab -
        echo "✓ Agent drift job removed."
    else
        echo "No agent drift job found."
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
    install-agent-labeling)
        install_agent_labeling
        ;;
    install-agent-export)
        install_agent_export
        ;;
    install-agent-drift)
        install_agent_drift
        ;;
    install-agent)
        install_agent_drift
        install_agent_labeling
        install_agent_export
        echo ""
        echo "Agent logs: logs/agent/cron_<workflow>_YYYYMMDD.log"
        ;;
    remove)
        remove_collect
        remove_scrape
        remove_monitor
        remove_backup
        remove_agent_labeling
        remove_agent_export
        remove_agent_drift
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
    remove-agent-labeling)
        remove_agent_labeling
        ;;
    remove-agent-export)
        remove_agent_export
        ;;
    remove-agent-drift)
        remove_agent_drift
        ;;
    remove-agent)
        remove_agent_labeling
        remove_agent_export
        remove_agent_drift
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
        if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT daily_labeling"; then
            echo "Agent Lab:  ACTIVE (daily at 6:30am)"
        else
            echo "Agent Lab:  NOT INSTALLED"
        fi
        if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT website_export"; then
            echo "Agent Exp:  ACTIVE (daily at 7am)"
        else
            echo "Agent Exp:  NOT INSTALLED"
        fi
        if crontab -l 2>/dev/null | grep -q "$AGENT_SCRIPT drift_monitoring"; then
            echo "Agent Drf:  ACTIVE (daily at 5:30am)"
        else
            echo "Agent Drf:  NOT INSTALLED"
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
        echo "  install-monitor  - Add monitoring job (standalone)"
        echo "  install-backup   - Add database backup job"
        echo "  install-agent    - Add all agent workflow jobs"
        echo "  install-agent-labeling - Add agent labeling job"
        echo "  install-agent-export   - Add agent export job"
        echo "  install-agent-drift    - Add agent drift monitoring job"
        echo "  remove-collect   - Remove collection job"
        echo "  remove-scrape    - Remove scrape job"
        echo "  remove-monitor   - Remove monitoring job"
        echo "  remove-backup    - Remove backup job"
        echo "  remove-agent     - Remove all agent jobs"
        echo "  remove-agent-labeling  - Remove agent labeling job"
        echo "  remove-agent-export    - Remove agent export job"
        echo "  remove-agent-drift     - Remove agent drift job"
        echo ""
        echo "Schedule:"
        echo "  NewsData (API + scrape):   midnight, 6am, noon, 6pm"
        echo "  GDELT (free API + scrape): 3am, 9am, 3pm, 9pm"
        echo "  Backup (daily rotation):   2am"
        echo "  Monitoring (standalone):   6am UTC daily"
        echo "  Agent drift monitoring:    5:30am daily"
        echo "  Agent labeling:            6:30am daily"
        echo "  Agent website export:      7am daily"
        exit 1
        ;;
esac
