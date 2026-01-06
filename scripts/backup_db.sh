#!/bin/bash
# Database backup script for ESG News Classifier
# Creates compressed PostgreSQL dumps with rotation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"
CONTAINER_NAME="${CONTAINER_NAME:-esg_news_db}"
DB_NAME="${POSTGRES_DB:-esg_news}"
DB_USER="${POSTGRES_USER:-postgres}"

# Retention settings
KEEP_DAILY=7      # Keep daily backups for 7 days
KEEP_WEEKLY=4     # Keep weekly backups for 4 weeks
KEEP_MONTHLY=3    # Keep monthly backups for 3 months

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  backup     Create a new backup (default)"
    echo "  restore    Restore from a backup file"
    echo "  list       List available backups"
    echo "  rotate     Clean up old backups based on retention policy"
    echo "  status     Show backup status and disk usage"
    echo ""
    echo "Options:"
    echo "  --file FILE    Specify backup file for restore"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 backup                           # Create new backup"
    echo "  $0 list                             # List all backups"
    echo "  $0 restore --file backups/daily/esg_news_20250105_120000.sql.gz"
    echo "  $0 rotate                           # Clean old backups"
}

check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container '$CONTAINER_NAME' is not running"
        log_info "Start it with: docker compose up -d postgres"
        exit 1
    fi
}

create_backup() {
    check_container

    # Create backup directories
    mkdir -p "$BACKUP_DIR"/{daily,weekly,monthly}

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DATE=$(date +%Y%m%d)
    DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
    DAY_OF_MONTH=$(date +%d)

    BACKUP_FILE="esg_news_${TIMESTAMP}.sql.gz"
    DAILY_PATH="$BACKUP_DIR/daily/$BACKUP_FILE"

    log_info "Creating backup: $BACKUP_FILE"

    # Create the backup using pg_dump through docker
    docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" \
        --format=plain \
        --no-owner \
        --no-privileges \
        | gzip > "$DAILY_PATH"

    if [ $? -eq 0 ]; then
        BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1)
        log_info "Backup created successfully: $DAILY_PATH ($BACKUP_SIZE)"

        # Create weekly backup on Sundays
        if [ "$DAY_OF_WEEK" -eq 7 ]; then
            cp "$DAILY_PATH" "$BACKUP_DIR/weekly/$BACKUP_FILE"
            log_info "Weekly backup created"
        fi

        # Create monthly backup on the 1st
        if [ "$DAY_OF_MONTH" -eq "01" ]; then
            cp "$DAILY_PATH" "$BACKUP_DIR/monthly/$BACKUP_FILE"
            log_info "Monthly backup created"
        fi

        # Run rotation after successful backup
        rotate_backups

        return 0
    else
        log_error "Backup failed!"
        rm -f "$DAILY_PATH"
        return 1
    fi
}

restore_backup() {
    local backup_file="$1"

    if [ -z "$backup_file" ]; then
        log_error "No backup file specified"
        log_info "Use: $0 restore --file <backup_file>"
        list_backups
        exit 1
    fi

    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi

    check_container

    log_warn "This will restore the database from: $backup_file"
    log_warn "Current data will be OVERWRITTEN!"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Restore cancelled"
        exit 0
    fi

    log_info "Creating pre-restore backup..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PRE_RESTORE="$BACKUP_DIR/pre_restore_${TIMESTAMP}.sql.gz"
    docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" | gzip > "$PRE_RESTORE"
    log_info "Pre-restore backup saved to: $PRE_RESTORE"

    log_info "Restoring from: $backup_file"

    # Drop and recreate database
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "CREATE DATABASE ${DB_NAME};"
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

    # Restore from backup
    gunzip -c "$backup_file" | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME"

    if [ $? -eq 0 ]; then
        log_info "Restore completed successfully!"

        # Show record counts
        log_info "Verifying restore..."
        docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c \
            "SELECT 'articles' as table_name, COUNT(*) as count FROM articles
             UNION ALL SELECT 'brand_labels', COUNT(*) FROM brand_labels
             UNION ALL SELECT 'article_chunks', COUNT(*) FROM article_chunks;"
    else
        log_error "Restore failed!"
        log_info "You can restore the pre-restore backup with:"
        log_info "  $0 restore --file $PRE_RESTORE"
        exit 1
    fi
}

list_backups() {
    log_info "Available backups:"
    echo ""

    for period in daily weekly monthly; do
        echo "=== ${period^^} ==="
        if [ -d "$BACKUP_DIR/$period" ] && [ "$(ls -A "$BACKUP_DIR/$period" 2>/dev/null)" ]; then
            ls -lh "$BACKUP_DIR/$period"/*.sql.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        else
            echo "  (no backups)"
        fi
        echo ""
    done
}

rotate_backups() {
    log_info "Rotating old backups..."

    # Rotate daily backups (keep last N days)
    if [ -d "$BACKUP_DIR/daily" ]; then
        find "$BACKUP_DIR/daily" -name "*.sql.gz" -mtime +$KEEP_DAILY -delete 2>/dev/null
        local daily_count=$(ls -1 "$BACKUP_DIR/daily"/*.sql.gz 2>/dev/null | wc -l)
        log_info "Daily backups: $daily_count (keeping last $KEEP_DAILY days)"
    fi

    # Rotate weekly backups (keep last N weeks)
    if [ -d "$BACKUP_DIR/weekly" ]; then
        find "$BACKUP_DIR/weekly" -name "*.sql.gz" -mtime +$((KEEP_WEEKLY * 7)) -delete 2>/dev/null
        local weekly_count=$(ls -1 "$BACKUP_DIR/weekly"/*.sql.gz 2>/dev/null | wc -l)
        log_info "Weekly backups: $weekly_count (keeping last $KEEP_WEEKLY weeks)"
    fi

    # Rotate monthly backups (keep last N months)
    if [ -d "$BACKUP_DIR/monthly" ]; then
        find "$BACKUP_DIR/monthly" -name "*.sql.gz" -mtime +$((KEEP_MONTHLY * 30)) -delete 2>/dev/null
        local monthly_count=$(ls -1 "$BACKUP_DIR/monthly"/*.sql.gz 2>/dev/null | wc -l)
        log_info "Monthly backups: $monthly_count (keeping last $KEEP_MONTHLY months)"
    fi
}

show_status() {
    log_info "Backup Status"
    echo ""

    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log_warn "No backups directory found"
        log_info "Run '$0 backup' to create your first backup"
        return
    fi

    # Count backups
    local daily_count=$(ls -1 "$BACKUP_DIR/daily"/*.sql.gz 2>/dev/null | wc -l)
    local weekly_count=$(ls -1 "$BACKUP_DIR/weekly"/*.sql.gz 2>/dev/null | wc -l)
    local monthly_count=$(ls -1 "$BACKUP_DIR/monthly"/*.sql.gz 2>/dev/null | wc -l)

    echo "Backup counts:"
    echo "  Daily:   $daily_count (keep $KEEP_DAILY days)"
    echo "  Weekly:  $weekly_count (keep $KEEP_WEEKLY weeks)"
    echo "  Monthly: $monthly_count (keep $KEEP_MONTHLY months)"
    echo ""

    # Show disk usage
    local total_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    echo "Total backup size: $total_size"
    echo ""

    # Show most recent backup
    local latest=$(ls -t "$BACKUP_DIR"/daily/*.sql.gz 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local latest_time=$(stat -c %y "$latest" 2>/dev/null | cut -d. -f1)
        echo "Most recent backup: $(basename "$latest")"
        echo "  Created: $latest_time"
    fi

    # Show current database size
    echo ""
    check_container
    local db_size=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | tr -d ' ')
    echo "Current database size: $db_size"
}

# Parse command line arguments
COMMAND="${1:-backup}"
RESTORE_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        backup|restore|list|rotate|status)
            COMMAND="$1"
            shift
            ;;
        --file)
            RESTORE_FILE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Execute command
case $COMMAND in
    backup)
        create_backup
        ;;
    restore)
        restore_backup "$RESTORE_FILE"
        ;;
    list)
        list_backups
        ;;
    rotate)
        rotate_backups
        ;;
    status)
        show_status
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
