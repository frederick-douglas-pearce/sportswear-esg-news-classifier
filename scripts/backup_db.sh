#!/bin/bash
# Database backup script for ESG News Classifier
# Creates compressed PostgreSQL dumps with rotation

# `pipefail` is load-bearing, not stylistic: without it a pipeline reports only its
# LAST command's status, so `pg_dump | gzip > file` reports gzip's success even when
# pg_dump died mid-stream and the archive is truncated. See issue #89.
set -eo pipefail

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

# Exit codes. Intended as the contract for cron and for src/agent/runner.py;
# nothing consumes it yet (`run_backup_status` has no caller), so treat this as
# the definition a future consumer binds to rather than a live binding.
#
#   0  the command did what it says
#   1  the command failed. Pre-existing and unchanged: bad arguments, missing
#      archive, pg_dump/gunzip failure. Deliberately NOT reused below -- `1` is
#      this script's generic failure, so mapping it to a specific cause would
#      collapse unrelated signals into one verdict. Where the failure came from
#      an external command, that command's own reason is on stderr.
#   2  a query this script needed failed, so a fact it depends on was never
#      established. Two paths reach it: the `docker ps` check, and the
#      database-size query in `show_status`. Whichever failed, that command's
#      own output is printed rather than a guess at the cause. `unknown` in the
#      Health Verdict Contract (docs/AGENT.md) -- never `healthy`. Retrying may
#      heal this.
#   3  Docker answered, and the container is not in its running list.
#      `degraded` in that same contract: a check ran and found a real problem.
#      Retrying will not heal it. Note the vocabulary's own caveat -- `degraded`
#      does not by itself fail a workflow run; what to do about it is the
#      consumer's decision, and no consumer exists yet.
#
# Not produced here: src/agent/runner.py synthesizes -1 for a timeout or an
# exception, so a consumer needs a default branch as well as these four.
EXIT_CANNOT_CHECK=2
EXIT_CONTAINER_ABSENT=3

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

# Report what is known about the container, WITHOUT exiting. Returns 0,
# $EXIT_CANNOT_CHECK or $EXIT_CONTAINER_ABSENT. The mutating commands use the
# fatal `check_container` wrapper below; `status` calls this directly so a
# read-only query can still report what it did learn.
#
# Three shapes here are load-bearing rather than stylistic:
#
# 1. `listing=$(...)` declared on its own line. `local listing=$(...)` returns
#    `local`'s status, not the command's, which would make the failure branch
#    unreachable -- the rule issue #90 is about.
# 2. `|| status=$?` rather than reading `$?` inside an `if ! ...; then` body,
#    where it is the negation's status and always 0.
# 3. `[[ ]]` with "$CONTAINER_NAME" quoted, which matches it literally.
#    Unquoted it would glob.
#
# On the replaced `docker ps ... | grep -q "^${CONTAINER_NAME}$"`: `grep` read
# the name as a regex, so `esg.news_db` matched `esgxnews_db`. Capturing
# mid-pipeline stderr needs a temp file or process substitution, so the pipeline
# could not show docker's message under a label or attribute it to this check --
# it went to stderr unredirected, not nowhere. D006 also recorded a `pipefail`
# SIGPIPE inversion in that construct, introduced by #89 and deferred here; it
# measured the inversion real but unreachable at this scale.
check_container_state() {
    local listing status=0
    # 2>&1 so the failure branch can show docker's own message. On the success
    # path it also folds in any docker warning, which whole-line matching makes
    # harmless here -- but see show_status, where a captured value is RENDERED
    # and the same fold needs a shape guard.
    listing=$(docker ps --format '{{.Names}}' 2>&1) || status=$?

    if [ "$status" -ne 0 ]; then
        log_error "Could not query Docker, so container '$CONTAINER_NAME' was never checked (docker exited $status)"
        # Gated: a header followed by nothing renders absence as presence.
        if [ -n "$listing" ]; then
            log_error "Docker reported:"
            printf '%s\n' "$listing"
        else
            log_error "Docker produced no output."
        fi
        return "$EXIT_CANNOT_CHECK"
    fi

    if [[ $'\n'"$listing"$'\n' != *$'\n'"$CONTAINER_NAME"$'\n'* ]]; then
        log_error "Container '$CONTAINER_NAME' is not running"
        # Show the listing rather than only naming the container. "Not in the
        # running list" is also what a misconfigured CONTAINER_NAME or a
        # different compose prefix looks like, and those are the states where
        # starting the container changes nothing -- the operator needs to see
        # the real names to tell which they are in. Hence "if it is stopped"
        # rather than an unconditional instruction.
        if [ -n "$listing" ]; then
            log_error "Running containers:"
            printf '%s\n' "$listing"
        else
            log_error "No containers are running."
        fi
        log_info "If it is stopped, start it with: docker compose up -d postgres"
        return "$EXIT_CONTAINER_ABSENT"
    fi

    return 0
}

# Fatal wrapper for the commands that mutate: neither `backup` nor `restore` can
# proceed without the container. It propagates the SPECIFIC code rather than a
# hardcoded 1, or the contract above would be defeated at the call site.
check_container() {
    local status=0
    check_container_state || status=$?
    [ "$status" -eq 0 ] || exit "$status"
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

    # Create the backup using pg_dump through docker.
    #
    # The pipeline is the `if` CONDITION rather than a bare statement followed by
    # `if [ $? -eq 0 ]`. Both halves matter:
    #   * `set -o pipefail` (top of file) makes this report pg_dump's failure
    #     instead of gzip's success, so a truncated archive fails the backup;
    #   * a command in an `if` condition is exempt from `set -e`, which is what
    #     makes the `else` branch reachable.
    #
    # As a bare pipeline that `else` was unreachable TWO different ways, and only
    # the second is a `set -e` abort:
    #   * pg_dump dying mid-stream left the pipeline reporting gzip's 0, so `$?`
    #     was read, was 0, and the SUCCESS branch ran -- this is the #89 case;
    #   * gzip itself failing (disk full) made the pipeline non-zero, which under
    #     `set -e` aborted the SCRIPT before `$?` could be read at all.
    # Either way the `rm -f` below was dead code and the partial file survived to
    # be reported as the latest backup by `list`/`status`. Stating only the abort
    # would teach the belief that caused #89 -- that `set -e` alone notices a
    # mid-pipeline death.
    #
    # The two bullets at the top -- detection and reachability -- are asserted by
    # tests/test_backup_db_script.py::test_backup_failure_exits_nonzero_and_removes_partial_archive
    # Route 2 (gzip itself failing) has no test: it is the route `set -e` already
    # caught before this change, so nothing here regressed it.
    if docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" \
        --format=plain \
        --no-owner \
        --no-privileges \
        | gzip > "$DAILY_PATH"; then
        # `|| BACKUP_SIZE=` is deliberate. This is the only plain (non-`local`)
        # assignment in the file whose right-hand side is a PIPELINE -- other plain
        # assignments exist (`TIMESTAMP=$(date ...)`), but `pipefail` cannot reach
        # them. So under `pipefail` a `du` failure propagates to
        # `set -e` and aborts INSIDE the success branch: a good archive is already
        # on disk, but there is no success line, no weekly/monthly copy, no
        # rotation, and the `rm -f` below is in the `else` and never runs. The size
        # is cosmetic and must not be able to fail a backup that worked.
        BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1) || BACKUP_SIZE="unknown"
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
    if ! docker exec "$CONTAINER_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" | gzip > "$PRE_RESTORE"; then
        log_error "Pre-restore backup failed - aborting before any destructive change"
        rm -f "$PRE_RESTORE"
        exit 1
    fi
    log_info "Pre-restore backup saved to: $PRE_RESTORE"

    log_info "Restoring from: $backup_file"

    # Drop and recreate database
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "CREATE DATABASE ${DB_NAME};"
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

    # Restore from backup - same `if <pipeline>` form as create_backup, and for the
    # same two reasons. Without it a truncated or corrupt archive made gunzip fail
    # while psql exited 0, and the success branch below reported
    # "Restore completed successfully!" over partially-loaded data.
    #
    # Asserted by
    # tests/test_backup_db_script.py::test_restore_reports_failure_on_corrupt_archive
    if gunzip -c "$backup_file" | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME"; then
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
            # `|| true`: the enclosing guard tests `ls -A` (is the directory
            # non-empty), not the *.sql.gz glob, so a directory holding only
            # non-archive files makes this `ls` fail. Under `pipefail` that would
            # abort the whole listing.
            # Asserted by
            # tests/test_backup_db_script.py::test_list_backups_survives_a_failing_glob_under_pipefail
            ls -lh "$BACKUP_DIR/$period"/*.sql.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || true
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

    # Show current database size.
    #
    # `status` is read-only, so a Docker problem must not discard the on-disk
    # facts printed above -- but it must not exit 0 either. `ScriptResult.success`
    # (src/agent/runner.py) is `exit_code == 0`, so exiting 0 here would make
    # "Docker unreachable" indistinguishable from a clean status to the only
    # consumer that would ever read it. So: report what was learned, say plainly
    # what was not, and exit with the specific code.
    echo ""
    local status=0
    check_container_state || status=$?
    if [ "$status" -ne 0 ]; then
        echo "Current database size: could not be determined"
        exit "$status"
    fi

    # The previous form was `local db_size=$(... | tr -d ' ')` with
    # `2>/dev/null`: `local` returned its own status rather than the pipeline's,
    # so a psql that refused the connection was indistinguishable from a
    # successful query, and the reason was discarded. An empty size was printed
    # as though it were a fact, with exit 0.
    #
    # Two variables, because the whitespace strip is lossy and the failure path
    # needs the original: stripping "psql: error: connection to server failed"
    # the same way runs it together into one unreadable token.
    local db_size_raw db_size
    db_size_raw=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>&1) || status=$?
    db_size="${db_size_raw//[[:space:]]/}"

    # Assert the SHAPE, not just non-emptiness. `pg_size_pretty` renders digits
    # then a unit, so after the strip a value is exactly that and nothing else.
    #
    # The unit list is taken from a live server (PostgreSQL 16.11) rather than
    # from memory, because a missing unit turns a working size into a permanent
    # "could not be determined" -- a false negative introduced by the guard:
    #   1 -> "1 bytes"  (plural even at 1, so no singular spelling to allow)
    #   10 kB / 10 MB / 1024 GB / 1024 TB / 1024 PB
    # `PB` is easy to omit and PostgreSQL does emit it. The optional decimal is
    # defensive: `pg_database_size` returns bigint, so this call site gets the
    # integer-only overload, but `pg_size_pretty(numeric)` exists.
    #
    # Non-emptiness alone is not enough because `2>&1`
    # folds psql's stderr into this capture: a server NOTICE, a psql startup
    # warning, or a docker shim banner on an otherwise SUCCESSFUL query would
    # be welded onto the number and printed as the size, on exit 0 -- the same
    # "unestablished value rendered as a fact" this change exists to remove.
    #
    # The trade, stated because it is a real cost: a benign warning now costs
    # the value rather than corrupting it. That is the safe direction -- the
    # script says it could not determine the size instead of reporting a
    # garbled one.
    if [ "$status" -ne 0 ] || [[ ! $db_size =~ ^[0-9]+(\.[0-9]+)?(bytes|kB|MB|GB|TB|PB)$ ]]; then
        echo "Current database size: could not be determined"
        # Report psql's status only when psql is what failed. On the
        # shape/emptiness path psql exited 0, and naming an exit status as the
        # reason would assert a cause that is not the one that fired.
        if [ "$status" -ne 0 ]; then
            log_error "Could not read the database size from container '$CONTAINER_NAME' (psql exited $status)"
        else
            log_error "Could not read the database size from container '$CONTAINER_NAME': psql exited 0 but returned no usable size"
        fi
        if [ -n "$db_size_raw" ]; then
            log_error "psql reported:"
            printf '%s\n' "$db_size_raw"
        else
            log_error "psql produced no output."
        fi
        exit "$EXIT_CANNOT_CHECK"
    fi

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
