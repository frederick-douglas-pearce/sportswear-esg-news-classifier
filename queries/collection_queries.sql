-- ============================================================================
-- Collection Queries
-- Queries for monitoring data collection pipeline and scraping progress
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Overview Stats
-- ---------------------------------------------------------------------------

-- Total articles collected (all sources)
SELECT COUNT(*) AS total_articles FROM articles;

-- Articles by scrape status
-- Shows how many articles are pending, successfully scraped, or failed
SELECT
    scrape_status,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM articles
GROUP BY scrape_status
ORDER BY count DESC;

-- Articles by data source
-- Shows distribution between NewsData.io and GDELT
SELECT
    source_name,
    COUNT(*) AS articles,
    MIN(published_at)::date AS earliest,
    MAX(published_at)::date AS latest
FROM articles
GROUP BY source_name
ORDER BY articles DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Collection Runs
-- ---------------------------------------------------------------------------

-- Recent collection runs with statistics
-- Shows API calls, articles fetched, duplicates, and scrape results
SELECT
    started_at AT TIME ZONE 'America/Los_Angeles' AS started_local,
    status,
    api_calls_made,
    articles_fetched,
    articles_duplicates,
    articles_scraped,
    articles_scrape_failed,
    EXTRACT(EPOCH FROM (completed_at - started_at))::int AS duration_seconds
FROM collection_runs
ORDER BY started_at DESC
LIMIT 10;

-- Collection runs summary by day
-- Aggregates daily collection statistics
SELECT
    started_at::date AS date,
    COUNT(*) AS runs,
    SUM(api_calls_made) AS total_api_calls,
    SUM(articles_fetched) AS total_fetched,
    SUM(articles_duplicates) AS total_duplicates,
    SUM(articles_scraped) AS total_scraped
FROM collection_runs
GROUP BY started_at::date
ORDER BY date DESC
LIMIT 14;

-- Failed collection runs
-- Identify runs that encountered errors
SELECT
    started_at,
    status,
    error_message,
    api_calls_made,
    articles_fetched
FROM collection_runs
WHERE status != 'completed'
ORDER BY started_at DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- Scraping Progress
-- ---------------------------------------------------------------------------

-- Articles pending scrape
-- These articles have metadata but no full_content yet
SELECT COUNT(*) AS pending_scrape
FROM articles
WHERE scrape_status = 'pending';

-- Recently failed scrapes
-- Articles where scraping failed (paywall, blocked, etc.)
SELECT
    LEFT(title, 50) AS title,
    url,
    scrape_error,
    created_at::date
FROM articles
WHERE scrape_status = 'failed'
ORDER BY created_at DESC
LIMIT 20;

-- Scrape success rate by source
-- Identify which news sources have better scraping success
SELECT
    source_name,
    COUNT(*) AS total,
    SUM(CASE WHEN scrape_status = 'success' THEN 1 ELSE 0 END) AS scraped,
    SUM(CASE WHEN scrape_status = 'failed' THEN 1 ELSE 0 END) AS failed,
    ROUND(100.0 * SUM(CASE WHEN scrape_status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) AS success_rate
FROM articles
GROUP BY source_name
HAVING COUNT(*) >= 5
ORDER BY success_rate DESC;

-- ---------------------------------------------------------------------------
-- Time-based Analysis
-- ---------------------------------------------------------------------------

-- Articles collected per day
SELECT
    created_at::date AS collection_date,
    COUNT(*) AS articles
FROM articles
GROUP BY created_at::date
ORDER BY collection_date DESC
LIMIT 30;

-- Articles by publication date
-- Shows the date range of articles in the database
SELECT
    published_at::date AS publish_date,
    COUNT(*) AS articles
FROM articles
WHERE published_at IS NOT NULL
GROUP BY published_at::date
ORDER BY publish_date DESC
LIMIT 30;

-- Collection gaps
-- Find days with no collection activity
WITH date_series AS (
    SELECT generate_series(
        (SELECT MIN(created_at)::date FROM articles),
        CURRENT_DATE,
        '1 day'::interval
    )::date AS date
)
SELECT ds.date AS missing_date
FROM date_series ds
LEFT JOIN articles a ON a.created_at::date = ds.date
WHERE a.id IS NULL
ORDER BY ds.date DESC;
