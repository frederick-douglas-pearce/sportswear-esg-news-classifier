-- ============================================================================
-- Article Queries
-- Queries for exploring and analyzing collected articles
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Article Overview
-- ---------------------------------------------------------------------------

-- Recent articles with key metadata
SELECT
    LEFT(title, 60) AS title,
    source_name,
    brands_mentioned,
    scrape_status,
    labeling_status,
    published_at::date AS published
FROM articles
ORDER BY created_at DESC
LIMIT 20;

-- Articles with full content available
-- These are ready for labeling
SELECT
    id,
    LEFT(title, 50) AS title,
    LENGTH(full_content) AS content_length,
    array_length(brands_mentioned, 1) AS num_brands,
    published_at::date
FROM articles
WHERE scrape_status = 'success'
  AND full_content IS NOT NULL
ORDER BY created_at DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Brand Analysis
-- ---------------------------------------------------------------------------

-- Articles per brand (from brands_mentioned array)
-- Shows which brands have the most coverage
SELECT
    unnest(brands_mentioned) AS brand,
    COUNT(*) AS article_count
FROM articles
WHERE brands_mentioned IS NOT NULL
GROUP BY brand
ORDER BY article_count DESC;

-- Articles mentioning multiple brands
-- Useful for finding comparison/industry articles
SELECT
    LEFT(title, 60) AS title,
    brands_mentioned,
    array_length(brands_mentioned, 1) AS num_brands,
    published_at::date
FROM articles
WHERE array_length(brands_mentioned, 1) > 1
ORDER BY array_length(brands_mentioned, 1) DESC, published_at DESC
LIMIT 20;

-- Brand co-occurrence matrix
-- Shows which brands are frequently mentioned together
SELECT
    b1.brand AS brand1,
    b2.brand AS brand2,
    COUNT(*) AS co_occurrences
FROM articles a,
     LATERAL unnest(a.brands_mentioned) AS b1(brand),
     LATERAL unnest(a.brands_mentioned) AS b2(brand)
WHERE b1.brand < b2.brand  -- Avoid duplicates and self-joins
  AND array_length(brands_mentioned, 1) > 1
GROUP BY b1.brand, b2.brand
ORDER BY co_occurrences DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Content Analysis
-- ---------------------------------------------------------------------------

-- Article length distribution
-- Understand the size of scraped content
SELECT
    CASE
        WHEN LENGTH(full_content) < 1000 THEN 'short (<1k chars)'
        WHEN LENGTH(full_content) < 3000 THEN 'medium (1k-3k chars)'
        WHEN LENGTH(full_content) < 6000 THEN 'long (3k-6k chars)'
        ELSE 'very long (>6k chars)'
    END AS content_size,
    COUNT(*) AS articles,
    ROUND(AVG(LENGTH(full_content))) AS avg_chars
FROM articles
WHERE full_content IS NOT NULL
GROUP BY 1
ORDER BY MIN(LENGTH(full_content));

-- Articles by keyword in title
-- Search for specific topics (replace 'sustainability' with your keyword)
SELECT
    LEFT(title, 70) AS title,
    brands_mentioned,
    published_at::date
FROM articles
WHERE LOWER(title) LIKE '%sustainability%'
ORDER BY published_at DESC
LIMIT 20;

-- Full-text search in content
-- Find articles containing specific terms (replace search term as needed)
SELECT
    LEFT(title, 60) AS title,
    brands_mentioned,
    LEFT(full_content, 200) AS content_preview
FROM articles
WHERE full_content ILIKE '%carbon neutral%'
ORDER BY published_at DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- Source Analysis
-- ---------------------------------------------------------------------------

-- Top news sources by article count
SELECT
    source_name,
    COUNT(*) AS articles,
    COUNT(CASE WHEN scrape_status = 'success' THEN 1 END) AS scraped,
    MIN(published_at)::date AS earliest,
    MAX(published_at)::date AS latest
FROM articles
GROUP BY source_name
ORDER BY articles DESC
LIMIT 25;

-- Source quality ranking
-- Sources with best scraping success and content length
SELECT
    source_name,
    COUNT(*) AS total_articles,
    ROUND(100.0 * SUM(CASE WHEN scrape_status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) AS scrape_success_pct,
    ROUND(AVG(LENGTH(full_content))) AS avg_content_length
FROM articles
GROUP BY source_name
HAVING COUNT(*) >= 5
ORDER BY scrape_success_pct DESC, avg_content_length DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Specific Article Lookup
-- ---------------------------------------------------------------------------

-- Find article by ID (replace UUID)
-- SELECT * FROM articles WHERE id = 'your-uuid-here';

-- Find articles by URL pattern
SELECT id, title, url
FROM articles
WHERE url LIKE '%example.com%'
LIMIT 10;

-- View full article content (replace UUID)
-- SELECT title, full_content FROM articles WHERE id = 'your-uuid-here';
