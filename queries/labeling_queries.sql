-- ============================================================================
-- Labeling Queries
-- Queries for monitoring and analyzing the LLM labeling pipeline
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Labeling Progress Overview
-- ---------------------------------------------------------------------------

-- Article labeling status distribution
-- Shows progress through the labeling pipeline
SELECT
    labeling_status,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM articles
GROUP BY labeling_status
ORDER BY count DESC;

-- Articles ready for labeling
-- Scraped articles that haven't been labeled yet
SELECT COUNT(*) AS ready_to_label
FROM articles
WHERE scrape_status = 'success'
  AND labeling_status = 'pending'
  AND full_content IS NOT NULL;

-- Recently labeled articles
SELECT
    LEFT(title, 50) AS title,
    labeling_status,
    labeled_at AT TIME ZONE 'America/Los_Angeles' AS labeled_local,
    array_length(brands_mentioned, 1) AS brands
FROM articles
WHERE labeled_at IS NOT NULL
ORDER BY labeled_at DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Brand Labels Analysis
-- ---------------------------------------------------------------------------

-- Brand labels by category
-- Shows ESG category distribution across all brands
SELECT
    brand,
    COUNT(*) AS total_labels,
    SUM(CASE WHEN environmental THEN 1 ELSE 0 END) AS environmental,
    SUM(CASE WHEN social THEN 1 ELSE 0 END) AS social,
    SUM(CASE WHEN governance THEN 1 ELSE 0 END) AS governance,
    SUM(CASE WHEN digital_transformation THEN 1 ELSE 0 END) AS digital
FROM brand_labels
GROUP BY brand
ORDER BY total_labels DESC;

-- Category breakdown with percentages
SELECT
    brand,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN environmental THEN 1 ELSE 0 END) / COUNT(*), 1) AS env_pct,
    ROUND(100.0 * SUM(CASE WHEN social THEN 1 ELSE 0 END) / COUNT(*), 1) AS social_pct,
    ROUND(100.0 * SUM(CASE WHEN governance THEN 1 ELSE 0 END) / COUNT(*), 1) AS gov_pct,
    ROUND(100.0 * SUM(CASE WHEN digital_transformation THEN 1 ELSE 0 END) / COUNT(*), 1) AS digital_pct
FROM brand_labels
GROUP BY brand
HAVING COUNT(*) >= 3
ORDER BY total DESC;

-- Sentiment analysis by brand and category
-- Shows positive/neutral/negative sentiment distribution
SELECT
    brand,
    'environmental' AS category,
    SUM(CASE WHEN environmental_sentiment = 1 THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN environmental_sentiment = 0 THEN 1 ELSE 0 END) AS neutral,
    SUM(CASE WHEN environmental_sentiment = -1 THEN 1 ELSE 0 END) AS negative
FROM brand_labels
WHERE environmental = true
GROUP BY brand
UNION ALL
SELECT
    brand,
    'social' AS category,
    SUM(CASE WHEN social_sentiment = 1 THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN social_sentiment = 0 THEN 1 ELSE 0 END) AS neutral,
    SUM(CASE WHEN social_sentiment = -1 THEN 1 ELSE 0 END) AS negative
FROM brand_labels
WHERE social = true
GROUP BY brand
UNION ALL
SELECT
    brand,
    'governance' AS category,
    SUM(CASE WHEN governance_sentiment = 1 THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN governance_sentiment = 0 THEN 1 ELSE 0 END) AS neutral,
    SUM(CASE WHEN governance_sentiment = -1 THEN 1 ELSE 0 END) AS negative
FROM brand_labels
WHERE governance = true
GROUP BY brand
UNION ALL
SELECT
    brand,
    'digital' AS category,
    SUM(CASE WHEN digital_sentiment = 1 THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN digital_sentiment = 0 THEN 1 ELSE 0 END) AS neutral,
    SUM(CASE WHEN digital_sentiment = -1 THEN 1 ELSE 0 END) AS negative
FROM brand_labels
WHERE digital_transformation = true
GROUP BY brand
ORDER BY brand, category;

-- Overall sentiment distribution
SELECT
    'environmental' AS category,
    SUM(CASE WHEN environmental_sentiment = 1 THEN 1 ELSE 0 END) AS positive,
    SUM(CASE WHEN environmental_sentiment = 0 THEN 1 ELSE 0 END) AS neutral,
    SUM(CASE WHEN environmental_sentiment = -1 THEN 1 ELSE 0 END) AS negative,
    COUNT(*) FILTER (WHERE environmental = true) AS total
FROM brand_labels
UNION ALL
SELECT 'social',
    SUM(CASE WHEN social_sentiment = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN social_sentiment = 0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN social_sentiment = -1 THEN 1 ELSE 0 END),
    COUNT(*) FILTER (WHERE social = true)
FROM brand_labels
UNION ALL
SELECT 'governance',
    SUM(CASE WHEN governance_sentiment = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN governance_sentiment = 0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN governance_sentiment = -1 THEN 1 ELSE 0 END),
    COUNT(*) FILTER (WHERE governance = true)
FROM brand_labels
UNION ALL
SELECT 'digital_transformation',
    SUM(CASE WHEN digital_transformation_sentiment = 1 THEN 1 ELSE 0 END),
    SUM(CASE WHEN digital_transformation_sentiment = 0 THEN 1 ELSE 0 END),
    SUM(CASE WHEN digital_transformation_sentiment = -1 THEN 1 ELSE 0 END),
    COUNT(*) FILTER (WHERE digital_transformation = true)
FROM brand_labels;

-- ---------------------------------------------------------------------------
-- Confidence Analysis
-- ---------------------------------------------------------------------------

-- Confidence score distribution
SELECT
    CASE
        WHEN confidence_score >= 0.9 THEN 'high (>=0.9)'
        WHEN confidence_score >= 0.7 THEN 'medium (0.7-0.9)'
        WHEN confidence_score >= 0.5 THEN 'low (0.5-0.7)'
        ELSE 'very low (<0.5)'
    END AS confidence_level,
    COUNT(*) AS count,
    ROUND(AVG(confidence_score), 3) AS avg_confidence
FROM brand_labels
GROUP BY 1
ORDER BY MIN(confidence_score) DESC;

-- Low confidence labels (may need review)
SELECT
    bl.brand,
    LEFT(a.title, 50) AS article_title,
    bl.environmental,
    bl.social,
    bl.governance,
    bl.digital_transformation,
    bl.confidence_score
FROM brand_labels bl
JOIN articles a ON bl.article_id = a.id
WHERE bl.confidence_score < 0.7
ORDER BY bl.confidence_score ASC
LIMIT 20;

-- Average confidence by brand
SELECT
    brand,
    COUNT(*) AS labels,
    ROUND(AVG(confidence_score), 3) AS avg_confidence,
    ROUND(MIN(confidence_score), 3) AS min_confidence,
    ROUND(MAX(confidence_score), 3) AS max_confidence
FROM brand_labels
GROUP BY brand
ORDER BY avg_confidence DESC;

-- ---------------------------------------------------------------------------
-- Labeling Runs
-- ---------------------------------------------------------------------------

-- Recent labeling runs with statistics
SELECT
    started_at AT TIME ZONE 'America/Los_Angeles' AS started_local,
    status,
    articles_processed,
    brands_labeled,
    llm_calls_made,
    total_input_tokens,
    total_output_tokens,
    ROUND(estimated_cost_usd::numeric, 4) AS cost_usd
FROM labeling_runs
ORDER BY started_at DESC
LIMIT 10;

-- Labeling run summary (totals)
SELECT
    COUNT(*) AS total_runs,
    SUM(articles_processed) AS total_articles,
    SUM(brands_labeled) AS total_labels,
    SUM(llm_calls_made) AS total_llm_calls,
    SUM(total_input_tokens) AS total_input_tokens,
    SUM(total_output_tokens) AS total_output_tokens,
    ROUND(SUM(estimated_cost_usd)::numeric, 2) AS total_cost_usd
FROM labeling_runs
WHERE status = 'completed';

-- Cost per article analysis
SELECT
    started_at::date AS date,
    articles_processed,
    ROUND(estimated_cost_usd::numeric, 4) AS cost,
    ROUND((estimated_cost_usd / NULLIF(articles_processed, 0))::numeric, 4) AS cost_per_article
FROM labeling_runs
WHERE articles_processed > 0
ORDER BY started_at DESC
LIMIT 10;

-- ---------------------------------------------------------------------------
-- Labels with Article Context
-- ---------------------------------------------------------------------------

-- Full label details with article info
SELECT
    bl.brand,
    LEFT(a.title, 40) AS title,
    bl.environmental AS env,
    bl.environmental_sentiment AS env_sent,
    bl.social,
    bl.social_sentiment AS soc_sent,
    bl.governance AS gov,
    bl.governance_sentiment AS gov_sent,
    bl.digital_transformation AS digital,
    bl.digital_transformation_sentiment AS dig_sent,
    bl.confidence_score,
    a.published_at::date AS published
FROM brand_labels bl
JOIN articles a ON bl.article_id = a.id
ORDER BY bl.created_at DESC
LIMIT 20;

-- Articles with no ESG categories identified
-- These might be false positives in collection
SELECT
    bl.brand,
    LEFT(a.title, 60) AS title,
    bl.confidence_score
FROM brand_labels bl
JOIN articles a ON bl.article_id = a.id
WHERE bl.environmental = false
  AND bl.social = false
  AND bl.governance = false
  AND bl.digital_transformation = false
ORDER BY bl.created_at DESC;

-- Articles with multiple ESG categories
-- Rich ESG content
SELECT
    bl.brand,
    LEFT(a.title, 50) AS title,
    (CASE WHEN bl.environmental THEN 1 ELSE 0 END +
     CASE WHEN bl.social THEN 1 ELSE 0 END +
     CASE WHEN bl.governance THEN 1 ELSE 0 END +
     CASE WHEN bl.digital_transformation THEN 1 ELSE 0 END) AS category_count,
    bl.confidence_score
FROM brand_labels bl
JOIN articles a ON bl.article_id = a.id
WHERE (CASE WHEN bl.environmental THEN 1 ELSE 0 END +
       CASE WHEN bl.social THEN 1 ELSE 0 END +
       CASE WHEN bl.governance THEN 1 ELSE 0 END +
       CASE WHEN bl.digital_transformation THEN 1 ELSE 0 END) >= 2
ORDER BY category_count DESC, bl.confidence_score DESC
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Skipped and Failed Articles
-- ---------------------------------------------------------------------------

-- Articles that were skipped during labeling
SELECT
    LEFT(title, 60) AS title,
    labeling_status,
    labeling_error,
    LENGTH(full_content) AS content_length
FROM articles
WHERE labeling_status = 'skipped'
ORDER BY labeled_at DESC
LIMIT 20;

-- Failed labeling attempts
SELECT
    LEFT(title, 60) AS title,
    labeling_error,
    labeled_at
FROM articles
WHERE labeling_status = 'failed'
ORDER BY labeled_at DESC;

-- ---------------------------------------------------------------------------
-- Historical Data Review (Post-Policy Change Analysis)
-- Use these queries to identify articles that may need relabeling after
-- prompt/policy changes (e.g., tangential brand mentions, stock articles)
-- ---------------------------------------------------------------------------

-- Count articles by status before a specific date
-- Update the timestamp to match your policy change date
SELECT
    COUNT(*) AS count,
    labeling_status
FROM articles
WHERE labeled_at < '2025-12-26 00:13:57+00'
GROUP BY labeling_status;

-- Find potentially mislabeled stock/financial articles (labeled)
-- These may be tangential brand mentions that should be false_positive
SELECT
    a.title,
    a.source_name,
    bl.brand,
    a.labeled_at
FROM articles a
JOIN brand_labels bl ON bl.article_id = a.id
WHERE a.labeled_at < '2025-12-26 00:13:57+00'
AND (
    a.title ILIKE '%stock%' OR
    a.title ILIKE '%shares%' OR
    a.title ILIKE '%trading%' OR
    a.title ILIKE '%short interest%'
)
ORDER BY a.labeled_at DESC;

-- Find skipped articles that may be false positives
-- Note: labeled_at is NULL for skips, so use created_at for filtering
SELECT
    a.title,
    a.source_name,
    a.brands_mentioned
FROM articles a
WHERE a.labeling_status = 'skipped'
AND (
    a.title ILIKE '%stock%' OR
    a.title ILIKE '%shares%' OR
    a.title ILIKE '%trading%' OR
    a.title ILIKE '%short interest%' OR
    a.title ILIKE '%former%' OR
    a.title ILIKE '%ex-%' OR
    a.title ILIKE '%appoints%' OR
    a.title ILIKE '%joins%'
)
ORDER BY a.created_at DESC;
