-- ============================================================================
-- Scorecard History Queries
-- Queries for analyzing historical brand ESG scores and trends
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Basic Overview
-- ---------------------------------------------------------------------------

-- Scorecard snapshots overview
SELECT
    period_end,
    period_days,
    articles_in_period,
    articles_after_dedup,
    duplicates_removed,
    created_at AT TIME ZONE 'America/Los_Angeles' AS created_local
FROM scorecard_snapshots
ORDER BY period_end DESC
LIMIT 30;

-- Total snapshots by period length
SELECT
    period_days,
    COUNT(*) AS snapshot_count,
    MIN(period_end) AS earliest,
    MAX(period_end) AS latest
FROM scorecard_snapshots
GROUP BY period_days
ORDER BY period_days;

-- ---------------------------------------------------------------------------
-- Brand Score History
-- ---------------------------------------------------------------------------

-- Nike's scores over the last 30 days
SELECT
    ss.period_end,
    sbs.total,
    sbs.environmental,
    sbs.social,
    sbs.governance,
    sbs.digital_transformation,
    sbs.rank_position,
    sbs.medal
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE sbs.brand = 'Nike'
  AND ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ss.period_end ASC;

-- All brands on a specific date
SELECT
    sbs.brand,
    sbs.total,
    sbs.environmental AS env,
    sbs.social AS soc,
    sbs.governance AS gov,
    sbs.digital_transformation AS dig,
    sbs.article_count,
    sbs.rank_position,
    CASE
        WHEN sbs.is_top_performer THEN 'TOP'
        WHEN sbs.is_bottom_performer THEN 'BOTTOM'
        ELSE ''
    END AS status,
    sbs.medal
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_end = CURRENT_DATE
  AND ss.period_days = 14
ORDER BY sbs.rank_position;

-- Compare brand rankings across two dates
WITH date1 AS (
    SELECT sbs.brand, sbs.total, sbs.rank_position
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE ss.period_end = CURRENT_DATE - INTERVAL '7 days'
      AND ss.period_days = 14
),
date2 AS (
    SELECT sbs.brand, sbs.total, sbs.rank_position
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE ss.period_end = CURRENT_DATE
      AND ss.period_days = 14
)
SELECT
    COALESCE(d1.brand, d2.brand) AS brand,
    d1.total AS total_7d_ago,
    d2.total AS total_today,
    d2.total - COALESCE(d1.total, 0) AS score_change,
    d1.rank_position AS rank_7d_ago,
    d2.rank_position AS rank_today,
    COALESCE(d1.rank_position, 999) - COALESCE(d2.rank_position, 999) AS rank_improvement
FROM date1 d1
FULL OUTER JOIN date2 d2 ON d1.brand = d2.brand
ORDER BY rank_improvement DESC;

-- ---------------------------------------------------------------------------
-- Trend Analysis
-- ---------------------------------------------------------------------------

-- Brand trend classification (last 14 days vs previous 14 days)
WITH recent AS (
    SELECT sbs.brand, AVG(sbs.total) AS avg_total
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE ss.period_days = 14
      AND ss.period_end >= CURRENT_DATE - INTERVAL '14 days'
    GROUP BY sbs.brand
),
previous AS (
    SELECT sbs.brand, AVG(sbs.total) AS avg_total
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE ss.period_days = 14
      AND ss.period_end >= CURRENT_DATE - INTERVAL '28 days'
      AND ss.period_end < CURRENT_DATE - INTERVAL '14 days'
    GROUP BY sbs.brand
)
SELECT
    r.brand,
    ROUND(p.avg_total::numeric, 2) AS previous_avg,
    ROUND(r.avg_total::numeric, 2) AS recent_avg,
    ROUND((r.avg_total - COALESCE(p.avg_total, 0))::numeric, 2) AS change,
    CASE
        WHEN r.avg_total > COALESCE(p.avg_total, 0) + 1 THEN 'Improving'
        WHEN r.avg_total < COALESCE(p.avg_total, 0) - 1 THEN 'Declining'
        ELSE 'Stable'
    END AS trend
FROM recent r
LEFT JOIN previous p ON r.brand = p.brand
ORDER BY change DESC;

-- Volatility analysis (standard deviation of scores)
SELECT
    sbs.brand,
    COUNT(*) AS data_points,
    ROUND(AVG(sbs.total)::numeric, 2) AS avg_score,
    ROUND(STDDEV(sbs.total)::numeric, 2) AS score_volatility,
    MIN(sbs.total) AS min_score,
    MAX(sbs.total) AS max_score
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sbs.brand
HAVING COUNT(*) >= 7
ORDER BY score_volatility DESC;

-- ---------------------------------------------------------------------------
-- Medal History
-- ---------------------------------------------------------------------------

-- Medal count by brand (last 90 days)
SELECT
    sbs.brand,
    SUM(CASE WHEN sbs.medal = 'gold' THEN 1 ELSE 0 END) AS gold_medals,
    SUM(CASE WHEN sbs.medal = 'silver' THEN 1 ELSE 0 END) AS silver_medals,
    SUM(CASE WHEN sbs.medal = 'bronze' THEN 1 ELSE 0 END) AS bronze_medals,
    COUNT(sbs.medal) AS total_medals
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '90 days'
  AND sbs.medal IS NOT NULL
GROUP BY sbs.brand
ORDER BY gold_medals DESC, silver_medals DESC, bronze_medals DESC;

-- Recent medal winners
SELECT
    ss.period_end,
    sbs.brand,
    sbs.medal,
    sbs.total
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND sbs.medal IS NOT NULL
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ss.period_end DESC,
    CASE sbs.medal
        WHEN 'gold' THEN 1
        WHEN 'silver' THEN 2
        WHEN 'bronze' THEN 3
    END;

-- Longest gold medal streak per brand
WITH gold_days AS (
    SELECT
        sbs.brand,
        ss.period_end,
        ROW_NUMBER() OVER (PARTITION BY sbs.brand ORDER BY ss.period_end) -
        ROW_NUMBER() OVER (PARTITION BY sbs.brand, sbs.medal ORDER BY ss.period_end) AS grp
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE sbs.medal = 'gold'
      AND ss.period_days = 14
),
streaks AS (
    SELECT
        brand,
        MIN(period_end) AS streak_start,
        MAX(period_end) AS streak_end,
        COUNT(*) AS streak_length
    FROM gold_days
    GROUP BY brand, grp
)
SELECT
    brand,
    MAX(streak_length) AS longest_gold_streak,
    (SELECT streak_start FROM streaks s2 WHERE s2.brand = s.brand ORDER BY streak_length DESC LIMIT 1) AS started,
    (SELECT streak_end FROM streaks s2 WHERE s2.brand = s.brand ORDER BY streak_length DESC LIMIT 1) AS ended
FROM streaks s
GROUP BY brand
HAVING MAX(streak_length) > 1
ORDER BY longest_gold_streak DESC;

-- ---------------------------------------------------------------------------
-- Top/Bottom Performer Analysis
-- ---------------------------------------------------------------------------

-- How often each brand appears in top 3
SELECT
    sbs.brand,
    COUNT(*) FILTER (WHERE sbs.is_top_performer) AS top_performer_days,
    COUNT(*) FILTER (WHERE sbs.is_bottom_performer) AS bottom_performer_days,
    COUNT(*) AS total_days,
    ROUND(100.0 * COUNT(*) FILTER (WHERE sbs.is_top_performer) / COUNT(*), 1) AS top_pct
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sbs.brand
ORDER BY top_performer_days DESC;

-- Days spent in each position
SELECT
    sbs.brand,
    sbs.rank_position,
    COUNT(*) AS days_at_position
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sbs.brand, sbs.rank_position
ORDER BY sbs.brand, sbs.rank_position;

-- ---------------------------------------------------------------------------
-- Deduplication Statistics
-- ---------------------------------------------------------------------------

-- Daily deduplication stats
SELECT
    period_end,
    articles_in_period,
    articles_after_dedup,
    duplicates_removed,
    ROUND(100.0 * duplicates_removed / NULLIF(articles_in_period, 0), 1) AS dedup_rate_pct,
    similarity_threshold
FROM scorecard_snapshots
WHERE period_days = 14
ORDER BY period_end DESC
LIMIT 30;

-- Average deduplication rate over time
SELECT
    DATE_TRUNC('week', period_end)::date AS week,
    ROUND(AVG(articles_in_period)::numeric, 1) AS avg_articles,
    ROUND(AVG(duplicates_removed)::numeric, 1) AS avg_dupes_removed,
    ROUND(AVG(100.0 * duplicates_removed / NULLIF(articles_in_period, 0))::numeric, 1) AS avg_dedup_rate_pct
FROM scorecard_snapshots
WHERE period_days = 14
GROUP BY DATE_TRUNC('week', period_end)
ORDER BY week DESC
LIMIT 12;

-- ---------------------------------------------------------------------------
-- Category-Level Analysis
-- ---------------------------------------------------------------------------

-- Category score trends for a brand
SELECT
    ss.period_end,
    sbs.environmental,
    sbs.social,
    sbs.governance,
    sbs.digital_transformation
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE sbs.brand = 'Nike'
  AND ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ss.period_end ASC;

-- Average category scores by brand
SELECT
    sbs.brand,
    ROUND(AVG(sbs.environmental)::numeric, 2) AS avg_env,
    ROUND(AVG(sbs.social)::numeric, 2) AS avg_soc,
    ROUND(AVG(sbs.governance)::numeric, 2) AS avg_gov,
    ROUND(AVG(sbs.digital_transformation)::numeric, 2) AS avg_dig,
    ROUND(AVG(sbs.total)::numeric, 2) AS avg_total
FROM scorecard_brand_scores sbs
JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
WHERE ss.period_days = 14
  AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sbs.brand
ORDER BY avg_total DESC;

-- Category strength index (which category contributes most to brand's score)
WITH brand_averages AS (
    SELECT
        sbs.brand,
        AVG(sbs.environmental) AS avg_env,
        AVG(sbs.social) AS avg_soc,
        AVG(sbs.governance) AS avg_gov,
        AVG(sbs.digital_transformation) AS avg_dig,
        AVG(sbs.total) AS avg_total
    FROM scorecard_brand_scores sbs
    JOIN scorecard_snapshots ss ON sbs.snapshot_id = ss.id
    WHERE ss.period_days = 14
      AND ss.period_end >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY sbs.brand
)
SELECT
    brand,
    CASE GREATEST(avg_env, avg_soc, avg_gov, avg_dig)
        WHEN avg_env THEN 'Environmental'
        WHEN avg_soc THEN 'Social'
        WHEN avg_gov THEN 'Governance'
        WHEN avg_dig THEN 'Digital'
    END AS strongest_category,
    ROUND(avg_total::numeric, 2) AS avg_total
FROM brand_averages
WHERE avg_total != 0
ORDER BY avg_total DESC;
