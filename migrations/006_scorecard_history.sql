-- Migration: Add scorecard history tables
-- Created: 2026-02-10
-- Purpose: Store sustainability scorecard results over time for historical trend analysis
--
-- Usage:
--   psql $DATABASE_URL -f migrations/006_scorecard_history.sql
--
-- Tables:
--   scorecard_snapshots - Daily snapshot metadata (period, article counts, settings)
--   scorecard_brand_scores - Per-brand scores for each snapshot

-- =============================================================================
-- Table 1: scorecard_snapshots (Parent)
-- One row per daily snapshot
-- =============================================================================

CREATE TABLE IF NOT EXISTS scorecard_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_days INTEGER NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    articles_in_period INTEGER NOT NULL DEFAULT 0,
    articles_after_dedup INTEGER NOT NULL DEFAULT 0,
    duplicates_removed INTEGER NOT NULL DEFAULT 0,
    similarity_threshold FLOAT NOT NULL DEFAULT 0.70,
    require_label_match BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Unique constraint: one snapshot per period type per day
CREATE UNIQUE INDEX IF NOT EXISTS ix_scorecard_snapshots_period_day
ON scorecard_snapshots(period_days, period_end);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS ix_scorecard_snapshots_period_end
ON scorecard_snapshots(period_end DESC);

-- Index for created_at queries
CREATE INDEX IF NOT EXISTS ix_scorecard_snapshots_created_at
ON scorecard_snapshots(created_at DESC);

-- Add comments
COMMENT ON TABLE scorecard_snapshots IS 'Daily sustainability scorecard snapshots with metadata';
COMMENT ON COLUMN scorecard_snapshots.period_days IS 'Rolling window size (e.g., 14 days)';
COMMENT ON COLUMN scorecard_snapshots.period_start IS 'Start date of scoring period';
COMMENT ON COLUMN scorecard_snapshots.period_end IS 'End date of scoring period (snapshot date)';
COMMENT ON COLUMN scorecard_snapshots.articles_in_period IS 'Total articles before deduplication';
COMMENT ON COLUMN scorecard_snapshots.articles_after_dedup IS 'Articles remaining after deduplication';
COMMENT ON COLUMN scorecard_snapshots.duplicates_removed IS 'Number of duplicate articles filtered';
COMMENT ON COLUMN scorecard_snapshots.similarity_threshold IS 'Cosine similarity threshold used for deduplication';
COMMENT ON COLUMN scorecard_snapshots.require_label_match IS 'Whether label matching was required for deduplication';


-- =============================================================================
-- Table 2: scorecard_brand_scores (Child)
-- One row per brand per snapshot
-- =============================================================================

CREATE TABLE IF NOT EXISTS scorecard_brand_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL REFERENCES scorecard_snapshots(id) ON DELETE CASCADE,
    brand VARCHAR(100) NOT NULL,
    environmental INTEGER NOT NULL DEFAULT 0,
    social INTEGER NOT NULL DEFAULT 0,
    governance INTEGER NOT NULL DEFAULT 0,
    digital_transformation INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    article_count INTEGER NOT NULL DEFAULT 0,
    rank_position INTEGER NOT NULL,
    is_top_performer BOOLEAN NOT NULL DEFAULT FALSE,
    is_bottom_performer BOOLEAN NOT NULL DEFAULT FALSE,
    medal VARCHAR(10),  -- 'gold', 'silver', 'bronze', or NULL
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Unique constraint: one score per brand per snapshot
CREATE UNIQUE INDEX IF NOT EXISTS ix_scorecard_brand_scores_snapshot_brand
ON scorecard_brand_scores(snapshot_id, brand);

-- Index for brand-based queries
CREATE INDEX IF NOT EXISTS ix_scorecard_brand_scores_brand
ON scorecard_brand_scores(brand);

-- Index for snapshot_id (foreign key)
CREATE INDEX IF NOT EXISTS ix_scorecard_brand_scores_snapshot_id
ON scorecard_brand_scores(snapshot_id);

-- Index for querying top/bottom performers
CREATE INDEX IF NOT EXISTS ix_scorecard_brand_scores_performers
ON scorecard_brand_scores(is_top_performer, is_bottom_performer);

-- Add comments
COMMENT ON TABLE scorecard_brand_scores IS 'Per-brand ESG scores for each scorecard snapshot';
COMMENT ON COLUMN scorecard_brand_scores.environmental IS 'Environmental category score';
COMMENT ON COLUMN scorecard_brand_scores.social IS 'Social category score';
COMMENT ON COLUMN scorecard_brand_scores.governance IS 'Governance category score';
COMMENT ON COLUMN scorecard_brand_scores.digital_transformation IS 'Digital transformation category score';
COMMENT ON COLUMN scorecard_brand_scores.total IS 'Sum of all category scores';
COMMENT ON COLUMN scorecard_brand_scores.article_count IS 'Number of articles mentioning this brand';
COMMENT ON COLUMN scorecard_brand_scores.rank_position IS '1-based rank by total score';
COMMENT ON COLUMN scorecard_brand_scores.is_top_performer IS 'In top 3 with positive score';
COMMENT ON COLUMN scorecard_brand_scores.is_bottom_performer IS 'In bottom 3 with negative score';
COMMENT ON COLUMN scorecard_brand_scores.medal IS 'Award tier: gold, silver, bronze, or NULL';
