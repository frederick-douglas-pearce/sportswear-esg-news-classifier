-- Migration: Record per-outcome article counts on labeling runs
-- Description: labeling_runs stored only articles_processed, so the daily_labeling
--              workflow had to scrape labeled/skipped/failed counts out of the
--              script's stdout. When a retry ran and found nothing to do, the
--              retry's zeros replaced the real numbers and the quality gate went
--              blind (issue #81). These columns make the run row self-describing.
-- Run: psql $DATABASE_URL -f migrations/007_labeling_run_outcomes.sql

ALTER TABLE labeling_runs
    ADD COLUMN IF NOT EXISTS articles_labeled INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS articles_skipped INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS articles_false_positive INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS articles_failed INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS articles_deduplicated INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS articles_left_pending INTEGER DEFAULT 0;

COMMENT ON COLUMN labeling_runs.articles_labeled IS
    'Articles that received at least one brand label';
COMMENT ON COLUMN labeling_runs.articles_skipped IS
    'Articles processed but skipped for lack of ESG content. Excludes false '
    'positives, which have their own column so the report can show both '
    'without double-counting.';
COMMENT ON COLUMN labeling_runs.articles_false_positive IS
    'Articles rejected as non-sportswear brand matches';
COMMENT ON COLUMN labeling_runs.articles_failed IS
    'Articles that errored during processing';
COMMENT ON COLUMN labeling_runs.articles_deduplicated IS
    'Articles dropped as title duplicates before processing';
COMMENT ON COLUMN labeling_runs.articles_left_pending IS
    'Articles whose processing raised before any status update, so they are '
    'still pending and a retry can still pick them up. Distinct from '
    'articles_failed, which have been marked failed and are spent.';

-- Existing rows keep 0 for these columns. Historical runs cannot be
-- reconstructed: the breakdown was never persisted anywhere.

-- Index the window the workflow queries (runs since a given start time).
CREATE INDEX IF NOT EXISTS idx_labeling_runs_started_at
    ON labeling_runs (started_at DESC);
