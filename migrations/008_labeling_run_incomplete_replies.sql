-- Migration: Record how many LLM replies left out a brand on each labeling run
-- Description: the labeling run summary has printed "Incomplete replies" since #192 PR1,
--              but the count was not stored, so a run row could not show whether the
--              model judged every brand it was sent. Stored from #192 PR2, ahead of the
--              v1.11.0 promotion, so runs under v1.10.0 give a figure to compare with.
-- Run: psql $DATABASE_URL -f migrations/008_labeling_run_incomplete_replies.sql

-- Nullable with no default: NULL means the run predates this column (not recorded),
-- which a default of 0 would hide.
ALTER TABLE labeling_runs
    ADD COLUMN IF NOT EXISTS incomplete_replies INTEGER;

COMMENT ON COLUMN labeling_runs.incomplete_replies IS
    'LLM replies that left out a brand that was sent (src/labeling/disposition.py). '
    'Under prompt v1.10.0 and earlier the prompt tells the model to leave out sportswear '
    'brands with no ESG content, so a left-out brand is expected; from v1.11.0 every brand '
    'is to be judged, so it is non-compliance. Read it with prompt_version. NULL: not recorded.';
