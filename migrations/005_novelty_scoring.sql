-- Migration: Add novelty scoring columns to classifier_predictions
-- Date: 2026-01-23
-- Description: Adds columns to track article topic novelty relative to training data

-- Add novelty scoring columns
ALTER TABLE classifier_predictions
ADD COLUMN IF NOT EXISTS novelty_score FLOAT,
ADD COLUMN IF NOT EXISTS novelty_cluster INTEGER;

-- Index for monitoring queries (filter by classifier type and novelty)
CREATE INDEX IF NOT EXISTS idx_classifier_predictions_novelty
ON classifier_predictions(classifier_type, novelty_score)
WHERE novelty_score IS NOT NULL;

-- Index for finding high-novelty articles
CREATE INDEX IF NOT EXISTS idx_classifier_predictions_high_novelty
ON classifier_predictions(classifier_type, created_at)
WHERE novelty_score > 0.7;

-- Add comments for documentation
COMMENT ON COLUMN classifier_predictions.novelty_score IS
'0-1 score indicating how novel the article topic is relative to training data. Higher = more novel. NULL if novelty scoring not enabled.';

COMMENT ON COLUMN classifier_predictions.novelty_cluster IS
'Index of the nearest training cluster centroid. Used for debugging and topic analysis.';
