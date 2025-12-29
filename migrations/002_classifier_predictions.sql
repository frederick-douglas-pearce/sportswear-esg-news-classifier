-- Migration: Create classifier_predictions table
-- Description: Stores ML classifier predictions for audit trail and analysis
-- Run: psql $DATABASE_URL -f migrations/002_classifier_predictions.sql

CREATE TABLE IF NOT EXISTS classifier_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,

    -- Classifier identification
    classifier_type VARCHAR(20) NOT NULL,  -- 'fp', 'ep', 'esg'
    model_version VARCHAR(100),             -- e.g., 'RF_tuned_v1'

    -- Prediction result
    probability FLOAT NOT NULL,             -- Raw probability from model (0.0-1.0)
    prediction BOOLEAN NOT NULL,            -- Binary prediction result
    threshold_used FLOAT NOT NULL,          -- Threshold used for this prediction

    -- FP-specific fields (nullable for other classifiers)
    risk_level VARCHAR(20),                 -- 'low', 'medium', 'high' for FP classifier

    -- EP/ESG-specific fields (nullable for FP)
    esg_categories JSONB,                   -- For ESG multi-label predictions

    -- Decision tracking
    action_taken VARCHAR(50) NOT NULL,      -- 'skipped_llm', 'continued_to_llm', 'failed'
    skip_reason VARCHAR(255),               -- Reason for skipping LLM (if applicable)

    -- Error handling
    error_message TEXT,                     -- If prediction failed

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_classifier_type CHECK (classifier_type IN ('fp', 'ep', 'esg')),
    CONSTRAINT valid_probability CHECK (probability >= 0.0 AND probability <= 1.0),
    CONSTRAINT valid_threshold CHECK (threshold_used >= 0.0 AND threshold_used <= 1.0),
    CONSTRAINT valid_action CHECK (action_taken IN ('skipped_llm', 'continued_to_llm', 'failed'))
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS ix_classifier_predictions_article_id
    ON classifier_predictions(article_id);
CREATE INDEX IF NOT EXISTS ix_classifier_predictions_classifier_type
    ON classifier_predictions(classifier_type);
CREATE INDEX IF NOT EXISTS ix_classifier_predictions_created_at
    ON classifier_predictions(created_at);
CREATE INDEX IF NOT EXISTS ix_classifier_predictions_action_taken
    ON classifier_predictions(action_taken);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS ix_classifier_predictions_type_action
    ON classifier_predictions(classifier_type, action_taken);

-- Comments for documentation
COMMENT ON TABLE classifier_predictions IS 'Stores ML classifier predictions for audit trail and analysis';
COMMENT ON COLUMN classifier_predictions.classifier_type IS 'Type of classifier: fp (False Positive), ep (ESG Pre-filter), esg (ESG Multi-label)';
COMMENT ON COLUMN classifier_predictions.action_taken IS 'Action taken based on prediction: skipped_llm (bypassed LLM), continued_to_llm (proceeded to Claude), failed (classifier error)';
COMMENT ON COLUMN classifier_predictions.skip_reason IS 'Human-readable explanation when LLM was skipped (e.g., "High-confidence false positive: probability 0.12 < threshold 0.30")';
COMMENT ON COLUMN classifier_predictions.risk_level IS 'FP classifier confidence level: low (prob < 0.3), medium (0.3-0.6), high (>= 0.6)';
