-- Migration: Add prompt versioning support
-- Description: Track prompt versions used in labeling runs and brand labels
-- Date: 2025-01-09

-- Add prompt_version columns to labeling_runs
ALTER TABLE labeling_runs
ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(20),
ADD COLUMN IF NOT EXISTS prompt_system_hash VARCHAR(12),
ADD COLUMN IF NOT EXISTS prompt_user_hash VARCHAR(12);

-- Add prompt_version to brand_labels
ALTER TABLE brand_labels
ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(20);

-- Add indexes for querying by prompt version
CREATE INDEX IF NOT EXISTS ix_labeling_runs_prompt_version
    ON labeling_runs(prompt_version);
CREATE INDEX IF NOT EXISTS ix_brand_labels_prompt_version
    ON brand_labels(prompt_version);

-- Add comments for documentation
COMMENT ON COLUMN labeling_runs.prompt_version IS 'Version of prompt templates used (e.g., v1.0.0)';
COMMENT ON COLUMN labeling_runs.prompt_system_hash IS 'SHA256 hash prefix of system prompt for verification';
COMMENT ON COLUMN labeling_runs.prompt_user_hash IS 'SHA256 hash prefix of user prompt for verification';
COMMENT ON COLUMN brand_labels.prompt_version IS 'Version of prompt used to generate this label';

-- Verification query (run after migration to confirm changes)
-- SELECT column_name, data_type FROM information_schema.columns
-- WHERE table_name IN ('labeling_runs', 'brand_labels')
-- AND column_name LIKE 'prompt%'
-- ORDER BY table_name, column_name;
