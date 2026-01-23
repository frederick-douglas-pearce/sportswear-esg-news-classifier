-- Migration: Add reranking columns to label_evidence
-- Created: 2026-01-20
-- Purpose: Support cross-encoder reranking for evidence quality improvement
--
-- Usage:
--   psql $DATABASE_URL -f migrations/004_rerank_scores.sql

-- Add rerank_score column for cross-encoder relevance scores
ALTER TABLE label_evidence ADD COLUMN IF NOT EXISTS rerank_score FLOAT;

-- Add match_method column to track how evidence was matched
-- Values: 'exact', 'fuzzy', 'embedding', 'combined', 'none'
ALTER TABLE label_evidence ADD COLUMN IF NOT EXISTS match_method VARCHAR(20);

-- Create index for efficient sorting by rerank_score
-- NULLS LAST ensures unscored evidence appears after scored evidence
CREATE INDEX IF NOT EXISTS idx_label_evidence_rerank_score
ON label_evidence(rerank_score DESC NULLS LAST);

-- Add comment documenting the columns
COMMENT ON COLUMN label_evidence.rerank_score IS 'Cross-encoder relevance score (0-1), used for evidence ranking';
COMMENT ON COLUMN label_evidence.match_method IS 'Method used to match evidence: exact, fuzzy, embedding, combined, none';
