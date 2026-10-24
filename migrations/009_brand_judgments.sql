-- Migration: Store each brand's content type from the labeling reply
-- Description: prompt v1.12.0 (D037) gives every brand sent a content_type, but only
--              sportswear brands with an ESG category reached brand_labels, so the type of a
--              product, financial or non-sportswear brand was lost. One row per sent brand
--              the latest reply judged, on every outcome the LLM decides (D038, #192).
--              The labeling pipeline's init_db() creates the table too; apply this anyway.
-- Run: psql $DATABASE_URL -f migrations/009_brand_judgments.sql

CREATE TABLE IF NOT EXISTS brand_judgments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    labeling_run_id UUID NOT NULL REFERENCES labeling_runs(id),
    brand VARCHAR(100) NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    reasoning TEXT NOT NULL,
    confidence FLOAT,
    gated_categories VARCHAR[] NOT NULL DEFAULT '{}',
    prompt_version VARCHAR(20) NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- One judgment per sent brand per article; a relabel replaces the article's rows
CREATE UNIQUE INDEX IF NOT EXISTS ix_brand_judgments_article_brand
ON brand_judgments(article_id, brand);

-- Per-brand selection by type (e.g. a brand's product articles)
CREATE INDEX IF NOT EXISTS ix_brand_judgments_brand_content_type
ON brand_judgments(brand, content_type);

-- Which run wrote a judgment (a backfill's runs are tagged in labeling_runs.config)
CREATE INDEX IF NOT EXISTS ix_brand_judgments_labeling_run_id
ON brand_judgments(labeling_run_id);

COMMENT ON TABLE brand_judgments IS
    'Per-brand content type from the latest LLM reply (prompt v1.12.0 and later, D037/D038). '
    'brand is the sent spelling. The sportswear verdict derives from content_type: '
    'product, financial and business are sportswear.';
