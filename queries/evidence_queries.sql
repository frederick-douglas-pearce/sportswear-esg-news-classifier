-- ============================================================================
-- Evidence Queries
-- Queries for exploring label evidence and chunk data
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Evidence Overview
-- ---------------------------------------------------------------------------

-- Total evidence excerpts by category
SELECT
    category,
    COUNT(*) AS evidence_count,
    ROUND(AVG(relevance_score), 3) AS avg_relevance
FROM label_evidence
GROUP BY category
ORDER BY evidence_count DESC;

-- Evidence by match quality
-- Shows how well evidence was matched to chunks
SELECT
    CASE
        WHEN relevance_score = 1.0 THEN 'exact match'
        WHEN relevance_score >= 0.85 THEN 'high confidence'
        WHEN relevance_score >= 0.7 THEN 'medium confidence'
        WHEN relevance_score > 0 THEN 'low confidence'
        ELSE 'no match'
    END AS match_quality,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS percentage
FROM label_evidence
GROUP BY 1
ORDER BY MIN(relevance_score) DESC;

-- Evidence with and without chunk links
-- Shows how many evidence excerpts were matched to source chunks
SELECT
    CASE WHEN chunk_id IS NOT NULL THEN 'linked' ELSE 'unlinked' END AS status,
    COUNT(*) AS count
FROM label_evidence
GROUP BY 1;

-- ---------------------------------------------------------------------------
-- Evidence by Brand
-- ---------------------------------------------------------------------------

-- Evidence excerpts for a specific brand (replace 'Nike' with brand name)
SELECT
    le.category,
    LEFT(le.excerpt, 100) AS evidence,
    le.relevance_score,
    bl.confidence_score AS label_confidence
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
WHERE bl.brand = 'Nike'
ORDER BY le.category, le.relevance_score DESC;

-- Evidence count by brand and category
SELECT
    bl.brand,
    le.category,
    COUNT(*) AS evidence_count,
    ROUND(AVG(le.relevance_score), 2) AS avg_relevance
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
GROUP BY bl.brand, le.category
ORDER BY bl.brand, evidence_count DESC;

-- Brands with most evidence
SELECT
    bl.brand,
    COUNT(DISTINCT bl.id) AS labels,
    COUNT(le.id) AS total_evidence,
    ROUND(AVG(le.relevance_score), 2) AS avg_relevance
FROM brand_labels bl
LEFT JOIN label_evidence le ON le.brand_label_id = bl.id
GROUP BY bl.brand
ORDER BY total_evidence DESC;

-- ---------------------------------------------------------------------------
-- Chunk Analysis
-- ---------------------------------------------------------------------------

-- Chunk statistics per article
SELECT
    a.id AS article_id,
    LEFT(a.title, 40) AS title,
    COUNT(ac.id) AS chunk_count,
    SUM(ac.token_count) AS total_tokens,
    ROUND(AVG(ac.token_count)) AS avg_chunk_tokens
FROM articles a
JOIN article_chunks ac ON ac.article_id = a.id
GROUP BY a.id, a.title
ORDER BY chunk_count DESC
LIMIT 20;

-- Chunks with embeddings vs without
SELECT
    CASE WHEN embedding IS NOT NULL THEN 'has embedding' ELSE 'no embedding' END AS status,
    COUNT(*) AS chunk_count
FROM article_chunks
GROUP BY 1;

-- Average chunk size distribution
SELECT
    CASE
        WHEN token_count < 200 THEN 'small (<200 tokens)'
        WHEN token_count < 400 THEN 'medium (200-400 tokens)'
        WHEN token_count < 600 THEN 'target (400-600 tokens)'
        ELSE 'large (>600 tokens)'
    END AS size_bucket,
    COUNT(*) AS chunks,
    ROUND(AVG(token_count)) AS avg_tokens
FROM article_chunks
GROUP BY 1
ORDER BY MIN(token_count);

-- ---------------------------------------------------------------------------
-- Evidence Traceability
-- ---------------------------------------------------------------------------

-- View evidence with its source chunk text
-- Shows the full chain: article → chunk → evidence
SELECT
    LEFT(a.title, 40) AS article_title,
    bl.brand,
    le.category,
    LEFT(le.excerpt, 80) AS evidence_excerpt,
    LEFT(ac.chunk_text, 100) AS source_chunk,
    le.relevance_score
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
JOIN articles a ON bl.article_id = a.id
LEFT JOIN article_chunks ac ON le.chunk_id = ac.id
WHERE le.chunk_id IS NOT NULL
ORDER BY a.title, bl.brand
LIMIT 20;

-- Evidence excerpts that couldn't be matched to chunks
-- These may indicate Claude paraphrased heavily or hallucinated
SELECT
    bl.brand,
    le.category,
    LEFT(le.excerpt, 100) AS unmatched_evidence,
    le.relevance_score
FROM label_evidence le
JOIN brand_labels bl ON le.brand_label_id = bl.id
WHERE le.chunk_id IS NULL
ORDER BY bl.brand, le.category
LIMIT 20;

-- ---------------------------------------------------------------------------
-- Quality Analysis
-- ---------------------------------------------------------------------------

-- Evidence quality by labeling run
-- Compare evidence matching quality across different labeling sessions
SELECT
    lr.id AS run_id,
    lr.started_at::date AS run_date,
    COUNT(le.id) AS total_evidence,
    ROUND(AVG(le.relevance_score), 3) AS avg_relevance,
    SUM(CASE WHEN le.chunk_id IS NOT NULL THEN 1 ELSE 0 END) AS linked_count
FROM labeling_runs lr
JOIN brand_labels bl ON bl.labeled_at >= lr.started_at
                     AND bl.labeled_at <= COALESCE(lr.completed_at, NOW())
JOIN label_evidence le ON le.brand_label_id = bl.id
GROUP BY lr.id, lr.started_at
ORDER BY lr.started_at DESC;

-- Most common evidence patterns by category
-- See what types of evidence Claude extracts for each category
SELECT
    category,
    LEFT(excerpt, 80) AS evidence_pattern,
    COUNT(*) AS occurrences
FROM label_evidence
GROUP BY category, LEFT(excerpt, 80)
HAVING COUNT(*) > 1
ORDER BY category, occurrences DESC;
