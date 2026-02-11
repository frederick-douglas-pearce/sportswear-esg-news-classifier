# Changelog

This document tracks significant changes to the ESG News Classifier pipeline, including new features, policy updates, and migrations.

## 2026

### 2026-02-11: Scorecard History Storage

Added database storage for daily scorecard snapshots, enabling historical trend analysis and brand performance tracking over time.

**New tables:**
- `scorecard_snapshots` - Daily snapshot metadata (period, article counts, dedup settings)
- `scorecard_brand_scores` - Per-brand scores with category breakdown, rank, and medals

**New module:** `src/scorecard/`
- `ScorecardDatabase` class with methods for saving and querying scorecard history
- Query methods: `get_brand_score_history()`, `get_medal_history()`, `get_deduplication_stats()`

**Integration:**
- `website_export` workflow now saves scorecard to database after each export
- Step is skipped in dry-run mode
- Uses upsert semantics (safe to re-run on same day)

**Key design decisions:**
- Only brands with labeled articles during the period are stored (not all 50 tracked brands)
- Full brand coverage analysis can be achieved by joining with `brand_labels` table
- All category scores stored (E, S, G, D) plus total, rank, and medal status

**Migration:** `psql $DATABASE_URL -f migrations/006_scorecard_history.sql`

**Query examples:** See `queries/scorecard_queries.sql` for trend analysis, medal history, etc.

### 2026-01-27: Domain Blacklist for Data Collection

Added ability to block low-quality news sources from data collection.

**New features:**
- `BLOCKED_DOMAINS` list in `src/data_collection/config.py`
- `is_blocked_domain()` helper function in `collector.py`
- Articles from blocked domains filtered during API collection
- `articles_blocked_domain` stat tracked in `CollectionStats`

**Initial blocklist:** `openpr.com` (AI-generated market reports with no real journalism)

**To add a blocked domain:** Edit `BLOCKED_DOMAINS` in `src/data_collection/config.py`

### 2026-01-26: Sustainability Scorecard for Website

Added a "Sportswear Sustainability Scorecard" to the ESG news website, ranking brands based on recent news sentiment.

**New features:**
- Scorecard calculation in `scripts/export_website_feed.py`
- Article deduplication using sentence embeddings (all-MiniLM-L6-v2)
- Top 3 performers (positive scores only) with medal badges
- Back 3 performers (negative scores only)
- Category breakdown per brand (E, S, G, D)
- Date range filter on website (7/14/30 days, All, custom)

**Scoring:** Positive=+2 pts, Neutral=+1 pt, Negative=-1 pt

**CLI options:**
- `--no-scorecard` - Skip scorecard generation
- `--scorecard-period-days N` - Custom period (default: 14)
- `--no-dedupe` - Disable article deduplication
- `--similarity-threshold N` - Custom similarity threshold (default: 0.75)

### 2026-01-20: Cross-Encoder Reranking for Evidence Quality

Integrated cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve evidence matching quality. The reranker jointly encodes (excerpt, chunk) pairs for more accurate relevance scoring than bi-encoder embeddings.

**New features:**
- `src/labeling/reranker.py` - CrossEncoderReranker class with lazy model loading
- `rerank_score` and `match_method` columns in `label_evidence` table
- Website export sorts evidence by `rerank_score` (falling back to `relevance_score`)
- Configurable top-N evidence per category in export (`--top-n-evidence`)
- Backfill script for existing articles: `scripts/backfill_rerank_scores.py`

**Configuration:**
- `RERANK_ENABLED=true` (default) - Enable/disable reranking
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` - Model to use
- `RERANK_TOP_K=10` - Candidates to rerank per excerpt
- `RERANK_WEIGHT=0.6` - Weight for combined score: `(1-w)*initial + w*rerank`

**Migration:** `psql $DATABASE_URL -f migrations/004_rerank_scores.sql`

### 2026-01-16: Expanded Stock Article Classification Guidelines

Clarified criteria for distinguishing between `false_positive` (pure metrics) and `skipped` (substantive content) for stock/finance articles. See [LABELING.md](./LABELING.md#stock-article-classification) for detailed guidelines.

### 2026-01-14: Clarified is_sportswear_brand Policy for Stock Articles

`is_sportswear_brand` is about **substantive content**, not just identity:
- `true` → Article has substantive content (products, business news, strategy, analyst commentary with reasoning)
- `false` → Brand refers to something else OR pure stock metrics only (no substantive content)

---

## 2025

### 2025-12-29: MLOps Improvements

Added `src/mlops/` module: MLflow tracking, Evidently drift detection, webhook alerts, daily monitoring cron job.

### 2025-12-29: FP Classifier Batch API

Optimized to batch API calls (N articles → 1 call). Fixed Docker deployment issues.

### 2025-12-28: FP Classifier Pre-filter Integration

FP classifier as optional pre-filter: articles with probability < threshold marked `false_positive`, skip LLM.
- `FP_CLASSIFIER_ENABLED=true`, `FP_SKIP_LLM_THRESHOLD=0.5`
- Migration: `psql $DATABASE_URL -f migrations/002_classifier_predictions.sql`

### 2025-12-26: Added skipped_at Timestamp & Tangential Brand Mention Guidance

Added `skipped_at` column for tracking. Updated prompts to identify false positives for tangential brand mentions (biographical, stock-only, incidental references).

Migration: `ALTER TABLE articles ADD COLUMN skipped_at TIMESTAMP WITH TIME ZONE;`
