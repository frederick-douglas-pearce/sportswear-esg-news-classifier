# Changelog

This document tracks significant changes to the ESG News Classifier pipeline, including new features, policy updates, and migrations.

## 2026

### 2026-09-07: A failed drift check no longer reports "all classifiers healthy"

FP drift monitoring failed on 221 of 230 scheduled runs since 2026-01-17, and on 217 of those the
workflow simultaneously logged `No action needed - all classifiers healthy` and exited 0. The
safety net had never produced a valid result, so nothing would have reported FP degradation at any
point in the project's life.

**Why a failure looked like health.** Four defects compounded, each of which turned a missing
signal into a benign one:

- `data/reference/fp_reference.parquet` was written 2026-01-17, before `novelty_score` existed.
  `_evidently_drift_check` guarded the novelty *stats* block on `current_data` alone and then read
  `reference_data["novelty_score"]`, so every run raised `KeyError: 'novelty_score'`. The
  column-list block four lines up already guarded both frames; this one did not.
- The script printed `Error running drift analysis: {e}` to **stdout** and returned 1, while the
  workflow logged the command's **stderr** -- producing 221 log lines reading
  `FP drift check failed: ` with the diagnosis nowhere.
- Exit 1 meant *both* "drift detected" and "the analysis raised", and `ScriptResult.success` is
  `exit_code == 0`, so the workflow could not tell them apart and fell back to scraping the
  human-readable report. `evaluate_drift_results` then read
  `context.get("fp_drift_detected", False)` and the report step read
  `context.get("fp_healthy", not context.get("fp_drift_detected", False))` -- with both keys
  absent, `not False` is **True**.
- The EP check ran against a classifier with **zero predictions, ever**. Two empty frames compared
  to each other returned `drift_detected=False` and passed vacuously.

**The rule the fix is built on:** a missing value never resolves to healthy. Concretely:

- `DriftReport` gains a typed `indeterminate` field, set wherever nothing was measured. A verdict
  that was never produced is now distinguishable from one that was produced and was clean.
- A three-value exit-code contract (`src/mlops/exit_codes.py`): `0` no drift, `1` drift detected,
  `2` indeterminate. Drift detected is non-retryable -- it is a result, and before this it was
  retried with exponential backoff before being reported.
- Errors go to stderr with a traceback; the workflow logs the **tail** of that stream, since a
  traceback's diagnosis is at its end (the lesson of #81, same runner).
- `HealthVerdict` (`healthy | degraded | unknown | skipped`) in `src/agent/health.py`. It is
  deliberately **not** a `WorkflowStatus` member: a health verdict is not a lifecycle state, and
  archives carrying `status: unknown` would need migrating.
- The workflow derives its verdict from the exit code, not from scraping stdout; `_parse_drift_output`
  is deleted. A run whose exit code claims a verdict but whose summary is missing or inconsistent
  is treated as `unknown` -- a health claim with no evidence behind it is not a health claim.
- EP is skipped explicitly behind `AGENT_EP_DRIFT_ENABLED` (default off) with a stated reason.
  A config gate rather than a row count: EP-on-hold is a governance decision and zero predictions
  is only its symptom.
- A terminal `fail_on_unknown_verdict` step marks the workflow FAILED when any verdict is not
  explicitly healthy/degraded/skipped -- including absent. It runs *last* so the summary is printed
  and the alert sent first. It is a bridge, to be removed once #74 gives verdicts a first-class
  escalation path (D008).
- The FP reference is regenerated over 90 days (934 rows, complete `novelty_score`, plus the brand
  columns the old file lacked).

Also: a column present in the current data but missing from the reference is now logged as *not
assessed* and recorded in the report details, so fixing the crash does not leave a partial
comparison silently reported as a whole one.

Follow-ups filed rather than folded in: #94 (minimum sample size), #95 (vacuously-healthy when
every check is skipped), #96 (a forgotten EP flag leaves EP dark), #97 (the reference window
overlaps the window it is compared against).

### 2026-09-06: A failed pg_dump no longer records a good backup

`scripts/backup_db.sh` ran `pg_dump | gzip > "$PATH"` under `set -e` with no `pipefail`. A
pipeline reports its *last* command's status, so a `pg_dump` that died mid-stream was masked by
`gzip` exiting 0: a truncated archive was written, "Backup created successfully" was printed,
rotation ran, and `list`/`status` reported the partial file as the latest good backup. The same
shape on the restore path printed "Restore completed successfully!" over partially-loaded data.

**The second half, which `pipefail` alone does not fix.** The `if [ $? -eq 0 ]` handlers were
unreachable -- by two different routes, and only one of them is a `set -e` abort. When `pg_dump`
died the pipeline reported `gzip`'s 0, so `$?` was read, was 0, and the *success* branch ran; when
`gzip` itself failed the pipeline was non-zero and `set -e` aborted before `$?` could be read. The
first is the case this fix is about. Either way the `else` -- including the `rm -f "$DAILY_PATH"`
cleanup -- was dead code, so adding `pipefail` on its own would only make the failure louder while
leaving the truncated file on disk. Each of the three pipelines is now the `if` **condition**,
which is exempt from `set -e`, so the cleanup branch actually runs.

Also: the pre-restore safety copy aborts the restore and removes its partial file rather than
proceeding; the `ls *.sql.gz | awk` listing is guarded with `|| true`, the only *top-level*
pipeline `pipefail` would newly abort; and `BACKUP_SIZE=$(du -h ... | cut -f1)` is guarded with
`|| BACKUP_SIZE="unknown"`. That last one is the subtle site: it is the only plain (non-`local`)
assignment whose *right-hand side is a pipeline* -- other plain assignments exist, but `pipefail`
has nothing to reach in them -- so under `pipefail` a `du` failure would abort inside the success
branch and invert the invariant this change exists to establish: a good archive on disk, a non-zero
exit, no rotation, and the cleanup unreachable in the `else`. The `local var=$(cmd | cmd)` sites in `rotate` and
`status` are unaffected: `local` returns its own exit status, so the pipeline's never reaches
`set -e`.

`check_container`'s `docker ps | grep -q` also changes behaviour under `pipefail` -- it is a change
of *value* rather than an abort, since there the pipeline's status is the branch condition. It is
checked, measured and filed as #93 rather than fixed here. The inversion is a *race*, not a size
threshold: it fires whenever `docker ps` is still writing when `grep -q` matches and stops reading.
Exceeding the ~64 KiB pipe buffer is merely the reliable way to force it. What makes it unreachable
here is that real `docker ps` emits its whole list -- 30 bytes, two running containers -- in one
buffered write and exits before `grep` can close the pipe. Note this is a hazard the change
*introduces*: on `main` that pipeline runs under plain `set -e`, where its status cannot flip the
branch. It is deferred rather than pre-existing, and rewriting it is a behaviour change none of
this issue's acceptance criteria ask for.

Instance 7 of the `silent-success` class (#72), and the only one whose outcome is data loss rather
than a missed alert. Regression tests in `tests/test_backup_db_script.py` stub `docker` and `du` on
`PATH` to force a mid-pipeline failure, asserting both non-zero exit **and** no leftover archive -- a test
asserting only the exit code passes against a fix that leaves the partial file. (#89)

### 2026-09-05: Stop the labeling retry from erasing a run's results

The `daily_labeling` report for 2026-09-05 read `0 processed / 0 labeled / 0 failed`
for a run that had labeled 5 articles, skipped 6, deduplicated 1 and failed 1.

**Mechanism.** One article (`e6ae677f`) failed with `Failed to parse LLM response`.
`label_articles.py` returned `0 if not stats.errors else 1`, so a single bad article
failed the whole script. The agent runner read exit 1 as a total failure and retried
the identical command — but labeling consumes the rows it selects, so the retry found
nothing pending, exited 0 in 2.25s, and its empty stdout replaced the real numbers.
`check_labeling_quality` then derived `error_rate` from those zeros, which made
`high_error_rate` unreachable on exactly the runs that had failures. The `labeling_runs`
row held `status='partial'`, `articles_processed=12` the whole time; nothing read it.

Present in **13 of the last 230 daily runs (~6%)**: 20260118, 20260123, 20260210,
20260211, 20260212, 20260225, 20260307, 20260417, 20260419, 20260420, 20260423,
20260617, 20260905.

**Changes (issue #81):**
- `scripts/label_articles.py` returns a distinct exit code (`2`) when the batch is
  spent — some articles reached a terminal status and none are left pending, so a
  retry can only erase the result. `src/labeling/exit_codes.py` holds the contract;
  the runner accepts code 2 as final via `non_retryable_exit_codes`. Exit `1` still
  means "a retry still has articles to work through" and keeps its retries (#51).
  The decision reads `articles_processed` (terminal, spent) against
  `articles_left_pending` (raised before any status update, still retryable) — the
  two are tracked separately because they have different bases.
- `labeling_runs` gained `articles_labeled`, `articles_skipped`,
  `articles_false_positive`, `articles_failed`, `articles_deduplicated` and
  `articles_left_pending` (migration `007_labeling_run_outcomes.sql`), so a run row
  describes itself. `check_labeling_quality`, `generate_report` and the LLM analysis
  read those columns instead of scraping stdout, and report which source they used.
  A dry run writes no run row, so stdout remains the fallback.
- Only rows with `completed_at` set are summed. A row still `running` was written by
  a process killed mid-flight (the 1800s subprocess timeout), and its zeros are the
  absence of a result rather than a result. The window is bounded at both ends so a
  concurrently started labeling run is not attributed to the workflow.
- Quality metrics carry `partial_failure`, which fires on a run that did work *and*
  errored even when the error rate is under the 10% threshold (1/12 was), and
  `metrics_degraded`, which fires when a non-dry run produced no usable run row.
  Both now reach the emailed notification, not just the console summary.
- The runner logs the **tail** of stderr on failure, not the first 500 characters.
  Tracebacks are at the end; the head captured only the startup banner, so the
  original diagnosis had to be reproduced by hand.
- LLM pricing moved to one table in `src/labeling/config.py`. Spend came from the
  database ($1/$5 per MTok) while the FP savings estimate came from stdout ($3/$15),
  so a report could claim it saved more than it spent. Model-aware pricing is #54.

The parse failure that triggered this is #82; replacing prose-JSON parsing with
structured output is #83. Historical `labeling_runs` rows keep `0` in the new
columns — the breakdown was never persisted and cannot be reconstructed.

**Migration:** `psql $DATABASE_URL -f migrations/007_labeling_run_outcomes.sql`

### 2026-09-05: Recover LLM responses that quote the article verbatim

Article `e6ae677f` failed labeling with `Failed to parse LLM response`. The model had
copied a quoted passage straight out of the article into an evidence excerpt:

```
"I think the store expansion will slow down," Morningstar analyst David Swartz said, ...
```

The article's opening quote mark merged with the JSON string opener, leaving a bare
`"` after `slow down,`. JSON ends the string there and chokes on ` Morningstar`.
`_fix_json` only repaired trailing commas and unquoted keys, so the whole response
was discarded. Evidence excerpts are verbatim article text, which makes this the
malformation most likely to recur.

**Changes (issue #82):**
- `_escape_interior_quotes` walks the document tracking string state. A quote not
  followed (past whitespace) by `,` `}` `]` `:` or `"` is interior text and gets
  escaped; otherwise it closes the string.
- `_recover_json` tries that repair alone before the full `_fix_json`. The regex
  passes are not string-aware — the unquoted-key pattern rewrites `, revenue:` inside
  a string value — so they are a fallback, which recovers documents the combined pass
  destroyed.
- Parse failures log the text around the error position. The first failure is logged
  against the model's original output, since repairs shift every offset after the
  edit and a window from repaired text can point at parser-introduced damage.

**Why `"` is a terminator.** Valid JSON never places two strings side by side, so
including it costs nothing on well-formed input. Excluding it made `["a" "b"]` — a
dropped comma — escape both inner quotes, merge the two excerpts into one corrupted
string, and *validate*, sending a garbled excerpt to the published feed. Failing to
parse is recoverable; silently rewriting an excerpt is not.

A quote that is both interior and followed by one of those characters (`"He said
"yes", loudly."`) still defeats the heuristic. It stays unparseable rather than
parsing wrongly. Structured output, which removes the class, is tracked in #83.

### 2026-07-01: Migrate labeling model to Claude Haiku 4.5 (cost downgrade)

Migrated the labeling model from Claude Sonnet 4.6 (`claude-sonnet-4-6`) to Claude
Haiku 4.5, pinned to the dated snapshot `claude-haiku-4-5-20251001` for
reproducibility (labels persist `model_version`). Motivation: cost — Haiku 4.5 is
~67% cheaper ($1/$5 vs $3/$15 per MTok), shares the Sonnet 4.6 tokenizer family
(no token-inflation penalty), and still supports `temperature=0` (the determinism
lever the scorecard relies on).

**Why not Sonnet 5:** its ~30% tokenizer inflation erased the introductory
discount (net *more* expensive than 4.6 after the intro window ends 2026-08-31)
and it rejects non-default `temperature` with a 400 — removing determinism.
Evaluated and rejected in favor of the Haiku cost-downgrade.

**Validation (issue #53):** a retrospective contrastive eval
(`scripts/eval_model_migration.py`) re-labeled 162 articles against the Sonnet 4.6
/ v1.9.0 baseline. Results: **0.0% newly-labeled rate** (no scorecard-inflation
risk — critical, since the FP/EP pre-filters are disabled so the labeler is the
only junk gate), **0 parse/truncation failures**, **93.9% exact sentiment
agreement** (≥ 4.6's 92.4%), and outcome disagreements dominated by Haiku
*correctly* rejecting brand-collision / financial / marketing false positives that
4.6 mislabeled. Statistical power on positives is low (30 labeled, 1 human
anchor), compensated by a **post-flip drift-monitoring window**.

**Changes:** new prompt version **v1.10.0** (byte-identical text to v1.9.0, model
only), promoted to production; all five Sonnet 4.6 call sites swapped to Haiku
(labeling — eval-gated; agent analysis, workflow-learning, experiment reflection —
ungated, low-stakes). Reusable model-migration skill tracked in #54; Allbirds
corporate-pivot labeling decision-boundary deferred to #55.

### 2026-06-01: Fix Jekyll build failure from mojibake control characters in feed

The website's GitHub Pages "Deploy site" build began failing with
`_data/esg_news.json: control characters are not allowed at line 1 column 1
(Psych::SyntaxError)`. Jekyll parses `_data/*.json` with its YAML parser (Ruby
Psych), which rejects C1 control characters (U+0080–U+009F).

**Root cause:** Scraped article text contained Windows-1252 smart punctuation
(’ “ ” —) stored as raw C1 control bytes — *mojibake* introduced when
`newspaper` misdetected a page's charset. This pre-existing data problem was
*unmasked* by the same-day prettier-JSON change (below): the old `json.dump`
default escaped non-ASCII as `\uXXXX` (harmless to YAML), whereas
`dumps_prettier` emits raw UTF-8, so the C1 bytes reached the committed file.

**Fix (defense-in-depth):**
- New shared `src/data_collection/text_normalize.py` (`normalize_text`,
  `repair_mojibake`, `find_illegal_chars`) repairs mojibake via `ftfy` and
  strips YAML-illegal control characters. Idempotent; used at every layer.
- **Ingest:** the scraper and `Database.upsert_article` normalize content,
  title, and description before storage.
- **One-time repair:** `scripts/repair_text_encoding.py` (`--dry-run`) repairs
  existing rows in `articles`, `article_chunks`, `brand_labels`, and
  `label_evidence`.
- **Export guard:** `export_website_feed.guard_feed_data` repairs any residual
  control characters in the assembled feed (and Atom fields) just before
  serialization, logging a non-blocking warning (with article id + field path)
  whenever it has to — a signal that ingest-time normalization missed a case.
- **Validator:** `website_export.validate_export` now counts articles correctly
  for the dict-shaped feed (previously always reported 0) and YAML-parses the
  feed (PyYAML mirrors Jekyll's Psych) so a build-breaking feed fails validation
  and blocks the push instead of shipping silently.

### 2026-06-01: Prettier-compatible JSON feed export

The website feed JSON is now emitted in Prettier's formatting directly by the
generator, fixing a `prettier --check` CI failure on the GitHub Pages repo that
began 2026-05-28.

**Root cause:** `export_website_feed.write_json` always used
`json.dump(indent=2)`, which expands every array (including single-item arrays)
onto multiple lines. Previously the `website_export` cron masked this by running
`npx prettier --write` before committing. The 2026-05-27 worktree hardening
(commit `b14994f`) moved the cron into a dedicated `-feed` git worktree that
lacks `node_modules`, so Prettier could no longer resolve the
`@shopify/prettier-plugin-liquid` plugin declared in `.prettierrc` and silently
failed (warning-only), committing the raw multi-line JSON.

**Fix:** `write_json` now serializes via `dumps_prettier`, a small encoder that
reproduces Prettier's JSON output (printWidth 150, flat containers when they
fit / break otherwise, `{ }` brace spacing, raw UTF-8, exponent normalization).
Validated byte-for-byte against Prettier 3.1.1 and 3.8.3 over the full
production feed. The now-redundant `npx prettier --write` step was removed from
the `website_export` workflow.

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
