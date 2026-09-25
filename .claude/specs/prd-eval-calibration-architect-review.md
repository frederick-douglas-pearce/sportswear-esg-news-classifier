# Architect Review — Epic: calibrate eval/monitoring against human review

Review of [prd-eval-calibration.md](prd-eval-calibration.md) against the live code (2026-09-24).
Citations are `file:line` against the tree at `999c25d`.

Coordinator verification: the config/registry threshold mismatch in #115 was confirmed —
`models/fp_classifier_config.json` threshold `0.475`, no `version` key; `models/registry.json`
production `v2.5.0` threshold `0.5429`.

---

## Cross-cutting findings

**C1 — `skipped` is a POSITIVE for the FP task.** The FP positive class is "genuine sportswear-brand
article". Ground-truth mapping from `articles.labeling_status`:
- positive = `{labeled, skipped}` (`skipped` = genuine brand, no ESG content, `fix_label.py:169`)
- negative = `{false_positive}`
- exclude = `{unlabelable, deduplicated, pending, chunked, embedded}`

Treating `skipped` as negative fabricates a large false-positive rate. Put this mapping in the Story 4 AC verbatim.

**C2 — the join is circular unless split by `action_taken`.** `false_positive` is written from two
sources: the FP classifier's own skip (`action_taken='skipped_llm'`, `pipeline.py:839-855`) and the
LLM's `is_sportswear_brand=False` verdict (`pipeline.py:952-965`). Only the second is independent
ground truth. The realized-performance query must join `classifier_predictions` where
`action_taken='continued_to_llm'` → precision of the "continue" decision (measurable today). Recall
is not measurable today because FP-skipped rows were labeled `false_positive` by the classifier
itself — the selection bias the exploration sample fixes.

**C3 — the exploration sample needs no schema change.** Routing lives at `pipeline.py:404`
(`should_continue = result.probability >= threshold`) in `_run_fp_prefilter_batch`. Override a small
deterministic random share of would-be-skips to `should_continue=True`, tag
`skip_reason='exploration_sample'`. The monitor identifies the cohort with
`action_taken='continued_to_llm' AND probability < threshold_used` (`threshold_used` is per-row,
`models.py:297`). The monitor is read-only and cannot create the sample — routing must be in the pipeline.

**C4 — minimum sample / CI is the whole game.** Wilson score interval on realized recall (denominator
= LLM-confirmed positives; for skip-recall, explored positives). Fire `retrain` only when the CI is
distinguishably below baseline, gated on a minimum count on the correct denominator. **Below the
floor → `HealthVerdict.SKIPPED` with reason, never `UNKNOWN`** (`health.py:80-86`); UNKNOWN fails the
run at the gate and would fail every night during data accrual.

**C5 — drift stays informational by changing the decision layer, not `monitoring.py`.**
`monitoring.py:660` is a correct distribution signal — leave it. The retrain reflex is in the
workflow: `evaluate_drift_results` (`drift_monitoring.py:318`) and `send_drift_alerts`
(`drift_monitoring.py:356-366`). Tiering = a separate performance verdict + rewiring those two
functions. Drift-mechanism bugs stay with #72/#136.

**C6 — build order `4-L1 → #115 → 1 → 4-L2 → 2 → 6 → 3 → 8 → 5 → 7` holds, with refinements:**
(a) split 4-L1 so the email-noise fix ships first with zero data dependency, exploration routing
ships in parallel to start accrual, the performance query lands once data exists; (b) keep Story 2's
sample and Story 4's exploration sample separate; (c) runbook sections are written with each story —
slot "8" is only the validated consolidation.

---

## #115 — FP predictions all `model_version='unknown'` (ship 2nd)

**Root cause.** `get_model_info()` (`src/deployment/base.py:240-264`) returns no `version`;
`ModelInfoResponse` (`scripts/predict.py:253-262`) has no `version` field; `ClassifierClient`
falls back to `"unknown"` (`classifier_client.py:148`); `pipeline.py:399` reads
`model_info.get("version","unknown")`.

**Approach.** Make the registry `production` pointer authoritative: loader (`base.py:57`) pulls
version + threshold from `models/registry.json`; surface `version` in `get_model_info()` and
`/model/info`. `pipeline.py:399` then records the real version.

**Owner flag — config/registry mismatch.** Config threshold `0.475` vs registry `v2.5.0` `0.5429`.
The skip decision uses `labeling_settings.fp_skip_llm_threshold` (0.53), not the model config, so
0.475 may be dead for the skip path but live for the API's `is_sportswear` boolean. Surface; don't
silently "fix".

**Backfill impossible.** Existing rows can't be attributed to a version. Document as an irrecoverable
gap; version-attributed metrics apply going forward only.

**Tests.** `/model/info` returns non-`unknown`; saved `ClassifierPrediction` carries the version
(mock client); DB-gated persistence check. **Size:** S–M, one PR.

---

## Story 1 — Correction audit log

**Approach.**
- `migrations/008_label_corrections.sql`; table per AC plus nullable `source_ref` (so Story 3 needs no
  second migration). Indexes on `article_id`, `created_at`, `review_session_id`, `reason_code`. Model
  after `ClassifierPrediction` (~`models.py:326`); `VALID_REASON_CODES` beside
  `VALID_LABELING_STATUSES` (`models.py:32`). `review_session_id` nullable FK (activated by Story 2).
- `scripts/fix_label.py`: `update_articles` (`L118-159`) already uses one session + one commit — add
  `session.add(LabelCorrection(...))` in the loop before commit (transactional for free). Add
  `--reason` (required), `--reason-code` (choices), `--reviewer` (default from git `user.email`/env).
  Add brand-level subcommands for `brand_labels` fields (`models.py:159-169`).
- `.claude/skills/review-labels/SKILL.md` Step 5 (`L312-325`): all corrections through `fix_label.py`,
  forbid raw `UPDATE`. Update CLAUDE.md "Label Corrections".

**Design.**
- **D-1.1** — required `reason_code` enum (with `other`) *and* required free-text `reason`. Enum makes
  aggregation cheap; prose alone can't be counted.
- **D-1.2** — record prior timestamp in `old_value` and stop clobbering `labeled_at` (set `skipped_at`
  only if null). Overwrites at `fix_label.py:149-151` corrupt the timeline review-labels keys off.
- **D-1.3** — one audit row per changed field; directly aggregatable by `reason_code`/`field`.

**Tests.** DB-gated (`RUN_DB_TESTS`, `tests/test_scorecard_database.py:24-29` pattern): audit row on
status change; rollback writes nothing; brand sentiment path; reason-required rejection (unit).
**Size:** M, one PR.

**Risks.**
- `is_sportswear_brand` is not a stored column — `brand_labels` holds only sportswear brands
  (`pipeline.py:1030`). "Is/isn't sportswear" corrections are status changes + `brand_labels` row
  add/delete; the audit model needs a row-created/deleted kind.
- **Orphaned `brand_labels` on `labeled→skipped/false_positive` still feed the scorecard export.**
  Decide: cascade (delete/flag) or document as a known gap. Scorecard-correctness issue.

---

## Story 2 — Review session record + unbiased random sample

**Approach.** `migrations/009_review_sessions.sql`: `review_sessions` (id, started_at, completed_at,
prompt_version, reviewer, trigger, notes, sample_frame, sample_size, articles_reviewed,
articles_confirmed); activate the FK; add `review_session_sample(session_id, article_id,
was_confirmed, correction_id nullable)` for per-article outcomes (the denominator). Drive from the
review-labels skill; corrections via `fix_label.py --review-session-id`. No scheduler (that was #6's
`review_queue.py`, cut).

**Design.**
- **D-2.1** — DB table, confirmed.
- **D-2.2** — fixed small n (~10–20/session), accumulated; proportional sampling is unstable at small n.
- **D-2.3** — review against existing labels; **do NOT merge with Story 4's exploration sample**.
  Different frames, estimands and stages; drawing Story 2 from would-be-skips biases it toward the
  FP-negative tail. Legitimate coordination is the reverse: over-sample explored articles for human review.

**Risk.** A frame of only `labeled` excludes the LLM's negative decisions (`skipped`,
`false_positive`) where false-skip/false-FP errors live. For "LLM error rate per prompt version" use
all articles that reached the LLM (`continued_to_llm`) or `{labeled, skipped, false_positive}`.

**Tests.** DB-gated session lifecycle; seeded uniformity test; error-rate query on a fixture.
**Size:** M, one PR.

---

## Story 3 — Backfill corrections from transcripts

**Approach.** One-off `scripts/backfill_corrections.py`. Parse transcripts for the `update_articles`
stdout pattern (`fix_label.py:153`) and raw `UPDATE` SQL; write rows with `source='reconstructed'`,
`source_ref` = transcript + timestamp, inferred `reason_code` else `reconstructed_unknown`.

**Dedup** on `(article_id, field, new_value)` keeping the earliest timestamp; reconcile against the
~34 upper bound. Cross-check against current DB state and flag disagreements rather than overwrite.

**D-3.1** — automate the `fix_label.py` parse; hand-enter the 3 raw-SQL cases or document the gap.

**Risk.** The stdout line has **no UUID**, only a 60-char title prefix — key on a UUID from nearby
`show <id>` commands when present, title-prefix otherwise, flag ambiguous matches. These rows feed
ground truth for Stories 4/5; a mis-mapped row is worse than a missing one. **Timebox hard or defer**;
downstream consumers must be able to filter `source='reconstructed'`.

**Tests.** Pure-function parser/dedup fixtures; DB-gated load. **Size:** S–M, one PR.

---

## Story 4 — Realized-performance monitor + tiered alerts

**Approach (Layer 1 in three increments):**
1. **Email fix (first, no data dependency).** `evaluate_drift_results` (`drift_monitoring.py:283-330`):
   drift-only `DEGRADED` → informational recommendation. `send_drift_alerts` (`L345-426`): stop
   emailing on the drift verdict; drift is report-only (`generate_drift_report`, `L442-462`).
2. **Exploration routing (parallel, to accrue data).** Override at `pipeline.py:404` (C3);
   `FP_EXPLORATION_RATE` setting; deterministic by hash of `article_id`.
3. **Performance query + verdict (once data exists).** `check_fp_realized_performance` step producing
   `fp_performance_verdict` from a named query over `classifier_predictions ⋈ articles` (C1/C2).
   Compose: perf `DEGRADED` → retrain (cite drop, n, CI); perf `HEALTHY` + drift `DEGRADED` →
   informational; perf `SKIPPED` + drift `DEGRADED` → "shifted; not yet measurable".

**HealthVerdict integration.** Same vocabulary (`health.py:48-86`); add to
`fail_on_unresolved_verdicts` (`drift_monitoring.py:550-553`, factory `base.py:112-206`);
insufficient sample = SKIPPED (C4); emit `.value` strings. No new table — metrics recomputed from the
query or read from the run archive (`src/agent/archive.py`). Design the query around `classifier_type`
so the planned ESG multi-label classifier can reuse it.

**Design.**
- **D-4.1 (crux) — frozen baseline.** Freeze a *reference window*, not a scalar. Registry offline test
  metrics (e.g. `test_recall=0.9966` for v2.5.0) are a different instrument and population
  (held-out-vs-human) — comparing to realized-vs-LLM would be apples-to-oranges. After promotion,
  record `{model_version, prompt_version, window_start, window_end, n}` and recompute the baseline
  with the same query as the live window. Location options: `models/registry.json` (reuses
  `register_model.py`, but writes generated data into a hand-curated committed file) vs a small DB
  table or `data/reference/realized_baselines.json`. Architect leans registry; table is the fallback.
  Until #115, the baseline is version-blind and must say so.
- **D-4.2** — informational drift does not email; email only for performance `DEGRADED` and check
  failure (`UNKNOWN`). Runbook covers the burn-in period: drift-only → manual `/review-labels` spot-check.
- **D-4.3** — L1/L2 split plus the three L1 increments.
- **D-4.4** — `core_metrics_drifted`/`brand_metrics_drifted` exist (`monitoring.py:602,607`) but aren't
  surfaced (`monitor_drift.py:122-130`, `REQUIRED_SUMMARY_FIELDS` `drift_monitoring.py:45-52`).
  **#136 owns the plumbing; Story 4 consumes it.** If #136 hasn't landed, add as an optional field.
- **D-4.5 (elevated) — exploration.** Placement in the pipeline (C3). Fixed small ε
  (`FP_EXPLORATION_RATE`, e.g. 0.05–0.10), chosen so the window accrues ≥ the C4 floor of explored
  positives; trade-off is LLM cost vs CI width. Don't reuse Story 2's sample.

**Tests.** Fixture join asserting C1 mapping and C2 filter; drift-only → informational; measured
degradation → retrain with cited drop; archive carries drifted columns; recall returns SKIPPED below
floor. **Size:** L, multiple PRs.

**Risk.** Between shipping tiering and accruing exploration data, real regression is effectively
unmonitored. Accept (strictly better than false alarms) and document in the runbook.

---

## Story 5 — Drift-alarm calibration backtest

Notebook for exploration + a committed decision record. Join archived drift history to Story 4
realized-performance history; compute each candidate rule's own precision/recall vs actual drops.
- **D-5.1** — gate on a count (e.g. ≥100 windows with realized recall at n ≥ `DRIFT_MIN_SAMPLE_SIZE`), not a date.
- **D-5.2** — decision lands in `docs/`/CHANGELOG, not a notebook cell.

**Risk.** The inputs aren't recorded historically — the clock starts when Story 4/#136 begin
recording, not from the existing archive. **Size:** M–L, time-gated; do not schedule now.

---

## Story 6 — Gold eval set as promotion gate

`data/evaluation/golden_set.jsonl` + thin `scripts/eval_gold_set.py`. Records embed a content
snapshot + expected per-brand category/sentiment/FP decision + `failure_class`, seeded from
confirmed corrections. Runs `ArticleLabeler(prompt_version=…)`, reports per-class agreement and
regression count with `--compare-to`. Decisions only, not evidence text.
- **D-6.1** — start ~30–50 records; grow via a `gold_candidate` flag set at Story 2 review time.
- **D-6.2** — spike: check what closed #38 left behind; reuse only if it computes per-example agreement.
- **D-6.3** — local pre-promotion gate inside `model_training`, not CI (paid APIs, secrets).

**Tests.** Fixture set + stubbed labeler. **Risk.** Comparison runs both versions (2× API cost);
seeding from corrections biases toward past failure modes. **Size:** M, one PR.

---

## Story 7 — MLflow logging (optional, last)

Behind `MLFLOW_ENABLED`, via `src/mlops/tracking.py`, keyed by `model_version` (#115). Key test:
no-op and no exception when off. **Size:** S. Fine to drop.

---

## Story 8 — Runbook

`docs/` runbook linked from alert emails (`src/agent/notifications.py`). Sections written with each
story; slot 8 = consolidation + validation. Include the burn-in note. **Risk:** validation needs a
replay path (e.g. `monitor_drift.py` on a crafted window) so it isn't blocked on a live alert.
**Size:** S on top of incremental sections.

---

## Forward-compatibility

- #136 and Story 4 both touch `print_summary_json`/`REQUIRED_SUMMARY_FIELDS` — resolve ownership up front.
- Story 4's performance verdict plugs into the #72 `silent-success` contract; SKIPPED-vs-UNKNOWN keeps
  it from reintroducing vacuous health.
- Planned ESG multi-label classifier: keep the realized-performance query `classifier_type`-generic.
- D-4.1 done as a scalar hardcode would rot exactly like the thresholds this epic replaces.

---

## Addendum — Stories 9 & 10 (filed as #154, #155), 2026-09-24

Reviewed against `e391972`. Full resolved design is in the issue bodies; summary:

- **Only the FP export is contaminated** (`export_training_data.py:52-118`). `skipped_llm` → `false_positive` exists only on the FP path (`pipeline.py:414`, `:832`); EP and esg-labels exports use LLM-written statuses only.
- **#154: derive provenance, do not store it.** It is a pure function of `classifier_predictions.action_taken`, `label_corrections` and `review_session_sample.was_confirmed`; a column would drift (cf. #135). Same predicate as #144 C2, defined once in `src/data_collection/database.py`. The owner-attested cutoff is an export parameter, not provenance.
- **The carve-back must read `review_session_sample.was_confirmed`**: `fix_label.py` writes nothing on a confirmed-correct skip.
- Exclude by default; `--include-classifier-labels` for ablation. Re-run fp2 tuning after (class balance). Attach an as-of contamination caveat to v2.5.0; headline recall minimally affected, negative-class precision and threshold at risk.
- **#155:** `ArticleLabeler.label_article()` (`labeler.py:227`) directly: no chunk/embed/DB. Empty `brand_analyses` = POSITIVE (`pipeline.py:924-930`). LLM census of all ~390; human reviews all disagreements + ~60 agreements stratified by probability band (0/60 → <5% at 95%) + census of the pre-cutoff cohort if ≤ ~100; per-month error rates. LLM opinions to a one-off output file, not a table.
- `review-fps` is a separate thin skill writing the same tables; recurring via #142. Add `fp_skipped` to #146 `sample_frame`.
