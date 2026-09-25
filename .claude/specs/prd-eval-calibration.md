# Epic draft — Calibrate the eval/monitoring framework against human review

> FILED as epic **#152**; stories **#140–#151**, **#154** (Story 9), **#155** (Story 10);
> **#115** in the milestone. Spec tracked by PR **#153** on branch
> `docs/152-eval-calibration-spec` (edit the working-tree spec file). All stories now filed;
> [`eval-calibration-new-issues.md`](eval-calibration-new-issues.md) is superseded by #154/#155.
>
> **Cross-reference note:** confirmed numbers — **#143 = Story 1** (audit log), **#146 =
> Story 2** (review session), **#142 = Story 4·L1b** (exploration sample), **#154 = Story 9**
> (label provenance), **#155 = Story 10** (`review-fps`). Annotate the rest when filing follow-ups.
>
> **Single source of truth:** this spec. Architect rationale + file:line citations in
> [`prd-eval-calibration-architect-review.md`](prd-eval-calibration-architect-review.md)
> (Stories 9 & 10 under "## Addendum — Stories 9 & 10").
>
> **Change note 1:** original "Story 0 — preserve transcripts" dropped (`cleanupPeriodDays:
> 3650`); optional snapshot folded into Story 3.
>
> **Change note 2 (owner):** Q1–Q5 resolved.
>
> **Change note 3 (architect review, owner exceptions):** baseline in a small DB table (not
> registry.json); orphaned `brand_labels` documented not deleted (data hygiene); threshold
> mismatch its own `priority:high` bug; #115 kept small.
>
> **Change note 4 (new scope):** added Story 9 (label provenance + stop exporting classifier-
> only labels) and Story 10 (one-time review of classifier-only negatives + `review-fps`
> skill) — the FP classifier trains and is evaluated on its own predictions.
>
> **Change note 5 (architect review of Stories 9 & 10 folded in, 2026-09-24):** provenance is
> **derived at query time** to one of **`{llm, pre_classifier, fp_classifier, human}`** (per the
> filed #154; no `label_source` column, no new table); the owner-attested cutoff is an **export
> parameter only**, not provenance, and the per-month hypothesis Story 10 tests; exclude
> classifier-only labels **by default with `--include-classifier-labels`**, carving human-
> confirmed skips back via `review_session_sample.was_confirmed`; **only the FP export** is
> affected; Story 10 uses `ArticleLabeler.label_article()` directly (no DB mutation) and writes
> LLM opinions to a one-off file. `D-9.n`/`D-10.n` replaced with the resolved design below.

---

## Epic issue (tracker) — #152

**Title:** `Epic: calibrate eval/monitoring against human review — stop drift-only retrain alerts`

**Labels:** `evaluation`, `mlops` · **Milestone (new):** `eval-calibration`

### Backlog actions decided (2026-09-24)

- **Close #1** (golden set) → superseded by Story 6.
- **Close #6** (formalize human review) → superseded by Stories 1+2.
- **Keep #3 open**, deferred (subsystem not adopted; parts fold into 4/5).
- **Keep #5 open**, deferred (LLM-as-judge — different, unvalidated ground-truth source).
- **File the NEW FP-threshold SSOT bug** (below): `priority:high`, this milestone, early with #115.
- **#115** in the milestone, **kept small** (version surfacing only).
- **Story 9 = #154** and **Story 10 = #155** filed.

### Pending from the owner

- **Owner-attested cutoff date** (before which FP training data was human-reviewed during
  fine-tuning). Likely ~late Jan 2026 (registry FP v2.5.0 2026-01-25 / v2.6.0 2026-01-26). Used as
  the #154 export **parameter** and the per-month hypothesis #155 tests. **Still pending.**

### NEW bug to file — FP threshold single source of truth

**Working title:** `FP threshold has no single source of truth (config 0.475 vs registry 0.5429 vs env-skip 0.53)`
**Labels:** `bug`, `mlops`, `priority:high` · **Milestone:** `eval-calibration` · **Sequence:** early, alongside #115.

`models/fp_classifier_config.json` threshold `0.475` (no `version` key); `models/registry.json`
production `v2.5.0` threshold `0.5429`; LLM-skip uses env `fp_skip_llm_threshold = 0.53`. **Goal:**
registry `production` pointer is the single source for version + threshold. **Explicit decision to
document:** does the LLM-skip threshold *derive from* the registry production threshold or remain an
**independent operating-point** setting? Cross-reference #115 (version only).

### The problem

The owner gets daily drift emails recommending FP-classifier retraining, but `/review-labels`
rarely finds a real problem. Three independent defects:

1. **Retrain is recommended on distribution shift alone.** `drift_detected = core_drifted > 0`
   (`monitoring.py:660`) so `DRIFT_THRESHOLD` never gates the core columns; `drift_monitoring.py:318`
   recommends retraining on drift alone. Per the architect (C5) the fix belongs in the workflow
   **decision layer**, not `monitoring.py:660` (a correct distribution signal).
2. **No durable record of ground truth.** `fix_label.py update` mutates status in place, prints old
   value to stdout only, records no reason/reviewer/session, overwrites timestamps, changes only
   status. Review coverage is unrecorded; review queries target suspicious articles, so the
   correction rate is not the population error rate.
3. **The FP classifier trains — and is evaluated — on its own predictions
   (coordinator-verified).** `scripts/export_training_data.py:101` exports **all**
   `labeling_status='false_positive'` articles as FP negatives. Of **941** false_positive articles,
   **390** carry a `classifier_predictions` row with `classifier_type='fp'` AND
   `action_taken='skipped_llm'` — decided by the classifier alone, never seen by the LLM or a human.
   Training/evaluating on these bakes prior-model errors in as ground truth, invisibly. By skip
   month: 2025-12:6, 2026-01:91, 2026-02:51, 03:44, 04:65, 05:45, 06:21, 07:29, 08:23, 09:15.
   Positives (`labeled`+`skipped`): **4122**. #154 (Story 9) and #155 (Story 10) fix this.

   **Owner context:** the owner reviewed FP output during training/fine-tuning and corrected errors,
   so **early data should be clean; the last 2–3 months are unvalidated** (validated window likely
   ends ~late January, ≈293 unvalidated). **Those confirmations were never recorded**, so the
   pre-cutoff cleanliness is an **owner attestation** (coarse, weaker than per-article verification),
   handled as an **export parameter** — *not* stamped on articles as provenance — and treated as a
   **hypothesis #155 tests per month** (it samples pre-cutoff articles too).

### Goal

Turn human review into durable ground truth; re-wire the monitoring **decision layer** so drift is
informational and only measured performance degradation recommends retraining; **and break the
self-training feedback loop** so the FP classifier trains and is evaluated on independent ground
truth, not its own past predictions.

### Two distinct eval relations (architect's ground-truth mapping)

- **(a) FP classifier vs. LLM label** — from `articles.labeling_status` (**C1**, verbatim in Story 4
  ACs): positive = `{labeled, skipped}`; negative = `{false_positive}`; exclude =
  `{unlabelable, deduplicated, pending, chunked, embedded}`. **Precision of "continue" measurable
  today; recall not** (**C2**): `false_positive` comes from the FP skip (`action_taken='skipped_llm'`)
  and from the LLM's `is_sportswear_brand=False`; only the second is independent. The realized-
  performance query joins where `action_taken='continued_to_llm'`; recall needs #142's exploration
  sample. **The same `skipped_llm` rows are the training-data contamination in defect 3 — #154/#155
  clean them.** The predicate that separates classifier-only from independent labels is defined once
  in `src/data_collection/database.py` and **shared between #154's export filter and Story 4's C2
  filter.**
- **(b) LLM label vs. human** — what `/review-labels` audits. Gold set (Story 6) + unbiased error
  rate (Story 2) live here.

### Relationship to the existing backlog

| Issue | Relationship |
|---|---|
| **#1** golden set | **Superseded by Story 6 → close.** |
| **#3** labeling distribution monitoring | **Keep open, deferred.** |
| **#5** LLM-as-judge | **Keep open, deferred.** |
| **#6** formalize human review | **Superseded by Stories 1+2 → close.** |
| **#38** Sonnet-4.6 contrastive gate (closed) | **Precedent** for Story 6. |
| **#54** model-migration skill | **Consumer** of Story 6's gate. |
| **#115** FP `model_version='unknown'` | **Prereq of Story 4 L2**, early; **kept small**. |
| **NEW** FP threshold SSOT bug | **priority:high**, this milestone, early with #115. |
| **#136** which drift code path produced a verdict | **Owns drifted-column plumbing**; Story 4 consumes. |
| **#141** | **Related to #154** (per coordinator) — confirm scope; not described here. |
| **#142** exploration sample (= Story 4·L1b) | **Feeds #155's recurring FP-review inflow** (clean-negative replenishment). |
| **#143** correction/label audit log (= Story 1) | Human-source signal for #154's derived provenance; #155 corrections write it. |
| **#146** review session record (= Story 2) | #155 records sessions here; **`sample_frame` must add `fp_skipped`**; its `review_session_sample.was_confirmed` is the human carve-back signal #154 reads. |
| **#147** | **Prereq (with #146) of the future ESG multi-label epic** — out of scope here. |
| **#154** Story 9 — label provenance + stop exporting classifier-only labels | This epic; early parallel cluster. |
| **#155** Story 10 — one-time review + `review-fps` | This epic; after #143/#146/#154. |
| **#72** epic scheduled-work-reports-success | **Sibling**; Story 4's verdict plugs into its contract. |

### Keep it lean

No new mega-package; no parallel store. **New tables in this epic:** `label_corrections` (Story 1/
#143), `review_sessions` + `review_session_sample` (Story 2/#146), `realized_baselines` (Story 4,
D-4.1). **#154 adds NO column and NO table** — provenance is **derived at query time** to one of
`{llm, pre_classifier, fp_classifier, human}` from `classifier_predictions.action_taken` +
`label_corrections` + `review_session_sample`, via a predicate defined once in
`src/data_collection/database.py` (shared with Story 4's C2 filter). **#155 adds NO table** — it calls
`ArticleLabeler.label_article()` directly (no pipeline/DB mutation), writes LLM opinions to a **one-off
output file**, and reuses `review_sessions`/`review_session_sample`/`label_corrections`, extending only
#146's `sample_frame` vocabulary with `fp_skipped`.

### Success criteria

- [ ] Daily email no longer recommends retraining on distribution shift alone; retrain only when the
      realized-performance verdict is `DEGRADED` vs a frozen baseline window (Story 4).
- [ ] Informational (drift-only) states do not email; email fires for performance `DEGRADED` and
      check failure (`UNKNOWN`) (Story 4).
- [ ] Realized recall reported only against the exploration sample; below the floor → `SKIPPED` (Story 4).
- [ ] Every correction captured with field, old/new, reason + reason_code, reviewer, source, session
      (Story 1).
- [ ] Each review session records coverage + per-article outcomes over `{labeled, skipped,
      false_positive}` → unbiased LLM error rate per prompt version (Story 2).
- [ ] A human-reviewed gold set gates prompt/model promotion locally in `model_training` (Story 6).
- [ ] `docs/` runbook (linked from the alert email) with a replay path (Story 8).
- [ ] **The FP training/eval export (only the `fp` dataset) excludes classifier-only negatives by
      default (with `--include-classifier-labels`), carves human-confirmed skips back via
      `review_session_sample.was_confirmed`, and reports the counts it excluded per month; provenance
      derives to `{llm, pre_classifier, fp_classifier, human}` at query time — no `label_source`
      column, no new table (#154).**
- [ ] **All ~390 classifier-only negatives get an LLM second opinion (`ArticleLabeler.label_article()`,
      empty `brand_analyses` = positive); the human reviews all disagreements + ~60 stratified
      agreements + a pre-cutoff census (≤~100); per-month error rates are produced; corrections go
      through the audit log and review-session tables via a thin `review-fps` skill (#155).**

### Story / issue checklist (build order)

- [ ] Story 4 · L1a — email-noise fix (decision layer; ships first)
- [ ] #115 (version) · NEW FP-threshold bug · **Story 9 (#154 — label provenance + export fix)** · Story 4·L1b (#142) — early, parallel
- [ ] Story 1 (#143) — correction audit log
- [ ] Story 4 · L1c — realized-performance query + verdict
- [ ] Story 4 · L2 — version-attributed metrics + human-correction override
- [ ] Story 2 (#146) — review session record + unbiased sample (+ `fp_skipped` frame)
- [ ] **Story 10 (#155) — one-time review of classifier-only negatives + `review-fps` skill**
- [ ] Story 6 — gold eval gate
- [ ] Story 3 — backfill from transcripts (hard timebox)
- [ ] Story 8 — runbook consolidation + validation
- [ ] Story 5 — drift-alarm calibration backtest (time-gated)
- [ ] Story 7 — MLflow logging (optional, last)

### Build order (final)

`4·L1a → {#115, FP-threshold bug, #154, 4·L1b(#142)} → 1(#143) → 4·L1c → 4·L2 → 2(#146) → #155 → 6 → 3 → 8 → 5 → 7`

- **#154 (Story 9)** joins the early parallel cluster (bug, `priority:high`, no dependencies) — fix
  the export before any FP retraining.
- **#155 (Story 10)** lands after **#143**, **#146** (incl. the `fp_skipped` frame) and **#154**; it
  precedes Story 6 so reviewed disagreements can seed the gold set.

---

## #115 — Surface real `model_version` (ship 2nd, small)

Version surfacing only. Registry `production` pointer authoritative for version: loader (`base.py:57`)
pulls version; add `version` to `get_model_info()` (`base.py:240-264`) and `ModelInfoResponse`
(`predict.py:253-262`); `pipeline.py:399` records it instead of the `"unknown"` fallback
(`classifier_client.py:148`). **Threshold reconciliation is the separate bug.** Backfill impossible
(document; version-attributed metrics forward-only). Tests: `/model/info` non-`unknown`; saved
prediction carries version; DB-gated. **Size:** S–M.

---

## Story 1 — Correction audit log (#143)

**Labels:** `enhancement`, `mlops`, `priority:high`

### User story
As the owner doing label review, I want every correction recorded in an append-only audit table —
field, old/new value, why, reviewer, session — so human review is durable, queryable ground truth.

### Approach
- `migrations/008_label_corrections.sql`; table per AC **plus nullable `source_ref`**. Indexes on
  `article_id`, `created_at`, `review_session_id`, `reason_code`. Model after `ClassifierPrediction`
  (~`models.py:326`); `VALID_REASON_CODES` beside `VALID_LABELING_STATUSES` (`models.py:32`).
  `review_session_id` nullable FK.
- `fix_label.py`: `update_articles` (`L118-159`) already single-session/single-commit — add
  `session.add(LabelCorrection(...))` before commit. Add `--reason` (required), `--reason-code`,
  `--reviewer` (default git `user.email`/env), `--review-session-id`. Brand subcommands
  (`models.py:159-169`).
- `.claude/skills/review-labels/SKILL.md` Step 5 (`L312-325`): route all corrections through
  `fix_label.py`, forbid raw `UPDATE`. Update CLAUDE.md.

### Acceptance criteria
- Given `label_corrections` (`id`, `article_id`, `brand` nullable, `field`, `kind`
  (`field_update`|`row_created`|`row_deleted`), `old_value`, `new_value`, `reason` required,
  `reason_code` required enum incl. `other`, `reviewer`, `review_session_id` nullable FK, `source`
  (`human`|`reconstructed`), `source_ref` nullable, `created_at`), when a correction is applied, then
  one audit row per changed field is inserted in the **same transaction**; on rollback none persists.
- Given `--status skipped`, prior status + prior timestamp recorded and `labeled_at` **not clobbered**
  (`skipped_at` set only if null) (D-1.2).
- Brand/category subcommands write change + audit row together; raw `UPDATE` no longer required.
- `/review-labels` Step 5 routes through `fix_label.py`, forbids raw `UPDATE`.
- Correction missing `reason` or `reason_code` is rejected.

### Design decisions
- **D-1.1** required enum + free text. **D-1.2** record prior timestamp, stop clobbering `labeled_at`.
  **D-1.3** one row per field.

### Dependencies
None hard. FK activated by Story 2. Foundation for Stories 2, 3, 4·L2, 6, #154, #155.

### Definition of done
- [ ] Migration + model + `VALID_REASON_CODES`.
- [ ] `fix_label.py` transactional audit rows; new flags; brand subcommands.
- [ ] SKILL.md Step 5 + CLAUDE.md updated.
- [ ] DB-gated tests: audit on status change; rollback writes nothing; brand sentiment; reason required.
- [ ] Runbook "how to correct a label" drafted.
- [ ] Branch `feature/<N>-correction-audit-log`; PR squash-merged; CI green. **Size:** M.

### Risks
- `is_sportswear_brand` not a stored column (`brand_labels` holds only sportswear brands,
  `pipeline.py:1030`) — hence the `kind` field.
- **Orphaned `brand_labels` on `labeled→skipped/false_positive`: DOCUMENT, do NOT delete** (owner) —
  **data hygiene, not a scorecard bug** (export filters `labeling_status=='labeled'`,
  `export_website_feed.py:587`). Note in runbook. **`fix_label` writes nothing on a confirmed-correct
  skip — that confirmation lives in Story 2's `review_session_sample.was_confirmed`, which #154 reads.**

---

## Story 2 — Review session record + unbiased random sample (#146)

**Labels:** `enhancement`, `evaluation`, `priority:medium`

### User story
As the owner, I want each `/review-labels` session to record coverage plus a small uniform random
sample, so I can compute an unbiased LLM error rate per prompt version.

### Approach
`migrations/009_review_sessions.sql`: `review_sessions` (`id`, `started_at`, `completed_at`,
`prompt_version`, `reviewer`, `trigger`, `notes`, `sample_frame`, `sample_size`, `articles_reviewed`,
`articles_confirmed`); activate the Story 1 FK; add `review_session_sample(session_id, article_id,
was_confirmed, correction_id nullable)`. No scheduler.

### Acceptance criteria
- Session row on start; corrections reference it.
- Uniform random sample over the frame **`{labeled, skipped, false_positive}`** under the current
  prompt version (frame + size recorded).
- Completed session yields per prompt version: reviewed, confirmed-correct, corrections by
  field/reason, unbiased error-rate estimate + n.
- Error rate defined by the query over the three tables; frame frozen per session; no hardcoded target.

### Design decisions
- **D-2.1** DB table. **D-2.2** fixed small n (~10–20/session). **D-2.3** review **against existing
  labels**; **do NOT merge** with Story 4's exploration sample. **`sample_frame` is an extensible
  vocabulary — #155 adds `fp_skipped`.** `review_session_sample.was_confirmed` is the human carve-back
  signal #154 reads (a confirmed-correct skip writes no correction row).

### Dependencies
Story 1. Feeds Stories 5, 6, #154 (carve-back signal), #155.

### Definition of done
- [ ] Migration `009`; FK; `review_session_sample` (incl. `was_confirmed`).
- [ ] Skill opens/records/closes a session and surfaces the sample.
- [ ] Documented per-prompt-version error-rate query.
- [ ] DB-gated tests: lifecycle; seeded uniformity; error-rate on a fixture.
- [ ] Runbook "how to run a review session" drafted.
- [ ] Branch `feature/<N>-review-session-record`; PR squash-merged; CI green. **Size:** M.

---

## Story 3 — Backfill corrections from transcripts (`source=reconstructed`)

**Labels:** `enhancement`, `mlops`, `priority:low`

One-off `scripts/backfill_corrections.py`. Parse the `update_articles` stdout pattern
(`fix_label.py:153`) + raw `UPDATE` SQL; write `source='reconstructed'`, `source_ref`, inferred
`reason_code` else `reconstructed_unknown`. Dedup on `(article_id, field, new_value)` earliest
timestamp; cross-check vs current DB, **flag disagreements**. **UUID caveat:** stdout line has no
UUID, only a 60-char title prefix — key on a nearby `show <id>` when present, flag ambiguous matches.
Consumers filter `source='reconstructed'`. **Hard timebox or defer.** DoD: reconstructed rows +
`source_ref`; disagreements flagged; dedup vs the ~34 upper bound; tests (parser/dedup fixtures +
DB-gated load); runbook provenance note. Branch `feature/<N>-backfill-corrections`. **Size:** S–M.

---

## Story 4 — Realized-performance monitor + tiered alerts

**Labels:** `enhancement`, `mlops`, `priority:high`

### User story
Report the FP classifier's realized performance vs the LLM labels (human corrections overriding) and
recommend retraining only when it degrades — drift alone stays informational.

### Approach — Layer 1 in three increments
1. **L1a — email fix (first).** Change the **decision layer**: `evaluate_drift_results`
   (`drift_monitoring.py:283-330`) → drift-only `DEGRADED` becomes informational; `send_drift_alerts`
   (`L345-426`) → stop emailing on the drift verdict (report-only). Leave `monitoring.py:660` (C5).
2. **L1b — exploration routing (#142).** Override at `pipeline.py:404` in `_run_fp_prefilter_batch`:
   force a small deterministic random share of would-be-skips to `should_continue=True`, tag
   `skip_reason='exploration_sample'`, deterministic by `article_id` hash, rate `FP_EXPLORATION_RATE`.
   No schema change (C3).
3. **L1c — performance query + verdict.** `check_fp_realized_performance` → `fp_performance_verdict`
   from a named query over `classifier_predictions ⋈ articles`.

### Layer 2 (after #115 + Story 1)
Version-attributed metrics (needs #115); human correction (Story 1) overrides the LLM label.

### Acceptance criteria
- **C1 mapping, verbatim:** positive `{labeled, skipped}`; negative `{false_positive}`; exclude
  `{unlabelable, deduplicated, pending, chunked, embedded}`.
- **C2 filter, verbatim:** join `classifier_predictions` where `action_taken='continued_to_llm'` (the
  shared predicate in `database.py`, also used by #154).
- Recall denominator = explored positives (`action_taken='continued_to_llm' AND probability <
  threshold_used`, `models.py:297`); otherwise not a bare number.
- **C4:** Wilson CI vs baseline gated on a minimum count; **below floor → `HealthVerdict.SKIPPED` with
  reason, never `UNKNOWN`** (`health.py:80-86`).
- Compose: perf `HEALTHY`+drift `DEGRADED` → informational (no email); perf `SKIPPED`+drift `DEGRADED`
  → "shifted; not yet measurable" (no email); perf `DEGRADED` → retrain, cite drop/n/CI (email).
- Drifted column(s) recorded — **#136 owns plumbing**, Story 4 consumes (optional field if #136 not
  yet landed).
- HealthVerdict vocabulary; added to `fail_on_unresolved_verdicts` (`drift_monitoring.py:550-553`);
  query keyed on `classifier_type`.
- (L2) version attribution once #115 lands; else flagged version-blind.

### Design decisions
- **D-4.1 (crux):** baseline = a reference **window** in a small DB table `realized_baselines` (owner
  override; not `registry.json`) — promotion writes `{model_version, prompt_version, window_start,
  window_end, n}`, recomputed by the same query as the live window.
- **D-4.2** informational drift doesn't email; email for perf `DEGRADED` and check failure. Runbook
  covers burn-in.
- **D-4.3** L1/L2 + three L1 increments. **D-4.4** #136 owns drifted-column surfacing. **D-4.5**
  exploration placement at `pipeline.py:404`; fixed small ε; **do not reuse Story 2's sample**.

### Dependencies
L1a/L1b: none. L1c: accrued exploration data. L2: #115 + Story 1. Consumes #136.

### Definition of done
- [ ] L1a decision layer rewired; L1b exploration routing; L1c step + verdict + `realized_baselines`.
- [ ] Drifted-column consumed (or optional pending #136); L2 attribution + override once #115/Story 1.
- [ ] Tests: C1/C2 fixture join; drift-only → informational; degradation → retrain w/ drop; SKIPPED
      below floor.
- [ ] CLAUDE.md + runbook drafted. Branches per increment. **Size:** L, multiple PRs.

### Risk
Between L1a tiering and accrued exploration data, real regression is effectively unmonitored — accept,
document the burn-in.

---

## Story 5 — Drift-alarm calibration backtest

**Labels:** `research`, `mlops`, `priority:low`

Notebook + committed decision record. Join archived drift history to Story 4 realized-performance
history; per candidate rule (PSI/Wasserstein, multiple-testing correction, N consecutive windows)
compute its own precision/recall vs actual drops. **D-5.1** gate on a **count** (≥100 windows at n ≥
`DRIFT_MIN_SAMPLE_SIZE`). **D-5.2** decision in `docs/`/CHANGELOG. **Clock starts at Story 4·L1b/#136
recording.** Time-gated; don't schedule now. **Size:** M–L.

---

## Story 6 — Human-reviewed gold eval set as promotion + prompt-regression gate

**Labels:** `enhancement`, `evaluation`, `priority:medium`

`data/evaluation/golden_set.jsonl` + thin `scripts/eval_gold_set.py`. Content snapshot + expected
per-brand category/sentiment/FP decision + `failure_class`, seeded from confirmed corrections (Stories
1/3, **and #155's reviewed disagreements**). Runs `ArticleLabeler(prompt_version=…)`, reports
per-class agreement + regression count with `--compare-to`. Decisions only. **D-6.1** ~30–50 records;
grow via a `gold_candidate` flag at Story 2/#155 review. **D-6.2** spike closed #38. **D-6.3** local
pre-promotion gate inside `model_training`, not CI. Feeds `model_training`. **Size:** M.

---

## Story 7 — MLflow logging of realized metrics per model version (optional, last)

**Labels:** `enhancement`, `mlops`, `priority:low`

Behind `MLFLOW_ENABLED`, via `src/mlops/tracking.py`, keyed by `model_version` (#115). Key test: no-op
+ no exception when off. Dependencies: Story 4 + #115. **Size:** S.

---

## Story 8 — Runbook: "alert fired → what now"

**Labels:** `documentation`, `mlops`, `priority:medium`

`docs/` runbook linked from alert emails (`src/agent/notifications.py`). Sections written with each
story; slot 8 = consolidation + validation via a **replay path** (`monitor_drift.py` on a crafted
window). Burn-in note. Dependencies: incremental across 1/2/3/4. **Size:** S.

---

## Story 9 — Label provenance + stop exporting classifier-only labels (#154)

**Labels:** `bug`, `mlops`, `priority:high` · **Blocked by:** none · **Relates to:** #143, #141, #142

### User story
As the owner, I want the FP training/eval export to stop treating labels the classifier produced
itself as ground truth, so retraining and evaluation rest on independent judgement instead of the
model's own past predictions.

### Context
`export_training_data.py:101` exports **all** `false_positive` articles as FP negatives; **390 of
941** are classifier-only (`classifier_type='fp' AND action_taken='skipped_llm'`, no LLM/human).
Positives (`labeled`+`skipped`) = **4122**; excluding the 390 leaves ~551 negatives (class balance
shifts — see below). "Provenance" here is a **derived query-time property**, not a stored field: an
article's label source is inferred to one of **`{llm, pre_classifier, fp_classifier, human}`** (per the
filed #154; `pre_classifier` = a `false_positive` with no `fp` classifier-prediction row, i.e. labeled
before the classifier was in the pipeline) from `classifier_predictions.action_taken` +
`label_corrections` + `review_session_sample`. The **owner-attested cutoff is not provenance** — it is
an **export parameter** (a date, still pending from the owner) and a **per-month hypothesis #155 tests**.

### Acceptance criteria (Given/When/Then)
- **Given** the shared predicate in `src/data_collection/database.py` (the same one Story 4's C2 filter
  uses), **when** it is applied, **then** each `false_positive` resolves to one of
  `{llm, pre_classifier, fp_classifier, human}`, and a **classifier-only** (`fp_classifier`) label is a
  `false_positive` whose only decision is a `classifier_type='fp' AND action_taken='skipped_llm'` row
  with no LLM label and no human signal — with **NO `label_source` column and NO new table** (derived
  at query time).
- **Given** the FP training export, **when** the `fp` dataset is exported, **then** classifier-only
  negatives are **excluded by default**, an **`--include-classifier-labels`** escape hatch restores the
  old behavior, and the export **reports the excluded count per skip-month**.
- **Given** a human confirmed a classifier skip was correct (which writes **no** `label_corrections`
  row), **when** the export decides carve-backs, **then** it reads **`review_session_sample.was_confirmed`**
  to keep those human-verified negatives — `label_corrections` alone would miss them.
- **Given** the owner-attested cutoff **parameter**, **when** an export tier includes pre-cutoff data,
  **then** that tier is a distinct, documented **weaker** tier (an attestation), not a per-article
  provenance stamp; the cutoff date is a parameter, pending owner confirmation.
- **Given** the eval/holdout split, **when** it is built, **then** it applies the **same exclusion** as
  training.
- **Given** only the FP task is contaminated, **when** other datasets are exported, **then** the
  **`esg-prefilter` (EP) and `esg-labels` exports are unchanged** (verified clean).
- **Given** the exclusion changes class balance, **when** it lands, **then** it triggers a **re-run of
  fp2 model-selection/tuning** and an **as-of contamination caveat is attached to v2.5.0** (registry /
  model card).

### Dependencies
None hard (can start immediately). Reads #143's audit log and #146's `review_session_sample` for human
signals (degrades gracefully before they land). Identifies the cohort **#155** reviews; the excluded
negatives are replenished over time by **#142**'s exploration sample once LLM-labeled.

### Definition of done
- [ ] Shared classifier-only predicate in `src/data_collection/database.py` (used by the export and
      Story 4's C2 filter); no `label_source` column, no new table; derives
      `{llm, pre_classifier, fp_classifier, human}`.
- [ ] `export_training_data.py` `fp` dataset excludes classifier-only negatives by default;
      `--include-classifier-labels` escape hatch; carve-back reads `review_session_sample.was_confirmed`;
      per-month excluded counts; eval/holdout applies the same exclusion.
- [ ] Owner-attested-cutoff export parameter implemented + documented as weaker-than-per-article (date
      pending owner).
- [ ] EP / esg-labels exports confirmed unaffected (test or explicit note).
- [ ] fp2 tuning re-run triggered post-exclusion; v2.5.0 tagged with an as-of contamination caveat.
- [ ] Tests: fixture classifier-only negative → excluded + counted; human-confirmed skip (via
      `was_confirmed`) → retained; escape hatch restores old behavior; EP/esg-labels unchanged.
- [ ] CLAUDE.md export section + runbook note (class-balance impact called out).
- [ ] Branch `fix/<N>-label-provenance`; PR squash-merged; CI green. **Size:** M.

### Resolved design (architect addendum)
- **Provenance derived, not stored** — `{llm, pre_classifier, fp_classifier, human}`; single predicate
  in `src/data_collection/database.py`, shared with Story 4's C2 filter (was D-9.1).
- **Owner-attested cutoff is an export parameter + per-month hypothesis (#155), not provenance** (was
  D-9.2).
- **Exclude by default + `--include-classifier-labels`; human carve-back reads
  `review_session_sample.was_confirmed`** because a confirmed-correct skip writes no correction row
  (was D-9.3).
- **Only the FP export is affected**; EP and esg-labels are clean.
- **Re-run fp2 after exclusion; attach an as-of contamination caveat to v2.5.0** (class-balance
  consequence of D-9.3).
- **#142's exploration negatives are the clean replenishment** once LLM-labeled — handled by the
  derived predicate, no special case (was D-9.4).

---

## Story 10 — One-time review of classifier-only negatives + `review-fps` skill (#155)

**Labels:** `enhancement`, `evaluation`, `priority:high` · **Blocked by:** #143, #146 (+ `fp_skipped`
frame), #154

### User story
As the owner, I want a one-time LLM-plus-human review of the ~390 classifier-only FP negatives — and a
repeatable `review-fps` entry point for the FP-skip inflow that follows — so the contaminated
negatives are re-grounded in independent judgement and future FP-skips keep getting audited.

### Context
#154's derived predicate identifies the ~390. An LLM **census** (all ~390) is cheap (~390 Haiku calls,
a few dollars). The human then reviews the disagreements plus a stratified sample of agreements,
prioritizing the decision boundary, and a pre-cutoff sample so "early data is clean" is **tested per
month, not assumed**. Corrections go through **#143**; sessions via **#146** with a new **`fp_skipped`**
sample frame. Afterwards, ongoing FP-skip inflow comes from **#142**'s exploration sample, making FP
review a **recurring session type**. A thin **`review-fps`** skill is a separate entry point but
**writes the same tables** as `review-labels`.

### Acceptance criteria (Given/When/Then)
- **Given** the ~390 classifier-only negatives, **when** the one-time pass runs, **then** it calls
  **`ArticleLabeler.label_article()` directly (no pipeline, no DB mutation)** for a **census of all
  ~390**, and maps **empty `brand_analyses` → POSITIVE** for the FP task (`pipeline.py:924-930`).
- **Given** the LLM opinions, **when** they are stored, **then** they go to a **one-off output file**,
  **not a table**.
- **Given** LLM vs classifier verdicts, **when** the human queue is built, **then** it includes **all
  disagreements**, plus **~60 agreements stratified by probability band**, plus a **census of the
  pre-cutoff cohort if it is ≤~100** (otherwise a sample), prioritizing rows near the **0.53**
  threshold; and it reports **per-month error rates**.
- **Given** a human correction, **when** applied, **then** it goes through **#143**'s audit log
  (`source='human'`, no raw `UPDATE`).
- **Given** a review session, **when** opened, **then** it is recorded via **#146** with
  `sample_frame='fp_skipped'` — a value #146's vocabulary must permit.
- **Given** the `review-fps` skill, **when** it runs, **then** it writes the same `review_sessions` /
  `review_session_sample` / `label_corrections` tables as `review-labels` (no parallel store), and is
  reusable for the recurring #142 inflow.

### Dependencies
Blocked by #143, #146 (+ `fp_skipped` frame) and #154. Feeds Story 6 (disagreements → gold candidates)
and #154 (reviewed labels flip the derived provenance; the per-month hypothesis result can refine the
cutoff parameter). Ongoing inflow from #142.

### Definition of done
- [ ] One-time LLM census over the ~390 via `ArticleLabeler.label_article()`; empty `brand_analyses`
      = positive; opinions written to a one-off output file.
- [ ] Human queue: all disagreements + ~60 stratified agreements + pre-cutoff census (≤~100), near-0.53
      prioritized; per-month error rates reported; corrections via #143; session via #146 with
      `sample_frame='fp_skipped'`.
- [ ] #146's `sample_frame` vocabulary extended to include `fp_skipped`.
- [ ] `review-fps` thin skill writes the shared tables (no parallel store); recurring via #142.
- [ ] Tests: queue ordering (disagreements first; near-threshold priority); empty-`brand_analyses` →
      positive mapping; session + corrections write the shared tables.
- [ ] Runbook "reviewing FP-skips" section.
- [ ] Branch `feature/<N>-review-fps`; PR squash-merged; CI green. **Size:** M.

### Resolved design (architect addendum)
- **LLM second opinion = `ArticleLabeler.label_article()` directly, no pipeline/DB mutation**; empty
  `brand_analyses` = POSITIVE (`pipeline.py:924-930`) (was D-10.1).
- **LLM opinions → a one-off output file, not a table** (was D-10.2).
- **Human coverage:** all disagreements + ~60 agreements stratified by probability band + census of the
  pre-cutoff cohort if ≤~100; per-month error rates (was D-10.3).
- **`review-fps` is a separate thin skill, recurring via #142; add `fp_skipped` to #146's
  `sample_frame`** (was D-10.4).

---

## Scope cuts and pushback

1. LLM-as-judge (#5) deferred; #3's subsystem not adopted; #6's YAML store + scheduler cut (#6 close);
   gold set small + seeded (#1 close); Story 5 time-gated; Story 7 optional; Story 0 dropped; #115 kept
   small (threshold is its own bug).
2. **#154 derives provenance rather than storing it** — no new column/table, one shared predicate; and
   it **excludes** classifier-only negatives rather than salvaging them in place (#155 re-grounds them).
3. **#155 reuses the review-session + audit tables** and calls `label_article()` directly (no DB
   mutation, opinions to a file) — the `review-fps` skill is only a separate entry point.

## Out of scope — future epics

- **ESG multi-label classifier revisit (~853 labeled articles).** A **separate future epic** whose
  prerequisites are **#146** and **#147**. The realized-performance query is kept `classifier_type`-
  generic so it can be reused there.

## Resolved planning decisions (2026-09-24)

- **Q1** `Epic:` tracker (#152) + `eval-calibration` milestone. **Q2** close #1/#6, keep #3/#5. **Q3**
  FP skip active (`fp_skip_llm_threshold=0.53`) → Story 4 exploration sample. **Q4** #115 early, small.
  **Q5** runbook in `docs/`.

## Resolved design decisions from the architect review (owner exceptions applied)

Stories 1–8: baseline in a small **DB table** `realized_baselines` (not `registry.json`); orphaned
`brand_labels` **documented not deleted** (data hygiene; export filters `labeling_status=='labeled'`);
threshold mismatch its own `priority:high` bug (#115 kept small); C1 mapping + C2 filter verbatim in
Story 4 ACs; exploration routing at `pipeline.py:404`, separate from Story 2's sample; SKIPPED not
UNKNOWN below floor; Story 4 L1 as three increments (email fix first); Story 2 frame `{labeled,
skipped, false_positive}`; Story 3 timebox + UUID caveat; Story 5 clock at L1b/#136; #136 owns
drifted-column plumbing.

Stories 9–10 (addendum): provenance **derived** to `{llm, pre_classifier, fp_classifier, human}` (no
column/table), single predicate in `database.py` shared with Story 4 C2; owner-attested cutoff an
**export parameter + per-month hypothesis**, not provenance; exclude classifier-only labels **by
default** with `--include-classifier-labels`, carve-back via `review_session_sample.was_confirmed`;
**FP export only** (EP/esg-labels clean); re-run fp2 + as-of v2.5.0 contamination caveat; #155 uses
`ArticleLabeler.label_article()` directly (empty `brand_analyses` = positive), LLM census of ~390,
human reviews all disagreements + ~60 stratified agreements + pre-cutoff census (≤~100), per-month
error rates, opinions to a one-off file; `review-fps` a separate thin skill recurring via #142; add
`fp_skipped` to #146's `sample_frame`.

## Forward-compatibility

- #136 owns `print_summary_json`/`REQUIRED_SUMMARY_FIELDS`; Story 4 consumes.
- Story 4's verdict plugs into #72's `silent-success` contract; SKIPPED-vs-UNKNOWN avoids vacuous
  health.
- **The classifier-only predicate is defined once in `src/data_collection/database.py` and shared by
  #154's export filter and Story 4's C2 realized-performance filter — one definition, two consumers.**
- Realized-performance query kept `classifier_type`-generic for the future ESG multi-label epic.
- **#154 (derived provenance, honest export) + #155 (FP-skip review) make the training/eval data
  self-cleaning as #142's exploration sample flows in** — the feedback loop that produced the 390
  contaminated negatives is closed going forward, not patched once.
- A scalar-hardcoded D-4.1 would rot like the thresholds this epic replaces — hence the same-query
  reference window.
