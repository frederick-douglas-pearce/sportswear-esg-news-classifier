# Epic draft — Calibrate the eval/monitoring framework against human review

> FINALIZED + architect-reviewed. No GitHub issues created. Repo:
> `frederick-douglas-pearce/sportswear-esg-news-classifier`. Author: fpearce.
> Date: 2026-09-24.
>
> **Single source of truth:** this spec. The architect review at
> [`prd-eval-calibration-architect-review.md`](prd-eval-calibration-architect-review.md)
> is the rationale reference (file:line citations against `999c25d`); its findings are
> folded into the stories below.
>
> **Change note 1:** original "Story 0 — preserve transcripts (urgent)" dropped —
> `cleanupPeriodDays: 3650`, transcripts not at risk. Optional snapshot folded into
> Story 3. Oldest surviving transcript 2026-05-07 covers prompt v1.10.0 (since 2026-07-01).
>
> **Change note 2 (owner, 2026-09-24):** Q1–Q5 resolved (see "Resolved planning decisions").
>
> **Change note 3 (architect review folded in, owner exceptions applied, 2026-09-24):**
> Adopted the review in full except three owner overrides — (1) the frozen baseline lives
> in a small **DB table**, not `registry.json`; (2) orphaned `brand_labels` are
> **documented, not deleted** (data hygiene, not a scorecard bug — export already filters
> `labeling_status=='labeled'`); (3) the config/registry/env **threshold mismatch becomes
> its own `priority:high` bug** in this milestone, and **#115 stays small** (version
> surfacing only). Details in-line and in "Resolved design decisions".

---

## Epic issue (tracker)

**Title:** `Epic: calibrate eval/monitoring against human review — stop drift-only retrain alerts`

**Labels:** `evaluation`, `mlops` (no `epic:` label — see convention note)
**Milestone (new — confirmed):** `eval-calibration`

### Convention note

This repo does **not** use an `epic:` label. Its one existing epic, #72, is an
`Epic:`-titled tracker issue grouped by a **milestone** (`silent-success`). Confirmed with
the owner: this epic follows that precedent — an `Epic:`-titled tracker + a new
`eval-calibration` milestone.

### Backlog actions decided (2026-09-24)

When this epic's stories are filed:
- **Close #1** (golden set) as superseded by Story 6, cross-referenced.
- **Close #6** (formalize human review) as superseded by Stories 1+2, cross-referenced.
- **Keep #3 open**, deferred, cross-referenced (distribution-monitor subsystem not adopted;
  useful parts fold into Stories 4/5).
- **Keep #5 open**, deferred, cross-referenced (LLM-as-judge — different, unvalidated
  ground-truth source).
- **File a NEW bug** (see below): *FP threshold has no single source of truth* —
  `priority:high`, `eval-calibration` milestone, sequenced **early** alongside #115.
- **#115** is pulled into this milestone but **kept small** (version surfacing only); the
  threshold mismatch is handled by the new bug, not by #115.

### NEW bug to file — FP threshold single source of truth

**Working title:** `FP threshold has no single source of truth (config 0.475 vs registry 0.5429 vs env-skip 0.53)`
**Labels:** `bug`, `mlops`, `priority:high` · **Milestone:** `eval-calibration` · **Sequence:** early, alongside #115.

**Facts (verified by coordinator + architect):**
- `models/fp_classifier_config.json` threshold `0.475`, **no `version` key**.
- `models/registry.json` production `v2.5.0` threshold `0.5429`.
- The LLM-skip decision uses env `fp_skip_llm_threshold = 0.53` (`labeling_settings`), not
  the model config — so `0.475` may be dead for the skip path but live for the API's
  `is_sportswear` boolean.

**Goal:** the registry `production` pointer is the single source for **version + threshold**.
**Decision to make explicitly and document:** does the LLM-skip threshold *derive from* the
registry production threshold, or remain an **independent operating-point** setting? Either is
defensible; the requirement is that the choice is explicit and recorded, not incidental.
Cross-reference #115 (which only surfaces the version). This bug does the threshold
reconciliation; #115 does not.

### The problem

The owner receives daily drift emails recommending FP-classifier retraining, but running
`/review-labels` rarely surfaces a substantive labeling problem. Two independent defects:

1. **The retrain recommendation is driven by distribution shift alone, not by evidence of
   degraded performance.** On the Evidently path (`src/mlops/monitoring.py`),
   `drift_detected = core_drifted > 0` counts how many of 3 core columns have p<0.05
   (verified `monitoring.py:660`), so `DRIFT_THRESHOLD` never gates the core columns and any
   single core column with p<0.05 fires the alert. `drift_monitoring.py:318` then emits
   `"Consider retraining affected classifiers"` on drift alone. Per the architect (C5):
   `monitoring.py:660` is a *correct distribution signal* — the defect is in the workflow
   **decision layer** (`evaluate_drift_results` `drift_monitoring.py:283-330` and
   `send_drift_alerts` `L345-426`), which is where the fix belongs. Drift-mechanism bugs stay
   with #72/#136.

2. **There is no durable record of the ground truth that would let us calibrate the alert.**
   `scripts/fix_label.py update` mutates `articles.labeling_status` in place; the old value
   only prints to stdout; no reason/reviewer/session; it overwrites `skipped_at`/`labeled_at`
   with `now`; it can only change status — sentiment/category/brand fixes are ad-hoc SQL with
   no trace. Review *coverage* is not recorded (only a date in
   `.claude/skills/review-labels/last_run.txt`), and review queries target suspicious
   articles, so the correction rate is **not** the population error rate.

### Goal

Turn human review into durable ground truth, and re-wire the monitoring **decision layer** so
that **distribution drift is informational and only measured performance degradation
recommends retraining**: (a) capture corrections + review sessions durably; (b) replace the
drift→retrain reflex with a realized-performance signal (with human corrections overriding the
LLM); (c) stand up a human-reviewed gold set that gates prompt/model promotion.

### Two distinct eval relations (do not conflate them) — with the architect's ground-truth mapping

- **(a) FP classifier vs. LLM label.** Ground-truth mapping from `articles.labeling_status`
  (architect **C1** — put verbatim in Story 4 ACs):
  - positive = `{labeled, skipped}` (`skipped` = genuine brand, no ESG content, `fix_label.py:169`)
  - negative = `{false_positive}`
  - exclude = `{unlabelable, deduplicated, pending, chunked, embedded}`

  **Precision of the "continue" decision is measurable today; recall is not** (architect **C2**).
  `false_positive` is written from two sources: the FP classifier's own skip
  (`action_taken='skipped_llm'`, `pipeline.py:839-855`) and the LLM's `is_sportswear_brand=False`
  verdict (`pipeline.py:952-965`). Only the second is independent ground truth. The realized-
  performance query joins `classifier_predictions` where **`action_taken='continued_to_llm'`**.
  Recall is unmeasurable today because FP-skipped rows were labeled by the classifier itself —
  the selection bias the **exploration sample** fixes. The FP skip is confirmed active
  (`fp_classifier_enabled=True`, `fp_skip_llm_threshold=0.53`).
- **(b) LLM label vs. human.** What `/review-labels` audits. Human corrections audit the LLM
  labeler, not the FP classifier directly. The gold set (Story 6) and the unbiased error rate
  (Story 2) live here.

### Relationship to the existing backlog

| Issue | Relationship |
|---|---|
| **#1** golden set (evaluation, P:high) | **Superseded by Story 6 → close.** |
| **#3** labeling distribution monitoring (evaluation, mlops) | **Keep open, deferred.** Useful parts fold into Stories 4/5; subsystem not adopted. |
| **#5** LLM-as-judge (evaluation) | **Keep open, deferred.** Different, unvalidated ground-truth source. |
| **#6** formalize human review (evaluation, P:low) | **Superseded by Stories 1+2 → close.** YAML store + `review_queue.py` scheduler cut. |
| **#38** Sonnet-4.6 contrastive gate (closed) | **Precedent** for Story 6. Spike what it left behind (D-6.2). |
| **#54** reusable model-migration skill | **Consumer** of Story 6's gate. |
| **#115** FP `model_version='unknown'` (bug, mlops, silent-success) | **Prerequisite of Story 4 Layer 2**, sequenced early. **Kept small: version surfacing only** (registry pointer → `get_model_info()`/`/model/info` → `pipeline.py:399`). Threshold mismatch handled by the new bug above. |
| **NEW** FP threshold SSOT bug | **priority:high**, this milestone, early alongside #115 (see above). |
| **#136** which drift code path produced a verdict | **Owns the drifted-column plumbing** (`core_metrics_drifted`/`brand_metrics_drifted` at `monitoring.py:602,607`, surfacing via `REQUIRED_SUMMARY_FIELDS` `drift_monitoring.py:45-52`). Story 4 **consumes** it. |
| **#72** epic scheduled-work-reports-success (silent-success) | **Sibling, distinct failure mode.** Story 4's performance verdict plugs into #72's contract; SKIPPED-vs-UNKNOWN keeps it from reintroducing vacuous health. |

### Keep it lean

CLAUDE.md warns against machinery (`src/experiment_log/`: 19 models, a store/tracker/CLI, and
`data/experiments/` never held an entry). Every story is scoped to one table or one thin
script, reusing existing infrastructure. No new `src/evaluation/` mega-package; no parallel
YAML store. New tables in this epic: `label_corrections` (Story 1), `review_sessions` +
`review_session_sample` (Story 2), and a small `realized_baselines` table (Story 4, D-4.1).

### Success criteria

- [ ] The daily email no longer recommends retraining on distribution shift alone; retrain is
      recommended only when the realized-performance verdict is `DEGRADED` vs a frozen baseline
      window (Story 4).
- [ ] Informational (drift-only) states do **not** email; email fires for performance
      `DEGRADED` and for check failure (`UNKNOWN`) (Story 4, D-4.2).
- [ ] Realized FP recall is reported only against an exploration sample; below the minimum
      count it is `SKIPPED` (not `UNKNOWN`), never a bare number (Story 4).
- [ ] Every correction is captured with field, old/new value, reason + reason_code, reviewer,
      source, and (when applicable) review session (Story 1).
- [ ] Each review session records coverage + per-article outcomes over the frame
      `{labeled, skipped, false_positive}`, yielding an unbiased LLM error rate per prompt
      version (Story 2).
- [ ] A human-reviewed gold set gates prompt-version and model promotion, locally inside
      `model_training` (Story 6).
- [ ] An operator receiving an alert has a `docs/` runbook (linked from the email) with a
      replay path (Story 8).

### Story / issue checklist (build order; fill #N on creation)

- [ ] Story 4 · L1a — email-noise fix (decision layer only; ships first)
- [ ] #115 — surface real `model_version` (small)  **+**  NEW bug — FP threshold SSOT (early, parallel)
- [ ] Story 4 · L1b — exploration routing at `pipeline.py:404` (starts data + Story 5 clock)
- [ ] Story 1 — correction audit log
- [ ] Story 4 · L1c — realized-performance query + verdict (once explored data exists)
- [ ] Story 4 · L2 — version-attributed metrics (needs #115) + human-correction override (needs Story 1)
- [ ] Story 2 — review session record + unbiased sample
- [ ] Story 6 — gold eval gate
- [ ] Story 3 — backfill from transcripts (hard timebox)
- [ ] Story 8 — runbook consolidation + validation
- [ ] Story 5 — drift-alarm calibration backtest (time-gated; clock started at L1b/#136)
- [ ] Story 7 — MLflow logging (optional, last)

### Build order (final)

`4·L1a → {#115, FP-threshold bug} → 4·L1b → 1 → 4·L1c → 4·L2 → 2 → 6 → 3 → 8 → 5 → 7`

- **4·L1a** ships first: it fixes the email noise with **zero data dependency**.
- **#115** and the **FP-threshold bug** ship next (early); they are independent of each other
  and of 4·L1b, so they can go in parallel.
- **4·L1b** (exploration routing) ships as early as possible in parallel — it starts data
  accrual and Story 5's recording clock; it depends on nothing.
- **4·L1c** needs accrued exploration data; **4·L2** needs #115 (version) + Story 1 (override).
- Runbook sections are written *with each story*; slot **8** is only the validated consolidation.

---

## #115 — Surface real `model_version` (ship 2nd, small)

**Scope (kept small per owner):** version surfacing only. Registry `production` pointer becomes
authoritative for version: loader (`src/deployment/base.py:57`) pulls version from
`models/registry.json`; add `version` to `get_model_info()` (`base.py:240-264`) and
`ModelInfoResponse` (`scripts/predict.py:253-262`); `pipeline.py:399` records the real version
instead of the `"unknown"` fallback (`classifier_client.py:148`).
**Threshold reconciliation is NOT in #115** — it is the separate FP-threshold SSOT bug.
**Backfill impossible:** existing rows can't be attributed to a version — document as an
irrecoverable gap; version-attributed metrics apply going forward only.
**Tests:** `/model/info` returns non-`unknown`; saved `ClassifierPrediction` carries the version
(mock client); DB-gated persistence check. **Size:** S–M, one PR.

---

## Story 1 — Correction audit log (DB table + `fix_label.py` writes it)

**Labels:** `enhancement`, `mlops`, `priority:high`

### User story
As the owner doing label review, I want every correction recorded in an append-only audit
table — field, old/new value, why, reviewer, and review session — so human review becomes
durable, queryable ground truth instead of a vanishing stdout line.

### Context
`fix_label.py update` can only change `labeling_status`, mutates in place, prints old value to
stdout only, overwrites `skipped_at`/`labeled_at` with `now`, records no reason/reviewer/session.
Sentiment/category/brand fixes are ad-hoc SQL. **Why a DB table, not #6's YAML:** corrections
must join to `articles`/`brand_labels`/`classifier_predictions` for Stories 4/6; a parallel YAML
store rots (`experiment_log` precedent). **Why app-level, not a trigger:** a trigger can't
capture reason/reviewer.

### Approach (from architect)
- `migrations/008_label_corrections.sql`; table per AC **plus a nullable `source_ref`** so Story 3
  needs no second migration. Indexes on `article_id`, `created_at`, `review_session_id`,
  `reason_code`. Model after `ClassifierPrediction` (~`models.py:326`); `VALID_REASON_CODES` beside
  `VALID_LABELING_STATUSES` (`models.py:32`). `review_session_id` nullable FK (activated by Story 2).
- `fix_label.py`: `update_articles` (`L118-159`) already uses one session + one commit — add
  `session.add(LabelCorrection(...))` in the loop before commit (transactional for free). Add
  `--reason` (required), `--reason-code` (choices), `--reviewer` (default from git `user.email`/env),
  `--review-session-id` (nullable). Add brand-level subcommands for `brand_labels` fields
  (`models.py:159-169`).
- `.claude/skills/review-labels/SKILL.md` Step 5 (`L312-325`): route all corrections through
  `fix_label.py`, forbid raw `UPDATE`. Update CLAUDE.md "Label Corrections".

### Acceptance criteria
- Given the new append-only `label_corrections` table (`id`, `article_id`, `brand` nullable,
  `field`, `kind` (`field_update` | `row_created` | `row_deleted` — see risk), `old_value`,
  `new_value`, `reason` (required free text), `reason_code` (required enum incl. `other`),
  `reviewer`, `review_session_id` nullable FK, `source` (`human`|`reconstructed`), `source_ref`
  nullable, `created_at`), when a correction is applied, then exactly one audit row per changed
  field is inserted in the **same transaction** as the mutation; on rollback, no audit row persists.
- Given a `--status skipped` transition, when applied, then the prior status **and prior
  timestamp** are recorded in `old_value`, and `labeled_at` is **no longer clobbered**
  (`skipped_at` set only if null) — the overwrite at `fix_label.py:149-151` corrupts the timeline
  review-labels keys off (D-1.2).
- Given brand/category subcommands, when the owner corrects a `brand_labels` field, then the change
  and its audit row are written together; raw `UPDATE` is no longer required.
- Given `/review-labels`, when it documents Step 5, then it routes corrections through `fix_label.py`
  and forbids raw `UPDATE`.
- A correction with no `reason` **or** no `reason_code` is rejected.

### Design decisions
- **D-1.1** — required `reason_code` enum (with `other`) **and** required free-text `reason`
  (enum for cheap aggregation; prose can't be counted).
- **D-1.2** — record prior timestamp in `old_value`; stop clobbering `labeled_at`; set `skipped_at`
  only if null.
- **D-1.3** — one audit row per changed field (directly aggregatable by `reason_code`/`field`).

### Dependencies
None hard. `review_session_id` FK activated by Story 2. Foundation for Stories 2, 3, 4·L2, 6.

### Definition of done
- [ ] Migration `008_label_corrections.sql`; model + `VALID_REASON_CODES` in `models.py`.
- [ ] `fix_label.py` writes an audit row per changed field transactionally; `--reason`
      (required), `--reason-code`, `--reviewer`, `--review-session-id`; brand/category subcommands.
- [ ] `/review-labels` SKILL.md Step 5 + CLAUDE.md updated.
- [ ] Tests (DB-gated, `tests/test_scorecard_database.py:24-29` pattern): audit row on status
      change; rollback writes nothing; brand sentiment path; reason/reason_code-required rejection.
- [ ] Runbook section "how to correct a label" drafted.
- [ ] Branch `feature/<N>-correction-audit-log`; PR squash-merged; CI green. **Size:** M.

### Risks
- `is_sportswear_brand` is **not a stored column** — `brand_labels` holds only sportswear brands
  (`pipeline.py:1030`). "Is/isn't sportswear" corrections are a status change **plus** a
  `brand_labels` row add/delete; hence the audit model needs a `kind` (`row_created`/`row_deleted`),
  not just `field_update`.
- **Orphaned `brand_labels` on `labeled→skipped/false_positive`: DOCUMENT, do NOT delete** (owner
  decision — the trail is preferred). This is **data hygiene, not a scorecard bug**: the scorecard
  export already filters `Article.labeling_status == "labeled"`
  (`scripts/export_website_feed.py:587`), so rows under a non-`labeled` article are not scored.
  (Corrects the architect's "scorecard-correctness" framing.) Note the behavior in the runbook.

---

## Story 2 — Review session record + unbiased random sample

**Labels:** `enhancement`, `evaluation`, `priority:medium`

### User story
As the owner, I want each `/review-labels` session to record what it covered plus a small uniform
random sample, so I can compute an unbiased LLM error rate per prompt version — distinct from the
correction rate on deliberately-suspicious articles.

### Approach (from architect)
`migrations/009_review_sessions.sql`: `review_sessions` (`id`, `started_at`, `completed_at`,
`prompt_version`, `reviewer`, `trigger`, `notes`, `sample_frame`, `sample_size`,
`articles_reviewed`, `articles_confirmed`); activate the Story 1 FK; add
`review_session_sample(session_id, article_id, was_confirmed, correction_id nullable)` for the
per-article outcomes (the denominator). Driven from the review-labels skill; corrections carry
`--review-session-id`. **No scheduler** (#6's `review_queue.py` is cut).

### Acceptance criteria
- Given a review session, when it starts, then a `review_sessions` row is created and corrections
  reference it via `review_session_id`.
- Given the unbiased sample, when drawn, then it is a **uniform random** draw over the frame
  **`{labeled, skipped, false_positive}`** — i.e. all articles that reached the LLM — under the
  current prompt version (frame + size recorded), so false-skip/false-FP errors are in scope, not
  just `labeled`.
- Given a completed session, when queried, then it yields per prompt version: articles reviewed,
  confirmed-correct, corrections by field/reason, and an unbiased error-rate estimate with its n.
- The error rate is defined by *this query over `review_sessions` + `review_session_sample` +
  `label_corrections`*; the frame is frozen per session — no hardcoded accuracy target in code.

### Design decisions
- **D-2.1** — DB table (confirmed).
- **D-2.2** — fixed small n (~10–20/session), accumulated; proportional sampling is unstable at small n.
- **D-2.3** — review **against existing labels**. **Do NOT merge with Story 4's exploration
  sample** — different frames, estimands and stages; drawing from would-be-skips biases toward the
  FP-negative tail. Legitimate coordination is the reverse: over-sample explored articles *for human
  review*.

### Dependencies
Story 1 (corrections FK). Feeds Stories 5 and 6.

### Definition of done
- [ ] Migration `009`; FK activated; `review_session_sample` table.
- [ ] review-labels skill (or thin CLI) opens/records/closes a session and surfaces the sample.
- [ ] Documented per-prompt-version reviewed / confirmed / unbiased-error-rate query.
- [ ] Tests (DB-gated): session lifecycle; seeded uniformity; error-rate query on a fixture.
- [ ] Runbook section "how to run a review session" drafted.
- [ ] Branch `feature/<N>-review-session-record`; PR squash-merged; CI green. **Size:** M.

---

## Story 3 — Backfill corrections from transcripts (`source=reconstructed`)

**Labels:** `enhancement`, `mlops`, `priority:low`

### User story
As the owner, I want past corrections reconstructed from transcripts and loaded as
`source=reconstructed`, so calibration has more history — without pretending reconstructed rows are
as trustworthy as fresh ones.

### Approach (from architect)
One-off `scripts/backfill_corrections.py`. Parse transcripts for the `update_articles` stdout
pattern (`fix_label.py:153`) and raw `UPDATE` SQL; write rows with `source='reconstructed'`,
`source_ref` = transcript + timestamp, inferred `reason_code` else `reconstructed_unknown`.
**Dedup** on `(article_id, field, new_value)` keeping the earliest timestamp; reconcile against the
~34 upper bound. Cross-check against current DB state and **flag disagreements rather than
overwrite**. Optional: snapshot parsed transcripts to a gitignored path (convenience, not safety —
transcripts are not decaying).

### Acceptance criteria
- Given transcripts, when parsed, then each correction becomes a `label_corrections` row with
  `source='reconstructed'` and a `source_ref`.
- Given a reconstructed correction, when loaded, then it is cross-checked against current
  `brand_labels`/`articles` and flagged on disagreement (not silently applied).
- Given overlapping transcripts, when backfill runs, then rows are de-duplicated on
  `(article_id, field, new_value)`, earliest timestamp kept.
- Raw-`UPDATE`-SQL corrections that can't be parsed automatically are listed for manual entry.

### Design decisions & risk
- **D-3.1** — automate the `fix_label.py`-output parse; hand-enter the 3 raw-SQL cases or document
  the gap.
- **UUID-matching caveat (architect):** the stdout line has **no UUID**, only a 60-char title
  prefix. Key on a UUID from a nearby `show <id>` command when present, title-prefix otherwise, and
  **flag ambiguous matches**. A mis-mapped row is worse than a missing one (these feed Stories 4/5
  ground truth). Downstream consumers must filter `source='reconstructed'`.
- **Hard timebox or defer** (owner + architect). Opportunistic; does not block Story 4 or 6.

### Definition of done
- [ ] Backfill loads reconstructed rows; all marked `source='reconstructed'` with `source_ref`.
- [ ] Disagreements reported, not applied; dedup verified against the ~34 upper bound; ambiguous
      title-only matches flagged.
- [ ] Tests: pure-function parser/dedup fixtures; DB-gated load.
- [ ] Runbook note "where transcripts live / reconstructed provenance" drafted.
- [ ] Branch `feature/<N>-backfill-corrections`; PR squash-merged; CI green. **Size:** S–M.

---

## Story 4 — Realized-performance monitor + tiered alerts (stops the false emails)

**Labels:** `enhancement`, `mlops`, `priority:high`

### User story
As the owner, I want the monitor to report the FP classifier's realized performance vs the LLM
labels (human corrections overriding where present) and recommend retraining **only** when that
degrades — drift alone stays informational — so I stop getting baseless retrain emails.

### Approach — Layer 1 in three increments (architect C6/D-4.3)
1. **L1a — email fix (first, no data dependency).** Change the **decision layer**, not
   `monitoring.py`: `evaluate_drift_results` (`drift_monitoring.py:283-330`) makes a drift-only
   `DEGRADED` an *informational* recommendation; `send_drift_alerts` (`L345-426`) **stops emailing
   on the drift verdict** (drift is report-only via `generate_drift_report` `L442-462`).
   `monitoring.py:660` is a correct distribution signal — leave it (C5).
2. **L1b — exploration routing (parallel, to accrue data + start Story 5's clock).** Override at
   `pipeline.py:404` (`should_continue = result.probability >= threshold`) in
   `_run_fp_prefilter_batch`: force a small deterministic random share of would-be-skips to
   `should_continue=True`, tagged `skip_reason='exploration_sample'`. Deterministic by hash of
   `article_id`; rate `FP_EXPLORATION_RATE` (~0.05–0.10). **No schema change** (C3). The monitor is
   read-only and cannot create the sample — routing must live in the pipeline.
3. **L1c — performance query + verdict (once data exists).** A `check_fp_realized_performance` step
   produces `fp_performance_verdict` from a named query over `classifier_predictions ⋈ articles`.

### Layer 2 (after #115 + Story 1)
Attribute metrics to the real `model_version` (needs #115); where a human correction exists
(Story 1), use it as ground truth over the LLM label.

### Acceptance criteria
- **C1 mapping, verbatim.** Ground truth from `articles.labeling_status`: positive =
  `{labeled, skipped}`; negative = `{false_positive}`; exclude =
  `{unlabelable, deduplicated, pending, chunked, embedded}`. Treating `skipped` as negative
  fabricates a large false-positive rate and is a defect.
- **C2 filter, verbatim.** The realized-performance query joins `classifier_predictions` where
  **`action_taken='continued_to_llm'`** (precision of the "continue" decision — measurable today).
  Rows with `action_taken='skipped_llm'` are the classifier's own skips and are **not** independent
  ground truth.
- Given the FP skip is active, when **recall** is computed, then its denominator is the **explored
  positives** cohort — identified by `action_taken='continued_to_llm' AND probability < threshold_used`
  (`threshold_used` is per-row, `models.py:297`); recall is otherwise unmeasurable and must not be
  reported as a bare number.
- **C4 floor + CI.** Fire a `retrain` recommendation only when a **Wilson score interval** on
  realized recall/precision is distinguishably below the baseline, gated on a **minimum count** on
  the correct denominator. **Below the floor → `HealthVerdict.SKIPPED` with a reason, never
  `UNKNOWN`** (`health.py:80-86`) — UNKNOWN fails the run at the gate and would fail every night
  during data accrual.
- Given a drift-only run (no measured performance drop), when composing, then: perf `HEALTHY` +
  drift `DEGRADED` → informational (no email); perf `SKIPPED` + drift `DEGRADED` → "shifted; not yet
  measurable" (no email); perf `DEGRADED` → retrain, citing the measured drop, n and CI (email).
- Given any drift verdict, when archived, then the **drifted column(s)** are recorded — **#136 owns
  this plumbing** (`core_metrics_drifted`/`brand_metrics_drifted` `monitoring.py:602,607`; surfacing
  via `REQUIRED_SUMMARY_FIELDS` `drift_monitoring.py:45-52`). Story 4 **consumes** it; if #136
  hasn't landed, Story 4 adds it as an optional field.
- **HealthVerdict integration:** `fp_performance_verdict` uses the shared vocabulary
  (`health.py:48-86`), is added to `fail_on_unresolved_verdicts` (`drift_monitoring.py:550-553`,
  factory `base.py:112-206`), and emits `.value` strings. Query is keyed on `classifier_type` so the
  planned ESG multi-label classifier can reuse it.
- (Layer 2) Given #115 landed, metrics are attributed to the serving `model_version`; before that,
  Layer 1 explicitly flags metrics as version-blind.

### Design decisions
- **D-4.1 (crux) — frozen baseline = a reference *window*, in a small DB table (owner override).**
  Do **not** use registry offline test metrics (e.g. `test_recall=0.9966` for v2.5.0) — that is a
  different instrument/population (held-out vs human) and would be apples-to-oranges. On promotion,
  write a row `{model_version, prompt_version, window_start, window_end, n}` to a small
  `realized_baselines` **DB table**; recompute the baseline metric from **the same query as the live
  window**. (Architect leaned toward `registry.json`; owner chose a DB table to avoid writing
  generated data into a hand-curated committed file and because a same-query window can't rot like a
  scalar.) Until #115, the baseline is version-blind and says so.
- **D-4.2** — informational drift does not email; email only for performance `DEGRADED` and for
  check failure (`UNKNOWN`). The runbook covers the burn-in: drift-only → manual `/review-labels`
  spot-check.
- **D-4.3** — L1/L2 split plus the three L1 increments above.
- **D-4.4** — **#136 owns** the drifted-column surfacing; Story 4 consumes. Resolve
  `print_summary_json`/`REQUIRED_SUMMARY_FIELDS` ownership with #136 up front.
- **D-4.5 (elevated) — exploration.** Placement at `pipeline.py:404` (C3); fixed small ε
  (`FP_EXPLORATION_RATE`) chosen so the window accrues ≥ the C4 floor of explored positives;
  trade-off is LLM cost vs CI width. **Do not reuse Story 2's sample.**

### Dependencies
L1a: none. L1b: none. L1c: accrued exploration data (from L1b). L2: **#115** + Story 1. Consumes
**#136**. Baseline table introduced here (D-4.1).

### Definition of done
- [ ] L1a: decision layer rewired; drift-only is informational and does not email.
- [ ] L1b: exploration routing at `pipeline.py:404`; `FP_EXPLORATION_RATE`; deterministic by
      `article_id` hash; `skip_reason='exploration_sample'`.
- [ ] L1c: `check_fp_realized_performance` step + `fp_performance_verdict`; C1/C2 query; Wilson CI;
      SKIPPED below floor; `realized_baselines` table + promotion writes the window row.
- [ ] Drifted-column recording consumed (or added optionally pending #136).
- [ ] L2: version attribution once #115 lands; human-correction override once Story 1 lands.
- [ ] Tests: fixture join asserting C1 mapping + C2 filter; drift-only → informational; measured
      degradation → retrain with cited drop; archive carries drifted columns; recall returns SKIPPED
      below floor.
- [ ] CLAUDE.md drift-monitoring section updated; runbook "what a drift-only vs performance alert
      means" drafted.
- [ ] Branches per increment (e.g. `feature/<N>-drift-email-tiering`, `-fp-exploration-routing`,
      `-fp-realized-performance`); PRs squash-merged; CI green. **Size:** L, multiple PRs.

### Risk
Between shipping L1a tiering and accruing exploration data, real regression is effectively
unmonitored. Accept (strictly better than false alarms) and document the burn-in in the runbook.

---

## Story 5 — Drift-alarm calibration backtest

**Labels:** `research`, `mlops`, `priority:low`

### User story
As the owner, I want a backtest quantifying what distribution shift actually predicts about
performance drops, so drift thresholds are set from evidence, not the p<0.05-on-any-core-column
reflex.

### Approach (from architect)
A notebook for exploration **plus a committed decision record**. Join archived drift history to
Story 4 realized-performance history; compute each candidate rule's own precision/recall vs actual
drops. Candidates: effect sizes (PSI/Wasserstein) over significance, multiple-testing correction,
"N consecutive windows".

### Acceptance criteria
- Given accrued drift history joined to realized-performance history, when the backtest runs, then it
  reports, per candidate rule, how often a drift alarm preceded an actual drop vs fired spuriously
  (the alarm's own precision/recall).
- Given the output, when thresholds are chosen, then the rule + its backtested false-alarm rate land
  in a **decision record in `docs/`/CHANGELOG** (not a notebook cell); the instrument and data window
  are named, not a magic threshold.
- The backtest states the n it ran on and that it is data-limited.

### Design decisions & risk
- **D-5.1** — gate on a **count** (e.g. ≥100 windows with realized recall at n ≥
  `DRIFT_MIN_SAMPLE_SIZE`), not a date.
- **D-5.2** — decision lands in `docs/`/CHANGELOG.
- **Clock:** the inputs aren't recorded historically — **the recording clock starts when Story 4·L1b
  / #136 begin recording**, not from the existing archive. Do not schedule now.

### Dependencies
Stories 1, 2, 4 + accumulated time. Overlaps #3's z-score/trend ideas. **Size:** M–L, time-gated.

### Definition of done
- [ ] Backtest produces alarm precision/recall per candidate rule on ≥ the D-5.1 count.
- [ ] Chosen rule + false-alarm rate recorded in `docs/`/CHANGELOG.
- [ ] `monitoring.py`/`drift_monitoring.py` thresholds updated to the backtested rule (or a
      documented decision to defer).
- [ ] Branch `research/<N>-drift-calibration-backtest`; PR squash-merged; CI green.

---

## Story 6 — Human-reviewed gold eval set as promotion + prompt-regression gate

**Labels:** `enhancement`, `evaluation`, `priority:medium`

### User story
As the owner, I want a small human-reviewed gold set a candidate prompt/model must pass before
promotion, so changes are gated on measured regression against known-correct labels instead of
ad-hoc spot-checking.

### Approach (from architect)
`data/evaluation/golden_set.jsonl` + thin `scripts/eval_gold_set.py`. Records embed a **content
snapshot** + expected per-brand category/sentiment/FP decision + `failure_class`, seeded from
confirmed corrections (Stories 1/3). Runs `ArticleLabeler(prompt_version=…)`, reports per-class
agreement + regression count with `--compare-to`. **Decisions only, not evidence text.**

### Acceptance criteria
- Given the committed gold set, when a candidate prompt/model is evaluated, then the script reports
  per-failure-class + overall agreement and a **regression count** vs current production.
- Given the `model_training` promotion step (and prompt-version promotion), when a candidate is
  promoted, then passing the gate (no undocumented regressions) is a precondition — mirroring the
  #38 contrastive gate.
- The gate names its instrument (gold set + script) and pass rule (no regressions vs current
  production, or documented) — no hardcoded accuracy floor.
- Evidence text is not checked (too brittle); only category/sentiment/FP decisions.

### Design decisions & risk
- **D-6.1** — start ~30–50 records; grow via a `gold_candidate` flag set at Story 2 review time.
- **D-6.2** — spike: check what closed **#38** left behind; reuse only if it computes per-example
  agreement.
- **D-6.3** — **local pre-promotion gate inside `model_training`, not CI** (paid APIs, secrets).
- Risk: comparison runs both versions (2× API cost); seeding from corrections biases toward past
  failure modes — note this.

### Dependencies
Story 1 (candidate articles). Relates to #1 (supersedes), #38 (precedent), #54 (consumer). Feeds
`model_training`.

### Definition of done
- [ ] `data/evaluation/golden_set.jsonl` (~30–50, content snapshots) + documented schema, seeded
      from confirmed corrections.
- [ ] `scripts/eval_gold_set.py` with `--compare-to`; per-class agreement + regression count.
- [ ] Wired as a local gate into prompt-version + model promotion; CLAUDE.md updated.
- [ ] Tests: fixture set + stubbed labeler.
- [ ] Branch `feature/<N>-gold-eval-gate`; PR squash-merged; CI green. **Size:** M.

---

## Story 7 — MLflow logging of realized metrics per model version (optional, last)

**Labels:** `enhancement`, `mlops`, `priority:low`

### User story
As the owner, I want realized FP metrics + review-session summaries logged to MLflow per model
version, visible in the training tooling — if/when MLflow is on.

### Approach
Behind `MLFLOW_ENABLED`, via `src/mlops/tracking.py`, keyed by `model_version` (#115). Convenience
layer over Stories 2 and 4; fine to drop.

### Acceptance criteria
- Given `MLFLOW_ENABLED=true`, realized metrics + review-session summaries are logged keyed by
  `model_version`; given it is false, nothing is logged and nothing errors (the key test).

### Dependencies
Story 4 + #115. Optional. **Size:** S.

### Definition of done
- [ ] Logging behind the flag; no-op + no exception when off (tested).
- [ ] Branch `feature/<N>-mlflow-realized-metrics`; PR squash-merged; CI green.

---

## Story 8 — Runbook: "alert fired → what now"

**Labels:** `documentation`, `mlops`, `priority:medium`

### User story
As the owner/operator, I want a `docs/` runbook saying what to do when an alert fires, linked from
the alert email, so an alert leads to a decision instead of another ignored email.

### Approach (from architect)
Runbook in **`docs/`** linked from alert emails (`src/agent/notifications.py`). Sections are written
**with each story** (Story 1: correct a label; Story 2: run a review session; Story 3: reconstructed
provenance; Story 4: drift-only vs performance, and the burn-in note). Slot 8 = consolidation +
validation.

### Acceptance criteria
- Given the per-story sections exist, when this completes, then a single `docs/` runbook connects
  them: alert type → meaning → confirm via review → correct / retrain / do nothing.
- Given a **replay path** (e.g. `monitor_drift.py` on a crafted window — so validation isn't blocked
  on a live alert), when the author walks the runbook, then every step is executable as written and
  gaps are fixed.
- Given an alert email, when read, then it links to the relevant `docs/` runbook section.

### Dependencies
Incremental across Stories 1/2/3/4; validated consolidation after Story 4. **Size:** S on top of the
incremental sections.

### Definition of done
- [ ] `docs/` runbook committed; burn-in note included.
- [ ] Validated via a replay path; validation noted.
- [ ] `src/agent/notifications.py` alert emails link to it.
- [ ] Branch `docs/<N>-alert-runbook`; PR squash-merged; CI green.

---

## Scope cuts and pushback

1. **LLM-as-judge (#5) deferred** (kept open, cross-referenced) — different, unvalidated
   ground-truth source; revisit after Stories 1/2/6 have real ground truth.
2. **#3's distribution-monitor subsystem not adopted** (kept open) — tiered alerting + which-column
   recording fold into Stories 4/5 on the existing path.
3. **#6's YAML store + `review_queue.py` scheduler cut** (#6 superseded → close) — DB tables instead.
4. **Gold set small + seeded** (#1 superseded → close), not #1's 50-100/15-class build.
5. **Story 5 time-gated** — recording clock starts at Story 4·L1b/#136; do not schedule now.
6. **Story 7 optional, last.**
7. **Story 0 dropped** — transcripts not at risk; optional snapshot folded into Story 3.
8. **#115 kept small** (version surfacing only); the threshold mismatch is its own `priority:high`
   bug rather than being smuggled into #115.

## Resolved planning decisions (2026-09-24)

- **Q1 — epic convention:** `Epic:` tracker + new `eval-calibration` milestone. No `epic:` label.
- **Q2 — supersede vs keep:** close #1 and #6; keep #3 and #5 open, deferred.
- **Q3 — FP skip in prod:** active (`fp_classifier_enabled=True`, `fp_skip_llm_threshold=0.53`) →
  Story 4 requires the exploration sample; recall otherwise inflated/unmeasurable.
- **Q4 — #115 sequencing:** early; prerequisite of Story 4 Layer 2; kept small (version surfacing).
- **Q5 — runbook home:** `docs/`.

## Resolved design decisions from the architect review (2026-09-24)

Adopted the review in full **except** the three owner overrides:
- **D-4.1 baseline** lives in a **small DB table** (`realized_baselines`), not `registry.json`;
  the promotion step writes the window row and the baseline metric is recomputed by the same query
  as the live window.
- **Orphaned `brand_labels`** are **documented, not deleted** (trail preferred). Reframed as **data
  hygiene, not a scorecard bug** — export already filters `labeling_status=='labeled'`
  (`export_website_feed.py:587`).
- **Threshold mismatch** (`config 0.475` vs `registry 0.5429` vs `env-skip 0.53`) becomes its own
  **`priority:high`** bug in this milestone; **#115 stays small** (version surfacing only).

Everything else adopted verbatim: C1 status mapping + C2 `continued_to_llm` filter in Story 4 ACs;
exploration routing at `pipeline.py:404`, kept **separate** from Story 2's sample; SKIPPED (not
UNKNOWN) below the sample floor; Story 4 L1 as three increments (email fix first); Story 2 frame
`{labeled, skipped, false_positive}`; Story 3 hard timebox + UUID-matching caveat; Story 5 clock
starts at Story 4·L1b/#136; **#136 owns** the drifted-column plumbing (Story 4 consumes).

## Forward-compatibility (architect)

- #136 and Story 4 both touch `print_summary_json`/`REQUIRED_SUMMARY_FIELDS` — resolve ownership up
  front (#136 owns).
- Story 4's performance verdict plugs into #72's `silent-success` contract; SKIPPED-vs-UNKNOWN keeps
  it from reintroducing vacuous health.
- Keep the realized-performance query `classifier_type`-generic for the planned ESG multi-label
  classifier.
- A scalar-hardcoded D-4.1 would rot exactly like the thresholds this epic replaces — hence the
  same-query reference window.
