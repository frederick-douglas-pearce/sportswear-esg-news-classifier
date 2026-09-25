> **SUPERSEDED (2026-09-24):** these two stories are now filed as **#154** (Story 9) and **#155**
> (Story 10). Kept for history; the authoritative spec is
> [`prd-eval-calibration.md`](prd-eval-calibration.md). Do not file from this document.

# eval-calibration — new issue bodies to file

Two new stories for epic **#152** (`eval-calibration` milestone). Full context in
[`prd-eval-calibration.md`](prd-eval-calibration.md). Do not file duplicates of the already-filed
stories (#140–#151) — these two are additions from the 2026-09-24 scope change.

Cross-reference key: **#143** = correction/label audit log (Story 1), **#146** = review session
record (Story 2), **#142** = exploration sample (Story 4·L1b). Fill the `<N>` self-references and
confirm **#141** / **#147** scope at filing.

---

## Issue 1

**Title:** Label provenance + stop exporting classifier-only labels

**Labels:** `bug`, `mlops`, `priority:high`
**Milestone:** `eval-calibration`
**Blocked by:** none
**Relates to:** #152 (epic), #143 (audit log = human-source record), #141, #142 (exploration replenishment), #115

**Body:**

## Summary

The FP classifier trains — and is evaluated — on its own predictions. `scripts/export_training_data.py:101`
exports **all** `labeling_status='false_positive'` articles as FP negatives. Of **941** false_positive
articles, **390** carry a `classifier_predictions` row with `classifier_type='fp'` AND
`action_taken='skipped_llm'` — decided by the classifier alone, never seen by the LLM or a human.
Training/evaluating on these bakes the prior model's errors in as ground truth, invisibly (the eval
split has the same contamination).

Classifier-only negatives by skip month: `2025-12:6, 2026-01:91, 2026-02:51, 03:44, 04:65, 05:45,
06:21, 07:29, 08:23, 09:15` (390 total). Positives (`labeled`+`skipped`) = **4122**; excluding the
390 leaves ~551 negatives, so the fix also shifts class balance.

This story (a) records **where each status label came from** and (b) makes the FP training/eval
export **exclude classifier-only negatives** and **report what it excluded**.

## Context / provenance sources

Provenance to represent: `llm`, `fp_classifier`, `human` (from #143's audit log), and
`owner_attested_cutoff` — a coarse attestation that articles before an owner-confirmed date were
human-reviewed during training/fine-tuning. **The attestation is weaker than per-article human
verification and must not be represented as if it were per-article.** The owner will confirm the
cutoff (likely ~late Jan 2026 per registry FP v2.5.0 2026-01-25 / v2.6.0 2026-01-26). "Early data is
clean" is a hypothesis Story 10 (`review-fps`) tests, not an assumption.

## Acceptance criteria

- **Given** an article's status label, **when** provenance is computed, **then** it resolves to one
  of `{llm, fp_classifier, human, owner_attested_cutoff}` from existing signals
  (`classifier_predictions.action_taken`, `labeling_status`, #143's `label_corrections`), and a
  **classifier-only negative** is identifiable as a `false_positive` whose only decision is a
  `classifier_type='fp' AND action_taken='skipped_llm'` row with no LLM label and no human
  correction.
- **Given** the FP training export (`export_training_data.py:101`), **when** the `fp` dataset is
  exported, **then** classifier-only negatives are **excluded** and the export **reports the excluded
  count** (total and by skip-month).
- **Given** human-verified labels exist (#143), **when** exporting, **then** the export can prefer
  human-verified labels and marks provenance on what it emits.
- **Given** an owner-attested cutoff date, **when** provenance is assigned, **then** pre-cutoff labels
  may be tagged `owner_attested_cutoff` (distinct from `human`), recorded as an attestation, not
  per-article verification.
- **Given** the eval/holdout split, **when** it is built, **then** it applies the **same exclusion**
  as training, so evaluation is not contaminated.

## Dependencies

None hard — can start immediately. Reads #143's audit log for the `human` source (degrades
gracefully before #143 lands). Identifies the cohort **Story 10** reviews; its exclusion is
replenished over time by **#142**'s exploration sample.

## Definition of done

- [ ] Provenance derivable per article; source vocabulary defined + documented (incl. attestation caveat).
- [ ] `export_training_data.py` `fp` dataset excludes classifier-only negatives; prints excluded
      counts (total + by month); eval/holdout applies the same exclusion.
- [ ] Human-verified preference/flagging (reads #143).
- [ ] Owner-attested-cutoff representation + documented as weaker-than-per-article.
- [ ] Tests: fixture classifier-only negative → excluded + counted; human-verified negative →
      retained/preferred; cutoff tagging.
- [ ] CLAUDE.md export section + runbook note (class-balance impact called out).
- [ ] Branch `fix/<N>-label-provenance`; PR squash-merged; CI green.

## Design questions (for the architect — reviewing in parallel; do not pre-decide)

- **D-9.1** Provenance derived on the fly from joins vs a materialized column/view/small table.
- **D-9.2** How the owner-attested cutoff is represented (config date / small-table row / special
  `label_corrections` row) so it is auditable and cannot masquerade as per-article verification.
- **D-9.3** Exclude classifier-only negatives outright vs downweight/keep-but-flag — excluding
  ~390/941 shifts class balance (4122 pos vs ~551 neg); flag the training-set impact for the owner.
- **D-9.4** Whether #142's exploration-sample negatives (once LLM-labeled) are the clean
  replenishment for the excluded negatives.

---

## Issue 2

**Title:** One-time review of classifier-only negatives + `review-fps` skill

**Labels:** `enhancement`, `evaluation`, `priority:high`
**Milestone:** `eval-calibration`
**Blocked by:** #143 (audit log), #146 (review session + `fp_skipped` frame), Story 9 (label provenance — file first)
**Relates to:** #152 (epic), #142 (ongoing exploration inflow), Story 6 (gold candidates)

**Body:**

## Summary

Re-ground the ~390 classifier-only FP negatives (identified by the label-provenance story) in
independent judgement, and add a repeatable `review-fps` entry point for the FP-skip inflow that
follows. An LLM second opinion is cheap (~390 Haiku calls, a few dollars); the human then reviews the
disagreements plus a sample of agreements, prioritizing the decision boundary.

## Context

Corrections go through **#143**'s audit log; sessions are recorded via **#146** using a **new sample
frame `fp_skipped`** (which #146's frame vocabulary must be extended to allow). Afterwards, ongoing
FP-skip inflow comes from **#142**'s exploration sample, making FP review a **recurring session
type**. The `review-fps` skill is a fine separate entry point but **must write the same tables** as
`review-labels` — **no parallel store**. It also **tests the "early data is clean" hypothesis** by
including a sample of pre-cutoff articles (feeding back into the provenance story's owner-attested
cutoff).

## Acceptance criteria

- **Given** the ~390 classifier-only negatives (from the provenance story), **when** the one-time
  pass runs, **then** each gets an LLM second opinion (Haiku), recorded so LLM-vs-classifier
  agreement/disagreement is computable.
- **Given** LLM and classifier verdicts, **when** the human queue is built, **then** it surfaces
  **disagreements first**, then a **random sample of agreements**, and **prioritizes** rows with
  `probability` just under **0.53** (the skip threshold).
- **Given** the "early data is clean" hypothesis, **when** the queue is built, **then** it includes a
  **sample of pre-cutoff articles** so the hypothesis is tested, not assumed.
- **Given** a human correction, **when** applied, **then** it goes through **#143**'s audit log
  (`source='human'`, no raw `UPDATE`).
- **Given** a review session, **when** opened, **then** it is recorded via **#146** with
  `sample_frame='fp_skipped'` — a value #146's vocabulary must permit.
- **Given** the `review-fps` skill, **when** it runs, **then** it writes the same `review_sessions` /
  `review_session_sample` / `label_corrections` tables as `review-labels` (no parallel store).
- **Given** #142's exploration sample produces ongoing FP-skipped→LLM-labeled inflow, **then** FP
  review is a recurring session type using the same `fp_skipped` frame.

## Dependencies

Blocked by #143, #146 (incl. the `fp_skipped` frame extension) and the label-provenance story. Feeds
Story 6 (reviewed disagreements → gold candidates) and the provenance story (reviewed labels update
provenance). Ongoing inflow from #142.

## Definition of done

- [ ] One-time LLM second-opinion pass over the classifier-only negatives; disagreements computed.
- [ ] Human review queue: disagreements → sampled agreements, near-0.53 prioritized, incl. a
      pre-cutoff sample; corrections via #143; session via #146 with `sample_frame='fp_skipped'`.
- [ ] #146's `sample_frame` vocabulary extended to include `fp_skipped`.
- [ ] `review-fps` skill writes the shared tables (no parallel store).
- [ ] Findings feed the provenance story (reviewed negatives re-sourced) and optionally Story 6.
- [ ] Tests: queue ordering (disagreements first; near-threshold priority); session + corrections
      write the shared tables.
- [ ] Runbook "reviewing FP-skips" section.
- [ ] Branch `feature/<N>-review-fps`; PR squash-merged; CI green.

## Design questions (for the architect — reviewing in parallel; do not pre-decide)

- **D-10.1** LLM second opinion as a normal labeling pass (reuse `ArticleLabeler`/pipeline) vs a
  lighter bespoke call. Reuse is leaner.
- **D-10.2** How LLM-vs-classifier verdicts are stored so the queue can be built and disagreements
  feed Story 6 — reuse `classifier_predictions`/labels vs a transient artifact.
- **D-10.3** Pre-cutoff sample design — how many, and does confirming/refuting "early data is clean"
  change the owner-attested cutoff (feedback into the provenance story)?
- **D-10.4** `review-fps` as a separate skill vs a mode/flag on `review-labels` — constraint is
  identical table writes and one shared frame vocabulary.
