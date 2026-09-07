# Decision Log

Decisions made during planning and execution of this project's epics. Each entry records the decision, alternatives considered, rationale, and date.

---

## D001: Value Hierarchy for AgentFluent Workflow Epic

**Date:** 2026-05-07
**Context:** Setting priorities for the agentfluent-workflow epic, which serves both engineering improvement and marketing goals.
**Decision:** Quality of improvement (1st) > marketing storytelling (2nd) > iteration velocity (3rd).
**Alternatives considered:**
- Speed-first: maximize sessions quickly to generate data. Rejected -- produces low-quality data and forced narratives.
- Marketing-first: optimize for publishable results. Rejected -- risks cherry-picking and artificial experiments.
**Rationale:** The user explicitly stated: "quality of improvement should be the most important goal to achieve" and "good stories to tell for marketing purposes should generally be prioritized over speed of delivery." Genuine improvements produce authentic stories; the reverse is not guaranteed.

---

## D002: Wait for AgentFluent v0.6 Before Baseline

**Date:** 2026-05-07
**Context:** AgentFluent v0.6 adds quality signals (`USER_CORRECTION`, `FILE_REWORK`, `REVIEWER_CAUGHT`) and date-range filtering (`--since`/`--until`). The baseline could be captured with v0.5 (current) or v0.6.
**Decision:** Wait for v0.6 to ship before locking the baseline so quality signals are available from the start.
**Alternatives considered:**
- Capture v0.5 baseline now, re-run after v0.6. Rejected -- creates two baselines, complicates the narrative, and the v0.5 baseline lacks the quality axis that makes the story compelling.
**Rationale:** v0.6 is expected to ship within days. The quality axis is the most interesting diagnostic dimension for the case study (architect agent usage shows up in quality signals, not just cost). A short wait produces a much stronger starting point.

---

## D003: Backlog Integration -- Existing Issues Stay Flat

**Date:** 2026-05-07
**Context:** The repo has 8 open issues (#1-#7, #12). The question is whether to retroactively label them under the new epic.
**Decision:** Option A + light C. Existing issues stay flat and unlabeled-by-epic. New epic creates only new issues. But Phase stories note which existing issues are natural candidates to tackle *during* a phase to produce two-for-one session data (e.g., implementing #5 LLM-as-judge during Phase 1 generates both engineering value and AgentFluent measurement data).
**Alternatives considered:**
- A (pure separation): New epic is fully independent. Simple but misses synergy.
- B (retroactive labeling): Relabel existing issues under the epic. Rejected -- changes their meaning and adds noise to the epic's scope.
- C (full integration): Make existing issues children of the epic. Rejected -- conflates two different scopes.
**Rationale:** The user agreed with "A + light C." This keeps the epic cleanly scoped while acknowledging that the best session data comes from real engineering work on existing issues.

---

## D004: Marketing Form Factor -- Case Study + Blog Posts in `social/`

**Date:** 2026-05-07
**Context:** Marketing artifacts need a home. Options ranged from README excerpts to standalone docs to blog posts.
**Decision:** Two parallel tracks starting from Phase 0:
- **Track B -- Case study:** `social/case-study-agentfluent.md`. Living document; each phase appends a section. Source of truth for marketing narrative.
- **Track C -- Blog posts:** `social/posts/YYYY-MM-DD-slug.md`. Each post drafted in this project, then transferred to the personal website's `_posts/` folder when ready to publish.
Both live in `social/` (gitignored). README excerpt (Track A) deferred -- derivable from case study later.
**Alternatives considered:**
- A only (README excerpt): Too limited for the depth of content.
- B only (case study): No publication path.
- C only (blog posts): No persistent narrative thread across phases.
- D (docs/ folder): Rejected by user -- `docs/` is git-tracked, marketing drafts should not be in version control.
**Rationale:** User specified: "doc file should be in a new folder named 'social' that is gitignored" and "the blog posts also go in the 'social' folder but the posts' ultimate purpose is being published on my personal website." The `social/` directory is already created and gitignored.

---

## D005: Revise `social/` Gitignore Policy -- Commit Case Study, Keep Baselines and Posts Local

**Date:** 2026-05-09
**Context:** D004 placed the case study, blog drafts, and baseline JSONs all in a fully gitignored `social/` directory. Architect reviews of #15-#18 (specifically the blocking concern raised on #16) flagged that the case study cannot be committed under that policy, which contradicts the dual-purpose theme: the case study is the canonical narrative artifact and should be version-controlled alongside the project. Independent security analysis on #15 confirmed that baseline JSONs should remain gitignored due to local file paths, tool arguments, and error messages in AgentFluent's output.
**Decision:** Replace the blanket `social/` gitignore with a negation-rule policy:
- `social/case-study-agentfluent.md` -> committed (canonical case study; tracked artifact)
- `social/posts/` -> gitignored (drafts published via the `github_pages` repo; no need for two committed copies)
- `social/baselines/` -> gitignored (security: JSON contains local file paths, tool args, error messages)

`.gitignore` updated to:
```
social/*
!social/case-study-agentfluent.md
```

**Alternatives considered:**
- Commit everything in `social/`. Rejected -- baseline JSONs leak local paths and tool args; the marginal credibility gain is not worth the security exposure.
- Keep blanket `social/` ignore. Rejected -- prevents the case study from being version-controlled, contradicts the dual-purpose theme of the epic, and creates a workflow problem for #16 (no diff to review).
- Selective explicit ignores (`social/posts/`, `social/baselines/`). Viable alternative; rejected for now in favor of the negation-rule allowlist (tighter by default). Can be revisited if more committable artifacts are added.

**Rationale:** The case study is the source-of-truth narrative artifact for the AgentFluent epic. Committing it makes the engineering and marketing tracks visibly connected in the repo, supporting the "every workflow improvement is also a data point" theme. Blog drafts are transient (their final home is the `github_pages` repo). Baseline JSONs have the highest information density per byte but also the highest local-path exposure -- the case study cites their numbers inline, making the JSONs reference data rather than canonical artifacts.

**Supersedes:** D004's "both live in `social/` (gitignored)" provision specifically. The rest of D004 (case study and blog post structure, voice, format, transfer path for posts) remains in effect.

---

## D006: `backup_db.sh` Failure Semantics -- `pipefail` Plus Reachable Failure Branches (#89)

**Date:** 2026-09-06
**Code references:** by symbol and quoted construct, per [D007](#d007-decision-records-cite-symbols-and-quoted-constructs-not-line-numbers). Where "before" is meant, it is the state at base commit `3bb8fbe`.

**Context:** PR 0 of the `silent-success` epic (#72). `scripts/backup_db.sh` opened with `set -e` and
no `pipefail`, so a `pg_dump` failure inside the `pg_dump ... | gzip > "$DAILY_PATH"` pipeline in
`create_backup()` was invisible -- a pipeline reports its last command's status, and `gzip` exits 0
on the partial stream, so a truncated archive was recorded as a good backup. The same shape in
`restore_backup()` (the pre-restore `pg_dump | gzip > "$PRE_RESTORE"` copy, and
`gunzip -c "$backup_file" | ... psql`) loaded a partial dump and reported
"Restore completed successfully!".

**The second half, and the mechanism stated correctly.** The `if [ $? -eq 0 ]` handlers in both
functions were unreachable -- but *not*, as three drafts of this entry claimed, because `set -e`
aborted first. That is only one of two routes, and it is not the one #89 is about:
- **`pg_dump` dies mid-stream** (the #89 case): the pipeline reports `gzip`'s 0, nothing aborts,
  `$?` **is** read, it **is** 0, and the *success* branch runs. Rotation runs. The truncated archive
  is reported by `list`/`status` as the latest backup.
- **`gzip` itself fails** (disk full): the pipeline is non-zero, and `set -e` aborts the *script*
  before `$?` can be read. (`create_backup` is called bare from the `case` dispatch, not in a
  tested context, so there is no enclosing construct for `set -e` to stop at.)

Either way the `else` -- including the `rm -f "$DAILY_PATH"` cleanup -- was dead code. Recording
both routes because stating only the abort teaches that `set -e` alone notices a mid-pipeline
death, which is the belief that produced #89 in the first place.

**Decision:**
1. `set -e` -> `set -eo pipefail` (AC1, all three pipelines).
2. Make the failure branches reachable by putting each pipeline in the `if` condition
   (`if <pipeline>; then ... else ... fi`), which exempts it from `set -e` while `pipefail` still
   reports the true status.
3. Wrap the pre-restore copy in `restore_backup()` the same way. **Rationale narrowed by the
   architect:** `pipefail` alone already aborts before the `DROP DATABASE`, so this does *not*
   protect against the destructive drop; its actual value is removing the partial
   `pre_restore_*.sql.gz` and emitting an operator-facing message.
4. Guard the `ls -lh ...*.sql.gz | awk` listing in `list_backups()` with `|| true` -- the only
   top-level pipeline `pipefail` would newly *abort*. Its enclosing guard tests `ls -A` (directory
   non-empty), not the `*.sql.gz` glob, so a directory of non-archive files would abort the listing.
5. Guard `BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1)` with `|| BACKUP_SIZE="unknown"`. See the
   audit below: of the sites `pipefail` changes, this is the dangerous one.
6. The AC3 restore test must assert the failure-guidance branch **ran**, not merely that the success
   line is absent -- otherwise it is satisfied by `pipefail` alone and the restructure is unguarded.

**Audit result (AC5) -- three sites change under `pipefail`, not one.**

`local var=$(cmd | cmd)` returns `local`'s own status, so the pipeline status never reaches
`set -e`. Every `local`-assigned command substitution -- the `ls | wc -l` sites in
`rotate_backups()`, and `ls | wc -l`, `du -sh | cut`, `ls -t | head -1`, `stat | cut`,
`psql | tr` in `show_status()` -- is unaffected. `list_backups()` holds no `local` assignment at
all: its `$(ls -A ...)` sits inside a `[ ... ]` test in an `if` condition, so it is safe by
`if`-exemption rather than by `local`, and its one pipeline is the `|| true` site above. (An
earlier draft of this entry listed `list_backups()` among the `local`-assigned functions. It is
not one.)

The three sites that **do** change (over and above the three pipelines this change deliberately
restructures):

0. **The `ls -lh ...*.sql.gz | awk` listing in `list_backups()`** -- the only *top-level* pipeline
   `pipefail` would newly abort. Guarded with `|| true` per decision 4 and covered by
   `tests/test_backup_db_script.py::test_list_backups_survives_a_failing_glob_under_pipefail`.
   Listed here for completeness: decision 4 already handles it, and an earlier draft of this
   heading counted only the two below.
1. **`BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1)` in `create_backup()`** -- the only plain
   (non-`local`) assignment whose right-hand side is a *pipeline*, so its status now propagates to
   `set -e`. (Other plain assignments exist, such as `TIMESTAMP=$(date ...)`; `pipefail` has
   nothing to reach in them.) An earlier draft called this
   "correct behaviour on a just-written file". It is not merely correct-and-boring: it aborts
   *inside* the success branch, after a good archive is on disk, so the result is the AC2 invariant
   inverted -- archive kept, non-zero exit, no success line, no rotation, and the `rm -f` in the
   `else` never reached. The *script* says nothing; `du`'s own stderr still reaches the operator,
   so "no error message at all" (an earlier draft) overstated it. Guarded per decision 5 and covered by
   `tests/test_backup_db_script.py::test_backup_survives_a_du_failure_after_the_archive_is_written`.
2. **`check_container()`'s `docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"`** --
   `pipefail` changes this pipeline's *status*, and there the status **is** the branch condition, so
   this is a change of *value*, not of abort behaviour. **Stated, not fixed here** -- see below.

**`check_container()` -- checked and stated, deferred to #93.** The first draft of this audit
recorded it as "exempt (`if !` condition)". `if !` exempts a command from a `set -e` *abort*; it
does not stop `pipefail` from changing the pipeline's status. Correct conclusion, wrong mechanism,
and the wrong mechanism is what kept it untested.

What the correction then found, and what it got wrong in turn, is recorded in #93 with the
measurements. In summary: the SIGPIPE inversion is **real but unreachable here**. The mechanism is a *race*,
not a size threshold: it fires whenever `docker ps` is still writing when `grep -q` matches and
stops reading. Exceeding the ~64 KiB pipe buffer is simply the reliable way to force that (measured:
running at 48,894 bytes, inverted at 108,894), but the acceptance-gate verifier inverted the same
construct with **two lines** of output by inserting a delay after the matching line. What makes it
unreachable here is not the byte count as such but that real `docker ps` emits its whole list --
measured at 30 bytes, two running containers -- in one buffered write and exits before `grep` can
close the pipe. 30/30 runs of the real construct reported RUNNING. A second draft claimed it "would have failed the
nightly cron backup on a healthy system"; it would not have. A third claimed that removing the pipe
was a *prerequisite* for reporting a Docker failure separately from an absent container; that is
false too -- `PIPESTATUS` exposes per-stage status and the `if !` condition keeps the function alive
to read it.

AC5 asks that the other pipelines be **checked and stated**. This one is checked, stated, measured
and filed. Rewriting it is a behaviour change no acceptance criterion here asks for, and the
attempt produced three successive wrong justifications and a new signal-collapse of its own -- so it
belongs in its own issue with its own tests, which is #93.

**Alternatives considered:**
- **`ERR` trap** for the failure branches -- rejected: needs `set -E` for function inheritance, and
  cannot scope the `rm -f` to the one file being written.
- **`PIPESTATUS` inspection** -- rejected *for `create_backup()`*, where the pipeline is bare and
  `set -e` aborts before it can be read. Note the scope of that rejection: it does **not** hold for
  a pipeline in an `if` condition, and applying it there was one of the errors above.
- **`|| { ...; }`** instead of `if ... then ... else ... fi` -- equivalent in effect, rejected on
  readability for a multi-statement success branch.
- **`pipefail` alone, leaving the `if [ $? -eq 0 ]` handlers in place** -- rejected, and this is the
  substantive alternative: it makes the failure *louder* without making the cleanup reachable, so a
  truncated archive still survives on disk and is still reported as the latest backup. It satisfies
  AC1 and leaves AC2 unmet.
- **Rewriting `check_container()` in this PR** -- rejected per the above; filed as #93.

**Rationale:** the two halves of the defect have different causes and need different fixes, and
fixing only the visible one is worse than it looks. `pipefail` addresses *detection* -- the script
can now see that `pg_dump` failed. Putting the pipeline in the `if` condition addresses
*reachability* -- the handler that cleans up after that failure can now run. Shipping only the first
converts "silently wrong" into "loudly wrong with the bad artifact still on disk", which still loses
data on the next restore.

**Conscious exclusions**, and they are excluded for different reasons -- an earlier draft gave both
the same one, which was false of the second:
- The `DROP DATABASE` / `CREATE DATABASE` / `CREATE EXTENSION` sequence in `restore_backup()` can
  leave the database dropped-but-not-recreated (#91). These are **not pipelines**, `pipefail` does
  not touch them, and `set -e` already aborts loudly -- genuinely outside AC1-AC5.
- `check_container()`'s signal collapse (#93). This one **is** a pipeline, and it **is** one of the
  three sites above whose behaviour `pipefail` changes. It is excluded not because AC5 fails to
  reach it but because AC5 asks that such sites be *checked and stated*, which is done above and in
  #93. Two things an earlier draft left unsaid: the hazard is one this change **introduces**, not
  one it inherits -- on `main` that pipeline runs under plain `set -e`, where its status cannot
  flip the branch -- and its *signal collapse* is **untested**. The happy path runs in every test
  in the file, via the fake `docker ps`; what has no test is the collapsed verdict, because a test
  asserting the correct verdict would fail against the code as it stands. That is exactly what #93 is for.

**Forward compatibility:** the `if <pipeline>; then ...` restructure gives #80 ("assert the output
is *good*") a home -- `gzip -t`, size and row-count checks compose into the `then` branch without
touching the failure path.

---

## D007: Decision Records Cite Symbols and Quoted Constructs, Not Line Numbers

**Date:** 2026-09-06
**Context:** D006 was written alongside the diff it describes, and its `file:line` citations were
stale before the entry landed -- they resolved against base `3bb8fbe` but not against the file
shipping in the same PR. Bare line numbers are the one citation form that rots *silently*: the
reference still points at a line, just the wrong one, so a reader is misdirected rather than
alerted. For a project whose current epic is about artifacts that lie about themselves, that failure
mode is the wrong one to keep.

**Decision:** in `.claude/specs/decisions.md` and `social/postmortems/`, cite code as follows.
1. **Symbol first** -- `check_container()`, "the `pg_dump | gzip` pipeline in `create_backup()`".
   Function names survive line moves and in-function edits.
2. **Quote the construct when precision matters** -- `set -e` without `pipefail`, `if [ $? -eq 0 ]`.
   A stale quote fails *loudly*: grep finds nothing and the reader knows it moved.
3. **Pin a line number to an immutable ref when one is genuinely needed** -- `path@<sha>:<line>` or a
   commit permalink, never a bare `:NN` that implicitly means "current".
4. **Label which side of a diff a reference describes.** A decision record straddles a change, so
   "before"/"after" is not inferable from the reference itself.

**Alternatives considered:**
- **Keep bare line numbers, refresh them on edit** -- rejected: nothing enforces the refresh, and
  the failure is silent, so this is exactly the unreachable-cleanup pattern in documentation form.
- **Drop code references entirely** -- rejected: the specificity is what makes these records usable.
- **SHA-permalink everything** -- rejected as the default: correct but heavy, and it makes an entry
  unreadable outside a browser. Reserved for point 3.

**Rationale:** prefer citation forms that fail loudly over forms that fail silently. This is the
same principle the code in D006 adopts -- a truncated backup should fail rather than be recorded as
good -- applied to the project's own documentation.

---

## D008: Drift Monitoring Reports an Indeterminate Verdict Instead of Health (#71)

**Date:** 2026-09-07
**Code references:** by symbol and quoted construct, per [D007](#d007-decision-records-cite-symbols-and-quoted-constructs-not-line-numbers). Where "before" is meant, it is the state at base commit `a9603ad`.

**Context:** Issue #71, order 1 of the `silent-success` epic (#72). The `drift_monitoring`
workflow failed on **223 of 232** scheduled runs since 2026-01-17 and on **219** of those
simultaneously logged `No action needed - all classifiers healthy` and exited 0 (re-counted from
`logs/agent/cron_drift_monitoring_*.log` on 2026-09-07; the issue's 221/230/217 were measured
2026-09-05). It produced a real verdict twice, on 2026-01-19 and 2026-01-23, and never again.

Four defects compound. **The current behaviour of each was reproduced at `a9603ad`; the
*historical* attribution was not, and one inherited claim was wrong** -- see the correction under
defect 1.

- `data/reference/fp_reference.parquet` (written 2026-01-17) carries 8 columns and no
  `novelty_score`, while `_evidently_drift_check` guards the novelty **stats** block on
  `current_data` alone and then reads `reference_data["novelty_score"]`. That asymmetry, not the
  staleness by itself, is the live raise site.

  **Correction (review round 1).** An earlier draft of this record, and of the changelog, said this
  raised "on every run from 2026-01-17". It cannot have: `novelty_score` was added to
  `classifier_predictions` on **2026-01-25** (commit `4c17395`). The earlier failures have other
  logged causes. The claim was inherited from #71's body and repeated as though verified -- which
  is the defect class this epic is about, committed in the record of fixing it.

  **Three** stats reads had the asymmetric shape, not one. `novelty_score` is merely the one that
  fired; a reference sharing only `brand_*` columns would have died on `reference_prob_mean` first.
  The first revision of this fix corrected `novelty_score` alone and asserted the other two were
  already symmetric.
- `monitor_drift.py` `main()` prints `Error running drift analysis: {e}` to **stdout** and returns
  1; `check_fp_drift` logs `result.stderr`, which is empty. Hence **220** log lines reading
  `FP drift check failed: ` with nothing after the colon; the 3 non-empty ones name causes
  (`Evidently not installed`, a missing `uv`) that never reached the analysis.
- `evaluate_drift_results` reads `context.get("fp_drift_detected", False)` and
  `generate_drift_report` reads `context.get("fp_healthy", not context.get("fp_drift_detected", False))`.
  With both keys absent -- a check that never ran -- the second is `not False`, i.e. **healthy**.
- `classifier_predictions` has never held an `ep` row, so the EP check finds nothing to compare.
  No comparison actually occurs: `check_drift` returns from its empty-frame branch *before*
  dispatching to either checker, so the `drift_detected=False` it reports is fabricated.

**The governing rule the architect named, adopted for the whole change:** *a fix for a
silent-success bug must not itself infer health from absence anywhere in its new code path.*
Every decision below is an application of it.

**Decision:**

1. **`DriftReport` gains a typed `indeterminate: bool` field**, set at `check_drift`'s empty-frame
   branch and `_evidently_drift_check`'s "No columns available" branch. **This overrides the plan's
   first draft**, which inferred indeterminacy from `"error" in report.details`. `details` is a
   free-form grab-bag also carrying `columns_checked`, `reference_size` and per-column p-values;
   deriving control flow from a key's presence in it is the same meaning-hidden-in-a-dict shape the
   epic removes, and #74/#76 **will** consume this signal programmatically across ~1,040 archived
   run YAMLs (both issues are open and unimplemented at the time of writing).
   `drift_detected` is already a first-class field; "could the analysis produce a verdict" is
   equally load-bearing.
2. **A three-value exit-code contract** in a new `src/mlops/exit_codes.py`: `0` no drift, `1` drift
   detected, `2` indeterminate. Before, exit 1 meant *both* "drift detected" and "the analysis
   raised", and `ScriptResult.success` is `exit_code == 0`, so the workflow could not tell them
   apart and fell back to scraping stdout. The module exports its **own**
   `NON_RETRYABLE_EXIT_CODES`, rather than reusing `src/labeling/exit_codes.py`'s, so each contract
   stays self-describing. The two modules are not a DRY violation: they encode different domain
   contracts that merely share the integers 0/1/2.
3. **`HealthVerdict` (`healthy | degraded | unknown | skipped`) in a new `src/agent/health.py`, enum
   only.** Not a `WorkflowStatus` member -- #72 constraint 1 -- because a health verdict is not a
   lifecycle state and archives written with `status: unknown` would need migration. The
   exit-code-to-verdict mapping stays next to the drift workflow: #77/#78/#79 will have different
   exit-code semantics and must not be made to depend on drift's.
4. **A terminal step `fail_on_unknown_verdict`** raises when any verdict is not *explicitly*
   `healthy`/`degraded`/`skipped`, so `Workflow.run`'s existing `try/except` marks the workflow
   FAILED. Positioned last so `send_drift_alerts` and `generate_drift_report` run and
   `complete_step` first -- `_execute_step` calls `complete_step` only on the non-raising path, so a
   raise inside the report step would discard the report from the run archive that #75/#76 read.
   Testing `== "unknown"` alone was rejected: a handler returning a dict without its verdict key
   leaves the value `None` and would sail past, re-creating the bug at the last gate. Not
   `skip_on_dry_run`.
5. **EP is gated by config (`AGENT_EP_DRIFT_ENABLED`, default false), not by a data count.**
   EP-on-hold is a governance decision, not a data artifact: zero predictions is the symptom,
   "on hold per CLAUDE.md" is the cause, and the flag encodes the cause. A data check would
   re-enable monitoring the instant one stray `ep` row landed, then compare it against a
   nonexistent reference and emit low-value `unknown` alerts.
6. **A verdict with no evidence is not a verdict.** Exit 0 or 1 with no parseable, schema-valid
   JSON summary resolves to `unknown`, not `healthy`; required summary fields are read fail-loud
   rather than with `.get(default)`.

**Alternatives considered:**

- **Infer indeterminacy from `details["error"]`** -- rejected per (1); brittle, untyped, and
  invisible to the archive consumers in #74/#76. Matching on the error *string* was rejected as
  worse still.
- **Spell the health vocabulary locally in `drift_monitoring.py` and let #74 promote it** --
  rejected: "a term introduced in one workflow and not the others" is precisely the hazard this
  project's `ARCHITECT_TRIGGERS` names, and #71 exercises all four members, so the enum is not
  speculative.
- **Raise inside the check step, or inside `generate_drift_report`** -- rejected: the first
  starves AC2 of its alert and summary, the second discards the report from the archive.
- **Wait for #73 to supply the non-zero workflow status** -- rejected: #73 is ordered after #71,
  and a 231-day live incident should not wait on a foundation issue. The device is independent of
  `base.py`, survives #73, and is **a bridge for #74 to delete** once verdicts have a first-class
  escalation path. Recorded here so it does not linger as orphaned dead code.
- **Data-driven EP skip** -- rejected per (5). Note the rejection is not that "0 rows to skipped"
  is itself the epic's bug (the bug is 0 rows to *healthy*), but that it encodes the symptom
  instead of the cause and re-enables itself on noise.
- **Keep `_parse_drift_output` alongside the structured path** -- rejected: two sources for one
  metric, and its `elif "healthy" in line_lower` is a second home for the absence-is-benign
  defect.

**Amendment (review round 1): the guards had to land on BOTH code paths.** The decision as first
implemented set `indeterminate` only inside `_evidently_drift_check`. But `mlops_settings.
evidently_enabled` defaults to **false**, and `_setup_evidently` silently falls back to the legacy
path on `ImportError` -- and `_legacy_drift_check` ended with
`max(drift_scores) if drift_scores else 0.0`, so "nothing was comparable" became drift score 0.0,
then no drift, then exit 0, then healthy. That is #71's exact shape surviving on the path taken by
default, inside the change written to remove it. `_legacy_drift_check` now returns `indeterminate`
on an empty score set, and the Evidently path additionally guards the case where a snapshot yields
no readable `ValueDrift` metric at all.

**Rationale:** the epic's thesis is that a missing signal must never collapse into "healthy". Every
decision above moves a signal from *inferred by absence* to *stated explicitly and typed*: the
indeterminate flag, the exit code, the verdict enum, the fail-loud field reads, and a terminal step
that treats an absent verdict exactly as it treats a failed one.

**Deferred, with rationale, rather than dropped -- each filed as its own issue so this record
resolves in both directions:**
- **[#94](https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier/issues/94)** --
  a minimum-sample-size floor. Exit 2 covers empty frames and absent columns, not "enough rows to
  compute, too few to mean anything".
- **[#95](https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier/issues/95)** --
  "all healthy" being vacuously true when every check is skipped. Note this is **implemented here**
  (`evaluate_drift_results` computes `bool(checked) and not drifted and not unknown`, pinned by
  `test_all_skipped_is_not_healthy`); what is deferred is generalising it beyond drift.
- **[#96](https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier/issues/96)** --
  a forgotten `AGENT_EP_DRIFT_ENABLED` leaving EP dark after it resumes.
- **[#97](https://github.com/frederick-douglas-pearce/sportswear-esg-news-classifier/issues/97)** --
  the reference window always contains the window it is compared against (7.6% overlap at 90 days),
  biasing every comparison toward "no drift".
