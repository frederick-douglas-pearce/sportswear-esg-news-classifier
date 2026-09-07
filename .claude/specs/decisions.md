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
- **`gzip` itself fails** (disk full): the pipeline is non-zero, and `set -e` aborts the function
  before `$?` can be read.

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
   audit below: this is the second site `pipefail` changes, and it is the dangerous one.
6. The AC3 restore test must assert the failure-guidance branch **ran**, not merely that the success
   line is absent -- otherwise it is satisfied by `pipefail` alone and the restructure is unguarded.

**Audit result (AC5) -- two sites change under `pipefail`, not one.**

`local var=$(cmd | cmd)` returns `local`'s own status, so the pipeline status never reaches
`set -e`. Every `local`-assigned command substitution -- the `ls | wc -l` sites in
`rotate_backups()`, and `ls | wc -l`, `du -sh | cut`, `ls -t | head -1`, `stat | cut`,
`psql | tr` in `show_status()` -- is unaffected. `list_backups()` holds no `local` assignment at
all: its `$(ls -A ...)` sits inside a `[ ... ]` test in an `if` condition, so it is safe by
`if`-exemption rather than by `local`, and its one pipeline is the `|| true` site above. (An
earlier draft of this entry listed `list_backups()` among the `local`-assigned functions. It is
not one.)

The two sites that **do** change:
1. **`BACKUP_SIZE=$(du -h "$DAILY_PATH" | cut -f1)` in `create_backup()`** -- the only plain
   (non-`local`) assignment whose right-hand side is a *pipeline*, so its status now propagates to
   `set -e`. (Other plain assignments exist, such as `TIMESTAMP=$(date ...)`; `pipefail` has
   nothing to reach in them.) An earlier draft called this
   "correct behaviour on a just-written file". It is not merely correct-and-boring: it aborts
   *inside* the success branch, after a good archive is on disk, so the result is the AC2 invariant
   inverted -- archive kept, non-zero exit, no success line, no rotation, and the `rm -f` in the
   `else` never reached, with no error message at all. Guarded per decision 5 and covered by
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
30 bytes on this host, three containers -- in a single buffered write and exits before `grep` can
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

**Conscious exclusions:** the `DROP DATABASE` / `CREATE DATABASE` / `CREATE EXTENSION` sequence in
`restore_backup()` can leave the database dropped-but-not-recreated (#91). `check_container()`'s
signal collapse (#93). Neither is a pipeline, and both are outside AC1-AC5.

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

