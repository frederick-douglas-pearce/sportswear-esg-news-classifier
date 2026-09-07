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
**Context:** PR 0 of the `silent-success` epic (#72). `scripts/backup_db.sh:5` sets `set -e` without
`pipefail`, so a `pg_dump` failure inside `pg_dump | gzip > "$PATH"` is invisible -- `gzip` exits 0
on the partial stream and a truncated archive is recorded as a good backup (`:80`, `:138`), or a
partial dump is loaded and reported as "Restore completed successfully!" (`:149`). Compounding it,
the `if [ $? -eq 0 ]` handlers at `:82` and `:151` are unreachable: under `set -e` a non-zero
pipeline aborts the function before `$?` is read, so the `else` at `:102-105` -- including the
`rm -f "$DAILY_PATH"` cleanup -- is dead code. Architect review requested at the dev-loop's
step-4 gate.

**Decision:**
1. `set -e` -> `set -eo pipefail` (AC1, all three pipelines).
2. Make the failure branches reachable by putting each pipeline in the `if` condition
   (`if <pipeline>; then ... else ... fi`), which exempts it from `set -e` while `pipefail` still
   reports the true status. Chosen over an `ERR` trap (needs `set -E`, cannot scope the `rm` to one
   file), `PIPESTATUS` inspection (unreachable -- `set -e` aborts first), and `|| { ... }` (less
   readable for a multi-statement success branch).
3. Wrap the pre-restore copy at `:138` as well. **Rationale narrowed by the architect:** `pipefail`
   alone already aborts before the `DROP DATABASE` at `:144`, so this does *not* protect against
   the destructive drop; its actual value is removing the partial `pre_restore_*.sql.gz` and
   emitting an operator-facing message.
4. Guard `:175` (`ls -lh ...*.sql.gz | awk`) with `|| true` -- the only top-level pipeline that
   `pipefail` would newly abort.
5. The AC3 restore test must assert the failure-guidance branch **ran** (`Restore failed!` plus the
   pre-restore path), not merely that the success line is absent. Without it the test is satisfied
   by `pipefail` alone and the restore restructure is unguarded.

**Audit result (AC5):** `local var=$(cmd | cmd)` returns `local`'s own status, so the pipeline
status never reaches `set -e`. Every `ls | wc -l` / `du | cut` / `ls -t | head -1` site
(`:189, :196, :203, :220-222, :231, :236, :238, :247`) is `local`-assigned and therefore unaffected
by `pipefail`. `:83` is the one *plain* assignment where the status does propagate (correct
behaviour on a just-written file). `:52` is exempt (`if !` condition). Confirmed twice: by architect
review and by an empirical bash harness.

**Conscious exclusion:** `:144-146` (`DROP` / `CREATE` / `CREATE EXTENSION`) can leave the database
dropped-but-not-recreated if a middle statement fails. Not pipelines, and `set -e` aborts loudly
rather than silently, so outside this defect class and outside #80's scope. Recorded, not fixed.

**Forward compatibility:** the `if <pipeline>; then ...` restructure gives #80 ("assert the output
is *good*") a home -- `gzip -t`, size and row-count checks drop into the `then` branch by
composition, without touching the failure path this change establishes.
