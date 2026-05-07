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
