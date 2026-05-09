# PRD: AgentFluent-Driven Workflow Upgrade

**Status:** Draft
**Date:** 2026-05-07
**Author:** PM Agent
**Decision log:** See `decisions.md` D001-D004 for scoping decisions.
**Backlog:** Epic #14 on GitHub; child stories #15-#24 linked below.

---

## 1. Theme

**"Use the tool, tell the story."**

This project serves a dual purpose. The sportswear ESG news classifier is a real, production system with real engineering work ahead (evaluation, model improvements, MLOps). At the same time, every Claude Code session on this project produces session data that AgentFluent can analyze. Workflow improvements guided by AgentFluent diagnostics generate before/after measurement data, which becomes the raw material for marketing content.

The engineering work and the marketing work are two views of the same activity:

```
Implement improvement  --->  Session JSONL  --->  AgentFluent analyze
       |                                                |
       v                                                v
  Better project                              Measurable delta
       |                                                |
       +----------> Case study + blog posts <-----------+
```

One-line pitch: **"Every workflow improvement is also a data point. Every data point is also a story."**

### Why this epic exists

AgentFluent baseline analysis of this project revealed:

- **100% Opus usage** with zero cost-tier routing. Every task -- trivial reads, grep searches, boilerplate edits -- runs at Opus token rates.
- **Zero subagent invocations.** The user has pm.md and architect.md agents configured and scoring 100/100 in `agentfluent config-check`, but they have never been invoked on this project.
- **43% Bash tool concentration.** High Bash reliance often correlates with retry loops and manual orchestration that subagents or skills could handle.
- **Only 2 sessions analyzed** ($4.42 total). Insufficient data for AgentFluent's pattern signals to fire.

These findings represent concrete improvement opportunities. The question is not *whether* to improve but *in what order* and *how to measure* the improvement.

### Why dual-purpose works here

Most "eat your own dogfood" initiatives fail because the dogfooding is artificial. Here, the classifier project has a genuine backlog of engineering work (8 open issues, active data collection, model iteration). AgentFluent analysis happens after sessions organically -- no contrived tasks needed. The marketing content emerges from documenting what actually happened, not from staging demonstrations.

## 2. Goals

1. **Establish a quantitative baseline** of AgentFluent diagnostics on this project's sessions, including v0.6 quality signals.
2. **Execute at least one measured improvement cycle** -- implement a workflow change guided by AgentFluent diagnostics, measure the before/after delta with `agentfluent diff`, document the result.
3. **Produce a living case study document** that accumulates evidence across phases.
4. **Draft at least two blog posts** suitable for the user's personal website -- one establishing context ("where we started"), one documenting the first experiment.
5. **Integrate pm and architect agents** into the project's workflow so they appear in session data for AgentFluent to analyze.

## 3. Non-Goals

- Modifying AgentFluent itself (that's a separate repo and backlog).
- Rushing through sessions to generate data -- organic pace, quality of improvement matters most.
- Fabricating or cherry-picking results for marketing -- document what actually happens, including failures.
- Modifying CLAUDE.md or existing project conventions (requires its own explicit story if needed).
- Resolving all 8 existing open issues -- the epic creates only new issues; existing issues are tackled organically when they produce useful session data.
- Publishing blog posts -- drafts are produced; publication is a separate transfer step.

## 4. In Scope (by Phase)

### Phase 0: Baseline and Instrumentation

**Goal:** Capture the "before" snapshot and establish marketing artifact scaffolding.

**Prerequisite:** AgentFluent v0.6 must ship first (quality signals: `USER_CORRECTION`, `FILE_REWORK`, `REVIEWER_CAUGHT`; date-range filtering: `--since`/`--until`). See D002.

| Issue | Title | Effort | Notes |
|-------|-------|--------|-------|
| #15 | Run AgentFluent v0.6 baseline and save snapshot | S | Run `agentfluent analyze --project classifier --format json > social/baselines/phase0-baseline.json`. Also save human-readable CLI output. Record key metrics in case study. |
| #16 | Create case study scaffold in `social/` | S | Create `social/case-study-agentfluent.md` with sections: Intro, Baseline, (Phase 1 placeholder), Conclusion. Populate Intro and Baseline from #15 data. |
| #17 | Draft "where we started" blog post | M | Draft `social/posts/YYYY-MM-DD-agentfluent-baseline.md` with al-folio frontmatter. Covers: what the project is, what AgentFluent found, what we plan to do. Must be prettier-clean. |
| #18 | Document Phase 0 session data in case study | S | After #15-#17 are done, run `agentfluent analyze` again to capture Phase 0 itself as data. Append observations to case study. |

### Phase 1: First Experiment

**Goal:** Execute one AgentFluent-guided improvement, measure the delta, tell the story.

**Prerequisite:** Phase 0 complete. The specific improvement target is determined by baseline diagnostics -- the stories below are parameterized.

| Issue | Title | Effort | Notes |
|-------|-------|--------|-------|
| #19 | Select improvement target from baseline diagnostics | S | Review Phase 0 baseline. Select the highest-confidence AgentFluent recommendation. Document selection rationale in the case study. Candidate improvements (will be narrowed at execution time): subagent delegation, cost-tier routing, skill-based task offloading. |
| #20 | Architect review of implementation approach | S | Open architect review: tag story with `needs-architect-review`, invoke architect agent. Architect posts `## Architect Review` comment on the story issue. Block implementation until review is posted. This session itself generates session data showing architect agent usage. |
| #21 | Implement the selected improvement | M-L | Implement on a feature branch per project conventions. The improvement targets a specific AgentFluent diagnostic. Scope depends on #19 outcome. Existing open issues (#1-#7) are natural candidates to work on during this phase to produce two-for-one session data. |
| #22 | Run post-experiment measurement | S | Run `agentfluent analyze --project classifier --since <phase1-start> --format json > social/baselines/phase1-post.json`. Run `agentfluent diff social/baselines/phase0-baseline.json social/baselines/phase1-post.json`. Save diff output. |
| #23 | Append Phase 1 results to case study | S | Document: what was tried, what the diff showed, what surprised us, what we'd do differently. Include raw numbers. |
| #24 | Draft Phase 1 blog post | M | Draft `social/posts/YYYY-MM-DD-agentfluent-first-experiment.md`. Narrative structure: hypothesis, method, results, reflection. Must be prettier-clean. |

### Phase 2+ (Sketch -- detail deferred)

**Goal:** Subsequent improvement cycles following the same measure-implement-measure-story pattern.

Each phase follows the same template:
1. Review current diagnostics and select next improvement target
2. Architect review
3. Implement
4. Measure with `agentfluent diff`
5. Append to case study, draft blog post

Phase 2+ stories will be created after Phase 1 completes and results are assessed. Potential targets (dependent on what Phase 1 reveals):

- Cost-tier routing (if still flagged after Phase 1)
- Skill-based task offloading for repetitive workflows
- MCP audit findings (if AgentFluent flags configured-but-unused servers)
- Quality signal reduction (if `USER_CORRECTION` or `FILE_REWORK` rates are high)

## 5. Stretch Scope

| Item | Rationale |
|------|-----------|
| Automated baseline capture script | A shell script that wraps `agentfluent analyze` + saves timestamped JSON. Only if manual capture becomes tedious. |
| Cross-phase trend visualization | A notebook that plots diagnostic signal trends across phases. Only if 3+ phases produce enough data points. |

## 6. Out of Scope / Deferred

| Item | Rationale |
|------|-----------|
| README excerpt (marketing Track A) | Can be derived from the case study later. Not needed until the case study has substance. |
| LinkedIn/Twitter posts | Derived from blog posts at publication time. Not a development artifact. |
| AgentFluent feature requests | File on the AgentFluent repo if discovered. Don't scope into this epic. |
| Blog post publication (transfer to website) | Each blog draft story specifies the transfer path but the actual publication is a manual step outside this epic's scope. |
| Existing open issues (#1-#7, #12) | Stay flat/unlabeled-by-epic. Phase stories note which existing issues are natural candidates to tackle during a phase for two-for-one session data. |

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| AgentFluent v0.6 ships late | Phase 0 blocked | Phase 0 prerequisite is explicit. No work is wasted -- the epic issue and marketing scaffolding can be set up in advance, but baseline capture waits for v0.6. |
| Too few sessions for pattern signals | AgentFluent diagnostics show `null` (as they do today with 2 sessions) | Organic session accumulation over 2-4 weeks. Each phase generates 3-6 sessions. After Phase 1, the project should have 8-12 sessions -- enough for signals to fire. |
| Improvement shows no measurable delta | Phase 1 experiment "fails" -- no signal reduction | Document the null result honestly. A well-documented null result is still good marketing content ("here's what we tried, here's why it didn't move the needle, here's what we learned"). |
| Marketing artifacts feel forced | Case study reads like a sales pitch rather than genuine engineering narrative | The value hierarchy (D001) puts quality of improvement first. If the improvement is real, the story writes itself. Review drafts for authenticity before publication. |
| Architect agent review adds overhead without value | Workflow slows down for ceremony | Architect review is scoped to implementation stories only (#20, #21). If the first review is low-value, reassess the gate for Phase 2+. |

## 8. Dependencies

```
[AgentFluent v0.6 ships] -----> [Phase 0: #15 baseline]
                                        |
                                  #15 --> #16 (case study scaffold uses baseline data)
                                  #15 --> #17 (blog post uses baseline data)
                                  #16 + #17 --> #18 (captures Phase 0 sessions)
                                        |
                                        v
                                 [Phase 1]
                                  #19 (select target from baseline)
                                  #19 --> #20 (architect review of chosen target)
                                  #20 --> #21 (implement)
                                  #21 --> #22 (post-measurement)
                                  #22 --> #23 (case study appendix)
                                  #22 --> #24 (blog post)
                                        |
                                        v
                                 [Phase 2+ -- created after Phase 1]
```

Phase 0 and Phase 1 are fully serial. Within each phase, the dependency chain is linear except where noted.

## 9. Architect Engagement Protocol

Every implementation story (#21 and equivalents in Phase 2+) requires an architect review gate:

1. **Before coding:** Tag the story issue with `needs-architect-review`.
2. **Invoke architect agent:** The developer opens a session with the architect agent to review the implementation approach.
3. **Architect posts review:** The architect agent posts a `## Architect Review` comment on the story issue with: recommended approach, risks, and any constraints.
4. **Unblock coding:** Remove the `needs-architect-review` label. Implementation proceeds.

This protocol serves dual purposes:
- Engineering value: catches design issues before implementation
- AgentFluent value: generates subagent session data that demonstrates architect agent usage patterns

## 10. Marketing Artifact Specifications

### Case Study (`social/case-study-agentfluent.md`)

A living Markdown document. Each phase appends a section. Structure:

```
# AgentFluent Case Study: Sportswear ESG News Classifier

## Introduction
[Project context, why we chose this project, what AgentFluent is]

## Baseline (Phase 0)
[AgentFluent findings, key metrics, identified opportunities]

## Experiment 1: [Title] (Phase 1)
[Hypothesis, method, results, reflection]

## Experiment N: [Title] (Phase N)
...

## Conclusions
[Cumulative impact, lessons learned]
```

### Blog Posts (`social/posts/YYYY-MM-DD-slug.md`)

Each post uses al-folio frontmatter:

```yaml
---
layout: post
title: "Post Title"
date: 2026-MM-DD 12:00:00-0800
description: "Short description for blog index card"
tags: ["agentfluent", "claude-code", "agent-quality"]
categories: ["external-services"]
featured: false
---
```

Voice: technical-but-accessible, matching the tone of the user's existing CodeFluent project page. Data-driven with specific numbers. Honest about failures.

**Transfer path to personal website:**
1. Copy `social/posts/<slug>.md` to `~/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_posts/<slug>.md`
2. Run `npx prettier . --check` in the website repo (fix any issues)
3. Commit on a branch, open PR, merge

**Acceptance criterion for all blog post stories:** Draft must pass `npx prettier --check` before the story is considered done.

### Baseline Snapshots (`social/baselines/`)

JSON output from `agentfluent analyze --format json`. Naming convention: `phase{N}-{pre|post}.json`. These are the raw data that the case study and blog posts reference.

## 11. Success Criteria

The epic is successful when:

1. **Baseline exists.** A v0.6 AgentFluent baseline JSON is saved and its key findings are documented in the case study.
2. **At least one experiment is measured.** `agentfluent diff` output exists showing a before/after comparison for one workflow change.
3. **Case study has substance.** At least two sections (Baseline + one Experiment) with real data, not placeholders.
4. **At least two blog post drafts exist.** Both prettier-clean and ready for transfer to the personal website.
5. **Architect agent appears in session data.** At least one session shows architect agent invocation, visible in `agentfluent analyze` output.
6. **pm agent appears in session data.** This epic's planning sessions themselves count.
7. **Quality of improvement is genuine.** The workflow change implemented in Phase 1 produces a real engineering benefit to the classifier project, not just session data for marketing.

## 12. Release Checklist

Phase 0:
- [x] AgentFluent v0.6 confirmed shipped
- [x] #15: Baseline JSON saved to `social/baselines/phase0-baseline.json`
- [ ] #16: Case study scaffold created with populated Baseline section
- [ ] #17: "Where we started" blog post drafted and prettier-clean
- [ ] #18: Phase 0 session data captured in case study

Phase 1:
- [ ] #19: Improvement target selected and documented
- [ ] #20: Architect review comment posted on implementation story
- [ ] #21: Implementation merged via PR (CI passing, squash merge)
- [ ] #22: Post-experiment JSON saved to `social/baselines/phase1-post.json`
- [ ] #22: `agentfluent diff` output saved and analyzed
- [ ] #23: Phase 1 case study section written with real data
- [ ] #24: Phase 1 blog post drafted and prettier-clean
