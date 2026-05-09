# AgentFluent Case Study: Sportswear ESG News Classifier

## Introduction

This case study documents an experiment in using AgentFluent — a diagnostic tool for Claude Code session data — to drive measured workflow improvements on a real engineering project. The project is the **Sportswear ESG News Classifier**: a multi-label text classification system that ingests news articles about 50 sportswear and outdoor brands and tags them across Environmental, Social, Governance, and Digital Transformation categories. It is an end-to-end pipeline (NewsData.io and GDELT collection, full-text scraping, paragraph chunking with OpenAI embeddings, Claude Sonnet labeling with cross-encoder reranking on the evidence step, FastAPI prediction services for two production classifiers), and it has a real backlog of engineering work — model evaluation, MLOps, feed exports to a public website. The work is not contrived for marketing.

The choice of project matters. AgentFluent is most useful when it can analyze sessions that reflect how someone actually works: messy, multi-day, with detours and reworks. The classifier project meets that bar. Every Claude Code session run against this repository produces a JSONL of tool calls, agent invocations, model usage, and quality signals. AgentFluent ingests those JSONLs and surfaces patterns: which tools dominate, which subagents retry the same call repeatedly, where reviewers caught issues, which configured MCP servers never get used, where cost concentrates by model.

The dual-purpose framing is the point. The engineering work and the workflow-quality work are two views of the same activity. Implement an improvement, capture a session, run AgentFluent, measure the delta, document the result. One activity, two outputs: a better classifier and a real data point about agent-assisted development. This document is the place where the second output accumulates.

The case study itself follows a fixed shape. Each phase appends a new section. The Baseline section below captures the "before" snapshot that all subsequent experiments will be compared against. Subsequent experiment sections record a hypothesis, a method, the measured before/after delta, and an honest reflection — including null results when an improvement does not move the needle.

## Baseline (Phase 0)

The baseline snapshot was captured on **2026-05-09** using AgentFluent **v0.6.0** in full-history mode (`agentfluent analyze --project classifier --diagnostics`, no date filter). Full-history mode is deliberate: subsequent measurements use `--since` to isolate post-baseline sessions, so the baseline serves as the reference point for all `agentfluent diff` runs. Raw output lives in `social/baselines/phase0-baseline-cli.txt` (human-readable) and `social/baselines/phase0-baseline.json` (structured), both gitignored for security reasons (the JSON contains absolute local file paths and tool arguments). The numbers below are extracted from those files.

### Sample size and cold-start caveat

**Sessions analyzed: 2.** That is a very small sample. AgentFluent's pattern signals (`USER_CORRECTION` rate, `FILE_REWORK` rate) are designed to fire when there is enough data to be statistically meaningful, and two sessions is below that threshold. The trace-level diagnostics below — `retry_loop`, `tool_error_sequence`, `mcp_unused_server`, `reviewer_caught` — fire on individual sequences rather than rates and are reported here as-is. Pattern-level signals are expected to populate as session count grows over Phase 1 and beyond. This is a cold-start baseline; it is honest about being one.

### Token usage and cost

| Metric | Value |
| ----- | ----- |
| Total tokens | 22,381,743 |
| Cache read tokens | 21,038,512 |
| Cache creation tokens | 1,136,716 |
| Output tokens | 205,397 |
| Input tokens | 1,118 |
| **Cache efficiency** | **94.9%** |
| API calls | 174 |
| **Total cost (API rate)** | **$22.76** |

API rate is the pay-per-token equivalent. The real cost on a fixed-price subscription plan is independent of usage; the dollar figure is reported because it is the comparable unit across plan types and is what `agentfluent diff` will compare across phases.

### Cost by model

| Model | Origin | Tokens | Cost |
| ----- | ----- | ----- | ----- |
| claude-opus-4-7 | parent | 14,921,794 | $14.44 |
| claude-opus-4-6 | subagent | 3,041,883 | $4.57 |
| claude-opus-4-6 | parent | 4,418,066 | $3.75 |

All sessions ran on Opus, with no routing to lower-cost tiers (Sonnet, Haiku) for trivial reads, greps, or boilerplate edits. Whether that is the right policy is one of the open questions a future experiment could answer.

### Tool concentration

| Tool | Calls | % of total |
| ----- | ----- | ----- |
| Bash | 115 | **61.8%** |
| Read | 25 | 13.4% |
| Edit | 9 | 4.8% |
| Agent | 8 | 4.3% |
| TaskUpdate | 8 | 4.3% |
| Write | 8 | 4.3% |
| Grep | 4 | 2.2% |
| TaskCreate | 4 | 2.2% |
| ToolSearch | 3 | 1.6% |
| Glob | 2 | 1.1% |
| **Total** | **186** | |

Ten unique tools, 186 total calls, **61.8% concentrated on Bash**. Heavy Bash use often indicates that the agent is shelling out for tasks dedicated tools handle better — `cat`/`head` instead of `Read`, `sed`/`awk` instead of `Edit`, `grep` instead of `Grep`. This is a documented anti-pattern in the project's CLAUDE.md, but the metric suggests it still happens often enough to dominate the tool mix.

### Subagent activity

| Agent type | Invocations | Tokens | Avg tokens/call | Total duration |
| ----- | ----- | ----- | ----- | ----- |
| pm | 4 | 209,915 | 52,478 | 5,841.9s |
| architect | 4 | 169,086 | 42,271 | 506.6s |
| **Total** | **8** | | | |

**Agent token share: 1.7% of total.** Both `pm` and `architect` agents are configured (in `~/.claude/agents/`) and were invoked in this baseline window — a notable change from the project's pre-AgentFluent state, where these agents were configured but never used. Eight invocations across two sessions is a healthy starting cadence, but the imbalance between `pm` duration (5,842s) and `architect` duration (507s) suggests `pm` is doing substantially more open-ended work per call. The agent-token percentage being only 1.7% means the parent thread still owns most of the heavy lifting.

### Diagnostic signals

| Agent | Type | Severity | Detail |
| ----- | ----- | ----- | ----- |
| pm | retry_loop | warning | retried `Grep` 3 times |
| pm | retry_loop | warning | retried `Read` 3 times |
| pm | tool_error_sequence | warning | 2 consecutive tool errors |
| architect | retry_loop | warning | retried `Glob` 4 times |
| architect | reviewer_caught | info | review surfaced 4 finding-keywords |
| architect | reviewer_caught | info | review surfaced 3 finding-keywords |
| architect | reviewer_caught | info | review surfaced 2 finding-keywords |
| architect | reviewer_caught | info | review surfaced 2 finding-keywords |
| (global) | mcp_unused_server | info | `playwright` configured in `.claude.json` with 0 tool calls across 8 sessions |

Three warning-level signals (all on subagents, all retry-related) and six info-level signals.

### Identified improvement opportunities

The baseline surfaces several candidates for future experiments. None is yet committed to — the Phase 1 selection happens in story #19 and will be documented in the next section.

1. **Subagent prompt hardening.** Both `pm` and `architect` produced retry-loop warnings (Grep ×3, Read ×3, Glob ×4) and `pm` had two consecutive tool errors. AgentFluent's recommendation is explicit: the prompts in `~/.claude/agents/{pm,architect}.md` mention error handling but do not give the agent specific stop conditions or alternative-tool fallbacks for repeated failures. A prompt revision is a small, well-scoped change with a measurable target (retry-loop count → 0 in the next analyze run).

2. **Reviewer-caught findings on architect.** Four `reviewer_caught` info signals on `architect`, with finding-keyword counts of 2, 2, 3, and 4. The recommendation is to investigate whether those findings are actionable and whether the parent prompt should require follow-through on architect feedback. This is more open-ended than (1) and may need its own diagnostic dive before becoming an experiment.

3. **Unused MCP server.** `playwright` is configured in `~/.claude.json` but had 0 tool calls across all 8 analyzed sessions. AgentFluent flags this as cost/config drift. The fix is trivial (remove from `mcpServers` or set `disabled: true`); the value is a small reduction in startup tax and one less moving part in the configuration.

4. **Heavy Bash concentration (61.8%).** Bash dominates tool use by a large margin. Some Bash use is appropriate (running tests, git operations), but 61.8% is high enough to suggest the agent is reaching for shell when a dedicated tool would be cleaner. The remediation path here is not a single change — it is either a CLAUDE.md tightening (already attempted: the file currently calls out "do not use cat/head/tail/sed/awk") or a structural intervention like a permissions/skill change. This is the largest opportunity by call volume but also the one with the least obvious single-step fix.

These four are listed in roughly increasing order of scope. The next section will record which one (or which combination) Phase 1 targets, and why.

## Experiment 1: [Title TBD] (Phase 1)

_To be populated after Phase 1 target selection (story #19) and post-experiment measurement (story #22)._

## Conclusions

_To be populated after at least one experiment cycle is complete._
