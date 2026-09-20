# Agent Orchestrator

The ESG News Classifier includes a custom-built agent orchestrator that automates daily operations, reducing manual maintenance to near-zero while ensuring data quality and system health.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Step Failure Contract](#step-failure-contract)
  - [Health Verdict Contract](#health-verdict-contract)
- [Workflows](#workflows)
  - [Daily Labeling](#daily-labeling)
  - [Drift Monitoring](#drift-monitoring)
  - [Website Export](#website-export)
  - [Run Audit](#run-audit)
  - [Model Training](#model-training)
- [CLI Usage](#cli-usage)
- [Scheduling with Cron](#scheduling-with-cron)
- [Configuration](#configuration)
- [Notifications](#notifications)
- [State Management](#state-management)
- [LLM Intelligence](#llm-intelligence)
  - [Knowledge Base & Heuristic Learning](#knowledge-base--heuristic-learning)
- [Troubleshooting](#troubleshooting)

## Overview

### Why a Custom Agent?

Off-the-shelf workflow tools (Airflow, Prefect, Dagster) are powerful but add significant operational overhead for a single-developer project:

| Approach | Pros | Cons |
|----------|------|------|
| Airflow/Prefect | Rich features, DAG visualization | Heavy infrastructure, maintenance overhead |
| Simple cron scripts | Easy setup | No state, no retries, scattered logic |
| **Custom agent** | Lightweight, tailored features, LLM intelligence | Development effort |

The custom agent provides:

- **Lightweight**: Single Python module (~1,500 LOC), no external services
- **YAML state management**: Human-readable workflow state and history
- **LLM intelligence**: Claude analyzes labeling results for quality assurance
- **Unified notifications**: Email (Resend) + webhooks (Slack/Discord)
- **Checkpointing**: Workflows can pause for human review and resume

### Benefits

| Metric | Without Agent | With Agent |
|--------|---------------|------------|
| Daily maintenance time | ~30 min/day | ~0 min/day |
| Model drift detection | Manual checks | Automated with alerts |
| Labeling quality assurance | Periodic manual review | Daily LLM analysis |
| Website updates | Manual export/push | Automated commit/push |
| Visibility into operations | Check logs manually | Email summaries daily |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Orchestrator                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   Workflow   │   │    State     │   │   Runner     │         │
│  │   Registry   │   │   Manager    │   │   (Scripts)  │         │
│  └──────────────┘   └──────────────┘   └──────────────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   Workflow Engine                     │        │
│  │    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │        │
│  │    │Step1│──│Step2│──│Step3│──│Step4│──│StepN│      │        │
│  │    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘      │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Notification │ │     LLM      │ │   Reports    │             │
│  │   Manager    │ │   Analyzer   │ │   (JSON)     │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│         │                │                                       │
│         ▼                ▼                                       │
│     Email/Slack    Claude                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step Failure Contract

A step handler has **two** ways to fail, and both mark the step FAILED and the workflow FAILED.
Anything else is recorded as success.

| Channel | Effect on the loop | Use when |
|---------|--------------------|----------|
| `raise` | Aborts the run; remaining steps stay `pending` | The step cannot meaningfully continue |
| `return StepFailure(error, context)` | **Continues** to the next step | Later steps still need to run — e.g. a terminal step that aggregates failures and notifies |

Sketch, in the shape of `daily_labeling.run_labeling` (see that function for the real one — it
returns eight context keys, and dropping any of them is the footgun in point 3):

```python
from src.agent.workflows.base import StepFailure

def run_labeling(workflow, context):
    result = run_label_articles(batch_size=pending, dry_run=context.get("dry_run", False))
    if not result.success:
        return StepFailure(
            error=f"labeling exited {result.exit_code}: {result.stderr[-2000:]}",
            context={"labeling_success": False, "labeling_exit_code": result.exit_code, ...},
        )
    return {"labeling_success": True, ...}
```

Four properties adopters depend on:

1. **Returning a plain dict always means success.** A handler that catches its own error and
   returns `{"..._success": False}` is recorded COMPLETED and the run archives as
   `status: completed, error: null`. That is the defect `StepFailure` exists to remove.
2. **`error` is always on the step**, as `step.error`. `context` is recorded in *both* places —
   merged into the workflow context, and stored as that step's `step.result`. Per-step attribution
   is why: the workflow context is a flat namespace several steps write the same keys into
   (`website_export` has three writers of `"error"` alone), so the archive could not otherwise say
   which step contributed what. A step that fails by *raising* has no payload and leaves
   `step.result` as `None`.
3. **`StepFailure.context` fully replaces the dict return**, so it must carry every key downstream
   steps read — dropping one is silent. A returned failure does not halt the loop, so those steps
   will run: **guarding on context flags is a requirement on the downstream step, not something
   the framework does for you, and today's workflows do not all have such guards.**
4. **A FAILED step beats a pause.** If any step has failed, the run is FAILED and archived even
   when a later step pauses. Otherwise a failure followed by an approval step would report `paused`
   with a null error and exit 0 — a failed run that reads as waiting for a human.

`Workflow._finalize()` is the single place step outcomes become a workflow status — `run()`,
`resume()` and both of their `except` branches call it, so the two paths cannot drift.
`WorkflowState.error` echoes the failed steps' own errors (each truncated in the summary; the full
text stays on the step).

**What this replaces.** Three workflows hand-rolled a terminal raising step to get the same effect:
`website_export.send_error_notification`, `daily_labeling.send_notification` (narrower — it fires
only when *every* notification channel failed, so it says nothing about whether labeling worked),
and `drift_monitoring.fail_on_unknown_verdict` (added in #71). Only `model_training` has none.
`drift_monitoring`'s was migrated in #74 onto the shared gate below; the remaining two are
#77's and #78's work.

### Health Verdict Contract

A **health verdict** answers *"what did this check find?"*. It is not a workflow lifecycle state:
`WorkflowStatus` says where a workflow *is*, a `HealthVerdict` says what one of its checks *saw*.
`unknown` is deliberately **not** a `WorkflowStatus` member — that would write archives carrying
`status: unknown` that every existing reader would need migrating for (epic #72, D008, D010).

**The four values, spelled as they appear on the wire** (`src/agent/health.py`). These strings are
the contract, not just the Python enum — a shell script or any non-Python consumer binds to the same
four spellings rather than inventing a fifth:

| Value | Meaning | Fails the run? |
|-------|---------|----------------|
| `healthy` | The check ran and found nothing wrong | no |
| `degraded` | The check ran and found a real problem — actionable | no; it is reported through the workflow's own alerting |
| `unknown` | The check produced **no verdict**: it raised, had nothing to compare, or returned a result with no evidence | **yes** |
| `skipped` | The check was deliberately not run, for a reason that must be stated | no — but it never counts toward "all healthy" either |

**Three rules, and each exists because it was once violated:**

1. **Absence is `unknown`, never `healthy`.** `as_verdict()` holds this rule and every entry point
   applies it: `verdict_of(context, key)` for a single read, and `summarize()` / `unresolved()` on
   each value they are handed. Absent, `None`, an unrecognised string, or junk all coerce to
   `unknown`, logged. Reading a health judgement off a key a failed check never wrote is how a
   broken drift check reported "all classifiers healthy" (#71).

   **This is why the aggregates normalize rather than compare by identity.** Verdicts are stored as
   `.value` strings (rule 3), so the natural call — read them out of a context and aggregate — hands
   `summarize()` strings. Identity comparison matched none of them: every subject counted as
   *checked*, and `summarize({"fp": "healthy", "ep": "skipped"})` reported healthy with the skipped
   check counted as a pass. Equality alone would not have fixed it either, since an unrecognised
   spelling would still match no branch and read as checked. Coercion is what makes an unknown
   spelling — including a future one from a non-Python consumer — fail safe.
2. **"All healthy" means at least one check ran.** `summarize(verdicts)` computes
   `all_checked_healthy` as *"at least one check ran and every check that ran passed"* — never
   `all(...)` over the non-skipped checks, which Python reports as `True` for an empty sequence. A
   run in which everything was skipped would otherwise report healthy having checked nothing (#95).
3. **Store `verdict.value`, never the enum member, and keep anything context-bound primitive.**
   The state file is written with `yaml.dump` but read with `yaml.safe_load`. A member — or a
   dataclass — serializes fine and then fails to load, landing in `StateManager._load`'s bare
   `except`, **which resets every workflow's state**. `HealthSummary` is a `TypedDict` for exactly
   this reason: a real `dict` at runtime, so it round-trips.

**Failing the run on an unresolved verdict** is `fail_on_unresolved_verdicts()` in
`workflows/base.py`. It builds a terminal step handler that returns `StepFailure` when any verdict
is unresolved, so the run is FAILED by the same `_finalize()` path as any other step failure:

```python
# Built once at module level, so the symbol stays importable by tests.
fail_on_unknown_verdict = fail_on_unresolved_verdicts(
    ("fp", "ep"),
    describe=_unmonitored_message,
)

StepDefinition(
    name="fail_on_unknown_verdict",
    description="Fail the workflow if any check produced no verdict",
    handler=fail_on_unknown_verdict,
    # Deliberately NOT skip_on_dry_run: a dry run should still surface that a
    # check could not tell us anything.
)
```

**Register it after every step that writes a verdict** — in practice, last. The gate reads verdicts
out of the context, so a subject whose check has not run yet reads as `unknown` and fails the run
spuriously. That is the entire ordering constraint.

It is **not** about losing the report from the archive. Returning `StepFailure` does not halt the
loop (Step Failure Contract, property 3 above), so steps after the gate still run and are still
recorded — measured, with the gate registered second of four: the alert still sent, the report step
still `COMPLETED` with its result in the archive. The raising gate this replaced *did* have that
property, and an earlier draft of this section carried its rationale over unchanged.

**The signal→verdict mapping stays in the workflow, not in `health.py`.** Each scheduled check has
its own contract — drift reads exit codes (`src/mlops/exit_codes.py`), labeling will read rates,
the export will read a written file — and pushing those semantics into the shared module would make
every workflow depend on drift's. `health.py` owns the vocabulary and the subject-agnostic
operations (`as_verdict`, `verdict_of`, `summarize`, `unresolved`) and nothing else. It is also an **import
leaf**: `workflows/base.py` imports it, never the reverse. The reverse is a cycle **today**, not a
future hazard — `workflows/__init__` eagerly imports every workflow module and `drift_monitoring`
already imports `health`, so `health` importing `workflows.base` breaks `import src.agent.health`
immediately (D010).

### Module Structure

```
src/agent/
├── __init__.py
├── __main__.py          # CLI entry point
├── config.py            # Configuration from environment
├── state.py             # YAML-based state management
├── archive.py           # Reader over the run archive (import leaf, writes nothing)
├── health.py            # HealthVerdict vocabulary + summarize/verdict_of (import leaf)
├── runner.py            # Script execution with retries
├── notifications.py     # Email (Resend) + webhook notifications
├── llm.py               # Claude integration
└── workflows/
    ├── __init__.py
    ├── base.py              # Workflow base class, registry, StepFailure, verdict gate
    ├── daily_labeling.py    # Daily labeling workflow
    ├── drift_monitoring.py  # Classifier drift detection
    ├── website_export.py    # Jekyll feed export
    ├── run_audit.py         # Liveness: a scheduled workflow that stopped running
    └── model_training.py    # Model training with notebook pause + heuristic updates

src/experiment_log/
├── __init__.py
├── __main__.py          # CLI entry point (uv run python -m src.experiment_log)
├── models.py            # Pydantic models (ExperimentEntry, Decision, KnowledgeBase, etc.)
├── store.py             # YAML-based storage with CRUD, index, knowledge base
├── tracker.py           # Experiment lifecycle orchestration
├── reflection.py        # LLM-based reflection via Claude API
└── cli.py               # Replay-time CLI (heuristics, record-decision, update-heuristic)

src/workflow_learning/
├── ...
├── analyzer.py          # Claude analysis (steps + decision extraction)
├── skill_generator.py   # SKILL.md generation with KB lookup directives
└── experiment_bridge.py # Bridge: decisions → experiment log + knowledge base
```

## Workflows

### Daily Labeling

**Schedule**: 6:30 AM daily
**Purpose**: Process pending articles through ML classifiers and LLM labeling, then generate quality reports.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `check_collection_status` | Query database for collection runs and pending articles from last 24h |
| 2. `run_labeling` | Run labeling pipeline on all pending articles |
| 3. `check_labeling_quality` | Calculate error rates, detect anomalies |
| 4. `run_llm_analysis` | Claude analyzes results for potential errors |
| 5. `generate_report` | Generate JSON summary report |
| 6. `save_report` | Save report to `reports/daily_labeling/` |
| 7. `send_notification` | Email summary via Resend |

**Output Example** (email summary):
```
DAILY LABELING WORKFLOW SUMMARY
============================================================
Generated: 2026-01-19T14:30:36.741283+00:00

Collection (24h):
  Runs: 8
  Fetched: 14
  Scraped: 12

Labeling:
  Processed: 31
  Labeled: 12
  Skipped: 8
  False Positives: 9
  Failed: 2
  Cost: $0.4500

LLM Analysis:
  Status: Completed
  Potential Errors: 2
  Patterns Detected: 3
  Improvement Suggestions: 2
```

### Drift Monitoring

**Schedule**: 5:30 AM daily
**Purpose**: Check FP and EP classifiers for data drift before labeling runs.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `check_fp_drift` | Run drift detection for FP classifier; map its exit code to a `HealthVerdict` |
| 2. `check_ep_drift` | Same for EP -- **skipped by default** with a stated reason (`AGENT_EP_DRIFT_ENABLED`, see below) |
| 3. `evaluate_drift_results` | Determine if action is needed, from the verdicts |
| 4. `send_drift_alerts` | Alert on detected drift **and** on any check that produced no verdict |
| 5. `generate_drift_report` | Generate summary report |
| 6. `fail_on_unknown_verdict` | Fail the workflow if any verdict is not explicitly healthy/degraded/skipped |

**Health verdicts**: see the [Health Verdict Contract](#health-verdict-contract) for the shared
vocabulary and its rules. What is drift-specific is how a verdict is *reached*: the
`_VERDICT_BY_EXIT_CODE` table over `scripts/monitor_drift.py`'s exit codes, plus `unknown` for an
exit code outside that contract or a summary with no evidence behind it, and
`skipped` for EP. This is where the vocabulary was first needed — issue #71,
where a failed check reported "all classifiers healthy" — and #74 generalized it.

**Step 6 runs last on purpose**, for the reason the shared contract gives. It is no longer
hand-rolled here: #74 replaced the local implementation with `fail_on_unresolved_verdicts` from
`workflows/base.py`, which is the retirement D008 asked for. The step *name* is unchanged so the
archive key stays stable for #75/#76.

**EP is on hold.** `classifier_predictions` has never held an `ep` row, so the check compared
nothing and reported healthy. It is now gated on `AGENT_EP_DRIFT_ENABLED` (default `false`) and
reports `skipped` with the reason in `AGENT_EP_DRIFT_SKIP_REASON`.

**Alert Triggers**: **any** core metric (`probability`, `prediction`, `novelty_score`) drifts,
**or** the `brand_*` drift score exceeds the configured threshold (default: 0.1), **or** a check
produced no verdict.

That describes the **Evidently** path (`EVIDENTLY_ENABLED=true`). There, core drift is a *count*
test, not a threshold test -- one core metric drifting is enough -- and the reported `drift_score`
is `max(core_drift_score, brand_drift_score)`, with both components kept in `details`. It used to
be described as a single threshold rule while the reported score was `core_drift_score` alone, so
brand-only drift alerted with "score 0.0 exceeds 0.15" (issue #71).

On the **default** path (`EVIDENTLY_ENABLED=false`, which is also where `_setup_evidently` falls
back on `ImportError`), `_legacy_drift_check` assesses the same four signal groups, but **scores the core ones
differently, and deliberately** (#102, D017). Core drift is an *effect size* there --
`probability` and `novelty_score` by KS statistic, `prediction` by rate difference -- and the core
score is the largest of those magnitudes against the threshold, not a count of significant tests.
The two paths are **not** expected to produce the same number. `brand_*` columns are assessed per
column by chi-square and enter as their own component, OR'd into the verdict, exactly as on the
Evidently path -- and the reported `drift_score` is `max(core, brand)`, so it can be either
quantity. `details["drift_score_source"]` records which one it was.

Until #102 the legacy path compared `probability` and `prediction` and nothing else, while
`columns_missing_from_reference` stayed empty because `novelty_score` and the `brand_*` columns
*are* in the reference -- they were simply never looked at, so a total distributional shift in
`novelty_score` reported as healthy.

Both paths now record **`columns_assessed`** (what produced a reading) and
**`columns_skipped`** (a column that was present but could not be assessed, each entry carrying
its own reason). **A skipped column is never counted as evidence of no drift** -- it is left out
of `drift_scores` and out of the brand denominator rather than scored as "not drifted".

⚠ **The two paths do not populate `columns_skipped` for the same reasons, so it is not a
like-for-like field between them.** The legacy path records a column it declined to test, for
whatever reason it declined, and names the cause. The Evidently path sees only a returned scalar,
so it records what it observed — a value it could not read, or one that is not finite — and never
why; an entry reading `p-value is not finite` on that path may have been produced by any of the
causes the legacy path distinguishes. A `brand_*` column absent from the reference is still
dropped there silently, before `columns_to_check` is built, so it appears in neither
`columns_skipped` nor `metrics_unreadable`.

⚠ **Which of the two paths a given deployment actually takes is not recorded anywhere in the
report.** It depends on the environment, and the `ImportError` fallback can change it without
announcing it — so a `drift_score` cannot be interpreted without knowing which instrument produced
it. Tracked as #136.

⚠ **These two fields do not reach the summary or the run archive yet.**
`print_summary_json` emits a fixed key set that excludes `details`, and `REQUIRED_SUMMARY_FIELDS`
is that same set. So **within-window** partial coverage -- a column present but unusable on a given
day -- does not reach the workflow. Carrying it across is #104. (They *are* visible in the
`--output` JSON, and in the webhook alert, which renders every `details` key.)

**Minimum sample size** (`DRIFT_MIN_SAMPLE_SIZE`, default 30 -- issue #94). Enough rows to compute
a statistic is not enough rows for it to mean anything. Two scopes, two outcomes:

- a whole **frame** below the floor -- the reference and the current window are checked
  independently -- makes the verdict `indeterminate`, not healthy;
- a single **column** below it, counted after its NaN are dropped, is skipped and recorded, and
  the check still returns a verdict from whatever else it could measure.

⚠ **A rare `brand_*` column is NOT filtered out for being rare**, and this is deliberate.
`brand_li-ning` carries 3 positives in the shipped 934-row reference; against a quiet week it
returns p=1.0 with a smallest expected cell of 0.22, which reads like a column that cannot detect
anything and only enlarges the denominator of `brand_drift_score`. It is power-*asymmetric*, not
powerless: it cannot evidence a decrease, and three positives in a 73-row window take it to
p=0.0011. A rare brand suddenly appearing is the drift this project most wants to hear about, so a
minimum-expected-cell floor would remove the dead reading and that detection together (D017).

### Website Export

**Schedule**: 7:00 AM daily
**Purpose**: Export labeled articles to Jekyll site and push to GitHub Pages.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `export_feeds` | Generate JSON and Atom feeds |
| 2. `validate_export` | Validate JSON/XML syntax |
| 3. `commit_and_push` | Git commit and push to website repo |
| 4. `send_error_notification` | Email only if export failed |

**Output Files**:
- `_data/esg_news.json` - JSON feed for Jekyll data files
- `assets/feeds/esg_news.atom` - Atom/RSS feed

### Run Audit

**Schedule**: Every 6 hours
**Purpose**: Two questions over one data source — has a scheduled workflow *stopped running*,
and is one *running and failing every time*.

Liveness is the only such net in the system, and it covers a gap none of the other mechanisms
can. #73 makes a failed step fail its run and #74 makes an unresolved check fail its run — both
read a run that *happened*. A job that never runs writes no archive for either of them to read,
so its silence is indistinguishable from success. This workflow is the external observer that
compares expected against actual.

The failure-streak check (#75) closes the other half: a job that keeps running and keeps
failing archives every run correctly as failed, and nothing reads archives, so it hides behind
a green light just the same.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `check_liveness` | Compare each expected workflow's newest archived run against its cadence |
| 2. `send_stale_alerts` | Notify on stalls, an unreadable archive, or an unusable config |
| 3. `check_failure_streaks` | Count each watched workflow's trailing failed runs (#75) |
| 4. `send_failure_escalations` | Escalate a workflow that has failed N runs in a row |
| 5. `generate_audit_report` | Summarize from recorded verdicts rather than inferred health |
| 6. `fail_on_unknown_verdict` | Fail the run if any audited subject produced no verdict |

**Alerting exactly once** (D014, corrected by D015). Each pass records a per-workflow
high-water mark — the newest run id it has alerted for — in this workflow's own archived
context, so no new store is introduced. A pass that *suppresses* a repeat must carry that mark
forward, or the ledger forgets and the next pass re-alerts. The mark advances only when the
alert reached a configured channel or when no channel is configured at all; a configured
channel that was attempted and failed leaves the mark alone so the next pass retries.

A workflow that is both stalled and failing is escalated once, by the stall alert — its streak
is still recorded, but the second page is suppressed.

**Verdict mapping** (D012). A workflow with no run inside its interval — including one that
has never run at all — is `degraded`, not `unknown`. `unknown` is reserved for *the archive
could not be read*, because that is the only case where the auditor genuinely cannot tell.
Mapping a stall to `unknown` would trip the terminal gate and fail the auditor's own run at
the exact moment it succeeded at detecting a dead job, leaving a FAILED status where an alert
naming the workflow should be.

**Configuration** lives in `AgentSettings`. *Which* jobs are watched is deliberately not
env-overridable — that is a governance decision, not a deployment knob. The escalation
threshold is, because it is an operator's sensitivity knob rather than a statement about
which work is watched.

| Setting | Env | Meaning |
|---------|-----|---------|
| `audit_expected_interval_hours` | — | Workflow → how often it is expected to run |
| `audit_grace_hours` | — | Added to the interval before a run is called stale |
| `audit_skipped_workflows` | — | Workflow → why it is deliberately not audited |
| `consecutive_failure_threshold_raw` | `AGENT_CONSECUTIVE_FAILURE_THRESHOLD` | Failed runs in a row before escalating (default `2`) |

The threshold is held as a raw string and parsed by `config.parse_failure_threshold` at the
point of use, so a value below 1 or a non-number fails `check_failure_streaks` with a
`StepFailure` naming the variable. It is deliberately **not** validated in `__post_init__`:
`agent_settings` is built at module scope, so raising there took down every agent workflow over
a knob one step reads (D015.1). Nothing guards the other direction — a large N is a detector
that never fires and says nothing about it.

The failure-streak check watches `audit_expected_interval_hours` only, which is a narrower set
than the workflows that archive runs. **Which workflows should be watched is an open question**
(D016): two reasons have been given for excluding `model_training` and both were false, so the
reason is recorded as unsettled rather than restated a third time. Whether `run_audit` should
count its own past failures is also unsettled.

A test asserts every workflow `setup_cron.sh` schedules appears in one of the two dicts, so a
newly scheduled job cannot end up with no detector — this epic's defect reproduced inside the
auditor's own configuration.

**How late an alert can be.** A stall is reported no earlier than `interval + grace` after the
last run, and no later than one audit period after that; the bound is
`interval + grace + audit period`. Grace is sized for start-time jitter — `run_id` is stamped
when a run starts, so a long job does not age its own archive. All three terms matter, so
tuning one alone will not give you a particular latency. The cost of a sub-daily cadence is
repetition: a job that stays dead is re-alerted each window.

**What it does not cover.** The auditor is a cron job on the same host as the workflows it
audits. One workflow dying while its siblings keep running is the case it exists for and is
detected while it is happening. A total host or cron outage takes the auditor down too, so
the gap is reported on recovery rather than at the time. A process cannot observe its own
absence; closing that needs an off-host dead-man's-switch, which is out of scope. For the
same reason the auditor is in its own skip set.

**Related one-shot sweep.** `uv run python scripts/audit_archive.py` answers a different
question — which archived runs reported `completed` while carrying a failure signal. It is
deliberately a script with no schedule: its value is a single retroactive pass.

It exits `0` (nothing found), `1` (findings listed on stdout), or `2` (the sweep did not run).
`2` covers both an unreadable archive and an invalid command line, because argparse uses `2`
for usage errors and this script does not reclaim it. What the code guarantees is the
distinction that matters to a caller: `0` means *checked every record it could read, and found
nothing*, never *could not look at the archive at all*. An individual record that cannot be read
or parsed is logged and skipped rather than raised — one bad file must not blind the reader to
the rest — so a `WARNING` on stderr is the only trace of it. Splitting `2` into distinct causes is a repo-wide convention question — `src/mlops/`
carries its own `0/1/2` contract — filed as #127 to settle alongside #80, which consumes exit
codes at cron boundaries.

### Model Training

**Schedule**: Manual (triggered when new training data is available)
**Purpose**: Automate model retraining with human-in-the-loop notebook review.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `export_training_data` | Export FP and EP training datasets |
| 2. `check_data_quality` | Validate record counts, class balance; create experiment log entries |
| 3. `notify_and_pause` | Send email, pause for notebook execution |
| 4. *User runs notebooks* | Manual: fp1 → fp2 → fp3 (or ep1 → ep2 → ep3) |
| 5. `compare_models` | Compare new model metrics to production; record observations in experiment log |
| 6. `prompt_promotion` | Pause for promotion approval |
| 7. `promote_model` | Update model registry |
| 8. `trigger_deployment` | Trigger GitHub Actions deployment |
| 9. `finalize_experiments` | Record reward, LLM reflection, update heuristic counters, and complete experiment log entries |

**Resume Command**:
```bash
uv run python -m src.agent continue model_training
```

## CLI Usage

The agent provides a command-line interface for managing workflows:

```bash
# List available workflows
uv run python -m src.agent list

# Run a workflow
uv run python -m src.agent run daily_labeling

# Run with dry-run (no side effects)
uv run python -m src.agent run daily_labeling --dry-run

# Resume a paused workflow
uv run python -m src.agent continue model_training

# Check workflow status
uv run python -m src.agent status

# View workflow history
uv run python -m src.agent history

# View history for specific workflow
uv run python -m src.agent history daily_labeling
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `list` | Show all registered workflows |
| `run <workflow>` | Execute a workflow |
| `run <workflow> --dry-run` | Execute without side effects |
| `continue <workflow>` | Resume a paused workflow |
| `status` | Show current workflow status |
| `history` | Show completed workflow runs |
| `history <workflow>` | Show history for specific workflow |

## Scheduling with Cron

### Install Cron Jobs

```bash
# Install all agent cron jobs
./scripts/setup_cron.sh install-agent

# Check cron status
./scripts/setup_cron.sh status

# Remove cron jobs
./scripts/setup_cron.sh remove-agent
```

### Default Schedule

| Time | Workflow | Purpose |
|------|----------|---------|
| 5:30 AM | `drift_monitoring` | Check classifier health before labeling |
| 6:30 AM | `daily_labeling` | Process new articles |
| 7:00 AM | `website_export` | Update live feed |
| Every 6 hours | `run_audit` | Detect a workflow that stopped running, or that keeps failing |

### Cron Configuration

The cron jobs use wrapper scripts that handle environment setup:

```bash
# Example cron entry (from setup_cron.sh)
30 5 * * * /path/to/scripts/cron_agent.sh drift_monitoring >> /path/to/logs/agent/cron_drift_monitoring_$(date +\%Y\%m\%d).log 2>&1
```

The wrapper script:
1. Sets up the correct PATH for `uv`
2. Changes to the project directory
3. Activates the virtual environment
4. Runs the workflow
5. Captures output to dated log files

## Configuration

All agent settings are configured via environment variables:

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_STATE_DIR` | Directory for state files | `~/.esg-agent` |
| `AGENT_DRY_RUN` | Enable dry-run mode globally | `false` |
| `AGENT_MAX_RETRIES` | Max retries for failed steps | `3` |
| `AGENT_RETRY_DELAY` | Delay between retries (seconds) | `5` |
| `AGENT_DEFAULT_TIMEOUT` | Script timeout (seconds) | `600` |
| `AGENT_CONSECUTIVE_FAILURE_THRESHOLD` | Failed runs in a row before `run_audit` escalates | `2` |

### LLM Analysis Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_LLM_ANALYSIS` | Enable Claude analysis of labeling | `true` |
| `AGENT_LLM_ERROR_THRESHOLD` | Error rate threshold to trigger analysis | `0.0` (always run) |
| `AGENT_LLM_MODEL` | Model for LLM analysis | `claude-haiku-4-5-20251001` |
| `ANTHROPIC_API_KEY` | API key for Claude | Required |

### Notification Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_EMAIL_ENABLED` | Enable email notifications | `false` |
| `AGENT_EMAIL_RECIPIENT` | Email recipient address | Required if enabled |
| `AGENT_EMAIL_SENDER` | Email sender address | Required if enabled |
| `RESEND_API_KEY` | Resend.com API key | Required for email |

### Path Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_PROJECT_ROOT` | Project root directory | Auto-detected |
| `AGENT_LOGS_DIR` | Log directory (relative to project) | `logs/agent` |
| `AGENT_WEBSITE_REPO_PATH` | Jekyll website repository path | None |

### Example .env Configuration

```bash
# Agent Core
AGENT_STATE_DIR=/home/user/.esg-agent
AGENT_MAX_RETRIES=3

# LLM Analysis
AGENT_LLM_ANALYSIS=true
AGENT_LLM_ERROR_THRESHOLD=0.0
ANTHROPIC_API_KEY=sk-ant-...

# Email Notifications (via Resend)
AGENT_EMAIL_ENABLED=true
AGENT_EMAIL_RECIPIENT=your@email.com
AGENT_EMAIL_SENDER=esg-agent@yourdomain.com
RESEND_API_KEY=re_...

# Website Export
AGENT_WEBSITE_REPO_PATH=/path/to/your-github-pages-repo
```

## Notifications

The agent supports multiple notification channels:

### Email (Resend)

[Resend](https://resend.com) is the recommended email provider (3,000 emails/month free):

```bash
# Enable email
AGENT_EMAIL_ENABLED=true
AGENT_EMAIL_RECIPIENT=your@email.com
AGENT_EMAIL_SENDER=esg-agent@yourdomain.com
RESEND_API_KEY=re_...
```

### Webhooks (Slack/Discord)

For drift alerts and failures:

```bash
# Slack webhook
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...

# Discord webhook
ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Notification Types

| Type | Trigger | Channels |
|------|---------|----------|
| Labeling Summary | Daily after labeling | Email |
| Drift Alert | When drift exceeds threshold | Email + Webhook |
| Check Failed | When a scheduled check produces no verdict (severity `error`) | Email + Webhook |
| Export Error | When website export fails | Email |
| Training Ready | When data export completes | Email |
| Promotion Complete | After model promotion | Email |

## State Management

Workflow state is stored in YAML format for human readability:

### State Directory Structure

```
~/.esg-agent/
├── state.yaml           # Current workflow state
└── history/
    ├── daily_labeling_20260119_143001.yaml
    ├── drift_monitoring_20260119_133001.yaml
    └── website_export_20260119_150002.yaml
```

### State File Format

```yaml
workflow: daily_labeling
status: completed
started_at: '2026-01-19T14:30:01.234567+00:00'
completed_at: '2026-01-19T14:30:37.891234+00:00'
current_step: send_notification
dry_run: false
context:
  collection_runs_24h: 8
  articles_fetched_24h: 14
  articles_pending: 31
  labeling_success: true
  # ... more context data
steps_completed:
  - check_collection_status
  - run_labeling
  - check_labeling_quality
  - run_llm_analysis
  - generate_report
  - save_report
  - send_notification
```

### Checkpointing

Workflows that require human intervention can pause and resume:

```python
# In workflow step
workflow.state.pause_workflow(
    workflow.name,
    reason="Waiting for manual notebook execution",
)
```

Resume with:
```bash
uv run python -m src.agent continue model_training
```

## LLM Intelligence

The agent integrates Claude for two purposes:

1. **Labeling analysis** (daily_labeling workflow): Detects potential errors in article labeling
2. **Experiment reflection** (model_training workflow): Analyzes completed training experiments to assess hypothesis outcomes, identify surprises, and suggest next steps

### Labeling Analysis

Claude analyzes labeling results and detects potential issues:

### What It Analyzes

- **Recent labeling samples**: Last 24 hours of labeled, skipped, and false positive articles
- **Labeling statistics**: Error rates, false positive rates, processing counts
- **Article content**: Title, content, brand mentions, assigned labels

### What It Detects

1. **Potential Labeling Errors**: Articles that may have been incorrectly classified
2. **Pattern Detection**: Systematic issues affecting multiple articles
3. **False Positive Analysis**: Common causes of false positives by brand
4. **Improvement Suggestions**: Actionable recommendations for prompts, pre-filtering, or labeling criteria

### Example LLM Analysis Output

```json
{
  "overall_assessment": "The labeling system shows good precision with 0% false positives, but appears overly conservative...",
  "potential_errors": [
    {
      "article_id": "1c129532-b27d-467f-a012-86851c551e14",
      "issue": "Nike's pickleball signing could have ESG implications but was skipped",
      "severity": "medium",
      "recommendation": "Review ESG criteria to include strategic partnerships"
    }
  ],
  "patterns_detected": [
    {
      "pattern": "Financial/analyst coverage articles consistently skipped",
      "affected_count": "6-7 articles",
      "recommendation": "Develop criteria to identify governance insights in financial analysis"
    }
  ],
  "improvement_suggestions": [
    {
      "area": "labeling",
      "suggestion": "Develop more nuanced ESG detection criteria...",
      "priority": "high",
      "effort": "medium"
    }
  ]
}
```

### Labeling Analysis Configuration

```bash
# Always run LLM analysis (default)
AGENT_LLM_ERROR_THRESHOLD=0.0

# Only run when error rate exceeds 10%
AGENT_LLM_ERROR_THRESHOLD=0.10

# Disable LLM analysis
AGENT_LLM_ANALYSIS=false
```

### Experiment Reflection

After a model training run completes, the `finalize_experiments` step optionally calls Claude to reflect on the experiment. This is gated by the same `AGENT_LLM_ANALYSIS` setting.

The reflector receives the full experiment context (production state, training data stats, observed metrics, promotion outcome) and returns:

- **Summary**: What happened and why
- **Hypothesis result**: confirmed / refuted / inconclusive
- **Surprises**: Unexpected findings
- **Next steps**: Recommended actions
- **Confidence**: Assessment reliability (low / medium / high)

Reflections are stored in the experiment YAML files at `data/experiments/{classifier}/exp_*.yaml` and are available for future knowledge base extraction.

### Knowledge Base & Heuristic Learning

The experiment log includes a per-classifier knowledge base (`data/experiments/knowledge/{classifier}_knowledge.yaml`) that accumulates patterns and heuristics from both workflow recordings and training experiments.

**How knowledge flows in:**

1. **From workflow recordings** (analyze time): When the user records a notebook walkthrough and narrates decisions (e.g., "NER features aren't helping, F2 dropped, so I'm removing them"), the workflow learning analyzer extracts structured `ExtractedDecision` objects. The `experiment_bridge` module then:
   - Saves each decision as a `Decision` entry in the experiment log
   - Seeds `Pattern` entries when a decision has a trigger + observed outcome
   - Seeds `Heuristic` entries when a decision has a trigger + chosen option + reasoning
   - New entries start at `confidence="low"` and upgrade with successful applications

2. **From training experiments** (finalize time): The `finalize_experiments` step loads all decisions recorded during an experiment and updates matching heuristic counters (`times_applied`, `times_successful`). After enough successful applications (3+ with 70%+ success rate), heuristic confidence upgrades to `"high"`.

**How knowledge flows out:**

1. **In generated SKILL.md files**: Checkpoint steps (those with both `success_criteria` and `on_failure`) include a KB lookup directive:
   ```bash
   uv run python -m src.experiment_log heuristics --classifier fp
   ```
   This allows the agent to consult past decisions before making choices during skill replay.

2. **Via CLI during replay**: The agent (or user) can record decisions and update heuristics:
   ```bash
   # Look up heuristics before a decision point
   uv run python -m src.experiment_log heuristics --classifier fp

   # Record a decision made during replay
   uv run python -m src.experiment_log record-decision \
     --classifier fp --experiment-id fp_20260226_143000 \
     --phase feature_engineering \
     --trigger "NER features not contributing" \
     --chosen "remove_ner" \
     --reasoning "F2 unchanged with NER"

   # Update heuristic outcome after observing result
   uv run python -m src.experiment_log update-heuristic \
     --classifier fp \
     --trigger "NER features not contributing" \
     --success true
   ```

**Architecture:**

```
Analyze time (workflow recording → knowledge):
  Screenpipe recording
    → RecordingAnalyzer (extended prompt → steps + decisions)
    → SkillGenerator (SKILL.md with KB lookup directives)
    → ExperimentBridge (decisions → Decision entries + Pattern/Heuristic seeds)

Replay time (knowledge → agent decisions):
  Agent follows SKILL.md
    → At checkpoints: consult heuristics CLI
    → Records decisions: record-decision CLI
    → finalize_experiments: updates heuristic counters from decisions
```

## Troubleshooting

### Common Issues

#### Cron Job Not Running

**Symptom**: Workflow doesn't execute at scheduled time

**Check**:
```bash
# Verify cron is installed
./scripts/setup_cron.sh status

# Check cron logs
cat /var/log/syslog | grep CRON

# Check workflow logs
cat logs/agent/cron_daily_labeling_$(date +%Y%m%d).log
```

**Common causes**:
- `uv` not in cron's PATH (fixed in `runner.py` with `_find_uv_path()`)
- Environment variables not loaded
- Incorrect file permissions

#### Email Not Sending

**Check**:
```bash
# Verify settings
echo $AGENT_EMAIL_ENABLED
echo $RESEND_API_KEY

# Test manually
uv run python -c "
from src.agent.notifications import NotificationManager, Notification, NotificationType
notifier = NotificationManager()
result = notifier.send(Notification(
    notification_type=NotificationType.WORKFLOW_COMPLETE,
    subject='Test',
    message='Test message',
    severity='info'
))
print(result)
"
```

#### Workflow Stuck in Paused State

**Check**:
```bash
# View current status
uv run python -m src.agent status

# Check state file
cat ~/.esg-agent/state.yaml
```

**Resolution**:
```bash
# Resume the workflow
uv run python -m src.agent continue <workflow_name>

# Or reset state (use with caution)
rm ~/.esg-agent/state.yaml
```

#### LLM Analysis Failing

**Check**:
```bash
# Verify API key
echo $ANTHROPIC_API_KEY

# Check logs for error details
grep -i "llm\|claude\|anthropic" logs/agent/daily_labeling.log
```

**Common causes**:
- Missing or invalid `ANTHROPIC_API_KEY`
- Rate limiting (reduce analysis frequency)
- No samples to analyze (check database for recent articles)

### Viewing Logs

```bash
# Latest workflow log
cat logs/agent/daily_labeling.log

# Dated cron log
cat logs/agent/cron_daily_labeling_$(date +%Y%m%d).log

# Follow logs in real-time
tail -f logs/agent/daily_labeling.log
```

### Workflow History

```bash
# View recent workflow runs
uv run python -m src.agent history

# View archived state files
ls -la ~/.esg-agent/history/
```

### Manual Workflow Execution

For debugging, run workflows manually with verbose output:

```bash
# Run with logging enabled
PYTHONUNBUFFERED=1 uv run python -m src.agent run daily_labeling 2>&1 | tee manual_run.log

# Dry run to test without side effects
uv run python -m src.agent run daily_labeling --dry-run
```
