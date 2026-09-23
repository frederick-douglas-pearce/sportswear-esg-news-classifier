# Changelog

This document tracks significant changes to the ESG News Classifier pipeline, including new features, policy updates, and migrations.

## 2026

### 2026-09-23: The test suite can no longer write into the real agent run archive

`AgentSettings.history_dir` is the directory `run_audit` reads to decide whether a scheduled job
is still alive. Keeping test runs out of it was left to each agent test module (#124).

**What changed:**

- **`tests/conftest.py` gives every test its own empty agent state dir** (`_isolate_agent_state`,
  autouse), and clears the in-memory state of the `state.state_manager` singleton. It is scoped per test rather than per session, so one test's archive cannot make a
  workflow look alive to another test.
- **A `pytest_configure` hook sets `AGENT_STATE_DIR` before `src.agent` is imported**, so the
  import-time `agent_settings` and `state.state_manager` singletons are never built against the
  real `~/.esg-agent`.
- **`tests/test_agent_state_isolation.py` pins both**, using only synthetic workflow names.
- The per-file `history_dir` fixtures are kept. Their docstrings now say which ones are assert
  handles and which are redundant.

D020 in `.claude/specs/decisions.md` records the mechanism and the per-test scope.

### 2026-09-19: A NaN p-value stops counting as evidence of no drift on the Evidently path

`float('nan')` is an instance of `float`, so it passed the Evidently path's unreadable-metric
guard; `nan < threshold` is then `False`, and the column was appended to `columns_assessed` and
counted toward `total_core`/`total_brand`. A column nobody could compute a statistic for was
counted as evidence that it did not drift — the same p=1.0 coercion the comment above that guard
says was removed, arriving by a different route. Evidently 0.7.18 returns `nan` from
`ValueDrift(method="chisquare")` for a column constant at the same value in both frames (#103) —
an input `_categorical_p_value` refuses outright on the legacy path.

**What changed:**

- **A non-finite metric value is skipped, not scored.** `math.isfinite` — not `np.isfinite`, which
  raises on non-numerics and returns `np.bool_` — in its own guard after the readability check, so
  the ordering is structural rather than a positional property of an `or` chain.
- **It is recorded in `details["columns_skipped"]` as `"p-value is not finite"`** — a description
  of what *this* path observed, naming no cause, because a returned scalar cannot tell you which
  cause produced it. **This is not the legacy path's reason for the same input**: that path rejects
  a column constant in both frames at its category check, writing `"one category in both frames"`,
  and reaches its own finite check only for a table it actually ran chi-square on. The two paths
  still do not share a reason vocabulary, and a reader of `columns_skipped` cannot tell which path
  wrote an entry (#136).
- **`details["metrics_unreadable"]` stays narrow** — only values that could not be read — and can
  now be a strict subset of `columns_skipped`. `columns_skipped` is the wider of the two, but not a
  complete inventory: a core column absent from the reference is in
  `columns_missing_from_reference`, and a `brand_*` column absent from the reference is in neither,
  because it never reaches `columns_to_check` (#105).
- **The `total_core == 0` verdict stops naming a cause that did not occur.** Skipping a non-finite
  metric makes that branch reachable on an input where every metric *was* read, so its message is
  now "no core drift metrics were **usable**" rather than "could not be **read**". That string is
  the only report-derived prose that reaches the operator email, the run archive and the CI
  summary — `columns_skipped` reaches none of them until #104 — so the old wording would have
  pointed the reader at a renamed metric when the cause was a constant column.
- **The brand denominator no longer absorbs a skipped column.** `brand_drift_score` was diluted by
  every NaN brand metric, biasing brand drift toward healthy.

**What this does NOT fix, and is not claimed to.** The two drift paths still disagree on the input
that motivates this change. `probability` and `novelty_score` are configured for `ks`, and KS
returns a **finite 1.0** for a column constant in both frames — so the Evidently path still assesses
it, `total_core` is still non-zero, and the verdict is still healthy where `_legacy_drift_check`
calls the same frames indeterminate. Catching that means inspecting the input frames rather than the
returned scalar, which is the "assessed set ≠ offered set ⇒ no verdict" cross-check reserved for
**#105**. Also filed there: `total_brand` can now reach 0 with brand columns offered by a second and
far likelier route than the pre-existing unreadable-metric one, making `brand_drift_score` a
fabricated 0.0.

**This supersedes the closing note of the 2026-09-18 entry below**, which said an unreadable metric
was the only skip reason that can occur on the Evidently path. There are two now, and neither
record is a complete inventory of what went unassessed.

This fix moves no live number today: on the live 7-day window as of 2026-09-19 all 18 offered
columns are still assessed and `columns_skipped` is empty. Verified by running both code arms
against the live database seconds apart — the reports are identical but for their timestamps. That
is a property of the current window against the current reference, not of either alone, and one
quiet week would change it.

### 2026-09-18: The legacy drift path assesses every signal group, and an unmeasured column stops reading as health

`_legacy_drift_check` compared `probability` and `prediction` and nothing else, while
`columns_missing_from_reference` stayed empty because `novelty_score` and the `brand_*` columns
*are* in the reference — they were simply never looked at. A total distributional shift in
`novelty_score` reported as healthy.

**What changed:**

- **All four signal groups are assessed on that path.** `novelty_score` joins the core score by KS
  statistic; `brand_*` columns are assessed per column by chi-square and enter as their own
  component, OR'd into the verdict.
- **Core stays an effect size and brand does not redefine it.** The reported `drift_score` is
  `max(core, brand)` — which is what stops brand-only drift alerting as "score 0.0 exceeds 0.15" —
  so the number can be either quantity, and `details["drift_score_source"]` records which.
  **Adding a term to a max can only raise it, so scores from before this change are not
  level-comparable with scores after it.**
- **No column that could not be measured is counted as evidence of no drift.** All three core
  columns go through one guard; a `brand_*` column the reference cannot answer for, or that the
  chi-square declines to test, is recorded in `details["columns_skipped"]` with its own reason
  rather than scored. `details["columns_assessed"]` records what did produce a reading.
- **Chi-square counts are aligned by label.** A bool-versus-int mismatch indexed positionally, and
  since `value_counts()` sorts by count descending it read the most-frequent category rather than
  the one asked for — returning "no drift" on a total flip.
- **A minimum sample size, with two scopes** (`#94`, `DRIFT_MIN_SAMPLE_SIZE`, default 30). A whole
  frame below the floor — reference and current window checked independently — makes the verdict
  `indeterminate` rather than healthy. A single column below it, counted after its NaN are
  dropped, is skipped and recorded, and the check still returns a verdict.
- **A rare brand column is not filtered out for being rare.** `brand_li-ning` (3 positives in the
  shipped 934-row reference) returns p=1.0 against a quiet week and looks like dead weight in the
  denominator, but it is power-*asymmetric* rather than powerless: three positives in a 73-row
  window take it to p=0.0011. A minimum-expected-cell floor would discard that detection along
  with the dead reading, so none is applied.

Both paths now report `columns_assessed` and `columns_skipped`; neither reaches the machine-readable
summary yet (#104). The two fields are not like-for-like across the paths — on the Evidently path
the only skip reason that can occur is an unreadable metric, and a `brand_*` column absent from the
reference is still dropped there silently. Which of the two paths produced a verdict is still
unrecorded (#136).

### 2026-09-16: A workflow that runs and fails every time is escalated

#73 fails a run with a failed step, #74 fails a run with an unresolved check, and #76 detects a
job that has stopped running. None of them watched a job that keeps running and keeps failing:
every run is correctly archived as failed, and nothing reads archives, so it hides behind a
green light just the same. `run_succeeded()` shipped with #76 and had no production caller;
this is its first.

**What changed:**

- **`archive.consecutive_failures()`** — the peer of `latest_run_per_workflow`: the same
  group-by-workflow reduction over `iter_runs`, reducing to the length of the trailing
  not-succeeded run. Count only; the threshold, the alert and the once-only bookkeeping are
  policy and live in `run_audit`. An unreadable archive raises rather than returning an empty
  result, because "no consecutive failures anywhere" is the silent all-clear this epic is about.
- **Two steps in `run_audit`**, mirroring its existing check/alert split:
  `check_failure_streaks` computes and `send_failure_escalations` alerts under
  `skip_on_dry_run`. Liveness asks whether a job ran; the streak asks whether it kept
  succeeding.
- **"Exactly once" is a carried-forward high-water mark** in the auditor's own archived
  context — reuse of a write that already happens, not a new store. The half that is easy to
  miss: a pass that *suppresses* must carry the mark forward, or the ledger forgets and the
  third pass re-alerts. That defect is invisible to a two-pass test, so the regression test
  spans three.
- **Every exit carries the ledger**, including the `StepFailure`, whose `context` replaces the
  result rather than merging with one. Without that, one transient bad pass re-armed every
  streak in the system.
- **Delivery gates the mark.** Every notifier swallows its exception and returns `False`, so a
  dead host or a rejected key is silent; the mark is left unadvanced and the next pass retries.
  A console-only result — the no-channel-configured default — advances it instead, since
  re-escalating to a console nobody reads on every pass forever is a busy-loop, not a signal.
- **The report tells the truth on both of its surfaces.** A live workflow failing every run
  left `generate_audit_report` rendering "No action needed" *and* archiving
  `all_checked_healthy: true` — this epic's defect committed by the report. Both now carry the
  streak result. A workflow that is stalled *and* failing is paged once, by the stall alert.
- **`AGENT_CONSECUTIVE_FAILURE_THRESHOLD`** (default 2) is parsed at the point of use, so a bad
  value fails the one step that reads it. Validating it in `AgentSettings.__post_init__` raised
  from module scope and stopped every agent workflow — including the auditor meant to notice —
  over a knob one step reads.

Design gates: **D014**, corrected in four places by **D015** after code review.

### 2026-09-14: The run archive is read — a workflow that stops running is now detected

`StateManager._archive_workflow` has always written each terminal run's full state to
`~/.esg-agent/history/`. Nothing read it back. Two failure modes were sitting in data that
was already on disk: a run archived as `completed` while its own record carried a failure
signal, and — the one with no detector anywhere in the system — a workflow that stopped
producing runs at all. #73, #74 and #75 all read a run that *happened*; a job that never
runs writes nothing for them to read, so its silence is indistinguishable from success.

**What changed:**

- **`src/agent/archive.py`** — a reader over the archive, writing nothing and holding no
  state. `iter_runs()` yields runs oldest-first (ordering is contract: #75 needs a
  time-ordered per-workflow sequence), tolerating a malformed record rather than letting one
  bad file blind it, and raising rather than returning empty when the directory itself
  cannot be read. `run_succeeded()` is the shared classifier #75 is to be built on.
- **Extraction is separated from policy.** `failure_signals()` returns *kinded* evidence and
  is deliberately broad, because the one-shot sweep that consumes it is read by a human.
  `run_succeeded()` is a narrow policy over the disqualifying kinds, because #75's automatic
  alerting is to be built on it: it must not read ordinary non-failure context — `alerts_sent: false`,
  `alerts_skipped: true`, `reason: nothing_to_report` — as failure, and it keys on the
  current contract (a failed run, a failed step, a step error, an unresolved verdict) rather
  than on the pre-#73/#74 `<name>_success: false` form, which belongs to archaeology.
- **`src/agent/workflows/run_audit.py`** — the scheduled liveness check, installed by
  `./scripts/setup_cron.sh install-agent`. It reuses #74's `HealthVerdict` rather than
  inventing a third spelling of "could not tell". It runs several times a day, not once:
  the auditor can only report a stall at the moments cron runs it, so its own period is part
  of the detection bound, which is `interval + audit_grace_hours + audit period`. Grace is
  sized for start-time jitter; the audit period is the other half, and both have to be set
  together to get a given latency.
- **A future-dated archive no longer silences a workflow permanently.** A newest run dated
  ahead of now — a clock that moved, a record that did not come from a run — gives a
  negative age, which passes every freshness comparison there is. Left alone, that workflow
  could never be reported stale again: this epic's own failure mode inside the detector
  written for it. It gets its own branch and reports `degraded`, because clamping the age to
  zero would produce the same silence.
- **The auditor refuses a cadence config it cannot audit honestly.** A config where no
  workflow has an expected interval, or where a workflow is both audited and skipped, fails
  the run and alerts instead of archiving `completed` having established nothing.
- **A stalled workflow is `degraded`, not `unknown`** (D012). Mapping it to `unknown` would
  trip `fail_on_unresolved_verdicts` and fail the auditor's own run at the moment it
  *succeeded* at detecting a dead job — making a correct detection indistinguishable from
  the auditor malfunctioning, and leaving a failed status where an alert naming the workflow
  should be. `unknown` is reserved for the case where the archive cannot be read at all.
- **`scripts/audit_archive.py`** — the one-shot retroactive sweep for runs that reported
  success over an embedded failure. Deliberately a script with no schedule: its value is a
  single pass over existing history, and once failures propagate correctly at the source a
  recurring version would mostly re-report the same records. **It lists instances and never
  a rate** — a proportion would need a corpus, and the directory is not one; a list has no
  denominator, and a spurious line is one a reader can dismiss. `--kind` takes only the
  listed signal kinds, so a typo cannot filter everything out and then report a clean
  archive.
- **Two guards turn silent drift into a red check.** Round-trip tests build real
  `WorkflowState` objects, serialize them through `to_dict`, and assert the reader's
  classification — covering `context` as well as `status`, `steps` and `error`, since
  `context` is where #73's and #74's signals live — so a field renamed in `state.py` cannot
  quietly degrade the reader to "sees no signal". And a test asserts every workflow
  `setup_cron.sh` schedules is either in the cadence config or in an explicit skip set, so a
  newly scheduled job cannot end up with no detector.

**Known limits, filed rather than claimed closed:** test-harness archive isolation (#124), and
`history_dir` creating the directory on read, which makes a deleted archive read as an empty
one — it alerts either way, but names the wrong cause (#125).

**What it does not cover, because "liveness check" invites the wrong assumption.** The
auditor is itself a cron job on the same host as the workflows it audits. It detects one
workflow stopping while its siblings keep running — the case it exists for — but during a
total host or cron outage it is down too, and reports the gap on recovery rather than at the
time. A process cannot observe its own absence; closing that gap needs an off-host
dead-man's-switch, which is out of scope. For the same reason the auditor is in its own skip
set. The limitation is stated in the module docstring rather than engineered around.

### 2026-09-11: `backup_db.sh` reports "could not check Docker" distinctly from "container is not running"

`check_container()` ran `docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"`
and printed *"Container '…' is not running"* plus `docker compose up -d postgres` for four
different states: the daemon down, the caller outside the `docker` group, `docker` absent
from `PATH`, and the container genuinely stopped. The remediation is correct only in the
last. Docker's error text still reached the terminal on stderr, but unlabelled and not
attributable to this check — capturing it is what the pipeline prevented. D006 recorded
this construct as "stated, not fixed here" and deferred it to #93.

**What changed:**

- **The pipeline is gone**, replaced by a captured query
  (`listing=$(docker ps --format '{{.Names}}' 2>&1) || status=$?`). That is what allows the
  failing command's own output to be shown under a label, and it removes two further
  hazards in the same construct: `grep` read the container name as a *regex*, so
  `esg.news_db` matched `esgxnews_db`; and under `pipefail` a `grep -q` that matched early
  could `SIGPIPE` `docker ps` and invert the verdict — a hazard #89 introduced here, which
  D006 measured as real but unreachable at this scale.
- **An exit-code contract**, in the script header and in
  [docs/DATABASE.md](DATABASE.md#exit-codes): `2` = a needed query failed, so the state was
  never established (`unknown`); `3` = Docker answered and the container is absent
  (`degraded`). `1` stays generic — naming it for a specific cause would make "backup file
  not found" and "unknown command" report that cause.
- **No guess at the cause on the `2` path.** The script shows the failing command's output
  and offers no remediation, because `docker` exits `1` for both a stopped daemon and a
  permissions problem. An earlier draft of PR #92 printed "Is the Docker daemon running?"
  for every non-zero status, reproducing the misattribution one level down.
- **The `3` path shows the running-container listing** and hedges its hint, because "not in
  the running list" is also what a misconfigured `CONTAINER_NAME` or a different compose
  prefix looks like.
- **Whole-line matching is `[[ ]]` with the name quoted**, which matches it literally.
  Unquoted, a `CONTAINER_NAME` of `*` would glob and match any listing. `grep -qxF` was
  rejected because `-F` reads a newline in the *pattern* as alternative patterns; the
  chosen form is narrower there rather than immune.
- **`status` reports partial results instead of aborting, and covers both of its Docker
  calls.** The second was the quieter defect: `local db_size=$(docker exec … | tr -d ' ')`
  returned `local`'s status rather than the pipeline's and `2>/dev/null` discarded psql's
  reason, so a container that was up while Postgres refused connections printed an empty
  size **and exited 0**. `status` now keeps the on-disk facts, says the size could not be
  determined, and exits the specific code. The size is validated by *shape* rather than
  non-emptiness, because folding stderr into the capture can otherwise weld a warning onto
  the number and render it as a fact.
- **The failure branches now have tests.** Before this the branch had none, so the one
  verdict the script got right was the one nothing checked.

### 2026-09-09: The health verdict becomes a shared contract with a first-class escalation path

#71 introduced `HealthVerdict` (`healthy | degraded | unknown | skipped`) and wired it into
`drift_monitoring`. It was shared in name only: `src/agent/health.py` had exactly two importers —
the drift workflow and its test — and the step that turns an unresolved verdict into a failed run
was hand-rolled inside that workflow, which said so in its own docstring ("This is a bridge. Once
#74 gives verdicts a first-class escalation path in the base runner, it should be deleted"). Each
of the four remaining adopters (#77, #78, #79, and the shell-side #93) would otherwise have written
its own third state.

**What changed:**

- **`health.py` gains the subject-agnostic operations** — `verdict_of` (absent/`None`/unrecognised
  coerce to `unknown`), `summarize` (aggregate), `unresolved` (the gate's predicate). The
  signal→verdict *mapping* stays per-workflow: drift keeps `_VERDICT_BY_EXIT_CODE`, because a
  labeling or export check has nothing to do with drift's exit codes. D008 phrased this boundary as
  "enum only"; D010 revises the phrasing and keeps the invariant it was protecting.
- **`fail_on_unresolved_verdicts()` in `workflows/base.py`** builds the terminal gate any workflow
  can register. It returns `StepFailure` rather than raising, making it the first production adopter
  of #73's contract: the same FAILED outcome through the same `_finalize()`, plus the payload on
  `StepState.result` that #75/#76 will read, minus a traceback that describes nothing.
- **`drift_monitoring`'s bridge is retired** onto it. The step's registered **name** is unchanged,
  and the failure wording is preserved verbatim, so nothing an archive reader might match on moves.
- **`summarize` is non-vacuous by construction.** `all_checked_healthy` is "at least one check ran
  and every check that ran passed", never `all(...)` over the non-skipped checks — `True` for an
  empty sequence, which would report a run that checked nothing as healthy. That invariant already
  shipped inside drift; putting it in the shared helper is what stops the next adopter writing the
  vacuous form. #95 stays open for the run-level half.
- **`HealthSummary` is a `TypedDict`, not a dataclass**, and everything the gate writes to the
  context is a primitive. The state file is written with `yaml.dump` and read with
  `yaml.safe_load`, so a richer object serializes cleanly and then fails to load on the *next* run,
  landing in `StateManager._load`'s catch-all `except Exception` — which resets every workflow's
  state. The test for this exercises a real `StateManager` save→load round trip, because that is
  the path production actually takes; asserting on `yaml.safe_dump` instead would test a function
  the agent never calls. (An earlier draft justified it by claiming `safe_dump` would have *passed*
  on the object that breaks — measured, it raises `RepresenterError` on a dataclass, a `NamedTuple`
  and an enum member, so it would have caught them.)
- **The aggregate helpers normalize their input rather than comparing it by identity.** Verdicts
  are stored as `.value` strings, so the natural call — read them out of a context and aggregate —
  handed `summarize()` strings, which matched no `is` branch: every subject counted as *checked*,
  and a run whose only other check was skipped reported healthy. One shared `as_verdict()` now
  holds the coerce-unrecognised-to-`unknown` rule and `verdict_of`, `summarize` and `unresolved` all
  route through it, so an unrecognised spelling — including a future one from a non-Python
  consumer — fails safe instead of reading as a passing check.
- **The gate's payload is a `TypedDict`** (`UnresolvedVerdictReport`), for the same reason
  `HealthSummary` is one. Its shape had been described in prose, and the prose had drifted. Both
  paths return the whole shape, so the archive carries the same keys whether a run passed or failed.
- **Subjects are validated as `str` at construction; reason values are coerced at read time.** The
  asymmetry is deliberate — a subject comes from the workflow author and a wrong one is worth
  refusing outright, while a reason is whatever was in the context at runtime.

- **The contract is documented as a contract** — `docs/AGENT.md` gains a Health Verdict Contract
  section beside the Step Failure Contract, listing the four canonical *string* values so a
  non-Python consumer (#93) binds to the same spellings, and `health.py` finally appears in the
  module tree.

**Also fixed:** an archived run written before the vocabulary existed had no test proving it still
loads. It does, and it does not acquire a healthy reading it never earned — the free-text
"all classifiers healthy" in an old archive still resolves to `unknown`.

**Why `health.py` stays an import leaf:** `workflows/__init__` eagerly imports every workflow
module **and `drift_monitoring` already imports `health`**, so had the gate lived in `health.py`
(importing `workflows.base`), `import src.agent.health` would fail **today** — the existing adopter
alone closes the loop, with no second one required. A test pins the direction by parsing `health.py`
for any import out of the `agent` package, which catches a lazy in-function import too.

See [D010](../.claude/specs/decisions.md) for the full decision record.

### 2026-09-08: A step that reports its own failure now marks the workflow FAILED

`Workflow._execute_step()` marked a step FAILED only when its handler *raised*. A handler that
caught its own error and returned a dict was recorded COMPLETED, so `run()`'s "all steps completed"
test passed and the run archived `status: completed, error: null`.

The run archive under `~/.esg-agent/history/` holds runs of exactly that shape: archived
`status: completed` with `error: null`, carrying a `*_success: false` context key, and **no step
recorded FAILED** — which is the mechanism itself. They are concentrated in `drift_monitoring`,
whose instance `2f30ab2` (#71, the commit before this one) has already closed; the remainder are in
`daily_labeling` and `website_export`, the latter predating that workflow's own terminal raise.

**No counts are quoted here deliberately.** The archive is not a stable corpus to measure against:
the test suite writes into the same directory, under production workflow names as well as synthetic
ones, so any figure is stale on the next test run and is inflated by artifacts that look like
scheduled runs. Earlier drafts of this entry quoted such figures and were wrong three times. The
case for fixing this in the base runner does not rest on a rate in any event — it is forward-looking:
five stories are about to bind to this contract, and the alternative is a fourth hand-rolled
terminal raise.

**What changed:**

- **`StepFailure(error, context)`** — a handler can now signal failure through its return value.
  The base runner marks the step FAILED and the run FAILED. Raising still works unchanged.
- A returned failure **does not halt** the loop, matching what a failure dict does today. That is
  what an aggregate-then-notify terminal step depends on; downstream steps guard on context flags.
- **One `_finalize()`** turns step outcomes into a workflow verdict, called from `run()`,
  `resume()` and both `except` branches. `resume()` previously had no **non-exception** failure
  branch: it completed the run when every step had completed and did nothing otherwise, so a
  resumed run carrying a failed step stayed RUNNING and was never archived (the archive is written
  only by `complete_workflow`/`fail_workflow`). A resumed step that *raised* was always caught and
  failed by its `except`, so that gap was **latent, not live** — nothing on `main` could produce a
  FAILED step without an exception escaping to that handler. `StepFailure` is what would have made
  it reachable.
- **A FAILED step beats a pause.** `_finalize` fails and archives even when the workflow is PAUSED,
  if any step failed. Without this, a `StepFailure` followed by any pausing step reported `paused`
  with a null error, wrote no archive, and exited 0 — the same defect class this change removes,
  reachable as soon as a workflow with an approval step adopts the contract.
- **`WorkflowState.error` echoes the real step errors** instead of the bare "Not all steps
  completed", truncated per step in the summary; the full text stays in `step.error`. A run where
  one step returns a failure and a later one raises now reports both.
- **A step name missing from persisted state no longer kills the finalizer.** `resume()` loads
  state written by an earlier process, so a step added since that run started has no record;
  indexing it raised `KeyError` from inside `_finalize` — including from the `except` handler that
  calls it — which would have left the run RUNNING and unarchived.

**Not changed:** no member was added to `WorkflowStatus` — `unknown`/`check_failed` is a health
verdict, not a lifecycle state (#74). The hand-rolled terminal-raise workarounds are **not**
migrated here; tests demonstrate the shapes re-expressed via `StepFailure`, and the migration
belongs to #74/#77/#78.

**There are three such workarounds, not two:** `website_export.send_error_notification`,
`daily_labeling.send_notification`, and `drift_monitoring.fail_on_unknown_verdict` — the last added
by #71 in `2f30ab2`, the commit immediately before this one. Only `model_training` has none. (This
entry and D009 first said `drift_monitoring` had none, a sentence inherited from #73's body, which
was written before #71 landed and was not re-checked against the tree.)

Worth recording, because it changes what one of those workarounds is worth:
`daily_labeling.send_notification` raises only when *every* notification channel fails. It says
nothing about whether labeling worked — so a failed labeling run whose report was delivered archives
as `completed`. It is not a guard for this defect class, though #73's issue body treats it as one.

Decision record: `.claude/specs/decisions.md` D009. Issue #73.

### 2026-09-07: A failed drift check no longer reports "all classifiers healthy"

FP drift monitoring failed on 223 of 232 scheduled runs since 2026-01-17, and on 219 of those the
workflow simultaneously logged `No action needed - all classifiers healthy` and exited 0. (The
issue's 221/230/217 were measured on 2026-09-05; these are the same logs re-counted on
2026-09-07.)

It produced a real verdict at least once -- **2026-01-19**, drift score 0.1315 -- so
"never produced a valid result" was wrong; #71's own title says "worked once". A second run,
2026-01-23, completed without a failure line but recorded `fp_drift_score: 0.0`, which is exactly
the value both fabricating paths emit; the script's stdout was never archived, so **whether it
measured anything cannot be determined** and it is not counted as a verdict here. Of the nine runs
with no failure line, the other seven are June 2026 `uv` DNS failures that never reached Python
(the #51 class), which fail loudly and are not this defect.

**Why a failure looked like health.** Four defects compounded, each of which turned a missing
signal into a benign one:

- `data/reference/fp_reference.parquet` was written 2026-01-17, before `novelty_score` existed.
  `_evidently_drift_check` guarded the novelty *stats* block on `current_data` alone and then read
  `reference_data["novelty_score"]`, raising `KeyError: 'novelty_score'`. This can only have been
  the cause from **2026-01-25**, when `novelty_score` was added to `classifier_predictions`
  (commit `4c17395`). It cannot explain the six earlier failures: three of those logged other
  causes (`Evidently not installed`, a missing `uv`) and the remaining three (01-21, 01-22, 01-24)
  logged an empty message, so **their cause is not recoverable from the logs** -- that being
  defect 2 below. **Three** stats reads had that asymmetric shape,
  not one; `novelty_score` is simply the one that fired, because a reference sharing only
  `brand_*` columns would have died on `reference_prob_mean` first.
- The script printed `Error running drift analysis: {e}` to **stdout** and returned 1, while the
  workflow logged the command's **stderr** -- producing 220 log lines reading
  `drift check failed: ` with nothing after the colon, out of 224. The 4 non-empty ones name
  causes that never reached the analysis (`Evidently not installed`, a missing `uv`).
- Exit 1 meant *both* "drift detected" and "the analysis raised", and `ScriptResult.success` is
  `exit_code == 0`, so the workflow could not tell them apart and fell back to scraping the
  human-readable report. `evaluate_drift_results` then read
  `context.get("fp_drift_detected", False)` and the report step read
  `context.get("fp_healthy", not context.get("fp_drift_detected", False))` -- with both keys
  absent, `not False` is **True**.
- The EP check ran against a classifier with **zero predictions, ever**. Two empty frames compared
  to each other returned `drift_detected=False` and passed vacuously.

**The rule the fix is built on:** a missing value never resolves to healthy. Concretely:

- `DriftReport` gains a typed `indeterminate` field, set wherever nothing was measured -- on
  **both** the Evidently and the legacy paths, the latter being the one taken by default
  (`EVIDENTLY_ENABLED` defaults to false, and Evidently's absence silently falls back to it). A
  verdict that was never produced is now distinguishable from one that was produced and was
  clean.
- A three-value exit-code contract (`src/mlops/exit_codes.py`): `0` no drift, `1` drift detected,
  `2` indeterminate. Drift detected is non-retryable -- it is a result, and before this it was
  retried with exponential backoff before being reported.
- Errors go to stderr with a traceback; the workflow logs the **tail** of that stream, since a
  traceback's diagnosis is at its end (the lesson of #81, same runner).
- `HealthVerdict` (`healthy | degraded | unknown | skipped`) in `src/agent/health.py`. It is
  deliberately **not** a `WorkflowStatus` member: a health verdict is not a lifecycle state, and
  archives carrying `status: unknown` would need migrating.
- The workflow derives its verdict from the exit code, not from scraping stdout; `_parse_drift_output`
  is deleted. A run whose exit code claims a verdict but whose summary is missing or inconsistent
  is treated as `unknown` -- a health claim with no evidence behind it is not a health claim.
- EP is skipped explicitly behind `AGENT_EP_DRIFT_ENABLED` (default off) with a stated reason.
  A config gate rather than a row count: EP-on-hold is a governance decision and zero predictions
  is only its symptom.
- A terminal `fail_on_unknown_verdict` step marks the workflow FAILED when any verdict is not
  explicitly healthy/degraded/skipped -- including absent. It runs *last* so the summary is printed
  and the alert sent first. It is a bridge, to be removed once #74 gives verdicts a first-class
  escalation path (D008).
- The FP reference is regenerated over 90 days (934 rows, complete `novelty_score`, plus the brand
  columns the old file lacked).

Also: a column present in the current data but missing from the reference is now logged as *not
assessed* and recorded in the report details, so fixing the crash does not leave a partial
comparison silently reported as a whole one.

Four further instances of the same class were found by review *inside this fix* and closed here:

- **A missing reference dataset used to answer itself.** `check_drift` split the current window in
  half on `FileNotFoundError` and compared the halves -- which agree by construction, so it always
  read as "no drift", with nothing recording that there was no baseline. Now indeterminate;
  `--create-reference` is the way to establish a baseline. This is **latent**, not live: it needs a
  classifier that has predictions and lacks a reference, and today `fp` is the only one with
  predictions and its reference is tracked. It goes live when EP resumes.
- **`drift_score` did not match the detection that reported it.** `drift_detected` reads core drift
  *or* brand drift, but the report carried the core score alone, so brand-only drift alerted with
  "score 0.0 exceeds 0.15". Now `max(core, brand)`, with both kept in the details.
- **An unreadable Evidently metric counted as "no drift".** Coercing a missing value to `p=1.0`
  made it both non-drifting and a contributor to the metric count, so the "nothing was readable"
  guard could not fire. Now skipped and recorded.
- **The CI summary printed "✅ Healthy" for an unreadable report.** Every `jq` returns an empty
  string on a truncated or malformed file, which matched no branch and fell through to the healthy
  default. Reproduced before fixing. Also, `|| echo DRIFT_DETECTED=true` reported exit 2
  ("could not assess") as drift; replaced with a `case` on the real exit code.

And `alerts_sent` now means *delivered*: both notification helpers return one bool per channel, and
every channel could fail while the step still recorded success.

Follow-ups filed rather than folded in: #94 (minimum sample size), #95 (vacuously-healthy when
every check is skipped), #96 (a forgotten EP flag leaves EP dark), #97 (the reference window
overlaps the window it is compared against), #99 (an unguarded `--output` write turns a disk error
into "drift detected"), #100 (`_to_builtin` type gaps), #101 (nothing consumes the new
`CHECK_INDETERMINATE` CI output).

### 2026-09-06: A failed pg_dump no longer records a good backup

`scripts/backup_db.sh` ran `pg_dump | gzip > "$PATH"` under `set -e` with no `pipefail`. A
pipeline reports its *last* command's status, so a `pg_dump` that died mid-stream was masked by
`gzip` exiting 0: a truncated archive was written, "Backup created successfully" was printed,
rotation ran, and `list`/`status` reported the partial file as the latest good backup. The same
shape on the restore path printed "Restore completed successfully!" over partially-loaded data.

**The second half, which `pipefail` alone does not fix.** The `if [ $? -eq 0 ]` handlers were
unreachable -- by two different routes, and only one of them is a `set -e` abort. When `pg_dump`
died the pipeline reported `gzip`'s 0, so `$?` was read, was 0, and the *success* branch ran; when
`gzip` itself failed the pipeline was non-zero and `set -e` aborted before `$?` could be read. The
first is the case this fix is about. Either way the `else` -- including the `rm -f "$DAILY_PATH"`
cleanup -- was dead code, so adding `pipefail` on its own would only make the failure louder while
leaving the truncated file on disk. Each of the three pipelines is now the `if` **condition**,
which is exempt from `set -e`, so the cleanup branch actually runs.

Also: the pre-restore safety copy aborts the restore and removes its partial file rather than
proceeding; the `ls *.sql.gz | awk` listing is guarded with `|| true`, the only *top-level*
pipeline `pipefail` would newly abort; and `BACKUP_SIZE=$(du -h ... | cut -f1)` is guarded with
`|| BACKUP_SIZE="unknown"`. That last one is the subtle site: it is the only plain (non-`local`)
assignment whose *right-hand side is a pipeline* -- other plain assignments exist, but `pipefail`
has nothing to reach in them -- so under `pipefail` a `du` failure would abort inside the success
branch and invert the invariant this change exists to establish: a good archive on disk, a non-zero
exit, no rotation, and the cleanup unreachable in the `else`. The `local var=$(cmd | cmd)` sites in `rotate` and
`status` are unaffected: `local` returns its own exit status, so the pipeline's never reaches
`set -e`.

`check_container`'s `docker ps | grep -q` also changes behaviour under `pipefail` -- it is a change
of *value* rather than an abort, since there the pipeline's status is the branch condition. It is
checked, measured and filed as #93 rather than fixed here. The inversion is a *race*, not a size
threshold: it fires whenever `docker ps` is still writing when `grep -q` matches and stops reading.
Exceeding the ~64 KiB pipe buffer is merely the reliable way to force it. What makes it unreachable
here is that real `docker ps` emits its whole list -- 30 bytes, two running containers -- in one
buffered write and exits before `grep` can close the pipe. Note this is a hazard the change
*introduces*: on `main` that pipeline runs under plain `set -e`, where its status cannot flip the
branch. It is deferred rather than pre-existing, and rewriting it is a behaviour change none of
this issue's acceptance criteria ask for.

Instance 7 of the `silent-success` class (#72), and the only one whose outcome is data loss rather
than a missed alert. Regression tests in `tests/test_backup_db_script.py` stub `docker` and `du` on
`PATH` to force a mid-pipeline failure, asserting both non-zero exit **and** no leftover archive -- a test
asserting only the exit code passes against a fix that leaves the partial file. (#89)

### 2026-09-05: Stop the labeling retry from erasing a run's results

The `daily_labeling` report for 2026-09-05 read `0 processed / 0 labeled / 0 failed`
for a run that had labeled 5 articles, skipped 6, deduplicated 1 and failed 1.

**Mechanism.** One article (`e6ae677f`) failed with `Failed to parse LLM response`.
`label_articles.py` returned `0 if not stats.errors else 1`, so a single bad article
failed the whole script. The agent runner read exit 1 as a total failure and retried
the identical command — but labeling consumes the rows it selects, so the retry found
nothing pending, exited 0 in 2.25s, and its empty stdout replaced the real numbers.
`check_labeling_quality` then derived `error_rate` from those zeros, which made
`high_error_rate` unreachable on exactly the runs that had failures. The `labeling_runs`
row held `status='partial'`, `articles_processed=12` the whole time; nothing read it.

Present in **13 of the last 230 daily runs (~6%)**: 20260118, 20260123, 20260210,
20260211, 20260212, 20260225, 20260307, 20260417, 20260419, 20260420, 20260423,
20260617, 20260905.

**Changes (issue #81):**
- `scripts/label_articles.py` returns a distinct exit code (`2`) when the batch is
  spent — some articles reached a terminal status and none are left pending, so a
  retry can only erase the result. `src/labeling/exit_codes.py` holds the contract;
  the runner accepts code 2 as final via `non_retryable_exit_codes`. Exit `1` still
  means "a retry still has articles to work through" and keeps its retries (#51).
  The decision reads `articles_processed` (terminal, spent) against
  `articles_left_pending` (raised before any status update, still retryable) — the
  two are tracked separately because they have different bases.
- `labeling_runs` gained `articles_labeled`, `articles_skipped`,
  `articles_false_positive`, `articles_failed`, `articles_deduplicated` and
  `articles_left_pending` (migration `007_labeling_run_outcomes.sql`), so a run row
  describes itself. `check_labeling_quality`, `generate_report` and the LLM analysis
  read those columns instead of scraping stdout, and report which source they used.
  A dry run writes no run row, so stdout remains the fallback.
- Only rows with `completed_at` set are summed. A row still `running` was written by
  a process killed mid-flight (the 1800s subprocess timeout), and its zeros are the
  absence of a result rather than a result. The window is bounded at both ends so a
  concurrently started labeling run is not attributed to the workflow.
- Quality metrics carry `partial_failure`, which fires on a run that did work *and*
  errored even when the error rate is under the 10% threshold (1/12 was), and
  `metrics_degraded`, which fires when a non-dry run produced no usable run row.
  Both now reach the emailed notification, not just the console summary.
- The runner logs the **tail** of stderr on failure, not the first 500 characters.
  Tracebacks are at the end; the head captured only the startup banner, so the
  original diagnosis had to be reproduced by hand.
- LLM pricing moved to one table in `src/labeling/config.py`. Spend came from the
  database ($1/$5 per MTok) while the FP savings estimate came from stdout ($3/$15),
  so a report could claim it saved more than it spent. Model-aware pricing is #54.

The parse failure that triggered this is #82; replacing prose-JSON parsing with
structured output is #83. Historical `labeling_runs` rows keep `0` in the new
columns — the breakdown was never persisted and cannot be reconstructed.

**Migration:** `psql $DATABASE_URL -f migrations/007_labeling_run_outcomes.sql`

### 2026-09-05: Recover LLM responses that quote the article verbatim

Article `e6ae677f` failed labeling with `Failed to parse LLM response`. The model had
copied a quoted passage straight out of the article into an evidence excerpt:

```
"I think the store expansion will slow down," Morningstar analyst David Swartz said, ...
```

The article's opening quote mark merged with the JSON string opener, leaving a bare
`"` after `slow down,`. JSON ends the string there and chokes on ` Morningstar`.
`_fix_json` only repaired trailing commas and unquoted keys, so the whole response
was discarded. Evidence excerpts are verbatim article text, which makes this the
malformation most likely to recur.

**Changes (issue #82):**
- `_escape_interior_quotes` walks the document tracking string state. A quote not
  followed (past whitespace) by `,` `}` `]` `:` or `"` is interior text and gets
  escaped; otherwise it closes the string.
- `_recover_json` tries that repair alone before the full `_fix_json`. The regex
  passes are not string-aware — the unquoted-key pattern rewrites `, revenue:` inside
  a string value — so they are a fallback, which recovers documents the combined pass
  destroyed.
- Parse failures log the text around the error position. The first failure is logged
  against the model's original output, since repairs shift every offset after the
  edit and a window from repaired text can point at parser-introduced damage.

**Why `"` is a terminator.** Valid JSON never places two strings side by side, so
including it costs nothing on well-formed input. Excluding it made `["a" "b"]` — a
dropped comma — escape both inner quotes, merge the two excerpts into one corrupted
string, and *validate*, sending a garbled excerpt to the published feed. Failing to
parse is recoverable; silently rewriting an excerpt is not.

A quote that is both interior and followed by one of those characters (`"He said
"yes", loudly."`) still defeats the heuristic. It stays unparseable rather than
parsing wrongly. Structured output, which removes the class, is tracked in #83.

### 2026-07-01: Migrate labeling model to Claude Haiku 4.5 (cost downgrade)

Migrated the labeling model from Claude Sonnet 4.6 (`claude-sonnet-4-6`) to Claude
Haiku 4.5, pinned to the dated snapshot `claude-haiku-4-5-20251001` for
reproducibility (labels persist `model_version`). Motivation: cost — Haiku 4.5 is
~67% cheaper ($1/$5 vs $3/$15 per MTok), shares the Sonnet 4.6 tokenizer family
(no token-inflation penalty), and still supports `temperature=0` (the determinism
lever the scorecard relies on).

**Why not Sonnet 5:** its ~30% tokenizer inflation erased the introductory
discount (net *more* expensive than 4.6 after the intro window ends 2026-08-31)
and it rejects non-default `temperature` with a 400 — removing determinism.
Evaluated and rejected in favor of the Haiku cost-downgrade.

**Validation (issue #53):** a retrospective contrastive eval
(`scripts/eval_model_migration.py`) re-labeled 162 articles against the Sonnet 4.6
/ v1.9.0 baseline. Results: **0.0% newly-labeled rate** (no scorecard-inflation
risk — critical, since the FP/EP pre-filters are disabled so the labeler is the
only junk gate), **0 parse/truncation failures**, **93.9% exact sentiment
agreement** (≥ 4.6's 92.4%), and outcome disagreements dominated by Haiku
*correctly* rejecting brand-collision / financial / marketing false positives that
4.6 mislabeled. Statistical power on positives is low (30 labeled, 1 human
anchor), compensated by a **post-flip drift-monitoring window**.

**Changes:** new prompt version **v1.10.0** (byte-identical text to v1.9.0, model
only), promoted to production; all five Sonnet 4.6 call sites swapped to Haiku
(labeling — eval-gated; agent analysis, workflow-learning, experiment reflection —
ungated, low-stakes). Reusable model-migration skill tracked in #54; Allbirds
corporate-pivot labeling decision-boundary deferred to #55.

### 2026-06-01: Fix Jekyll build failure from mojibake control characters in feed

The website's GitHub Pages "Deploy site" build began failing with
`_data/esg_news.json: control characters are not allowed at line 1 column 1
(Psych::SyntaxError)`. Jekyll parses `_data/*.json` with its YAML parser (Ruby
Psych), which rejects C1 control characters (U+0080–U+009F).

**Root cause:** Scraped article text contained Windows-1252 smart punctuation
(’ “ ” —) stored as raw C1 control bytes — *mojibake* introduced when
`newspaper` misdetected a page's charset. This pre-existing data problem was
*unmasked* by the same-day prettier-JSON change (below): the old `json.dump`
default escaped non-ASCII as `\uXXXX` (harmless to YAML), whereas
`dumps_prettier` emits raw UTF-8, so the C1 bytes reached the committed file.

**Fix (defense-in-depth):**
- New shared `src/data_collection/text_normalize.py` (`normalize_text`,
  `repair_mojibake`, `find_illegal_chars`) repairs mojibake via `ftfy` and
  strips YAML-illegal control characters. Idempotent; used at every layer.
- **Ingest:** the scraper and `Database.upsert_article` normalize content,
  title, and description before storage.
- **One-time repair:** `scripts/repair_text_encoding.py` (`--dry-run`) repairs
  existing rows in `articles`, `article_chunks`, `brand_labels`, and
  `label_evidence`.
- **Export guard:** `export_website_feed.guard_feed_data` repairs any residual
  control characters in the assembled feed (and Atom fields) just before
  serialization, logging a non-blocking warning (with article id + field path)
  whenever it has to — a signal that ingest-time normalization missed a case.
- **Validator:** `website_export.validate_export` now counts articles correctly
  for the dict-shaped feed (previously always reported 0) and YAML-parses the
  feed (PyYAML mirrors Jekyll's Psych) so a build-breaking feed fails validation
  and blocks the push instead of shipping silently.

### 2026-06-01: Prettier-compatible JSON feed export

The website feed JSON is now emitted in Prettier's formatting directly by the
generator, fixing a `prettier --check` CI failure on the GitHub Pages repo that
began 2026-05-28.

**Root cause:** `export_website_feed.write_json` always used
`json.dump(indent=2)`, which expands every array (including single-item arrays)
onto multiple lines. Previously the `website_export` cron masked this by running
`npx prettier --write` before committing. The 2026-05-27 worktree hardening
(commit `b14994f`) moved the cron into a dedicated `-feed` git worktree that
lacks `node_modules`, so Prettier could no longer resolve the
`@shopify/prettier-plugin-liquid` plugin declared in `.prettierrc` and silently
failed (warning-only), committing the raw multi-line JSON.

**Fix:** `write_json` now serializes via `dumps_prettier`, a small encoder that
reproduces Prettier's JSON output (printWidth 150, flat containers when they
fit / break otherwise, `{ }` brace spacing, raw UTF-8, exponent normalization).
Validated byte-for-byte against Prettier 3.1.1 and 3.8.3 over the full
production feed. The now-redundant `npx prettier --write` step was removed from
the `website_export` workflow.

### 2026-02-11: Scorecard History Storage

Added database storage for daily scorecard snapshots, enabling historical trend analysis and brand performance tracking over time.

**New tables:**
- `scorecard_snapshots` - Daily snapshot metadata (period, article counts, dedup settings)
- `scorecard_brand_scores` - Per-brand scores with category breakdown, rank, and medals

**New module:** `src/scorecard/`
- `ScorecardDatabase` class with methods for saving and querying scorecard history
- Query methods: `get_brand_score_history()`, `get_medal_history()`, `get_deduplication_stats()`

**Integration:**
- `website_export` workflow now saves scorecard to database after each export
- Step is skipped in dry-run mode
- Uses upsert semantics (safe to re-run on same day)

**Key design decisions:**
- Only brands with labeled articles during the period are stored (not all 50 tracked brands)
- Full brand coverage analysis can be achieved by joining with `brand_labels` table
- All category scores stored (E, S, G, D) plus total, rank, and medal status

**Migration:** `psql $DATABASE_URL -f migrations/006_scorecard_history.sql`

**Query examples:** See `queries/scorecard_queries.sql` for trend analysis, medal history, etc.

### 2026-01-27: Domain Blacklist for Data Collection

Added ability to block low-quality news sources from data collection.

**New features:**
- `BLOCKED_DOMAINS` list in `src/data_collection/config.py`
- `is_blocked_domain()` helper function in `collector.py`
- Articles from blocked domains filtered during API collection
- `articles_blocked_domain` stat tracked in `CollectionStats`

**Initial blocklist:** `openpr.com` (AI-generated market reports with no real journalism)

**To add a blocked domain:** Edit `BLOCKED_DOMAINS` in `src/data_collection/config.py`

### 2026-01-26: Sustainability Scorecard for Website

Added a "Sportswear Sustainability Scorecard" to the ESG news website, ranking brands based on recent news sentiment.

**New features:**
- Scorecard calculation in `scripts/export_website_feed.py`
- Article deduplication using sentence embeddings (all-MiniLM-L6-v2)
- Top 3 performers (positive scores only) with medal badges
- Back 3 performers (negative scores only)
- Category breakdown per brand (E, S, G, D)
- Date range filter on website (7/14/30 days, All, custom)

**Scoring:** Positive=+2 pts, Neutral=+1 pt, Negative=-1 pt

**CLI options:**
- `--no-scorecard` - Skip scorecard generation
- `--scorecard-period-days N` - Custom period (default: 14)
- `--no-dedupe` - Disable article deduplication
- `--similarity-threshold N` - Custom similarity threshold (default: 0.75)

### 2026-01-20: Cross-Encoder Reranking for Evidence Quality

Integrated cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve evidence matching quality. The reranker jointly encodes (excerpt, chunk) pairs for more accurate relevance scoring than bi-encoder embeddings.

**New features:**
- `src/labeling/reranker.py` - CrossEncoderReranker class with lazy model loading
- `rerank_score` and `match_method` columns in `label_evidence` table
- Website export sorts evidence by `rerank_score` (falling back to `relevance_score`)
- Configurable top-N evidence per category in export (`--top-n-evidence`)
- Backfill script for existing articles: `scripts/backfill_rerank_scores.py`

**Configuration:**
- `RERANK_ENABLED=true` (default) - Enable/disable reranking
- `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` - Model to use
- `RERANK_TOP_K=10` - Candidates to rerank per excerpt
- `RERANK_WEIGHT=0.6` - Weight for combined score: `(1-w)*initial + w*rerank`

**Migration:** `psql $DATABASE_URL -f migrations/004_rerank_scores.sql`

### 2026-01-16: Expanded Stock Article Classification Guidelines

Clarified criteria for distinguishing between `false_positive` (pure metrics) and `skipped` (substantive content) for stock/finance articles. See [LABELING.md](./LABELING.md#stock-article-classification) for detailed guidelines.

### 2026-01-14: Clarified is_sportswear_brand Policy for Stock Articles

`is_sportswear_brand` is about **substantive content**, not just identity:
- `true` → Article has substantive content (products, business news, strategy, analyst commentary with reasoning)
- `false` → Brand refers to something else OR pure stock metrics only (no substantive content)

---

## 2025

### 2025-12-29: MLOps Improvements

Added `src/mlops/` module: MLflow tracking, Evidently drift detection, webhook alerts, daily monitoring cron job.

### 2025-12-29: FP Classifier Batch API

Optimized to batch API calls (N articles → 1 call). Fixed Docker deployment issues.

### 2025-12-28: FP Classifier Pre-filter Integration

FP classifier as optional pre-filter: articles with probability < threshold marked `false_positive`, skip LLM.
- `FP_CLASSIFIER_ENABLED=true`, `FP_SKIP_LLM_THRESHOLD=0.5`
- Migration: `psql $DATABASE_URL -f migrations/002_classifier_predictions.sql`

### 2025-12-26: Added skipped_at Timestamp & Tangential Brand Mention Guidance

Added `skipped_at` column for tracking. Updated prompts to identify false positives for tangential brand mentions (biographical, stock-only, incidental references).

Migration: `ALTER TABLE articles ADD COLUMN skipped_at TIMESTAMP WITH TIME ZONE;`
