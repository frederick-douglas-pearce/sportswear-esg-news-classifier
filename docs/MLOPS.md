# MLOps: Experiment Tracking & Monitoring

This document provides detailed information about the optional MLOps features for experiment tracking and production monitoring.

> **Quick Start:** For a high-level overview, see the [main README](../README.md#mlops).

## Overview

The project includes optional MLOps features that use **graceful degradation** - they work when disabled with no code changes required.

## MLflow Experiment Tracking

Track training experiments with hyperparameters, metrics, and model artifacts.

### Enable MLflow

```bash
# In .env
MLFLOW_ENABLED=true
MLFLOW_TRACKING_URI=sqlite:///mlruns.db  # Local SQLite tracking
# Or use a remote server:
# MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### Training with MLflow

```bash
# Train with automatic MLflow logging
uv run python scripts/train.py --classifier fp --verbose

# View experiments in MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

### What Gets Logged

- Training parameters (model type, hyperparameters, target recall)
- Metrics (test F2, recall, precision, threshold)
- Artifacts (pipeline, config JSON)
- Run metadata (timestamp, classifier type)

### Programmatic Usage

```python
from src.mlops import ExperimentTracker

tracker = ExperimentTracker("fp")
with tracker.start_run(run_name="fp-v1.2.0"):
    # Your training code...
    tracker.log_params({"n_estimators": 200, "max_depth": 20})
    tracker.log_metrics({"test_f2": 0.974, "test_recall": 0.988})
    tracker.log_artifact("models/fp_classifier_pipeline.joblib")
```

## Evidently AI Drift Monitoring

Detect prediction drift and data quality issues in production.

### Enable Evidently

```bash
# In .env
EVIDENTLY_ENABLED=true
DRIFT_THRESHOLD=0.1  # Alert if drift score > 10%
```

### Running Drift Monitoring

```bash
# Production drift check from database (recommended)
uv run python scripts/monitor_drift.py --classifier fp --from-db

# Extended analysis with HTML report
uv run python scripts/monitor_drift.py --classifier fp --from-db --days 30 --html-report

# Create reference dataset from production data: the 30 days ending where the
# default comparison window starts, so it does not contain that window
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30

# Or pin the window's end explicitly (exclusive, 00:00 UTC) for a reproducible baseline
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 90 --reference-end-date 2026-09-01

# Check reference dataset stats
uv run python scripts/monitor_drift.py --classifier fp --reference-stats

# Legacy: from local log files (for local API testing)
uv run python scripts/monitor_drift.py --classifier fp --logs-dir logs/predictions
```

### Data Sources

- `--from-db`: Load predictions from `classifier_predictions` database table (recommended for production)
- `--logs-dir`: Load from local JSONL log files (for local API development)

### Exit-code contract

`monitor_drift.py` distinguishes three outcomes (`src/mlops/exit_codes.py`). The agent workflow
reads the exit code, never the report text.

| Code | Meaning | Retried? |
|------|---------|----------|
| `0` | The check ran; no drift | n/a |
| `1` | The check ran; **drift detected** -- a result, not an error | No: re-running returns the same answer |
| `2` | **Indeterminate** -- no verdict was produced (the analysis raised, or there was nothing to compare) | Yes: the cause may be transient |

Exit 2 never reads as healthy. Before this contract (issue #71) exit 1 meant both "drift detected"
and "the analysis raised", so a failed check was indistinguishable from a clean one and the drift
safety net reported "all classifiers healthy" on 219 of the 223 runs that failed (out of 232).

Errors go to **stderr**, with a traceback; the agent runner logs the tail of that stream.

### Monitor Output

```
============================================================
DRIFT MONITORING REPORT - FP
============================================================

Timestamp: 2025-12-29 10:30:45
Drift Detected: NO
Drift Score: 0.0000 (threshold: 0.1000)

HTML Report: reports/monitoring/fp/drift_report_20251229_103045.html

============================================================
✅ Status: Healthy - no significant drift detected
--- drift summary (machine-readable) ---
{"classifier": "fp", "exit_code": 0, "indeterminate": false, "drift_detected": false, "drift_score": 0.0, "threshold": 0.1, "error": null, "reference_window": {...}, "reference_observed": {...}, "reference_overlaps_current": false, "columns_assessed": [...], "columns_skipped": {}, "columns_missing_from_reference": [], "metrics_unreadable": [], "columns_checked": [...]}
```

The last line is a single-line JSON summary the `drift_monitoring` workflow consumes via
`ScriptResult.parsed_output`. A run whose exit code claims a verdict but whose summary is absent,
incomplete, or inconsistent with the exit code is treated as `unknown` rather than trusted.

The summary also names what the check assessed (issue #104): `columns_assessed`, `columns_skipped`,
`columns_missing_from_reference`, `metrics_unreadable`, and `columns_checked` (the columns offered
for comparison on the Evidently path, before its per-column input checks). Every summary carries every key; a run that fails before printing a
summary has none. An empty list or object means that measurement ran and found nothing. `null` means
it did not run on the path taken, or was not recorded. For example, a run with no reference or too
few rows never reaches per-column assessment, so its `columns_assessed` and `columns_skipped` are
`null`. An Evidently run with no column in common also has `columns_assessed` of `null`, but it
checks brand columns against the reference first, so its `columns_skipped` records any the
reference lacks (`{}` when there are none; #105). `metrics_unreadable` is `null` wherever no
Evidently metric snapshot was produced, and `columns_missing_from_reference` is `null` when there is
no reference to compare against. The agent workflow copies these fields into its context, report and
run archive. The fields are a record: the workflow neither requires them nor rejects a summary over them. What partial coverage
means for the verdict is decided upstream of them, in the check itself (#105, D023): an offered
**core** column that was not assessed makes the result indeterminate (exit 2) on both paths. A
`brand_*` shortfall is recorded and does not change the verdict. When no brand column was assessed
`brand_drift_score` stays `0.0`, and `brand_assessed_count` of `0` is what says nothing was measured.

### Keeping the reference dataset current

A reference written against an older schema is the failure that started #71: `novelty_score` was
added to `classifier_predictions` after `fp_reference.parquet` was written, and the mismatch raised
`KeyError` from 2026-01-25 (when the column was added) until 2026-09-06, the last run before the
fix. The failure logs are empty -- that is defect 2 of #71 -- so the attribution is from the code
and the schema dates, not from a logged traceback. Two guards now exist, on **both** the Evidently
and the legacy code paths, and neither removes the need to regenerate:

- a **core** column (`probability`, `prediction`, `novelty_score`) present in the current data but
  missing from the reference is **logged as not assessed** and recorded in
  `details["columns_missing_from_reference"]`, which reaches the machine-readable summary and the
  agent's run archive, so a partial comparison is not reported as a whole one. `brand_*` columns are
  not tracked this way: they come and go with `TRACKED_BRANDS` and would bury the signal;
- a **core column offered and not assessed** returns `indeterminate`, i.e. exit 2 -- on both
  paths (#105, D023). That covers a reference sharing no core column, and also any core column
  that was present in both frames and came back unusable: too few values after the NaN drop, the
  same constant in both frames, a metric whose value cannot be read or is non-finite, or a column
  no recognised metric came back for (for example after an Evidently metric rename). Each is
  recorded in `details["columns_skipped"]` with its reason. `details["metrics_unreadable"]` is the
  narrower record: only the ones whose value could not be read. A core column **absent from the
  reference** is the exception. It is never offered, stays in `columns_missing_from_reference`,
  and does not make the result indeterminate: that loss is already logged and fixed by
  `--create-reference`, while #105 is about losses that were silent;
- a column that is **present but cannot be assessed** is recorded in
  `details["columns_skipped"]` with its own reason and left out of the score entirely.
  `details["columns_assessed"]` is the complementary record of what produced a reading. Counting
  an unassessable column as "not drifted" is the shape these two exist to prevent (#102). Both
  fields exist on both paths. Both now record a `brand_*` column absent from the reference as
  `not in reference`, and both apply the same input checks to core columns. Reasons for a skipped
  `brand_*` column still differ between the paths, so the two records are not like-for-like;
- **too few rows** is a floor with two scopes (`DRIFT_MIN_SAMPLE_SIZE`, default 30 — #94). Enough
  rows to compute a statistic is not enough for it to mean anything. A whole *frame* below the
  floor — the reference and the current window are checked independently — returns `indeterminate`
  rather than healthy. A single *column* below it, counted after its NaN are dropped, is skipped
  and recorded. For a `brand_*` column the check still returns a verdict from whatever else it
  could measure; for a core column the result is `indeterminate`, on both paths (#105);
- **no reference dataset at all** returns `indeterminate` as well. This used to split the current
  window in half and compare the halves -- two samples from one window, which agree by
  construction and so always read as "no drift". Use `--create-reference` to establish a
  baseline.

Regenerate after any change to a column the drift check reads: `CORE_DRIFT_COLUMNS`
(`src/mlops/monitoring.py`) from `classifier_predictions`, and the `brand_*` columns derived from
`articles.brands_mentioned` via `TRACKED_BRANDS`:

```bash
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 90
```

The reference window ends where the default comparison window (`DEFAULT_DRIFT_WINDOW_DAYS`,
`src/mlops/config.py`) starts, so a reference does not contain the default comparison window
(issue #97). A check run with a longer `--days` can still overlap it. `--exclude-recent-days N` or
`--reference-end-date YYYY-MM-DD` move the end, and `--create-reference` prints the resolved
window. The requested window is stored inside the parquet (`attrs["reference_window"]`). Every
drift report that loaded a reference carries `reference_window`, `reference_observed` and
`reference_overlaps_current`, both in its details and in the machine-readable summary. The agent
workflow copies them into its context, report and run archive. An overlap is recorded, and it does
not change the verdict. A reference built before this was recorded reports
`reference_window: null`, never a window inferred from its data.

The EP check stays skipped while `AGENT_EP_DRIFT_ENABLED=false`, but the skip counts `ep`
predictions in the drift window first (issue #96). Below `DRIFT_MIN_SAMPLE_SIZE` it stays
`skipped`, and a nonzero count is named in the reason. At or above the floor the verdict is
`unknown` and the drift workflow fails, because EP is running unmonitored. A count that could not
be taken is `unknown` too; the count runs with a connect and a statement timeout, so a database
that stops answering fails the check instead of hanging it.

### Model version in `classifier_predictions`

`model_version` records the version of the artifact the classifier API **loaded**, not the
registry's production pointer (#115, D026). Every script that records a version in
`models/registry.json` (`register_model.py --update-registry`, `retrain.py` promotion,
`promote_model.py`) stamps `version` and `pipeline_sha256` into the artifact's
`<type>_classifier_config.json`. The API reports that version only when `pipeline_sha256` is present
and matches the joblib it loaded. `/model/info` returns it as `version`, with the loaded joblib's short
hash as `artifact_sha256`.

The stamp is written to the working tree. An image picks it up only when it is built from a tree that
holds the stamped config: a local `docker compose build` reads the working tree, while `deploy.yml`
builds from the committed tree, so commit the stamped config with its joblib before deploying. Making
the registry pointer decide what is built is #172.

`pipeline_sha256` binds the version to the joblib only. Config fields such as `threshold` are not
covered (#141). The MLflow run that `register_model.py` logs carries the version as its `version`
tag, but the config file it logs is the one from before the stamp.

When no version can be reported, one of these is recorded instead. The constants live in
`src/deployment/versioning.py` (`NON_VERSION_SENTINELS` is the set):

| Value | Meaning | What to do |
|---|---|---|
| a registry version string (e.g. `v2.5.0`) | a registered artifact whose hash matched | nothing |
| `unversioned` | the loaded artifact has no version, or its config's `pipeline_sha256` is missing or does not match the loaded joblib | register the artifact, then rebuild the image from a tree holding the stamped config |
| `unreported` | the API answered, but its `/model/info` has no `version` field (the image predates it) | rebuild the image |
| `unavailable` | the model-info fetch failed (the client logs why), or the FP batch step failed — the API call, result handling, or saving the prediction; such a row is `action_taken='failed'` and its `error_message` says which | read the log or `error_message` |
| `disabled` | the FP classifier is turned off | configuration |
| `unknown` | legacy value from before #115; never written now | nothing |

When the FP API answered but reported no usable version (`unversioned`, `unreported`, or `unavailable`
from a failed model-info fetch), the labeling pipeline logs one WARNING for the batch. A batch whose
API call fails logs only its batch-failure WARNING; a failure after an unusable version was read
(result handling or the save) logs both; after a verified version, only the batch-failure WARNING. `disabled` rows are written without a warning. A missing version does not stop the pre-filter.

**What changes at merge.** The labeling pipeline runs from the tree, so the first labeling run after
merge records `unreported` for the currently deployed `fp-classifier-api` image (its `/model/info`
has no `version` field), and `unavailable` for a failed fetch or batch, where it used to record
`unknown`. A rebuild from the current tree reports `unversioned`, because the committed
`models/fp_classifier_config.json` carries no stamp. A real version is recorded only after an
artifact is registered with this in place and the image is rebuilt from a tree holding its stamped
config. #176 registered the served FP bytes as `v2.4.0` and committed them with the pointer and
the stamped config, so an FP image rebuilt from the tree reports `v2.4.0`.

`model_version` is not a drift column (`CORE_DRIFT_COLUMNS` in `src/mlops/monitoring.py`), so a change
in the values it records does not require regenerating the drift reference. An EP image rebuilt from
this tree reports `unversioned` until an EP artifact is registered and the image is rebuilt from a
tree holding its stamped config.

### Served, registered and committed models

**Invariant: the registry pointer (`models/registry.json` → `<clf>.production`) decides which model is
built and deployed.** The artifact in `models/<clf>_classifier_*` must be the pointer's registered bytes,
and every image must be built from those bytes. No build enforces it until #172's build guard lands
(D027). For FP, #176 reconciled the pointer with the committed artifact (`v2.4.0`, the bytes the
labeling pipeline is served), and `tests/test_committed_model_registry.py` checks in CI that the
committed FP config's stamp names the pointer and hashes the committed joblib. EP is not reconciled:
its registered bytes are not recoverable (#180), and that test marks EP as an expected failure until
#180 lands.

The served, registered and committed models are three separate things and can differ. Which ones
differ today, with evidence, is recorded on #175. The steps where they can come apart are listed
below, each with a proposed fix (or none) and a proposed owner (an issue, or `deferred`). #172's ACs
are finalized after #175 (D027 §1), so a `#172` entry here is a proposal until #172 accepts it:

| Divergence point | Where | Proposed fix | Proposed owner |
|---|---|---|---|
| Candidates and production share `models/<clf>_classifier_*`. The fp3/ep3 deployment cells and `train.py` (default `--output-dir models`) overwrite it unstamped, and never write `models/registry.json` | `src/fp3_nb/deployment.py`, `src/ep3_nb/deployment.py`, `scripts/train.py` | the build guard refuses bytes that are not the pointer's registered bytes | #172 |
| `register_model.py --update-registry` stamps the shared config even without `--set-production`; `promote_model.py` stamps it even without `--production` | `scripts/register_model.py`, `scripts/promote_model.py` | the build guard refuses a config whose stamp does not name the pointer (D027 §2) | #172 |
| `retrain.py` promotion moves the pointer and dispatches `deploy.yml` with no commit or push, so CI builds whatever `main` holds | `scripts/retrain.py` | dispatch only when the ref `deploy.yml` builds (the default branch; no `--ref` is passed) holds the pointer's bytes | #172 |
| `deploy.yml` skips deploying on a patch bump: the pointer moves, the served image does not | `.github/workflows/deploy.yml` | remove the skip for model changes | #172 |
| `retrain.py` skips triggering a deploy on a patch bump, with the same effect | `scripts/retrain.py` | remove the skip for model changes | #172 |
| The `model_training` promote step registers without `--set-production`, so the pointer never moves | `src/agent/workflows/model_training.py` | pass `--set-production` (failure handling stays #79) | #172 |
| `trigger_deployment` reads a `version` key the registry does not have, so every workflow deploy is tagged `unknown` | `src/agent/workflows/model_training.py` | pass the pointer | #172 |
| Local `docker compose build` and `scripts/deploy_cloudrun.sh` build from the working tree, so they can serve unregistered bytes | `docker-compose.yml`, `scripts/deploy_cloudrun.sh` | the build guard runs in the Dockerfile's builder stage, which both use | #172 |
| A locally registered but uncommitted pointer and artifact agree with each other, so a working-tree build passes the guard. The local compose container is the labeling pipeline's serving path | `docker-compose.yml`, `scripts/deploy_cloudrun.sh`, `models/registry.json` | decide whether the guard also binds committed state (raised on #172 for its finalization) | deferred |
| `retrain.py` numbers the next version from the production pointer, so it can collide with, and overwrite, an existing non-production version. With the pointer at `v2.4.0`, its default next version is `v2.5.0`, which exists; a retrain before #172 must register under a version above the highest existing one | `scripts/retrain.py` | immutable registry versions | #172 |
| A promotion commit can carry `registry.json` without the artifact, or the artifact without the pointer | commit discipline | the build guard fails a tree whose artifact is not the pointer's registered bytes (FP reconciled in #176) | #172 |
| A commit message can name a version the registry does not record | commit discipline | not checked; git history and the registry remain the evidence | deferred |
| `pipeline_sha256` does not bind the config's `threshold` | `src/deployment/versioning.py` | one source of truth for the FP threshold | #141 |
| `pipeline_sha256` does not bind the `src/fp1_nb`/`src/ep1_nb` transformer code the pickle imports, `BRANDS`, or library versions. v2.4.0's pickle predates a transformer attribute and needed a compatibility default to run under today's code (D028); `tests/test_committed_model_registry.py` loads the committed FP artifact and compares its probabilities with values captured from the served container | `src/deployment/versioning.py`, `src/fp1_nb/feature_transformer.py` | record the code SHA at registration | deferred |
| The notebooks' other outputs in `models/` (`fp_feature_config.json`, `fp_training_config.json`, …) describe the last candidate, not the pointer's artifact. The experiment log's state snapshot pairs the pointer's config with the candidate's `fp_feature_config.json`, and a retrain reads the candidate's `fp_training_config.json` | `src/experiment_log/tracker.py`, `src/deployment/training_config.py` | not decided | deferred |
| `deploy.yml`'s cleanup keeps only `:latest` and one other digest, deleting older images that show what was served | `.github/workflows/deploy.yml` | keep images tagged with a registered version | deferred |
| Cloud Run is not on the labeling path (`FP_CLASSIFIER_URL` points at the local container), so "deployed" there is not what serves labels. Its last revision is inferred (#175) to hold v2.5.0's bytes under an `:unknown` tag, while the pointer names `v2.4.0` | `src/labeling/config.py`, `deploy.yml` | decide whether Cloud Run remains a serving target | deferred |

### What Gets Monitored

- Probability distribution drift (KS test or Evidently)
- Prediction rate shifts
- Data quality issues (missing values, outliers)

## Automated Monitoring

Set up daily drift monitoring with cron or GitHub Actions.

### Local Cron Setup

```bash
# Install monitoring cron job (runs daily at 6am UTC)
./scripts/setup_cron.sh install-monitor

# Check status
./scripts/setup_cron.sh status

# Remove monitoring job
./scripts/setup_cron.sh remove-monitor

# View logs
tail -f logs/monitoring/fp_monitoring_$(date +%Y%m%d).log
```

### GitHub Actions

The project includes `.github/workflows/monitoring.yml` for automated drift monitoring:

```yaml
# Runs daily at 6am UTC
# Monitors FP and EP classifiers
# Uploads HTML reports as artifacts
# Sends alerts via webhook if drift detected
```

**Required GitHub Secrets:**
- `ALERT_WEBHOOK_URL` - Slack/Discord webhook for alerts

**Manual Workflow Trigger:**

```bash
# Trigger via GitHub CLI
gh workflow run monitoring.yml --field classifier=fp --field days=7
```

## Webhook Alerts

Receive Slack or Discord notifications when drift is detected.

### Configure Alerts

```bash
# In .env
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_ON_DRIFT=true
ALERT_ON_TRAINING=false  # Optional: alert after training
```

### Alert Example (Slack)

```
⚠️ ESG Classifier Alert
━━━━━━━━━━━━━━━━━━━━━━━━
Drift Detected
Drift detected! Score: 0.1523 (threshold: 0.1000)

Classifier: fp | 2025-12-29 10:30:45

Drift Score: 0.1523
Threshold: 0.1000
Reference Size: 1000
Current Size: 250
```

### Programmatic Alerts

```python
from src.mlops import send_drift_alert

send_drift_alert(
    classifier_type="fp",
    drift_score=0.15,
    threshold=0.10,
    details={"reference_size": 1000, "current_size": 250}
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_ENABLED` | Enable MLflow experiment tracking | `false` |
| `MLFLOW_TRACKING_URI` | MLflow server URI or local path | `sqlite:///mlruns.db` |
| `MLFLOW_EXPERIMENT_PREFIX` | Prefix for experiment names | `esg-classifier` |
| `EVIDENTLY_ENABLED` | Enable Evidently drift detection | `false` |
| `EVIDENTLY_REPORTS_DIR` | Directory for HTML reports | `reports/monitoring` |
| `DRIFT_THRESHOLD` | Drift score threshold for alerts | `0.1` |
| `REFERENCE_DATA_DIR` | Directory for reference datasets | `data/reference` |
| `REFERENCE_WINDOW_DAYS` | Days of data for reference | `30` |
| `AGENT_EP_DRIFT_ENABLED` | Run the EP classifier drift check. Off while EP is on hold (see the EP paragraph above) | `false` |
| `AGENT_EP_DRIFT_SKIP_REASON` | Reason recorded when the EP check is skipped | (see `src/agent/config.py`) |
| `ALERT_WEBHOOK_URL` | Slack/Discord webhook URL | - |
| `ALERT_ON_DRIFT` | Send alert on drift detection | `true` |
| `ALERT_ON_TRAINING` | Send alert after training | `false` |
