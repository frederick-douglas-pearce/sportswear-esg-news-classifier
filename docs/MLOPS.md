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
to the Evidently report). Every summary carries every key; a run that fails before printing a
summary has none. An empty list or object means that measurement ran and found nothing. `null` means
it did not run on the path taken, or was not recorded. For example, a run with no reference or too
few rows, or an Evidently run with no column in common, never reaches per-column assessment, so its
`columns_assessed` and `columns_skipped` are `null`. `metrics_unreadable` is `null` wherever no
Evidently metric snapshot was produced, and `columns_missing_from_reference` is `null` when there is
no reference to compare against. The agent workflow copies these fields into its context, report and
run archive. The drift alert includes whichever of them name something not assessed. They are a
record: partial coverage does not change the verdict (turning it into one is #105), and the workflow
neither requires them nor rejects a summary over them.

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
- a reference sharing *no* comparable core column returns `indeterminate`, i.e. exit 2 -- on both
  paths. On the Evidently path this also covers the case where core columns are offered but no
  core metric comes back usable: a metric whose value cannot be read, or that comes back
  non-finite, is skipped and recorded in `details["columns_skipped"]` rather than counted as
  p=1.0, so it does not prop up `total_core` and the guard actually fires. Without it the score
  would be a fabricated 0.0. `details["metrics_unreadable"]` is the narrower record — only the
  ones whose value could not be read. `columns_skipped` is the wider of the two, but it is **not**
  a complete inventory of what went unassessed: a core column absent from the reference is in
  `columns_missing_from_reference`, and a `brand_*` column absent from the reference is in neither,
  because it never reaches `columns_to_check` (#105);
- a column that is **present but cannot be assessed** is recorded in
  `details["columns_skipped"]` with its own reason and left out of the score entirely.
  `details["columns_assessed"]` is the complementary record of what produced a reading. Counting
  an unassessable column as "not drifted" is the shape these two exist to prevent (#102). Both
  fields exist on both paths, but they are **not like-for-like**: the legacy path records any
  column it declined to test, while on the Evidently path a `brand_*` column absent from the
  reference is still dropped silently;
- **too few rows** is a floor with two scopes (`DRIFT_MIN_SAMPLE_SIZE`, default 30 — #94). Enough
  rows to compute a statistic is not enough for it to mean anything. A whole *frame* below the
  floor — the reference and the current window are checked independently — returns `indeterminate`
  rather than healthy. A single *column* below it, counted after its NaN are dropped, is skipped
  and recorded, and the check still returns a verdict from whatever else it could measure;
- **no reference dataset at all** returns `indeterminate` as well. This used to split the current
  window in half and compare the halves -- two samples from one window, which agree by
  construction and so always read as "no drift". Use `--create-reference` to establish a
  baseline.

Regenerate after any change to what is written to `classifier_predictions`:

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
