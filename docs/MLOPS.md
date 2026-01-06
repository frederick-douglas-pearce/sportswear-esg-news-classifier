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

# Create reference dataset from production data
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30

# Check reference dataset stats
uv run python scripts/monitor_drift.py --classifier fp --reference-stats

# Legacy: from local log files (for local API testing)
uv run python scripts/monitor_drift.py --classifier fp --logs-dir logs/predictions
```

### Data Sources

- `--from-db`: Load predictions from `classifier_predictions` database table (recommended for production)
- `--logs-dir`: Load from local JSONL log files (for local API development)

### Monitor Output

```
============================================================
DRIFT MONITORING REPORT - FP
============================================================

Timestamp: 2025-12-29 10:30:45
Drift Detected: NO
Drift Score: 0.0523 (threshold: 0.1000)

HTML Report: reports/monitoring/fp/drift_report_20251229_103045.html

============================================================
✅ Status: Healthy - no significant drift detected
```

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
| `ALERT_WEBHOOK_URL` | Slack/Discord webhook URL | - |
| `ALERT_ON_DRIFT` | Send alert on drift detection | `true` |
| `ALERT_ON_TRAINING` | Send alert after training | `false` |
