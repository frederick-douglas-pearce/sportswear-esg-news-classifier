# Agent Orchestrator

The ESG News Classifier includes a custom-built agent orchestrator that automates daily operations, reducing manual maintenance to near-zero while ensuring data quality and system health.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Workflows](#workflows)
  - [Daily Labeling](#daily-labeling)
  - [Drift Monitoring](#drift-monitoring)
  - [Website Export](#website-export)
  - [Model Training](#model-training)
- [CLI Usage](#cli-usage)
- [Scheduling with Cron](#scheduling-with-cron)
- [Configuration](#configuration)
- [Notifications](#notifications)
- [State Management](#state-management)
- [LLM Intelligence](#llm-intelligence)
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
- **LLM intelligence**: Claude Sonnet analyzes labeling results for quality assurance
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
│     Email/Slack    Claude Sonnet                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/agent/
├── __init__.py
├── __main__.py          # CLI entry point
├── config.py            # Configuration from environment
├── state.py             # YAML-based state management
├── runner.py            # Script execution with retries
├── notifications.py     # Email (Resend) + webhook notifications
├── llm.py               # Claude Sonnet integration
└── workflows/
    ├── __init__.py
    ├── base.py              # Workflow base class + registry
    ├── daily_labeling.py    # Daily labeling workflow
    ├── drift_monitoring.py  # Classifier drift detection
    ├── website_export.py    # Jekyll feed export
    └── model_training.py    # Model training with notebook pause
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
| 4. `run_llm_analysis` | Claude Sonnet analyzes results for potential errors |
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
| 1. `check_fp_drift` | Run Evidently drift detection for FP classifier |
| 2. `check_ep_drift` | Run Evidently drift detection for EP classifier |
| 3. `evaluate_drift_results` | Determine if action is needed |
| 4. `send_drift_alerts` | Send alerts if drift detected |
| 5. `generate_drift_report` | Generate summary report |

**Alert Trigger**: Drift score exceeds configured threshold (default: 0.1)

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

### Model Training

**Schedule**: Manual (triggered when new training data is available)
**Purpose**: Automate model retraining with human-in-the-loop notebook review.

**Steps**:

| Step | Description |
|------|-------------|
| 1. `export_training_data` | Export FP and EP training datasets |
| 2. `check_data_quality` | Validate record counts, class balance |
| 3. `notify_and_pause` | Send email, pause for notebook execution |
| 4. *User runs notebooks* | Manual: fp1 → fp2 → fp3 (or ep1 → ep2 → ep3) |
| 5. `compare_models` | Compare new model metrics to production |
| 6. `prompt_promotion` | Pause for promotion approval |
| 7. `promote_model` | Update model registry |
| 8. `trigger_deployment` | Trigger GitHub Actions deployment |

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

### LLM Analysis Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_LLM_ANALYSIS` | Enable Claude analysis of labeling | `true` |
| `AGENT_LLM_ERROR_THRESHOLD` | Error rate threshold to trigger analysis | `0.0` (always run) |
| `AGENT_LLM_MODEL` | Model for LLM analysis | `claude-sonnet-4-20250514` |
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

The agent integrates Claude Sonnet to analyze labeling results and detect potential issues:

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

### Configuration

```bash
# Always run LLM analysis (default)
AGENT_LLM_ERROR_THRESHOLD=0.0

# Only run when error rate exceeds 10%
AGENT_LLM_ERROR_THRESHOLD=0.10

# Disable LLM analysis
AGENT_LLM_ANALYSIS=false
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
