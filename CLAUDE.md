# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Interaction Preferences

- Explain trade-offs between alternative approaches before committing to one
- Push back if the proposed approach seems suboptimal — suggest a better path
- Flag assumptions being made, especially about intent, scope, or constraints

## Development Workflow

This project uses a branch-based workflow. Do NOT commit directly to main.

1. Work should reference a GitHub issue. Create one if none exists: `gh issue create`
2. Create a feature branch: `git checkout -b <type>/<issue-number>-<description> main`
3. Branch prefixes: `feature/`, `fix/`, `docs/`, `refactor/`
4. Make commits on the branch (imperative mood: "Add...", "Fix...", "Update...")
5. Push and create a PR: `git push -u origin <branch>` then `gh pr create`
6. CI must pass before merging (runs full test suite)
7. Squash merge: `gh pr merge --squash --delete-branch`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## Failure Postmortems

Incidents are logged to `social/postmortems/` (gitignored via the existing `social/*` rule).
The ledger feeds blog posts and interview prep, so entries are written to be candid — that is
why they stay out of the public repo.

**Trigger — the only rule that keeps this cheap:** write an entry when the fix required
**changing your mind about the cause**. Not every bug. A fix that went the way you expected
teaches nothing; that is a `docs/CHANGELOG.md` entry, not a postmortem.

**Write it the same day**, from the issue and PR text, which already contain most of it. The one
field that cannot be reconstructed later is *what I believed that was wrong* — memory discards it
first, and it is the point of the exercise.

**A candidate is not a post; a claim is.** One incident is a bug report. Entries bind
`instance-of:` to a claim in `social/postmortems/claims.md`, and a claim becomes write-ready at
`MIN_INSTANCES` = 2. A claim with one instance is recorded and waits.

Three independent axes, do not collapse them: `subsystem:` is where it broke
(`ingestion · labeling · classifiers · mlops · agent · publication · database · ci · tooling`),
`instance-of:` is what it proves, `signal:` is why it is interesting.

`status:` is the human's field. Nothing else edits it.

Full conventions, field vocabulary, the two-voice split, and the data-provider guardrail:
[social/postmortems/README.md](social/postmortems/README.md).

**Build no machinery for this.** Markdown files in a folder. `src/experiment_log/` is 19 Pydantic
models with a store, tracker, LLM reflection module and CLI, and `data/experiments/` has never
held a single entry — that is the precedent to avoid repeating.

## Project Overview

ESG News Classifier for sportswear brands - a multi-label text classification system that categorizes news articles into ESG (Environmental, Social, Governance) categories for 50 sportswear/outdoor brands (see `src/data_collection/config.py` for full list).

## Commands

```bash
# Setup
uv sync                                    # Install dependencies
uv sync --extra dev                        # Install dev dependencies (testing)
cp .env.example .env                       # Create environment file (add API keys)
docker compose up -d                       # Start PostgreSQL + pgvector

# Data Collection (use uv run to execute in venv)
uv run python scripts/collect_news.py                            # NewsData.io collection (requires API key)
uv run python scripts/collect_news.py --source gdelt             # GDELT collection (free, no key needed)
uv run python scripts/collect_news.py --source gdelt --timespan 6h  # GDELT with 6-hour window
uv run python scripts/collect_news.py --dry-run --max-calls 5    # Test without saving
uv run python scripts/collect_news.py --scrape-only              # Only scrape pending articles
uv run python scripts/gdelt_backfill.py                          # 3-month historical backfill

# Article Labeling (requires ANTHROPIC_API_KEY and OPENAI_API_KEY)
uv run python scripts/label_articles.py --stats                  # View labeling statistics
uv run python scripts/label_articles.py --dry-run --batch-size 5 # Test without saving
uv run python scripts/label_articles.py --batch-size 10          # Label batch of articles
uv run python scripts/label_articles.py --article-id UUID        # Label specific article

# Label Corrections (after /review-labels)
uv run python scripts/fix_label.py show <article-id>              # View article details + content
uv run python scripts/fix_label.py update <id> --status skipped   # Correct labeling status
uv run python scripts/fix_label.py update <id1> <id2> --status skipped  # Batch update
uv run python scripts/fix_label.py statuses                       # List valid statuses

# Export Training Data
uv run python scripts/export_training_data.py --dataset fp       # False positive classifier data
uv run python scripts/export_training_data.py --dataset esg-prefilter  # ESG pre-filter data
uv run python scripts/export_training_data.py --dataset esg-labels     # Multi-label ESG data

# ML Classifier Training & API
uv run python scripts/train.py --classifier fp                 # Train FP classifier
uv run python scripts/train.py --classifier ep                 # Train EP classifier
CLASSIFIER_TYPE=fp uv run python scripts/predict.py            # Start FP API (port 8000)
CLASSIFIER_TYPE=ep uv run python scripts/predict.py            # Start EP API (port 8000)

# Testing
uv run pytest                              # Run all tests
uv run pytest -v                           # Run with verbose output
uv run pytest --cov=src                    # Run with coverage report
RUN_DB_TESTS=1 uv run pytest tests/test_database.py  # Run database tests (requires PostgreSQL)
RUN_MODEL_TESTS=1 uv run pytest tests/test_deduplication.py  # Run opt-in tests that download the real sentence-transformer model (network)

# Scheduled Collection (cron)
./scripts/setup_cron.sh install            # Set up both cron jobs
./scripts/setup_cron.sh status             # Check cron status
./scripts/setup_cron.sh remove             # Remove both cron jobs

# Cross-Encoder Reranking Backfill
uv run python scripts/backfill_rerank_scores.py --dry-run    # Preview backfill
uv run python scripts/backfill_rerank_scores.py              # Run backfill
uv run python scripts/backfill_rerank_scores.py --batch-size 100  # Custom batch size

# Database Backup
./scripts/backup_db.sh backup              # Create a new backup
./scripts/backup_db.sh list                # List available backups
./scripts/backup_db.sh status              # Show backup status and disk usage

# MLOps - Drift Monitoring
uv run python scripts/monitor_drift.py --classifier fp --from-db              # Production drift check (7 days)
uv run python scripts/monitor_drift.py --classifier fp --from-db --html-report  # Generate Evidently HTML report
uv run python scripts/monitor_drift.py --classifier fp --from-db --create-reference --days 30  # Create reference dataset

# MLOps - MLflow (when MLFLOW_ENABLED=true)
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db  # Start MLflow UI (http://localhost:5000)

# Model Registration
uv run python scripts/register_model.py --classifier fp --version v2.2.0  # Register in MLflow
uv run python scripts/register_model.py --classifier fp --bump minor --update-registry  # Auto-version

# Website Feed Export
uv run python scripts/export_website_feed.py --format both \
  --json-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/_data/esg_news.json \
  --atom-output /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io/assets/feeds/esg_news.atom

# Agent Orchestrator
uv run python -m src.agent list                    # List available workflows
uv run python -m src.agent run daily_labeling      # Run daily labeling workflow
uv run python -m src.agent run drift_monitoring    # Run drift monitoring workflow
uv run python -m src.agent run website_export      # Run website export workflow
uv run python -m src.agent run model_training      # Run model training workflow (pauses for notebooks)
uv run python -m src.agent run daily_labeling --dry-run  # Dry run (no side effects)
uv run python -m src.agent continue model_training # Resume paused workflow after notebooks
uv run python -m src.agent status                  # Show workflow status
uv run python -m src.agent history                 # Show workflow history
./scripts/setup_cron.sh install-agent              # Install all agent cron jobs
./scripts/setup_cron.sh status                     # Check cron status

# Workflow Learning (record workflows to generate Agent Skills)
uv run python -m src.workflow_learning start "workflow-name"        # Start recording session
uv run python -m src.workflow_learning start "workflow-name" -d "description"
uv run python -m src.workflow_learning stop                         # Stop recording session
uv run python -m src.workflow_learning list                         # List all sessions
uv run python -m src.workflow_learning show <session-id>            # Show session details
uv run python -m src.workflow_learning analyze <session-id>         # Analyze and generate skill
uv run python -m src.workflow_learning analyze <session-id> --skill-name "custom-name"
uv run python -m src.workflow_learning analyze <session-id> --refine "skill-name"  # Refine existing skill with new recording
uv run python -m src.workflow_learning delete <session-id>          # Delete a session

# Experiment Log CLI (knowledge base queries + decision recording)
uv run python -m src.experiment_log heuristics --classifier fp                     # Look up learned heuristics
uv run python -m src.experiment_log record-decision --classifier fp \
  --experiment-id EXP_ID --phase feature_engineering \
  --trigger "NER features not contributing" --chosen "remove_ner" \
  --reasoning "F2 unchanged with NER"                                              # Record a decision
uv run python -m src.experiment_log update-heuristic --classifier fp \
  --trigger "NER features not contributing" --success true                          # Update heuristic outcome
```

## Status Reporting & SQL Queries

For collection/labeling status reporting, see [docs/COLLECTION.md](docs/COLLECTION.md#status-reporting).

SQL query examples in `queries/` folder: `collection_queries.sql`, `labeling_queries.sql`, `article_queries.sql`, `evidence_queries.sql`, `scorecard_queries.sql`

## Architecture

### Data Collection Pipeline (`src/data_collection/`)
- `config.py` - Settings, brands list, keywords, blocked domains, and API configuration
- `api_client.py` - NewsData.io API wrapper with OR-grouped query generation
- `gdelt_client.py` - GDELT DOC 2.0 API wrapper (free, 3 months history)
- `scraper.py` - Full article text extraction with language detection
- `database.py` - PostgreSQL operations with SQLAlchemy
- `models.py` - SQLAlchemy models (Article, CollectionRun, ArticleChunk, BrandLabel, LabelEvidence, LabelingRun)
- `collector.py` - Orchestrates API collection + scraping with in-memory deduplication and domain filtering

**Domain Blacklist:** Low-quality or AI-generated content sources are blocked via `BLOCKED_DOMAINS` in `config.py`. Articles from blocked domains are filtered during collection and tracked in `CollectionStats.articles_blocked_domain`.

### Labeling Pipeline (`src/labeling/`)
- `config.py` - Labeling settings, Claude prompts, ESG category definitions
- `models.py` - Pydantic models for LLM response parsing
- `chunker.py` - Paragraph-based article chunking with tiktoken token counting
- `embedder.py` - OpenAI embedding wrapper with batch processing
- `labeler.py` - Claude wrapper for ESG classification
- `classifier_client.py` - HTTP client for FP/EP classifier APIs
- `evidence_matcher.py` - Links evidence excerpts to chunks via similarity matching
- `reranker.py` - Cross-encoder reranking for improved evidence quality
- `pipeline.py` - Orchestrates FP pre-filter → chunking → embedding → labeling → reranking

### Prompt Versioning (`prompts/labeling/`)

```
prompts/labeling/
├── registry.json          # Version registry with metadata
├── v1.0.0/ through v1.8.0/
    ├── config.json, system_prompt.txt, user_prompt.txt
```

**To update prompts:** Create new version directory → Update files → Update registry.json → Update `src/labeling/config.py` (runtime prompt)

### Scripts (`scripts/`)
- `collect_news.py` - CLI for NewsData.io/GDELT data collection
- `label_articles.py` - CLI for LLM-based article labeling
- `export_training_data.py` - Export labeled data for ML training (JSONL format)
- `train.py` - Unified training script for FP/EP classifiers (with MLflow integration)
- `predict.py` - Unified FastAPI service for all classifiers
- `retrain.py` - Retrain models with version management
- `register_model.py` - Register models in MLflow without retraining
- `monitor_drift.py` - Monitor prediction drift with Evidently AI
- `export_website_feed.py` - Export labeled articles as JSON/Atom feeds

### MLOps Module (`src/mlops/`)
- `config.py` - MLOps settings from environment variables
- `tracking.py` - MLflow experiment tracking wrapper
- `monitoring.py` - Evidently-based drift detection
- `reference_data.py` - Reference dataset management
- `alerts.py` - Webhook notifications for Slack/Discord

### Agent Orchestrator (`src/agent/`)
- `config.py` - Agent settings (state dir, email, retries, LLM)
- `state.py` - YAML-based state management with checkpointing
- `runner.py` - Script execution wrapper with retry logic
- `notifications.py` - Unified notifications (Resend email + webhooks)
- `llm.py` - Claude integration for labeling analysis
- `workflows/` - Workflow definitions:
  - `base.py` - Workflow base class and registry
  - `daily_labeling.py` - Collection check → labeling → quality metrics → reports
  - `drift_monitoring.py` - FP/EP classifier drift detection with alerts
  - `website_export.py` - JSON/Atom feed generation + scorecard history storage
  - `model_training.py` - Data export → quality check → pause → comparison → promotion → deploy → experiment finalization
- `__main__.py` - CLI entry point (run, continue, status, list, history)

### Scorecard History (`src/scorecard/`)
- `database.py` - ScorecardDatabase class for save/query operations
- Stores daily scorecard snapshots with brand scores for trend analysis
- Integrated into `website_export` workflow (saves after each export)

### Experiment Log (`src/experiment_log/`)

YAML-based experiment tracking for ML agent learning, following an RL-parallel structure (state → action → observation → reward → reflection).

- `models.py` - 19 Pydantic models (ExperimentEntry, Decision, KnowledgeBase, etc.)
- `store.py` - YAML-based ExperimentStore with CRUD, index, knowledge base management
- `tracker.py` - ExperimentTracker: lifecycle orchestration (create → observe → reward → reflect → complete)
- `reflection.py` - ExperimentReflector: LLM-based reflection on completed experiments via Claude API
- `cli.py` - CLI utilities for replay-time knowledge base queries and decision recording
- `__main__.py` - CLI entry point (`uv run python -m src.experiment_log <command>`)

**CLI commands:** `heuristics` (look up learned heuristics), `record-decision` (save a decision during replay), `update-heuristic` (update outcome counters and confidence).

**Integration with model training workflow:** The `model_training` workflow automatically creates experiment entries at `check_data_quality`, records observations at `compare_models`, and finalizes with reward + optional LLM reflection + heuristic counter updates in the `finalize_experiments` step. All tracking is wrapped in try/except so failures never block the training workflow. Experiment IDs persist in workflow context across pause/resume.

**Data directory:** `data/experiments/{classifier}/exp_*.yaml` with `data/experiments/index.yaml` for quick lookup.

### Workflow Learning (`src/workflow_learning/`)

Records user workflows via Screenpipe and generates replayable Agent Skills using Claude analysis.

- `config.py` - Settings (Screenpipe URL, output directories, analysis model)
- `models.py` - Pydantic models (RecordingSession, ScreenContent, AudioTranscript, WorkflowStep, ExtractedDecision)
- `screenpipe_client.py` - Screenpipe REST API wrapper for screen OCR and audio transcription
- `session_manager.py` - YAML-based session state persistence
- `analyzer.py` - Claude integration for extracting workflow steps and decisions from recordings
- `skill_generator.py` - Generates SKILL.md files with KB lookup directives at checkpoint steps
- `experiment_bridge.py` - Bridges extracted decisions to experiment log (Decision entries + Pattern/Heuristic seeds)
- `__main__.py` - CLI entry point (start, stop, list, show, analyze, delete)

**Prerequisites:** Screenpipe must be running (`screenpipe` command) to record screen content and audio.

**Usage Flow:**
1. Start Screenpipe: `screenpipe`
2. Start recording: `uv run python -m src.workflow_learning start "model-training"`
3. Demonstrate workflow while narrating what you're doing
4. Stop recording: `uv run python -m src.workflow_learning stop`
5. Analyze and generate skill: `uv run python -m src.workflow_learning analyze <session-id>`
6. Review generated skill: `.claude/skills/learned/<workflow-name>/SKILL.md`
7. (Optional) Refine with additional recording: record another session, then `analyze <new-id> --refine "skill-name"`

**Skill Refinement:** Multiple recording sessions can contribute to a single skill. Use `--refine` to merge a new recording into an existing skill — the analyzer preserves existing steps while adding detail or new steps from the new recording. This supports notebook-heavy workflows where a first session captures the overall flow and subsequent sessions add detail about specific cells, metrics, and decision points.

**Notebook-Aware Skills:** When recordings involve Jupyter notebooks, the analyzer generates steps with `tool_type: "jupyter"` that reference `mcp__ide__executeCode` for cell execution, plus checkpoint steps with `success_criteria` and `on_failure` guidance for metric verification.

**Decision Extraction & Knowledge Bridge:** The analyzer extracts structured decisions from narration (e.g., "NER features aren't helping, removing them") as `ExtractedDecision` objects. At analyze time, the `experiment_bridge` module saves these as `Decision` entries in the experiment log and seeds `Pattern`/`Heuristic` entries in the knowledge base. Generated SKILL.md files include KB lookup directives (`uv run python -m src.experiment_log heuristics --classifier {classifier}`) at checkpoint steps so the agent can consult past decisions during future replays.

**Output Directories:**
- Session state: `data/workflow_recordings/sessions/`
- Generated skills: `.claude/skills/learned/`

**Limitations (MVP):**
- Running analyze with same skill name overwrites previous skill
- Refinement requires an existing skill file at `.claude/skills/learned/<name>/SKILL.md`

### ML Classifier Notebooks (`notebooks/`)

**False Positive Classifier (3 notebooks):** fp1_EDA_FE.ipynb → fp2_model_selection_tuning.ipynb → fp3_model_evaluation_deployment.ipynb
- **Production (v2.5.0):** Random Forest with TF-IDF + LSA + NER + proximity + brand features (Test F2: 0.987, Recall: 99.7%)

**ESG Pre-filter Classifier (3 notebooks):** ep1_EDA_FE.ipynb → ep2_model_selection_tuning.ipynb → ep3_model_evaluation_deployment.ipynb
- **Production (v1.0.0):** Logistic Regression with TF-IDF + LSA features (Test F2: 0.931, Recall: 100%) — on hold, insufficient data for significant improvement

**Notebook Standards:** All imports in Setup section, grouped: stdlib → third-party → project modules

### Notebook Utilities
- `src/fp1_nb/` - EDA & feature engineering: data_utils, eda_utils, preprocessing, feature_transformer, ner_analysis, modeling
- `src/fp2_nb/` - Model selection: overfitting_analysis
- `src/fp3_nb/` - Deployment: threshold_optimization, deployment
- `src/ep1_nb/`, `src/ep2_nb/`, `src/ep3_nb/` - Same structure for EP classifier

### Test Suite (`tests/`)
Core tests: test_api_client, test_gdelt_client, test_scraper, test_collector, test_database, test_chunker, test_labeler, test_embedder, test_evidence_matcher, test_labeling_pipeline, test_deployment, test_explainability, test_mlops_*, test_retrain, test_agent_*, test_scorecard_database, test_experiment_log/*, test_workflow_learning/*, test_integration

### Database Schema
- **articles**: Article metadata + scraped content + labeling_status
- **collection_runs**: Collection run statistics
- **article_chunks**: Chunked text with embeddings (pgvector)
- **brand_labels**: Per-brand ESG labels with sentiment
- **label_evidence**: Supporting excerpts linked to chunks (rerank_score, match_method)
- **labeling_runs**: Labeling batch tracking
- **classifier_predictions**: ML classifier predictions audit trail
- **scorecard_snapshots**: Daily scorecard metadata (period, article counts, dedup settings)
- **scorecard_brand_scores**: Per-brand scores per snapshot (category scores, rank, medals)

### Environment Variables
```
# Data Collection
NEWSDATA_API_KEY, DATABASE_URL, MAX_API_CALLS_PER_DAY=200, SCRAPE_DELAY_SECONDS=2
GDELT_TIMESPAN=3m, GDELT_MAX_RECORDS=250

# Labeling
ANTHROPIC_API_KEY, OPENAI_API_KEY, LABELING_MODEL=claude-haiku-4-5-20251001
EMBEDDING_MODEL=text-embedding-3-small, LABELING_BATCH_SIZE=10

# FP Classifier Pre-filter
FP_CLASSIFIER_ENABLED=false, FP_CLASSIFIER_URL=http://localhost:8000
FP_SKIP_LLM_THRESHOLD=0.5, FP_CLASSIFIER_TIMEOUT=30.0

# Cross-Encoder Reranking (evidence quality improvement)
RERANK_ENABLED=true, RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_K=10, RERANK_WEIGHT=0.6

# MLOps
MLFLOW_ENABLED=false, MLFLOW_TRACKING_URI=sqlite:///mlruns.db
EVIDENTLY_ENABLED=false, DRIFT_THRESHOLD=0.1
REFERENCE_DATA_DIR=data/reference, REFERENCE_WINDOW_DAYS=30
ALERT_WEBHOOK_URL, ALERT_ON_DRIFT=true

# Agent Orchestrator
AGENT_EMAIL_ENABLED=false, AGENT_EMAIL_RECIPIENT=, AGENT_EMAIL_SENDER=
RESEND_API_KEY=  # Recommended for email (resend.com, 3000/month free)
AGENT_LLM_ANALYSIS=true  # Enable Claude analysis of labeling results
AGENT_LLM_ERROR_THRESHOLD=0.0  # 0.0 = always run, >0 = only if error_rate exceeds
AGENT_LLM_MODEL=claude-haiku-4-5-20251001  # Model for LLM analysis

# Workflow Learning
SCREENPIPE_API_URL=http://localhost:3030  # Screenpipe REST API
WORKFLOW_RECORDING_DIR=data/workflow_recordings  # Session storage
WORKFLOW_SKILLS_DIR=.claude/skills/learned  # Generated skills output
WORKFLOW_ANALYSIS_MODEL=claude-haiku-4-5-20251001  # Model for analysis
```

## ESG Category Structure

- **Environmental**: carbon_emissions, waste_management, sustainable_materials
- **Social**: worker_rights, diversity_inclusion, community_engagement
- **Governance**: ethical_sourcing, transparency, board_structure
- **Digital Transformation**: technology innovation, digital initiatives

Sentiment values: +1 (positive), 0 (neutral), -1 (negative)

## ML Classifier Opportunities

1. **False Positive Classifier** ✅ - Filter non-sportswear brand matches (Production v2.5.0, Test F2: 0.987)
2. **ESG Pre-filter Classifier** ✅ - Identify ESG content before Claude (Production v1.0.0, Test F2: 0.931) — on hold
3. **ESG Multi-label Classifier** - Planned

## Project Phases

1. **Data Collection** ✅ - NewsData.io/GDELT, scraping, PostgreSQL+pgvector
2. **LLM-Based Labeling** ✅ - Claude classification, evidence extraction, embeddings
3. **Model Development** (Current) - FP classifier ✅, EP classifier ✅, ESG multi-label planned
4. **Deployment & MLOps** ✅ - FastAPI, Docker, Cloud Run, drift monitoring

## Deployment

### Docker
```bash
docker build --build-arg CLASSIFIER_TYPE=fp -t fp-classifier-api .
docker run -p 8000:8000 -e CLASSIFIER_TYPE=fp fp-classifier-api
# Or: docker compose up fp-classifier-api ep-classifier-api
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Classify multiple articles |

### CI/CD
GitHub Actions → Google Cloud Run. Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`, `GCP_REGION`

## Website Feed Export

JSON/Atom feeds for Jekyll/al-folio site. Website repo: `/home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io`

### Worktree Setup (one-time)

The `website_export` cron workflow must run in a dedicated git worktree pinned to `main`, separate from any interactive checkout. This isolates unattended pushes from in-progress feature-branch work.

```bash
cd /home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io
git worktree add ../frederick-douglas-pearce.github.io-feed main
# Then point AGENT_WEBSITE_REPO_PATH at the new worktree path:
# AGENT_WEBSITE_REPO_PATH=/home/fdpearce/Documents/Projects/git/github_pages/frederick-douglas-pearce.github.io-feed
```

The workflow's first step (`prepare_worktree`) asserts the worktree is on `main` and fast-forward-pulls BEFORE `export_feeds` writes any files (so the FF pull never has to contend with uncommitted feed changes). `commit_and_push` then runs a defense-in-depth branch check, scopes its status check to the two feed paths, and pushes with an explicit `origin HEAD:main` refspec. Any failure causes `send_error_notification` to dispatch an alert and raise, so the workflow ends in `WorkflowStatus.FAILED` rather than masquerading as a clean run. Set `AGENT_WEBSITE_EXPECTED_BRANCH` if your publishing branch is not `main`.

### Export Commands
```bash
# Full export with scorecard (default)
uv run python scripts/export_website_feed.py --format both \
  --json-output ~/website/_data/esg_news.json \
  --atom-output ~/website/assets/feeds/esg_news.atom

# Export without scorecard
uv run python scripts/export_website_feed.py --format json --no-scorecard

# Custom scorecard period (default: 14 days)
uv run python scripts/export_website_feed.py --format json --scorecard-period-days 30

# Disable article deduplication for scorecard
uv run python scripts/export_website_feed.py --format json --no-dedupe

# Custom similarity threshold for deduplication (default: 0.56)
uv run python scripts/export_website_feed.py --format json --similarity-threshold 0.70
```

### Sustainability Scorecard

The website displays a "Sportswear Sustainability Scorecard" ranking brands based on ESG news sentiment over a rolling 14-day window.

**Scoring System:**
| Sentiment | Points |
|-----------|--------|
| Positive (+1) | +2 points |
| Neutral (0) | +1 point |
| Negative (-1) | -1 point |

**Key Rules:**
- Each article contributes **one score per brand per category** (Environmental, Social, Governance, Digital)
- Multiple evidence excerpts per category are for documentation only - they don't add extra points
- **Top Performers**: Top 3 brands with positive total scores (with medals: 🥇🥈🥉)
- **Back of the Pack**: Bottom 3 brands with negative total scores
- A brand cannot appear in both lists

**Article Deduplication:**
Similar news stories from different sources are deduplicated before scoring using sentence embeddings (all-MiniLM-L6-v2, 384-dim). Articles with cosine similarity >= 0.56 are considered duplicates; only the first is counted. This threshold was tuned on 200 labeled pairs to achieve ~92% recall with 73% precision.

**Website Features:**
- Date range filter with presets (7, 14, 30 days, All) and custom date inputs
- Filter by brand, category, and sentiment
- Collapsible evidence sections showing supporting excerpts

**Files:**
- `scripts/export_website_feed.py` - Export script with scorecard calculation
- `_pages/esg-news.md` - Jekyll page template
- `assets/js/esg_news_filter.js` - Client-side filtering
- `_sass/_esg_news.scss` - Scorecard and filter styling

## Changelog

For full changelog, see [docs/CHANGELOG.md](docs/CHANGELOG.md).

**Recent changes:**
- **2026-09-05**: JSON parser recovers LLM responses that quote the article verbatim - interior-quote escaping, staged repair, error-position logging (#82)
- **2026-02-26**: Workflow learning ↔ experiment log bridge - decision extraction from narration, KB seeding, heuristic CLI, replay-time knowledge consultation
- **2026-02-26**: Experiment log workflow integration - automatic experiment tracking in model training workflow with LLM reflection
- **2026-02-26**: Experiment log schema - YAML-based experiment store with RL-parallel structure (state/action/observation/reward/reflection)
- **2026-02-16**: Workflow learning module - record workflows via Screenpipe and generate Agent Skills with Claude analysis
- **2026-02-11**: Scorecard history storage - daily snapshots saved to `scorecard_snapshots` and `scorecard_brand_scores` tables
- **2026-01-27**: Domain blacklist for data collection - block low-quality sources via `BLOCKED_DOMAINS`
- **2026-01-26**: Sustainability scorecard for website - brand ranking with deduplication
- **2026-01-20**: Cross-encoder reranking for evidence quality

**Stock article classification guidelines:** See [docs/LABELING.md#stock-article-classification](docs/LABELING.md#stock-article-classification)
