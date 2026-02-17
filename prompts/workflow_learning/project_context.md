# Project Context: ESG News Classifier

## CLI Commands

All commands use `uv run` to execute within the project virtual environment.

### Data Collection
- `uv run python scripts/collect_news.py` - Collect news articles (NewsData.io)
- `uv run python scripts/collect_news.py --source gdelt` - Collect from GDELT (free)
- `uv run python scripts/collect_news.py --scrape-only` - Only scrape pending articles

### Labeling
- `uv run python scripts/label_articles.py --stats` - View labeling statistics
- `uv run python scripts/label_articles.py --batch-size 10` - Label batch of articles

### Training Data Export
- `uv run python scripts/export_training_data.py --dataset fp` - False positive classifier data
- `uv run python scripts/export_training_data.py --dataset esg-prefilter` - ESG pre-filter data
- `uv run python scripts/export_training_data.py --dataset esg-labels` - Multi-label ESG data

### ML Training & Prediction
- `uv run python scripts/train.py --classifier fp` - Train FP classifier
- `uv run python scripts/train.py --classifier ep` - Train EP classifier
- `CLASSIFIER_TYPE=fp uv run python scripts/predict.py` - Start classifier API (port 8000)

### MLOps
- `uv run python scripts/monitor_drift.py --classifier fp --from-db` - Check prediction drift
- `uv run python scripts/retrain.py --classifier fp` - Retrain with version management
- `uv run python scripts/register_model.py --classifier fp --bump minor` - Register model in MLflow

### Agent Orchestrator
- `uv run python -m src.agent run daily_labeling` - Run daily labeling workflow
- `uv run python -m src.agent run drift_monitoring` - Run drift monitoring workflow
- `uv run python -m src.agent run website_export` - Run website export workflow
- `uv run python -m src.agent run model_training` - Run model training workflow (pauses for notebooks)
- `uv run python -m src.agent continue model_training` - Resume paused workflow
- `uv run python -m src.agent status` - Show workflow status

### Website Feed Export
- `uv run python scripts/export_website_feed.py --format both` - Export JSON + Atom feeds

### Testing
- `uv run pytest` - Run all tests
- `uv run pytest tests/test_workflow_learning/` - Run workflow learning tests

## Available Tools

- **Jupyter MCP Server**: Execute notebook cells programmatically via Claude Code's MCP integration. Use this instead of manually opening notebooks in a browser.
- **Agent Orchestrator** (`src/agent/`): Automated workflow runner with state management, retries, and notifications.
- **Claude Code CLI**: The primary interface for interacting with the project. Supports tool use, file editing, and running commands.

## Jupyter MCP Server Tools

When a workflow involves Jupyter notebooks, use these MCP tools instead of manual browser steps:

### Execute notebook code
Tool: `mcp__ide__executeCode`
- Executes Python code in the active Jupyter kernel
- State persists across calls (variables, imports carry over)
- Returns cell output (text, errors, figures)
- Usage: Run cells sequentially, check output between cells

### Get diagnostics
Tool: `mcp__ide__getDiagnostics`
- Returns language diagnostics (errors, warnings) from the IDE

### Notebook Execution Patterns
- Open notebook in IDE first, then use executeCode to run cells
- Run setup/import cells first, then data loading, then analysis
- After model training cells: check metrics in output
- After plotting cells: take screenshots to review figures
- Key metrics to watch: F2 score, recall, precision, AUC

### Notebook Workflow Pattern
1. Open notebook file in IDE
2. Execute cells sequentially using mcp__ide__executeCode
3. At checkpoint cells: inspect output for expected metrics/patterns
4. If results don't meet criteria: adjust parameters and rerun relevant cells
5. After final cells: verify all success criteria met

## Key Conventions

- Always use `uv run` to execute Python scripts (ensures correct virtual environment)
- Project modules are under `src/` (e.g., `src.agent`, `src.data_collection`, `src.labeling`)
- Notebook utility code lives in `src/{notebook}_nb/` directories (e.g., `src/fp1_nb/`)
- ML notebooks follow a 3-notebook pattern: EDA/FE -> Model Selection -> Evaluation/Deployment
- Configuration via environment variables (see `.env` file)
- Database: PostgreSQL with pgvector extension
- Models saved to `models/` directory
- Training data in `data/` directory (JSONL format)
