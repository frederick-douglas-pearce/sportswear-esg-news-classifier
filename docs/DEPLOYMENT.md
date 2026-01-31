# Classifier Deployment Guide

This document covers the complete deployment workflow for the FP (False Positive) and EP (ESG Pre-filter) classifiers, including local deployment, Docker containerization, cloud deployment, and model lifecycle management.

## Table of Contents

- [Deployment Architecture](#deployment-architecture)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Labeling Pipeline Integration](#labeling-pipeline-integration)
- [API Reference](#api-reference)
- [Model Deployment Workflow](#model-deployment-workflow)
- [Semantic Versioning](#semantic-versioning)
- [Daily Retraining Workflow](#daily-retraining-workflow)

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FP Classifier API                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Client ──► FastAPI ──► FPClassifier ──► sklearn Pipeline        │
│              │               │               │                   │
│              │               │               ├── FPFeatureTransformer │
│              │               │               │   (sentence-transformer + NER) │
│              │               │               │                   │
│              │               │               └── RandomForestClassifier │
│              │               │                                   │
│              ▼               ▼                                   │
│         /predict         Threshold                               │
│         /predict/batch   (set in .env)                           │
│         /health                                                  │
│         /model/info                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Local Deployment

### Without Docker

```bash
# Install dependencies
uv sync

# Run training (optional - model already trained)
uv run python scripts/train.py --classifier fp --verbose

# Start FP API server
CLASSIFIER_TYPE=fp uv run python scripts/predict.py
# Or with uvicorn directly:
CLASSIFIER_TYPE=fp uv run uvicorn scripts.predict:app --host 0.0.0.0 --port 8000

# Start EP API server (different port)
CLASSIFIER_TYPE=ep uv run uvicorn scripts.predict:app --host 0.0.0.0 --port 8001

# Access API docs
open http://localhost:8000/docs
```

## Docker Deployment

```bash
# Build and start the API service
docker compose build fp-classifier-api
docker compose up -d fp-classifier-api

# Check health
curl http://localhost:8000/health

# View logs
docker logs fp-classifier-api

# Stop
docker compose down fp-classifier-api
```

**Note**: Docker image size depends on the transformer method (~150MB for TF-IDF/NER, ~1GB with sentence-transformers). The build auto-detects dependencies from the model config.

### Model Registry & Auto-Detection

The model registry (`models/registry.json`) tracks all model versions. When deploying, the Dockerfile automatically reads the transformer method from the model config to install only the required dependencies.

**Transformer Method → Dependencies:**

| Method Pattern | Dependencies | Image Size |
|----------------|--------------|------------|
| `tfidf_lsa` | scikit-learn only | ~150MB |
| `*_ner*` | spaCy (en_core_web_sm) | ~150MB |
| `sentence_transformer*` | spaCy + sentence-transformers | ~1GB |

## Labeling Pipeline Integration

Enable the FP classifier as a pre-filter in the labeling pipeline to reduce LLM costs:

```bash
# 1. Start the FP classifier API
docker compose up -d fp-classifier-api

# 2. Enable pre-filter in .env
FP_CLASSIFIER_ENABLED=true
FP_CLASSIFIER_URL=http://localhost:8000
FP_SKIP_LLM_THRESHOLD=0.5  # Skip LLM for articles <50% sportswear probability

# 3. Run labeling - FP classifier automatically filters false positives
uv run python scripts/label_articles.py --batch-size 20
```

The pipeline uses batch API calls for efficiency (1 call per batch, not per article). Articles below the threshold are marked as `false_positive` and skip LLM labeling, saving ~$0.01-0.02 per article.

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for container orchestration |
| `/model/info` | GET | Model metadata and performance metrics |
| `/predict` | POST | Classify single article |
| `/predict/batch` | POST | Classify multiple articles in one request |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}

# Get model info
curl http://localhost:8000/model/info

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Nike releases new sustainability initiative",
    "content": "The athletic footwear giant unveiled plans...",
    "brands": ["Nike"]
  }'
# {"is_sportswear":true,"probability":0.935,"risk_level":"high","threshold":0.605}

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"title": "Nike shoe sales soar", "content": "...", "brands": ["Nike"]},
      {"title": "Adidas sponsors tournament", "content": "...", "brands": ["Adidas"]}
    ]
  }'
```

### Response Fields

| Field | Description |
|-------|-------------|
| `is_sportswear` | Boolean - true if article is about sportswear brands |
| `probability` | Float (0-1) - classifier confidence |
| `risk_level` | "low" (<0.3), "medium" (0.3-0.6), "high" (>=0.6) |
| `threshold` | Classification threshold used (default: 0.605) |

### Training Commands

```bash
# Train FP classifier with default settings
uv run python scripts/train.py --classifier fp --verbose

# Train EP classifier
uv run python scripts/train.py --classifier ep --verbose

# Train with custom parameters
uv run python scripts/train.py \
  --classifier fp \
  --data-path data/fp_training_data.jsonl \
  --target-recall 0.98 \
  --output-dir models \
  --verbose

# Retrain and auto-promote if model improves
uv run python scripts/retrain.py --classifier fp --auto-promote
```

## Model Deployment Workflow

This section describes the complete workflow for developing, training, evaluating, and deploying model updates.

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DEVELOP: Run notebooks with new data                         │
│    fp1 → fp2 → fp3 (or ep1 → ep2 → ep3)                        │
│    Exports: feature transformer, training config, CV metrics    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAIN: Run train.py                                          │
│    → Trains model using notebook config                         │
│    → Registers in MLflow Model Registry (if enabled)            │
│    → Saves artifacts to models/                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. EVALUATE: Compare to production                              │
│    - Check F2, recall, precision vs current production          │
│    - Review MLflow UI for metrics comparison                    │
│    - Decide if improvement justifies promotion                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. PROMOTE: Move to production (retrain.py)                     │
│    → Archives old production model                              │
│    → Updates models/registry.json                               │
│    → Promotes in MLflow Model Registry                          │
│    → Redeploy API service                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Development (Notebooks)

```bash
# Export new training data
uv run python scripts/export_training_data.py --dataset fp

# Run the 3-notebook pipeline
# 1. fp1_EDA_FE.ipynb - EDA and feature engineering
# 2. fp2_model_selection_tuning.ipynb - Model selection and tuning
# 3. fp3_model_evaluation_deployment.ipynb - Final evaluation
```

**Notebook outputs (saved to `models/`):**
- `fp_feature_transformer.joblib` - Fitted feature transformer
- `fp_training_config.json` - Model type, hyperparameters, feature method
- `fp_cv_metrics.json` - Cross-validation metrics
- `fp_best_classifier.joblib` - Best classifier from CV

### Phase 2: Training (train.py)

```bash
# Train using config exported from notebooks
uv run python scripts/train.py --classifier fp --verbose
```

**What train.py does:**
1. Loads training config from `models/fp_training_config.json`
2. Fits feature transformer on training data only (prevents leakage)
3. Trains classifier on train+val combined
4. Optimizes threshold for target recall (default: 99%)
5. Logs to MLflow (if `MLFLOW_ENABLED=true`)
6. Registers model in MLflow Model Registry
7. Saves pipeline and config to `models/`

### Phase 3: Evaluation (Decide to Promote)

**Promotion Criteria:**

| Criterion | Threshold | Check |
|-----------|-----------|-------|
| Test F2 Score | ≥ production F2 | Compare train.py output to registry |
| Test Recall | ≥ target (99% FP, 98% EP) | train.py output |
| No regression | F2 decrease < 1% | Manual comparison |

```bash
# View current production metrics
cat models/registry.json | python -m json.tool

# Or use MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

**When NOT to promote:**
- F2 score decreased by more than 1%
- Recall dropped below target threshold
- Training data had quality issues
- Model shows overfitting (large train-test gap)

### Phase 4: Promotion (retrain.py)

```bash
# Daily retraining with new data (minor version bump)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl

# Major version bump for new architecture
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --major

# Patch version bump for threshold fix
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --patch

# Auto-promote if metrics improve
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --auto-promote
```

**What retrain.py does:**
1. Determines next version based on bump type (major/minor/patch)
2. Runs `train.py` to train new model
3. Compares metrics to current production
4. Prompts for confirmation (or auto-promotes with `--auto-promote`)
5. Copies model to production location
6. Updates `models/registry.json`
7. Promotes in MLflow Model Registry (if enabled)

## Semantic Versioning

| Version Type | When to Use | Example |
|--------------|-------------|---------|
| **Major** (`--major`) | Breaking changes: new architecture, different features | RandomForest → XGBoost |
| **Minor** (default) | Improvements: new training data, hyperparameter tuning | Daily retraining |
| **Patch** (`--patch`) | Bug fixes: config corrections, threshold tweaks | Threshold 0.53 → 0.54 |

**Version format:** `vMAJOR.MINOR.PATCH` (e.g., `v1.2.3`)

## Daily Retraining Workflow

```bash
# 1. Export fresh training data (includes new labeled articles)
uv run python scripts/export_training_data.py --dataset fp

# 2. Retrain with auto-promote (promotes if F2 improves)
uv run python scripts/retrain.py --classifier fp --data data/fp_training_data.jsonl --auto-promote

# 3. Rebuild and redeploy Docker container (if promoted)
docker compose build fp-classifier-api
docker compose up -d fp-classifier-api
```

**Automation with cron:**
```bash
# Add to crontab for daily retraining at 2am
0 2 * * * cd /path/to/project && ./scripts/retrain_daily.sh fp >> logs/retrain.log 2>&1
```

### Post-Deployment Verification

```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. Verify model info
curl http://localhost:8000/model/info

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Nike sustainability report", "content": "...", "brands": ["Nike"]}'

# 4. Monitor for drift (next day)
uv run python scripts/monitor_drift.py --classifier fp --days 1
```
