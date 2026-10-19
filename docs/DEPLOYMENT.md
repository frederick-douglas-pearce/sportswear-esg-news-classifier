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

# Start EP API server (different port). EP is on hold, with no production
# version and no committed artifact (#180); this needs an EP artifact in models/.
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

### The registry pointer decides what is built

The production pointer in `models/registry.json` (`<clf>.production`) decides which model an image
packages (#172, D034). The Dockerfile's builder stage runs a build guard
(`check_build_artifact` in `src/deployment/versioning.py`) after installing the Python dependencies
and before the step that may download spaCy or sentence-transformers models, and the build fails
unless:

1. the pointer is not `null`;
2. the pointer names a version registered in `models/registry.json`, and that entry records a
   `pipeline_sha256`;
3. `models/<clf>_classifier_pipeline.joblib` has that sha256;
4. `models/<clf>_classifier_config.json` is stamped with the pointer's `version` and that
   `pipeline_sha256`;
5. that config's `threshold` equals the entry's `threshold` exactly (#141, D035).

Check 4's hash half makes the image report the pointer's version from `/model/info` rather than
`unversioned`. Every supported build runs the guard, because each builds the same Dockerfile:
`docker build`, `docker compose build`, `deploy.yml` and `scripts/deploy_cloudrun.sh`. The runtime stage
copies the two files the guard checked from the builder stage, and `registry.json` stays in the
builder; nothing reads the pointer at serve time. To run the same check without Docker:

```bash
python -I src/deployment/versioning.py --classifier fp --models-dir models   # exit 0 build, 1 refused
```

**When the guard refuses.** The message names the classifier and what it found, and the pointer once
it has been read:

| Refusal | Meaning | Reconcile |
|---|---|---|
| `Registry not found` / `<clf> is not in …registry.json` / `<clf> in … has no production key` / `<clf>'s record … is not an object` / `… production pointer … is …, not a version name` | the registry is missing, has no usable entry for the classifier, or is malformed | restore `models/registry.json` from git; build only a classifier the registry lists |
| `<clf> has no production version (its pointer is null …)` | the classifier has no production version (EP while it is on hold, #180) | nothing may be built until a version is registered and promoted; files in `models/` do not change this |
| `… is not a registered version` | the pointer names a version with no entry | point `production` at a registered version (`promote_model.py --pointer-only`) |
| `… has no recorded pipeline_sha256` | the pointer's entry predates #189 | record that version's hash through its own reconciliation, with byte evidence and a D-entry (D033), or move the pointer to a version that records one |
| `… is not <clf> <version>'s bytes: its sha256 is …, and the registry records …` | `models/` holds other bytes, such as a notebook or `train.py` output | to ship them, register and promote them (`register_model.py --classifier <clf> --bump <major|minor|patch> --update-registry --set-production`) and commit the registry, joblib and config together; otherwise restore the pointer's bytes from git |
| `… is stamped version=…, pipeline_sha256=…` | the config is not the pointer's stamped config | restore the pointer's config from git, or register and promote the artifact as above |
| `… has threshold …, and <clf> <version>'s registry entry records …` | the config's threshold is not the one registered for the pointer's version | restore the pointer's config from git; to ship a new threshold, register and promote the artifact as above (a registered version's threshold is never rewritten, D033) |
| `… has no numeric threshold: it holds …` | the config or the pointer's entry has no usable `threshold` | restore the file from git; the tooling never rewrites a registered entry (D033), so an entry that records no threshold needs its own hand edit with a D-entry, or move the pointer to a version that records one |
| `… registry record has no versions object` / `… registry entry is not an object` | the registry's entries for the classifier are malformed | restore `models/registry.json` from git |
| `Pipeline not found` / `Config not found` / `… is not valid JSON` / `… does not hold a JSON object` / `… cannot be read` | a file is missing, unreadable or malformed | restore it from git |

To restore the committed bytes, run
`git checkout HEAD -- models/<clf>_classifier_pipeline.joblib models/<clf>_classifier_config.json`.
This overwrites whatever candidate is in `models/`, so copy it elsewhere first if you want to keep it.

**Smoke-testing a candidate locally.** `SKIP_MODEL_GUARD=true` (exactly; any other value runs the
guard) skips the guard and packages the files as they are:

```bash
docker build --build-arg SKIP_MODEL_GUARD=true -t fp-candidate .
docker run -p 8002:8000 fp-candidate
```

Use another image name and port. Never build a candidate through `docker compose build`: that replaces
`fp-classifier-api`, the image the labeling pipeline calls. `deploy.yml`, `docker-compose.yml` and
`scripts/deploy_cloudrun.sh` never pass the argument. Such an image claims no version it does not have:
it reports the version its config is stamped with only when the stamp's hash matches the packaged joblib (as a registered version's files are), and `unversioned` otherwise.

**Recording served probabilities after a promotion.** `tests/test_committed_model_registry.py`
compares the committed FP artifact's probabilities with values recorded for the pointer's version,
and fails by name when the pointer has none. In the promotion PR, build the guarded image from the
promotion tree, send the test's four articles to its `/predict/batch`, record the probabilities under
the new version in `SERVED_FP_PROBABILITIES`, and post the request and response on the issue (as on
#176). Keep the old version's values, so a rollback stays covered.

**What the guard does not check.** It binds the joblib's bytes, the config stamp and the config's
`threshold`, not:
- **committed state.** A pointer and artifact registered locally but not committed agree with each
  other, so a working-tree build passes. What remains is a reproducibility gap, not a wrong model,
  and checking git state would need `.git` in the build context (D034). `deploy.yml` builds the
  committed tree.
- **the code the pickle imports** (`src/fp1_nb`, library versions; D028). The load-and-predict test
  covers that for the committed FP artifact.

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

# Train EP classifier (EP is on hold, with no production version and no committed
# artifact, #180; this needs models/ep_training_config.json from the ep2 notebook,
# and none is committed)
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
# Export new training data to the file the notebooks read
uv run python scripts/export_training_data.py --dataset fp -o data/fp_training_data.jsonl

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
# 1. Export fresh training data (includes new labeled articles).
#    Since #154 it excludes classifier-only negatives, unjudged skips and keyword-sweep negatives
#    by default, which shrinks both classes and shifts the class balance: read the printed
#    by-source breakdown and class ratio before retraining (docs/LABELING.md, FP label provenance).
uv run python scripts/export_training_data.py --dataset fp -o data/fp_training_data.jsonl

# 2. Retrain and promote: not available for FP today. retrain.py exits before training (#186),
#    and its promotion moves the pointer and dispatches deploy.yml with no commit, so CI builds
#    whatever main holds. The build guard passes when main agrees with itself; failing a deploy
#    whose served version is not the new pointer is #190 (docs/MLOPS.md). Its Verdict also
#    compares against a production F2 measured before #154's exclusion; #154 PR2's retrain
#    compares like-for-like.

# 3. Rebuild and redeploy the container after a promotion
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
