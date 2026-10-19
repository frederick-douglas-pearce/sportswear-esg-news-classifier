# Multi-Classifier API - Multi-Stage Docker Build
#
# Build:
#   docker build -t fp-classifier-api .                              # FP (default)
#   docker build --build-arg CLASSIFIER_TYPE=ep -t ep-classifier-api .  # EP (refused
#                                                       # while EP has no production version)
#
# Run:
#   docker run -p 8000:8000 fp-classifier-api
#
# Build guard (#172, D034): in every supported build, the builder stage refuses unless
# models/<type>_classifier_pipeline.joblib and _config.json are the registered,
# stamped bytes of the version models/registry.json's production pointer names.
# docs/DEPLOYMENT.md says how to reconcile a refusal. registry.json is copied into the builder only, never into the
# runtime image.
#
# EP is on hold, with no production version and no committed artifact (#180).
# Its pointer is null, so the guard refuses an EP build by name, whatever
# models/ep_* files are present.
#
# Local candidate smoke test only (never through docker compose, which would
# replace the image the labeling pipeline calls):
#   docker build --build-arg SKIP_MODEL_GUARD=true -t fp-candidate .
#   docker run -p 8002:8000 fp-candidate
# The bypass packages the files as they are. The API reports the version the
# config is stamped with only when the stamp's hash matches the packaged joblib
# (as a registered version's files are), and `unversioned` otherwise.
#
# Classifier Types:
#   fp  - False Positive Brand Classifier (~150MB image)
#   ep  - ESG Pre-filter Classifier (~150MB image)
#   esg - ESG Multi-label Classifier (future)
#
# Dependency Detection:
#   The build automatically detects which dependencies are needed by reading
#   the transformer_method from {classifier}_classifier_config.json:
#   - Methods containing "ner" → spaCy (en_core_web_sm)
#   - Methods containing "sentence_transformer" → sentence-transformers + spaCy
#   - Methods with only "tfidf"/"lsa" → scikit-learn only (no extra downloads)

# Build argument for classifier type
ARG CLASSIFIER_TYPE=fp

# ==============================================================================
# Stage 1: Builder - Install dependencies and download models
# ==============================================================================
FROM python:3.12-slim AS builder

# Re-declare ARG after FROM to make it available
ARG CLASSIFIER_TYPE

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install production dependencies
RUN uv sync --frozen --no-dev --no-editable

# Build guard inputs. After uv sync, so the uv sync layer stays cached across
# model and registry changes; a change to any of them re-runs the guard and every
# step after it (D034). The bracket globs let an absent
# artifact match nothing, so a classifier with no files still reaches the guard
# and its refusal by name.
COPY models/registry.json models/${CLASSIFIER_TYPE}_classifier_pipeline.jobli[b] models/${CLASSIFIER_TYPE}_classifier_config.jso[n] /build/models/
COPY src/deployment/versioning.py /build/versioning.py

# Refuse unless the files are the registry pointer's registered bytes. Run by
# path under -I, so it imports nothing from src. Only SKIP_MODEL_GUARD=true
# (exactly) skips it; deploy.yml and docker-compose.yml never pass it.
ARG SKIP_MODEL_GUARD=false
RUN if [ "$SKIP_MODEL_GUARD" = "true" ]; then \
        echo "WARNING: SKIP_MODEL_GUARD=true: the build guard did not run. This image is an unchecked candidate for a local smoke test; never build it through docker compose."; \
    else \
        python -I /build/versioning.py --classifier "${CLASSIFIER_TYPE}" --models-dir /build/models; \
    fi

# Download dependencies based on transformer_method in config
# - Methods containing "ner" need spaCy
# - Methods containing "sentence_transformer" need sentence-transformers + spaCy
# - Methods with only "tfidf"/"lsa" need nothing extra (scikit-learn already installed)
# Always create .cache directory so COPY doesn't fail
RUN mkdir -p /root/.cache && \
    TRANSFORMER_METHOD=$(python -c "import json; print(json.load(open('/build/models/${CLASSIFIER_TYPE}_classifier_config.json'))['transformer_method'])") && \
    echo "Detected transformer method: ${TRANSFORMER_METHOD}" && \
    if echo "$TRANSFORMER_METHOD" | grep -q "sentence_transformer"; then \
        echo "Installing spaCy + sentence-transformers for ${TRANSFORMER_METHOD}..." && \
        uv pip install --python /app/.venv/bin/python en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
        /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    elif echo "$TRANSFORMER_METHOD" | grep -q "ner"; then \
        echo "Installing spaCy for ${TRANSFORMER_METHOD}..." && \
        uv pip install --python /app/.venv/bin/python en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl; \
    else \
        echo "No extra dependencies needed for ${TRANSFORMER_METHOD}"; \
    fi

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:3.12-slim

# Re-declare ARG after FROM
ARG CLASSIFIER_TYPE

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder (includes all dependencies)
# Use --chown to set ownership during copy (avoids slow chown -R)
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy model cache from builder (spacy models, sentence-transformers for FP)
# For EP classifier, this copies an empty directory (created in builder stage)
COPY --from=builder --chown=appuser:appuser /root/.cache /home/appuser/.cache

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV CLASSIFIER_TYPE=${CLASSIFIER_TYPE}

# Copy unified prediction script
COPY --chown=appuser:appuser scripts/predict.py scripts/

# Copy deployment module (shared by all classifiers)
COPY --chown=appuser:appuser src/deployment/ src/deployment/
COPY --chown=appuser:appuser src/__init__.py src/__init__.py

# Copy classifier-specific modules
# FP needs fp1_nb for feature transformer
# EP needs ep1_nb for feature transformer
RUN mkdir -p src/fp1_nb src/ep1_nb src/data_collection && chown -R appuser:appuser src/

COPY --chown=appuser:appuser src/fp1_nb/ src/fp1_nb/
COPY --chown=appuser:appuser src/ep1_nb/ src/ep1_nb/
COPY --chown=appuser:appuser src/data_collection/config.py src/data_collection/config.py
# Create minimal __init__.py (avoids importing models which needs SQLAlchemy)
RUN echo 'from .config import settings' > src/data_collection/__init__.py && chown appuser:appuser src/data_collection/__init__.py

# Copy the model artifacts the builder's guard checked, not the build context's
COPY --from=builder --chown=appuser:appuser /build/models/${CLASSIFIER_TYPE}_classifier_pipeline.joblib /build/models/${CLASSIFIER_TYPE}_classifier_config.json models/

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Run the unified API with classifier type from environment
CMD ["python", "scripts/predict.py"]
