# Multi-Classifier API - Multi-Stage Docker Build
#
# Build:
#   docker build -t fp-classifier-api .                              # FP (default)
#   docker build --build-arg CLASSIFIER_TYPE=ep -t ep-classifier-api .  # EP
#
# Run:
#   docker run -p 8000:8000 fp-classifier-api
#   docker run -p 8001:8000 ep-classifier-api
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

# Copy config file to detect transformer method
COPY models/${CLASSIFIER_TYPE}_classifier_config.json /tmp/config.json

# Download dependencies based on transformer_method in config
# - Methods containing "ner" need spaCy
# - Methods containing "sentence_transformer" need sentence-transformers + spaCy
# - Methods with only "tfidf"/"lsa" need nothing extra (scikit-learn already installed)
# Always create .cache directory so COPY doesn't fail
RUN mkdir -p /root/.cache && \
    TRANSFORMER_METHOD=$(python -c "import json; print(json.load(open('/tmp/config.json'))['transformer_method'])") && \
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
    fi && \
    rm /tmp/config.json

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

# Copy model artifacts for the specified classifier
COPY --chown=appuser:appuser models/${CLASSIFIER_TYPE}_classifier_pipeline.joblib models/
COPY --chown=appuser:appuser models/${CLASSIFIER_TYPE}_classifier_config.json models/

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Run the unified API with classifier type from environment
CMD ["python", "scripts/predict.py"]
