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
#   fp  - False Positive Brand Classifier (~220MB image, needs sentence-transformers + spaCy)
#   ep  - ESG Pre-filter Classifier (~150MB image, TF-IDF only)
#   esg - ESG Multi-label Classifier (future)

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

# Download models conditionally based on classifier type
# FP classifier needs spaCy and sentence-transformers
# EP classifier only needs scikit-learn (already installed)
# Always create .cache directory so COPY doesn't fail
# Use uv pip install for spacy model since uv venvs don't include pip
RUN mkdir -p /root/.cache && \
    if [ "$CLASSIFIER_TYPE" = "fp" ]; then \
        echo "Downloading models for FP classifier..." && \
        uv pip install --python /app/.venv/bin/python en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
        /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    else \
        echo "Skipping spaCy/sentence-transformer download for ${CLASSIFIER_TYPE} classifier"; \
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
COPY --from=builder /app/.venv /app/.venv

# Copy model cache from builder (sentence-transformers, spacy)
# For EP classifier, this copies an empty directory (created in builder stage)
COPY --from=builder /root/.cache /home/appuser/.cache

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV CLASSIFIER_TYPE=${CLASSIFIER_TYPE}

# Copy unified prediction script
COPY scripts/predict.py scripts/

# Copy deployment module (shared by all classifiers)
COPY src/deployment/ src/deployment/
COPY src/__init__.py src/__init__.py

# Copy classifier-specific modules
# FP needs fp1_nb for feature transformer
# EP needs ep1_nb for feature transformer
RUN mkdir -p src/fp1_nb src/ep1_nb src/data_collection

COPY src/fp1_nb/ src/fp1_nb/
COPY src/ep1_nb/ src/ep1_nb/
COPY src/data_collection/config.py src/data_collection/config.py
COPY src/data_collection/__init__.py src/data_collection/__init__.py

# Copy model artifacts for the specified classifier
COPY models/${CLASSIFIER_TYPE}_classifier_pipeline.joblib models/
COPY models/${CLASSIFIER_TYPE}_classifier_config.json models/

# Set ownership for non-root user
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the unified API with classifier type from environment
CMD ["python", "scripts/predict.py"]
