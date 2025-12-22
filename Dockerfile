# FP Brand Classifier API - Multi-Stage Docker Build
#
# Build: docker build -t fp-classifier-api .
# Run:   docker run -p 8000:8000 fp-classifier-api
#
# The image includes:
# - Trained model pipeline (~110MB)
# - Sentence-transformers for embeddings
# - spaCy with en_core_web_sm model for NER
# - FastAPI + uvicorn for serving

# ==============================================================================
# Stage 1: Builder - Install dependencies and download models
# ==============================================================================
FROM python:3.12-slim AS builder

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
# We use --no-dev to exclude development dependencies
RUN uv sync --frozen --no-dev --no-editable

# Download spaCy model
RUN /app/.venv/bin/python -m spacy download en_core_web_sm

# Download sentence-transformer model (cached in .cache)
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:3.12-slim

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
COPY --from=builder /root/.cache /home/appuser/.cache

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/appuser/.cache/huggingface

# Copy application code
COPY predict.py .
COPY src/deployment/ src/deployment/
COPY src/fp1_nb/ src/fp1_nb/
COPY src/data_collection/config.py src/data_collection/config.py
COPY src/__init__.py src/__init__.py
COPY src/data_collection/__init__.py src/data_collection/__init__.py

# Copy model artifacts
COPY models/fp_classifier_pipeline.joblib models/
COPY models/fp_classifier_config.json models/

# Set ownership for non-root user
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["python", "predict.py"]
