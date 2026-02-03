# syntax=docker/dockerfile:1.7

#############################
# Builder stage
#############################
FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages required for compiling wheels (tokenizers, transformers) and Postgres client libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Prefetch embedding model so runtime pods avoid first-request downloads
RUN python - <<'PYTHON'
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-base-en-v1.5")
PYTHON

COPY . .

#############################
# Runtime stage
#############################
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTENCE_TRANSFORMERS_BACKEND=torch \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    USE_TORCH=1 \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# Runtime dependencies (no build toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
 && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and cached models from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /root/.cache /root/.cache

# Copy project source
COPY . .

# Drop privileges
RUN addgroup --system docs && adduser --system --ingroup docs docs
USER docs

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
