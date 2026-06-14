FROM python:3.12-slim

WORKDIR /app

# Build tools needed for some Python packages (e.g. faiss, tokenizers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (maximise layer cache reuse)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "." \
    && pip install --no-cache-dir faiss-cpu

# Copy application source
COPY data/       data/
COPY offline/    offline/
COPY serving/    serving/

# Copy model artifacts produced by the offline pipeline.
# Weights (.pt) are gitignored but must be present in the build context.
# Run `make rqvae sasrec` before building this image.
COPY artifacts/rqvae/config.json           artifacts/rqvae/config.json
COPY artifacts/rqvae/semantic_ids.parquet  artifacts/rqvae/semantic_ids.parquet
COPY artifacts/rqvae/model.pt              artifacts/rqvae/model.pt
COPY artifacts/sasrec/config.json          artifacts/sasrec/config.json
COPY artifacts/sasrec/model.pt             artifacts/sasrec/model.pt

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
