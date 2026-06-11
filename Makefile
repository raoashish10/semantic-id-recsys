.PHONY: install data embeddings rqvae sasrec pipeline serve fmt

install:
	pip install -e ".[dev]"

# ── Offline pipeline steps (can run individually or via `pipeline`) ──────────

data:
	python -m data.download
	python -m data.preprocess

embeddings:
	python -m offline.embeddings.generate

rqvae:
	python -m offline.rqvae.train

sasrec:
	python -m offline.sasrec.train

ann:
	python -m offline.ann.build

ranking:
	python -m offline.ranking.train

precompute:
	python -m offline.precompute

# Run the full offline pipeline via Prefect
pipeline:
	python -m offline.pipeline

# ── Online serving ────────────────────────────────────────────────────────────

serve:
	uvicorn serving.api.main:app --reload --port 8000

# ── Infra ────────────────────────────────────────────────────────────────────

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

fmt:
	ruff format . && ruff check --fix .
