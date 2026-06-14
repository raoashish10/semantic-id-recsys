.PHONY: install data embeddings rqvae sasrec ann ranking evaluate index precompute pipeline serve up down fmt test load-test docker-build

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

evaluate:
	python -m offline.evaluate --gate

index:
	python -m offline.pipeline --from index

precompute:
	python -m offline.pipeline --from precompute

# Run the full offline pipeline end-to-end via Prefect
pipeline:
	python -m offline.pipeline

# ── Online serving ────────────────────────────────────────────────────────────

serve:
	uvicorn serving.api.main:app --reload --port 8000

# ── Infra ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t recsys-mlops:latest .

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

test:
	COLD_START_LLM_ENABLED=false pytest tests/ --ignore=tests/load -v --cov=serving --cov=offline --cov-report=term-missing

load-test:
	locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 60s --html tests/load/report.html

fmt:
	ruff format . && ruff check --fix .
