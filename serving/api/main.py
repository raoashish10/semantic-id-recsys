"""FastAPI serving application.

Start: uvicorn serving.api.main:app --reload --port 8000

The app loads the SASRec model once at startup and keeps it in memory.
The Redis item store is used for fast item ↔ semantic ID lookups.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from rich.console import Console

from serving.api.routes import router
from serving.api.state import AppState
from serving.metrics import (
    CACHE_HIT_COUNT,
    COLD_START_COUNT,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

console = Console()
state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    console.print("[bold]Loading model and connecting to Redis...[/bold]")
    state.load()
    console.print("[green]Ready[/green]")
    yield


app = FastAPI(
    title="RecSys Serving API",
    description="Online recommendation serving with RQ-VAE semantic IDs + SASRec",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    if request.url.path == "/recommend" and request.method == "POST":
        # Determine path type from response body tag (set by route handler via header)
        path_type = response.headers.get("X-Serving-Path", "warm")
        REQUEST_COUNT.labels(path_type=path_type).inc()
        REQUEST_LATENCY.labels(path_type=path_type).observe(latency)
        response.headers["X-Serving-Latency-Ms"] = f"{latency * 1000:.1f}"
        if path_type == "cache_hit":
            CACHE_HIT_COUNT.inc()
        elif path_type == "cold_start":
            method = response.headers.get("X-Cold-Start-Method", "prefix_fallback")
            COLD_START_COUNT.labels(method=method).inc()

    return response


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Serving-Latency-Ms", "X-Serving-Path", "X-Cold-Start-Method"],
)

app.include_router(router)
app.state.recsys = state
