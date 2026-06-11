"""FastAPI serving application.

Start: uvicorn serving.api.main:app --reload --port 8000

The app loads the SASRec model once at startup and keeps it in memory.
The Redis item store is used for fast item ↔ semantic ID lookups.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from rich.console import Console

from serving.api.routes import router
from serving.api.state import AppState

console = Console()
state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    console.print("[bold]Loading model and connecting to Redis...[/bold]")
    state.load()
    console.print("[green]Ready[/green]")
    yield
    # Cleanup (if needed) goes here


app = FastAPI(
    title="RecSys Serving API",
    description="Online recommendation serving with RQ-VAE semantic IDs + SASRec",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.state.recsys = state
