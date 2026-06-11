"""Precompute recommendations for every user and store them in Redis.

Runs as the final step of the offline pipeline (after sasrec training).
At serving time, GET /recommend checks this cache first; a miss falls back
to real-time inference and writes the result back here.

Reads  : data/processed/sequences.parquet  (user_id → item_id history)
         artifacts/sasrec/model.pt + config.json
         artifacts/rqvae/config.json
Uses   : Redis (must be running)

Run: python -m offline.precompute
"""

import json
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.progress import track

from offline.sasrec.model import SASRec
from serving.inference import beam_recommend, build_input
from serving.store.redis_client import ItemStore, REC_TTL

SEQUENCES = Path("data/processed/sequences.parquet")
SASREC_CFG = Path("artifacts/sasrec/config.json")
SASREC_WEIGHTS = Path("artifacts/sasrec/model.pt")

TOP_K = 50   # precompute more than typical top_k so the cache stays useful
             # even if the caller asks for top_k=20 or top_k=50
BATCH_SIZE = 128

console = Console()


def precompute(top_k: int = TOP_K, ttl: int = REC_TTL) -> None:
    # ── Load model ────────────────────────────────────────────────────────────
    with SASREC_CFG.open() as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    model = SASRec(
        num_codes=cfg["num_codes"],
        num_levels=cfg["num_levels"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(SASREC_WEIGHTS, map_location=device))
    model.eval()

    num_levels = cfg["num_levels"]
    max_len = cfg["max_len"]

    store = ItemStore()
    if not store.ping():
        raise RuntimeError("Redis not reachable — run `make up` first")

    # ── Load user sequences ───────────────────────────────────────────────────
    seqs_df = pd.read_parquet(SEQUENCES)
    console.print(f"[bold]Precomputing recs for {len(seqs_df):,} users[/bold]  top_k={top_k}  ttl={ttl}s")

    hits, skipped = 0, 0

    for _, row in track(seqs_df.iterrows(), total=len(seqs_df), description="Precomputing"):
        user_id = row["user_id"]
        session = row["item_ids"]

        inp, n_resolved = build_input(session, store, num_levels, max_len, device)
        if inp is None:
            skipped += 1
            continue

        recs = beam_recommend(model, store, inp, num_levels, top_k)
        if not recs:
            skipped += 1
            continue

        store.set_user_recs(user_id, [r.model_dump() for r in recs], ttl=ttl)
        hits += 1

    console.print(
        f"[green]Done: {hits:,} users indexed, {skipped:,} skipped (no catalog overlap)[/green]"
    )


if __name__ == "__main__":
    precompute()
