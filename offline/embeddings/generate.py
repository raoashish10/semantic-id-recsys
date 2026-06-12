"""Generate sentence-transformer embeddings for catalog items.

Only embeds items that appear in the processed interaction sequences —
embedding the full Amazon catalog (112k items) is unnecessary and slow on
CPU since the RQ-VAE, SASRec, and feature store only use the filtered set.

Reads  : data/processed/items.parquet
         data/processed/sequences.parquet  (to find the active item set)
Writes : artifacts/embeddings/item_embeddings.npy   (float32, shape [N, D])
         artifacts/embeddings/item_ids.json          (list of item_id strings, same order)

Run: python -m offline.embeddings.generate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sentence_transformers import SentenceTransformer

ITEMS = Path("data/processed/items.parquet")
SEQUENCES = Path("data/processed/sequences.parquet")
OUT_DIR = Path("artifacts/embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

console = Console()


def generate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only embed items that appear in the interaction sequences
    seqs = pd.read_parquet(SEQUENCES)
    active_ids: set[str] = set()
    for item_ids in seqs["item_ids"]:
        active_ids.update(item_ids)

    all_items = pd.read_parquet(ITEMS)
    items = all_items[all_items["item_id"].isin(active_ids)].reset_index(drop=True)
    console.print(
        f"[bold]Generating embeddings for {len(items):,} catalog items[/bold]"
        f"  ({len(all_items):,} total, {len(all_items) - len(items):,} skipped — not in sequences)"
    )
    console.print(f"  model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        items["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(OUT_DIR / "item_embeddings.npy", embeddings.astype(np.float32))
    with (OUT_DIR / "item_ids.json").open("w") as f:
        json.dump(items["item_id"].tolist(), f)

    console.print(f"[green]Saved embeddings {embeddings.shape} → {OUT_DIR}[/green]")


if __name__ == "__main__":
    generate()
