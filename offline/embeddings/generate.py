"""Generate sentence-transformer embeddings for every item.

Reads  : data/processed/items.parquet
Writes : artifacts/embeddings/item_embeddings.npy   (float32, shape [N, D])
         artifacts/embeddings/item_ids.json          (list of item_id strings, same order)

Run: python -m offline.embeddings.generate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer

ITEMS = Path("data/processed/items.parquet")
OUT_DIR = Path("artifacts/embeddings")

# all-MiniLM-L6-v2: 384-dim, fast on CPU, good quality for product text
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

console = Console()


def generate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(ITEMS)
    console.print(f"[bold]Generating embeddings for {len(items):,} items[/bold]")
    console.print(f"  model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = items["text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit norm — cosine similarity becomes dot product
        convert_to_numpy=True,
    )

    np.save(OUT_DIR / "item_embeddings.npy", embeddings.astype(np.float32))
    with (OUT_DIR / "item_ids.json").open("w") as f:
        json.dump(items["item_id"].tolist(), f)

    console.print(f"[green]Saved embeddings {embeddings.shape} → {OUT_DIR}[/green]")


if __name__ == "__main__":
    generate()
