"""Build the FAISS ANN index from sentence-transformer item embeddings.

This is the offline retrieval artifact. At serving time, a user session is
summarised into a single query embedding (mean of session item embeddings),
then the index returns the top-200 nearest neighbors as retrieval candidates.

We use IndexFlatIP (inner-product / cosine) because sentence-transformer
embeddings are L2-normalised, so inner product = cosine similarity.

Reads  : artifacts/embeddings/item_embeddings.npy
         artifacts/embeddings/item_ids.json
Writes : artifacts/ann/index.faiss
         artifacts/ann/item_ids.json

Run: python -m offline.ann.build
"""

import json
from pathlib import Path

import faiss
import numpy as np
from rich.console import Console

EMB_DIR = Path("artifacts/embeddings")
OUT_DIR = Path("artifacts/ann")

console = Console()


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMB_DIR / "item_embeddings.npy").astype(np.float32)
    with (EMB_DIR / "item_ids.json").open() as f:
        item_ids = json.load(f)

    N, D = embeddings.shape
    console.print(f"[bold]Building FAISS index[/bold]  items={N:,}  dim={D}")

    # Embeddings from generate.py are already L2-normalised (normalize_embeddings=True),
    # so IndexFlatIP gives cosine similarity directly.
    index = faiss.IndexFlatIP(D)
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    with (OUT_DIR / "item_ids.json").open("w") as f:
        json.dump(item_ids, f)

    console.print(f"[green]Index built: {index.ntotal:,} vectors → {OUT_DIR}[/green]")


if __name__ == "__main__":
    build()
