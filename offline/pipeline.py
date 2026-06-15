"""Prefect offline pipeline — runs all steps in order.

Usage
-----
  python -m offline.pipeline              # run end-to-end
  python -m offline.pipeline --from rqvae # start from a specific step

Steps
-----
  download   → raw data
  preprocess → cleaned parquet files
  embeddings → item_embeddings.npy
  rqvae      → semantic_ids.parquet + model.pt
  sasrec     → sasrec model.pt
  ann        → FAISS index + item_ids.json
  ranking    → MLP ranker model.pt
  evaluate   → Recall@K / NDCG@K logged to MLflow; fails if below baseline (gate)
  index      → Redis: semantic IDs + feature store + prefix index populated
  precompute → recs:{user_id} keys written to Redis via SASRec beam search
"""

import argparse
from pathlib import Path

from prefect import flow, task
from rich.console import Console

console = Console()

STEPS = [
    "download",
    "preprocess",
    "embeddings",
    "rqvae",
    "sasrec",
    "ann",
    "ranking",
    "evaluate",
    "index",
    "precompute",
]


@task(name="download")
def download_task() -> None:
    from data.download import download_meta, download_interactions

    download_meta()
    download_interactions()


@task(name="preprocess")
def preprocess_task() -> None:
    from data.preprocess import build_items, build_sequences

    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    build_items().to_parquet(processed / "items.parquet", index=False)
    build_sequences().to_parquet(processed / "sequences.parquet", index=False)


@task(name="embeddings")
def embeddings_task() -> None:
    from offline.embeddings.generate import generate

    generate()


@task(name="rqvae")
def rqvae_task() -> None:
    from offline.rqvae.train import train

    train()


@task(name="sasrec")
def sasrec_task() -> None:
    from offline.sasrec.train import train

    train()


@task(name="ann")
def ann_task() -> None:
    from offline.ann.build import build

    build()


@task(name="ranking")
def ranking_task() -> None:
    from offline.ranking.train import train

    train()


@task(name="precompute")
def precompute_task() -> None:
    from offline.precompute import precompute

    precompute()


@task(name="evaluate")
def evaluate_task() -> None:
    from offline.evaluate import run

    run(k=10, gate=True)


@task(name="index")
def index_task() -> None:
    """Populate Redis with:
    - semantic ID ↔ item_id mappings  (for SASRec beam search path)
    - prefix:{c0}:{c1} sets           (for cold-start fallback)
    - item embeddings in feature store (for FAISS audit use)
    """
    import json
    import numpy as np
    import pandas as pd
    from serving.store.redis_client import ItemStore

    store = ItemStore()
    sid_df = pd.read_parquet("artifacts/rqvae/semantic_ids.parquet")
    items_df = pd.read_parquet("data/processed/items.parquet")

    embeddings = np.load("artifacts/embeddings/item_embeddings.npy").astype(np.float32)
    with open("artifacts/embeddings/item_ids.json") as f:
        emb_item_ids = json.load(f)
    id_to_emb = dict(zip(emb_item_ids, embeddings))

    meta = items_df.set_index("item_id")[["title"]].to_dict("index")
    code_cols = [c for c in sid_df.columns if c.startswith("c")]

    count = 0
    for _, row in sid_df.iterrows():
        codes = tuple(int(row[c]) for c in code_cols)
        item_id = row["item_id"]
        title = meta.get(item_id, {}).get("title", "")
        store.set_item(item_id, codes, title)
        store.add_to_prefix_index(
            item_id, codes
        )  # prefix:{c0}:{c1}   — cold-start (c0,c1)
        store.add_to_prefix3_index(
            item_id, codes
        )  # prefix3:{c0}:{c1}:{c2} — SASRec beam search
        if item_id in id_to_emb:
            store.set_item_features(item_id, id_to_emb[item_id])
        count += 1

    console.print(
        f"[green]Indexed {count:,} items (semantic IDs + feature store + prefix index) in Redis[/green]"
    )


@flow(name="recsys-offline-pipeline")
def pipeline(start_from: str = "download") -> None:
    idx = STEPS.index(start_from)
    active = STEPS[idx:]

    console.print(f"[bold]Running steps: {' → '.join(active)}[/bold]")

    if "download" in active:
        download_task()
    if "preprocess" in active:
        preprocess_task()
    if "embeddings" in active:
        embeddings_task()
    if "rqvae" in active:
        rqvae_task()
    if "sasrec" in active:
        sasrec_task()
    if "ann" in active:
        ann_task()
    if "ranking" in active:
        ranking_task()
    if "evaluate" in active:
        evaluate_task()
    if "index" in active:
        index_task()
    if "precompute" in active:
        precompute_task()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start_from", default="download", choices=STEPS)
    args = parser.parse_args()
    pipeline(start_from=args.start_from)
