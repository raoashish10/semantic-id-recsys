"""Offline evaluation: Recall@K and NDCG@K on the held-out test split.

How it works
------------
For each user in the test split:
  1. Feed their full history (minus the last item) through the trained SASRec model.
  2. Expand the top predicted (c0,c1,c2) prefixes into candidate item_ids via the
     prefix3:{c0}:{c1}:{c2} Redis index (same logic as online serving).
  3. Compare against the true held-out last item.
  4. Accumulate Recall@K and NDCG@K.

Promotion gate
--------------
Pass --gate to fail (exit code 1) when Recall@10 drops below the baseline stored
in the MLflow run tag "baseline_recall_at_10". On first run (no baseline) the
current value is written as the baseline and the gate always passes.

Run:
  python -m offline.evaluate              # eval + log to MLflow
  python -m offline.evaluate --gate       # eval + promote-or-fail
  python -m offline.evaluate --k 20       # use K=20 instead of 10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console

from offline.sasrec.dataset import load_dataset
from offline.sasrec.model import SASRec
from serving.store.redis_client import ItemStore

SEQUENCES = Path("data/processed/sequences.parquet")
SID_PATH = Path("artifacts/rqvae/semantic_ids.parquet")
SASREC_CFG = Path("artifacts/sasrec/config.json")
SASREC_WEIGHTS = Path("artifacts/sasrec/model.pt")
MLFLOW_URI = "http://localhost:5001"

console = Console()


def _expand_prefixes(
    logits_per_level: list[torch.Tensor],
    store: ItemStore,
    num_levels: int,
    top_per_level: int,
    top_k: int,
) -> list[str]:
    """Convert per-level logits to a ranked list of candidate item_ids."""
    top_codes = [
        logits_per_level[lvl][0, -1].topk(top_per_level).indices.tolist()
        for lvl in range(num_levels)
    ]

    seen_prefixes: set[tuple] = set()
    candidates: list[str] = []
    # Use a large srandmember limit to sample the full bucket — the catalog is tiny
    # so this is still fast. Small limit (e.g. top_k=10) gives Recall≈0 by chance.
    fetch_limit = max(top_k * 20, 200)

    def try_prefix(prefix: tuple) -> None:
        if prefix in seen_prefixes:
            return
        seen_prefixes.add(prefix)
        for item_id in store.get_items_by_prefix3(*prefix, limit=fetch_limit):
            if item_id not in candidates:
                candidates.append(item_id)

    # Greedy top-1 first, then diversify across levels
    try_prefix(tuple(top_codes[lvl][0] for lvl in range(num_levels)))
    for c0 in top_codes[0]:
        try_prefix((c0,) + tuple(top_codes[lvl][0] for lvl in range(1, num_levels)))
    for c1 in top_codes[1]:
        try_prefix((top_codes[0][0], c1) + tuple(top_codes[lvl][0] for lvl in range(2, num_levels)))

    return candidates[:top_k]


def _get_predicted_prefixes(
    logits_per_level: list[torch.Tensor],
    num_levels: int,
    top_per_level: int,
) -> set[tuple]:
    """Return the set of (c0,c1,c2) tuples the model would explore in beam search."""
    top_codes = [
        logits_per_level[lvl][0, -1].topk(top_per_level).indices.tolist()
        for lvl in range(num_levels)
    ]
    prefixes = set()
    prefixes.add(tuple(top_codes[lvl][0] for lvl in range(num_levels)))
    for c0 in top_codes[0]:
        prefixes.add((c0,) + tuple(top_codes[lvl][0] for lvl in range(1, num_levels)))
    for c1 in top_codes[1]:
        prefixes.add((top_codes[0][0], c1) + tuple(top_codes[lvl][0] for lvl in range(2, num_levels)))
    return prefixes


@torch.no_grad()
def evaluate(k: int = 10, top_per_level: int = 8) -> dict[str, float]:
    """Run evaluation on the test split. Returns metrics dict."""
    with SASREC_CFG.open() as f:
        cfg = json.load(f)

    num_codes = cfg["num_codes"]
    num_levels = cfg["num_levels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASRec(
        num_codes=num_codes,
        num_levels=num_levels,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(SASREC_WEIGHTS, map_location=device, weights_only=True))
    model.eval()

    _, _, test_ds = load_dataset(SEQUENCES, SID_PATH, cfg["max_len"], num_levels=num_levels)
    store = ItemStore()

    console.print(f"[bold]Evaluating on {len(test_ds):,} test users  K={k}[/bold]")

    # Build item_id → codes lookup from parquet (for ground-truth resolution)
    sid_df = pd.read_parquet(SID_PATH)
    code_cols = [c for c in sid_df.columns if c.startswith("c")][:num_levels]
    id_to_codes = {
        row["item_id"]: tuple(int(row[c]) for c in code_cols)
        for _, row in sid_df.iterrows()
    }
    # Build reverse: 3-level prefix → set of item_ids (for ground truth lookup)
    prefix_to_items: dict[tuple, list[str]] = {}
    for item_id, codes in id_to_codes.items():
        prefix_to_items.setdefault(codes, []).append(item_id)

    # Load the full sequences for ground-truth last items
    seqs_df = pd.read_parquet(SEQUENCES)
    ground_truths: list[str] = []
    for seq in seqs_df["item_ids"]:
        valid = [it for it in seq if it in id_to_codes]
        if len(valid) >= 3:
            ground_truths.append(valid[-1])  # true next item

    n_users = min(len(test_ds), len(ground_truths))
    recall_hits = 0
    ndcg_sum = 0.0
    prefix_hits = 0

    for i in range(n_users):
        inp, _, mask = test_ds[i]
        inp = inp.unsqueeze(0).to(device)   # (1, T, L)
        mask = mask.unsqueeze(0).to(device) # (1, T)

        logits = model(inp, mask)

        gt = ground_truths[i]
        gt_codes = id_to_codes.get(gt)

        # Prefix3 recall: does the model's beam include the target's (c0,c1,c2) cluster?
        if gt_codes is not None:
            predicted_prefixes = _get_predicted_prefixes(logits, num_levels, top_per_level)
            if gt_codes in predicted_prefixes:
                prefix_hits += 1

        # Item-level recall: is the exact item in the expanded candidate list?
        candidates = _expand_prefixes(logits, store, num_levels, top_per_level, k)
        if gt in candidates:
            rank = candidates.index(gt) + 1  # 1-indexed
            recall_hits += 1
            ndcg_sum += 1.0 / math.log2(rank + 1)

    recall = recall_hits / n_users if n_users > 0 else 0.0
    ndcg = ndcg_sum / n_users if n_users > 0 else 0.0
    prefix_recall = prefix_hits / n_users if n_users > 0 else 0.0

    metrics = {
        f"recall_at_{k}": recall,
        f"ndcg_at_{k}": ndcg,
        f"prefix3_recall_at_{k}": prefix_recall,
        "n_test_users": n_users,
    }
    console.print(
        f"  Recall@{k}={recall:.4f}  NDCG@{k}={ndcg:.4f}  "
        f"Prefix3-Recall@{k}={prefix_recall:.4f}  (over {n_users:,} users)"
    )
    return metrics


def run(k: int = 10, gate: bool = False) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)

    metrics = evaluate(k=k)
    recall_key = f"recall_at_{k}"
    current_recall = metrics[recall_key]

    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.log_params({"k": k})
        mlflow.log_metrics(metrics)

        # ── Promotion gate ────────────────────────────────────────────────────
        baseline_tag = f"baseline_{recall_key}"
        baseline_run = _get_latest_baseline(baseline_tag)

        if baseline_run is None:
            # First eval — write current value as baseline
            mlflow.set_tag(baseline_tag, str(current_recall))
            console.print(f"[yellow]First eval run — baseline set to {current_recall:.4f}[/yellow]")
        else:
            baseline_val = float(baseline_run)
            passed = current_recall >= baseline_val * 0.95  # 5% regression tolerance
            status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
            console.print(
                f"Gate {status}  {recall_key}={current_recall:.4f}  "
                f"baseline={baseline_val:.4f}"
            )
            mlflow.set_tag("gate_passed", str(passed))

            if gate and not passed:
                raise SystemExit(
                    f"Promotion gate failed: {recall_key}={current_recall:.4f} "
                    f"< 95% of baseline={baseline_val:.4f}"
                )

            if passed and current_recall > baseline_val:
                mlflow.set_tag(baseline_tag, str(current_recall))
                console.print(f"Baseline updated to {current_recall:.4f}")

        console.print(f"[green]Run logged → {run.info.run_id}[/green]")


def _get_latest_baseline(tag_key: str) -> str | None:
    """Return the tag value from the most recent evaluate run that has it, or None."""
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=f"tags.mlflow.runName = 'evaluate' and tags.`{tag_key}` != ''",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs:
        return runs[0].data.tags.get(tag_key)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Cutoff for Recall@K and NDCG@K")
    parser.add_argument("--gate", action="store_true", help="Fail if Recall@K regresses vs baseline")
    args = parser.parse_args()
    run(k=args.k, gate=args.gate)
