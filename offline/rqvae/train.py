"""Train the RQ-VAE and write semantic IDs for every item.

Collision handling
------------------
A 3-level codebook with 256 codes per level gives 256^3 = 16.7M possible IDs
for a catalog of tens of thousands of items. In practice ~10% of items still
share a 3-level ID because the codebook doesn't distribute perfectly uniformly.

We resolve this with a sequential 4th token (Eugene Yan's approach):
  - Train only 3 levels — no retraining needed
  - After generating codes, detect collision groups
  - For each group of items sharing the same (c0, c1, c2), assign c3 = 0, 1, 2 ...
  - Items with a unique (c0, c1, c2) get c3 = 0
  - Uniqueness is now guaranteed by construction, not by a larger codebook

This is strictly better than adding a 4th learned VQ level because:
  1. No additional training
  2. Guaranteed unique — a 4th codebook can still collide
  3. The 4th token carries no semantic meaning (it's just a disambiguator),
     so there's no point wasting model capacity on it

Reads  : artifacts/embeddings/item_embeddings.npy
         artifacts/embeddings/item_ids.json
Writes : artifacts/rqvae/model.pt            (model weights, always 3 levels)
         artifacts/rqvae/semantic_ids.parquet (item_id → [c0, c1, c2, c3])
         artifacts/rqvae/config.json          (hyperparameters)

Run: python -m offline.rqvae.train
"""

import json
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console

from offline.rqvae.model import RQVAE

EMB_DIR = Path("artifacts/embeddings")
OUT_DIR = Path("artifacts/rqvae")
MLFLOW_URI = "http://localhost:5001"

CONFIG = {
    "hidden_dim": 128,
    "num_codes": 32,  # 32^3 = 32,768 possible 3-tuples for 1,693 items — avoids collapse
    "num_levels": 3,
    "commitment_cost": 2.0,  # higher value forces codebook spread; 0.25 collapses on small catalogs
    "lr": 3e-4,
    "batch_size": 256,
    "epochs": 100,
}

console = Console()


def load_embeddings() -> tuple[torch.Tensor, list[str]]:
    embeddings = np.load(EMB_DIR / "item_embeddings.npy")
    with (EMB_DIR / "item_ids.json").open() as f:
        item_ids = json.load(f)
    console.print(f"Loaded embeddings: {embeddings.shape}")
    return torch.tensor(embeddings, dtype=torch.float32), item_ids


def resolve_collisions(codes: np.ndarray) -> np.ndarray:
    """Append a sequential 4th token to guarantee every item has a unique ID.

    For each unique (c0, c1, c2) prefix:
      - If only one item maps to it: c3 = 0
      - If k items map to it:        c3 = 0, 1, ..., k-1  (insertion order)

    The 4th token is a disambiguator, not a learned codebook entry. Items in
    the same collision group still share a (c0, c1, c2) prefix, preserving the
    hierarchical clustering property for the items that don't collide.

    Returns: (N, 4) int array
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(codes):
        groups[tuple(row.tolist())].append(i)

    out = np.zeros((len(codes), 4), dtype=np.int64)
    out[:, :3] = codes
    for indices in groups.values():
        for seq, idx in enumerate(indices):
            out[idx, 3] = seq

    n_collisions = sum(len(v) for v in groups.values() if len(v) > 1)
    console.print(
        f"  Collision groups: {sum(1 for v in groups.values() if len(v) > 1)}  "
        f"({n_collisions / len(codes):.1%} of items disambiguated by c3)"
    )
    return out


def train(cfg: dict = CONFIG) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings, item_ids = load_embeddings()
    input_dim = embeddings.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    model = RQVAE(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_codes=cfg["num_codes"],
        num_levels=cfg["num_levels"],
        commitment_cost=cfg["commitment_cost"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    loader = DataLoader(
        TensorDataset(embeddings), batch_size=cfg["batch_size"], shuffle=True
    )

    # ── K-means codebook initialization ──────────────────────────────────────
    # Without this, small datasets (< 10k items) collapse — all inputs map to
    # the same nearest code because random initialization places all centroids
    # equidistant from a tight cluster. K-means seeds each level's codebook at
    # actual data centroids so gradient updates start from a spread-out state.
    console.print("[bold]Initialising codebooks with k-means...[/bold]")
    model.eval()
    with torch.no_grad():
        z_all = model.encoder(embeddings.to(device))
        residual = z_all.cpu().numpy()
        for level, quantizer in enumerate(model.quantizers):
            km = MiniBatchKMeans(
                n_clusters=cfg["num_codes"], n_init=5, random_state=level, max_iter=300
            )
            km.fit(residual)
            quantizer.codebook.weight.data = torch.tensor(
                km.cluster_centers_, dtype=torch.float32, device=device
            )
            # compute residual for next level using assigned codes
            assigned = quantizer.codebook(
                torch.tensor(km.labels_, dtype=torch.long, device=device)
            )
            residual = residual - assigned.cpu().numpy()
            used = len(set(km.labels_))
            console.print(
                f"  level {level}: {used}/{cfg['num_codes']} codes used after k-means init"
            )

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="rqvae"):
        mlflow.log_params(cfg)

        console.print(f"[bold]Training RQ-VAE for {cfg['epochs']} epochs[/bold]")
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                _, _, loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            scheduler.step()
            avg_loss = total_loss / len(embeddings)
            if epoch % 10 == 0 or epoch == 1:
                console.print(
                    f"  epoch {epoch:3d}/{cfg['epochs']}  loss={avg_loss:.5f}"
                )
                mlflow.log_metric("loss", avg_loss, step=epoch)

        # ── Generate 3-level codes ────────────────────────────────────────────
        model.eval()
        codes_3 = model.get_codes(embeddings.to(device)).cpu().numpy()  # (N, 3)

        n_unique_3 = len({tuple(r) for r in codes_3})
        collision_rate = 1.0 - n_unique_3 / len(codes_3)
        console.print(f"\n3-level collision rate: {collision_rate:.2%}")
        mlflow.log_metric("collision_rate_3level", collision_rate)

        # ── Resolve collisions with sequential 4th token ──────────────────────
        console.print("Resolving collisions with sequential 4th token...")
        codes_4 = resolve_collisions(codes_3)  # (N, 4) — guaranteed unique

        assert len({tuple(r) for r in codes_4}) == len(codes_4), (
            "resolve_collisions did not produce unique IDs"
        )
        console.print(
            f"[green]All {len(codes_4):,} items have unique 4-token semantic IDs[/green]"
        )

        # ── Save artifacts ────────────────────────────────────────────────────
        torch.save(model.state_dict(), OUT_DIR / "model.pt")

        # config records num_levels=3 (the trained model) and notes that the
        # saved semantic IDs always have 4 columns (c0–c3) where c3 is the
        # sequential disambiguator, not a learned codebook level
        # num_sid_tokens=3: SASRec uses only the 3 semantic levels (c0,c1,c2).
        # c3 is a sequential disambiguator — it has no semantic ordering and
        # its values can exceed num_codes, making it incompatible with the
        # shared token embedding table.
        with (OUT_DIR / "config.json").open("w") as f:
            json.dump({**cfg, "input_dim": input_dim, "num_sid_tokens": 3}, f, indent=2)

        df = pd.DataFrame({"item_id": item_ids})
        for lvl in range(4):
            df[f"c{lvl}"] = codes_4[:, lvl].astype(int)
        df.to_parquet(OUT_DIR / "semantic_ids.parquet", index=False)

        console.print(f"[green]Saved → {OUT_DIR}[/green]")
        mlflow.log_metric("collision_rate_after_resolution", 0.0)
        mlflow.log_artifact(str(OUT_DIR / "semantic_ids.parquet"))
        mlflow.log_artifact(str(OUT_DIR / "config.json"))


if __name__ == "__main__":
    train()
