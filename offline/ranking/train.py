"""Train the point-wise MLP ranker.

Training setup
--------------
For each user sequence, we create (user_emb, item_emb, label) triples:
  - user_emb:  mean of the embeddings of items seen so far in the session
  - pos_emb:   embedding of the *next* item the user actually interacted with
  - neg_emb:   embedding of a randomly sampled item from the catalog

Loss: BCE(ranker(user, pos), 1) + BCE(ranker(user, neg), 0)

This trains the ranker to score true next-items higher than random items —
the same objective the FAISS index is trying to achieve at the retrieval step,
but with a learned nonlinear function over the embedding pair rather than
plain cosine similarity.

Reads  : data/processed/sequences.parquet
         artifacts/embeddings/item_embeddings.npy
         artifacts/embeddings/item_ids.json
Writes : artifacts/ranking/model.pt
         artifacts/ranking/config.json

Run: python -m offline.ranking.train
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich.console import Console

from offline.ranking.model import Ranker

SEQUENCES = Path("data/processed/sequences.parquet")
EMB_DIR = Path("artifacts/embeddings")
OUT_DIR = Path("artifacts/ranking")
MLFLOW_URI = "http://localhost:5001"

CONFIG = {
    "hidden_dim": 256,
    "lr": 1e-3,
    "batch_size": 512,
    "epochs": 20,
    "neg_per_pos": 1,  # negative samples per positive example
}

console = Console()


class RankingDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[str]],
        id_to_emb: dict[str, np.ndarray],
        all_item_ids: list[str],
        neg_per_pos: int = 1,
    ):
        self.id_to_emb = id_to_emb
        self.all_item_ids = all_item_ids
        self.neg_per_pos = neg_per_pos
        self.examples: list[tuple[np.ndarray, np.ndarray, float]] = []

        rng = np.random.default_rng(42)
        for seq in sequences:
            valid = [it for it in seq if it in id_to_emb]
            if len(valid) < 2:
                continue
            # Use leave-last-2-out for train (mirror SASRec split)
            items = valid[:-2]
            for t in range(1, len(items)):
                user_emb = np.stack([id_to_emb[it] for it in items[:t]]).mean(axis=0)
                pos_emb = id_to_emb[items[t]]
                self.examples.append((user_emb, pos_emb, 1.0))
                for _ in range(neg_per_pos):
                    neg_id = rng.choice(all_item_ids)
                    neg_emb = id_to_emb[neg_id]
                    self.examples.append((user_emb, neg_emb, 0.0))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        u, i, label = self.examples[idx]
        return (
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(i, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def train(cfg: dict = CONFIG) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings = np.load(EMB_DIR / "item_embeddings.npy").astype(np.float32)
    with (EMB_DIR / "item_ids.json").open() as f:
        item_ids = json.load(f)
    id_to_emb = dict(zip(item_ids, embeddings))
    embedding_dim = embeddings.shape[1]

    seqs_df = pd.read_parquet(SEQUENCES)
    sequences = seqs_df["item_ids"].tolist()

    console.print("[bold]Building ranking dataset[/bold]")
    ds = RankingDataset(sequences, id_to_emb, item_ids, cfg["neg_per_pos"])
    console.print(f"  {len(ds):,} training examples")

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    model = Ranker(embedding_dim=embedding_dim, hidden_dim=cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="ranker"):
        mlflow.log_params({**cfg, "embedding_dim": embedding_dim})

        console.print(f"[bold]Training Ranker for {cfg['epochs']} epochs[/bold]")
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for user_emb, item_emb, labels in loader:
                user_emb = user_emb.to(device)
                item_emb = item_emb.to(device)
                labels = labels.to(device)
                logits = model(user_emb, item_emb)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg = total_loss / len(loader)
            if epoch % 5 == 0 or epoch == 1:
                console.print(f"  epoch {epoch:3d}/{cfg['epochs']}  loss={avg:.5f}")
                mlflow.log_metric("loss", avg, step=epoch)

    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    full_cfg = {**cfg, "embedding_dim": embedding_dim}
    with (OUT_DIR / "config.json").open("w") as f:
        json.dump(full_cfg, f, indent=2)
    console.print(f"[green]Ranker saved → {OUT_DIR}[/green]")


if __name__ == "__main__":
    train()
