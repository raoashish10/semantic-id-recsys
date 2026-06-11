"""Train SASRec on sequences rewritten as RQ-VAE semantic IDs.

Reads  : data/processed/sequences.parquet
         artifacts/rqvae/semantic_ids.parquet
         artifacts/rqvae/config.json
Writes : artifacts/sasrec/model.pt
         artifacts/sasrec/config.json

Run: python -m offline.sasrec.train
"""

import json
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import track

from offline.sasrec.dataset import load_dataset
from offline.sasrec.model import SASRec

SEQUENCES = Path("data/processed/sequences.parquet")
SID_PATH = Path("artifacts/rqvae/semantic_ids.parquet")
RQ_CFG = Path("artifacts/rqvae/config.json")
OUT_DIR = Path("artifacts/sasrec")
MLFLOW_URI = "http://localhost:5001"

CONFIG = {
    "hidden_dim": 128,
    "num_heads": 2,
    "num_layers": 2,
    "max_len": 50,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 256,
    "epochs": 30,
}

console = Console()


def cross_entropy_loss(
    logits_per_level: list[torch.Tensor],  # L × (B, T, num_codes)
    targets: torch.Tensor,                 # (B, T, L)
    padding_mask: torch.Tensor,            # (B, T) True = padding
) -> torch.Tensor:
    """Cross-entropy averaged over non-padding positions and all levels."""
    B, T, L = targets.shape
    total = torch.tensor(0.0, device=targets.device)
    valid = (~padding_mask).float().sum()

    for lvl, logits in enumerate(logits_per_level):
        # logits: (B, T, num_codes) → (B*T, num_codes)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets[:, :, lvl].reshape(-1),
            reduction="none",
        )
        # Zero out padding positions
        loss = loss * (~padding_mask).reshape(-1).float()
        total = total + loss.sum() / (valid + 1e-8)

    return total / L


@torch.no_grad()
def evaluate(model: SASRec, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute loss and Hit@10 on the validation set."""
    model.eval()
    total_loss, total_hit, total_count = 0.0, 0.0, 0

    for inp, tgt, mask in loader:
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        logits = model(inp, mask)
        loss = cross_entropy_loss(logits, tgt, mask)
        total_loss += loss.item()

        # Hit@10: check if the true next item's codes are all in the top-10 per level
        # (simplified: we check the last non-padding position only)
        seq_lens = (~mask).sum(dim=-1) - 1  # index of last real token
        for b, last in enumerate(seq_lens):
            if last < 0:
                continue
            hit = True
            for lvl, level_logits in enumerate(logits):
                top10 = level_logits[b, last].topk(10).indices
                if tgt[b, last, lvl] not in top10:
                    hit = False
                    break
            total_hit += int(hit)
            total_count += 1

    return {
        "loss": total_loss / len(loader),
        "hit@10": total_hit / max(total_count, 1),
    }


def train(cfg: dict = CONFIG) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with RQ_CFG.open() as f:
        rq_cfg = json.load(f)

    num_codes = rq_cfg["num_codes"]
    # num_sid_tokens is the actual token count per item (always 4: 3 learned + 1
    # sequential disambiguator). num_levels describes the trained RQ-VAE model (3).
    num_sid_tokens = rq_cfg.get("num_sid_tokens", rq_cfg["num_levels"])

    console.print(f"[bold]Loading datasets[/bold]  (num_codes={num_codes}, num_sid_tokens={num_sid_tokens})")
    train_ds, val_ds, _ = load_dataset(SEQUENCES, SID_PATH, cfg["max_len"])
    console.print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")

    model = SASRec(
        num_codes=num_codes,
        num_levels=num_sid_tokens,
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="sasrec"):
        mlflow.log_params({**cfg, "num_codes": num_codes, "num_levels": num_sid_tokens})

        best_hit = 0.0
        console.print(f"[bold]Training SASRec for {cfg['epochs']} epochs[/bold]")

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0

            for inp, tgt, mask in train_loader:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                logits = model(inp, mask)
                loss = cross_entropy_loss(logits, tgt, mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)

            if epoch % 5 == 0 or epoch == 1:
                metrics = evaluate(model, val_loader, device)
                console.print(
                    f"  epoch {epoch:3d}  train_loss={avg_loss:.4f}  "
                    f"val_loss={metrics['loss']:.4f}  hit@10={metrics['hit@10']:.4f}"
                )
                mlflow.log_metrics(
                    {"train_loss": avg_loss, **{f"val_{k}": v for k, v in metrics.items()}},
                    step=epoch,
                )
                if metrics["hit@10"] > best_hit:
                    best_hit = metrics["hit@10"]
                    torch.save(model.state_dict(), OUT_DIR / "model.pt")

        full_cfg = {
            **cfg,
            "num_codes": num_codes,
            "num_levels": num_sid_tokens,  # 4 tokens per item (3 learned + 1 disambiguator)
            "best_hit@10": best_hit,
        }
        with (OUT_DIR / "config.json").open("w") as f:
            json.dump(full_cfg, f, indent=2)

        mlflow.log_artifact(str(OUT_DIR / "config.json"))
        console.print(f"[green]Best Hit@10: {best_hit:.4f}  Saved → {OUT_DIR}[/green]")


if __name__ == "__main__":
    train()
