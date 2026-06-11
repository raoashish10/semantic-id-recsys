"""Download Amazon Reviews 2023 — All_Beauty category.

Uses the HuggingFace datasets mirror so no manual URL handling is needed.
Run: python -m data.download
"""

import json
from pathlib import Path

from datasets import load_dataset
from rich.console import Console

RAW = Path("data/raw")
console = Console()


def download_meta() -> None:
    """Product metadata: title, description, price, category."""
    console.print("[bold]Downloading product metadata...[/bold]")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_All_Beauty",
        split="full",
        trust_remote_code=True,
    )
    out = RAW / "meta.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    console.print(f"[green]Saved {len(ds):,} products → {out}[/green]")


def download_interactions() -> None:
    """User–item interactions: user_id, item_id, rating, timestamp."""
    console.print("[bold]Downloading user interactions...[/bold]")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "0core_rating_only_All_Beauty",
        split="full",
        trust_remote_code=True,
    )
    out = RAW / "interactions.jsonl"
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    console.print(f"[green]Saved {len(ds):,} interactions → {out}[/green]")


if __name__ == "__main__":
    download_meta()
    download_interactions()
