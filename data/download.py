"""Download Amazon Reviews 2023 — All_Beauty category.

Uses huggingface_hub to download the raw JSONL files directly, bypassing the
dataset loading script which datasets>=3.0 no longer supports.

Run: python -m data.download
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

RAW = Path("data/raw")
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

console = Console()


def _download(repo_path: str, dest: Path) -> None:
    if dest.exists():
        console.print(f"  [dim]Already exists: {dest}[/dim]")
        return
    console.print(f"  Downloading {repo_path} ...")
    tmp = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="dataset",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp, dest)
    console.print(f"  [green]→ {dest}[/green]")


def download_meta() -> None:
    """Product metadata: parent_asin, title, description, price, category."""
    console.print("[bold]Downloading product metadata...[/bold]")
    _download("raw/meta_categories/meta_All_Beauty.jsonl", RAW / "meta.jsonl")


def download_interactions() -> None:
    """User-item interactions: user_id, parent_asin, rating, timestamp."""
    console.print("[bold]Downloading user interactions...[/bold]")
    _download("raw/review_categories/All_Beauty.jsonl", RAW / "interactions.jsonl")


if __name__ == "__main__":
    download_meta()
    download_interactions()
