"""Download Amazon Reviews 2023 — Beauty_and_Personal_Care category.

Uses huggingface_hub to download the raw JSONL files directly, bypassing the
dataset loading script which datasets>=3.0 no longer supports.

Run: python -m data.download
"""

from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TimeRemainingColumn

RAW = Path("data/raw")
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
HF_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

# Cap interactions at 3M lines — enough for ~30K items after 3-core filter,
# well within the 2.9GB free space budget.
MAX_INTERACTION_LINES = 3_000_000

console = Console()


def _stream_download(url: str, dest: Path, max_lines: int | None = None) -> None:
    if dest.exists():
        console.print(f"  [dim]Already exists: {dest}[/dim]")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"  Streaming {url} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f, Progress(
            BarColumn(), DownloadColumn(), TimeRemainingColumn(), console=console
        ) as progress:
            task = progress.add_task("download", total=None)
            lines_written = 0
            for chunk in r.iter_lines():
                if max_lines and lines_written >= max_lines:
                    break
                f.write(chunk + b"\n")
                lines_written += 1
                if lines_written % 100_000 == 0:
                    progress.advance(task, 100_000)
    console.print(f"  [green]→ {dest} ({lines_written:,} lines)[/green]")


def download_meta() -> None:
    """Product metadata: parent_asin, title, description, price, category."""
    console.print("[bold]Downloading product metadata...[/bold]")
    _stream_download(
        f"{HF_BASE}/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
        RAW / "meta.jsonl",
    )


def download_interactions() -> None:
    """User-item interactions: user_id, parent_asin, rating, timestamp."""
    console.print("[bold]Downloading user interactions...[/bold]")
    _stream_download(
        f"{HF_BASE}/raw/review_categories/Beauty_and_Personal_Care.jsonl",
        RAW / "interactions.jsonl",
        max_lines=MAX_INTERACTION_LINES,
    )


if __name__ == "__main__":
    download_meta()
    download_interactions()
