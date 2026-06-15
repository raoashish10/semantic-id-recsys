"""Preprocess raw Amazon data into clean parquet files.

Outputs
-------
data/processed/items.parquet     — item_id, title, description (text for embeddings)
data/processed/sequences.parquet — user_id, item_ids (sorted by timestamp, 5-core filtered)

Run: python -m data.preprocess
"""

import json
from pathlib import Path

import pandas as pd
from rich.console import Console

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
console = Console()

MIN_INTERACTIONS = 3  # 3-core filter: drop users/items with fewer interactions
# All_Beauty is a sparse category (75th-percentile user has 1 interaction).
# 5-core collapses the dataset to ~50 users; 3-core keeps 1,500+ users
# all with ≥3 interactions, which is the minimum the dataset loader needs.
MIN_RATING = 4  # only treat ratings >= 4 as positive interactions


def build_items() -> pd.DataFrame:
    console.print("[bold]Building items table...[/bold]")
    rows = []
    with (RAW / "meta.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            title = r.get("title", "").strip()
            # Flatten description list if present
            desc_raw = r.get("description", [])
            desc = " ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw)
            if not title:
                continue
            rows.append(
                {"item_id": r["parent_asin"], "title": title, "description": desc}
            )

    df = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    # Concatenate title + description as the text field for embedding
    df["text"] = df["title"] + ". " + df["description"].str[:256]
    console.print(f"  {len(df):,} items")
    return df


def build_sequences() -> pd.DataFrame:
    console.print("[bold]Building interaction sequences...[/bold]")
    rows = []
    with (RAW / "interactions.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "user_id": r["user_id"],
                    "item_id": r["parent_asin"],
                    "rating": float(r.get("rating", 0)),
                    "timestamp": r["timestamp"],
                }
            )

    df = pd.DataFrame(rows)

    # Keep only positive interactions — a 1-star review is not a signal to
    # recommend similar items
    before = len(df)
    df = df[df["rating"] >= MIN_RATING]
    console.print(
        f"  {before - len(df):,} interactions dropped (rating < {MIN_RATING})"
    )

    # 5-core filter: iteratively drop until all users and items meet the threshold
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index
        filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(filtered) == len(df):
            break
        df = filtered

    # Build sorted sequences per user
    df = df.sort_values(["user_id", "timestamp"])
    sequences = (
        df.groupby("user_id")["item_id"]
        .apply(list)
        .reset_index()
        .rename(columns={"item_id": "item_ids"})
    )
    console.print(
        f"  {len(sequences):,} users, {df['item_id'].nunique():,} items after {MIN_INTERACTIONS}-core filter"
    )
    return sequences


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)

    items = build_items()
    items.to_parquet(PROCESSED / "items.parquet", index=False)
    console.print(f"[green]→ {PROCESSED}/items.parquet[/green]")

    sequences = build_sequences()
    sequences.to_parquet(PROCESSED / "sequences.parquet", index=False)
    console.print(f"[green]→ {PROCESSED}/sequences.parquet[/green]")
