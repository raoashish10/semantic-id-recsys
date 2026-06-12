"""Dataset for SASRec training with semantic IDs.

Each training example is a (input_sequence, target_sequence) pair where:
  - input_sequence: semantic IDs of items 0..T-1
  - target_sequence: semantic IDs of items 1..T   (shifted by one)

Sequences are padded on the left to max_len so all items in a batch have
the same shape. Left-padding is standard for sequential recommenders —
the model sees real items at the right end and padding at the left.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    """
    Args
        sequences:    list of item_id lists (sorted chronologically)
        id_to_codes:  dict mapping item_id → np.ndarray of shape (num_levels,)
        max_len:      maximum sequence length (left-padded)
        mode:         'train' uses all but last 2 items; 'val' uses last-but-one;
                      'test' uses the last item as target
    """

    def __init__(
        self,
        sequences: list[list[str]],
        id_to_codes: dict[str, np.ndarray],
        max_len: int = 50,
        mode: str = "train",
    ):
        self.max_len = max_len
        self.mode = mode
        self.num_levels = next(iter(id_to_codes.values())).shape[0]
        self.examples: list[tuple[np.ndarray, np.ndarray]] = []

        for seq in sequences:
            # Keep only items for which we have semantic IDs
            valid = [it for it in seq if it in id_to_codes]
            if len(valid) < 3:  # need at least 2 input + 1 target
                continue

            if mode == "train":
                items = valid[:-2]
            elif mode == "val":
                items = valid[:-1]
            else:  # test
                items = valid

            if len(items) < 2:
                continue

            codes_arr = np.stack([id_to_codes[it] for it in items])  # (T, L)
            # input: all but last; target: all but first (shifted)
            self.examples.append((codes_arr[:-1], codes_arr[1:]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp, tgt = self.examples[idx]  # (T, L), (T, L)
        T = inp.shape[0]

        # Left-pad to max_len
        pad_len = max(0, self.max_len - T)
        inp_padded = np.zeros((self.max_len, self.num_levels), dtype=np.int64)
        tgt_padded = np.zeros((self.max_len, self.num_levels), dtype=np.int64)
        padding_mask = np.ones(self.max_len, dtype=bool)  # True = padding

        if T > 0:
            actual = min(T, self.max_len)
            inp_padded[pad_len:pad_len + actual] = inp[-actual:]
            tgt_padded[pad_len:pad_len + actual] = tgt[-actual:]
            padding_mask[pad_len:pad_len + actual] = False

        return (
            torch.tensor(inp_padded, dtype=torch.long),  # (max_len, L)
            torch.tensor(tgt_padded, dtype=torch.long),  # (max_len, L)
            torch.tensor(padding_mask, dtype=torch.bool),  # (max_len,)
        )


def load_dataset(
    sequences_path: Path,
    semantic_ids_path: Path,
    max_len: int = 50,
    num_levels: int | None = None,
) -> tuple[SeqDataset, SeqDataset, SeqDataset]:
    """Load train/val/test splits from parquet files.

    Args:
        num_levels: if set, truncate semantic ID codes to this many levels.
                    Use this to drop the c3 disambiguator (which is not a
                    learned codebook token and may exceed the embedding vocab).
    """
    seqs_df = pd.read_parquet(sequences_path)
    ids_df = pd.read_parquet(semantic_ids_path)

    # Build item_id → codes mapping, optionally truncating to num_levels
    code_cols = [c for c in ids_df.columns if c.startswith("c")]
    if num_levels is not None:
        code_cols = code_cols[:num_levels]

    id_to_codes = {
        row["item_id"]: row[code_cols].values.astype(np.int64)
        for _, row in ids_df.iterrows()
    }

    sequences = seqs_df["item_ids"].tolist()

    train_ds = SeqDataset(sequences, id_to_codes, max_len, mode="train")
    val_ds = SeqDataset(sequences, id_to_codes, max_len, mode="val")
    test_ds = SeqDataset(sequences, id_to_codes, max_len, mode="test")

    return train_ds, val_ds, test_ds
