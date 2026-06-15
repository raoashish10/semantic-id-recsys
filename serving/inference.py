"""SASRec beam search — shared by the online API and the offline precompute step.

This module contains two functions used on every warm-user recommendation path:

  build_input()     — converts a list of item_id strings into a (1, T, L) semantic
                      ID tensor by looking up each item's codes in Redis

  beam_recommend()  — runs SASRec forward pass, expands the top predicted code
                      combinations, and resolves them to real items via Redis
                      reverse-lookup (sid:* keys)

Cold-start counterparts (session length < 3):
  serving/intent.py           — LLM-based session intent inference (Ollama, 200ms timeout)
  serving/intent_retrieval.py — prefix candidate retrieval + cosine re-ranking by intent
  serving/api/routes.py       — orchestrates the full fallback chain
"""

from __future__ import annotations

import torch

from serving.models import RecommendedItem
from serving.store.redis_client import ItemStore
from offline.sasrec.model import SASRec


def build_input(
    session: list[str],
    store: ItemStore,
    num_levels: int,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    """Convert item_ids to a (1, T, L) semantic ID tensor.

    Returns (tensor, num_resolved). Returns (None, 0) if no items resolve.
    """
    code_seq = []
    for item_id in session:
        codes = store.get_codes(item_id)
        if codes is not None:
            code_seq.append(codes)

    if not code_seq:
        return None, 0

    T = min(len(code_seq), max_len)
    inp = torch.zeros(1, T, num_levels, dtype=torch.long, device=device)
    for t, codes in enumerate(code_seq[-T:]):
        for lvl, c in enumerate(codes[:num_levels]):
            inp[0, t, lvl] = c

    return inp, len(code_seq)


def beam_recommend(
    model: SASRec,
    store: ItemStore,
    inp: torch.Tensor,
    num_levels: int,
    top_k: int,
    top_per_level: int = 5,
    exclude_ids: set[str] | None = None,
) -> list[RecommendedItem]:
    """SASRec beam search over semantic ID space → real items.

    Predicts the top-K (c0, c1, c2) prefix combinations, then resolves each
    to a set of real items via the prefix3:{c0}:{c1}:{c2} Redis index. This
    is correct for 3-level prediction where c3 is a sequential disambiguator
    that carries no semantic meaning for the model to predict.
    """
    with torch.no_grad():
        padding_mask = torch.zeros(inp.shape[:2], dtype=torch.bool, device=inp.device)
        logits_per_level = model(inp, padding_mask)

    top_codes: list[list[int]] = [
        logits_per_level[lvl][0, -1].topk(top_per_level).indices.tolist()
        for lvl in range(num_levels)
    ]

    seen_prefixes: set[tuple] = set()
    seen_items: set[str] = set(exclude_ids) if exclude_ids else set()
    results: list[RecommendedItem] = []

    def try_prefix(prefix: tuple[int, ...]) -> None:
        if prefix in seen_prefixes:
            return
        seen_prefixes.add(prefix)
        candidates = store.get_items_by_prefix3(*prefix, limit=top_k * 2)
        for item_id in candidates:
            if item_id in seen_items or len(results) >= top_k:
                continue
            seen_items.add(item_id)
            codes = store.get_codes(item_id)
            if codes is None:
                continue
            results.append(
                RecommendedItem(
                    item_id=item_id,
                    title=store.get_title(item_id),
                    semantic_id=codes,
                )
            )

    # Greedy: top-1 at each level
    try_prefix(tuple(top_codes[lvl][0] for lvl in range(num_levels)))

    # Expand level-0 for broad category diversity
    for c0 in top_codes[0]:
        try_prefix((c0,) + tuple(top_codes[lvl][0] for lvl in range(1, num_levels)))
        if len(results) >= top_k:
            break

    # Expand level-1 for within-category diversity
    for c1 in top_codes[1]:
        try_prefix(
            (top_codes[0][0], c1)
            + tuple(top_codes[lvl][0] for lvl in range(2, num_levels))
        )
        if len(results) >= top_k:
            break

    return results[:top_k]
