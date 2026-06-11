"""FAISS-based ANN retrieval — used for RQ-VAE audit, not live serving.

The serving API uses SASRec beam search (serving/inference.py) for both
precomputed and real-time recommendations. This module exists to support
offline auditing of RQ-VAE embedding quality:

  Audit use case:
    Given an item, retrieve its k nearest neighbors by raw sentence-transformer
    embedding (FAISS), then check whether those neighbors share a semantic ID
    prefix. If they do, the RQ-VAE has learned meaningful quantization — items
    that are similar in continuous embedding space also cluster together in the
    discrete semantic ID tree.

  Example (notebooks/audit_rqvae.ipynb):
    retriever = AnnRetriever()
    neighbors = retriever.retrieve(item_embedding, top_k=10)
    # compare semantic ID prefixes of neighbors vs. the query item
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

ANN_DIR = Path("artifacts/ann")


class AnnRetriever:
    def __init__(
        self,
        index_path: str | Path = ANN_DIR / "index.faiss",
        item_ids_path: str | Path = ANN_DIR / "item_ids.json",
    ):
        self.index = faiss.read_index(str(index_path))
        with open(item_ids_path) as f:
            self.item_ids: list[str] = json.load(f)

    def retrieve(
        self,
        session_embeddings: np.ndarray,  # (T, D) — one embedding per session item
        top_k: int = 200,
    ) -> list[tuple[str, float]]:
        """Return top_k (item_id, cosine_score) candidates for the session.

        The query is the mean of session item embeddings. Items already in the
        session are not filtered here — the route layer handles deduplication.
        """
        query = session_embeddings.mean(axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(query)  # safe no-op if already normalised
        scores, indices = self.index.search(query, top_k)
        return [
            (self.item_ids[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if 0 <= i < len(self.item_ids)
        ]
