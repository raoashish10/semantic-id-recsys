"""Intent-based candidate retrieval for cold-start recommendations.

Takes an IntentResult (from serving/intent.py), pulls candidate items from
the Redis prefix index for each predicted (c0, c1), then re-ranks them by
weighted cosine similarity between the LLM-inferred intent text and each
candidate's stored sentence-transformer embedding.

The sentence-transformer model is loaded lazily on first call and reused for
all subsequent calls — it is the same model used in the offline pipeline
(all-MiniLM-L6-v2, 384-dim, L2-normalized outputs).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from serving.intent import IntentResult
from serving.models import RecommendedItem
from serving.store.redis_client import ItemStore

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; normalizes both vectors to handle non-unit inputs."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))


def intent_based_recommend(
    intent_result: IntentResult,
    session_item_ids: list[str],
    store: ItemStore,
    top_k: int,
) -> list[RecommendedItem]:
    """Retrieve and rank candidates using LLM-inferred intent.

    For each predicted prefix in intent_result, fetches up to top_k*3 candidates
    from the Redis prefix set, scores each as:

        score = prefix.weight * cosine_similarity(intent_embedding, item_embedding)

    where intent_embedding is the all-MiniLM-L6-v2 encoding of intent_result.intent
    and item_embedding is the precomputed feat:{item_id} blob from Redis.

    Items already in the session are excluded. Returns top_k items sorted by
    descending score.

    Args:
        intent_result:    output from infer_session_intent()
        session_item_ids: item IDs already seen by the user (filtered from results)
        store:            ItemStore for all Redis lookups
        top_k:            number of recommendations to return
    """
    intent_emb: np.ndarray = _get_embed_model().encode(
        [intent_result.intent],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    session_set = set(session_item_ids)
    seen: set[str] = set()
    scored: list[tuple[float, RecommendedItem]] = []

    for prefix in intent_result.predicted_prefixes:
        candidates = store.get_items_by_prefix(prefix.c0, prefix.c1, limit=top_k * 3)
        for item_id in candidates:
            if item_id in session_set or item_id in seen:
                continue
            seen.add(item_id)

            item_emb = store.get_item_features(item_id)
            if item_emb is None:
                continue

            codes = store.get_codes(item_id)
            if codes is None:
                continue

            score = prefix.weight * _cosine_sim(intent_emb, item_emb)
            scored.append(
                (
                    score,
                    RecommendedItem(
                        item_id=item_id,
                        title=store.get_title(item_id),
                        semantic_id=codes,
                    ),
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
