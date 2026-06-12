"""Redis-backed item store for the online serving layer.

Key schema
----------
  item:{item_id}          → JSON  {"codes": [c0,c1,...], "title": "..."}   no TTL
  sid:{c0}:{c1}:{...}     → item_id string                                  no TTL
  feat:{item_id}          → bytes raw float32 embedding (feature store)     no TTL
  prefix:{c0}:{c1}        → Redis Set of item_ids sharing that prefix       no TTL
  recs:{user_id}          → JSON  [{"item_id":..,"title":..,"semantic_id":..}, ...]
                                                                             TTL = REC_TTL_SECONDS

Item/feature/prefix mappings never expire — stable between pipeline runs.
User rec keys expire after REC_TTL_SECONDS (default 24h). A miss triggers
real-time SASRec inference (warm users) or semantic ID prefix lookup (cold
start users with short sessions), and the result is written back here.

Two Redis connections are kept: one with decode_responses=True for JSON/string
keys, and one without for binary embedding blobs (feat:* keys).
"""

import json
import os
from typing import Optional

import numpy as np
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REC_TTL = int(os.getenv("REC_TTL_SECONDS", 86400))  # 24 hours


class ItemStore:
    def __init__(self, url: str = REDIS_URL):
        self._r = redis.from_url(url, decode_responses=True)   # strings / JSON
        self._rb = redis.from_url(url, decode_responses=False) # binary blobs

    # ── Write ─────────────────────────────────────────────────────────────────

    def set_item(self, item_id: str, codes: tuple[int, ...], title: str = "") -> None:
        """Index a single item in both directions."""
        forward_key = f"item:{item_id}"
        self._r.set(forward_key, json.dumps({"codes": list(codes), "title": title}))

        reverse_key = self._sid_key(codes)
        self._r.set(reverse_key, item_id)

    def _sid_key(self, codes: tuple[int, ...]) -> str:
        return "sid:" + ":".join(str(c) for c in codes)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_codes(self, item_id: str) -> Optional[list[int]]:
        """Look up semantic ID codes for an item_id. None if not found."""
        raw = self._r.get(f"item:{item_id}")
        if raw is None:
            return None
        return json.loads(raw)["codes"]

    def get_item_id(self, codes: tuple[int, ...]) -> Optional[str]:
        """Reverse lookup: semantic ID → item_id. None if not found."""
        return self._r.get(self._sid_key(codes))

    def get_title(self, item_id: str) -> str:
        raw = self._r.get(f"item:{item_id}")
        if raw is None:
            return ""
        return json.loads(raw).get("title", "")

    # ── Precomputed user recommendations ─────────────────────────────────────

    def set_user_recs(self, user_id: str, recs: list[dict], ttl: int = REC_TTL) -> None:
        """Store precomputed recommendations for a user with a TTL."""
        self._r.set(f"recs:{user_id}", json.dumps(recs), ex=ttl)

    def get_user_recs(self, user_id: str) -> Optional[list[dict]]:
        """Fetch precomputed recs. Returns None on miss (expired or never computed)."""
        raw = self._r.get(f"recs:{user_id}")
        if raw is None:
            return None
        return json.loads(raw)

    # ── Feature store (raw embeddings for retrieval + ranking) ───────────────

    # ── Semantic ID prefix index (cold-start fallback) ───────────────────────

    def add_to_prefix_index(self, item_id: str, codes: tuple[int, ...]) -> None:
        """Add item_id to the (c0, c1) prefix set — used for cold-start fallback."""
        self._r.sadd(f"prefix:{codes[0]}:{codes[1]}", item_id)

    def get_items_by_prefix(self, c0: int, c1: int, limit: int = 50) -> list[str]:
        """Return up to `limit` random item_ids sharing the (c0, c1) prefix."""
        return self._r.srandmember(f"prefix:{c0}:{c1}", limit) or []

    def add_to_prefix3_index(self, item_id: str, codes: tuple[int, ...]) -> None:
        """Add item_id to the (c0, c1, c2) prefix set — used by SASRec beam search."""
        self._r.sadd(f"prefix3:{codes[0]}:{codes[1]}:{codes[2]}", item_id)

    def get_items_by_prefix3(self, c0: int, c1: int, c2: int, limit: int = 50) -> list[str]:
        """Return up to `limit` random item_ids sharing the (c0, c1, c2) prefix."""
        return self._r.srandmember(f"prefix3:{c0}:{c1}:{c2}", limit) or []

    # ── Feature store (raw embeddings for FAISS auditing) ────────────────────

    def set_item_features(self, item_id: str, emb: np.ndarray) -> None:
        """Store the sentence-transformer embedding as raw bytes (no TTL)."""
        self._rb.set(f"feat:{item_id}", emb.astype(np.float32).tobytes())

    def get_item_features(self, item_id: str) -> Optional[np.ndarray]:
        """Return the item embedding, or None if not in the feature store."""
        raw = self._rb.get(f"feat:{item_id}")
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    @property
    def redis_client(self) -> redis.Redis:
        """Expose the text Redis connection for components that need to share it."""
        return self._r

    def ping(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False
