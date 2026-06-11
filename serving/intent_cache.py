"""Redis-backed cache for IntentResult objects.

Keyed by a fingerprint derived from the session item IDs, so the same set of
items always resolves to the same cache entry regardless of insertion order.

TTL is 300s — short enough that a user who adds a new item to their session
gets a fresh inference fairly quickly, long enough to absorb repeated requests
from the same short session.

The cache reuses the Redis client from an existing ItemStore; no new connection
objects are created.
"""

import hashlib

import redis

from serving.intent import IntentResult

_TTL_SECONDS = 300


class IntentCache:
    def __init__(self, redis_client: "redis.Redis[str]") -> None:
        self._r = redis_client

    @staticmethod
    def _fingerprint(item_ids: list[str]) -> str:
        content = ",".join(sorted(item_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, item_ids: list[str]) -> IntentResult | None:
        """Return a cached IntentResult for this set of item IDs, or None on miss."""
        raw = self._r.get(f"intent:{self._fingerprint(item_ids)}")
        if raw is None:
            return None
        return IntentResult.model_validate_json(raw)

    def set(self, item_ids: list[str], result: IntentResult) -> None:
        """Cache an IntentResult with a 300s TTL."""
        self._r.set(
            f"intent:{self._fingerprint(item_ids)}",
            result.model_dump_json(),
            ex=_TTL_SECONDS,
        )
