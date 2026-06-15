"""Shared pytest fixtures.

Provides:
  fake_store   — ItemStore backed by fakeredis (no real Redis needed in CI)
  mock_state   — AppState-like object with model/store/intent_cache mocked
  api_client   — FastAPI TestClient with mock state injected
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from serving.store.redis_client import ItemStore


# ── Fake Redis-backed ItemStore ───────────────────────────────────────────────


@pytest.fixture()
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture()
def fake_redis_binary():
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture()
def fake_store(fake_redis, fake_redis_binary):
    """ItemStore wired to fakeredis — no real Redis required."""
    store = ItemStore.__new__(ItemStore)
    store._r = fake_redis
    store._rb = fake_redis_binary
    return store


@pytest.fixture()
def populated_store(fake_store):
    """fake_store pre-loaded with 5 items across 2 prefix buckets."""
    items = [
        ("ITEM_A", (1, 2, 3, 0), "Product A"),
        ("ITEM_B", (1, 2, 3, 1), "Product B"),
        ("ITEM_C", (1, 2, 4, 0), "Product C"),
        ("ITEM_D", (5, 6, 7, 0), "Product D"),
        ("ITEM_E", (5, 6, 7, 1), "Product E"),
    ]
    for item_id, codes, title in items:
        fake_store.set_item(item_id, codes, title)
        fake_store.add_to_prefix_index(item_id, codes)
        fake_store.add_to_prefix3_index(item_id, codes)
        fake_store._rb.set(
            f"feat:{item_id}",
            np.random.randn(8).astype(np.float32).tobytes(),
        )
    return fake_store


# ── Mocked AppState for API tests ─────────────────────────────────────────────


@pytest.fixture()
def mock_state(populated_store):
    """AppState-compatible mock: real store wrapped for method overrides, mocked SASRec model."""
    state = MagicMock()
    # Wrap the real store so its methods work normally but can be overridden per-test
    state.store = MagicMock(wraps=populated_store)
    state.num_levels = 3
    state.device = torch.device("cpu")

    # SASRec model mock — returns random logits for 3 levels, 32 codes each
    def fake_forward(inp, mask):
        B, T, L = inp.shape
        return [torch.randn(B, T, 32) for _ in range(3)]

    state.model.side_effect = fake_forward
    state.model.max_len = 50

    # Intent cache — always miss so we skip LLM in tests
    state.intent_cache.get.return_value = None

    return state


@pytest.fixture()
def api_client(mock_state):
    """TestClient with mock state injected — no real model or Redis needed."""
    from serving.api.main import app

    original = getattr(app.state, "recsys", None)
    app.state.recsys = mock_state
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
    app.state.recsys = original
