"""Tests for cold-start routing and prefix fallback logic."""

from __future__ import annotations

import pytest

from serving.api.routes import _prefix_recommend, COLD_START_THRESHOLD


def test_prefix_recommend_returns_items_from_dominant_prefix(populated_store):
    # ITEM_A and ITEM_B share prefix (1,2); feed one of them as session
    recs = _prefix_recommend(["ITEM_A"], populated_store, top_k=5)
    assert len(recs) > 0
    returned_ids = {r.item_id for r in recs}
    # Neighbor ITEM_B or ITEM_C (all share c0=1,c1=2) should appear
    assert returned_ids & {"ITEM_B", "ITEM_C"}


def test_prefix_recommend_excludes_session_items(populated_store):
    recs = _prefix_recommend(["ITEM_A", "ITEM_B"], populated_store, top_k=10)
    returned_ids = {r.item_id for r in recs}
    assert "ITEM_A" not in returned_ids
    assert "ITEM_B" not in returned_ids


def test_prefix_recommend_empty_session(populated_store):
    recs = _prefix_recommend([], populated_store, top_k=5)
    assert recs == []


def test_prefix_recommend_unknown_item(populated_store):
    recs = _prefix_recommend(["GHOST_ITEM"], populated_store, top_k=5)
    assert recs == []


def test_prefix_recommend_respects_top_k(populated_store):
    recs = _prefix_recommend(["ITEM_D"], populated_store, top_k=1)
    assert len(recs) <= 1


def test_cold_start_threshold_value():
    # Explicit: threshold is 3 — changing this breaks the beam search semantic
    assert COLD_START_THRESHOLD == 3
