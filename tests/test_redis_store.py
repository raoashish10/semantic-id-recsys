"""Tests for ItemStore Redis CRUD operations (backed by fakeredis)."""

import numpy as np


def test_set_and_get_item(fake_store):
    fake_store.set_item("ITEM_X", (1, 2, 3, 0), "Widget X")
    assert fake_store.get_codes("ITEM_X") == [1, 2, 3, 0]
    assert fake_store.get_title("ITEM_X") == "Widget X"


def test_get_codes_missing_returns_none(fake_store):
    assert fake_store.get_codes("DOES_NOT_EXIST") is None


def test_get_title_missing_returns_empty(fake_store):
    assert fake_store.get_title("DOES_NOT_EXIST") == ""


def test_reverse_lookup(fake_store):
    fake_store.set_item("ITEM_Y", (4, 5, 6, 0), "Widget Y")
    assert fake_store.get_item_id((4, 5, 6, 0)) == "ITEM_Y"


def test_prefix2_index(fake_store):
    fake_store.set_item("A", (1, 2, 3, 0), "A")
    fake_store.set_item("B", (1, 2, 4, 0), "B")
    fake_store.set_item("C", (9, 9, 9, 0), "C")
    fake_store.add_to_prefix_index("A", (1, 2, 3, 0))
    fake_store.add_to_prefix_index("B", (1, 2, 4, 0))
    fake_store.add_to_prefix_index("C", (9, 9, 9, 0))

    results = fake_store.get_items_by_prefix(1, 2, limit=10)
    assert set(results) == {"A", "B"}


def test_prefix3_index(fake_store):
    fake_store.set_item("A", (1, 2, 3, 0), "A")
    fake_store.set_item("B", (1, 2, 3, 1), "B")
    fake_store.add_to_prefix3_index("A", (1, 2, 3, 0))
    fake_store.add_to_prefix3_index("B", (1, 2, 3, 1))

    results = fake_store.get_items_by_prefix3(1, 2, 3, limit=10)
    assert set(results) == {"A", "B"}


def test_prefix_returns_empty_for_unknown(fake_store):
    assert fake_store.get_items_by_prefix(99, 99) == []
    assert fake_store.get_items_by_prefix3(99, 99, 99) == []


def test_user_recs_roundtrip(fake_store):
    recs = [{"item_id": "X", "title": "T", "semantic_id": [1, 2, 3, 0]}]
    fake_store.set_user_recs("USER_1", recs)
    assert fake_store.get_user_recs("USER_1") == recs


def test_user_recs_miss_returns_none(fake_store):
    assert fake_store.get_user_recs("NO_SUCH_USER") is None


def test_feature_store_roundtrip(fake_store, fake_redis_binary):
    fake_store._rb = fake_redis_binary
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    fake_store.set_item_features("ITEM_Z", emb)
    recovered = fake_store.get_item_features("ITEM_Z")
    np.testing.assert_allclose(recovered, emb, rtol=1e-6)


def test_feature_store_miss_returns_none(fake_store):
    assert fake_store.get_item_features("GHOST") is None


def test_ping(fake_store):
    assert fake_store.ping() is True
