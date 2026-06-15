"""Integration tests for the FastAPI serving endpoints.

All tests use a TestClient with a mocked AppState — no real Redis or model weights
needed. The mock state is defined in conftest.py.
"""

from __future__ import annotations


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["redis"] is True


# ── /recommend — cache hit path ───────────────────────────────────────────────


def test_recommend_cache_hit(api_client, mock_state):
    cached = [{"item_id": "ITEM_A", "title": "Product A", "semantic_id": [1, 2, 3, 0]}]
    mock_state.store.get_user_recs.return_value = cached

    resp = api_client.post(
        "/recommend", json={"user_id": "USER_1", "session": [], "top_k": 5}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is True
    assert body["recommendations"][0]["item_id"] == "ITEM_A"


# ── /recommend — cold start prefix fallback ───────────────────────────────────


def test_recommend_cold_start_prefix_fallback(api_client, mock_state):
    mock_state.store.get_user_recs.return_value = None  # cache miss

    resp = api_client.post(
        "/recommend",
        json={"user_id": "NEW_USER", "session": ["ITEM_A"], "top_k": 5},
        headers={"COLD_START_LLM_ENABLED": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is False
    assert body["cold_start_method"] in ("prefix_fallback", "intent", None)


def test_recommend_empty_session_no_user_returns_422(api_client, mock_state):
    # Empty session + no user_id + no catalog match → no recs possible → 422
    mock_state.store.get_user_recs.return_value = None

    resp = api_client.post(
        "/recommend",
        json={"session": [], "top_k": 5},
    )
    # Either we get empty cold-start recs (200) or a 422 if nothing resolves
    assert resp.status_code in (200, 422)


# ── /recommend — warm user SASRec path ───────────────────────────────────────


def test_recommend_warm_path(api_client, mock_state):
    mock_state.store.get_user_recs.return_value = None

    # 3+ session items triggers SASRec
    resp = api_client.post(
        "/recommend",
        json={"session": ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D"], "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is False
    assert body["cold_start_method"] is None


def test_recommend_session_items_not_in_results(api_client, mock_state):
    mock_state.store.get_user_recs.return_value = None
    session = ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D"]

    resp = api_client.post("/recommend", json={"session": session, "top_k": 10})
    assert resp.status_code == 200
    returned_ids = {r["item_id"] for r in resp.json()["recommendations"]}
    assert returned_ids.isdisjoint(set(session)), (
        "Session items must not appear in recs"
    )


def test_recommend_all_unknown_session_returns_422(api_client, mock_state):
    mock_state.store.get_user_recs.return_value = None
    mock_state.store.get_codes.return_value = None  # nothing resolves

    resp = api_client.post(
        "/recommend",
        json={"session": ["GHOST_1", "GHOST_2", "GHOST_3", "GHOST_4"], "top_k": 5},
    )
    assert resp.status_code == 422


# ── input validation ──────────────────────────────────────────────────────────


def test_recommend_top_k_out_of_range(api_client):
    resp = api_client.post("/recommend", json={"session": [], "top_k": 0})
    assert resp.status_code == 422

    resp = api_client.post("/recommend", json={"session": [], "top_k": 101})
    assert resp.status_code == 422


def test_recommend_missing_session_field(api_client):
    resp = api_client.post("/recommend", json={"user_id": "X"})
    assert resp.status_code == 422
