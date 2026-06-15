"""Tests for SASRec build_input and beam_recommend."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from serving.inference import beam_recommend, build_input
from serving.models import RecommendedItem


# ── build_input ───────────────────────────────────────────────────────────────


def test_build_input_resolves_known_items(populated_store):
    inp, n = build_input(
        ["ITEM_A", "ITEM_C"],
        populated_store,
        num_levels=3,
        max_len=50,
        device=torch.device("cpu"),
    )
    assert inp is not None
    assert n == 2
    assert inp.shape == (1, 2, 3)


def test_build_input_truncates_to_num_levels(populated_store):
    # Items have 4 codes; build_input should keep only first num_levels=3
    inp, n = build_input(
        ["ITEM_A"],
        populated_store,
        num_levels=3,
        max_len=50,
        device=torch.device("cpu"),
    )
    assert inp.shape[-1] == 3


def test_build_input_returns_none_for_unknown_items(populated_store):
    inp, n = build_input(
        ["GHOST_ITEM"],
        populated_store,
        num_levels=3,
        max_len=50,
        device=torch.device("cpu"),
    )
    assert inp is None
    assert n == 0


def test_build_input_skips_unknown_mixed(populated_store):
    inp, n = build_input(
        ["ITEM_A", "GHOST", "ITEM_D"],
        populated_store,
        num_levels=3,
        max_len=50,
        device=torch.device("cpu"),
    )
    assert n == 2  # only 2 resolve


def test_build_input_respects_max_len(populated_store):
    inp, n = build_input(
        ["ITEM_A", "ITEM_C", "ITEM_D", "ITEM_E"],
        populated_store,
        num_levels=3,
        max_len=2,
        device=torch.device("cpu"),
    )
    assert inp.shape[1] == 2


# ── beam_recommend ────────────────────────────────────────────────────────────


def _make_model_mock(num_levels=3, num_codes=32):
    """SASRec mock that returns random logits."""
    model = MagicMock()
    model.max_len = 50

    def forward(inp, mask):
        B, T, L = inp.shape
        return [torch.randn(B, T, num_codes) for _ in range(num_levels)]

    model.__call__ = MagicMock(side_effect=forward)
    model.eval.return_value = None
    return model


def test_beam_recommend_returns_items(populated_store):
    model = _make_model_mock()
    inp = torch.zeros(1, 3, 3, dtype=torch.long)
    results = beam_recommend(model, populated_store, inp, num_levels=3, top_k=5)
    assert isinstance(results, list)
    assert all(isinstance(r, RecommendedItem) for r in results)


def test_beam_recommend_excludes_session_items(populated_store):
    model = _make_model_mock()
    inp = torch.zeros(1, 3, 3, dtype=torch.long)
    exclude = {"ITEM_A", "ITEM_B"}
    results = beam_recommend(
        model, populated_store, inp, num_levels=3, top_k=10, exclude_ids=exclude
    )
    returned_ids = {r.item_id for r in results}
    assert returned_ids.isdisjoint(exclude), (
        "Session items must not appear in recommendations"
    )


def test_beam_recommend_respects_top_k(populated_store):
    model = _make_model_mock()
    inp = torch.zeros(1, 3, 3, dtype=torch.long)
    results = beam_recommend(model, populated_store, inp, num_levels=3, top_k=2)
    assert len(results) <= 2


def test_beam_recommend_no_duplicates(populated_store):
    model = _make_model_mock()
    inp = torch.zeros(1, 3, 3, dtype=torch.long)
    results = beam_recommend(model, populated_store, inp, num_levels=3, top_k=10)
    item_ids = [r.item_id for r in results]
    assert len(item_ids) == len(set(item_ids)), "No duplicate items in results"
