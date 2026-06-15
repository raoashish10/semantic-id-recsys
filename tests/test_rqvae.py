"""Tests for RQ-VAE collision resolution logic."""

import numpy as np

from offline.rqvae.train import resolve_collisions


def test_no_collisions_all_c3_zero():
    codes = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    out = resolve_collisions(codes)
    assert out.shape == (3, 4)
    assert list(out[:, 3]) == [0, 0, 0]


def test_collisions_get_sequential_c3():
    # All 3 items share the same (c0, c1, c2)
    codes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    out = resolve_collisions(codes)
    assert sorted(out[:, 3].tolist()) == [0, 1, 2]


def test_mixed_collision_and_unique():
    # First 2 collide; third is unique
    codes = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    out = resolve_collisions(codes)
    # Unique item at index 2 gets c3=0
    assert out[2, 3] == 0
    # Colliding pair gets c3=0 and c3=1
    assert sorted(out[:2, 3].tolist()) == [0, 1]


def test_output_is_always_unique():
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 4, size=(50, 3))
    out = resolve_collisions(codes)
    tuples = [tuple(row) for row in out]
    assert len(set(tuples)) == len(tuples), (
        "resolve_collisions must produce unique 4-tuples"
    )


def test_first_three_cols_unchanged():
    codes = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    out = resolve_collisions(codes)
    np.testing.assert_array_equal(out[:, :3], codes)


def test_large_input_all_unique():
    # 256^3 possible 3-tuples — even a fully random 1000-item catalog should resolve
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 32, size=(1000, 3))
    out = resolve_collisions(codes)
    tuples = [tuple(row) for row in out]
    assert len(set(tuples)) == 1000
