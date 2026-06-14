"""Determinism fixture + rng_snapshot invariants (P0, invariant I6)."""

import random

import numpy as np
import torch

from conftest import rng_snapshot


def test_deterministic_rng_seeds_torch(deterministic_rng):
    got = torch.rand(3)
    torch.manual_seed(0)
    expected = torch.rand(3)
    assert torch.allclose(got, expected)


def test_deterministic_rng_seeds_numpy_and_python(deterministic_rng):
    got_np = np.random.rand(3)
    got_py = [random.random() for _ in range(3)]
    np.random.seed(0)
    random.seed(0)
    assert np.allclose(got_np, np.random.rand(3))
    assert got_py == [random.random() for _ in range(3)]


def test_rng_snapshot_restores_torch():
    torch.manual_seed(123)
    a = torch.rand(1)
    with rng_snapshot():
        _ = torch.rand(50)  # advance the RNG inside the block
    b = torch.rand(1)
    # b must equal the draw that would follow `a` had the inner block not run
    torch.manual_seed(123)
    assert torch.allclose(torch.rand(1), a)
    assert torch.allclose(torch.rand(1), b)


def test_rng_snapshot_restores_numpy_and_python():
    np.random.seed(7)
    random.seed(7)
    a_np = np.random.rand()
    a_py = random.random()
    with rng_snapshot():
        np.random.rand(20)
        [random.random() for _ in range(20)]
    b_np = np.random.rand()
    b_py = random.random()
    np.random.seed(7)
    random.seed(7)
    assert np.random.rand() == a_np
    assert random.random() == a_py
    assert np.random.rand() == b_np
    assert random.random() == b_py
