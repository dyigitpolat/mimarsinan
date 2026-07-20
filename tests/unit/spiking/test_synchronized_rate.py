"""[calculus §16] the synchronized-rate exactness theorem, locked as tests.

Two-window execution (integrate with firing disabled, then input-free
emission) makes the emitted count a function of input COUNTS alone — equal to
the strict LIF count staircase for ANY arrival pattern. Streaming execution
differs by a level-crossing statistic that vanishes only for constant input.
"""

from __future__ import annotations

import inspect

import torch

from mimarsinan.models.nn.activations.autograd import LIFCountStaircaseFunction
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_policies import LifSegmentPolicy


def _two_window_counts(charge_seq: torch.Tensor, theta: torch.Tensor,
                       T: int, v0: float = 0.0) -> torch.Tensor:
    """Reference chip discipline: integrate all cycles (fire-disable), then T
    emission cycles of strict fire-and-subtract."""
    V = torch.full(charge_seq.shape[1:], v0, dtype=torch.float64)
    for t in range(charge_seq.shape[0]):
        V = V + charge_seq[t] / theta
    fires = torch.zeros_like(V)
    for _ in range(T):
        f = (V > 1.0).to(V)
        fires += f
        V -= f
    return fires


def _streaming_counts(charge_seq: torch.Tensor, theta: torch.Tensor,
                      T: int, v0: float = 0.0) -> torch.Tensor:
    V = torch.full(charge_seq.shape[1:], v0, dtype=torch.float64)
    fires = torch.zeros_like(V)
    for t in range(charge_seq.shape[0]):
        V = V + charge_seq[t] / theta
        f = (V > 1.0).to(V)
        fires += f
        V -= f
    return fires


def _random_train(counts: torch.Tensor, T: int, gen: torch.Generator) -> torch.Tensor:
    """Binary (T, n) train with EXACTLY counts[i] spikes per channel, at
    uniformly random cycles — arbitrary arrival order."""
    n = counts.numel()
    train = torch.zeros(T, n, dtype=torch.float64)
    for i in range(n):
        pos = torch.randperm(T, generator=gen)[: int(counts[i])]
        train[pos, i] = 1.0
    return train


def test_two_window_equals_staircase_for_any_arrival():
    gen = torch.Generator().manual_seed(0)
    for T in (4, 8, 32):
        for _ in range(6):
            n_in, n_out = 23, 7
            w = torch.randn(n_out, n_in, generator=gen, dtype=torch.float64)
            b = torch.randn(n_out, generator=gen, dtype=torch.float64) * 0.3
            theta = torch.rand(n_out, generator=gen, dtype=torch.float64) * 2.5 + 0.3
            counts = torch.randint(0, T + 1, (n_in,), generator=gen)
            train = _random_train(counts, T, gen)
            charge = train @ w.T + b
            two = _two_window_counts(charge, theta, T)
            z = w @ (counts.to(torch.float64) / T) + b
            stair = LIFCountStaircaseFunction.apply(z, theta, T, True)
            kernel_counts = stair * T / theta
            torch.testing.assert_close(two, kernel_counts, atol=1e-9, rtol=0.0)


def test_two_window_is_arrival_order_invariant():
    gen = torch.Generator().manual_seed(1)
    T, n_in, n_out = 16, 12, 5
    w = torch.randn(n_out, n_in, generator=gen, dtype=torch.float64)
    b = torch.zeros(n_out, dtype=torch.float64)
    theta = torch.full((n_out,), 1.3, dtype=torch.float64)
    counts = torch.randint(0, T + 1, (n_in,), generator=gen)
    reference = None
    for _ in range(5):
        train = _random_train(counts, T, gen)
        two = _two_window_counts(train @ w.T + b, theta, T)
        if reference is None:
            reference = two
        else:
            torch.testing.assert_close(two, reference, atol=0.0, rtol=0.0)


def test_exact_integer_tie_uses_strict_semantics():
    T = 8
    theta = torch.tensor([1.0], dtype=torch.float64)
    train = torch.zeros(T, 1, dtype=torch.float64)
    train[:3, 0] = 1.0  # V_T = 3.0 exactly
    two = _two_window_counts(train, theta, T)
    assert float(two) == 2.0  # strict V > 1: an exact integer emits V_T - 1
    z = torch.tensor([3.0 / T], dtype=torch.float64)
    stair = LIFCountStaircaseFunction.apply(z, theta, T, True)
    assert float(stair * T / theta) == 2.0


def test_streaming_matches_sync_only_for_constant_input():
    T = 16
    theta = torch.tensor([1.0], dtype=torch.float64)
    constant = torch.full((T, 1), 0.22, dtype=torch.float64)
    torch.testing.assert_close(
        _streaming_counts(constant, theta, T),
        _two_window_counts(constant, theta, T), atol=0.0, rtol=0.0,
    )
    bursty = torch.zeros(T, 1, dtype=torch.float64)
    bursty[0, 0] = 3.4    # early positive burst...
    bursty[1:, 0] = -3.3 / (T - 1)  # ...cancelled later: total 0.1
    assert float(_two_window_counts(bursty, theta, T)) == 0.0
    assert float(_streaming_counts(bursty, theta, T)) > 0.0  # level-crossing overfire


def test_policy_walk_and_accessor_carry_the_discipline():
    from mimarsinan.chip_simulation.spiking_semantics import (
        lif_execution_synchronized,
    )

    assert LifSegmentPolicy(synchronized=True).synchronized is True
    assert LifSegmentPolicy().synchronized is False
    assert "synchronized" in inspect.signature(chip_aligned_segment_forward).parameters
    assert lif_execution_synchronized({}) is False
    assert lif_execution_synchronized(
        {"lif_execution_discipline": "synchronized"}
    ) is True
    assert lif_execution_synchronized(
        {"lif_execution_discipline": "streaming"}
    ) is False
