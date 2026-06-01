"""Tests for chip_simulation.spike_modes."""

import numpy as np
import torch

from mimarsinan.chip_simulation.recording.spike_modes import (
    to_deterministic_spikes,
    to_spikes,
    to_ttfs_latched_spikes,
    to_uniform_spikes,
)
from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_latched_spike_train


class TestSpikeModes:
    def test_uniform_saturates_at_one(self):
        rates = torch.tensor([[1.0]])
        T = 4
        for cycle in range(T):
            assert to_uniform_spikes(rates, cycle, T).item() == 1.0

    def test_to_spikes_dispatches_uniform(self):
        rates = torch.tensor([[0.5]])
        out = to_spikes(rates, cycle=0, simulation_length=4, spike_mode="Uniform")
        assert out.shape == rates.shape

    def test_deterministic_threshold(self):
        x = torch.tensor([[0.4, 0.6]])
        out = to_deterministic_spikes(x, threshold=0.5)
        assert out.tolist() == [[0.0, 1.0]]


class TestTtfsLatchedSpikes:
    def test_dispatches_via_to_spikes(self):
        rates = torch.tensor([[0.5]])
        out = to_spikes(rates, cycle=0, simulation_length=4, spike_mode="TTFS")
        assert out.shape == rates.shape

    def test_rate_zero_never_fires(self):
        rates = torch.tensor([[0.0]])
        for cycle in range(4):
            assert to_ttfs_latched_spikes(rates, cycle, 4).item() == 0.0

    def test_rate_one_latched_from_zero(self):
        rates = torch.tensor([[1.0]])
        for cycle in range(4):
            assert to_ttfs_latched_spikes(rates, cycle, 4).item() == 1.0

    def test_matches_numpy_latched_reference(self):
        # Per-cycle torch encoder stacked over T must equal the (N, D, S) numpy
        # reference used by the chip-simulation TTFS path.
        S = 8
        rates = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float64)
        stacked = torch.stack(
            [to_ttfs_latched_spikes(rates, c, S) for c in range(S)], dim=-1
        )  # (N, D, S)
        ref = ttfs_latched_spike_train(rates.numpy(), S)
        np.testing.assert_allclose(stacked.numpy(), ref, rtol=0, atol=0)
