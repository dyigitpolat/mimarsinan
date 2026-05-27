"""Tests for chip_simulation.spike_modes."""

import torch

from mimarsinan.chip_simulation.recording.spike_modes import (
    to_deterministic_spikes,
    to_spikes,
    to_uniform_spikes,
)


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
