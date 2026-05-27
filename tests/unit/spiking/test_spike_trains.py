"""Unit tests for mimarsinan.spiking.spike_trains."""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations import LIFActivation, uniform_encode_to_spike_train
from mimarsinan.spiking.spike_trains import (
    lif_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)


def test_uniform_spike_train_matches_legacy_encoder() -> None:
    T = 4
    rates = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).reshape(1, 5)
    legacy = uniform_encode_to_spike_train(rates, T)
    shared = uniform_spike_train(rates, T)
    torch.testing.assert_close(shared, legacy)


def test_lif_spike_train_matches_forward_spiking_when_flag_on() -> None:
    act = LIFActivation(T=4, activation_scale=torch.tensor(1.5))
    act.use_cycle_accurate_trains = True
    x = torch.linspace(-0.2, 1.8, 12).reshape(3, 4)
    via_method = act.forward_spiking(x)
    via_helper = lif_spike_train(x, act)
    torch.testing.assert_close(via_method, via_helper)


def test_rates_to_spike_train_uniform_mode() -> None:
    rates = torch.tensor([[0.5, 1.0]])
    train = rates_to_spike_train(rates, 4, spike_mode="Uniform", log_fallback=False)
    assert train.shape == (4, 1, 2)
    torch.testing.assert_close(train[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(train[2, 0, 0], torch.tensor(1.0))
