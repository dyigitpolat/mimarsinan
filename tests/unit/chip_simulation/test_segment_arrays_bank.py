"""Regression: the TTFS contract's bank-weight reconstruction
(``segment_ttfs_arrays_from_mapping``) must recover the true ``(axons, neurons)``
weights for a *non-square*, bank-backed core.

The bank is stored ``(axons, neurons)`` (see ``register_weight_bank``); the
reconstruction previously sliced it with the axon/neuron ranges swapped, which
is a no-op for square cores but zeroed every axon beyond ``used_neurons`` for
non-square cores (e.g. an offloaded conv: many axons, few output channels) -
silently corrupting the contract V and breaking SANA-FE TTFS parity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.chip_simulation.ttfs.segment_arrays import (
    segment_ttfs_arrays_from_mapping,
)


def _bank_backed_mapping(n_axons: int, n_neurons: int):
    rng = np.random.default_rng(0)
    # Bank weights laid out (axons, neurons), exactly as register_weight_bank.
    bank = rng.uniform(-0.5, 0.5, size=(n_axons, n_neurons)).astype(np.float64)
    core = SimpleNamespace(
        axons_per_core=n_axons, neurons_per_core=n_neurons,
        available_axons=0, available_neurons=0,
        threshold=1.0, latency=0, hardware_bias=None,
        core_matrix=bank.copy(),  # what the runtime/HCM actually wires
        axon_sources=[SpikeSource(-1, i, True, False, False) for i in range(n_axons)],
    )
    placement = {
        "weight_bank_id": 0, "axon_offset": 0, "neuron_offset": 0,
        "axons": n_axons, "neurons": n_neurons,
        "bank_axon_range": (0, n_axons), "bank_neuron_range": (0, n_neurons),
    }
    mapping = SimpleNamespace(
        cores=[core],
        weight_banks={0: bank},
        soft_core_placements_per_hard_core=[[placement]],
        output_sources=np.asarray(
            [SpikeSource(0, j, False, False, False) for j in range(n_neurons)]
        ),
    )
    return mapping, bank


def test_bank_reconstruction_nonsquare_matches_true_weights():
    n_axons, n_neurons = 10, 3  # non-square: exposes the swapped-range bug
    mapping, bank = _bank_backed_mapping(n_axons, n_neurons)

    arrays = segment_ttfs_arrays_from_mapping(mapping)
    core_params = np.asarray(arrays.core_params[0])

    # core_params is (neurons, axons) for ``v = input_signals @ core_params.T``.
    assert core_params.shape == (n_neurons, n_axons)
    np.testing.assert_allclose(core_params, bank.T, atol=1e-12)
    # Guard the original failure mode: weights beyond ``used_neurons`` were zeroed.
    assert np.any(core_params[:, n_neurons:] != 0.0), (
        "axons beyond used_neurons were zeroed - bank slice ranges still swapped"
    )


def test_bank_reconstruction_square_unchanged():
    n = 5
    mapping, bank = _bank_backed_mapping(n, n)
    arrays = segment_ttfs_arrays_from_mapping(mapping)
    np.testing.assert_allclose(np.asarray(arrays.core_params[0]), bank.T, atol=1e-12)
