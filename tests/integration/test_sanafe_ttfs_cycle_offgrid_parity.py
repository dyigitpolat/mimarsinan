"""Synchronized genuine SANA-FE == quantized analytical reference on OFF-GRID input.

The synchronized soma decodes inputs from single-spike timings, so off-grid
values are snapped to the 1/S grid on the wire. The analytical reference applies
the same snap (``ttfs_input_grid_quantize``, gated on the synchronized schedule);
this locks that the genuine backend matches it exactly, and that a continuous
(un-snapped) reference would NOT match — pinning the root cause so a refactor
cannot silently revert the contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _have_sanafe() -> bool:
    try:
        import sanafe  # noqa: F401
        return True
    except Exception:
        return False


def _cycle_plugin_built() -> bool:
    return Path(
        "build/mimarsinan_sanafe_plugins/libmimarsinan_ttfs_cycle_soma.so"
    ).is_file()


pytestmark = [
    pytest.mark.skipif(not _have_sanafe(),
                       reason="SANA-FE not installed (scripts/bootstrap_sanafe.sh)"),
    pytest.mark.skipif(not _cycle_plugin_built(),
                       reason="mimarsinan_ttfs_cycle_soma plugin not built"),
    pytest.mark.slow,
    pytest.mark.integration,
]

S = 16

OFF_GRID_RATES = np.array([[0.61, 0.333, 0.97, 0.05]], dtype=np.float64)


def _kernel(V: np.ndarray) -> np.ndarray:
    from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation
    return ttfs_quantized_activation(
        torch.tensor(V, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64), S,
    ).numpy()


def _grid(rates: np.ndarray) -> np.ndarray:
    from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_input_grid_quantize
    return ttfs_input_grid_quantize(rates, S)


def _from_spike_source(core_idx, neuron, *, is_input):
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    return SpikeSource(core_idx, neuron, is_input, False, False)


def _run_genuine(mapping, rates):
    from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
    runner = SanafeRunner(
        mapping=mapping, simulation_length=S, arch_preset="loihi",
        spiking_mode="ttfs_cycle_based", firing_mode="TTFS",
        ttfs_cycle_schedule="synchronized",
        thresholding_mode="<=", log_potential_trace=True,
    )
    rec = runner.run(rates.astype(np.float32), sample_index=0)
    return np.asarray(rec.segments[0].per_neuron_potential_trace)[:, -1]


def _single_core_mapping(W):
    n_ax, n_ne = W.shape
    core = SimpleNamespace(
        axons_per_core=n_ax, neurons_per_core=n_ne,
        available_axons=0, available_neurons=0, threshold=1.0,
        core_matrix=W.astype(np.float32),
        axon_sources=[_from_spike_source(-1, i, is_input=True) for i in range(n_ax)],
        hardware_bias=None, latency=0,
    )
    hcm = SimpleNamespace(
        cores=[core],
        output_sources=np.asarray([_from_spike_source(0, j, is_input=False)
                                   for j in range(n_ne)]),
    )
    stage = SimpleNamespace(
        kind="neural", name="t", hard_core_mapping=hcm, compute_op=None,
        input_map=[SimpleNamespace(node_id=-2, offset=0, size=n_ax)],
        output_map=[SimpleNamespace(node_id=0, offset=0, size=n_ne)],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    return SimpleNamespace(
        stages=[stage], get_neural_segments=lambda: [hcm],
        get_compute_ops=lambda: [], output_sources=hcm.output_sources,
        node_activation_scales={}, node_input_activation_scales={},
    )


def _two_core_mapping(W0, W1):
    c0 = SimpleNamespace(
        axons_per_core=W0.shape[0], neurons_per_core=W0.shape[1],
        available_axons=0, available_neurons=0, threshold=1.0,
        core_matrix=W0.astype(np.float32),
        axon_sources=[_from_spike_source(-1, i, is_input=True)
                      for i in range(W0.shape[0])],
        hardware_bias=None, latency=0,
    )
    c1 = SimpleNamespace(
        axons_per_core=W1.shape[0], neurons_per_core=W1.shape[1],
        available_axons=0, available_neurons=0, threshold=1.0,
        core_matrix=W1.astype(np.float32),
        axon_sources=[_from_spike_source(0, j, is_input=False)
                      for j in range(W1.shape[0])],
        hardware_bias=None, latency=1,
    )
    hcm = SimpleNamespace(
        cores=[c0, c1],
        output_sources=np.asarray([_from_spike_source(1, j, is_input=False)
                                   for j in range(W1.shape[1])]),
    )
    stage = SimpleNamespace(
        kind="neural", name="t", hard_core_mapping=hcm, compute_op=None,
        input_map=[SimpleNamespace(node_id=-2, offset=0, size=W0.shape[0])],
        output_map=[SimpleNamespace(node_id=1, offset=0, size=W1.shape[1])],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    return SimpleNamespace(
        stages=[stage], get_neural_segments=lambda: [hcm],
        get_compute_ops=lambda: [], output_sources=hcm.output_sources,
        node_activation_scales={}, node_input_activation_scales={},
    )


def test_synchronized_offgrid_matches_quantized_reference():
    rng = np.random.default_rng(23)
    for _ in range(5):
        W = rng.uniform(0.0, 0.6, size=(4, 3)).astype(np.float32)
        rates = rng.uniform(0.0, 1.0, size=(1, 4))
        expected = _kernel(_grid(rates) @ W.astype(np.float64))[0]
        genuine = _run_genuine(_single_core_mapping(W), rates)
        np.testing.assert_allclose(genuine, expected, atol=1e-6)


def test_synchronized_offgrid_differs_from_continuous_reference():
    """Pins the root cause: an un-snapped (continuous) reference would mismatch."""
    W = np.eye(4, dtype=np.float32)
    continuous = _kernel(OFF_GRID_RATES @ W.astype(np.float64))[0]
    quantized = _kernel(_grid(OFF_GRID_RATES) @ W.astype(np.float64))[0]
    assert not np.allclose(continuous, quantized, atol=1e-6)
    genuine = _run_genuine(_single_core_mapping(W), OFF_GRID_RATES)
    np.testing.assert_allclose(genuine, quantized, atol=1e-6)


def test_synchronized_offgrid_two_group_cascade():
    """Off-grid input through two latency groups: only the entry boundary snaps;
    interior single-spike timings are exact by construction."""
    rng = np.random.default_rng(31)
    W0 = rng.uniform(0.0, 0.6, size=(3, 2)).astype(np.float32)
    W1 = rng.uniform(0.0, 0.8, size=(2, 2)).astype(np.float32)
    rates = rng.uniform(0.0, 1.0, size=(1, 3))

    a0 = _kernel(_grid(rates) @ W0.astype(np.float64))
    a1 = _kernel(a0 @ W1.astype(np.float64))[0]
    genuine = _run_genuine(_two_core_mapping(W0, W1), rates)
    np.testing.assert_allclose(genuine[2:], a1, atol=1e-6)
