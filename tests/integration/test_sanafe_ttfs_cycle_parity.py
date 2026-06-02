"""Genuine single-spike SANA-FE TTFS-cycle backend == analytical reference.

``ttfs_cycle_based`` runs the genuine event-driven soma (no preset membrane) on
the synchronized ``S × num_groups`` schedule: each latency group fires in its own
S-cycle window, emitting exactly one timing-coded spike per neuron, and downstream
groups reconstruct V from the incoming spike *timings*. By the ReLU↔TTFS
equivalence the decoded result equals ``ttfs_quantized_activation`` composed in
topological order — so genuine SANA-FE output must match the analytical kernel
exactly for on-grid inputs (off-grid inputs differ only by the orthogonal,
expected ≤1-level input spike-time quantization).
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


def _kernel(V: np.ndarray) -> np.ndarray:
    from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation
    return ttfs_quantized_activation(
        torch.tensor(V, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64), S,
    ).numpy()


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


def test_single_core_genuine_matches_analytical_on_grid():
    rng = np.random.default_rng(7)
    for _ in range(5):
        W = rng.uniform(0.0, 0.6, size=(4, 3))
        rates = (rng.integers(0, S + 1, size=(1, 4)) / S).astype(np.float64)
        analytical = _kernel(rates @ W)[0]
        genuine = _run_genuine(_single_core_mapping(W), rates)
        np.testing.assert_allclose(genuine, analytical, atol=1e-9)


def test_two_core_cascade_genuine_matches_analytical():
    """Synchronized cross-group schedule: group 0 fires in [S,2S), group 1 decodes
    its single spikes in [2S,3S). Total timeline = 3·S."""
    rng = np.random.default_rng(11)
    W0 = rng.uniform(0.0, 0.6, size=(3, 2)).astype(np.float32)
    W1 = rng.uniform(0.0, 0.8, size=(2, 2)).astype(np.float32)
    rates = (rng.integers(0, S + 1, size=(1, 3)) / S).astype(np.float64)

    a0 = _kernel(rates @ W0)
    a1 = _kernel(a0 @ W1)[0]

    c0 = SimpleNamespace(
        axons_per_core=3, neurons_per_core=2, available_axons=0, available_neurons=0,
        threshold=1.0, core_matrix=W0,
        axon_sources=[_from_spike_source(-1, i, is_input=True) for i in range(3)],
        hardware_bias=None, latency=0,
    )
    c1 = SimpleNamespace(
        axons_per_core=2, neurons_per_core=2, available_axons=0, available_neurons=0,
        threshold=1.0, core_matrix=W1,
        axon_sources=[_from_spike_source(0, j, is_input=False) for j in range(2)],
        hardware_bias=None, latency=1,
    )
    hcm = SimpleNamespace(
        cores=[c0, c1],
        output_sources=np.asarray([_from_spike_source(1, 0, is_input=False),
                                   _from_spike_source(1, 1, is_input=False)]),
    )
    stage = SimpleNamespace(
        kind="neural", name="t", hard_core_mapping=hcm, compute_op=None,
        input_map=[SimpleNamespace(node_id=-2, offset=0, size=3)],
        output_map=[SimpleNamespace(node_id=1, offset=0, size=2)],
        schedule_segment_index=None, schedule_pass_index=None,
    )
    mapping = SimpleNamespace(
        stages=[stage], get_neural_segments=lambda: [hcm],
        get_compute_ops=lambda: [], output_sources=hcm.output_sources,
        node_activation_scales={}, node_input_activation_scales={},
    )
    genuine = _run_genuine(mapping, rates)  # [core0 n0, core0 n1, core1 n0, core1 n1]
    np.testing.assert_allclose(genuine[2:], a1, atol=1e-9)
