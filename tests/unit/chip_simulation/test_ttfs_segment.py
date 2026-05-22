import numpy as np
import torch

from mimarsinan.chip_simulation.ttfs_segment import (
    run_ttfs_continuous_segment,
    run_ttfs_quantized_segment,
    segment_ttfs_arrays_from_mapping,
    ttfs_core_membrane_voltages,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.softcore_mapping import HardCoreMapping


def _single_core_mapping():
    from mimarsinan.mapping.softcore_mapping import HardCore

    core = HardCore(axons_per_core=2, neurons_per_core=2)
    core.core_matrix = np.array([[2.0, -1.0], [0.5, 1.0]], dtype=np.float64)
    core.axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
    ]
    core.threshold = 2.0
    core.hardware_bias = np.array([0.1, -0.2], dtype=np.float64)
    core.latency = 0
    core.available_axons = 0
    core.available_neurons = 0
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core]
    mapping.output_sources = np.array([
        SpikeSource(0, 0),
        SpikeSource(0, 1),
    ])
    return mapping


def test_numpy_continuous_matches_torch_kernel_path():
    mapping = _single_core_mapping()
    seg = segment_ttfs_arrays_from_mapping(mapping)
    inp = np.array([[0.25, 0.75]], dtype=np.float64)
    out_np, _ = run_ttfs_continuous_segment(seg, inp)
    v = inp @ seg.core_params[0].T + seg.hw_biases[0]
    expected = np.clip(np.maximum(v, 0) / 2.0, 0, 1)
    np.testing.assert_allclose(out_np[0], expected[0], rtol=1e-9)


def test_cross_core_uses_upstream_activations_in_latency_order():
    """Regression: downstream cores must see prior-core TTFS activations, not zeros."""
    from mimarsinan.mapping.softcore_mapping import HardCore

    core0 = HardCore(axons_per_core=2, neurons_per_core=2)
    core0.core_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    core0.axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
    ]
    core0.threshold = 1.0
    core0.latency = 0
    core0.available_axons = 0
    core0.available_neurons = 0

    core1 = HardCore(axons_per_core=2, neurons_per_core=1)
    core1.core_matrix = np.array([[1.0, 1.0]], dtype=np.float64)
    core1.axon_sources = [
        SpikeSource(0, 0, is_input=False, is_off=False),
        SpikeSource(0, 1, is_input=False, is_off=False),
    ]
    core1.threshold = 1.0
    core1.latency = 1
    core1.available_axons = 0
    core1.available_neurons = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core0, core1]
    mapping.output_sources = np.array([SpikeSource(1, 0)])

    seg = segment_ttfs_arrays_from_mapping(mapping)
    inp = np.array([[0.5, 0.25]], dtype=np.float64)
    out_np, bufs = run_ttfs_continuous_segment(seg, inp)
    # core0 fires both neurons; core1 sums their activations (not zeros).
    assert bufs[0][0, 0] > 0.4
    assert bufs[0][0, 1] > 0.2
    assert out_np[0, 0] > 0.4


def test_cross_core_fill_when_axons_exceed_source_neurons():
    """Regression: axon dst span can be wider than source neuron buffer."""
    from mimarsinan.mapping.softcore_mapping import HardCore

    core0 = HardCore(axons_per_core=2, neurons_per_core=3)
    core0.core_matrix = np.eye(2, 3, dtype=np.float64)[:, :2]
    core0.axon_sources = [SpikeSource(-2, 0, is_input=True, is_off=False)] * 2
    core0.threshold = 1.0
    core0.latency = 0
    core0.available_axons = 0
    core0.available_neurons = 0

    core1 = HardCore(axons_per_core=10, neurons_per_core=2)
    core1.core_matrix = np.zeros((10, 2), dtype=np.float64)
    core1.axon_sources = [SpikeSource(0, j, is_input=False, is_off=False) for j in range(10)]
    core1.threshold = 1.0
    core1.latency = 1
    core1.available_axons = 0
    core1.available_neurons = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core0, core1]
    mapping.output_sources = np.array([SpikeSource(1, 0)])

    seg = segment_ttfs_arrays_from_mapping(mapping)
    inp = np.zeros((1, 2), dtype=np.float64)
    membrane = ttfs_core_membrane_voltages(seg, inp)
    assert membrane[0].shape == (1, 3)
    assert membrane[1].shape == (1, 2)


def test_quantized_matmul_uses_axon_dim_not_neuron_dim():
    """Regression: input_signals width must match core_params columns (axons)."""
    from mimarsinan.mapping.softcore_mapping import HardCore

    core = HardCore(axons_per_core=5, neurons_per_core=3)
    core.core_matrix = np.random.randn(5, 3).astype(np.float64)
    core.axon_sources = [SpikeSource(-2, i, is_input=True, is_off=False) for i in range(5)]
    core.threshold = 1.0
    core.latency = 0
    core.available_axons = 0
    core.available_neurons = 0
    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core]
    mapping.output_sources = np.array([SpikeSource(0, j) for j in range(3)])

    seg = segment_ttfs_arrays_from_mapping(mapping)
    assert seg.core_params[0].shape == (3, 5)
    inp = np.random.rand(2, 5).astype(np.float64)
    out_np, bufs = run_ttfs_quantized_segment(seg, inp, simulation_length=4)
    assert out_np.shape == (2, 3)
    assert bufs[0].shape == (2, 3)


def test_quantized_matches_ttfs_kernels():
    from mimarsinan.models.ttfs_kernels import ttfs_quantized_activation

    mapping = _single_core_mapping()
    seg = segment_ttfs_arrays_from_mapping(mapping)
    inp = np.array([[0.25, 0.75]], dtype=np.float64)
    s = 4
    out_np, _ = run_ttfs_quantized_segment(seg, inp, s)
    v_t = torch.tensor(inp @ seg.core_params[0].T + seg.hw_biases[0], dtype=torch.float64)
    th = torch.tensor(seg.thresholds[0], dtype=torch.float64)
    expected = ttfs_quantized_activation(v_t, th, s).numpy()
    np.testing.assert_allclose(out_np[0], expected[0], rtol=1e-9)
