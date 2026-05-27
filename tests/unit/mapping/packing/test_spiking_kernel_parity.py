"""TTFS kernel parity between models.spiking and chip_simulation paths."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.models.spiking.ttfs_kernels import (
    ttfs_quantized_activation,
    ttfs_quantized_activation_np,
)


def test_ttfs_kernels_torch_numpy_parity():
    v_np = np.array([[0.25, 0.75, 1.1]], dtype=np.float64)
    th_np = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    s = 8

    out_np = ttfs_quantized_activation_np(v_np, th_np, s)

    v_t = torch.tensor(v_np, dtype=torch.float64)
    th_t = torch.tensor(th_np, dtype=torch.float64)
    out_t = ttfs_quantized_activation(v_t, th_t, s).numpy()

    np.testing.assert_allclose(out_np, out_t, rtol=1e-12, atol=1e-12)


def test_ttfs_kernels_match_chip_simulation_segment_reference():
    from mimarsinan.chip_simulation.ttfs.ttfs_segment import (
        run_ttfs_quantized_segment,
        segment_ttfs_arrays_from_mapping,
    )
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

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

    seg = segment_ttfs_arrays_from_mapping(mapping)
    inp = np.array([[0.25, 0.75]], dtype=np.float64)
    s = 4
    out_np, _ = run_ttfs_quantized_segment(seg, inp, s)

    v_t = torch.tensor(inp @ seg.core_params[0].T + seg.hw_biases[0], dtype=torch.float64)
    th = torch.tensor(seg.thresholds[0], dtype=torch.float64)
    expected = ttfs_quantized_activation(v_t, th, s).numpy()
    np.testing.assert_allclose(out_np[0], expected[0], rtol=1e-9)
