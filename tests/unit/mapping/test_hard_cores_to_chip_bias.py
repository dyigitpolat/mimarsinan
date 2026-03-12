"""Tests for hard_cores_to_chip bias handling across all three modes.

Mode 1: hardware_bias field (dedicated bias register, no always-on row).
Mode 2: Legacy always-on row (has_bias_capability=True, no hardware_bias).
Mode 3: No bias (has_bias_capability=False, no hardware_bias).
"""

import numpy as np
import pytest

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping
from mimarsinan.mapping.mapping_utils import hard_cores_to_chip


def test_bias_capable_core_folds_bias_no_always_on():
    """When a HardCore has has_bias_capability=True, hard_cores_to_chip emits per-neuron bias
    and replaces the bias axon's always-on source with off (no k_always_on for that core).
    """
    # One hard core: 3 axons (2 inputs + 1 bias), 2 neurons. Last row = bias [0.5, -0.3].
    core_matrix = np.array([
        [0.1, 0.2],   # axon 0
        [0.3, 0.4],   # axon 1
        [0.5, -0.3],  # bias row
    ], dtype=np.float64)
    axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
        SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True),
    ]
    hc = HardCore(axons_per_core=3, neurons_per_core=2, has_bias_capability=True)
    hc.core_matrix = core_matrix
    hc.axon_sources = axon_sources
    hc.threshold = 1.0
    hc.latency = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [hc]
    mapping.output_sources = np.array([
        SpikeSource(0, 0, is_input=False, is_off=False),
        SpikeSource(0, 1, is_input=False, is_off=False),
    ])

    chip = hard_cores_to_chip(
        input_size=2,
        hardcore_mapping=mapping,
        axons_per_core=3,
        neurons_per_core=2,
        leak=0,
        weight_type=float,
    )

    assert len(chip.connections) == 1
    conn = chip.connections[0]
    # Last axon must be off, not always-on (bias folded into neuron.bias).
    assert len(conn.axon_sources) == 3
    assert conn.axon_sources[2].is_always_on_ is False
    assert conn.axon_sources[2].is_off_ is True

    assert len(chip.cores) == 1
    core = chip.cores[0]
    assert len(core.neurons) == 2
    assert core.neurons[0].bias == 0.5
    assert core.neurons[1].bias == -0.3
    # Weights should be 2 axons only (bias row dropped); padded to 3 with zero.
    assert len(core.neurons[0].weights) == 3
    assert core.neurons[0].weights[0] == 0.1
    assert core.neurons[0].weights[1] == 0.3
    assert core.neurons[0].weights[2] == 0.0


def test_no_bias_capability_keeps_always_on_and_zero_bias():
    """When has_bias_capability=False, last axon remains always-on and neuron bias is 0."""
    core_matrix = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, -0.3],
    ], dtype=np.float64)
    axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
        SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True),
    ]
    hc = HardCore(axons_per_core=3, neurons_per_core=2, has_bias_capability=False)
    hc.core_matrix = core_matrix
    hc.axon_sources = axon_sources
    hc.threshold = 1.0
    hc.latency = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [hc]
    mapping.output_sources = np.array([
        SpikeSource(0, 0, is_input=False, is_off=False),
        SpikeSource(0, 1, is_input=False, is_off=False),
    ])

    chip = hard_cores_to_chip(
        input_size=2,
        hardcore_mapping=mapping,
        axons_per_core=3,
        neurons_per_core=2,
        leak=0,
        weight_type=float,
    )

    assert len(chip.connections) == 1
    conn = chip.connections[0]
    assert conn.axon_sources[2].is_always_on_ is True

    assert len(chip.cores) == 1
    core = chip.cores[0]
    assert core.neurons[0].bias == 0.0
    assert core.neurons[1].bias == 0.0
    assert core.neurons[0].weights[2] == 0.5
    assert core.neurons[1].weights[2] == -0.3


def test_hardware_bias_field_mode1_no_always_on_row():
    """Mode 1: When HardCore has hardware_bias set, bias comes from the field,
    core_matrix is used as-is (no row peeling), and no always-on axon exists.
    """
    weight_mat = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ], dtype=np.float64)
    hw_bias = np.array([0.7, -0.5])

    hc = HardCore(axons_per_core=2, neurons_per_core=2, has_bias_capability=True)
    hc.core_matrix = weight_mat
    hc.hardware_bias = hw_bias
    hc.axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
    ]
    hc.threshold = 1.0
    hc.latency = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [hc]
    mapping.output_sources = np.array([
        SpikeSource(0, 0, is_input=False, is_off=False),
        SpikeSource(0, 1, is_input=False, is_off=False),
    ])

    chip = hard_cores_to_chip(
        input_size=2, hardcore_mapping=mapping,
        axons_per_core=2, neurons_per_core=2,
        leak=0, weight_type=float,
    )

    core = chip.cores[0]
    # Bias comes from hardware_bias field
    assert core.neurons[0].bias == pytest.approx(0.7)
    assert core.neurons[1].bias == pytest.approx(-0.5)
    # Full weight matrix used (no row peeled)
    assert core.neurons[0].weights[:2] == pytest.approx([0.1, 0.3])
    assert core.neurons[1].weights[:2] == pytest.approx([0.2, 0.4])
    # No always-on in connections
    conn = chip.connections[0]
    for src in conn.axon_sources:
        assert not src.is_always_on_


def test_hardware_bias_takes_priority_over_has_bias_capability():
    """Mode 1 takes priority: even with has_bias_capability=True and an
    always-on row present in axon_sources, hardware_bias field is used.
    """
    weight_mat = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],  # this row would be bias in Mode 2
    ], dtype=np.float64)
    hw_bias = np.array([0.9, -0.9])

    hc = HardCore(axons_per_core=3, neurons_per_core=2, has_bias_capability=True)
    hc.core_matrix = weight_mat
    hc.hardware_bias = hw_bias
    hc.axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
        SpikeSource(-2, 2, is_input=True, is_off=False),
    ]
    hc.threshold = 1.0
    hc.latency = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [hc]
    mapping.output_sources = np.array([
        SpikeSource(0, 0, is_input=False, is_off=False),
        SpikeSource(0, 1, is_input=False, is_off=False),
    ])

    chip = hard_cores_to_chip(
        input_size=3, hardcore_mapping=mapping,
        axons_per_core=3, neurons_per_core=2,
        leak=0, weight_type=float,
    )

    core = chip.cores[0]
    # hardware_bias takes priority
    assert core.neurons[0].bias == pytest.approx(0.9)
    assert core.neurons[1].bias == pytest.approx(-0.9)
    # All 3 rows are weights (not peeled)
    assert core.neurons[0].weights[:3] == pytest.approx([1.0, 3.0, 5.0])
    assert core.neurons[1].weights[:3] == pytest.approx([2.0, 4.0, 6.0])
