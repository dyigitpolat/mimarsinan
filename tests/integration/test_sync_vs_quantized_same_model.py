"""Same-model synchronized TTFS cycle vs ttfs_quantized equivalence contract."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.parity_contract import (
    ParityContractKind,
    classify_ttfs_quantized_sync_equivalence,
)
from mimarsinan.chip_simulation.ttfs.ttfs_executor import TtfsAnalyticalExecutor
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


def _two_core_mapping():
    core0 = HardCore(axons_per_core=2, neurons_per_core=2)
    core0.core_matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    core0.axon_sources = [
        SpikeSource(-2, 0, is_input=True, is_off=False),
        SpikeSource(-2, 1, is_input=True, is_off=False),
    ]
    core0.threshold = 1.0
    core0.hardware_bias = np.zeros(2, dtype=np.float64)
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
    core1.hardware_bias = np.zeros(1, dtype=np.float64)
    core1.latency = 1
    core1.available_axons = 0
    core1.available_neurons = 0

    mapping = HardCoreMapping(chip_cores=[])
    mapping.cores = [core0, core1]
    mapping.output_sources = np.array([SpikeSource(1, 0)])
    return mapping


@pytest.mark.parametrize(
    ("mode", "schedule", "expected"),
    [
        ("ttfs_quantized", "analytical", ParityContractKind.FUNCTIONAL_EQUIVALENCE),
        ("ttfs_cycle_based", "synchronized", ParityContractKind.FUNCTIONAL_EQUIVALENCE),
        ("ttfs_cycle_based", "cascaded", ParityContractKind.INAPPLICABLE),
        ("lif", None, ParityContractKind.INAPPLICABLE),
    ],
)
def test_contract_classifier(mode, schedule, expected):
    assert (
        classify_ttfs_quantized_sync_equivalence(
            spiking_mode=mode,
            schedule=schedule,
        )
        == expected
    )


def test_executor_quantized_and_synchronized_cycle_match():
    mapping = _two_core_mapping()
    inp = np.array([[0.25, 0.75], [0.10, 0.90]], dtype=np.float64)
    s = 8
    exec_ = TtfsAnalyticalExecutor()
    q = exec_.run_segment(mapping, inp, simulation_length=s, spiking_mode="ttfs_quantized")
    c = exec_.run_segment(mapping, inp, simulation_length=s, spiking_mode="ttfs_cycle_based")
    np.testing.assert_array_equal(c.inter_stage, q.inter_stage)
    for a, b in zip(c.per_core_activations, q.per_core_activations):
        np.testing.assert_array_equal(a, b)
