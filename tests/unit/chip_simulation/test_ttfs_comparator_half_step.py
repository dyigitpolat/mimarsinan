"""[E3] comparator-side half-step threads contract -> executor -> core staircase."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract
from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_input_grid_quantize
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.models.spiking.wire_semantics import ttfs_quantized_staircase_np

S = 4
X = np.array([[0.6, 0.15]], dtype=np.float64)


_W = np.diag([0.3, 1.0]).astype(np.float64)
"""Off-grid gain: grid-snapped inputs stay grid FIXED POINTS of both kernels,
so the shift is only observable through a pre-activation off the S-grid."""


def _single_stage_mapping():
    core = HardCore(axons_per_core=2, neurons_per_core=2)
    core.core_matrix = _W.copy()
    core.axon_sources = [
        SpikeSource(-2, i, is_input=True, is_off=False) for i in range(2)
    ]
    core.threshold = 1.0
    core.latency = 0
    core.available_axons = 0
    core.available_neurons = 0

    hcm = HardCoreMapping(chip_cores=[])
    hcm.cores = [core]
    hcm.output_sources = np.array([SpikeSource(0, 0), SpikeSource(0, 1)])

    return HybridHardCoreMapping(stages=[
        HybridStage(
            kind="neural",
            name="stage0",
            hard_core_mapping=hcm,
            input_map=[SegmentIOSlice(-2, 0, 2)],
            output_map=[SegmentIOSlice(0, 0, 2)],
        ),
    ])


def _contract(**overrides):
    cfg = {
        "spiking_mode": "ttfs_cycle_based",
        "thresholding_mode": "<=",
        "simulation_steps": S,
        "ttfs_cycle_schedule": "synchronized",
    }
    cfg.update(overrides)
    return SpikingDeploymentContract.from_pipeline_config(cfg)


def test_contract_flag_shifts_the_deployed_staircase():
    run = run_ttfs_hybrid_contract(
        _single_stage_mapping(), X, contract=_contract(comparator_half_step=True),
    )
    v = ttfs_input_grid_quantize(X, S) @ _W.T
    expected = ttfs_quantized_staircase_np(v, 1.0, S, comparator_half_step=True)
    np.testing.assert_array_equal(
        run.record.segments[0].cores[0].output_activation, expected[0],
    )
    # The shift is observable here: v[0] = 0.15 sits in the plain kernel's
    # dead zone at S=4 but fires under the shifted ladder.
    plain = ttfs_quantized_staircase_np(v, 1.0, S)
    assert not np.array_equal(expected, plain)


def test_contract_flag_absent_is_byte_identical_to_loose_kwargs():
    with_contract = run_ttfs_hybrid_contract(
        _single_stage_mapping(), X, contract=_contract(),
    )
    loose = run_ttfs_hybrid_contract(
        _single_stage_mapping(),
        X,
        simulation_length=S,
        spiking_mode="ttfs_cycle_based",
        ttfs_cycle_schedule="synchronized",
    )
    np.testing.assert_array_equal(
        with_contract.record.segments[0].cores[0].output_activation,
        loose.record.segments[0].cores[0].output_activation,
    )
    v = ttfs_input_grid_quantize(X, S) @ _W.T
    np.testing.assert_array_equal(
        loose.record.segments[0].cores[0].output_activation,
        ttfs_quantized_staircase_np(v, 1.0, S)[0],
    )
