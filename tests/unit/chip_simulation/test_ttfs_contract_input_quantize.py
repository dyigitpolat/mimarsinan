"""Synchronized TTFS contract sees the grid-quantized stage input (hardware contract)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_input_grid_quantize
from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
    run_ttfs_contract_neural_stage,
    run_ttfs_hybrid_contract,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping
from mimarsinan.pipelining.core.simulation_factory import record_ttfs_hcm_reference

S = 4
OFF_GRID = np.array([[0.6, 0.7]], dtype=np.float64)


def _single_stage_mapping():
    core = HardCore(axons_per_core=2, neurons_per_core=2)
    core.core_matrix = np.eye(2, dtype=np.float64)
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


def _stage0_seg_input(schedule):
    contract = run_ttfs_hybrid_contract(
        _single_stage_mapping(),
        OFF_GRID,
        simulation_length=S,
        spiking_mode="ttfs_cycle_based",
        ttfs_cycle_schedule=schedule,
    )
    mapping = _single_stage_mapping()
    state = {-2: OFF_GRID}
    inc = run_ttfs_contract_neural_stage(
        mapping, mapping.stages[0], 0, state,
        simulation_length=S, spiking_mode="ttfs_cycle_based",
        quantize_input_to_ttfs_grid=(schedule == "synchronized"),
    )
    return contract, inc.seg_input


def test_synchronized_contract_quantizes_raw_input():
    contract, seg_in = _stage0_seg_input("synchronized")
    expected = ttfs_input_grid_quantize(OFF_GRID, S)
    np.testing.assert_array_equal(seg_in, expected)
    np.testing.assert_array_equal(
        contract.record.segments[0].cores[0].output_activation,
        expected[0],
    )


def test_cascaded_contract_leaves_raw_input_continuous():
    contract, seg_in = _stage0_seg_input("cascaded")
    np.testing.assert_array_equal(seg_in, OFF_GRID)
    # Identity weights, threshold 1: activations are the quantized kernel of the
    # continuous input (analytical staircase), not the grid-snapped input.
    assert not np.array_equal(
        contract.record.segments[0].cores[0].output_activation,
        ttfs_input_grid_quantize(OFF_GRID, S)[0],
    )


@pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized"])
def test_analytical_modes_never_quantize_input(mode):
    contract = run_ttfs_hybrid_contract(
        _single_stage_mapping(),
        OFF_GRID,
        simulation_length=S,
        spiking_mode=mode,
        ttfs_cycle_schedule="synchronized",
    )
    baseline = run_ttfs_hybrid_contract(
        _single_stage_mapping(),
        OFF_GRID,
        simulation_length=S,
        spiking_mode=mode,
    )
    np.testing.assert_array_equal(
        contract.record.segments[0].cores[0].output_activation,
        baseline.record.segments[0].cores[0].output_activation,
    )


def _sync_output(round_mode):
    return run_ttfs_hybrid_contract(
        _single_stage_mapping(), OFF_GRID, simulation_length=S,
        spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized",
        spike_time_round=round_mode,
    ).record.segments[0].cores[0].output_activation


def _ttfs_quantized_output():
    return run_ttfs_hybrid_contract(
        _single_stage_mapping(), OFF_GRID, simulation_length=S,
        spiking_mode="ttfs_quantized",
    ).record.segments[0].cores[0].output_activation


def test_synchronized_ceil_is_ttfs_quantized():
    """With the ``ceil`` encode convention the segment-entry snap is a fixed point
    of the firing staircase, so synchronized == ttfs_quantized — the mathematical
    equivalence. (Round-encode vs ceil-fire was the ~7% NF↔SCM gap.)"""
    np.testing.assert_array_equal(_sync_output("ceil"), _ttfs_quantized_output())


def test_synchronized_round_diverges_from_ttfs_quantized():
    """The legacy round encode does NOT match ttfs_quantized — the bug this fixes."""
    assert not np.array_equal(_sync_output("round"), _ttfs_quantized_output())


def test_reference_passes_schedule_from_pipeline_config():
    class _Pipe:
        config = {
            "device": "cpu",
            "simulation_steps": S,
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "firing_mode": "TTFS",
            "thresholding_mode": "<=",
            "spike_generation_mode": "TTFS",
            "input_shape": (2,),
            "cycle_accurate_lif_forward": False,
        }

    _flow, ref = record_ttfs_hcm_reference(
        _Pipe(), _single_stage_mapping(),
        torch.tensor(OFF_GRID, dtype=torch.float64),
    )
    np.testing.assert_array_equal(
        ref.segments[0].cores[0].output_activation,
        ttfs_input_grid_quantize(OFF_GRID, S)[0],
    )
