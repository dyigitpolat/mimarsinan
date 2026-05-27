"""Shared TTFS hybrid contract runner and parity tolerance policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.ttfs_executor import (
    TtfsContractComputeStageResult,
    run_ttfs_contract_compute_stage,
    run_ttfs_contract_neural_stage,
    run_ttfs_hybrid_contract,
)
from mimarsinan.chip_simulation.ttfs_recorder import (
    CoreTtfsActivations,
    SegmentTtfsRecord,
    TtfsRunRecord,
    compare_ttfs_contract_records,
    compare_ttfs_hardware_records,
    compare_ttfs_records,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore_mapping import HardCore, HardCoreMapping
from mimarsinan.pipelining.simulation_factory import record_ttfs_hcm_reference


def _make_two_stage_mapping():
    core0 = HardCore(axons_per_core=1, neurons_per_core=1)
    core0.core_matrix = np.array([[2.0]], dtype=np.float64)
    core0.axon_sources = [SpikeSource(-2, 0, is_input=True, is_off=False)]
    core0.threshold = 2.0
    core0.latency = 0
    core0.available_axons = 0
    core0.available_neurons = 0

    core2 = HardCore(axons_per_core=1, neurons_per_core=1)
    core2.core_matrix = np.array([[1.0]], dtype=np.float64)
    core2.axon_sources = [SpikeSource(0, 0, is_input=False, is_off=False)]
    core2.threshold = 1.0
    core2.latency = 0
    core2.available_axons = 0
    core2.available_neurons = 0

    hcm0 = HardCoreMapping(chip_cores=[])
    hcm0.cores = [core0]
    hcm0.output_sources = np.array([SpikeSource(0, 0)])

    hcm1 = HardCoreMapping(chip_cores=[])
    hcm1.cores = [core0, core2]
    hcm1.output_sources = np.array([SpikeSource(1, 0)])

    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )

    return HybridHardCoreMapping(stages=[
        HybridStage(
            kind="neural",
            name="stage0",
            hard_core_mapping=hcm0,
            input_map=[SegmentIOSlice(-2, 0, 1)],
            output_map=[SegmentIOSlice(0, 0, 1)],
        ),
        HybridStage(
            kind="neural",
            name="stage1",
            hard_core_mapping=hcm1,
            input_map=[SegmentIOSlice(0, 0, 1)],
            output_map=[SegmentIOSlice(1, 0, 1)],
        ),
    ])


@pytest.fixture
def pipeline_config():
    return {
        "device": "cpu",
        "simulation_steps": 4,
        "spiking_mode": "ttfs",
        "firing_mode": "Default",
        "thresholding_mode": "<=",
        "spike_generation_mode": "Uniform",
        "input_shape": (1,),
        "cycle_accurate_lif_forward": False,
    }


def test_run_ttfs_hybrid_contract_matches_record_ttfs_hcm_reference(pipeline_config):
    mapping = _make_two_stage_mapping()
    inp = np.array([[0.5]], dtype=np.float64)

    direct = run_ttfs_hybrid_contract(
        mapping, inp, simulation_length=4, spiking_mode="ttfs", sample_index=0,
    )

    class _Pipe:
        config = pipeline_config

    _flow, ref = record_ttfs_hcm_reference(
        _Pipe(), mapping, torch.tensor([[0.5]], dtype=torch.float64),
    )

    assert not compare_ttfs_contract_records(direct.record, ref)


def test_contract_tolerance_stricter_than_hardware():
    ref = TtfsRunRecord(sample_index=0, simulation_length=4, spiking_mode="ttfs")
    ref.segments[1] = SegmentTtfsRecord(
        stage_index=1,
        stage_name="neural_segment_until:mean_reduce",
        schedule_segment_index=None,
        schedule_pass_index=None,
        seg_output=np.zeros(1),
        cores=[CoreTtfsActivations(30, 1, np.array([0.00043472066]))],
    )
    act = TtfsRunRecord(sample_index=0, simulation_length=4, spiking_mode="ttfs")
    act.segments[1] = SegmentTtfsRecord(
        stage_index=1,
        stage_name="neural_segment_until:mean_reduce",
        schedule_segment_index=None,
        schedule_pass_index=None,
        seg_output=np.zeros(1),
        cores=[CoreTtfsActivations(30, 1, np.array([0.0004347238]))],
    )

    assert compare_ttfs_records(ref, act)
    assert compare_ttfs_hardware_records(ref, act) == []
    assert compare_ttfs_contract_records(ref, act)


def test_compute_stage_returns_op_id_result():
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridStage, SegmentIOSlice
    from mimarsinan.mapping.ir import ComputeOp, IRSource

    import torch.nn as nn
    op = ComputeOp(
        id=7,
        name="identity",
        op_type="Identity",
        input_sources=np.array([IRSource(0, 0)], dtype=object),
        params={"module": nn.Identity()},
        output_shape=(1,),
    )
    mapping = _make_two_stage_mapping()
    mapping.stages.append(
        HybridStage(
            kind="compute",
            name="mean_op",
            compute_op=op,
            input_map=[],
            output_map=[],
        )
    )
    inp = np.array([[0.5]], dtype=np.float64)
    full = run_ttfs_hybrid_contract(
        mapping, inp, simulation_length=4, spiking_mode="ttfs",
    )
    assert 7 in full.record.compute_outputs
    stage = mapping.stages[-1]
    state = dict(full.state_buffer)
    result = run_ttfs_contract_compute_stage(mapping, stage, state, inp)
    assert isinstance(result, TtfsContractComputeStageResult)
    assert result.op_id == 7
    np.testing.assert_array_equal(result.output, full.record.compute_outputs[7])


@pytest.mark.parametrize("spiking_mode", ["ttfs", "ttfs_quantized"])
def test_incremental_neural_stage_matches_full_contract(spiking_mode):
    mapping = _make_two_stage_mapping()
    inp = np.array([[0.5]], dtype=np.float64)
    full = run_ttfs_hybrid_contract(
        mapping, inp, simulation_length=4, spiking_mode=spiking_mode,
    )
    state = {-2: inp}
    stage = mapping.stages[0]
    inc = run_ttfs_contract_neural_stage(
        mapping, stage, 0, state,
        simulation_length=4, spiking_mode=spiking_mode,
    )
    assert not compare_ttfs_contract_records(
        full.record,
        TtfsRunRecord(
            sample_index=0,
            simulation_length=4,
            spiking_mode=spiking_mode,
            segments={0: inc.segment_record},
        ),
    )
