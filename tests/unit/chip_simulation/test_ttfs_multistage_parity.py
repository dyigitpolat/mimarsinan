"""Multi-stage TTFS: inter-stage buffer must carry activations, not spike/T."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_semantics import (
    NeuralSegmentResult,
    lif_inter_stage_from_spike_counts,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.ttfs_executor import TtfsAnalyticalExecutor
from mimarsinan.chip_simulation.ttfs_recorder import compare_ttfs_records
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping
from mimarsinan.pipelining.simulation_factory import record_ttfs_hcm_reference


def _make_two_stage_mapping():
    """Stage0: 1 input -> core0 -> output. Stage1: reads core0 -> core2."""
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

    from mimarsinan.mapping.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )

    stage0 = HybridStage(
        kind="neural",
        name="stage0",
        hard_core_mapping=hcm0,
        input_map=[SegmentIOSlice(-2, 0, 1)],
        output_map=[SegmentIOSlice(0, 0, 1)],
    )
    stage1 = HybridStage(
        kind="neural",
        name="stage1",
        hard_core_mapping=hcm1,
        input_map=[SegmentIOSlice(0, 0, 1)],
        output_map=[SegmentIOSlice(1, 0, 1)],
    )
    return HybridHardCoreMapping(stages=[stage0, stage1])


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


def test_wrong_buffer_spike_over_t_breaks_stage1(pipeline_config):
    """Stage-0 per-core parity can pass while stage-1 fails if buffer uses spike/T."""
    mapping = _make_two_stage_mapping()
    inp = np.array([[0.5]], dtype=np.float64)
    state_wrong: dict = {-2: inp}
    executor = TtfsAnalyticalExecutor()
    T = int(pipeline_config["simulation_steps"])

    def on_neural(_idx, stage, buf):
        seg_in = assemble_segment_input_numpy(
            stage.input_map, buf, 1, dtype=np.float64,
        )
        result = executor.run_segment(
            stage.hard_core_mapping, seg_in,
            simulation_length=T, spiking_mode="ttfs",
        )
        if stage.name == "stage0":
            wrong = lif_inter_stage_from_spike_counts(
                np.zeros(1, dtype=np.int64),
                T,
            )
            store_segment_output_numpy(stage.output_map, buf, wrong)
        else:
            store_neural_segment_output("ttfs", stage.output_map, buf, result)

    run_hybrid_stages(mapping, state_wrong, on_neural=on_neural, on_compute=lambda *_: None)
    stage1_in_wrong = assemble_segment_input_numpy(
        mapping.stages[1].input_map, state_wrong, 1, dtype=np.float64,
    )

    state_right: dict = {-2: inp}
    run_hybrid_stages(
        mapping, state_right,
        on_neural=lambda _i, st, b: store_neural_segment_output(
            "ttfs", st.output_map, b,
            executor.run_segment(
                st.hard_core_mapping,
                assemble_segment_input_numpy(st.input_map, b, 1, dtype=np.float64),
                simulation_length=T, spiking_mode="ttfs",
            ),
        ),
        on_compute=lambda *_: None,
    )
    stage1_in_right = assemble_segment_input_numpy(
        mapping.stages[1].input_map, state_right, 1, dtype=np.float64,
    )

    assert stage1_in_wrong[0, 0] < stage1_in_right[0, 0]


def test_store_neural_segment_output_matches_executor(pipeline_config):
    mapping = _make_two_stage_mapping()
    state: dict = {-2: np.array([[0.5]], dtype=np.float64)}
    executor = TtfsAnalyticalExecutor()
    T = int(pipeline_config["simulation_steps"])

    run_hybrid_stages(
        mapping, state,
        on_neural=lambda _i, st, b: store_neural_segment_output(
            "ttfs", st.output_map, b,
            executor.run_segment(
                st.hard_core_mapping,
                assemble_segment_input_numpy(st.input_map, b, 1, dtype=np.float64),
                simulation_length=T, spiking_mode="ttfs",
            ),
        ),
        on_compute=lambda *_: None,
    )

    class _Pipe:
        config = pipeline_config

    import torch

    _flow, ref = record_ttfs_hcm_reference(
        _Pipe(), mapping, torch.tensor([[0.5]], dtype=torch.float64),
    )
    act_record = __import__(
        "mimarsinan.chip_simulation.ttfs_recorder", fromlist=["TtfsRunRecord"],
    ).TtfsRunRecord(
        sample_index=0, simulation_length=T, spiking_mode="ttfs",
    )
    for si, stage in enumerate(mapping.stages):
        seg_in = assemble_segment_input_numpy(
            stage.input_map, state, 1, dtype=np.float64,
        )
        result = executor.run_segment(
            stage.hard_core_mapping, seg_in,
            simulation_length=T, spiking_mode="ttfs",
        )
        from mimarsinan.chip_simulation.ttfs_recorder import (
            CoreTtfsActivations,
            SegmentTtfsRecord,
        )
        from mimarsinan.mapping.core_geometry import used_neurons

        cores = []
        for ci, core in enumerate(stage.hard_core_mapping.cores):
            n = used_neurons(core, min_one=True)
            if n <= 0:
                continue
            from mimarsinan.chip_simulation.ttfs_recorder import (
                normalize_core_output_activation,
            )

            cores.append(CoreTtfsActivations(
                core_index=ci,
                n_out_used=n,
                output_activation=normalize_core_output_activation(
                    result.per_core_activations[ci], n_out_used=n,
                ),
            ))
        act_record.segments[si] = SegmentTtfsRecord(
            stage_index=si, stage_name=stage.name,
            schedule_segment_index=None, schedule_pass_index=None,
            seg_output=result.inter_stage[0], cores=cores,
        )

    from mimarsinan.chip_simulation.ttfs_recorder import compare_ttfs_contract_records

    assert not compare_ttfs_contract_records(ref, act_record)
