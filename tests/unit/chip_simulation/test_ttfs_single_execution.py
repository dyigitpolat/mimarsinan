"""TTFS contract stage runs each segment ONCE (W3 wall: no membrane re-execution)."""

from __future__ import annotations

import numpy as np
import pytest

import mimarsinan.chip_simulation.ttfs.ttfs_segment as ttfs_segment_mod
from mimarsinan.chip_simulation.ttfs.segment_arrays import (
    segment_ttfs_arrays_from_mapping,
)
from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
    TtfsAnalyticalExecutor,
    run_ttfs_contract_neural_stage,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping


def _make_hcm(seed=0):
    rng = np.random.default_rng(seed)
    core0 = HardCore(axons_per_core=3, neurons_per_core=2)
    core0.core_matrix = rng.normal(size=(3, 2)).astype(np.float64)
    core0.axon_sources = [
        SpikeSource(-2, i, is_input=True, is_off=False) for i in range(3)
    ]
    core0.threshold = 1.5
    core0.latency = 0
    core0.available_axons = 0
    core0.available_neurons = 0

    core1 = HardCore(axons_per_core=2, neurons_per_core=2)
    core1.core_matrix = rng.normal(size=(2, 2)).astype(np.float64)
    core1.axon_sources = [
        SpikeSource(0, i, is_input=False, is_off=False) for i in range(2)
    ]
    core1.threshold = 1.0
    core1.latency = 1
    core1.available_axons = 0
    core1.available_neurons = 0

    hcm = HardCoreMapping(chip_cores=[])
    hcm.cores = [core0, core1]
    hcm.output_sources = np.array([SpikeSource(1, 0), SpikeSource(1, 1)])
    return hcm


def _make_stage(hcm):
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        HybridHardCoreMapping,
        HybridStage,
        SegmentIOSlice,
    )

    stage = HybridStage(
        kind="neural",
        name="stage0",
        hard_core_mapping=hcm,
        input_map=[SegmentIOSlice(-2, 0, 3)],
        output_map=[SegmentIOSlice(0, 0, 2)],
    )
    mapping = HybridHardCoreMapping(stages=[stage])
    return mapping, stage


class TestRunSegmentCarriesMembrane:
    @pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized"])
    def test_membrane_in_result_matches_membrane_voltages(self, mode):
        hcm = _make_hcm()
        inp = np.random.default_rng(1).uniform(0, 1, size=(4, 3))
        executor = TtfsAnalyticalExecutor()
        result = executor.run_segment(
            hcm, inp, simulation_length=4, spiking_mode=mode,
        )
        reference = executor.membrane_voltages(
            hcm, inp, simulation_length=4, spiking_mode=mode,
        )
        assert result.membrane_voltages is not None
        assert len(result.membrane_voltages) == len(reference)
        for got, want in zip(result.membrane_voltages, reference):
            np.testing.assert_array_equal(got, want)


class TestContractStageSingleExecution:
    def test_neural_stage_runs_the_segment_exactly_once(self, monkeypatch):
        hcm = _make_hcm()
        mapping, stage = _make_stage(hcm)
        calls = []
        original = ttfs_segment_mod._run_ttfs_segment_ordered

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            ttfs_segment_mod, "_run_ttfs_segment_ordered", counted
        )
        state = {-2: np.random.default_rng(2).uniform(0, 1, size=(4, 3))}
        out = run_ttfs_contract_neural_stage(
            mapping, stage, 0, state,
            simulation_length=4, spiking_mode="ttfs_quantized",
        )
        assert len(calls) == 1, (
            f"segment executed {len(calls)} times; membrane voltages must come "
            "from the same single execution"
        )
        assert out.membrane_voltages is not None
        assert 0 in state

    def test_stage_outputs_unchanged_by_single_execution(self):
        # Golden agreement: the stage result equals an independent
        # double-execution reference computed straight from the kernels.
        hcm = _make_hcm()
        mapping, stage = _make_stage(hcm)
        inp = np.random.default_rng(3).uniform(0, 1, size=(4, 3))
        state = {-2: inp.copy()}
        out = run_ttfs_contract_neural_stage(
            mapping, stage, 0, state,
            simulation_length=4, spiking_mode="ttfs_quantized",
        )
        seg = segment_ttfs_arrays_from_mapping(hcm)
        want_out, _, want_membrane = ttfs_segment_mod._run_ttfs_segment_ordered(
            seg, inp, simulation_length=4, spiking_mode="ttfs_quantized",
        )
        np.testing.assert_array_equal(out.neural_result.inter_stage, want_out)
        for got, want in zip(out.membrane_voltages, want_membrane):
            np.testing.assert_array_equal(got, want)


class TestSegmentArraysCache:
    def test_arrays_cached_per_mapping_instance(self):
        hcm = _make_hcm()
        first = segment_ttfs_arrays_from_mapping(hcm)
        second = segment_ttfs_arrays_from_mapping(hcm)
        assert first is second

    def test_distinct_mappings_get_distinct_arrays(self):
        first = segment_ttfs_arrays_from_mapping(_make_hcm(seed=0))
        second = segment_ttfs_arrays_from_mapping(_make_hcm(seed=1))
        assert first is not second
        assert not np.array_equal(
            first.core_params[0], second.core_params[0]
        )
