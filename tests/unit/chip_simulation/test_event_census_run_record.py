"""[U-S S-1] The event census of an executed run record: per core, the recorded input counts times the used columns.

The Hard Core Mapping executor's recorder sums, per core and axon, the exact signals the core integrated inside its
window (``CoreSpikeCounts.input_spike_count``). One event is one spike arriving at one occupied cell
(``chip_simulation/synaptic_events.py``), so a core's events are its recorded arrivals times its used columns. The
record is keyed by the executor's execution-unit index (level stages included, ``run_hybrid_stages``); the census
reads the same index (``enumerate_execution_stages``, the reference runners' SSOT), refuses by name a stage or a core
the mapping holds and the record lacks, and the recorder's batch-1 limit is lifted by a per-sample loop
(``forward_with_recordings``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import enumerate_execution_stages
from mimarsinan.chip_simulation.recording.compare import compare_records
from mimarsinan.chip_simulation.recording.records import CoreSpikeCounts, RunRecord, SegmentSpikeRecord
from mimarsinan.chip_simulation.synaptic_events import (
    EventCensusRefusal,
    census_by_unit,
    census_from_run_record,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy

from unit.mapping.residency.executed import runs as R
from unit.mapping.residency.executed import vehicles as V


def _core(axons_used: int, neurons_used: int, size: int = 8):
    """A packed hard core, duck-typed: the used geometry ``CoreOccupancy`` reads."""
    return SimpleNamespace(axons_per_core=size, neurons_per_core=size, available_axons=size - axons_used,
                           available_neurons=size - neurons_used, unusable_space=0)


def _counts(index: int, inputs) -> CoreSpikeCounts:
    inputs = np.asarray(inputs, dtype=np.int64)
    return CoreSpikeCounts(core_index=index, n_in_used=len(inputs), n_out_used=1, core_latency=0,
                           has_hardware_bias=True, n_always_on_axons=0, input_spike_count=inputs,
                           output_spike_count=np.zeros(1, dtype=np.int64))


def _segment(index: int, name: str, cores) -> SegmentSpikeRecord:
    empty = np.zeros(0, dtype=np.int64)
    return SegmentSpikeRecord(stage_index=index, stage_name=name, schedule_segment_index=None,
                              schedule_pass_index=None, seg_input_rates=np.zeros((1, 0)),
                              seg_input_spike_count=empty, seg_output_spike_count=empty, cores=list(cores))


def _neural(name: str, cores):
    return SimpleNamespace(kind="neural", name=name, hard_core_mapping=SimpleNamespace(cores=list(cores)))


def _mapping(*stages):
    return SimpleNamespace(stages=list(stages))


def _chain_run(semantics, *, count: int = 2, seed: int = 0, batch: int = 1):
    ir = V.ir_of(V.chain_repr(seed=0))
    mapping = V.build(ir, count, None, retimed_level_stages=R.retimed(semantics))
    flow = R.flow(mapping, **semantics).eval()
    x = torch.rand(batch, V.IN, generator=torch.Generator().manual_seed(seed))
    return mapping, flow, x


def neural_execution_units(mapping):
    return [(i, unit) for i, _stage, unit in enumerate_execution_stages(mapping.stages) if unit.kind == "neural"]


def _hand_census_by_unit(record: RunRecord, mapping) -> dict:
    out = {}
    for index, unit in neural_execution_units(mapping):
        recorded = {c.core_index: c for c in record.segments[index].cores}
        out[index] = 0
        for core_index, core in enumerate(unit.hard_core_mapping.cores):
            columns = CoreOccupancy.from_hard_core(core).neurons_used
            if columns > 0:
                out[index] += int(recorded[core_index].input_spike_count.sum()) * int(columns)
    return out


def _hand_census(record: RunRecord, mapping) -> int:
    return sum(_hand_census_by_unit(record, mapping).values())


class TestTheExecutionUnits:
    """The census reads the record by the index the executor hands each neural execution unit, one entry each."""

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.WINDOWED_LIF], ids=["streamed", "windowed"])
    @pytest.mark.parametrize("vehicle, count", [("chain", 2), ("deep_mismatch", 6)])
    def test_the_units_are_the_records_segments_core_for_core(self, semantics, vehicle, count):
        rep = V.chain_repr(seed=0) if vehicle == "chain" else V.branchy_repr(vehicle)
        mapping = V.build(V.ir_of(rep), count, None, retimed_level_stages=R.retimed(semantics))
        if semantics is R.WINDOWED_LIF:  # the windowed run executes level stages: the index skips the fused stage
            assert any(getattr(s, "retimed_level_stages", None) for s in mapping.stages)
        flow = R.flow(mapping, **semantics).eval()
        with torch.no_grad():
            _, record = flow.forward_with_recording(torch.rand(1, V.IN, generator=torch.Generator().manual_seed(1)))
        by_unit = census_by_unit(record, mapping)
        assert sorted(by_unit) == sorted(record.segments) == [index for index, _ in neural_execution_units(mapping)]
        assert by_unit == _hand_census_by_unit(record, mapping)
        assert sum(by_unit.values()) == census_from_run_record(record, mapping) > 0


class TestTheJoin:
    def test_events_are_each_cores_recorded_arrivals_times_its_used_columns(self):
        """Core 0: 3 + 5 arrivals on 4 used columns; core 1: 2 + 0 + 7 on 3. Nothing else is an event."""
        mapping = _mapping(_neural("seg", [_core(2, 4), _core(3, 3)]))
        record = RunRecord(sample_index=0, T=4, segments={0: _segment(0, "seg", [
            _counts(0, [3, 5]), _counts(1, [2, 0, 7])])})
        assert census_from_run_record(record, mapping) == (3 + 5) * 4 + (2 + 0 + 7) * 3

    def test_a_core_with_no_used_column_needs_no_record(self):
        """Arrivals on a core that occupies no column land on nothing (the SANA-FE census's rule)."""
        mapping = _mapping(_neural("seg", [_core(2, 4), _core(2, 0)]))
        record = RunRecord(sample_index=0, T=4, segments={0: _segment(0, "seg", [_counts(0, [1, 1])])})
        assert census_from_run_record(record, mapping) == 2 * 4

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.WINDOWED_LIF], ids=["streamed", "windowed"])
    def test_on_an_executed_run_it_is_the_recorders_input_total_times_columns(self, semantics):
        mapping, flow, x = _chain_run(semantics)
        with torch.no_grad():
            _, record = flow.forward_with_recording(x)
        events = census_from_run_record(record, mapping)
        assert events == _hand_census(record, mapping)
        assert events > 0                                  # the vehicle fires: no comparison is vacuous


class TestRefusalByName:
    def test_a_core_the_record_lacks_refuses_by_name(self):
        mapping = _mapping(_neural("seg_a", [_core(2, 4), _core(3, 3)]))
        record = RunRecord(sample_index=0, T=4, segments={0: _segment(0, "seg_a", [_counts(0, [1, 1])])})
        with pytest.raises(EventCensusRefusal, match=r"core 1.*seg_a|seg_a.*core 1") as refused:
            census_from_run_record(record, mapping)
        assert isinstance(refused.value, ToolchainRefusal)

    def test_a_stage_the_record_lacks_refuses_by_name(self):
        compute = SimpleNamespace(kind="compute", name="host", hard_core_mapping=None)
        mapping = _mapping(_neural("seg_a", [_core(1, 1)]), compute, _neural("seg_b", [_core(1, 1)]))
        record = RunRecord(sample_index=0, T=4, segments={0: _segment(0, "seg_a", [_counts(0, [1])])})
        with pytest.raises(EventCensusRefusal, match=r"2.*seg_b"):
            census_from_run_record(record, mapping)

    def test_counts_over_other_axons_than_the_core_uses_refuse_by_name(self):
        """The record's axon count must be the core's used axons: otherwise the index books disagree."""
        mapping = _mapping(_neural("seg", [_core(3, 2)]))
        record = RunRecord(sample_index=0, T=4, segments={0: _segment(0, "seg", [_counts(0, [1, 1])])})
        with pytest.raises(EventCensusRefusal, match="axons"):
            census_from_run_record(record, mapping)


class TestTheBatchOneLimitLifted:
    """``forward_with_recording`` records at batch 1; ``forward_with_recordings`` loops a batch sample by sample."""

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.WINDOWED_LIF], ids=["streamed", "windowed"])
    def test_a_batch_gives_one_record_per_sample_equal_to_its_single_run(self, semantics):
        mapping, flow, x = _chain_run(semantics, batch=3, seed=2)
        with torch.no_grad():
            out, records = flow.forward_with_recordings(x, first_sample_index=5)
        assert [r.sample_index for r in records] == [5, 6, 7]
        assert out.shape[0] == 3
        events = []
        for i, record in enumerate(records):
            with torch.no_grad():
                single_out, single = flow.forward_with_recording(x[i:i + 1], sample_index=5 + i)
            assert compare_records(single, record) == []
            assert torch.equal(out[i:i + 1], single_out)
            events.append(census_from_run_record(record, mapping))
        assert len(set(events)) > 1                       # the samples differ, so the per-sample census is read
