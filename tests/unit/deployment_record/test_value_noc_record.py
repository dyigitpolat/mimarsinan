"""KA-G3-1 (CH-D2, record plane): the record build seals the built program's value-domain NoC wire census, and the
record's value rule prices it as the candidate plane does.

The sealed quantity is the grid-free one, ``ScheduleRecord.value_noc_wire_hops`` (declared-only): per execution
stage, every distinct (source core, neuron) a core reads from another core of the stage, times the XY hops between
their tiles on the resolved floorplan (the SANA-FE geometry SSOT ``mapping/noc/value_traffic.py`` uses). The record's
value rule reads it as ``noc_value_bytes`` at ``activation_bits``, by the candidate's own rule: zero traffic is zero
whatever the grid, other traffic without the grid is withheld by name, and a record that seals no census (every
archived record) keeps the KA-G3-1 reason."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.value_domain import (
    NO_RECORD_WIRES, NO_VALUE_NOC_GRID, record_value_noc_wire_hops,
)
from mimarsinan.deployment_record.schema.schedule import ScheduleRecord
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments, value_noc_wire_hops
from mimarsinan.mapping.noc.fragments import execution_stage_placements
from mimarsinan.mapping.noc.value_traffic import stage_value_noc_wire_hops
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.packing.hybrid_build_scheduled import PlannedPassPackingRefusal
from mimarsinan.mapping.support.residency_rules import spec_runs

from unit.deployment_record.g3_fixtures import built_chain, compute, record_with, schedule, segment
from unit.mapping.residency.executed import vehicles as V


def _source(core, neuron, *, is_input=False, special=False):
    return SimpleNamespace(core_=core, neuron_=neuron, is_input_=is_input, is_off_=special, is_always_on_=False)


def _stage(cores):
    """A neural stage of hard cores, each a list of axon sources."""
    return SimpleNamespace(kind="neural", hard_core_mapping=SimpleNamespace(
        cores=[SimpleNamespace(axon_sources=list(sources)) for sources in cores]))


#: Core 0 holds the producer; core 1 reads neurons 0, 1, 2 of it (neuron 1 twice) and one input; core 3 reads
#: neuron 0 of core 0 and neuron 4 of core 1; core 2 reads only inputs and an off source.
_STAGE = _stage([
    [_source(-1, 0, is_input=True)],
    [_source(0, 0), _source(0, 1), _source(0, 1), _source(0, 2), _source(-1, 3, is_input=True)],
    [_source(-1, 1, is_input=True), _source(-1, 0, special=True)],
    [_source(0, 0), _source(1, 4)],
])


class TestTheBuiltCensus:
    def test_the_hops_equal_the_hand_census(self):
        """2 x 2-tile chip, one core per tile, column-major: core 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1).
        Core 1 reads 3 distinct wires of core 0 at 1 hop; core 3 reads one of core 0 at 2 hops and one of core 1
        at 1 hop: 3 + 2 + 1 = 6 value-hops."""
        assert stage_value_noc_wire_hops([_STAGE], cores_per_tile=1, mesh_height=2) == 6

    def test_cores_of_one_tile_cross_no_link_and_compute_stages_carry_none(self):
        stages = [_STAGE, SimpleNamespace(kind="compute", hard_core_mapping=None)]
        assert stage_value_noc_wire_hops(stages, cores_per_tile=4, mesh_height=1) == 0

    def test_each_stage_is_mapped_from_core_zero(self):
        """Two stages are two chips' worth of placements: a wire never crosses stages (a carry, not mesh traffic)."""
        assert stage_value_noc_wire_hops([_STAGE, _STAGE], cores_per_tile=1, mesh_height=2) == 12

    def test_the_built_plane_equals_the_candidate_plane_on_the_deployed_program(self):
        """One rule on both planes over the vehicles (no two consumer softcores of one hard core read one wire
        there, the stated boundary of ``value_traffic``): the built stages' census equals the candidate's census
        of its executed stage placements, planned with the walk's runs."""
        compared = 0
        for rep in [V.random_rep(seed) for seed in range(0, 30)] + [V.branchy_repr("skew3"), V.chain_repr(0)]:
            for count in (3, 5, 8):
                try:
                    built = V.build(V.ir_of(rep), count, None)
                except PlannedPassPackingRefusal:          # the residency class, refused by name
                    continue
                walk = LayoutIRMapping(max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS, collect_wire_census=True)
                specs = walk.collect_layout_softcores(rep)
                census = census_of_walk(walk)
                frag = collect_noc_fragments(
                    softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, count)],
                    census=census, runs=spec_runs(walk), **V.capabilities().layout_kwargs())
                stages = execution_stage_placements(specs, frag.pass_placements, retimed=False, runs=frag.runs)
                candidate = value_noc_wire_hops(pair_wires=census.pair_wires, stage_placements=stages,
                                                cores_per_tile=2, mesh_height=2)
                assert stage_value_noc_wire_hops(built.stages, cores_per_tile=2, mesh_height=2) == candidate, (
                    rep, count)
                compared += 1
        assert compared >= 40


_VALUE_PLATFORM = {"core_semantics": "mvm", "cores_per_tile_resolved": 1, "tile_grid_rows_resolved": 2,
                   "tile_grid_cols_resolved": 2}


class TestTheRecordBuild:
    def test_a_value_program_on_a_resolved_floorplan_seals_its_hops(self):
        stages = SimpleNamespace(stages=[_STAGE])
        assert record_value_noc_wire_hops(stages, _VALUE_PLATFORM) == 6

    @pytest.mark.parametrize("platform", [
        {"cores_per_tile_resolved": 1, "tile_grid_rows_resolved": 2},           # a spiking program
        {"core_semantics": "mvm"},                                             # no resolved floorplan
    ])
    def test_otherwise_it_seals_nothing(self, platform):
        assert record_value_noc_wire_hops(SimpleNamespace(stages=[_STAGE]), platform) is None

    def test_the_schedule_fragment_carries_the_sealed_census_and_only_then(self):
        mapping = built_chain()
        sealed = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0, timesteps=4,
                                              pass_transfer="collapse", value_noc_wire_hops=7)
        assert sealed.value_noc_wire_hops == 7
        data = sealed.to_dict()
        assert data["value_noc_wire_hops"] == 7
        assert ScheduleRecord.from_dict(data) == sealed
        plain = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0, timesteps=4,
                                             pass_transfer="collapse")
        assert "value_noc_wire_hops" not in plain.to_dict()


def _value_record(hops=None, **platform):
    stages = (segment(1, 0, 0, [100, 50], latencies=[0, 0]), compute(2), segment(3, 1, 0, [20], latencies=[0]))
    sealed = replace(schedule(stages), value_noc_wire_hops=hops)
    return record_with(sealed, core_semantics="mvm", **platform)


class TestTheRecordReading:
    def test_a_sealed_census_is_priced_at_the_value_grid(self):
        values = from_record(_value_record(6, activation_bits=4))
        assert values.get("noc_value_bytes").value == 6 * 4 / 8
        assert values.get("noc_value_bytes").provenance == "static"
        assert values.withheld_reason("noc_value_bytes") is None

    def test_a_sealed_zero_is_zero_without_the_grid(self):
        assert from_record(_value_record(0)).get("noc_value_bytes").value == 0.0

    def test_sealed_traffic_without_the_grid_is_withheld_by_the_grids_name(self):
        values = from_record(_value_record(6))
        assert not values.has("noc_value_bytes")
        assert values.withheld_reason("noc_value_bytes") == NO_VALUE_NOC_GRID

    def test_a_record_that_seals_no_census_keeps_the_ka_g3_1_reason(self):
        values = from_record(_value_record(None, activation_bits=8))
        assert values.withheld_reason("noc_value_bytes") == NO_RECORD_WIRES


class TestTheEmissionSite:
    """[MN1] The hard-core mapping step's emission (``emit_deployment_record_hcm``) seals the census the record build
    computes for the run's platform: a value-domain run on a resolved floorplan carries the key, a spiking run none.
    The partition and utilization fragments are stubbed; the schedule fragment is the step's own."""

    def _emit(self, monkeypatch, platform):
        from mimarsinan.pipelining.pipeline_steps.mapping import deployment_record_emission as emission

        monkeypatch.setattr(emission, "compute_partition_record", lambda *args, **kwargs: None)
        monkeypatch.setattr(emission, "utilization_record_from_mapping",
                            lambda *args, **kwargs: SimpleNamespace(to_dict=dict))
        entries = {}
        step = SimpleNamespace(pipeline=SimpleNamespace(config={"simulation_steps": 4, "cores": []}),
                               add_entry=lambda key, value, kind: entries.__setitem__(key, value))
        mapping = built_chain()
        emission.emit_deployment_record_hcm(
            step, mapping, platform_constraints={**platform, "weight_bits": 8},
            scm_fragment={"reuse_plan": {"params_reloaded": 0}, "relay_cores_inserted": 0},
            programming=SimpleNamespace(neural_stages=0, programming_events=0, params_programmed=0,
                                        params_unique=0),
            crossbar_report=None, spike_gate_result=None, accuracy=0.5, observes_values=True, model=None,
            ir_graph=None, input_shape=(1,), num_classes=2, encoding_placement="subsume")
        return mapping, entries["deployment_record_hcm"]["schedule"]

    def test_a_value_run_seals_the_census_it_computes(self, monkeypatch):
        mapping, schedule_dict = self._emit(monkeypatch, _VALUE_PLATFORM)
        hops = record_value_noc_wire_hops(mapping, _VALUE_PLATFORM)
        assert hops is not None and schedule_dict["value_noc_wire_hops"] == hops
        assert ScheduleRecord.from_dict(schedule_dict).value_noc_wire_hops == hops

    def test_a_spiking_run_seals_no_census(self, monkeypatch):
        spiking = {k: v for k, v in _VALUE_PLATFORM.items() if k != "core_semantics"}
        _mapping, schedule_dict = self._emit(monkeypatch, spiking)
        assert "value_noc_wire_hops" not in schedule_dict
