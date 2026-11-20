"""[U-ENERGY delta round] The verify's minors m-1, m-2, m-4 and m-6, each a named test. Every constant here is a test
constant.

- m-1: a load's census leaves out the producer's per-inference writes: a program with producer-written cores (an
  operand region) loads and reprograms exactly what the same program's static cells alone load.
- m-2: a sealed record whose segment totals differ from its cores' sums is priced by the cores and names the
  disagreement in each load's basis; a consistent record names nothing.
- m-4: the chip simulation's grounded weight-reuse band says that it holds no static power.
- m-6: on a chip of several write technologies the write law names a plane's own reason for the entries it does not
  state, and reads each pass's wall back where a census of the same program priced it."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.report import absolute_pricing_for_record
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.quantities.planes.loads import CANDIDATE_NO_ENTRIES, load_inputs
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema import SegmentRecord

BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0, "t_program_per_byte": 7.0,
        "p_static_per_core": 0.25, "p_static_global": 4.0}
LOAD_TERMS = ("latency_load_s", "energy_load_mj", "energy_static_load_mj", "latency_programming_s",
              "energy_programming_mj", "energy_static_programming_mj")


def physics(constants):
    return profile_from_dict({
        "format_version": 1, "name": "minors", "display_name": "minors", "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in constants.items()}})


def _q():
    values = {"synaptic_events": 1e6, "cores_physical": 8, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _bands(pricing, names=LOAD_TERMS):
    terms = {t.name: t for t in pricing.terms}
    return {n: (terms[n].band.low, terms[n].band.nominal, terms[n].band.high) for n in names if n in terms}


# ------------------------------------------------------------------------------------------------------------ m-1
def _rows(producer: bool):
    """Two written passes; with ``producer`` the second core of each holds an operand region of 30 and 12 cells
    beside its static cells (producer-written per inference)."""
    def row(i, cells, operand):
        extra = ({"producer_cells": sum(operand), "producer_cores": frozenset(k for k, o in enumerate(operand) if o),
                  "producer_core_cells": tuple(operand)} if producer else {})
        return PassRow(segment_index=i, resident=False, core_cells=tuple(cells), marks=frozenset(),
                       core_extents=tuple((c // 4, 4) for c in cells), compute_steps=3, **extra)
    return ProgramPasses(rows=(row(0, (40, 20), (0, 30)), row(1, (32, 16), (0, 12))))


class TestM1TheLoadsLeaveOutTheProducersWrites:
    def test_a_program_with_producer_written_cores_loads_its_static_cells_alone(self):
        platform = {"cores": [{"max_axons": 16, "max_neurons": 16, "count": 2}], "weight_bits": 4}
        ph = physics(BASE)
        with_operands = _bands(price_absolute(_q(), ph, loads=load_inputs(_rows(True), platform, weight_bits=4,
                                                                          label="test")))
        static_alone = _bands(price_absolute(_q(), ph, loads=load_inputs(_rows(False), platform, weight_bits=4,
                                                                         label="test")))
        assert set(with_operands) == set(LOAD_TERMS) == set(static_alone)
        assert with_operands == static_alone


# ------------------------------------------------------------------------------------------------------------ m-2
def _record(constants, **stage_totals):
    from unit.deployment_record.record_fixtures import make_rewritten_record
    record = make_rewritten_record()
    stages = []
    for stage in record.schedule.stages:
        if isinstance(stage, SegmentRecord):
            sums = {"connectivity_entries": sum(c.connectivity_entries for c in stage.cores),
                    "params_bytes": sum(c.params_bytes for c in stage.cores)}
            stage = replace(stage, **{**sums, **stage_totals})
        stages.append(stage)
    record = replace(record, schedule=replace(record.schedule, stages=tuple(stages)))
    return replace(record, identity=replace(record.identity, platform={
        **record.identity.platform, "platform_physics_resolved": physics(constants).to_dict()}))


class TestM2ASealedRecordsTotalsAgainstItsCores:
    CONSTANTS = {**BASE, "bytes_per_connectivity_entry": 8.0}

    def test_totals_that_differ_from_the_cores_are_named_in_each_loads_basis(self):
        consistent = absolute_pricing_for_record(_record(self.CONSTANTS))
        odd = absolute_pricing_for_record(_record(self.CONSTANTS, connectivity_entries=99, params_bytes=1))
        assert _bands(odd) == _bands(consistent)  # priced by the cores, as the design rules
        terms = {t.name: t for t in odd.terms}
        for name in ("latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"):
            text = terms[name].source
            assert "sealed totals" in text and "connectivity_entries 99" in text and "params_bytes 1" in text, name
            assert "its cores' sums" in text, name

    def test_a_record_whose_totals_are_its_cores_sums_names_nothing(self):
        terms = {t.name: t for t in absolute_pricing_for_record(_record(self.CONSTANTS)).terms}
        for name in ("latency_load_s", "energy_load_mj", "latency_programming_s", "energy_programming_mj"):
            assert "sealed totals" not in terms[name].source, name


# ------------------------------------------------------------------------------------------------------------ m-4
class TestM4TheGroundedWeightReuseBand:
    def test_the_band_says_it_holds_no_static_power(self):
        from mimarsinan.chip_simulation import weight_reuse_cost_model
        from mimarsinan.chip_simulation.cost_extraction import CostRecord
        for doc in (CostRecord.reuse_mj_band.__doc__, weight_reuse_cost_model.phase_cost_band.__doc__):
            assert "dynamic only, no static power" in (doc or ""), doc


# ------------------------------------------------------------------------------------------------------------ m-6
S, T = (16, 16), (8, 8)  # a core type of the chip's write physics, a core type of a named write physics


def _split_loads():
    platform = {"cores": [{"max_axons": 16, "max_neurons": 16, "count": 1, "write_granularity": "cell"},
                          {"max_axons": 8, "max_neurons": 8, "count": 1, "write_granularity": "cell",
                           "write_physics": "neurram"}], "weight_bits": 8}
    rows = tuple(PassRow(segment_index=i, resident=False, core_cells=cells, marks=frozenset(),
                         core_extents=tuple((c, 1) for c in cells), compute_steps=1, stage_latencies=(0,),
                         core_types=(S, T)) for i, cells in enumerate(((10, 7), (12, 5))))
    return load_inputs(ProgramPasses(rows=rows), platform, weight_bits=8, label="candidate",
                       entries_absent=CANDIDATE_NO_ENTRIES)


class TestM6TheSeveralTechnologiesWritePath:
    def test_the_planes_reason_for_unstated_entries_is_named(self):
        loads = _split_loads()
        assert loads.first.writes.by_technology is not None and len(loads.first.writes.by_technology.parts) == 2
        pricing = price_absolute(_q(), get_platform_physics("generic_estimated_22nm"), loads=loads)
        terms = {t.name: t for t in pricing.terms}
        assert "energy_load_mj" in terms
        assert CANDIDATE_NO_ENTRIES in terms["energy_load_mj"].source

    def test_each_passs_wall_is_read_back_across_the_two_loads(self):
        from mimarsinan.deployment_record.cost.absolute.walls.write_walls import PassWalls
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
        loads, ph = _split_loads(), get_platform_physics("generic_estimated_22nm")
        walls = PassWalls()
        first = price_write_inputs(WriteInputs.of(loads.first.writes), ph, lifetime=False, walls=walls)
        priced = walls.priced
        period = price_write_inputs(WriteInputs.of(loads.period.writes), ph, lifetime=False, walls=walls)
        assert first.exposed is not None and period.exposed is not None
        assert priced > 0 and walls.priced == priced  # the period's passes are read back, never priced again
        alone = price_write_inputs(WriteInputs.of(loads.period.writes), ph, lifetime=False)
        assert (period.exposed.low, period.exposed.nominal, period.exposed.high) == pytest.approx(
            (alone.exposed.low, alone.exposed.nominal, alone.exposed.high), rel=1e-12)
