"""[E-6, N2 follow-up F2] The sealed record states its operand regions per pass, so the record plane prices operand
programs.

N2's record schema stated no operand regions, so the record plane refused every write term of a platform that
declares an operand core type (``RECORD_OPERANDS_UNSTATED``) while the mapping plane priced them: Figure 1's energy
split and latency band could not be read off the sealed record. Each pass of a program that binds dynamic banks now
seals, per hard core, the cells of its bound-bank regions (``SegmentRecord.operand_core_cells``, declared-only: a
record of any other program keeps its bytes), and ``record_program_passes`` reads them as the mapping plane does
(``mapping_program_passes``): static cells and producer cells per tile, the same period census, the same write terms.
A record sealed without them still refuses by name.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import (
    RECORD_OPERANDS_UNSTATED, mapping_program_passes, record_program_passes,
)
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.quantities.write_law import plane_writes, record_write_census
from mimarsinan.deployment_record.schema import SegmentRecord
from mimarsinan.mapping.ir.dynamic_bank import operand_cells_by_core
from unit.chip_simulation.value_run.operand_programs import attention_program
from unit.deployment_record.g3_fixtures import constant, physics

GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)


@pytest.fixture(scope="module")
def program():
    return attention_program("softmax")


@pytest.fixture(scope="module")
def mapping(program):
    return program.packed_mapping(**GEOMETRY)


def _operand_chip(count: int = 10_000) -> dict:
    return {"cores": [{"max_axons": 64, "max_neurons": 64, "count": count, "operand": {"payload_bits": 8}}],
            "inference_batch_K": 1, "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}


def _record(schedule, platform):
    return SimpleNamespace(schedule=schedule, timing=SimpleNamespace(per_segment=(), s_global=1),
                           identity=SimpleNamespace(platform=platform),
                           utilization=SimpleNamespace(crossbar=SimpleNamespace(weight_bits=8)))


def _physics():
    return physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"),
                   endurance_cycles=constant(1e6, unit="cycles"))


def test_every_pass_seals_its_cores_operand_cells(mapping):
    schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
    banks = mapping.dynamic_bank_bindings
    neural = [s for s in mapping.stages if s.kind == "neural"]
    segments = schedule.segments()
    assert len(segments) == len(neural)
    for segment, stage in zip(segments, neural):
        assert segment.operand_core_cells == tuple(operand_cells_by_core(stage.hard_core_mapping, banks))
    assert sum(sum(s.operand_core_cells) for s in segments) > 0


def test_the_field_round_trips_and_stays_out_of_records_without_it(mapping):
    segment = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0).segments()[0]
    assert SegmentRecord.from_dict(segment.to_dict()) == segment
    bare = replace(segment, operand_core_cells=None)
    assert "operand_core_cells" not in bare.to_dict()


def test_the_record_plane_reads_the_mapping_planes_passes(mapping):
    record = _record(schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0), _operand_chip())
    sealed, built = record_program_passes(record), mapping_program_passes(mapping)
    assert sealed.operands_unstated is None
    assert [(r.core_cells, r.producer_cells, r.producer_cores, r.producer_core_cells) for r in sealed.rows] == [
        (r.core_cells, r.producer_cells, r.producer_cores, r.producer_core_cells) for r in built.rows]


def test_the_record_plane_prices_an_operand_program_as_the_mapping_plane(mapping):
    platform = _operand_chip()
    record = _record(schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0), platform)
    writes = record_write_census(record)
    assert writes is not None and writes.census is not None
    mapped = plane_writes(mapping_program_passes(mapping), platform, weight_bits=8, timesteps=1, label="mapping")
    # [U-ENERGY c-2] the sealed record states its written cores' connectivity entries and the mapping plane states
    # none: on a description without an entry width both planes price the same operand program; with the width the
    # record also prices its written cores' entries (the plane difference m-2 names)
    no_width = replace(_physics(), constants={k: v for k, v in _physics().constants.items()
                                              if k != "bytes_per_connectivity_entry"})
    sealed_terms = {t.name: t.band for t in price_write_inputs(WriteInputs.of(writes), no_width).terms}
    mapped_terms = {t.name: t.band for t in price_write_inputs(WriteInputs.of(mapped), no_width).terms}
    assert sealed_terms and sealed_terms.keys() == mapped_terms.keys()
    for name in sealed_terms:
        a, b = sealed_terms[name], mapped_terms[name]
        assert (a.low, a.nominal, a.high) == (b.low, b.nominal, b.high), name
    with_width = {t.name: t.band for t in price_write_inputs(WriteInputs.of(writes), _physics()).terms}
    mapped_width = {t.name: t.band for t in price_write_inputs(WriteInputs.of(mapped), _physics()).terms}
    assert with_width["energy_write_mj"].nominal >= mapped_width["energy_write_mj"].nominal


def test_a_record_sealed_without_operand_regions_still_refuses_by_name(mapping):
    schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
    stages = tuple(replace(s, operand_core_cells=None) if isinstance(s, SegmentRecord) else s for s in schedule.stages)
    record = _record(replace(schedule, stages=stages), _operand_chip())
    assert record_program_passes(record).operands_unstated == RECORD_OPERANDS_UNSTATED
    writes = record_write_census(record)
    assert writes is not None and writes.census is None and "operand regions" in str(writes.unavailable)


def test_a_program_without_bound_banks_seals_no_operand_field():
    from unit.chip_simulation.value_run.integer_programs import INSTANCES

    build, geometry = INSTANCES["lenet5"]
    mapping = build().packed_mapping(**geometry)
    schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
    assert all(s.operand_core_cells is None and "operand_core_cells" not in s.to_dict() for s in schedule.segments())
