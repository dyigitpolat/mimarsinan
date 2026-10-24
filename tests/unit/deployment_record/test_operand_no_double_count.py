"""N2 design §2.6 and §2.7: payload bytes are priced under programming terms only, never as carry; the streamed Q
and P cross host segments as host-boundary values; a host-sourced write adds the transfer of the payload bytes
into the chip, an on-chip write-back does not."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.cost.absolute.writes import price_write_terms
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, write_policy_of
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.mappers.operand import OperandContractionMapper
from mimarsinan.mapping.support.schedule.pass_carry import carried_wire_spans
from unit.chip_simulation.value_run.operand_programs import attention_program, operand_platform
from unit.deployment_record.g3_fixtures import constant, physics

GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)


@pytest.fixture(scope="module")
def program():
    return attention_program("softmax")


def _payload_nodes(mapping):
    return {b.source_node_id for b in mapping.dynamic_bank_bindings.values()}


def test_payload_values_enter_no_segment_as_a_boundary_input(program):
    mapping = program.packed_mapping(**GEOMETRY)
    writes = _payload_nodes(mapping)
    assert writes
    for stage in mapping.stages:
        if stage.kind == "neural":
            assert not {int(s.node_id) for s in stage.input_map if s.node_id is not None} & writes


def test_payload_cells_are_producer_writes_never_backing_store_or_carry(program):
    mapping = program.packed_mapping(**GEOMETRY)
    bindings = mapping.dynamic_bank_bindings
    passes = mapping_program_passes(mapping)
    stages = [s for s in mapping.stages if s.kind == "neural"]
    for row, stage in zip(passes.rows, stages):
        segment = stage.hard_core_mapping
        operand = [sum(p["axons"] * p["neurons"] for p in placements if p.get("weight_bank_id") in bindings)
                   for placements in segment.soft_core_placements_per_hard_core]
        rectangles = [CoreOccupancy.from_hard_core(core).cells_used for core in segment.cores]
        assert row.core_cells == tuple(r - o for r, o in zip(rectangles, operand))
        assert row.producer_cells == sum(operand)
    census = period_census(passes, weight_bits=8, capacity={(64, 64): 3}, bytes_of=lambda e: e).census
    regions = sum(p["axons"] * p["neurons"] for s in stages
                  for placements in s.hard_core_mapping.soft_core_placements_per_hard_core for p in placements
                  if p.get("weight_bank_id") in bindings)
    assert sum(p.written_cells_producer for p in census.passes) == regions
    for row, p in zip(passes.rows, census.passes):
        written = [] if p.resident_cores == p.cores else [i for i in range(p.cores) if i not in row.marks]
        assert p.payload_bytes == sum(params_bytes(row.core_cells[i], 8) for i in written)
    writes = _payload_nodes(mapping)
    assert not {node for node, *_ in carried_wire_spans(mapping.stages)} & writes


def test_q_and_p_cross_their_host_segments_as_host_boundary_values(program):
    mapping = program.identity_mapping()
    scores, values = [m for m in program.model.get_mapper_repr().execution_order()
                      if isinstance(m, OperandContractionMapper)][:2]
    stream_banks = {"scores": scores, "values": values}
    inputs = set()
    for stage in mapping.stages:
        if stage.kind == "neural":
            inputs |= {int(s.node_id) for s in stage.input_map if s.node_id is not None}
    for name, mapper in stream_banks.items():
        producers = mapper.streamed_producer_nodes(program.ir)
        assert producers and producers <= inputs, name


def _rows():
    return ProgramPasses(rows=(PassRow(
        segment_index=0, resident=False, core_cells=(0,), marks=frozenset(), core_extents=((0, 0),),
        compute_steps=1, stage_latencies=(0,), core_types=((32, 32),), producer_cells=64,
        producer_cores=frozenset({0})),))


def _priced(write_source):
    census = period_census(_rows(), weight_bits=8, capacity={(32, 32): 1})
    policy = WritePolicy(batch_k=4, write_granularity="cell", core_capacity=(((32, 32), 1),), physical_cores=1,
                         producer_write_source=write_source, payload_bits=8)
    target = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    return price_write_terms(census.census, target, policy, written_extents=census.written_extents,
                             most_written=census.most_written_core,
                             producer_slots=census.producer_slots), target


def test_a_host_sourced_write_adds_the_payload_transfer_and_an_on_chip_write_back_does_not():
    host, target = _priced("host")
    on_chip, _ = _priced("on_chip")
    assert host.energy is not None and on_chip.energy is not None
    transfer = target.band("e_dma_per_byte").nominal * 64 * 1e3
    assert host.energy.nominal - on_chip.energy.nominal == pytest.approx(transfer, rel=1e-12)
    assert "e_dma_per_byte" in host.energy.basis and "transfer" in host.energy.basis


def test_the_write_source_is_read_off_the_operand_core_type():
    platform = {**operand_platform(write_source="on_chip"), "inference_batch_K": 4}
    policy = write_policy_of(platform)
    assert policy is not None and policy.producer_write_source == "on_chip" and policy.payload_bits == 8
    host = write_policy_of({**operand_platform(), "inference_batch_K": 4})
    assert host is not None and host.producer_write_source == "host"
    plain = write_policy_of({"cores": [{"max_axons": 8, "max_neurons": 8}], "inference_batch_K": 4})
    assert plain is not None and plain.producer_write_source is None
