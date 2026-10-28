"""N2 design §2.6: the payload cells a pass's operand cores take per inference fill S0's
``PassCensus.written_cells_producer`` on the mapping and candidate planes and equal G7's census at C3; a plane that
cannot state them (the sealed record, the search's candidate path) refuses its write terms by name under an operand
core type; producer write energy and time do not amortize over K, backing-store terms do; the period census's one
slot rule places operand regions as it places static tiles, so the wear is exact and never divided by K; a lifetime
below ``min_lifetime_hours`` is inadmissible by name."""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.program_census import candidate_program_passes, mapping_program_passes
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.cost.absolute.writes import (
    ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES, LIFETIME_VERDICT, WriteInputs, price_write_inputs,
    price_write_terms,
)
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.quantities.write_law import (
    WritePolicy, cell_wear, plane_writes, record_write_census, write_quantities,
)
from mimarsinan.mapping.capability.ladder import HOST_PAYLOAD_WRITE, capability_level
from mimarsinan.mapping.capability.pricing_inputs import level_period_census
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc.fragments import collect_noc_fragments
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from unit.chip_simulation.value_run.operand_programs import attention_block, attention_program
from unit.deployment_record.g3_fixtures import constant, physics

GEOMETRY = dict(axons=64, neurons=64, count=3, passes=128)


def _physics(**extra):
    """generic 22 nm with a declared write time: every write term priced."""
    return physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"), **extra)


_VALUE = type("Semantics", (), {"retimed": False, "is_cycle": False, "is_cascade": False,
                                "latency_group_count": 1, "chip_latency": None})()


@pytest.fixture(scope="module")
def program():
    return attention_program("softmax")


def _bank_cells(ir) -> int:
    return sum(b.payload_shape[0] * b.payload_shape[1] for b in ir.dynamic_bank_bindings.values())


def _region_cells(mapping) -> int:
    """The cells of every placed operand region (each streamed instance holds its whole bank)."""
    banks = mapping.dynamic_bank_bindings
    return sum(p["axons"] * p["neurons"] for s in mapping.stages if s.kind == "neural"
               for placements in s.hard_core_mapping.soft_core_placements_per_hard_core for p in placements
               if p.get("weight_bank_id") in banks)


def test_the_mapping_plane_states_the_written_region_cells_per_inference(program):
    mapping = program.packed_mapping(**GEOMETRY)
    passes = mapping_program_passes(mapping)
    instances = {b: sum(1 for n in program.ir.get_neural_cores() if n.weight_bank_id == b)
                 for b in mapping.dynamic_bank_bindings}
    assert _region_cells(mapping) == sum(mapping.dynamic_bank_bindings[b].cells * k for b, k in instances.items())
    assert sum(row.producer_cells for row in passes.rows) == _region_cells(mapping)
    census = period_census(passes, weight_bits=8, capacity={(64, 64): 3}).census
    assert sum(p.written_cells_producer for p in census.passes) == _region_cells(mapping)


def test_producer_cells_never_count_as_backing_store_cells(program):
    mapping = program.packed_mapping(**GEOMETRY)
    rows = mapping_program_passes(mapping).rows
    from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
    used = sum(CoreOccupancy.from_hard_core(c).cells_used for s in mapping.stages if s.kind == "neural"
               for c in s.hard_core_mapping.cores)
    assert sum(sum(row.core_cells) for row in rows) == used - _region_cells(mapping)


def test_the_candidate_plane_states_the_same_bank_cells(program):
    softcores = list(program.ir.layout_softcores)
    noc = collect_noc_fragments(softcores=softcores, core_types=[LayoutHardCoreType(64, 64, 3)], census=None,
                                allow_scheduling=True, allow_neuron_splitting=True, allow_coalescing=False,
                                max_schedule_passes=128)
    passes = candidate_program_passes(softcores, noc, semantics=_VALUE, timesteps=None,
                                      dynamic_banks=frozenset(program.ir.dynamic_bank_bindings))
    assert sum(row.producer_cells for row in passes.rows) == _region_cells(program.packed_mapping(**GEOMETRY))


def test_the_payload_writes_equal_g7s_census_at_c3():
    """G7's C3 census writes each stationary operand once per inference: exactly what the program's payload
    write ops write; the packed program then places each bank in one region per streamed instance."""
    from mimarsinan.mapping.support.operand_write import OperandWrite
    model = attention_block()
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(copy.deepcopy(model), (3, 8, 8), 5, device="cpu", packaging=MVM_PACKAGING)
        repr_ = flow.get_mapper_repr()
        repr_.assign_perceptron_indices()
        g1_ir = IRMapping(q_max=127, firing_mode="Default").map(repr_)
    planned = level_period_census(g1_ir, capability_level("C3"), value_bits=8, payload_write=HOST_PAYLOAD_WRITE,
                                  bytes_of_cells=int)  # [CL4] the census's payload bytes; the cells compared here
    executed = attention_program("softmax")
    written = sum(n.params["module"].payload_cells() for n in executed.ir.get_compute_ops()
                  if isinstance(n.params.get("module"), OperandWrite))
    # [CL4] the level's period census is S0's PassPeriodCensus of its passes: their producer cells per inference
    assert sum(p.written_cells_producer for p in planned.passes) == _bank_cells(executed.ir) == written


def _static_row(segment, cells):
    return PassRow(segment_index=segment, resident=False, core_cells=(cells,), marks=frozenset(),
                   core_extents=((cells, 1),), compute_steps=1, stage_latencies=(0,), core_types=((64, 64),))


def _producer_row(segment, cells=20):
    return PassRow(segment_index=segment, resident=False, core_cells=(0,), marks=frozenset(), core_extents=((0, 0),),
                   compute_steps=1, stage_latencies=(0,), core_types=((32, 32),), producer_cells=cells,
                   producer_cores=frozenset({0}))


def _fixture_rows():
    """Two static passes (40 and 30 cells) on one 64 x 64 core rewrite from the backing store every period; one
    pass holds 20 producer cells on the 32 x 32 operand core."""
    return ProgramPasses(rows=(_static_row(0, 40), _producer_row(1), _static_row(2, 30)))


CAPACITY = {(64, 64): 1, (32, 32): 1}


def _policy(k: int, **extra) -> WritePolicy:
    return WritePolicy(batch_k=k, write_granularity="cell", core_capacity=tuple(sorted(CAPACITY.items())),
                       physical_cores=2, **extra)


def test_producer_terms_stay_constant_in_k_and_backing_store_terms_scale_by_one_over_k():
    census = period_census(_fixture_rows(), weight_bits=8, capacity=CAPACITY)
    energies = {}
    times = {}
    for k in (1, 16, 64):
        priced = price_write_terms(census.census, _physics(), _policy(k), written_extents=census.written_extents,
                                   most_written=census.most_written_core)
        assert priced.energy is not None and priced.exposed is not None
        energies[k], times[k] = priced.energy.nominal, priced.exposed.nominal
    static = (energies[1] - energies[64]) / (1 - 1 / 64)
    producer = energies[1] - static
    assert producer > 0 and static > 0
    assert energies[16] == pytest.approx(static / 16 + producer, rel=1e-12)
    static_t = (times[1] - times[64]) / (1 - 1 / 64)
    assert times[16] == pytest.approx(static_t / 16 + (times[1] - static_t), rel=1e-12)


def test_placed_operand_cores_make_the_producer_wear_exact():
    census = period_census(_fixture_rows(), weight_bits=8, capacity=CAPACITY)
    assert census.producer_slots == ((0, 1),)
    wear, reason = cell_wear(census.census, _policy(16), census.most_written_core, census.producer_slots)
    assert reason is None and wear is not None and wear.exact and wear.low == 1.0


def test_banks_rotated_through_fewer_operand_cores_wear_by_their_rotation():
    rotated = ProgramPasses(rows=(_static_row(0, 40), _producer_row(1), _producer_row(2), _static_row(3, 30)))
    census = period_census(rotated, weight_bits=8, capacity=CAPACITY)
    assert census.producer_slots == ((0, 2),)
    wear, _ = cell_wear(census.census, _policy(16), census.most_written_core, census.producer_slots)
    assert wear is not None and wear.exact and wear.low == 2.0


def _shared_core_rows(producer_passes: int = 7) -> ProgramPasses:
    """One 64 x 64 core: one static pass (40 cells, rewritten every period) and seven passes each holding a 20-cell
    operand region on that same core (the S-core passes of the executed ViT-Lite declaration have this shape)."""
    return ProgramPasses(rows=(_static_row(0, 40),) + tuple(
        replace(_producer_row(1 + i), core_types=((64, 64),)) for i in range(producer_passes)))


@pytest.mark.parametrize("k", [1, 16, 64])
def test_operand_passes_on_a_shared_core_type_wear_its_core_once_per_inference_each(k):
    """The integration review's reproducer: the core takes 7 producer writes per inference plus 1 / K static ones,
    never a band divided by K, so the floor refuses at every K."""
    census = period_census(_shared_core_rows(), weight_bits=8, capacity={(64, 64): 1})
    policy = WritePolicy(batch_k=k, write_granularity="cell", core_capacity=(((64, 64), 1),), physical_cores=1,
                         inference_rate_hz=30.0, min_lifetime_hours=5_000.0)
    assert census.producer_slots == ((1, 7),)
    wear, reason = cell_wear(census.census, policy, census.most_written_core, census.producer_slots)
    assert reason is None and wear is not None and wear.exact and wear.low == pytest.approx(7 + 1 / k)
    endurance = _physics(endurance_cycles=constant(1e9, unit="cycles", low=1e9, high=1e9))
    priced = price_write_terms(census.census, endurance, policy, written_extents=census.written_extents,
                               most_written=census.most_written_core, producer_slots=census.producer_slots)
    verdict = next(v for v in priced.verdicts if v.name == LIFETIME_VERDICT)
    assert verdict.verdict == "inadmissible"
    hours = next(t for t in priced.terms if t.name == "lifetime_hours")
    assert hours.band.high == pytest.approx(1e9 / ((7 + 1 / k) * 30 * 3600))
    quantities = write_quantities(census.census, policy, written_extents=census.written_extents,
                                  most_written=census.most_written_core, producer_slots=census.producer_slots)
    assert quantities["max_writes_per_cell_per_inference"].value == pytest.approx(7 + 1 / k)


def test_a_loaded_once_census_gives_each_operand_region_its_own_core():
    """One placement model per census: where every tile fits its core type, static tiles and operand regions each
    keep a core, so each operand cell is written once per inference. [integration of N2 with W03] A fit is loaded
    once where the planner pins it (``mark_fit_residency``, the floor at r = 0, marks every core of every pass): the
    rows carry those marks; the executed rotation rewrites an unmarked program of several writing passes."""
    rows = ProgramPasses(rows=tuple(replace(row, marks=frozenset({0})) for row in (
        _static_row(0, 40), _producer_row(1), _producer_row(2), _static_row(3, 30))))
    census = period_census(rows, weight_bits=8, capacity={(64, 64): 2, (32, 32): 2})
    assert census.most_written_core == 0 and census.producer_slots == ((0, 1),)
    wear, _ = cell_wear(census.census, _policy(16), census.most_written_core, census.producer_slots)
    assert wear is not None and wear.exact and wear.low == 1.0


def test_producer_cells_on_unplaced_cores_bound_the_wear_from_below_only():
    census = period_census(_fixture_rows(), weight_bits=8, capacity=CAPACITY)
    wear, _ = cell_wear(census.census, _policy(16), census.most_written_core)  # no producer placement stated
    assert wear is not None and wear.high is None and wear.low == 1.0


def _operand_chip(count: int = 3) -> dict:
    return {"cores": [{"max_axons": 64, "max_neurons": 64, "count": count, "operand": {"payload_bits": 8}}],
            "inference_batch_K": 1, "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}


def test_a_sealed_record_without_operand_regions_refuses_the_write_terms_of_an_operand_program(program):
    """A record that states no operand regions per pass (sealed before E-6 sealed them,
    ``test_record_operand_regions.py``) would price operand cells as static backing-store cells and admit an NVM
    chip the mapping plane refuses (integration review's reproducer), so its write terms refuse by name."""
    mapping = program.packed_mapping(**GEOMETRY)
    platform = _operand_chip(10_000)
    sealed = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)
    record = SimpleNamespace(
        schedule=replace(sealed, stages=tuple(replace(s, operand_core_cells=None) if hasattr(s, "operand_core_cells")
                                              else s for s in sealed.stages)),
        timing=SimpleNamespace(per_segment=(), s_global=1), identity=SimpleNamespace(platform=platform),
        utilization=SimpleNamespace(crossbar=SimpleNamespace(weight_bits=8)))
    writes = record_write_census(record)
    assert writes is not None and writes.census is None and "operand regions" in str(writes.unavailable)
    priced = price_write_inputs(WriteInputs.of(writes), _physics(endurance_cycles=constant(1e6, unit="cycles")))
    assert {r.name for r in priced.refusals} == {ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES, LIFETIME_VERDICT}
    assert not priced.verdicts
    mapped = plane_writes(mapping_program_passes(mapping), platform, weight_bits=8, timesteps=1, label="mapping")
    assert mapped is not None and mapped.census is not None  # the mapping plane states the regions


def test_the_candidate_plane_refuses_write_terms_it_cannot_lower(program):
    softcores = list(program.ir.layout_softcores)
    noc = collect_noc_fragments(softcores=softcores, core_types=[LayoutHardCoreType(64, 64, 3)], census=None,
                                allow_scheduling=True, allow_neuron_splitting=True, allow_coalescing=False,
                                max_schedule_passes=128)
    passes = candidate_program_passes(softcores, noc, semantics=_VALUE, timesteps=None)  # the search's call
    context = candidate_context_from_platform({**_operand_chip(), "weight_bits": 8}, passes=passes)
    assert context.writes is not None and context.writes.census is None
    assert "does not lower products" in str(context.writes.unavailable)


def test_producer_write_steps_never_span_two_passes():
    rows = ProgramPasses(rows=(_producer_row(0, cells=3), _producer_row(1, cells=3)))
    census = period_census(rows, weight_bits=8, capacity={(32, 32): 2})
    exposed = {}
    for lanes in (1, 2):
        policy = WritePolicy(batch_k=1, write_granularity="cell", write_units_in_parallel=lanes,
                             core_capacity=(((32, 32), 2),), physical_cores=2)
        priced = price_write_terms(census.census, _physics(), policy, written_extents=census.written_extents,
                                   most_written=census.most_written_core, producer_slots=census.producer_slots)
        assert priced.exposed is not None
        exposed[lanes] = priced.exposed.nominal
        events = write_quantities(census.census, policy, written_extents=census.written_extents)["write_events"]
        assert events.value == (6 if lanes == 1 else 4)  # ceil(3 / 2) per pass, twice; never ceil(6 / 2)
    assert exposed[2] / exposed[1] == pytest.approx(4 / 6)


def test_producer_cells_without_a_write_time_refuse_by_their_own_reason():
    census = period_census(_fixture_rows(), weight_bits=8, capacity=CAPACITY)
    target = physics("generic_estimated_22nm")
    assert target.has("t_program_per_byte") and not target.has("t_write_per_unit")
    priced = price_write_terms(census.census, target, _policy(1), written_extents=census.written_extents,
                               most_written=census.most_written_core, producer_slots=census.producer_slots)
    reason = next(r.reason for r in priced.refusals if r.name == LATENCY_WRITE)
    assert "t_write_per_unit" in reason and "backing-store bytes only" in reason


def test_a_lifetime_below_the_floor_is_inadmissible_by_name():
    census = period_census(_fixture_rows(), weight_bits=8, capacity=CAPACITY)
    endurance = _physics(endurance_cycles=constant(1e6, unit="cycles", low=1e6, high=1e9))
    priced = price_write_terms(census.census, endurance,
                               _policy(16, inference_rate_hz=30.0, min_lifetime_hours=43_800.0),
                               written_extents=census.written_extents, most_written=census.most_written_core,
                               producer_slots=census.producer_slots)
    verdict = next(v for v in priced.verdicts if v.name == LIFETIME_VERDICT)
    assert verdict.verdict == "inadmissible" and "min_lifetime_hours" in verdict.reason
    assert torch is not None
