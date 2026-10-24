"""N2b (N2 design §7 and §9, owner decision 4): per-core-type write technology.

A core type declares ``write_physics``, a shipped physics profile whose write rows price its cells (the static tiles
on an NVM, the operand cores on SRAM), beside its own write granularity and units. The write law prices each write
technology's share of a plane's period census by G3's one law and joins the shares: energies and per-pass write walls
add (the technologies share the chip's write ports), the lifetime is the shortest technology's, the floor verdict the
worst. A chip whose core types share one write technology states none and prices byte for byte as before. Refusals by
name: an unknown write physics, two technologies on one geometry, a plane that cannot attribute its writes to a
technology, operand core types of two technologies under the lowering price.

The new interface is imported inside each test so that each test fails on its own on a tree without it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.cost.absolute.writes import (
    ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES, LIFETIME_VERDICT, WriteInputs, price_write_inputs,
    price_write_terms,
)
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.quantities.write_law import plane_writes, write_policy_of, write_quantities
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
from unit.deployment_record.g3_fixtures import constant, physics

S, O, T = (64, 64), (32, 32), (16, 16)  # a static geometry, an operand geometry, a second static geometry
RATE, FLOOR = 30.0, 43_800.0
TABLE = Path(__file__).resolve().parents[2] / "fixtures" / "theorem_box" / "hypothesis_table.json"


def _row(cells, types, *, producer=None, resident=False, marks=(), steps=1, segment=0):
    """One pass: static cells per core, each core's geometry, payload cells per core (operand regions)."""
    producer = list(producer or [0] * len(cells))
    return PassRow(segment_index=segment, resident=resident, core_cells=tuple(cells), marks=frozenset(marks),
                   core_extents=tuple((max(c, 1), 1) for c in cells), compute_steps=steps, stage_latencies=(0,),
                   core_types=tuple(types), producer_cells=sum(producer),
                   producer_cores=frozenset(i for i, c in enumerate(producer) if c),
                   producer_core_cells=tuple(producer) if any(producer) else None)


def _core(kind, count, *, granularity="cell", write_physics=None, operand=False):
    ct = {"max_axons": kind[0], "max_neurons": kind[1], "count": count, "write_granularity": granularity}
    if write_physics is not None:
        ct["write_physics"] = write_physics
    if operand:
        ct["operand"] = {"payload_bits": 8, "write_source": "host"}
    return ct


def _platform(*cores, k=1, **extra):
    return {"cores": list(cores), "inference_batch_K": k, "inference_rate_hz": RATE, "min_lifetime_hours": FLOOR,
            **extra}


def _price(passes, platform, chip):
    writes = plane_writes(passes, platform, weight_bits=8, timesteps=None, label="hand")
    assert writes is not None
    return writes, price_write_inputs(WriteInputs.of(writes), chip)


def _term(priced, name):
    return next((t for t in priced.terms if t.name == name), None)


def _refusal(priced, name):
    return next((r.reason for r in priced.refusals if r.name == name), None)


def _verdict(priced):
    return next((v for v in priced.verdicts if v.name == LIFETIME_VERDICT), None)


def _resident():
    """Figure 1's pair at unit scale: static tiles loaded once (3 static cores for 3 static tiles, 2 operand cores
    for the 2 operand regions), each operand region rewritten once per inference."""
    return ProgramPasses(rows=(
        _row([100, 50, 0], [S, S, O], producer=[0, 0, 40]),
        _row([30, 0], [S, O], producer=[0, 40], segment=1),
    ))


def _rotating():
    """The same shapes and a third pass on fewer cores: the static tiles rotate through 2 static cores."""
    return ProgramPasses(rows=(
        _row([100, 50, 0], [S, S, O], producer=[0, 0, 40]),
        _row([30, 0], [S, O], producer=[0, 40], segment=1),
        _row([20, 10], [S, S], segment=2),
    ))


# --- declaration: the registry field, the theorem box, resolution ------------------------------------------------

def test_write_physics_is_a_cost_only_core_type_field_whose_absence_is_the_chips_physics():
    from mimarsinan.certification.randomized.cost_draws import PRICING_ONLY, cost_only_set, partition
    from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
    from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
    field = CORE_TYPE_SCHEMA.fields["write_physics"]
    assert field.decision_class == COST_ONLY and "chip" in field.absent_means
    assert "cores.write_physics" in cost_only_set() and "cores.write_physics" in PRICING_ONLY
    assert "cores.write_physics" in json.loads(TABLE.read_text())["cost_only"]
    partition()


def test_an_unknown_write_physics_refuses_by_name_at_resolution():
    from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    cores = [_core(S, 2), _core(O, 1, write_physics="no_such_profile")]
    with pytest.raises(DeclarationValueError, match="no_such_profile") as refused:
        build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": cores}))
    assert "write_physics" in str(refused.value) and "sram_cim_macro" in str(refused.value)


def test_the_resolved_surface_seals_each_named_write_physics_and_only_when_declared():
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    plain = build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": [_core(S, 2), _core(O, 1)]}))
    assert "write_physics_resolved" not in plain
    mixed = build_platform_constraints_resolved(merge_pipeline_config(
        {}, {"cores": [_core(S, 2, write_physics="neurram"), _core(O, 1, write_physics="sram_cim_macro")]}))
    assert mixed["write_physics_resolved"] == {
        name: get_platform_physics(name).to_dict() for name in ("neurram", "sram_cim_macro")}
    assert mixed["cores"][1]["write_physics"] == "sram_cim_macro"


def test_the_sealed_copy_prices_a_record_whatever_the_shipped_profile_says():
    """A record re-priced later reads the write physics its run sealed, as ``platform_physics_resolved`` is read."""
    from mimarsinan.deployment_record.quantities.write_technology import core_technologies
    sealed = get_platform_physics("sram_cim_macro").to_dict()
    sealed["constants"]["endurance_cycles"].update(low=2e16, nominal=2e16, high=2e16)
    platform = _platform(_core(S, 3), _core(O, 2, write_physics="sram_cim_macro", operand=True),
                         write_physics_resolved={"sram_cim_macro": sealed})
    technologies = core_technologies(platform)
    assert technologies is not None
    operand = next(t for t in technologies if t.core_type == O)
    assert operand.physics is not None and operand.physics.band("endurance_cycles").nominal == 2e16


def test_two_technologies_on_one_geometry_refuse_by_name():
    from mimarsinan.deployment_record.quantities.write_technology import WriteTechnologyError
    with pytest.raises(WriteTechnologyError, match="64 x 64"):
        write_policy_of(_platform(_core(S, 2), _core(S, 1, write_physics="sram_cim_macro")))
    with pytest.raises(WriteTechnologyError, match="64 x 64"):
        write_policy_of(_platform(_core(S, 2, granularity="cell"), _core(S, 1, granularity="row")))


def test_the_write_technology_replaces_every_write_row_and_nothing_else():
    from mimarsinan.deployment_record.platform_physics.resolve import WRITE_TECHNOLOGY_KEYS, with_write_technology
    chip, sram = get_platform_physics("neurram"), get_platform_physics("sram_cim_macro")
    typed = with_write_technology(chip, sram)
    for key in WRITE_TECHNOLOGY_KEYS:
        assert typed.has(key) == sram.has(key), key
        if sram.has(key):
            assert typed.constants[key] == sram.constants[key], key
    assert not typed.has("t_write_per_unit")  # NeuRRAM's write time is not the SRAM cells' write time
    assert typed.constants["e_mac"] == chip.constants["e_mac"]
    assert "e_dma_per_byte" not in WRITE_TECHNOLOGY_KEYS and "t_program_per_byte" not in WRITE_TECHNOLOGY_KEYS
    assert set(typed.constants) - set(WRITE_TECHNOLOGY_KEYS) == set(chip.constants) - set(WRITE_TECHNOLOGY_KEYS)


# --- identity: one write technology prices as before ---------------------------------------------------------------

@pytest.mark.parametrize("cores", [
    (_core(S, 4),),
    (_core(S, 3), _core(O, 2, operand=True)),
    (_core(S, 3, granularity="row"), _core(O, 2, granularity="row", operand=True)),
])
def test_a_chip_of_one_write_technology_states_none_and_prices_as_the_one_law(cores):
    policy = write_policy_of(_platform(*cores))
    assert policy is not None and policy.core_technologies is None
    passes = _resident() if len(cores) > 1 else ProgramPasses(rows=(_row([100, 50], [S, S]), _row([30], [S])))
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    writes, priced = _price(passes, _platform(*cores), chip)
    census = writes.census
    direct = price_write_terms(census.census, chip, writes.policy, written_extents=census.written_extents,
                               most_written=census.most_written_core, pass_core_types=census.pass_core_types,
                               producer_slots=census.producer_slots)
    assert priced.terms == direct.terms and priced.refusals == direct.refusals and priced.verdicts == direct.verdicts


def test_one_core_type_declaring_write_physics_prices_as_the_one_law_at_that_physics():
    from mimarsinan.deployment_record.platform_physics.resolve import with_write_technology
    chip = get_platform_physics("neurram")
    passes = ProgramPasses(rows=(_row([100, 50], [S, S]), _row([30], [S])))
    writes, priced = _price(passes, _platform(_core(S, 2, write_physics="generic_estimated_22nm")), chip)
    assert writes.policy.core_technologies is not None
    census = writes.census
    typed = with_write_technology(chip, get_platform_physics("generic_estimated_22nm"))
    plain = write_policy_of(_platform(_core(S, 2)))
    direct = price_write_terms(census.census, typed, plain, written_extents=census.written_extents,
                               most_written=census.most_written_core, pass_core_types=census.pass_core_types,
                               producer_slots=census.producer_slots)
    assert priced.terms == direct.terms and priced.refusals == direct.refusals and priced.verdicts == direct.verdicts


def test_the_one_law_refuses_by_name_a_policy_of_several_technologies():
    """No silent wrong output: G3's one-law pricer handed a mixed chip's policy prices nothing."""
    writes, _ = _price(_resident(), _platform(_core(S, 3), _core(O, 2, write_physics="sram_cim_macro",
                                                                operand=True)), get_platform_physics("neurram"))
    direct = price_write_terms(writes.census.census, get_platform_physics("neurram"), writes.policy)
    assert direct.energy is None and direct.exposed is None
    assert "write technolog" in (_refusal(direct, ENERGY_WRITE) or "")
    bare = WriteInputs(census=writes.census.census, policy=writes.policy)
    assert "write technolog" in (_refusal(price_write_inputs(bare, get_platform_physics("neurram")),
                                          LIFETIME_INFERENCES) or "")


# --- Figure 1's pair: NVM static tiles with an SRAM operand core type ------------------------------------------------

def test_nvm_static_tiles_with_an_sram_operand_core_type_are_admitted_where_all_nvm_is_not():
    chip = get_platform_physics("neurram")
    sram = get_platform_physics("sram_cim_macro")
    all_nvm = _platform(_core(S, 3), _core(O, 2, operand=True))
    mixed = _platform(_core(S, 3), _core(O, 2, write_physics="sram_cim_macro", operand=True))
    _, refused = _price(_resident(), all_nvm, chip)
    writes, admitted = _price(_resident(), mixed, chip)
    assert _verdict(refused).verdict == "inadmissible"
    assert _verdict(admitted).verdict == "admitted" and "32 x 32" in _verdict(admitted).reason
    # the operand cells take one write per inference on SRAM, the loaded-once static tiles none
    hours = _term(admitted, "lifetime_hours").band
    assert hours.nominal == pytest.approx(sram.band("endurance_cycles").nominal / (1.0 * RATE * 3600), rel=1e-12)
    energy = admitted.energy
    assert energy is not None
    assert energy.nominal == pytest.approx(sram.band("e_write_per_cell").nominal * 80 * 1e3, rel=1e-12)  # mJ
    assert "unpriced: e_dma_per_byte(host-sourced payload transfer)" in energy.basis  # NeuRRAM's chip, no DMA row
    assert "t_write_per_unit" in (_refusal(admitted, LATENCY_WRITE) or "")  # SRAM macro states no write time
    assert "32 x 32" in (_refusal(admitted, LATENCY_WRITE) or "")
    # the chip-wide wear is unchanged by the technology: one write per operand cell per inference
    quantities = write_quantities(writes.census.census, writes.policy, most_written=writes.census.most_written_core,
                                  producer_slots=writes.census.producer_slots)
    assert quantities["max_writes_per_cell_per_inference"].value == 1.0


def test_the_operand_writes_on_nvm_refuse_whatever_the_static_tiles_are():
    _, priced = _price(_resident(), _platform(_core(S, 3, write_physics="sram_cim_macro"), _core(O, 2, operand=True)),
                       get_platform_physics("neurram"))
    verdict = _verdict(priced)
    assert verdict.verdict == "inadmissible" and "32 x 32" in verdict.reason


@pytest.mark.parametrize("k", [1, 16, 64])
def test_the_operand_terms_of_the_mixed_chip_are_constant_in_k(k):
    mixed = _platform(_core(S, 3), _core(O, 2, write_physics="sram_cim_macro", operand=True), k=k)
    _, priced = _price(_resident(), mixed, get_platform_physics("neurram"))
    _, at_one = _price(_resident(), {**mixed, "inference_batch_K": 1}, get_platform_physics("neurram"))
    assert priced.energy == at_one.energy
    assert _term(priced, "lifetime_hours").band == _term(at_one, "lifetime_hours").band


# --- the join: G3's law per technology, energies and walls added, the shortest lifetime, the worst verdict ----------

def _hand_split(passes, written, kinds):
    """The census restricted to the cores of ``kinds``, built by hand from the rows and the written cores."""
    out = []
    for p, row in enumerate(passes.rows):
        mine = [i for i, kind in enumerate(row.core_types) if kind in kinds]
        w = [i for i in written.written_cores[p] if row.core_types[i] in kinds]
        producer = sum((row.producer_core_cells or (0,) * len(row.core_cells))[i] for i in mine)
        full = written.census.passes[p]
        out.append(PassCensus(position=p, segment_index=row.segment_index, resident=full.resident, cores=len(mine),
                              resident_cores=len(mine) - len(w),
                              written_cells_backing_store=sum(row.core_cells[i] for i in w),
                              written_cells_producer=producer, payload_bytes=sum(row.core_cells[i] for i in w),
                              compute_steps=full.compute_steps))
    return PassPeriodCensus(passes=tuple(out))


def test_the_mixed_energy_is_the_sum_of_g3s_law_per_technology():
    from mimarsinan.deployment_record.platform_physics.resolve import with_write_technology
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    sram = with_write_technology(chip, get_platform_physics("sram_cim_macro"))
    platform = _platform(_core(S, 2), _core(O, 1, write_physics="sram_cim_macro", operand=True), k=4)
    writes, priced = _price(_rotating(), platform, chip)
    plain = write_policy_of({**platform, "cores": [_core(S, 2), _core(O, 1, operand=True)]})
    static = price_write_terms(_hand_split(_rotating(), writes.census, {S}), chip, plain)
    operand = price_write_terms(_hand_split(_rotating(), writes.census, {O}), sram, plain)
    assert static.energy is not None and operand.energy is not None and priced.energy is not None
    for corner in ("low", "nominal", "high"):
        assert getattr(priced.energy, corner) == pytest.approx(
            getattr(static.energy, corner) + getattr(operand.energy, corner), rel=1e-12)
    assert "64 x 64" in priced.energy.basis and "32 x 32" in priced.energy.basis


def test_the_mixed_serial_write_time_adds_the_technologies_walls():
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    nvm = get_platform_physics("neurram")
    platform = _platform(_core(S, 2), _core(T, 2, write_physics="neurram"), k=2)
    passes = ProgramPasses(rows=(_row([100, 50, 7], [S, S, T]), _row([30, 9, 3], [S, T, T], segment=1),
                                 _row([20], [S], segment=2)))
    _, priced = _price(passes, platform, chip)
    step_s = chip.band("t_write_per_unit").nominal * chip.band("write_verify_iterations").nominal
    step_t = nvm.band("t_write_per_unit").nominal * nvm.band("write_verify_iterations").nominal
    expected = ((100 + 50 + 30 + 20) * step_s + (7 + 9 + 3) * step_t) / 2
    assert priced.exposed is not None and priced.exposed.nominal == pytest.approx(expected, rel=1e-12)


def test_overlapped_walls_add_per_pass_before_the_exposure_rule():
    """The technologies share the write ports: per pass the walls add, then max(0, t_w(q+1) / K - t_c(q)); adding the
    technologies' own exposures would hide a write that neither hides alone."""
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    nvm = get_platform_physics("neurram")
    step_s = chip.band("t_write_per_unit").nominal * chip.band("write_verify_iterations").nominal
    step_t = nvm.band("t_write_per_unit").nominal * nvm.band("write_verify_iterations").nominal
    cycle = chip.band("t_cycle").nominal
    wall_s, wall_t = 400 * step_s, 2 * step_t  # pass 1's walls per technology
    steps = int((max(wall_s, wall_t) + 0.5 * min(wall_s, wall_t)) / cycle)  # hides each alone, not both
    assert max(wall_s, wall_t) < steps * cycle < wall_s + wall_t
    passes = ProgramPasses(rows=(_row([1], [S], steps=steps), _row([400, 2], [S, T], steps=steps, segment=1)))
    platform = _platform(_core(S, 1), _core(T, 1, write_physics="neurram"),  # 2 static tiles on 1 core: rewritten
                         write_overlap={"mode": "overlapped", "shadow_capacity_fraction": 0.0})
    _, priced = _price(passes, platform, chip)
    expected = max(0.0, wall_s + wall_t - steps * cycle) + max(0.0, 1 * step_s - steps * cycle)
    assert priced.exposed is not None and priced.exposed.nominal == pytest.approx(expected, rel=1e-9)
    assert priced.exposed.nominal > 0


def test_the_mixed_lifetime_is_the_shortest_technologys_and_the_verdict_the_worst():
    chip = get_platform_physics("neurram")
    platform = _platform(_core(S, 2), _core(O, 1, write_physics="sram_cim_macro", operand=True), k=1)
    _, priced = _price(_rotating(), platform, chip)
    # static NVM cores rotate: the most-written takes 3 writes per period, against 1e6..1e9 cycles
    hours = _term(priced, "lifetime_hours").band
    nvm = chip.band("endurance_cycles")
    assert (hours.low, hours.high) == pytest.approx((nvm.low / 3 / (RATE * 3600), nvm.high / 3 / (RATE * 3600)))
    verdict = _verdict(priced)
    assert verdict.verdict == "inadmissible" and "64 x 64" in verdict.reason and "32 x 32" not in verdict.reason
    # NeuRRAM declares no commit energy: the static writes refuse the energy, by the technology's name
    assert priced.energy is None and "64 x 64" in (_refusal(priced, ENERGY_WRITE) or "")


def test_a_technology_without_endurance_refuses_the_chips_lifetime_by_name():
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    sealed = get_platform_physics("sram_cim_macro").to_dict()
    sealed["constants"].pop("endurance_cycles")  # a sealed copy without the row
    platform = _platform(_core(S, 2), _core(O, 1, write_physics="sram_cim_macro", operand=True),
                         write_physics_resolved={"sram_cim_macro": sealed})
    _, priced = _price(_rotating(), platform, chip)
    assert "32 x 32" in (_refusal(priced, LIFETIME_INFERENCES) or "")
    assert _term(priced, "lifetime_hours") is None
    assert "32 x 32" in (_refusal(priced, LIFETIME_VERDICT) or "")


def test_mixed_write_granularities_are_priced_per_core_type():
    """Before N2b the chip refused: one write law needed one granularity. Each type now writes in its own units; the
    chip-wide unit count, a sum of unlike units, stays unstated."""
    chip = physics("generic_estimated_22nm", t_write_per_unit=constant(10.0, unit="ns"))
    platform = _platform(_core(S, 2, granularity="cell"), _core(T, 2, granularity="row"), k=1)
    passes = ProgramPasses(rows=(_row([100, 7], [S, T]), _row([30, 9], [S, T], segment=1),
                                 _row([5, 6], [T, T], segment=2)))
    writes, priced = _price(passes, platform, chip)
    step = chip.band("t_write_per_unit").nominal * chip.band("write_verify_iterations").nominal
    rows_t = 7 + 9 + 5 + 6  # each core's used rectangle is cells x 1: as many rows as cells
    assert priced.exposed is not None and priced.exposed.nominal == pytest.approx((100 + 30 + rows_t) * step)
    quantities = write_quantities(writes.census.census, writes.policy,
                                  written_extents=writes.census.written_extents,
                                  most_written=writes.census.most_written_core)
    assert "written_units" not in quantities and "runtime_written_cells" in quantities


# --- refusals by name of planes that cannot attribute their writes ----------------------------------------------------

def test_an_untyped_plane_on_two_technologies_refuses_its_write_terms_by_name():
    rows = tuple(PassRow(segment_index=i, resident=False, core_cells=(10, 10, 10), marks=frozenset(),
                         core_extents=None, compute_steps=1, stage_latencies=(0,)) for i in range(4))
    platform = _platform(_core(S, 2), _core(O, 1, write_physics="sram_cim_macro"))
    _, priced = _price(ProgramPasses(rows=rows), platform, get_platform_physics("neurram"))
    for name in (ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES, LIFETIME_VERDICT):
        assert "core type" in (_refusal(priced, name) or ""), name


def test_payload_cells_on_two_technologies_without_per_core_cells_refuse_by_name():
    row = PassRow(segment_index=0, resident=False, core_cells=(0, 0), marks=frozenset(), core_extents=((1, 1),) * 2,
                  compute_steps=1, stage_latencies=(0,), core_types=(S, O), producer_cells=50,
                  producer_cores=frozenset({0, 1}))
    platform = _platform(_core(S, 1, operand=True), _core(O, 1, write_physics="sram_cim_macro", operand=True))
    _, priced = _price(ProgramPasses(rows=(row,)), platform, get_platform_physics("neurram"))
    assert "per core" in (_refusal(priced, ENERGY_WRITE) or "")


# --- the split conserves the census -------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_the_split_by_technology_conserves_the_census(seed):
    """Random censuses on three geometries of three technologies (residency marks, resident passes, operand regions
    on their own type; loaded once or rotating as the drawn counts fall): every share sums to the census."""
    from mimarsinan.deployment_record.quantities.write_technology import split_by_technology
    rng = random.Random(seed)
    rows, widest, pinned = [], {S: 0, T: 0, O: 0}, {S: 0, T: 0, O: 0}
    for segment in range(rng.randint(1, 6)):
        kinds = [k for k in (S, T, O) for _ in range(rng.randint(0, 3))] or [S]
        cells = [0 if (k == O and rng.random() < 0.5) else rng.randint(1, 99) for k in kinds]
        producer = [rng.randint(1, 50) if k == O else 0 for k in kinds]
        resident = rng.random() < 0.2
        marks = [i for i in range(len(kinds)) if rng.random() < 0.2]
        rows.append(_row(cells, kinds, producer=producer, resident=resident, marks=marks, segment=segment))
        for k in (S, T, O):
            widest[k] = max(widest[k], kinds.count(k))
            pinned[k] += 0 if resident else sum(1 for i in marks if kinds[i] == k)
    counts = {k: max(1, widest[k] + pinned[k] + rng.randint(0, 4)) for k in (S, T, O)}
    passes = ProgramPasses(rows=tuple(rows))
    policy = write_policy_of(_platform(_core(S, counts[S]), _core(T, counts[T], write_physics="neurram"),
                                       _core(O, counts[O], write_physics="sram_cim_macro", operand=True)))
    written = period_census(passes, weight_bits=8, capacity=counts)
    split = split_by_technology(passes, written, policy, weight_bits=8)
    assert split.refused is None and len(split.parts) == 3
    fields = ("cores", "resident_cores", "written_cells_backing_store", "written_cells_producer", "payload_bytes")
    for position, full in enumerate(written.census.passes):
        for name in fields:
            assert sum(getattr(part.census.passes[position], name) for part in split.parts) == getattr(full, name)
    assert written.most_written_core is not None and written.producer_slots is not None
    assert max(part.most_written or 0 for part in split.parts) == written.most_written_core
    assert set().union(*(set(part.producer_slots or ()) for part in split.parts)) == set(written.producer_slots)


# --- the lowering price reads the operand core type's write technology -----------------------------------------------

def _form():
    """m = 6, k = 5, n = 2 in one bank: B stationary places 6 instances of 10 cells, 60 cells per inference."""
    import torch
    from mimarsinan.mapping.support.contraction import Contraction
    from mimarsinan.mapping.support.operand_lowering import operand_rule_form
    form = operand_rule_form(Contraction(torch.bmm, "bmk,bkn->bmn", ((6, 5), (5, 2)), 60))
    assert form is not None
    return form


def _operand_platform(static_physics=None, operand_physics=None):
    """NeuRRAM's chip (no commit energy, no DMA row): only a declared write technology prices the regions."""
    return {"cores": [_core(S, 4, write_physics=static_physics), _core(O, 2, write_physics=operand_physics,
                                                                       operand=True)],
            "activation_bits": 8, "platform_physics_profile": "neurram"}


def test_the_lowering_price_reads_the_operand_core_types_write_physics():
    from mimarsinan.mapping.support.operand_form import B_STATIONARY, Lowering
    from mimarsinan.mapping.support.operand_lowering import lowering_price
    lowering = Lowering(_form(), B_STATIONARY)
    base = lowering_price(_operand_platform())(lowering)
    static_only = lowering_price(_operand_platform(static_physics="generic_estimated_22nm"))(lowering)
    operand = lowering_price(_operand_platform(operand_physics="generic_estimated_22nm"))(lowering)
    assert base == static_only == 0.0  # the static type's technology never prices operand regions
    generic = get_platform_physics("generic_estimated_22nm")
    writes = 60 * generic.band("e_write_per_cell").nominal * generic.band("write_verify_iterations").nominal
    assert operand == pytest.approx(writes, rel=1e-12)


def test_operand_core_types_of_two_technologies_refuse_the_lowering_price_by_name():
    from mimarsinan.deployment_record.quantities.write_technology import WriteTechnologyError
    from mimarsinan.mapping.support.operand_lowering import lowering_price
    platform = {"cores": [_core(S, 4, write_physics="sram_cim_macro", operand=True),
                          _core(O, 2, write_physics="hermes_64", operand=True)],
                "activation_bits": 8, "platform_physics_profile": "generic_estimated_22nm"}
    with pytest.raises(WriteTechnologyError, match="operand core types"):
        lowering_price(platform)


# --- the mapping plane, end to end ---------------------------------------------------------------------------------

def test_a_built_operand_program_on_a_mixed_chip_is_admitted_and_on_all_nvm_refused():
    from mimarsinan.deployment_record.build.program_census import mapping_program_passes
    from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
    from unit.chip_simulation.value_run.operand_programs import attention_program
    program = attention_program("softmax")
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=False))
    declared = [_core(S, 999), _core(T, 999, operand=True)]
    mapping = build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=declared, strategy=strategy)
    passes = mapping_program_passes(mapping)
    held = [row for row in passes.rows if row.producer_cells]
    assert held and all(row.producer_core_cells is not None for row in held)
    assert all(row.core_types[i] == T for row in held for i in row.producer_cores)
    chip = get_platform_physics("neurram")
    _, all_nvm = _price(passes, _platform(*declared), chip)
    mixed = _platform(declared[0], {**declared[1], "write_physics": "sram_cim_macro"})
    _, priced = _price(passes, mixed, chip)
    assert _verdict(all_nvm).verdict == "inadmissible"
    assert _verdict(priced).verdict == "admitted"
    cells = sum(row.producer_cells for row in passes.rows)
    sram = get_platform_physics("sram_cim_macro")
    assert priced.energy is not None
    assert priced.energy.nominal == pytest.approx(sram.band("e_write_per_cell").nominal * cells * 1e3, rel=1e-12)
