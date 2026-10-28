"""[E-6, N2 follow-up F5] Per-core-type compute physics: each core type's MAC events at its own technology's e_mac.

N2b priced writes per core type (``write_physics``) while the MAC energy stayed one ``e_mac`` per chip, so Figure 1's
mixed chip (NVM static tiles beside SRAM-CIM operand cores) priced the products' MACs at the NVM's e_mac and the
acceptance printed the per-type split as arithmetic beside. A core type now declares ``compute_physics``, a shipped
physics profile whose compute row (``e_mac``) prices that core type's MAC events (the value rule's committed cells,
read per tile from a plane's passes), on every plane that states its passes; a chip whose core types name none states
no split and prices byte for byte as before. Refusals by name: an unknown profile at resolution, a spiking chip (its
event census states no per-core-type split), a plane with no pass structure, a split that differs from the MAC
census, a named physics without the compute row.

The new interface is imported inside each test so that each test fails on its own on a tree without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from unit.deployment_record.g3_fixtures import quantities

S, O = (64, 64), (32, 32)
TABLE = Path(__file__).resolve().parents[2] / "fixtures" / "theorem_box" / "hypothesis_table.json"
VALUE = {"core_semantics": "mvm", "activation_bits": 8}


def _core(kind, count, *, compute_physics=None, operand=False):
    ct = {"max_axons": kind[0], "max_neurons": kind[1], "count": count}
    if compute_physics is not None:
        ct["compute_physics"] = compute_physics
    if operand:
        ct["operand"] = {"payload_bits": 8, "write_source": "host"}
    return ct


def _row(cells, types, producer=None, segment=0):
    producer = list(producer or [0] * len(cells))
    return PassRow(segment_index=segment, resident=False, core_cells=tuple(cells), marks=frozenset(),
                   core_extents=None, compute_steps=None, stage_latencies=(0,), core_types=tuple(types),
                   producer_cells=sum(producer), producer_cores=frozenset(i for i, c in enumerate(producer) if c),
                   producer_core_cells=tuple(producer) if any(producer) else None)


def _passes():
    """Two passes: static tiles (64 x 64) with 300 and 200 cells, operand cores (32 x 32) with 40 payload cells."""
    return ProgramPasses(rows=(_row([100, 200], [S, S]), _row([200, 0], [S, O], producer=[0, 40], segment=1)))


def _dynamic(pricing):
    return next(t.band for t in pricing.terms if t.name == "energy_dynamic_mj")


def _refusal(pricing, name):
    return next((r.reason for r in pricing.refusals if r.name == name), None)


def _q():
    return quantities(synaptic_events=540, timesteps=1)


def test_compute_physics_is_a_cost_only_core_type_field_whose_absence_is_the_chips_physics():
    from mimarsinan.certification.randomized.cost_draws import PRICING_ONLY, cost_only_set, partition
    from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
    from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
    field = CORE_TYPE_SCHEMA.fields["compute_physics"]
    assert field.decision_class == COST_ONLY and "chip" in field.absent_means
    assert "cores.compute_physics" in cost_only_set() and "cores.compute_physics" in PRICING_ONLY
    assert "cores.compute_physics" in json.loads(TABLE.read_text())["cost_only"]
    partition()


def test_an_unknown_compute_physics_refuses_by_name_at_resolution():
    from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    cores = [_core(S, 2), _core(O, 1, compute_physics="no_such_profile")]
    with pytest.raises(DeclarationValueError, match="no_such_profile") as refused:
        build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": cores}))
    assert "compute_physics" in str(refused.value) and "sram_cim_macro" in str(refused.value)


def test_the_resolved_surface_seals_each_named_compute_physics_and_only_when_declared():
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    plain = build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": [_core(S, 2), _core(O, 1)]}))
    assert "compute_physics_resolved" not in plain
    mixed = build_platform_constraints_resolved(merge_pipeline_config(
        {}, {"cores": [_core(S, 2, compute_physics="neurram"), _core(O, 1, compute_physics="sram_cim_macro")]}))
    assert mixed["compute_physics_resolved"] == {
        name: get_platform_physics(name).to_dict() for name in ("neurram", "sram_cim_macro")}


def test_a_chip_naming_no_compute_physics_states_no_split_and_prices_as_before():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    platform = {"cores": [_core(S, 2), _core(O, 1)], **VALUE}
    assert compute_split(platform, _passes()) is None
    chip = get_platform_physics("generic_estimated_22nm")
    assert price_absolute(_q(), chip, compute=None) == price_absolute(_q(), chip)


def test_the_split_reads_each_core_types_mac_events_from_the_passes():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    platform = {"cores": [_core(S, 2, compute_physics="neurram"), _core(O, 1, compute_physics="sram_cim_macro")],
                **VALUE}
    split = compute_split(platform, _passes())
    assert split is not None and split.unavailable is None
    assert {share.core_type: share.events for share in split.shares} == {S: 500, O: 40}
    assert {share.core_type: share.compute_physics for share in split.shares} == {S: "neurram", O: "sram_cim_macro"}


def test_each_core_types_mac_events_are_priced_at_its_own_e_mac():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    chip = get_platform_physics("generic_estimated_22nm")
    platform = {"cores": [_core(S, 2, compute_physics="neurram"), _core(O, 1, compute_physics="sram_cim_macro")],
                **VALUE}
    q = quantities(synaptic_events=540, timesteps=1)
    one = _dynamic(price_absolute(q, chip))
    split = _dynamic(price_absolute(q, chip, compute=compute_split(platform, _passes())))
    e = {name: get_platform_physics(name).band("e_mac") for name in ("generic_estimated_22nm", "neurram",
                                                                     "sram_cim_macro")}
    for corner in ("low", "nominal", "high"):
        expected = getattr(one, corner) + (500 * getattr(e["neurram"], corner) + 40 * getattr(e["sram_cim_macro"], corner)
                                           - 540 * getattr(e["generic_estimated_22nm"], corner)) * 1e3
        assert getattr(split, corner) == pytest.approx(expected, rel=1e-12)
    assert "sram_cim_macro" in split.basis and "neurram" in split.basis


def test_a_core_type_naming_no_compute_physics_takes_the_chips_e_mac():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    chip = get_platform_physics("neurram")
    platform = {"cores": [_core(S, 2), _core(O, 1, compute_physics="sram_cim_macro")], **VALUE}
    split = _dynamic(price_absolute(_q(), chip, compute=compute_split(platform, _passes())))
    one = _dynamic(price_absolute(_q(), chip))
    sram, nvm = get_platform_physics("sram_cim_macro").band("e_mac"), chip.band("e_mac")
    assert split.nominal == pytest.approx(one.nominal + 40 * (sram.nominal - nvm.nominal) * 1e3, rel=1e-12)


def test_a_split_that_differs_from_the_mac_census_refuses_the_energy_by_name():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    platform = {"cores": [_core(S, 2, compute_physics="neurram"), _core(O, 1)], **VALUE}
    pricing = price_absolute(quantities(synaptic_events=999, timesteps=1), get_platform_physics("neurram"),
                             compute=compute_split(platform, _passes()))
    assert "per-core-type" in (_refusal(pricing, "energy_per_inference_mj") or "")
    assert "differs" in _refusal(pricing, "energy_dynamic_mj")


def test_a_spiking_chip_naming_compute_physics_refuses_its_energy_by_name():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    platform = {"cores": [_core(S, 2, compute_physics="loihi"), _core(O, 1)], "core_semantics": "spiking"}
    split = compute_split(platform, _passes())
    assert split is not None and split.unavailable is not None and "spiking" in split.unavailable
    pricing = price_absolute(_q(), get_platform_physics("loihi"), compute=split)
    assert "spiking" in _refusal(pricing, "energy_per_inference_mj")


def test_a_plane_without_passes_refuses_by_name():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    platform = {"cores": [_core(S, 2, compute_physics="neurram")], **VALUE}
    split = compute_split(platform, None)
    assert split is not None and "pass structure" in split.unavailable


def test_a_named_physics_without_the_compute_row_refuses_by_name():
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    sealed = get_platform_physics("sram_cim_macro").to_dict()
    sealed["constants"] = {k: v for k, v in sealed["constants"].items() if k != "e_mac"}
    platform = {"cores": [_core(S, 2), _core(O, 1, compute_physics="sram_cim_macro")], **VALUE,
                "compute_physics_resolved": {"sram_cim_macro": sealed}}
    pricing = price_absolute(_q(), get_platform_physics("neurram"), compute=compute_split(platform, _passes()))
    assert "e_mac" in _refusal(pricing, "energy_per_inference_mj")


def test_the_record_plane_and_the_mapping_plane_state_one_split():
    """A built operand program on the mixed chip: the sealed record's passes give the mapping plane's split."""
    from types import SimpleNamespace

    from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
    from mimarsinan.deployment_record.build.program_census import mapping_program_passes, record_program_passes
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
    from unit.chip_simulation.value_run.operand_programs import attention_program
    program = attention_program("softmax")
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=False))
    declared = [_core(S, 999, compute_physics="neurram"),
                _core((16, 16), 999, compute_physics="sram_cim_macro", operand=True)]
    mapping = build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=declared, strategy=strategy)
    platform = {"cores": declared, **VALUE}
    record = SimpleNamespace(schedule=schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0),
                             timing=SimpleNamespace(per_segment=()), identity=SimpleNamespace(platform=platform))
    built, sealed = compute_split(platform, mapping_program_passes(mapping)), compute_split(
        platform, record_program_passes(record))
    assert built == sealed and built.unavailable is None
    cells = sum(c.cells_used for s in record.schedule.segments() for c in s.cores)
    assert sum(share.events for share in built.shares) == cells
    assert {share.core_type for share in built.shares if share.events} == {S, (16, 16)}
