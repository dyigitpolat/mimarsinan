"""[E-6, N2 follow-up F5] The hybrid declaration (NVM static tiles beside SRAM-CIM operand cores) and the SRAM-CIM
macro's write time, so Figure 1's mixed row prices its stacked energy split and its latency band.

- The SRAM-CIM macro's profile declares ``t_write_per_unit`` per 8-bit weight cell, derived from its paper
  (jiang2023a): a 1 kB block is 64 rows of 128 bits (p.2), a row is read or written in one clock cycle (p.3, "until
  the next read/write cycle"), and the macro runs at 20 MHz and 91 MHz (p.3-4, p.4); one 128-bit row of 16 weights per
  cycle gives 3.125 ns per cell at 20 MHz (nominal, the profile's energy operating point) and 0.687 ns at 91 MHz (low);
  the high end, one byte per 20 MHz cycle (50 ns), covers the write port width the paper does not state. With it the
  operand cores' producer writes price their exposed write time, where N2b refused it by name.
- The hybrid declaration (``tests/fixtures/design_point/hybrid_neurram_sram_cim.json``) is the neurram bundle with
  its cores overridden: NeuRRAM's 48 static cores of 256 x 256 cells, 128 x 256 signed weights of two cells each
  ([U-CR C-3] the block its chip description states; write and compute physics NeuRRAM's) beside one
  SRAM-CIM macro of 128 x 1024 as the operand core type (write and compute physics the macro's), every other constant
  NeuRRAM's. It is a declaration over a published target, not a target bundle: a bundle names a published target
  (``target_bundles``), and no published chip carries both core types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.deployment_record.platform_physics import get_platform_physics

STATIC, OPERAND = (128, 256), (128, 1024)  # [U-CR C-3, declared pin amendment] the static type counts weights


def test_the_macro_declares_a_derived_write_time_per_cell_from_its_paper():
    physics = get_platform_physics("sram_cim_macro")
    band = physics.band("t_write_per_unit")  # SI seconds
    assert band.nominal == pytest.approx(50e-9 / 16, rel=1e-9)  # one 128-bit row of 16 weights per 20 MHz cycle
    assert band.low == pytest.approx(1.0 / 91e6 / 16, rel=1e-9)  # the same row at 91 MHz
    assert band.high == pytest.approx(50e-9, rel=1e-9)  # one byte per 20 MHz cycle: the unstated port width
    constant = physics.constants["t_write_per_unit"]
    assert constant.unit == "ns" and constant.evidence_kind == "derived" and "jiang2023a" in constant.citation
    assert "read/write cycle" in constant.derivation and "cell" in constant.derivation


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "design_point" / "hybrid_neurram_sram_cim.json"


def _declaration():
    return json.loads(FIXTURE.read_text())["platform_constraints"]


def _resolved():
    from mimarsinan.config_schema.runtime import run_config
    from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
    return build_platform_constraints_resolved(run_config({}, _declaration()))


def test_the_hybrid_declaration_states_each_core_types_technologies():
    resolved = _resolved()
    assert resolved["platform_physics_resolved"]["name"] == "neurram"
    static, operand = resolved["cores"]
    assert (static["max_axons"], static["max_neurons"], static["count"]) == (*STATIC, 48)
    assert (static["cells_per_weight"], static["weight_cell_rows"]) == (2, 2)  # 256 x 256 cells (U-CR C-3)
    assert static["write_physics"] == static["compute_physics"] == "neurram" and "operand" not in static
    assert (operand["max_axons"], operand["max_neurons"]) == OPERAND
    assert operand["write_physics"] == operand["compute_physics"] == "sram_cim_macro"
    assert operand["operand"]["payload_bits"] == 8
    assert static["write_granularity"] == operand["write_granularity"] == "cell"
    assert resolved["target_faithfulness_class"].startswith("approximate") and "cores declared" in resolved[
        "target_faithfulness_class"]


def test_the_hybrid_declaration_resolves_with_both_technologies_sealed():
    resolved = _resolved()
    assert set(resolved["write_physics_resolved"]) == set(resolved["compute_physics_resolved"]) == {
        "neurram", "sram_cim_macro"}


@pytest.fixture(scope="module")
def built():
    """The softmax attention block packed on the bundle's two core types (counts large enough to hold it once)."""
    from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
    from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
    from unit.chip_simulation.value_run.operand_programs import attention_program
    program = attention_program("softmax")
    cores = [{**ct, "count": 999} for ct in _declaration()["target_overrides"]["cores"]["value"]]
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=True,
                                                        allow_scheduling=False))
    mapping = build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=cores, strategy=strategy)
    platform = {"cores": cores, "core_semantics": "mvm", "activation_bits": 8, "inference_batch_K": 1,
                "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}
    return mapping, platform, schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0)


def test_the_mixed_row_prices_its_stacked_split_and_latency_band_on_the_record_plane(built):
    from types import SimpleNamespace

    from mimarsinan.deployment_record.build.program_census import record_program_passes
    from mimarsinan.deployment_record.cost.absolute import price_absolute
    from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
    from mimarsinan.deployment_record.quantities.compute_technology import compute_split
    from mimarsinan.deployment_record.quantities.write_law import record_write_census
    from unit.deployment_record.g3_fixtures import quantities
    mapping, platform, schedule = built
    record = SimpleNamespace(schedule=schedule, timing=SimpleNamespace(per_segment=(), s_global=1),
                             identity=SimpleNamespace(platform=platform),
                             utilization=SimpleNamespace(crossbar=SimpleNamespace(weight_bits=8)))
    split = compute_split(platform, record_program_passes(record))
    assert split is not None and split.unavailable is None
    events = {share.core_type: share.events for share in split.shares}
    assert events[STATIC] > 0 and events[OPERAND] > 0
    writes = record_write_census(record)
    assert writes is not None and writes.census is not None
    chip = get_platform_physics("neurram")
    pricing = price_absolute(quantities(synaptic_events=split.events, timesteps=1), chip,
                             writes=WriteInputs.of(writes), compute=split)
    terms = {t.name: t.band for t in pricing.terms}
    refused = {r.name: r.reason for r in pricing.refusals}
    e = {name: get_platform_physics(name).band("e_mac").nominal for name in ("neurram", "sram_cim_macro")}
    assert terms["energy_dynamic_mj"].nominal == pytest.approx(
        (events[STATIC] * e["neurram"] + events[OPERAND] * e["sram_cim_macro"]) * 1e3, rel=1e-9)
    operand_cells = sum(sum(s.operand_core_cells) for s in schedule.segments())
    sram = get_platform_physics("sram_cim_macro")
    assert terms["energy_write_mj"].nominal == pytest.approx(
        operand_cells * sram.band("e_write_per_cell").nominal * sram.band("write_verify_iterations").nominal * 1e3,
        rel=1e-9)
    assert "latency_write_exposed_s" in terms, refused.get("latency_write_exposed_s")
    assert terms["latency_write_exposed_s"].nominal > 0
