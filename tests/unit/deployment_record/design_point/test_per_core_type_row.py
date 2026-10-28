"""[E-6, targeted pass, review M2] Figure 1's mixed row judged and priced by the instance: NeuRRAM static tiles beside
SRAM-CIM operand cores, each core type under its own write and compute technology.

The build refused the row at declare (``per_core_type_write_technology``): the instance priced one write technology per
row. Here a row whose core types each declare their ``write_physics`` is judged: the write law prices each
technology's share of the executed period census (``price_write_inputs`` over the census split by technology), each
core type states its own wear and lifetime (the row's are the shortest and the worst), K* is the priced boundary of
the binding technology, and the row prices on the record plane (a record of the executed program, priced by
``absolute_pricing_for_record``): each core type's MAC events at its own ``e_mac``, the operand writes at the macro's
cell energy, the exposed write time.

Two vehicles (``programs.attention_on_hybrid``, the small softmax block at test counts): static tiles rotated over one
NeuRRAM core (scheduled; the NeuRRAM share rewrites its most-written core M = 6 times per period), and static tiles
resident (8 NeuRRAM cores, 4 macros; nothing static rewritten, operand cells written every inference).
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.deployment_record.cost.write_lifetime import endurance_in_writes
from mimarsinan.deployment_record.design_point.chip_rows import PER_CORE_TYPE_WRITE_TECHNOLOGY, declaration_refusals
from mimarsinan.deployment_record.design_point.endurance import CORNERS, k_star
from mimarsinan.deployment_record.design_point.timeline import program_writes
from mimarsinan.deployment_record.design_point.wear import wear_problems
from mimarsinan.deployment_record.platform_physics import PlatformPhysics, get_platform_physics
from unit.chip_simulation.value_run.integer_programs import integer_inputs

from . import programs as P

STATIC = {"role": "static", "max_axons": 256, "max_neurons": 256, "count": 1, "write_technology": "neurram",
          "write_physics": "neurram"}
OPERAND = {"role": "operand", "max_axons": 128, "max_neurons": 1024, "count": 1, "write_technology": "sram_cim_macro",
           "write_physics": "sram_cim_macro", "operand": {"payload_bits": 8, "write_source": "host"}}
ROLES = {"neurram": "static", "sram_cim_macro": "operand"}
GRID = (1, 16, 64)


class TestTheDeclarationRoute:
    def test_core_types_that_each_declare_their_write_physics_are_buildable(self):
        assert declaration_refusals([STATIC, OPERAND]) == []

    def test_a_core_type_without_its_write_physics_still_refuses_by_name(self):
        bare = {k: v for k, v in OPERAND.items() if k != "write_physics"}
        assert [r["name"] for r in declaration_refusals([STATIC, bare])] == [PER_CORE_TYPE_WRITE_TECHNOLOGY]


def _judged(static, operand, scheduling, k=1):
    from mimarsinan.deployment_record.design_point.rows import judged_row
    program, hybrid, platform = P.attention_on_hybrid(static, operand, scheduling)
    platform = {**platform, "weight_bits": 8, "inference_batch_K": k}
    return judged_row(nf=program.model, ir_graph=program.ir, hybrid=hybrid, x_int=integer_inputs(program, 8),
                      platform=platform, physics=PlatformPhysics.from_dict(platform["platform_physics_resolved"]),
                      k_grid=GRID, weight_bits=8, roles=ROLES), platform, hybrid


def _row(judged, platform):
    chip = {"target": None, "display_name": "test", "weight_bits": 8, "claim_class": None, "arithmetic_class": "test",
            "faithfulness": {"class": "approximate", "named": []}, "backing_store": None,
            "core_types": [{"role": ROLES[c["write_physics"]], "max_axons": c["max_axons"],
                            "max_neurons": c["max_neurons"], "count": c["count"],
                            "write_technology": c["write_physics"]} for c in platform["cores"]]}
    return {"row": "mixed", "role": "figure", "title": "test", "beside_of": None, "chip": chip, **judged}


DECLARATIONS = {"inference_rate_hz": P.RATE_HZ, "min_lifetime_hours": P.FLOOR_H}


@pytest.fixture(scope="module")
def rotated():
    return _judged(1, 1, True)


@pytest.fixture(scope="module")
def resident():
    return _judged(8, 4, True)


class TestRotatedStaticTiles:
    """One NeuRRAM core under scheduling: the static tiles are rewritten every period, the operand cells every
    inference; NeuRRAM's endurance binds."""

    def test_each_core_type_states_its_own_wear_and_the_row_validates(self, rotated):
        row, platform, _ = rotated
        assert row["certificate"]["exact"], row["certificate"]
        writes = {e["write_technology"]: e for e in row["writes"]["per_core_type"]}
        lives = {e["write_technology"]: e for e in row["lifetime"]["per_core_type"]}
        assert set(writes) == set(lives) == {"neurram", "sram_cim_macro"}
        assert writes["neurram"]["role"] == "static" and writes["sram_cim_macro"]["role"] == "operand"
        assert writes["neurram"]["static_written_cells_per_period"] > 0
        assert writes["sram_cim_macro"]["operand_written_cells_per_inference"] > 0
        assert lives["neurram"]["endurance"]["cycles"][0] == get_platform_physics("neurram").band(
            "endurance_cycles").low
        assert row["lifetime"]["binding"] == "static (neurram)"
        assert wear_problems(_row(row, platform), DECLARATIONS) == []

    def test_k_star_is_the_static_technologys_priced_boundary(self, rotated):
        row, platform, hybrid = rotated
        writes = program_writes(hybrid, platform, weight_bits=8)
        static = next(p for p in writes.by_technology.parts if p.technology.write_physics == "neurram")
        most = int(static.most_written)
        assert most == 6
        # [U-E3, U-C C-1] the writes a cell sustains: the endurance in cycles over the technology's pulses per write
        # (``endurance_in_writes``, the one reading every endurance reader takes)
        endurance = endurance_in_writes(get_platform_physics("neurram"))
        assert endurance is not None
        reading = row["k_star"]
        by_k = {v["K"]: v for v in reading["verdicts_by_K"]}
        for corner, writes in zip(CORNERS, (endurance.low, endurance.nominal, endurance.high)):
            closed = max(1, math.ceil(most * P.RATE_HZ * 3600.0 * P.FLOOR_H / writes - 1e-9))
            assert reading["per_corner"][corner] == closed
            assert by_k[closed]["per_corner"][corner] == "admitted"
            if closed > 1:
                assert by_k[closed - 1]["per_corner"][corner] == "inadmissible"
        low = reading["per_corner"]["low"]
        assert low == 25  # 6 writes per period x 30/s x 43,800 h x 8.52 pulses per write / 1e10 cycles
        assert by_k[low - 1]["verdict"] == "straddles" and by_k[low]["verdict"] == "admitted"
        assert reading["checked"]

    def test_the_write_terms_price_per_technology_or_refuse_by_name(self, rotated):
        row, _platform, _ = rotated
        assert row["writes"]["latency_write_exposed_s"] is not None  # both technologies declare t_write_per_unit
        refused = {r["name"]: r["reason"] for r in row["writes"]["refusals"]}
        assert row["writes"]["energy_write_mj"] is None and "neurram" in refused["energy_write_mj"]  # no commit energy


    def test_the_record_of_a_program_cut_into_passes_states_its_carry_on_the_value_domain(self, rotated):
        """The executed record seals what crossed the pass boundaries: mvm cores carry no spikes, so the boundary
        collapses (``run_pass_transfer``'s rule) over one evaluation."""
        from mimarsinan.deployment_record.design_point.pricing import executed_record
        _row_, platform, hybrid = rotated
        record = executed_record(hybrid, platform, weight_bits=8, label="test")
        assert record.schedule.carry is not None and record.schedule.carry.transfer == "collapse"
        assert record.schedule.pass_count == len([s for s in hybrid.stages if s.kind == "neural"])


class TestResidentStaticTiles:
    """8 NeuRRAM cores and 4 macros under scheduling: the static tiles fit side by side (pinned), only the operand
    cells are written, every inference, on SRAM: admitted from K = 1 at every corner."""

    def test_admitted_from_k_one(self, resident):
        row, platform, _ = resident
        assert row["admission"]["verdict"] == "admitted"
        assert row["k_star"]["per_corner"] == {c: 1 for c in CORNERS}
        assert row["writes"]["static_written_cells_per_period"] == 0
        assert row["lifetime"]["binding"] == "operand (sram_cim_macro)"
        assert wear_problems(_row(row, platform), DECLARATIONS) == []

    def test_the_record_plane_prices_its_stacked_split_and_latency_band(self, resident):
        from mimarsinan.deployment_record.build.program_census import record_program_passes
        from mimarsinan.deployment_record.design_point.pricing import record_pricing
        from mimarsinan.deployment_record.design_point.pricing import executed_record
        from mimarsinan.deployment_record.quantities.compute_technology import compute_split
        _row_, _platform, hybrid = resident
        program, _hybrid, _ = P.attention_on_hybrid(8, 4, True)
        variants = {"host": P.resolved_hybrid(8, 4, scheduling=True, overrides=P.HOST_BLOCK)}
        from mimarsinan.deployment_record.build.partition import compute_partition_record
        import copy
        model = copy.deepcopy(program.model).float()  # the certificate ran the model in fp64
        partition = compute_partition_record(program.ir, model, (3, 8, 8), 5, encoding_placement="subsume")
        records = {name: executed_record(hybrid, {**platform, "weight_bits": 8}, weight_bits=8, label="test",
                                         partition=partition)
                   for name, platform in variants.items()}
        pricing = record_pricing(records, GRID)
        assert pricing["plane"].startswith("record")
        assert [(v["variant"], v["K"]) for v in pricing["variants"]] == [("host", k) for k in GRID]
        at_one = pricing["variants"][0]
        split = at_one["energy_split_mj"]
        record = records["host"]
        shares = {s.core_type: s.events for s in compute_split(record.identity.platform,
                                                               record_program_passes(record)).shares}
        e_mac = {n: get_platform_physics(n).band("e_mac").nominal for n in ("neurram", "sram_cim_macro")}
        assert shares[(256, 256)] > 0 and shares[(128, 1024)] > 0
        dynamic = (shares[(256, 256)] * e_mac["neurram"] + shares[(128, 1024)] * e_mac["sram_cim_macro"]) * 1e3
        assert split["dynamic"][1] >= dynamic * (1 - 1e-9)  # MAC events at each type's e_mac (+ any periphery)
        sram = get_platform_physics("sram_cim_macro")
        cells = sum(sum(s.operand_core_cells or ()) for s in record.schedule.segments())
        assert cells > 0
        assert split["operand_write"][1] == pytest.approx(
            cells * sram.band("e_write_per_cell").nominal * sram.band("write_verify_iterations").nominal * 1e3,
            rel=1e-9)
        assert split["static_write"] == [0.0, 0.0, 0.0]
        assert at_one["energy_mj"] is not None and at_one["split_closes"], at_one
        assert at_one["latency_s"] is not None and at_one["lifetime_verdict"] == "admitted"
        assert at_one["refusals"] == {"energy_per_inference_mj": None, "e2e_latency_s": None}
