"""The lane fidelity review's pricing findings (11_wave2/lanes/review_fidelity.md, F1 to F5): one lane-block rule on
both planes, the fused add's operand priced once (as lane work, not as crossbar cells), the pool lane's work as its
requantized outputs, the 45 nm block scaled to the target's node by its declared rule, and the chip-only energy
naming what it cannot price."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.absolute.output_stage import CHIP_ONLY_TERM, output_stage_note
from mimarsinan.deployment_record.platform_physics.lane_blocks import LaneBlockError, load_lane_block
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.deployment_record.quantities.lanes import NO_LANE_COUNT
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.ir.lane_law import operand_axons
from mimarsinan.mapping.lanes.install import LANE_REPORT
from mimarsinan.mapping.lanes.output_stage import planned_lane_ops
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from unit.chip_simulation.value_run.integer_programs import lenet5, mlp, residual_block_net
from unit.mapping.lanes.lane_programs import build_lane_program, cores, lane_platform
from unit.mapping.lanes.test_lane_reduction import transition_net

BLOCK = "ng45_os_lane"
LANES = {"classes": ["L", "E", "P"], "lanes_per_core": 4, "lane_block": BLOCK}


def _resnet(classes=("L", "E", "P")):
    return build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(classes))


class TestOperandPricedOnce:
    def test_lanes_on_and_off_price_equal_crossbar_cells_but_the_pool_cores(self):
        on, off = _resnet(), _resnet(None)
        report_on = CrossbarUtilizationReport.from_hybrid_mapping(build_identity_hybrid_mapping(ir_graph=on.ir))
        report_off = CrossbarUtilizationReport.from_hybrid_mapping(build_identity_hybrid_mapping(ir_graph=off.ir))
        pools = [c for c in cores(on) if c.perceptron_index is None]
        pool_cells = sum(c.get_input_count() * c.get_output_count() for c in pools)
        assert pools and report_on.cells_used - pool_cells == report_off.cells_used
        assert report_on.macs - pool_cells == report_off.macs
        assert report_on.axons_used - sum(c.get_input_count() for c in pools) == report_off.axons_used

    def test_every_operand_row_is_recorded_on_its_core_and_carried_by_each_fragment(self):
        from mimarsinan.mapping.lanes.ir.fusion import OPERAND_ROWS

        program = _resnet()
        fused = [c for c in cores(program) if getattr(c, OPERAND_ROWS, None) is not None]
        assert fused
        for core in fused:  # an owned core's own rows; a bank position core the bank's whole operand block
            rows = (program.ir.weight_banks[core.weight_bank_id].core_matrix.shape[1] if core.weight_bank_id is not None
                    else int((np.asarray(getattr(core, OPERAND_ROWS)) >= 0).sum()))
            assert operand_axons(core) == rows > 0, core.name
        identity = sum(operand_axons(c) for c in cores(program))
        mapping = program.packed_mapping(axons=256, neurons=8, count=99999)
        hard = [hc for s in mapping.stages if s.kind == "neural" for hc in s.hard_core_mapping.cores]
        assert sum(operand_axons(hc) for hc in hard) > identity, "split fragments keep every parent row"
        census = CrossbarUtilizationReport.from_hard_cores(hard)
        placed = sum(hc.axons_per_core - hc.available_axons for hc in hard)
        assert census.axons_used == placed - sum(operand_axons(hc) for hc in hard)

    def test_a_pool_lane_works_once_per_requantized_output(self):
        report = getattr(_resnet().ir, LANE_REPORT)
        assert report["lane_ops"]["pool"] == 16, "a global pool over 16 channels writes 16 outputs"


class TestPlannedLaneWork:
    @pytest.mark.parametrize("name,build", [
        ("residual_block", lambda p: build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=p)),
        ("lenet5", lambda p: build_lane_program(lenet5(), (1, 28, 28), num_classes=10, platform=p)),
        ("mlp", lambda p: build_lane_program(mlp(), (8,), num_classes=4, platform=p)),
        ("transition", lambda p: build_lane_program(transition_net(), (3, 16, 16), num_classes=10, platform=p)),
    ])
    def test_the_plan_on_the_model_counts_what_the_fusion_runs(self, name, build):
        platform = lane_platform()
        program = build(platform)
        assert planned_lane_ops(program.model.get_mapper_repr(), platform) == getattr(program.ir, LANE_REPORT)[
            "lane_ops"], name

    def test_lanes_not_executed_plan_no_work_and_no_declaration_plans_nothing(self):
        program = _resnet(None)
        declared = {"onchip_arithmetic": {"classes": ["L", "E", "P"]}}
        assert planned_lane_ops(program.model.get_mapper_repr(), declared) == {"clamp": 0, "add": 0, "pool": 0}
        assert planned_lane_ops(program.model.get_mapper_repr(), {}) is None


def _platform(**extra):
    return {"onchip_arithmetic": dict(LANES), "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 16},
            "cores_per_tile_resolved": 2, **extra}


class TestCandidatePlane:
    def _context(self, platform, lane_ops):
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform

        return candidate_context_from_platform(platform, lane_ops=lane_ops)

    def _quantities(self, context):
        from mimarsinan.deployment_record.quantities.from_candidate import from_candidate

        return from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                              context=context)

    def test_a_lane_chip_candidate_prices_its_lane_block_by_the_record_rule(self):
        from mimarsinan.deployment_record.quantities.lanes import lane_block_quantities

        context = self._context(_platform(), {"clamp": 10, "add": 5, "pool": 3})
        quantities = self._quantities(context)
        assert quantities.get("lanes_per_tile").value == 8.0 and quantities.get("lane_ops").value == 18.0
        assert quantities.get("lane_ops").provenance == "modeled"
        record_values, _ = lane_block_quantities(classes=("E", "L", "P"), lanes_per_core=4, cores_per_tile=2,
                                                 lane_ops=18, provenance="measured")
        assert {k: v.value for k, v in record_values.items()} == {
            k: quantities.get(k).value for k in ("lanes_per_tile", "lane_ops")}
        stage = context.output_stage
        assert stage is not None and stage["label"] == "estimated" and stage["deployed"] is None

    def test_a_float_program_on_a_lane_chip_prices_the_block_and_no_lane_work(self):
        platform = _platform()
        platform.pop("readout_law")
        quantities = self._quantities(self._context(platform, {"clamp": 10, "add": 5, "pool": 3}))
        assert quantities.get("lanes_per_tile").value == 8.0 and quantities.get("lane_ops").value == 0.0

    def test_a_chip_without_lanes_keeps_its_candidate_context(self):
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform

        context = candidate_context_from_platform({}, lane_ops=None)
        assert context.lane_quantities == () and context.output_stage is None

    def test_the_candidate_report_prints_its_output_stage_and_chip_only_energy(self):
        from mimarsinan.deployment_record.cost.absolute.report import candidate_cost_report

        rows = {k: {"nominal": v, "unit": u, "note": "test lane row"}
                for k, v, u in (("area_per_lane", 1.0, "um^2"), ("e_lane_op", 1.0, "pJ"), ("t_lane_op", 1.0, "ns"))}
        mac = {"e_mac": {"nominal": 0.1, "unit": "pJ", "note": "test compute energy"}}
        physics = resolve_platform_physics("", {**mac, **rows}, lane_block=BLOCK)
        context = self._context(_platform(), {"clamp": 10, "add": 5, "pool": 3})
        base = {"synaptic_events": 1e6, "cells_used": 1e6, "macs": 1e6, "host_macs": 0, "compute_op_count": 0,
                "tiles": 10.0}
        quantities = Quantities({**{k: QuantityValue(float(v), "static") for k, v in base.items()},
                                 **dict(context.lane_quantities)})
        report = candidate_cost_report(quantities, physics, output_stage=context.output_stage)
        notes = [n for n in report.notes if n.startswith("output stage")]
        assert len(notes) == 1 and notes[0].startswith("output stage (estimated): lanes ELP via lane block")
        assert "lane ops 18" in notes[0] and "chip-only energy" in notes[0] and "lane-free shape" in notes[0]
        assert any(t.name == CHIP_ONLY_TERM for t in report.energy)


class TestLaneCount:
    def test_lanes_without_a_lane_count_withhold_it_by_name_and_price_their_work(self):
        from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
        from mimarsinan.deployment_record.quantities.from_candidate import from_candidate

        platform = _platform()
        platform["onchip_arithmetic"] = {"classes": ["L", "E", "P"], "lane_block": BLOCK}
        context = candidate_context_from_platform(platform, lane_ops={"clamp": 4})
        quantities = from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                                    host_side_segment_count=None, context=context)
        assert not quantities.has("lanes_per_tile") and quantities.withheld_reason("lanes_per_tile") == NO_LANE_COUNT
        assert quantities.get("lane_ops").value == 4.0


class TestNodeScaling:
    def test_the_block_declares_its_node_supply_and_rule(self):
        block = load_lane_block(BLOCK)
        assert (block.technology_node_nm, block.supply_v) == (45.0, 1.1)
        assert dict(block.scaling) == {"area": "node^2", "energy": "node*V^2", "time": "none"}

    def test_the_22nm_bundle_prices_the_block_at_its_own_node_and_supply(self):
        physics = resolve_platform_physics("generic_estimated_22nm", {}, lane_block=BLOCK)
        assert physics is not None
        block = load_lane_block(BLOCK).constants
        area, energy, time = (physics.constants[k] for k in ("area_per_lane", "e_lane_op", "t_lane_op"))
        assert math.isclose(area.nominal, block["area_per_lane"]["nominal"] * (22 / 45) ** 2, rel_tol=1e-12)
        assert math.isclose(energy.nominal, block["e_lane_op"]["nominal"] * (22 / 45) * (0.8 / 1.1) ** 2,
                            rel_tol=1e-12)
        assert math.isclose(energy.low, block["e_lane_op"]["low"] * (22 / 45) * (0.8 / 1.1) ** 2, rel_tol=1e-12)
        assert time.nominal == block["t_lane_op"]["nominal"]
        assert "(22/45)^2" in area.note and "(0.8/1.1)^2" in energy.note and "unscaled (rule none)" in time.note

    def test_a_target_without_a_supply_scales_energy_by_the_node_and_says_so(self):
        physics = resolve_platform_physics("isaac_like", {}, lane_block=BLOCK)
        assert physics is not None
        energy = physics.constants["e_lane_op"]
        assert math.isclose(energy.nominal, load_lane_block(BLOCK).constants["e_lane_op"]["nominal"] * 32 / 45,
                            rel_tol=1e-12)
        assert "no supply" in energy.note

    def test_a_block_beside_physics_of_no_declared_node_is_refused_by_name(self):
        mac = {"e_mac": {"nominal": 0.1, "unit": "pJ", "note": "operator compute energy"}}
        with pytest.raises(LaneBlockError, match="undeclared\\) node"):
            resolve_platform_physics("", mac, lane_block=BLOCK)
        rows = {k: {"nominal": 1.0, "unit": u, "note": "operator lane row"}
                for k, u in (("area_per_lane", "um^2"), ("e_lane_op", "pJ"), ("t_lane_op", "ns"))}
        assert resolve_platform_physics("", {**mac, **rows}, lane_block=BLOCK) is not None, "declared rows: no block"

    def test_the_scaled_lane_keeps_the_toggles_share_of_a_node_scaled_tile(self):
        """arith_toggle.md section 6.1: L+E with its 64 x 64 macro table is 13,048 um^2, 11.3% of the 45 nm G3 tile
        (0.1155 mm^2). Scaled by node^2, as a tile is, the block keeps that share of the G3 tile scaled to 22 nm. The
        22 nm bundle's own 64-column slice is denser (0.09 um^2 per bitcell, no macro periphery): there the scaled
        lane is 20% of slice plus lane (fix_report.md), the profile's array estimate rather than the block."""
        physics = resolve_platform_physics("generic_estimated_22nm", {}, lane_block=BLOCK)
        assert physics is not None
        tile_22 = 0.1155e6 * (22 / 45) ** 2
        assert math.isclose(physics.constants["area_per_lane"].nominal / tile_22, 13047.9 / 0.1155e6, rel_tol=1e-12)
        assert 0.069 <= 13047.9 / 0.1155e6 <= 0.130, "the toggle's band"


class TestChipOnlyEnergyNamesItsUnpriced:
    def test_an_undeclared_chip_constant_is_named_in_the_term_and_the_note(self):
        rows = {k: {"nominal": v, "unit": u, "note": "test lane row"}
                for k, v, u in (("area_per_lane", 1.0, "um^2"), ("e_lane_op", 1.0, "pJ"), ("t_lane_op", 1.0, "ns"))}
        mac = {"e_mac": {"nominal": 0.1, "unit": "pJ", "note": "test compute energy"}}
        physics = resolve_platform_physics("", {**mac, **rows}, lane_block=BLOCK)
        values = {"synaptic_events": 1e6, "cells_used": 1e6, "macs": 1e6, "lane_ops": 1e5, "host_macs": 0,
                  "compute_op_count": 0, "lanes_per_tile": 4.0, "tiles": 10.0, "noc_value_bytes": 5e5}
        pricing = price_absolute(Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()}),
                                 physics)
        terms = {t.name: t for t in pricing.terms}
        assert "unpriced: e_noc_value_per_byte_hop(noc_value)" in terms[CHIP_ONLY_TERM].source
        assert "unpriced: e_noc_value_per_byte_hop" in terms["energy_dynamic_mj"].source
        stage = {"label": "executed", "classes": ["E", "L", "P"], "lane_block": BLOCK, "lanes_per_core": 4,
                 "deployed": {"lane_ops": {"add": 1}, "host_ops": 0, "host_round_trips": 0, "host_barriers": 0,
                              "host_bytes": 0}, "notes": []}
        assert "(unpriced: e_noc_value_per_byte_hop(noc_value))" in output_stage_note(stage, pricing)


    def test_a_lane_block_point_whose_headline_refuses_refuses_its_chip_only_term_by_name(self):
        physics = resolve_platform_physics("generic_estimated_22nm", {}, lane_block=BLOCK)
        values = {"synaptic_events": 1e6, "cells_used": 1e6, "macs": 1e6, "lane_ops": 1e5, "lanes_per_tile": 4.0,
                  "tiles": 10.0, "cores_physical": 40.0}  # no latency census: static power cannot be priced
        pricing = price_absolute(Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()}),
                                 physics)
        refused = {r.name: r.reason for r in pricing.refusals}
        assert CHIP_ONLY_TERM not in {t.name for t in pricing.terms}
        assert refused[CHIP_ONLY_TERM] == refused["energy_per_inference_mj"] and "static power" in refused[CHIP_ONLY_TERM]
        plain = {k: v for k, v in values.items() if k not in ("lane_ops", "lanes_per_tile")}
        pricing = price_absolute(Quantities({k: QuantityValue(float(v), "static") for k, v in plain.items()}), physics)
        assert CHIP_ONLY_TERM not in {r.name for r in pricing.refusals}, "no lane block, no chip-only term"

    def test_a_refused_write_energy_refuses_the_chip_only_term_where_the_host_refuses(self):
        """Carried minor N4 (lanes verify2, reproducer V4): where the host refuses the headline, a declared write
        policy whose write energy is refused refuses the chip-only term by that reason, as the host-priced branch
        does, instead of pricing the chip without its writes."""
        from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
        from mimarsinan.deployment_record.quantities.write_law import WritePolicy

        rows = {k: {"nominal": v, "unit": u, "note": "test lane row"}
                for k, v, u in (("area_per_lane", 1.0, "um^2"), ("e_lane_op", 1.0, "pJ"), ("t_lane_op", 1.0, "ns"))}
        declared = {"e_mac": {"nominal": 0.1, "unit": "pJ", "note": "test compute energy"},
                    "host_compute_rate": {"nominal": 1e9, "unit": "1", "note": "test host rate"}, **rows}
        physics = resolve_platform_physics("", declared, lane_block=BLOCK)
        values = {"synaptic_events": 1e6, "cells_used": 1e6, "macs": 1e6, "lane_ops": 1e5, "lanes_per_tile": 4.0,
                  "tiles": 10.0, "host_macs": 1e6, "compute_op_count": 1}
        quantities = Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})
        priced = price_absolute(quantities, physics)
        assert CHIP_ONLY_TERM in {t.name for t in priced.terms}, "control: no writes declared, the term is priced"
        writes = WriteInputs(census=None, policy=WritePolicy(batch_k=4), unavailable="no census (test)")
        pricing = price_absolute(quantities, physics, writes=writes)
        refused = {r.name: r.reason for r in pricing.refusals}
        assert "energy_per_inference_mj" in refused and "energy_write_mj" in refused
        assert CHIP_ONLY_TERM not in {t.name for t in pricing.terms}
        assert "the write energy is refused" in refused[CHIP_ONLY_TERM] and "no census (test)" in refused[CHIP_ONLY_TERM]


def test_operand_axon_counts_are_absent_without_lanes():
    program = _resnet(None)
    assert all(operand_axons(c) == 0 and not hasattr(c, "lane_operand_axons") for c in cores(program))
    mapping = program.packed_mapping(axons=256, neurons=8, count=99999)
    assert not any(hasattr(hc, "lane_operand_axons") for s in mapping.stages if s.kind == "neural"
                   for hc in s.hard_core_mapping.cores)
    assert np.isfinite(CrossbarUtilizationReport.from_hybrid_mapping(mapping).cells_used)
