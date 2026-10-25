"""W08: the output stage beside each priced point (lane work, host boundary bytes, host round trips, chip-only
energy), labelled executed or estimated, carried by the record only for a chip declaring lanes."""

from __future__ import annotations

import json
from dataclasses import replace

from mimarsinan.deployment_record.cost.absolute.metrics import price_absolute
from mimarsinan.deployment_record.cost.output_stage import CHIP_ONLY_TERM, output_stage_note
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.deployment_record.platform_physics.resolve import resolve_platform_physics
from mimarsinan.mapping.lanes.output_stage import EXECUTED, ESTIMATED, OutputStage, output_stage_of_program
from unit.chip_simulation.value_run.integer_programs import residual_block_net
from unit.deployment_record.record_fixtures import make_full_record
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform


def _lane_platform(**extra):
    platform = lane_platform(**extra)
    platform["onchip_arithmetic"].update(lanes_per_core=1, lane_block="ng45_os_lane")
    return platform


class TestOutputStageOfProgram:
    def test_a_lane_program_executes_its_lanes_and_crosses_the_host_twice(self):
        platform = _lane_platform()
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=platform)
        stage = output_stage_of_program(program.ir, platform)
        assert stage is not None and stage.label == EXECUTED and stage.placed is None
        d = stage.deployed
        assert d.host_ops == 0 and d.host_round_trips == 0 and d.neural_segments == 1
        assert d.boundary_elements["carry_in"] == d.boundary_elements["carry_out"] == 0
        assert d.boundary_elements["entry"] == 3 * 16 * 16 and d.boundary_elements["readout"] == 10
        assert set(d.lane_ops) == {"clamp", "add", "pool"} and d.lane_ops["pool"] == 16, "the pool's 16 outputs"
        assert any(note.startswith("operand rows:") for note in stage.notes), "the operand port's array overhead"
        again = OutputStage.from_dict(json.loads(json.dumps(stage.to_dict())))
        assert again == stage

    def test_a_chip_without_the_declaration_has_no_output_stage(self):
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(None))
        assert output_stage_of_program(program.ir, lane_platform(None)) is None

    def test_lanes_not_executed_print_the_placement_as_an_estimate(self):
        platform = lane_platform(None)
        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=platform)
        declared = {**platform, "onchip_arithmetic": {"classes": ["L", "E", "P"], "lanes_per_core": 1}}
        declared.pop("readout_law")  # the float value path: lanes declared, not executed
        stage = output_stage_of_program(program.ir, declared)
        assert stage is not None and stage.label == ESTIMATED
        assert stage.deployed.host_ops == 8 and stage.deployed.lane_ops_total == 0
        assert stage.placed is not None and stage.placed.host_ops == 0 and stage.placed.lane_ops_total > 0
        assert any("not executed" in note for note in stage.notes)


def _record_with(stage: dict) -> DeploymentRecord:
    record = make_full_record()
    platform = dict(record.identity.platform)
    platform["cores_per_tile_resolved"] = 2
    return replace(record, identity=replace(record.identity, platform=platform), output_stage=stage)


_STAGE = {"label": "executed", "classes": ["E", "L", "P"], "lanes_per_core": 4, "lane_block": "ng45_os_lane",
          "deployed": {"lane_ops": {"clamp": 10, "add": 5, "pool": 3}, "host_ops": 0, "host_round_trips": 0,
                       "host_barriers": 0, "neural_segments": 1,
                       "boundary_elements": {"entry": 3, "carry_in": 0, "carry_out": 0, "readout": 2},
                       "host_bytes": 5}, "placed": None, "notes": []}


class TestCandidatePoints:
    def test_a_level_row_is_an_estimated_placement(self):
        from mimarsinan.mapping.capability.census import level_census
        from mimarsinan.mapping.lanes.output_stage import declared_level, output_stage_of_level
        from mimarsinan.mapping.lanes.declaration import lane_declaration

        program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=lane_platform(None))
        platform = {"onchip_arithmetic": {"classes": ["L", "E", "P"], "lanes_per_core": 1}}
        row = level_census(program.ir, declared_level(lane_declaration(platform)), value_bits=4)
        stage = output_stage_of_level(row, platform)
        assert stage is not None and stage.label == ESTIMATED and stage.deployed.host_ops == 0
        assert stage.deployed.lane_ops_total == row.lane_ops
        assert output_stage_of_level(row, {}) is None


class TestRecordPlane:
    def test_the_record_carries_the_output_stage_only_when_set(self):
        plain = make_full_record()
        assert "output_stage" not in plain.to_dict()
        record = _record_with(_STAGE)
        data = record.to_dict()
        assert data["output_stage"]["label"] == "executed"
        assert DeploymentRecord.from_dict(json.loads(json.dumps(data))).output_stage == data["output_stage"]

    def test_the_record_prices_its_executed_lanes(self):
        quantities = from_record(_record_with(_STAGE))
        assert quantities.get("lanes_per_tile").value == 8.0 and quantities.get("lane_ops").value == 18.0
        assert not from_record(make_full_record()).has("lanes_per_tile")

    def test_lanes_not_executed_price_no_lane_work(self):
        stage = {**_STAGE, "label": "estimated"}
        assert from_record(_record_with(stage)).get("lane_ops").value == 0.0

    def test_the_off_ablation_prices_no_lanes_but_keeps_its_output_stage(self):
        stage = {**_STAGE, "label": "no lanes", "classes": [], "lanes_per_core": None, "lane_block": None}
        quantities = from_record(_record_with(stage))
        assert quantities.get("lanes_per_tile").value == 0.0 and quantities.get("lane_ops").value == 0.0

    def test_lanes_without_a_lane_count_withhold_the_lane_block_by_name_and_price_their_work(self):
        from mimarsinan.deployment_record.quantities.lanes import NO_LANE_COUNT

        stage = {**_STAGE, "lanes_per_core": None}
        quantities = from_record(_record_with(stage))
        assert not quantities.has("lanes_per_tile") and quantities.withheld_reason("lanes_per_tile") == NO_LANE_COUNT
        assert quantities.get("lane_ops").value == 18.0


class TestChipOnlyEnergy:
    def _quantities(self, **extra):
        values = {"synaptic_events": 1e6, "cells_used": 1e6, "macs": 1e6, "lane_ops": 1e5, "host_macs": 0,
                  "compute_op_count": 0, **extra}
        return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})

    def test_priced_beside_the_headline_for_a_lane_block_point(self):
        mac = {"e_mac": {"nominal": 0.1, "unit": "pJ", "note": "test-labelled compute energy"}}
        # Operator physics declares no node, so it declares its lane rows (a block's rows need a node to scale to).
        rows = {k: {"nominal": v, "unit": u, "note": "test-labelled lane row"}
                for k, v, u in (("area_per_lane", 3000.0, "um^2"), ("e_lane_op", 1.2, "pJ"), ("t_lane_op", 1.2, "ns"))}
        physics = resolve_platform_physics("", {**mac, **rows}, lane_block="ng45_os_lane")
        assert physics is not None
        with_lanes = price_absolute(self._quantities(lanes_per_tile=4.0, tiles=10.0), physics)
        terms = {t.name: t for t in with_lanes.terms}
        assert CHIP_ONLY_TERM in terms and "energy_per_inference_mj" in terms
        lane = 1e5 * physics.constants["e_lane_op"].nominal * 1e-12 * 1e3
        compute = 1e6 * 0.1 * 1e-12 * 1e3
        assert abs(terms[CHIP_ONLY_TERM].value - (compute + lane)) < 1e-12, "compute plus the lane block's work"
        assert terms[CHIP_ONLY_TERM].value == terms["energy_per_inference_mj"].value, "no host work, no host energy"
        without = price_absolute(self._quantities(), physics)
        assert CHIP_ONLY_TERM not in {t.name for t in without.terms}, "no lane block, no new term"
        note = output_stage_note(_STAGE, with_lanes)
        assert "chip-only energy" in note and "host round trips 0" in note and "lane ops 18" in note
