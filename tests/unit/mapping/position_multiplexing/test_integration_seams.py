"""Seams of the pack unit (W02 position multiplexing, the fix round's physical tiles) with N2's operand core types and
the closure's per-axon weight sign, on the integrated tree.

1. An operand bank's payload is rewritten every pass (N2 design §2.7), so no run of passes serves one copy of it, and
   the shape-only plane keeps a product as a host op where the executed plane lowers it onto operand cores. Declared
   ``position_multiplexing`` beside an operand core type is refused by name at every door (its legal set admits only
   false, as beside pruning and relays) and by the one pass planner on a segment holding operand banks; before, the
   doors admitted it and the planner silently gave such a segment the capacity split.
2. Under per-axon weight sign no instance is its bank's region on either plane (the closure's split owns its rows),
   so position multiplexing composes nothing there: both planes plan the replicated program (pinned, no change).
3. A fused core of k per-axon physical cores carries one sign register per member: each tile signs the used rows it
   holds (``tile_rows``), on the candidate, build-time record and sealed record planes (pinned, no change).
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.deployment_record.build.payload_sizes import core_payload_bytes
from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    mapping_program_passes,
    record_program_passes,
)
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.deployment_record.schema import SegmentRecord
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, segment_specs_for
from mimarsinan.mapping.packing.residency_tiles import tile_extents, tile_rows
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy, PositionMultiplexingError
from mimarsinan.mapping.support.position_multiplexing import shares_a_bank
from mimarsinan.mapping.support.schedule.pass_planner import plan_segment_passes

from unit.chip_simulation.value_run.operand_programs import attention_program
from unit.config_schema.test_target_doors import PC, _refused_by_name, _verdicts
from unit.deployment_record.record_fixtures import make_full_record
from unit.deployment_record.test_sign_register_bytes import _t0_61_mapping
from unit.mapping.packing.test_operand_sign_granularity import OPERAND, STATIC, SYNAPSE, _document

from . import vehicles as V

AXON = {"weight_sign_granularity": "per_axon"}


class _Semantics:
    retimed = False
    is_cycle = False
    is_cascade = False
    latency_group_count = None
    chip_latency = None


class TestPositionMultiplexingBesideAnOperandCoreType:
    def test_the_declaration_admits_only_false_beside_an_operand_core_type(self):
        entry = REGISTRY["position_multiplexing"]
        state = {"allow_scheduling": True, "lif_depth_balancing_relays": False}  # relays off (carried minor N1)
        assert tuple(entry.legal_values({**state, "cores": [STATIC, OPERAND]})) == (False,)
        assert tuple(entry.legal_values({**state, "cores": [STATIC]})) == (False, True)

    def test_every_door_refuses_it_by_name(self, tmp_path):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": True}
        _refused_by_name(doc, tmp_path, "position_multiplexing")

    def test_an_operand_chip_declaring_it_false_passes_every_door(self, tmp_path):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": False}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def _operand_specs(self):
        return [LayoutSoftCoreSpec(8, 8, latency_tag=0, segment_id=0, bank_id=7, core_role=OPERAND_ROLE,
                                   name=f"o{i}") for i in range(3)]

    def test_the_planner_refuses_it_on_a_segment_holding_operand_banks(self):
        with pytest.raises(PositionMultiplexingError, match="operand bank"):
            plan_segment_passes(self._operand_specs(), 4, core_types=[LayoutHardCoreType(16, 16, 4)],
                                position_multiplexing=True)

    def test_undeclared_an_operand_segment_takes_the_capacity_split_as_before(self):
        plan = plan_segment_passes(self._operand_specs(), 4, core_types=[LayoutHardCoreType(16, 16, 4)])
        assert plan.feasible and not plan.residency_applied and plan.run_heads is None

    def test_a_direct_build_of_an_operand_program_refuses_it_by_name(self):
        program = attention_program("softmax")
        caps = ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False, allow_scheduling=True,
                                max_schedule_passes=128, position_multiplexing=True)
        with pytest.raises(PositionMultiplexingError, match="operand bank"):
            build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=[dict(STATIC), dict(OPERAND)],
                                           strategy=MappingStrategy.resolve(caps), max_schedule_passes=128)


class TestPositionMultiplexingOnAPerAxonPool:
    #: Per-axon sign doubles a mixed row, so the vehicle's 145-row conv cores need 320 rows (no coalescing declared).
    CORES = [{"max_axons": 320, "max_neurons": 32, "count": 64, "has_bias": True, **AXON}]

    def test_no_instance_is_bank_backed_on_the_builders_plane(self):
        ir = V.integer_resnet().ir
        cores = list(ir.get_neural_cores())
        assert any(c.weight_bank_id is not None for c in cores)
        specs, _, _ = segment_specs_for(self.CORES)(cores, ir_graph=ir, hardware_bias=True)
        assert all(s.bank_id is None for s in specs) and not shares_a_bank(specs)

    def test_both_declarations_build_the_same_replicated_program(self):
        ir = V.integer_resnet().ir
        plain = V.neural(V.build(copy.deepcopy(ir), copy.deepcopy(self.CORES), False))
        declared = V.neural(V.build(copy.deepcopy(ir), copy.deepcopy(self.CORES), True))
        assert len(plain) == len(declared)
        assert [bool(s.schedule_weights_resident) for s in plain] == [bool(s.schedule_weights_resident)
                                                                       for s in declared]
        assert [len(s.hard_core_mapping.cores) for s in plain] == [len(s.hard_core_mapping.cores) for s in declared]


class TestTheSignRegisterOfAFusedCorePerTile:
    def test_a_core_signs_its_rows_member_by_member(self):
        assert tile_rows(25, ((10, 8),) * 3) == (10, 10, 5)
        assert tile_rows(0, ((10, 8),) * 3) == (0, 0, 0)
        assert tile_rows(643, ((256, 256),) * 3) == (256, 256, 131)

    def test_the_candidate_plane_signs_each_tile(self):
        specs = [LayoutSoftCoreSpec(25, 4, latency_tag=0, segment_id=0, name="w")]
        declared = [{"max_axons": 10, "max_neurons": 8, "count": 6, **AXON}]
        frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(10, 8, 6)], census=None,
                                     allow_scheduling=False, allow_neuron_splitting=False, allow_coalescing=True,
                                     max_schedule_passes=8, declared_cores=declared)
        (program,) = frag.pass_programs
        assert program.core_sign_rows == (25,) and program.core_tile_sign_rows == ((10, 10, 5),)
        (row,) = candidate_program_passes(specs, frag, semantics=_Semantics(), timesteps=None).rows
        assert row.core_sign_rows == (10, 10, 5) and row.core_cells == (40, 40, 20)
        # each member's payload is its cells and its own register, by the one rule
        assert [row.core_bytes(i, 4) for i in range(3)] == [core_payload_bytes(c, 4, r)
                                                            for c, r in zip((40, 40, 20), (10, 10, 5))]
        census = period_census(candidate_program_passes(specs, frag, semantics=_Semantics(), timesteps=None),
                               weight_bits=4, capacity={(10, 8): 3}).census
        assert [p.cores for p in census.passes] == [3] and all(p.resident for p in census.passes)

    def test_the_build_time_record_plane_signs_each_member(self):
        mapping = _t0_61_mapping()
        rows = mapping_program_passes(mapping).rows
        cores = [c for s in mapping.stages if s.kind == "neural" for c in s.hard_core_mapping.cores]
        fused = [c for c in cores if len(getattr(c, "fused_component_axons", None) or ()) > 1]
        assert fused, "t0_61 fuses its per-axon cores on ODIN"
        signed = [r for row in rows for r in row.core_sign_rows]
        assert len(signed) == sum(len(getattr(c, "fused_component_axons", None) or (0,)) for c in cores)
        expected = []
        for core in cores:
            members = getattr(core, "fused_component_axons", None) or (core.axons_per_core,)
            used = core.axons_per_core - core.available_axons
            expected += [r for r, _ in tile_extents(used, 0, tuple((int(a), 0) for a in members))]
        assert signed == expected and sum(signed) == 1285

    def test_the_sealed_record_plane_signs_each_declared_member(self):
        record = make_full_record()
        stages = list(record.schedule.stages)
        first = next(i for i, s in enumerate(stages) if isinstance(s, SegmentRecord))
        cores = list(stages[first].cores)
        cores[0] = replace(cores[0], axons=512, axons_used=300, sign_register_rows=300)
        stages[first] = replace(stages[first], cores=tuple(cores))
        fused = replace(record, schedule=replace(record.schedule, stages=tuple(stages)))
        row = record_program_passes(fused).rows[0]
        assert row.core_sign_rows[:2] == (256, 44)
        assert row.core_bytes(0, 8) + row.core_bytes(1, 8) == (
            core_payload_bytes(row.core_cells[0], 8, 256) + core_payload_bytes(row.core_cells[1], 8, 44))
        assert math.ceil(256 / 8) + math.ceil(44 / 8) == math.ceil(300 / 8)


#: Executed on-chip lanes with the pool class (WI-5e): the classes beside the integer path's readout law.
POOL_LANES = {"onchip_arithmetic": {"classes": ["L", "E", "P"]},
              "readout_law": {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}}


class TestPositionMultiplexingBesideAPoolLane:
    """[integration exactness review F3, then U-L] A pool lane core (``mapping/lanes/ir/pool_cores.py``) carried no
    layout record, so the multiplexing planner could not order it and the builder refused a lane program with an
    on-chip pool after training ("this IR carries no layout record for its cores"); the legal set then admitted only
    false beside executed P lanes, refused by name at every door. [U-L] Every core the lane fusion inserts now carries
    a layout record of its own (``lanes/ir/layout_records.py``), so executed P lanes keep both values, as L and E
    lanes and lanes off the integer path (no readout law: a pool stays a host op) do; ``unit/mapping/
    pool_lane_multiplexing`` builds, certifies and prices those programs."""

    STATE = {"allow_scheduling": True, "lif_depth_balancing_relays": False}  # relays off (carried minor N1)

    def test_the_declaration_admits_both_values_beside_executed_pool_lanes(self):
        entry = REGISTRY["position_multiplexing"]
        state = {**self.STATE, "cores": [STATIC]}
        assert tuple(entry.legal_values({**state, **POOL_LANES})) == (False, True)  # [U-L] was (False,)
        clamp_and_add = {**POOL_LANES, "onchip_arithmetic": {"classes": ["L", "E"]}}
        assert tuple(entry.legal_values({**state, **clamp_and_add})) == (False, True)
        float_path = {"onchip_arithmetic": POOL_LANES["onchip_arithmetic"]}
        assert tuple(entry.legal_values({**state, **float_path})) == (False, True)
        off_ablation = {**POOL_LANES, "onchip_arithmetic": {"classes": []}}
        assert tuple(entry.legal_values({**state, **off_ablation})) == (False, True)

    def test_a_malformed_lane_declaration_is_left_to_its_own_rule(self):
        entry = REGISTRY["position_multiplexing"]
        malformed = {**POOL_LANES, "onchip_arithmetic": {"classes": ["P"]}}  # P without L
        assert tuple(entry.legal_values({**self.STATE, "cores": [STATIC], **malformed})) == (False, True)

    def test_a_pool_lane_chip_declaring_it_true_passes_every_door(self, tmp_path):
        """[U-L] Was refused by name at every door ("position_multiplexing")."""
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": True, **POOL_LANES}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_a_pool_lane_chip_declaring_it_false_passes_every_door(self, tmp_path):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": False, **POOL_LANES}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_a_clamp_and_add_lane_chip_declaring_it_true_passes_every_door(self, tmp_path):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": True, **POOL_LANES,
                   "onchip_arithmetic": {"classes": ["L", "E"]}}
        assert _verdicts(doc, tmp_path) == ([], [], None)
