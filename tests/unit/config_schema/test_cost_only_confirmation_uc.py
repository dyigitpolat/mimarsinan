"""[U-C C-7] The owner's confirmations of 2026-10-27 (decisions C1 and C3), pinned.

C1: the owner-reviewed cost-only set of S0 (26 rows) grows by the lane block's two pricing fields to 28. The tree's set
holds those 28 and the ten rows that later amendments added, each under its own record, so the set this tree pins
(38 rows, the theorem box's) is that composition and no other ([U-E3] the fifth: U-E's ``cores.compute_physics``;
[U-S2] the sixth: U-S's ``event_census_samples``; [U-F4] the seventh: U-F4's ``pipelined_multiplexing``; [U-C4] the
eighth and ninth: U-C4's ``cores.cells_per_weight`` and ``cores.weight_cell_rows``, ``WEIGHT_CELLS``; [U-L2] the
tenth: U-L2's ``reduction_core_banks``, cost-only with position multiplexing's hypotheses).
C3: one geometry may carry a static and an operand core type (a chip may have identical arrays in two roles); every
door admits it."""

from __future__ import annotations

import copy
import json
import os

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.deployment_record.build.write_census import core_capacity_of
from mimarsinan.deployment_record.quantities.write_law import write_policy_of
from mimarsinan.mapping.support.operand_lowering import operand_declaration
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

#: The owner-reviewed S0 set (RA/10_wave1/s0_declarations/owner_review.md, "Cost-only set (26 rows)").
S0_SET = frozenset({
    "activity_factor", "allow_coalescing", "allow_neuron_splitting", "allow_scheduling", "core_value_granularity",
    "cores.count", "cores.max_axons", "cores.max_neurons", "cores.write_granularity", "cores.write_units_in_parallel",
    "cores_per_tile", "inference_batch_K", "max_axons", "max_neurons", "max_schedule_passes",
    "platform_physics_overrides", "platform_physics_profile", "residency_plan", "residency_plan.index",
    "residency_plan.step", "tile_grid_cols", "tile_grid_rows", "write_overlap", "write_overlap.mode",
    "write_overlap.shadow_capacity_fraction", "write_ports",
})
#: Decision C1 (2026-10-27, "confirm"): the lane block's two pricing fields (RA/11_wave2/lanes/step2_report.md sec. 10).
LANE_FIELDS = frozenset({"onchip_arithmetic.lane_block", "onchip_arithmetic.lanes_per_core"})
#: The later amendments, each with its own record: sign-off L1 (CL-SG), the N2 design's sec. 2.9 table (accepted by
#: the owner, 2026-10-22) for the operand write source and N2b's write technology, the pack unit's W02, [U-E3]
#: U-E's E-6 (finalization plan sec. 2.2, N2 follow-up F5: the per-core-type compute physics, a pricing-only key,
#: ``certification/randomized/cost_draws.py``), [U-S2] U-S's S-2 (finalization plan sec. 2.2, decision B6 of
#: 2026-10-27: the opt-in event census's sample count, a pricing-only key,
#: ``config_schema/registry/decision_classes.py`` and ``certification/randomized/cost_draws.py``), [U-F4] U-F4's
#: P-1 (owner answer "Build it", 2026-10-28: pipelined position multiplexing, a declared decision of the pass planner,
#: cost-only in ``config_schema/registry/decision_classes.py``, deferred by name in
#: ``certification/randomized/cost_draws.py``), and [U-L2] U-L2's B-1 (W02's banked reduction cores, cost-only in
#: ``decision_classes.py``, deferred by name in ``cost_draws.py``; final_units/ul2, plan item B-1), and [thru:k] the
#: throughput branch's layer copies (layer_replication, wavefront_group_cores: cost-only, deferred by name;
#: PL/thru/design.md (b)), [thru:w2] and its batch pipeline (batch_pipeline_slots: cost-only, deferred by name).
LATER = frozenset({"cores.weight_sign_granularity", "cores.operand.write_source", "cores.write_physics",
                   "position_multiplexing", "cores.compute_physics", "event_census_samples",
                   "pipelined_multiplexing", "reduction_core_banks", "layer_replication", "wavefront_group_cores",
                   "batch_pipeline_slots"})
#: [U-C4] The cells per stored weight (B3 intake open item 4): two pricing-only core-type fields, recorded in
#: RA/11_wave2/final_units/uc4/report.md.
WEIGHT_CELLS = frozenset({"cores.cells_per_weight", "cores.weight_cell_rows"})


def _cost_only():
    out = set()
    for key, entry in REGISTRY.items():
        if entry.decision_class == COST_ONLY:
            out.add(key)
        schema = entry.value_schema
        if schema is not None:
            from mimarsinan.config_schema.registry.value_schema import nested_schemas
            out |= {path for path, sub in nested_schemas(schema, key) if sub.decision_class == COST_ONLY}
    return out


class TestCostOnlySet:
    def test_the_confirmed_28_rows_are_the_s0_set_and_the_lane_block_s_two_fields(self):
        assert len(S0_SET) == 26 and len(S0_SET | LANE_FIELDS) == 28
        assert S0_SET | LANE_FIELDS <= _cost_only()

    def test_the_tree_s_set_is_the_confirmed_28_and_the_later_amendments(self):
        assert _cost_only() == S0_SET | LANE_FIELDS | LATER | WEIGHT_CELLS
        assert len(_cost_only()) == 41


ONE_GEOMETRY = [{"max_axons": 64, "max_neurons": 64, "count": 6, "has_bias": True},
                {"max_axons": 64, "max_neurons": 64, "count": 2, "has_bias": True,
                 "operand": {"payload_bits": 8, "write_source": "host"}}]


class TestOneGeometryTwoRoles:
    """Decision C3: a static and an operand core type of one geometry are admitted at every door."""

    def _document(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs",
                            "mvm_aq_lenet5.json")
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        pc = {k: v for k, v in doc["platform_constraints"].items() if k not in ("cores", "max_axons", "max_neurons")}
        doc["platform_constraints"] = {**pc, "cores": copy.deepcopy(ONE_GEOMETRY)}
        return doc

    def test_the_document_door_and_the_wizard_admit_it(self):
        doc = self._document()
        assert not [m for m in validate_deployment_config(copy.deepcopy(doc)) if "cores" in m]
        assert not [row for row in resolve_draft(copy.deepcopy(doc)).errors if "cores" in row["message"]]

    def test_the_run_door_admits_it_and_reads_the_operand_type(self):
        resolved = build_platform_constraints_resolved(merge_pipeline_config(
            {"core_semantics": "mvm"}, {"cores": copy.deepcopy(ONE_GEOMETRY), "weight_bits": 8}))
        assert [bool(c.get("operand")) for c in resolved["cores"]] == [False, True]
        assert operand_declaration(resolved).payload_bits == 8

    def test_the_write_law_pools_the_geometry_and_states_one_technology(self):
        platform = {"cores": copy.deepcopy(ONE_GEOMETRY), "inference_batch_K": 4}
        assert core_capacity_of(platform) == {(64, 64): 8}
        policy = write_policy_of(platform)
        assert policy is not None and policy.core_technologies is None
        assert policy.producer_write_source == "host" and policy.payload_bits == 8

    @pytest.mark.parametrize("roles", [(False, True), (True, False)])
    def test_the_order_of_the_two_types_does_not_matter(self, roles):
        cores = [dict(ONE_GEOMETRY[0]), dict(ONE_GEOMETRY[1])]
        if roles == (True, False):
            cores.reverse()
        resolved = build_platform_constraints_resolved(merge_pipeline_config(
            {"core_semantics": "mvm"}, {"cores": cores, "weight_bits": 8}))
        assert operand_declaration(resolved).payload_bits == 8
