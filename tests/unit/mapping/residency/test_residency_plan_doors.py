"""[integration pass 2 fix round, exactness review F2 and F3] A declared ``residency_plan`` beside an operand core type
or beside executed lanes is admitted at every door, because the build plans it.

F3. The scheduled builder refused a residency plan for a program with dynamic operand banks (N2), and the fix round
refused one beside an operand core type at every door: the plan pins cores by payload, and its units did not state
which cores hold operand banks (rewritten every inference, never pinned). The planner now states them and prices
their tiles (``residency_units``' producer cells, ``pinned_tiles``, the fronts' operand bound), so every door admits a
residency plan beside an operand core type. [G2] The one residency-plan rule at the doors is the per-axon one
(``platform_residency_errors``, ``test_per_axon_residency_door.py``): it keeps both of these verdicts.

F2. The residency planner took each layer's tile groups as units in layer order, in which a lane program with fused
adds (WI-5e class E) is no program order, so the build refused every step and the fix round refused a residency plan
beside executed E lanes at every door. The units are now ordered by the layer graph (``residency_units``, the F2
unit), so every door admits a residency plan beside executed lanes of any class, as it admits one beside clamp lanes
or lanes off the integer path.

Executed lanes and an operand core type still do not share a chip: that is the lane rule
(``lane_operand_conflict``), keyed on the lanes, with or without a residency plan.
"""

from __future__ import annotations

import pytest

from mimarsinan.config_schema.registry.core_type_doors import platform_lane_errors

from unit.config_schema.test_target_doors import PC, _verdicts
from unit.mapping.packing.test_operand_sign_granularity import OPERAND, STATIC, SYNAPSE, _document

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}
STEPS = ({"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1})
SCHEDULED = {"allow_scheduling": True}
ADD_LANES = {"onchip_arithmetic": {"classes": ["L", "E"]}, "readout_law": LAW}
POOL_LANES = {"onchip_arithmetic": {"classes": ["L", "E", "P"]}, "readout_law": LAW}
CLAMP_LANES = {"onchip_arithmetic": {"classes": ["L"]}, "readout_law": LAW}
SHARED_OPERAND = {**STATIC, "count": 12, "operand": {"payload_bits": 8}}


def _step_id(plan):
    return plan["step"]


class TestResidencyPlanBesideAnOperandCoreType:
    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_every_door_admits_it_beside_a_separate_operand_core_type(self, tmp_path, plan):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_every_door_admits_it_on_one_shared_type_declaring_the_operand(self, tmp_path, plan):
        doc = _document([SHARED_OPERAND])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_an_operand_chip_without_a_plan_and_a_static_chip_with_one_pass_every_door(self, tmp_path):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], **SCHEDULED}
        assert _verdicts(doc, tmp_path) == ([], [], None)
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": {"step": "floor"}}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_the_one_residency_plan_rule_at_the_doors_admits_operand_types_and_lanes(self):
        import importlib.util

        from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors

        plan = {**SCHEDULED, "residency_plan": {"step": "floor"}}
        assert platform_residency_errors({**plan, "cores": [{**STATIC, **SYNAPSE}, OPERAND]}) == []
        assert platform_residency_errors({**plan, "cores": [SHARED_OPERAND]}) == []
        assert platform_residency_errors({**plan, "cores": [STATIC], **POOL_LANES}) == []
        assert importlib.util.find_spec("mimarsinan.mapping.support." + "residency_declaration") is None


class TestResidencyPlanBesideFusedAddLanes:
    @pytest.mark.parametrize("lanes", (ADD_LANES, POOL_LANES), ids=("L_E", "L_E_P"))
    @pytest.mark.parametrize("plan", STEPS[:2], ids=_step_id)
    def test_every_door_admits_it(self, tmp_path, plan, lanes):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan, **lanes}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    @pytest.mark.parametrize("plan", STEPS[:2], ids=_step_id)
    def test_a_clamp_lane_chip_declaring_it_passes_every_door(self, tmp_path, plan):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan, **CLAMP_LANES}
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_an_operand_core_type_beside_executed_lanes_is_refused_by_the_lane_rule_alone(self):
        for plan in ({}, {"residency_plan": {"step": "floor"}}):
            errors = platform_lane_errors({**SCHEDULED, "cores": [STATIC, OPERAND], **plan, **ADD_LANES})
            assert len(errors) == 1 and "operand" in errors[0], errors
