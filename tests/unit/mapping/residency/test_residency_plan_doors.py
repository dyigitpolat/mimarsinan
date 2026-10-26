"""[integration pass 2 fix round, exactness review F2 and F3] A declared ``residency_plan`` that the build refuses by name
after training is refused by name at every door first.

F3. The scheduled builder refuses a residency plan for a program with dynamic operand banks (N2): the plan pins cores
by payload, and its units do not yet state which cores hold operand banks (rewritten every inference, never pinned).
Every door admitted the declaration beside an operand core type. Keyed on the declaration as the other operand rules
are (``position_multiplexing``'s legal set, ``lane_operand_conflict``), a residency plan beside an operand core type is
refused at every door under the planner's message.

F2. The residency planner takes each layer's tile groups as units in layer order (``residency_units``). A lane program
with fused adds (WI-5e class E) is no program order in that order, so the build refuses every step ("layer order is not
a program order"). Until the units are ordered by the layer graph, a residency plan beside executed E lanes is refused
at every door by name, keyed on the declaration as the pool-lane rule of ``position_multiplexing`` is. L lanes alone
(the floor and stateless steps build and run exact: the exactness review's ``lane_floor.json``) and lanes off the
integer path keep it.
"""

from __future__ import annotations

import pytest

from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors
from mimarsinan.mapping.support.residency_declaration import OPERAND_RESIDENCY_REFUSAL

from unit.config_schema.test_target_doors import PC, _refused_by_name, _verdicts
from unit.mapping.packing.test_operand_sign_granularity import OPERAND, STATIC, SYNAPSE, _document

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}
STEPS = ({"step": "floor"}, {"step": "stateless"})
SCHEDULED = {"allow_scheduling": True}
ADD_LANES = {"onchip_arithmetic": {"classes": ["L", "E"]}, "readout_law": LAW}
POOL_LANES = {"onchip_arithmetic": {"classes": ["L", "E", "P"]}, "readout_law": LAW}
CLAMP_LANES = {"onchip_arithmetic": {"classes": ["L"]}, "readout_law": LAW}


def _step_id(plan):
    return plan["step"]


class TestResidencyPlanBesideAnOperandCoreType:
    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_the_door_rule_names_it_once_with_the_planners_message(self, plan):
        errors = platform_residency_errors({**SCHEDULED, "cores": [STATIC, OPERAND], "residency_plan": plan})
        assert len(errors) == 1 and OPERAND_RESIDENCY_REFUSAL in errors[0] and "cores [1]" in errors[0], errors

    def test_the_rule_admits_either_declaration_alone(self):
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC], "residency_plan": {"step": "floor"}}) == []
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC, OPERAND]}) == []

    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_every_door_refuses_it_by_name(self, tmp_path, plan):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan}
        _refused_by_name(doc, tmp_path, OPERAND_RESIDENCY_REFUSAL)

    def test_an_operand_chip_without_a_plan_and_a_static_chip_with_one_pass_every_door(self, tmp_path):
        doc = _document([{**STATIC, **SYNAPSE}, OPERAND])
        doc[PC] = {**doc[PC], **SCHEDULED}
        assert _verdicts(doc, tmp_path) == ([], [], None)
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": {"step": "floor"}}
        assert _verdicts(doc, tmp_path) == ([], [], None)


class TestResidencyPlanBesideFusedAddLanes:
    @pytest.mark.parametrize("lanes", (ADD_LANES, POOL_LANES), ids=("L_E", "L_E_P"))
    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_the_door_rule_names_it_once(self, plan, lanes):
        errors = platform_residency_errors({**SCHEDULED, "cores": [STATIC], "residency_plan": plan, **lanes})
        assert len(errors) == 1 and "residency_plan" in errors[0] and "class E" in errors[0], errors

    def test_clamp_lanes_lanes_off_the_integer_path_and_no_plan_are_admitted(self):
        plan = {"residency_plan": {"step": "floor"}}
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC], **plan, **CLAMP_LANES}) == []
        float_path = {"onchip_arithmetic": ADD_LANES["onchip_arithmetic"]}
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC], **plan, **float_path}) == []
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC], **ADD_LANES}) == []

    def test_a_malformed_lane_declaration_is_left_to_its_own_rule(self):
        malformed = {"onchip_arithmetic": {"classes": ["E"]}, "readout_law": LAW}  # E without L
        assert platform_residency_errors({**SCHEDULED, "cores": [STATIC], "residency_plan": {"step": "floor"},
                                          **malformed}) == []

    @pytest.mark.parametrize("lanes", (ADD_LANES, POOL_LANES), ids=("L_E", "L_E_P"))
    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_every_door_refuses_it_by_name(self, tmp_path, plan, lanes):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan, **lanes}
        _refused_by_name(doc, tmp_path, "executed fused-add lanes")

    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    def test_a_clamp_lane_chip_declaring_it_passes_every_door(self, tmp_path, plan):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan, **CLAMP_LANES}
        assert _verdicts(doc, tmp_path) == ([], [], None)
