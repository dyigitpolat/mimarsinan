"""[U-L, L-1] The legal set admits the pool lane on the one-copy arm: beside executed P lanes, ``position_multiplexing``
and, under it, ``pipelined_multiplexing`` take both values, and every door admits a P-lane chip declaring them true.
The rule's other exclusions stay (pruning, depth-balancing relays, an operand core type, windowed LIF for the
pipelined form), and a pool lane never meets a spiking executor: the lane keys are value-domain keys, so a spiking
run declaring them is refused by name at the document and run doors and held dormant by the wizard (streamed LIF,
nevresim, SANA-FE and Loihi included)."""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY

from unit.config_schema.test_target_doors import PC, _load, _refused_by_name, _verdicts
from unit.mapping.packing.test_operand_sign_granularity import OPERAND, STATIC, _document

#: Executed on-chip lanes with the pool class (WI-5e): the classes beside the integer path's readout law.
POOL_LANES = {"onchip_arithmetic": {"classes": ["L", "E", "P"]},
              "readout_law": {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}}
STATE = {"allow_scheduling": True, "lif_depth_balancing_relays": False, "cores": [STATIC]}


class TestTheLegalSet:
    def test_position_multiplexing_takes_both_values_beside_executed_pool_lanes(self):
        assert tuple(REGISTRY["position_multiplexing"].legal_values({**STATE, **POOL_LANES})) == (False, True)

    def test_its_pipelined_form_takes_both_values_there_too(self):
        state = {**STATE, **POOL_LANES, "position_multiplexing": True}
        assert tuple(REGISTRY["pipelined_multiplexing"].legal_values(state)) == (False, True)

    @pytest.mark.parametrize("extra", [{"pruning": True}, {"lif_depth_balancing_relays": True},
                                       {"cores": [STATIC, OPERAND]}], ids=["pruning", "relays", "operand_core_type"])
    def test_the_rule_s_other_exclusions_stay_beside_pool_lanes(self, extra):
        state = {**STATE, **POOL_LANES, **extra}
        assert tuple(REGISTRY["position_multiplexing"].legal_values(state)) == (False,)
        assert tuple(REGISTRY["pipelined_multiplexing"].legal_values({**state, "position_multiplexing": True})) == (
            False,)


class TestTheDoors:
    @pytest.mark.parametrize("pipelined", [None, True], ids=["one_copy", "pipelined"])
    def test_a_pool_lane_chip_declaring_the_one_copy_arm_passes_every_door(self, tmp_path, pipelined):
        doc = _document([STATIC])
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": True, **POOL_LANES}
        if pipelined is not None:
            doc[PC]["pipelined_multiplexing"] = pipelined
        assert _verdicts(doc, tmp_path) == ([], [], None)

    def test_with_pruning_beside_it_every_door_still_refuses_it_by_name(self, tmp_path):
        doc = _document([STATIC])
        doc["deployment_parameters"] = {**doc["deployment_parameters"], "pruning": True, "pruning_fraction": 0.5}
        doc[PC] = {**doc[PC], "allow_scheduling": True, "position_multiplexing": True, **POOL_LANES}
        _refused_by_name(doc, tmp_path, "position_multiplexing")

    def test_a_spiking_run_declaring_pool_lanes_never_runs_them(self, tmp_path):
        """No spiking executor (streamed LIF, nevresim, SANA-FE, Loihi) meets a pool lane: the lane keys belong to
        the value domain, so the document door and the run door refuse them on a spiking run by the lane key's name,
        and the wizard holds them dormant (not run). Position multiplexing on a LIF run takes only false already
        (its recipe arms the depth-balancing relays)."""
        import copy

        from mimarsinan.config_schema.resolve import resolve_draft

        doc = _load("lif_deepmlp.json")
        doc[PC] = {**doc[PC], **POOL_LANES}
        document, _wizard, run = _verdicts(doc, tmp_path)
        assert any(m.startswith("onchip_arithmetic") for m in document), document
        assert isinstance(run, ToolchainRefusal) and "onchip_arithmetic" in str(run), repr(run)
        assert "onchip_arithmetic" in resolve_draft(copy.deepcopy(doc)).dormant
