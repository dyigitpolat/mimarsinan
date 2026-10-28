"""[G2] A declared ``residency_plan`` on a chip that packs per axon is refused by name at every door.

On a chip whose core types declare per-axon weight sign (``weight_sign_granularity: per_axon``, ODIN's), the builder
places a mixed-sign logical row as a W+ and a W- row; the residency planner's plan plane does not read those
sign-split rows, so the plan's period census (resident cores, backing-store cells) and the built program's record
disagree, with outputs exact. Until the two planes agree, the document door, the wizard and the run door refuse a
residency plan on such a chip by name, before any training spend. The same chip without a plan, and a chip that
packs per synapse with one, keep every door's verdict.
"""

from __future__ import annotations

import pytest

from unit.config_schema.test_target_doors import DP, PC, _refused_by_name, _verdicts
from unit.config_schema.test_target_doors import _document as _target_document
from unit.mapping.packing.test_operand_sign_granularity import AXON, STATIC, SYNAPSE, _document

STEPS = ({"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1})
SCHEDULED = {"allow_scheduling": True}
NAMED = "residency_plan is not composed with per-axon weight sign"
#: A chip packs per axon when any of its core types declares it (``pool_sign_granularity``'s fold).
PER_AXON_CHIPS = {
    "one_per_axon_type": [{**STATIC, **AXON}],
    "a_per_axon_type_beside_a_per_synapse_type": [{**STATIC, **SYNAPSE}, {**STATIC, **AXON, "max_axons": 128}],
}
PER_SYNAPSE_CHIPS = {"declared_per_synapse": [{**STATIC, **SYNAPSE}], "undeclared": [STATIC]}


def _step_id(plan):
    return plan["step"]


def _planned(cores, plan):
    doc = _document(cores)
    doc[PC] = {**doc[PC], **SCHEDULED, "residency_plan": plan}
    return doc


def _odin(**platform):
    """The spiking fixture on the ODIN target (per-axon cores), scheduling declared once, where the fixture does."""
    doc = _target_document("odin")
    doc[DP] = {**doc[DP], **SCHEDULED}
    doc[PC] = {**doc[PC], **platform}
    return doc


class TestEveryDoorRefusesItByName:
    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    @pytest.mark.parametrize("chip", sorted(PER_AXON_CHIPS))
    def test_a_value_chip_that_packs_per_axon(self, tmp_path, chip, plan):
        _refused_by_name(_planned(PER_AXON_CHIPS[chip], plan), tmp_path, NAMED)

    @pytest.mark.parametrize("plan", STEPS[:2], ids=_step_id)
    def test_the_odin_target(self, tmp_path, plan):
        _refused_by_name(_odin(residency_plan=plan), tmp_path, NAMED)


class TestEveryOtherVerdictIsKept:
    def test_a_per_axon_chip_without_a_plan_passes_every_door(self, tmp_path):
        for cores in PER_AXON_CHIPS.values():
            doc = _document(cores)
            doc[PC] = {**doc[PC], **SCHEDULED}
            assert _verdicts(doc, tmp_path) == ([], [], None)

    @pytest.mark.parametrize("plan", STEPS, ids=_step_id)
    @pytest.mark.parametrize("chip", sorted(PER_SYNAPSE_CHIPS))
    def test_a_chip_that_packs_per_synapse_keeps_its_plan(self, tmp_path, chip, plan):
        assert _verdicts(_planned(PER_SYNAPSE_CHIPS[chip], plan), tmp_path) == ([], [], None)

    def test_the_odin_target_without_a_plan_passes_every_door(self, tmp_path):
        assert _verdicts(_odin(), tmp_path) == ([], [], None)


class TestTheDoorRule:
    def test_it_names_the_per_axon_core_types_once(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors

        cores = PER_AXON_CHIPS["a_per_axon_type_beside_a_per_synapse_type"]
        errors = platform_residency_errors({**SCHEDULED, "cores": cores, "residency_plan": {"step": "floor"}})
        assert len(errors) == 1 and NAMED in errors[0] and "cores [1]" in errors[0], errors
        assert "until" in errors[0] and "per_synapse" in errors[0], errors

    def test_it_reads_only_a_plan_the_run_reads_on_a_per_axon_chip(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors

        axon = PER_AXON_CHIPS["one_per_axon_type"]
        assert platform_residency_errors({**SCHEDULED, "cores": axon}) == []
        assert platform_residency_errors({"allow_scheduling": False, "cores": axon,
                                          "residency_plan": {"step": "floor"}}) == []
        for cores in PER_SYNAPSE_CHIPS.values():
            assert platform_residency_errors({**SCHEDULED, "cores": cores, "residency_plan": {"step": "floor"}}) == []
        assert platform_residency_errors({**SCHEDULED, "cores": "not a list", "residency_plan": {"step": "floor"}}) == []


#: [Q-1] Pools beyond the two kinds above: no core type, a value the fold does not know (alone and beside a per-axon
#: type) and a null value, which the fold reads as a declared value it does not know.
FOLD_POOLS = {**PER_AXON_CHIPS, **PER_SYNAPSE_CHIPS, "no_core_types": [],
              "an_unknown_value": [{**STATIC, "weight_sign_granularity": "per_row"}],
              "an_unknown_value_beside_a_per_axon_type": [{**STATIC, **AXON},
                                                          {**STATIC, "weight_sign_granularity": "per_row"}],
              "a_null_value": [{**STATIC, "weight_sign_granularity": None}]}


class TestOneDefinitionOfPacksPerAxon:
    """[Q-1] The door decides with ``pool_sign_granularity`` (one definition of "the chip packs per axon"): it refuses
    exactly where the fold reads per axon, and where the fold refuses a value it refuses with the fold's own words."""

    @staticmethod
    def _rule(cores):
        from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors

        return platform_residency_errors({**SCHEDULED, "cores": cores, "residency_plan": {"step": "floor"}})

    @pytest.mark.parametrize("pool", sorted(FOLD_POOLS))
    def test_the_door_refuses_exactly_where_the_fold_reads_per_axon_or_refuses(self, pool):
        from mimarsinan.mapping.platform.platform_constraints import (
            PER_AXON, SignGranularityError, pool_sign_granularity,
        )

        errors = self._rule(FOLD_POOLS[pool])
        try:
            folded = pool_sign_granularity(FOLD_POOLS[pool])
        except SignGranularityError as refusal:
            assert errors == [str(refusal)], errors
        else:
            assert bool(errors) == (folded == PER_AXON), (pool, errors)

    def test_an_unknown_value_is_refused_by_the_folds_own_name(self):
        from mimarsinan.mapping.platform.platform_constraints import SignGranularityError, pool_sign_granularity

        unknown = FOLD_POOLS["an_unknown_value"]
        with pytest.raises(SignGranularityError) as raised:
            pool_sign_granularity(unknown)
        errors = self._rule(unknown)
        assert errors == [str(raised.value)] and "per_row" in errors[0], errors

    def test_the_per_axon_message_keeps_its_index_list(self):
        errors = self._rule(FOLD_POOLS["a_per_axon_type_beside_a_per_synapse_type"])
        assert len(errors) == 1 and NAMED in errors[0] and "cores [1]" in errors[0], errors

    def test_the_rule_reads_no_core_type_without_a_plan_the_run_reads(self):
        from mimarsinan.config_schema.registry.core_type_doors import platform_residency_errors

        for cores in (FOLD_POOLS["an_unknown_value"], FOLD_POOLS["a_null_value"]):
            assert platform_residency_errors({**SCHEDULED, "cores": cores}) == []
            assert platform_residency_errors({"allow_scheduling": False, "cores": cores,
                                              "residency_plan": {"step": "floor"}}) == []

    def test_at_the_doors_an_unknown_value_is_the_schemas_to_name_once(self, tmp_path):
        """The rule reads only the declarations their schemas admit, so a value outside the schema's options is
        named once, by the schema, at every door (unchanged)."""
        document, wizard, run = _refused_by_name(_planned(FOLD_POOLS["an_unknown_value"], STEPS[0]), tmp_path,
                                                 "is not a valid option")
        assert len(document) == 1 and len(wizard) == 1, (document, wizard)

    def test_at_the_doors_a_null_value_beside_a_plan_is_refused_by_the_folds_name(self, tmp_path):
        """A null value is admitted by the schema; the fold, which the builder runs at mapping after training, refuses
        it, so the residency door does too, before any training spend."""
        _refused_by_name(_planned(FOLD_POOLS["a_null_value"], STEPS[0]), tmp_path, "['None'] is not one of")
