"""The seam of N2's operand core types and the closure's per-axon weight sign (CL-SG) on one chip.

The per-axon split fixes each physical row's sign once, from the static weights, after compaction and before packing
(``softcore/sign_split.py``); an operand bank's payload is written per inference (N2), so its signs move with every
sample, and its matrix is a NaN placeholder until the executor writes it. The two are not composed on this tree. A
chip that declares per-axon weight sign beside an operand core type is refused by name, with that reason, at every
door (before any training spend), at Operand Lowering, and at the sign split of a direct build. Before this seam was
named, the split refused such a build as "a weight off the integer grid", the reason for unquantized weights."""

from __future__ import annotations

import copy

import pytest

from mimarsinan.config_schema.registry.entries_write import platform_sign_errors
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing.softcore.sign_split import SignSplitError
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.operand_lowering import OperandDeclarationError, operand_declaration
from unit.chip_simulation.value_run.operand_programs import attention_program
from unit.config_schema.test_target_doors import PC, VALUE, _load, _refused_by_name, _verdicts

STATIC = {"max_axons": 256, "max_neurons": 256, "count": 6}
OPERAND = {"max_axons": 240, "max_neurons": 240, "count": 4, "operand": {"payload_bits": 8}}
AXON = {"weight_sign_granularity": "per_axon"}
SYNAPSE = {"weight_sign_granularity": "per_synapse"}
#: The three ways per-axon sign meets an operand core type: beside it, on it, and one shared type declaring both.
COMPOSED = {
    "per_axon_static_beside_an_operand_type": [{**STATIC, **AXON}, OPERAND],
    "per_axon_operand_type": [STATIC, {**OPERAND, **AXON}],
    "one_shared_type_declaring_both": [{**STATIC, **AXON, "count": 12, "operand": {"payload_bits": 8}}],
}
NAMED = "per-axon weight sign and an operand core type"


def _platform(cores):
    return {"cores": copy.deepcopy(cores), "activation_bits": 8, "readout_ranges": "per_consumer_group"}


def _document(cores):
    doc = _load(VALUE)
    doc[PC] = {**{k: v for k, v in doc[PC].items() if k not in ("cores", "max_axons", "max_neurons")},
               "cores": copy.deepcopy(cores)}
    return doc


@pytest.mark.parametrize("name", sorted(COMPOSED))
def test_operand_lowering_refuses_per_axon_sign_beside_an_operand_core_type_by_name(name):
    with pytest.raises(OperandDeclarationError, match=NAMED):
        operand_declaration(_platform(COMPOSED[name]))


def test_an_operand_chip_declaring_per_synapse_sign_reads_as_before():
    declaration = operand_declaration(_platform([{**STATIC, **SYNAPSE}, {**OPERAND, **SYNAPSE}]))
    assert declaration is not None and declaration.payload_bits == 8


@pytest.mark.parametrize("name", sorted(COMPOSED))
def test_the_door_rule_names_the_composition_once(name):
    errors = platform_sign_errors({"cores": copy.deepcopy(COMPOSED[name])})
    assert len(errors) == 1 and NAMED in errors[0], errors


def test_the_door_rule_keeps_its_float_weight_refusal_and_admits_either_declaration_alone():
    assert platform_sign_errors({"cores": [{**STATIC, **AXON}]}) == []
    assert platform_sign_errors({"cores": [STATIC, OPERAND]}) == []
    assert platform_sign_errors({"cores": [{**STATIC, **SYNAPSE}, {**OPERAND, **SYNAPSE}]}) == []
    floats = platform_sign_errors({"cores": [{**STATIC, **AXON}], "weight_quantization": False})
    assert len(floats) == 1 and "needs weights on the integer grid" in floats[0]


@pytest.mark.parametrize("name", sorted(COMPOSED))
def test_every_door_refuses_the_composition_by_name(tmp_path, name):
    _refused_by_name(_document(COMPOSED[name]), tmp_path, NAMED)


def test_an_operand_core_type_on_a_per_synapse_chip_passes_every_door(tmp_path):
    assert _verdicts(_document([{**STATIC, **SYNAPSE}, OPERAND]), tmp_path) == ([], [], None)


@pytest.fixture(scope="module")
def program():
    return attention_program("softmax")


@pytest.mark.parametrize("name", sorted(COMPOSED))
def test_a_direct_build_names_the_dynamic_operand_bank_at_the_sign_split(program, name):
    """A build that bypasses Operand Lowering's declaration (the operand program lowered for a per-synapse chip,
    packed on a per-axon pool) stops at the split with the composition's reason, not the unquantized-weight one."""
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=True, max_schedule_passes=128))
    with pytest.raises(SignSplitError) as raised:
        build_hybrid_hard_core_mapping(ir_graph=program.ir, cores_config=copy.deepcopy(COMPOSED[name]),
                                       strategy=strategy, max_schedule_passes=128)
    message = str(raised.value)
    assert "dynamic operand bank" in message and "per inference" in message, message
    assert "quantize the weights" not in message, message
