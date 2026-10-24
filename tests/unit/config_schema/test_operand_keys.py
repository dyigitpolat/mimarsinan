"""N2 design §2.9: the operand core-type field and ``readout_ranges``, with their decision classes, absent
meanings and refusals; the theorem box's cost-only list equals the registry's (``write_source`` joins it)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.certification.randomized.cost_draws import PRICING_ONLY, cost_only_set, partition
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY, FUNCTION_BEARING
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError, schema_of, value_errors
from mimarsinan.mapping.support.operand_lowering import (
    PER_CONSUMER_GROUP,
    PER_PERCEPTRON,
    READOUT_RANGE_RULES,
    WRITE_SOURCES,
    OperandDeclarationError,
    operand_declaration,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

TABLE = Path(__file__).resolve().parents[2] / "fixtures" / "theorem_box" / "hypothesis_table.json"


def test_the_operand_field_is_function_bearing_with_its_two_nested_fields():
    operand = CORE_TYPE_SCHEMA.fields["operand"]
    assert operand.decision_class == FUNCTION_BEARING and "host" in operand.absent_means.lower()
    assert operand.fields["payload_bits"].decision_class == FUNCTION_BEARING
    assert operand.fields["payload_bits"].low == 2
    assert "payload_bits" in operand.required
    source = operand.fields["write_source"]
    assert source.decision_class == COST_ONLY and tuple(source.options) == WRITE_SOURCES
    assert source.absent_means == "host"


def test_readout_ranges_is_a_function_bearing_platform_key_absent_per_perceptron():
    entry = REGISTRY["readout_ranges"]
    assert entry.decision_class == FUNCTION_BEARING and entry.passthrough
    assert tuple(entry.resolved_options()) == READOUT_RANGE_RULES == (PER_PERCEPTRON, PER_CONSUMER_GROUP)
    assert "per_perceptron" in entry.empty_means


@pytest.mark.parametrize("operand,named", [
    ({"payload_bits": 1}, "cores[0].operand.payload_bits"),
    ({"payload_bits": 8, "write_source": "dma"}, "cores[0].operand.write_source"),
    ({"write_source": "host"}, "cores[0].operand.payload_bits"),
])
def test_an_illegal_operand_declaration_refuses_by_name(operand, named):
    cores = [{"max_axons": 64, "max_neurons": 64, "count": 2, "operand": operand}]
    errors = value_errors(schema_of(REGISTRY["cores"]), cores, "cores")
    assert any(m.startswith(named) for m in errors), errors
    with pytest.raises(DeclarationValueError, match=named.replace("[", r"\[").replace("]", r"\]")):
        build_platform_constraints_resolved(merge_pipeline_config({}, {"cores": cores}))


def test_an_illegal_readout_range_rule_refuses_by_name():
    errors = value_errors(schema_of(REGISTRY["readout_ranges"]), "per_channel", "readout_ranges")
    assert any(m.startswith("readout_ranges") for m in errors)


def test_the_resolved_platform_carries_both_declarations():
    cores = [{"max_axons": 64, "max_neurons": 64, "count": 2},
             {"max_axons": 64, "max_neurons": 64, "count": 1, "operand": {"payload_bits": 4}}]
    resolved = build_platform_constraints_resolved(merge_pipeline_config(
        {"core_semantics": "mvm"}, {"cores": cores, "readout_ranges": PER_CONSUMER_GROUP}))
    assert resolved["readout_ranges"] == PER_CONSUMER_GROUP
    assert resolved["cores"][1]["operand"] == {"payload_bits": 4}
    declaration = operand_declaration(resolved)
    assert declaration is not None and declaration.payload_bits == 4 and declaration.write_source == "host"


def test_no_operand_core_type_declares_no_lowering_and_two_widths_refuse():
    assert operand_declaration({"cores": [{"max_axons": 8, "max_neurons": 8}]}) is None
    two = {"cores": [{"max_axons": 8, "max_neurons": 8, "operand": {"payload_bits": 4}},
                     {"max_axons": 8, "max_neurons": 8, "operand": {"payload_bits": 8}}]}
    with pytest.raises(OperandDeclarationError, match="payload_bits"):
        operand_declaration(two)


def test_the_theorem_box_states_the_conditions_of_the_operand_cost_only_classes():
    table = json.loads(TABLE.read_text())
    finding = next(f for f in table["findings_on_classes"] if f["id"] == "operand_cost_only_conditions")
    assert "per_consumer_group" in finding["statement"] and "on_chip" in finding["statement"]
    assert "host-fed" in finding["statement"]


def test_the_stationary_choice_is_cost_only_exactly_at_equal_bits_with_consumer_groups():
    """The declaration's half of the condition; the product's half (both operands core-fed) is
    ``test_operand_lowering.test_a_host_fed_operand_keeps_the_declared_default_whatever_the_price``."""
    def declared(payload_bits, rule):
        return operand_declaration({"cores": [{"max_axons": 8, "max_neurons": 8,
                                               "operand": {"payload_bits": payload_bits}}],
                                    "activation_bits": 8, "readout_ranges": rule,
                                    "readout_law": {"adc_bits": 32, "range": "calibrated", "mult_bits": 16}})
    assert declared(8, PER_CONSUMER_GROUP).choice_is_cost_only
    assert not declared(4, PER_CONSUMER_GROUP).choice_is_cost_only
    assert not declared(8, PER_PERCEPTRON).choice_is_cost_only


def test_the_theorem_box_cost_only_list_equals_the_registrys():
    table = json.loads(TABLE.read_text())
    assert set(table["cost_only"]) == cost_only_set()
    assert "cores.operand.write_source" in cost_only_set()
    assert "cores.operand.write_source" in PRICING_ONLY
    assert "cores.operand" not in cost_only_set() and "cores.operand.payload_bits" not in cost_only_set()
    partition()
