"""Wave 2 WI-5a to WI-5d declarations: four registry keys (one a readout_law field), their schemas, decision
classes, absent meanings, both doors and the value-domain tag."""

from __future__ import annotations

import copy
import json
import os

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.build import passthrough_keys
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.entries_readout import READOUT_LAW_SCHEMA
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError, schema_of, value_errors
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.mapping.support.grid_declarations import (
    BIAS_AT_LAW,
    BIAS_RESOLUTIONS,
    OUTPUT_LAW_IDENTITY,
    OUTPUT_LAWS,
    WEIGHT_SCALE_FRACTIONAL,
    WEIGHT_SCALE_RULES,
    ParameterGrids,
    identity_output_law,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

NEW_KEYS = ("quantizer_state", "weight_scale_rule", "bias_resolution")
_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "deployment_configs")
STATE = {"file": "/tmp/state.json", "sha256": "ab" * 32}


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


class TestEntries:
    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_a_function_bearing_value_domain_platform_passthrough_with_no_default(self, key):
        entry = REGISTRY[key]
        assert entry.section == "platform_constraints" and entry.group == "hardware"
        assert entry.domain == "value" and entry.passthrough and not entry.has_default()
        assert entry.decision_class == FUNCTION_BEARING
        assert entry.empty_means
        assert key in passthrough_keys()

    def test_the_option_sets_are_the_consumers_vocabularies(self):
        assert tuple(REGISTRY["weight_scale_rule"].resolved_options()) == WEIGHT_SCALE_RULES == (
            "integer", "fractional")
        assert tuple(REGISTRY["bias_resolution"].resolved_options()) == BIAS_RESOLUTIONS == ("weight_grid", "law")

    def test_the_output_law_is_a_function_bearing_readout_law_field(self):
        field = READOUT_LAW_SCHEMA.fields["output_law"]
        assert tuple(field.options) == OUTPUT_LAWS == ("requantize", "identity")
        assert field.decision_class == FUNCTION_BEARING and "requantize" in field.absent_means
        assert "output_law" not in READOUT_LAW_SCHEMA.required

    def test_the_quantizer_state_schema_needs_a_path_and_a_digest(self):
        schema = schema_of(REGISTRY["quantizer_state"])
        assert value_errors(schema, STATE, "quantizer_state") == []
        assert any("sha256" in m for m in value_errors(schema, {"file": "/a.json"}, "quantizer_state"))
        assert any("file" in m for m in value_errors(schema, {"sha256": "ab" * 32}, "quantizer_state"))
        assert any("extra" in m for m in value_errors(schema, {**STATE, "extra": 1}, "quantizer_state"))


class TestReaders:
    def test_parameter_grids_read_the_two_keys(self):
        assert ParameterGrids.from_config({}) == ParameterGrids(False, False)
        grids = ParameterGrids.from_config({"weight_scale_rule": WEIGHT_SCALE_FRACTIONAL,
                                            "bias_resolution": BIAS_AT_LAW})
        assert grids == ParameterGrids(fractional_scale=True, law_bias=True) and grids.declared
        assert not ParameterGrids.from_config({"weight_scale_rule": "integer",
                                               "bias_resolution": "weight_grid"}).declared

    def test_identity_output_law(self):
        assert not identity_output_law(None)
        assert not identity_output_law({"adc_bits": 8, "range": "calibrated", "mult_bits": 16})
        assert identity_output_law({"output_law": OUTPUT_LAW_IDENTITY})


class TestDoors:
    def _value_document(self, **pc):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"].update(pc)
        return config

    @pytest.mark.parametrize("key,value", [
        ("weight_scale_rule", "fractional"), ("bias_resolution", "law"), ("quantizer_state", STATE),
    ])
    def test_a_legal_value_passes_both_doors_and_rides_the_resolved_surface(self, key, value):
        config = self._value_document(**{key: value})
        assert validate_deployment_config(config) == [] or all(key not in m for m in
                                                               validate_deployment_config(config))
        flat = merge_pipeline_config(dict(config["deployment_parameters"]), config["platform_constraints"])
        assert build_platform_constraints_resolved(flat)[key] == value

    @pytest.mark.parametrize("key,value", [
        ("weight_scale_rule", "half"), ("bias_resolution", "float"), ("quantizer_state", {"file": ""}),
    ])
    def test_an_illegal_value_refuses_by_name_at_both_doors(self, key, value):
        config = self._value_document(**{key: value})
        assert any(key in m for m in validate_deployment_config(config))
        with pytest.raises(DeclarationValueError, match=key):
            flat = merge_pipeline_config(dict(config["deployment_parameters"]), config["platform_constraints"])
            build_platform_constraints_resolved(flat)

    def test_an_illegal_output_law_refuses_by_name(self):
        law = {"adc_bits": 8, "range": "calibrated", "mult_bits": 16, "output_law": "float"}
        config = self._value_document(readout_law=law)
        assert any("readout_law.output_law" in m for m in validate_deployment_config(config))

    @pytest.mark.parametrize("key,value", [
        ("weight_scale_rule", "fractional"), ("bias_resolution", "law"), ("quantizer_state", STATE),
    ])
    def test_a_spiking_document_declaring_a_value_grid_key_is_refused(self, key, value):
        config = _load("lif_deepmlp.json")
        config["platform_constraints"][key] = copy.deepcopy(value)
        assert any(m.startswith(key) for m in validate_deployment_config(config))
