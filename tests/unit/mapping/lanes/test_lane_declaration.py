"""WI-5e: the ``onchip_arithmetic`` declaration, its registry key, both doors and its refusals by name."""

from __future__ import annotations

import copy
import json
import os

import pytest

from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.build import passthrough_keys
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY, FUNCTION_BEARING
from mimarsinan.config_schema.registry.entries_lanes import ONCHIP_ARITHMETIC_SCHEMA
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError, schema_of, value_errors
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.mapping.lanes.declaration import (
    BUILT_LANE_CLASSES,
    LANE_KEY,
    LaneDeclaration,
    LaneDeclarationError,
    lane_declaration,
    lanes_executed,
    parse_lane_declaration,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "fixtures", "deployment_configs")
LAW = {"adc_bits": 32, "range": "calibrated", "mult_bits": 24}


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


class TestParse:
    def test_absent_is_no_lanes(self):
        assert parse_lane_declaration(None) is None and lane_declaration({}) is None

    def test_the_lane_subset_and_the_lane_block(self):
        declaration = parse_lane_declaration({"classes": ["L", "E", "P"], "lanes_per_core": 4,
                                              "lane_block": "ng45_os_lane"})
        assert declaration == LaneDeclaration(frozenset("LEP"), 4, "ng45_os_lane") and declaration.any
        assert BUILT_LANE_CLASSES == ("L", "E", "P")

    def test_an_empty_class_list_is_the_off_ablation(self):
        declaration = parse_lane_declaration({"classes": []})
        assert declaration is not None and not declaration.any
        assert lanes_executed({LANE_KEY: {"classes": []}, "readout_law": LAW}) is None

    def test_lanes_run_on_the_integer_path_only(self):
        lanes = {"classes": ["L"]}
        assert lanes_executed({LANE_KEY: lanes}) is None, "a float value program keeps its host ops"
        assert lanes_executed({LANE_KEY: lanes, "readout_law": LAW}) == LaneDeclaration(frozenset("L"))

    @pytest.mark.parametrize("value,phrase", [
        ({"classes": ["A"]}, "no lowering here realizes"),
        ({"classes": ["L", "C"]}, "no lowering here realizes"),
        ({"classes": ["Q"]}, "unknown lane classes"),
        ({"classes": ["E"]}, "without L"),
        ({"classes": ["P", "E"]}, "without L"),
        ({"classes": ["L", "L"]}, "twice"),
        ({"classes": "LEP"}, "list of class letters"),
        ({"lanes": ["L"]}, "classes list"),
        ({"classes": ["L"], "lanes_per_core": 0}, "at least 1"),
        ({"classes": ["L"], "lane_block": ""}, "shipped lane block"),
        ({"classes": [], "lane_block": "ng45_os_lane"}, "prices none"),
    ])
    def test_refusals_by_name(self, value, phrase):
        with pytest.raises(LaneDeclarationError, match=phrase):
            parse_lane_declaration(value)


class TestRegistry:
    def test_a_function_bearing_value_domain_platform_passthrough_with_no_default(self):
        entry = REGISTRY[LANE_KEY]
        assert entry.section == "platform_constraints" and entry.domain == "value" and entry.passthrough
        assert not entry.has_default() and entry.empty_means and LANE_KEY in passthrough_keys()
        assert entry.decision_class == FUNCTION_BEARING

    def test_the_fields_and_their_classes(self):
        fields = ONCHIP_ARITHMETIC_SCHEMA.fields
        assert fields["classes"].decision_class == FUNCTION_BEARING
        assert fields["lanes_per_core"].decision_class == COST_ONLY
        assert fields["lane_block"].decision_class == COST_ONLY
        assert tuple(fields["classes"].item.options) == BUILT_LANE_CLASSES

    def test_the_schema_judges_the_value(self):
        schema = schema_of(REGISTRY[LANE_KEY])
        assert value_errors(schema, {"classes": ["L", "E", "P"], "lanes_per_core": 4,
                                     "lane_block": "ng45_os_lane"}, LANE_KEY) == []
        assert any("reserved" in m for m in value_errors(schema, {"classes": ["A"]}, LANE_KEY))
        assert any("classes" in m for m in value_errors(schema, {"lanes_per_core": 4}, LANE_KEY))


class TestDoors:
    def _value_document(self, **pc):
        config = _load("mvm_aq_lenet5.json")
        config["platform_constraints"].update(pc)
        return config

    def test_a_legal_declaration_passes_both_doors_and_rides_the_resolved_surface(self):
        value = {"classes": ["L", "E"], "lanes_per_core": 4}
        config = self._value_document(onchip_arithmetic=value)
        assert all(LANE_KEY not in m for m in validate_deployment_config(config))
        flat = merge_pipeline_config(dict(config["deployment_parameters"]), config["platform_constraints"])
        assert build_platform_constraints_resolved(flat)[LANE_KEY] == value

    def test_an_illegal_declaration_refuses_by_name_at_both_doors(self):
        config = self._value_document(onchip_arithmetic={"classes": ["X"]})
        assert any(LANE_KEY in m for m in validate_deployment_config(config))
        with pytest.raises(DeclarationValueError, match=LANE_KEY):
            flat = merge_pipeline_config(dict(config["deployment_parameters"]), config["platform_constraints"])
            build_platform_constraints_resolved(flat)

    def test_a_spiking_document_declaring_lanes_is_refused(self):
        config = _load("lif_deepmlp.json")
        config["platform_constraints"][LANE_KEY] = copy.deepcopy({"classes": ["L"]})
        assert any(m.startswith(LANE_KEY) for m in validate_deployment_config(config))
