"""[U-ADC A-1] The one declared key of readout-aware training: ``readout_aware_training``.

A value-domain registry entry of decision class function_bearing (true trains Weight Quantization's endpoint recovery
through the deployed readout law, so the deployed weights change), with no schema default and no derived default:
absent, a run resolves nothing for it, so every existing program and row stays byte for byte the base's. Both booleans
are admitted at the door on a value-domain (mvm) document, any other value is refused there by name, and a spiking
document declaring it is refused by the key's value-domain tag."""

from __future__ import annotations

import json
import pathlib

from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET, get_default_deployment_parameters
from mimarsinan.config_schema.registry import REGISTRY, effective_value, parse_deployment_document
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import FieldType
from mimarsinan.config_schema.resolve import resolve_draft
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.gui.wizard.emit import emit_deployment_config


KEY = "readout_aware_training"
ROOT = pathlib.Path(__file__).resolve().parents[3]
MVM_FIXTURE = ROOT / "tests" / "fixtures" / "integer_value" / "mvm_int_readout_lenet5.json"


def _mvm_document(**deployment):
    document = json.loads(MVM_FIXTURE.read_text())
    document["deployment_parameters"].update(deployment)
    return document


def _spiking_document(**deployment):
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "uadc_key",
        "generated_files_path": "./generated",
        "seed": 0,
        "pipeline_mode": "phased",
        "platform_constraints": {
            "cores": [{"max_axons": 784, "max_neurons": 256, "count": 8, "has_bias": True}],
            "max_axons": 784, "max_neurons": 256, "has_bias": True,
            "target_tq": 4, "simulation_steps": 4, "weight_bits": 5,
        },
        "deployment_parameters": {
            "model_type": "simple_mlp", "model_config_mode": "user", "hw_config_mode": "fixed",
            "spiking_family": "lif", "spiking_variant": "streamed", "weight_quantization": True,
            **deployment,
        },
    }


def test_the_key_is_one_value_domain_registry_entry():
    entry = REGISTRY[KEY]
    assert KEY in CONFIG_KEYS_SET
    assert entry.section == "deployment_parameters" and entry.domain == "value"
    assert entry.type is FieldType.BOOL
    assert entry.owner == "pipelining/pipeline_steps/quantization/readout_aware_training"
    assert entry.declarable and entry.exposure == "user"


def test_the_key_is_function_bearing():
    entry = REGISTRY[KEY]
    assert entry.decision_class == FUNCTION_BEARING
    assert "deployed weights" in (entry.decision_reason or "")


def test_absent_is_false_with_no_default_so_no_resolution_moves():
    entry = REGISTRY[KEY]
    assert not entry.has_default() and entry.derived_default is None
    assert KEY not in get_default_deployment_parameters()
    assert effective_value({}, KEY) is None
    assert (entry.empty_means or "").startswith("false")


def test_both_booleans_are_admitted_at_the_door_on_a_value_domain_document():
    for value in (True, False):
        document = _mvm_document(**{KEY: value})
        assert parse_deployment_document(document).unknown == []   # a known key, not an ignored one
        errors = validate_deployment_config(document, builder_registration=False)
        assert not [e for e in errors if KEY in e], errors


def test_a_value_other_than_a_boolean_is_refused_by_name_at_the_door():
    errors = validate_deployment_config(_mvm_document(**{KEY: "yes"}), builder_registration=False)
    assert [e for e in errors if KEY in e], errors


def test_a_spiking_document_declaring_it_is_refused_by_its_value_domain_tag():
    errors = validate_deployment_config(_spiking_document(**{KEY: True}), builder_registration=False)
    assert [e for e in errors if e.startswith(KEY) and "value-domain" in e], errors


def test_a_declaring_document_is_representable_in_the_wizard():
    """The representability test's four readings on a document that declares the key."""
    document = _mvm_document(**{KEY: True})
    assert parse_deployment_document(document).unknown == []
    assert json.dumps(emit_deployment_config(document), sort_keys=True) == json.dumps(document, sort_keys=True)
    resolution = resolve_draft(document)
    assert resolution.errors == [] and resolution.unknown_keys == []
    assert validate_deployment_config(document) == []
