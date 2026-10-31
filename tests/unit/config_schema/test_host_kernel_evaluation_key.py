"""[U-PS P-1] The one declared key of the per-sample host-kernel evaluation: ``host_kernel_evaluation``.

A registry entry of decision class function_bearing (per_sample changes the host values to the ulp, and so the
deployed outputs), with no schema default and no derived default: absent, a run resolves nothing for it and stamps
nothing on its program, so every existing program and row stays byte for byte the base's. Its vocabulary is the
consumer's (``mapping/support/host_kernel_evaluation.py``), and a value outside it is refused at the door by name."""

from __future__ import annotations

from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET, get_default_deployment_parameters
from mimarsinan.config_schema.registry import REGISTRY, effective_value
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING
from mimarsinan.config_schema.validation import validate_deployment_config
from mimarsinan.mapping.support.host_kernel_evaluation import (
    BATCHED,
    HOST_KERNEL_EVALUATION_KEY,
    HOST_KERNEL_EVALUATIONS,
    PER_SAMPLE,
)

KEY = HOST_KERNEL_EVALUATION_KEY


def _document(**deployment):
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "ups_key",
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


def test_the_key_is_one_registry_entry_of_the_consumer_s_vocabulary():
    entry = REGISTRY[KEY]
    assert KEY in CONFIG_KEYS_SET
    assert entry.section == "deployment_parameters"
    assert tuple(entry.options) == HOST_KERNEL_EVALUATIONS == (BATCHED, PER_SAMPLE)
    assert entry.owner == "mapping/support/host_kernel_evaluation"
    assert entry.declarable and entry.exposure == "user"


def test_the_key_is_function_bearing():
    entry = REGISTRY[KEY]
    assert entry.decision_class == FUNCTION_BEARING
    assert "deployed outputs" in (entry.decision_reason or "")


def test_absent_is_batched_with_no_default_so_no_resolution_moves():
    entry = REGISTRY[KEY]
    assert not entry.has_default() and entry.derived_default is None
    assert KEY not in get_default_deployment_parameters()
    assert effective_value({}, KEY) is None
    assert (entry.empty_means or "").startswith(BATCHED)


def test_both_values_are_admitted_at_the_door():
    for value in HOST_KERNEL_EVALUATIONS:
        errors = validate_deployment_config(_document(**{KEY: value}), builder_registration=False)
        assert not [e for e in errors if KEY in e], errors


def test_a_value_outside_the_vocabulary_is_refused_by_name_at_the_door():
    errors = validate_deployment_config(_document(**{KEY: "per_batch"}), builder_registration=False)
    assert [e for e in errors if KEY in e], errors
