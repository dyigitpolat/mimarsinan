"""config_schema under core_semantics='mvm': derivation, folding, document rules."""

import pytest

from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
)
from mimarsinan.config_schema.validation import validate_deployment_config


def _derive(dp, explicit=None):
    derive_deployment_parameters(dp, explicit)
    return dp


class TestMvmDerivation:
    def test_aq_defaults_off_without_activation_bits(self):
        dp = _derive({"core_semantics": "mvm", "weight_quantization": True})
        assert dp["activation_quantization"] is False
        assert dp["weight_quantization"] is True
        assert dp["pipeline_mode"] == "phased"

    def test_activation_bits_arms_aq(self):
        # The derivation runs on the FLAT merge, so the platform key is visible.
        dp = _derive({
            "core_semantics": "mvm", "weight_quantization": True,
            "activation_bits": 8,
        })
        assert dp["activation_quantization"] is True

    def test_activation_bits_arms_aq_on_float_weights(self):
        dp = _derive({
            "core_semantics": "mvm", "pipeline_mode": "vanilla",
            "activation_bits": 8,
        })
        assert dp["weight_quantization"] is False
        assert dp["activation_quantization"] is True

    def test_explicit_aq_true_fails_loud(self):
        with pytest.raises(ValueError, match="value-domain"):
            _derive({"core_semantics": "mvm", "activation_quantization": True})

    def test_explicit_aq_false_contradicts_activation_bits(self):
        with pytest.raises(ValueError, match="value-domain"):
            _derive({
                "core_semantics": "mvm", "activation_quantization": False,
                "activation_bits": 8,
            })

    def test_explicit_aq_true_agrees_with_activation_bits(self):
        dp = _derive({
            "core_semantics": "mvm", "activation_quantization": True,
            "activation_bits": 8,
        })
        assert dp["activation_quantization"] is True

    def test_vanilla_mvm_is_float(self):
        dp = _derive({"core_semantics": "mvm", "pipeline_mode": "vanilla"})
        assert dp["weight_quantization"] is False
        assert dp["activation_quantization"] is False

    def test_sim_enables_fold_false(self):
        dp = _derive({"core_semantics": "mvm"})
        assert dp["enable_nevresim_simulation"] is False
        assert dp["enable_sanafe_simulation"] is False
        assert dp["enable_loihi_simulation"] is False

    def test_explicit_spiking_backend_enable_fails_loud(self):
        with pytest.raises(ValueError, match="value-domain"):
            _derive({"core_semantics": "mvm", "enable_nevresim_simulation": True})

    def test_wq_knobs_fold_and_spiking_knobs_do_not(self):
        dp = _derive({"core_semantics": "mvm"})
        assert dp["wq_fast_rates"] == [0.5, 1.0]
        assert dp["wq_endpoint_recovery_steps"] == 600
        assert dp["optimization_driver"] == "fast"
        assert "lif_exact_qat" not in dp
        assert "lif_blend_fast" not in dp

    def test_spiking_derivation_is_untouched(self):
        dp = _derive({"spiking_mode": "lif", "weight_quantization": True})
        assert dp["activation_quantization"] is True
        assert dp["lif_exact_qat"] is True


def _mvm_document(dp_extra=None, pc_extra=None):
    dp = {
        "core_semantics": "mvm",
        "model_type": "lenet5",
        "model_config": {},
        **(dp_extra or {}),
    }
    pc = {"weight_bits": 8, **(pc_extra or {})}
    return {
        "data_provider_name": "MNIST_DataProvider",
        "experiment_name": "t",
        "generated_files_path": "./generated",
        "platform_constraints": pc,
        "deployment_parameters": dp,
        "start_step": "",
    }


class TestMvmDocumentRules:
    def test_clean_mvm_document_passes(self):
        assert validate_deployment_config(_mvm_document()) == []

    @pytest.mark.parametrize("key,value", [
        ("spiking_family", "lif"),
        ("spiking_variant", "synchronized"),
        ("firing_mode", "Default"),
        ("spike_generation_mode", "Uniform"),
        ("thresholding_mode", "<="),
        ("encoding_layer_placement", "subsume"),
        ("lif_exact_qat", True),
        ("spike_phase_dither", True),
        ("sync_exact_qat_theta", True),
    ])
    def test_event_domain_keys_are_rejected(self, key, value):
        errors = validate_deployment_config(_mvm_document({key: value}))
        assert any("mvm" in e and key in e for e in errors), errors

    @pytest.mark.parametrize("key,value", [
        ("spiking_mode", "lif"),
        ("ttfs_cycle_schedule", "cascaded"),
    ])
    def test_retired_keys_report_retired_never_silent(self, key, value):
        # under mvm a retired taxonomy key reports through the retired-key
        # rule (with its migration remedy), not the mvm domain rule.
        errors = validate_deployment_config(_mvm_document({key: value}))
        assert any("retired" in e and key in e for e in errors), errors

    @pytest.mark.parametrize("key", ["simulation_steps", "target_tq"])
    def test_temporal_platform_keys_are_rejected(self, key):
        errors = validate_deployment_config(_mvm_document(pc_extra={key: 32}))
        assert any("mvm" in e and key in e for e in errors), errors

    def test_spiking_document_is_untouched(self):
        doc = _mvm_document(
            {"spiking_family": "lif", "spiking_variant": "synchronized"}
        )
        doc["deployment_parameters"]["core_semantics"] = "spiking"
        doc["platform_constraints"]["simulation_steps"] = 32
        doc["platform_constraints"]["target_tq"] = 32
        assert validate_deployment_config(doc) == []

    def test_mvm_activation_bits_is_clean(self):
        errors = validate_deployment_config(
            _mvm_document(pc_extra={"activation_bits": 8})
        )
        assert errors == []

    def test_spiking_rejects_activation_bits(self):
        doc = _mvm_document({"spiking_mode": "lif"},
                            pc_extra={"activation_bits": 8})
        doc["deployment_parameters"]["core_semantics"] = "spiking"
        errors = validate_deployment_config(doc)
        assert any("activation_bits" in e and "target_tq" in e for e in errors), errors
