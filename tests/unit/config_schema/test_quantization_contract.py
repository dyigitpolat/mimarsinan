"""Quantization assembly contract: AQ/WQ are derived pipeline-assembly modes."""

import pytest

from mimarsinan.common.env import UNSAFE_QUANT_OVERRIDES_VAR
from mimarsinan.config_schema.defaults import PIPELINE_MODE_PRESETS, apply_preset
from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
    enforce_quantization_assembly_contract,
)

AQ_ON_MODES = ("lif", "ttfs_quantized", "ttfs_cycle_based")
AQ_OFF_MODES = ("ttfs",)


def _derive(dp):
    derive_deployment_parameters(dp)
    return dp


class TestEnvAccessor:
    def test_exactly_one_semantics(self, monkeypatch):
        from mimarsinan.common.env import unsafe_quant_overrides_enabled

        monkeypatch.delenv(UNSAFE_QUANT_OVERRIDES_VAR, raising=False)
        assert unsafe_quant_overrides_enabled() is False
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "true")
        assert unsafe_quant_overrides_enabled() is False
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "1")
        assert unsafe_quant_overrides_enabled() is True

    def test_var_name(self):
        assert UNSAFE_QUANT_OVERRIDES_VAR == "MIMARSINAN_UNSAFE_QUANT_OVERRIDES"


class TestActivationQuantizationDerivation:
    @pytest.mark.parametrize("mode", AQ_ON_MODES)
    def test_absent_key_derives_on_silently(self, mode):
        dp = _derive({"spiking_mode": mode, "weight_quantization": True})
        assert dp["activation_quantization"] is True

    @pytest.mark.parametrize("mode", AQ_OFF_MODES)
    def test_absent_key_derives_off_silently(self, mode):
        dp = _derive({"spiking_mode": mode, "weight_quantization": True})
        assert dp["activation_quantization"] is False

    @pytest.mark.parametrize("mode", AQ_ON_MODES)
    def test_consistent_explicit_value_passes(self, mode):
        dp = _derive({
            "spiking_mode": mode,
            "weight_quantization": True,
            "activation_quantization": True,
        })
        assert dp["activation_quantization"] is True

    def test_consistent_explicit_off_passes_for_analytical_ttfs(self):
        dp = _derive({
            "spiking_mode": "ttfs",
            "weight_quantization": True,
            "activation_quantization": False,
        })
        assert dp["activation_quantization"] is False

    def test_sync_schedule_also_derives_on(self):
        dp = _derive({
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "weight_quantization": True,
        })
        assert dp["activation_quantization"] is True


class TestActivationQuantizationContradictions:
    @pytest.mark.parametrize("mode", AQ_ON_MODES)
    def test_explicit_off_on_aq_forcing_mode_raises(self, mode):
        dp = {
            "spiking_mode": mode,
            "weight_quantization": True,
            "activation_quantization": False,
        }
        with pytest.raises(ValueError, match="activation_quantization"):
            derive_deployment_parameters(dp)

    def test_explicit_on_for_analytical_ttfs_raises(self):
        dp = {
            "spiking_mode": "ttfs",
            "weight_quantization": True,
            "activation_quantization": True,
        }
        with pytest.raises(ValueError, match="activation_quantization"):
            derive_deployment_parameters(dp)

    def test_explicit_on_under_float_weights_raises(self):
        dp = {
            "spiking_mode": "lif",
            "weight_quantization": False,
            "activation_quantization": True,
        }
        with pytest.raises(ValueError, match="float"):
            derive_deployment_parameters(dp)

    def test_error_names_the_derivation_rule_and_the_escape(self):
        dp = {
            "spiking_mode": "ttfs",
            "weight_quantization": True,
            "activation_quantization": True,
        }
        with pytest.raises(ValueError) as exc:
            derive_deployment_parameters(dp)
        message = str(exc.value)
        assert "spiking_mode" in message
        assert "deployment_derivation" in message
        assert UNSAFE_QUANT_OVERRIDES_VAR in message

    def test_float_path_explicit_off_is_consistent(self):
        # The legitimate fp pattern (t0_02/t0_08/t0_10): vanilla + wq off + aq off.
        dp = _derive({
            "spiking_mode": "ttfs",
            "pipeline_mode": "vanilla",
            "weight_quantization": False,
            "activation_quantization": False,
        })
        assert dp["pipeline_mode"] == "vanilla"
        assert dp["activation_quantization"] is False


class TestUnsafeOverride:
    def test_override_honors_explicit_value(self, monkeypatch, capsys):
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "1")
        dp = _derive({
            "spiking_mode": "lif",
            "weight_quantization": True,
            "activation_quantization": False,
        })
        assert dp["activation_quantization"] is False
        assert "[UNSAFE-OVERRIDE]" in capsys.readouterr().out

    def test_override_honors_explicit_on_for_analytical_ttfs(self, monkeypatch, capsys):
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "1")
        dp = _derive({
            "spiking_mode": "ttfs",
            "weight_quantization": True,
            "activation_quantization": True,
        })
        assert dp["activation_quantization"] is True
        assert "[UNSAFE-OVERRIDE]" in capsys.readouterr().out

    def test_non_1_value_does_not_unlock(self, monkeypatch):
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "yes")
        dp = {
            "spiking_mode": "lif",
            "weight_quantization": True,
            "activation_quantization": False,
        }
        with pytest.raises(ValueError):
            derive_deployment_parameters(dp)

    def test_consistent_config_stays_silent_under_override(self, monkeypatch, capsys):
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "1")
        _derive({"spiking_mode": "lif", "weight_quantization": True})
        assert "[UNSAFE-OVERRIDE]" not in capsys.readouterr().out


class TestDerivationIdempotency:
    """Derivation output re-derives cleanly: derived values never contradict."""

    @pytest.mark.parametrize("mode", AQ_ON_MODES + AQ_OFF_MODES)
    def test_rederive_quantized(self, mode):
        dp = _derive({"spiking_mode": mode, "weight_quantization": True})
        again = dict(dp)
        derive_deployment_parameters(again)
        assert again["activation_quantization"] == dp["activation_quantization"]
        assert again["weight_quantization"] == dp["weight_quantization"]

    def test_rederive_float(self):
        dp = _derive({"spiking_mode": "lif", "weight_quantization": False})
        again = dict(dp)
        derive_deployment_parameters(again)
        assert again["pipeline_mode"] == "vanilla"
        assert again["activation_quantization"] is False


class TestPresetNoLongerInjectsQuantFlags:
    """The phased preset must not fabricate explicit-looking AQ/WQ values."""

    def test_phased_preset_is_empty_of_quant_flags(self):
        assert "activation_quantization" not in PIPELINE_MODE_PRESETS["phased"]
        assert "weight_quantization" not in PIPELINE_MODE_PRESETS["phased"]

    def test_apply_preset_leaves_quant_keys_absent(self):
        params = {}
        apply_preset("phased", params)
        assert "activation_quantization" not in params
        assert "weight_quantization" not in params

    def test_phased_analytical_ttfs_without_explicit_aq_derives_off(self):
        from mimarsinan.config_schema import build_flat_pipeline_config

        flat = build_flat_pipeline_config(
            {"spiking_mode": "ttfs", "weight_quantization": True},
            {},
            pipeline_mode="phased",
        )
        assert flat["activation_quantization"] is False
        assert flat["weight_quantization"] is True
        assert flat["pipeline_mode"] == "phased"


class TestWeightQuantizationAssemblyContract:
    def test_wq_false_with_bits_and_no_vanilla_claim_raises(self):
        with pytest.raises(ValueError, match="weight_bits"):
            enforce_quantization_assembly_contract(
                {"weight_quantization": False},
                {"weight_bits": 5},
                pipeline_mode=None,
            )

    def test_wq_false_with_bits_under_phased_raises(self):
        # The t0_04/t0_07 fictional class.
        with pytest.raises(ValueError, match="weight_bits"):
            enforce_quantization_assembly_contract(
                {"weight_quantization": False, "activation_quantization": True},
                {"weight_bits": 5},
                pipeline_mode="phased",
            )

    def test_wq_false_with_bits_under_explicit_vanilla_passes(self):
        # The legitimate fp pattern (t0_02/t0_08/t0_10): vanilla arbitrates.
        enforce_quantization_assembly_contract(
            {"weight_quantization": False},
            {"weight_bits": 5},
            pipeline_mode="vanilla",
        )

    def test_wq_false_without_bits_passes(self):
        enforce_quantization_assembly_contract(
            {"weight_quantization": False},
            {},
            pipeline_mode=None,
        )

    def test_explicit_vanilla_with_explicit_wq_true_raises(self):
        with pytest.raises(ValueError, match="vanilla"):
            enforce_quantization_assembly_contract(
                {"weight_quantization": True},
                {"weight_bits": 5},
                pipeline_mode="vanilla",
            )

    def test_quantized_configs_pass(self):
        enforce_quantization_assembly_contract(
            {"weight_quantization": True},
            {"weight_bits": 5},
            pipeline_mode="phased",
        )
        enforce_quantization_assembly_contract(
            {},
            {"weight_bits": 8},
            pipeline_mode=None,
        )

    def test_error_names_rule_and_escape(self):
        with pytest.raises(ValueError) as exc:
            enforce_quantization_assembly_contract(
                {"weight_quantization": False},
                {"weight_bits": 5},
                pipeline_mode="phased",
            )
        message = str(exc.value)
        assert "bits-driven" in message
        assert "vanilla" in message
        assert UNSAFE_QUANT_OVERRIDES_VAR in message

    def test_unsafe_override_allows_legacy_float_collapse(self, monkeypatch, capsys):
        monkeypatch.setenv(UNSAFE_QUANT_OVERRIDES_VAR, "1")
        enforce_quantization_assembly_contract(
            {"weight_quantization": False},
            {"weight_bits": 5},
            pipeline_mode="phased",
        )
        assert "[UNSAFE-OVERRIDE]" in capsys.readouterr().out


class TestSessionEnforcesTheContract:
    def _config(self, tmp_path, *, pipeline_mode, dp_extra, pc_extra):
        dp = {"spiking_mode": "lif", "model_type": "simple_mlp", **dp_extra}
        cfg = {
            "seed": 0,
            "experiment_name": "quant_contract_test",
            "generated_files_path": str(tmp_path),
            "data_provider_name": "MNIST_DataProvider",
            "platform_constraints": {"weight_bits": 5, **pc_extra},
            "deployment_parameters": dp,
        }
        if pipeline_mode is not None:
            cfg["pipeline_mode"] = pipeline_mode
        return cfg

    def test_fictional_phased_float_config_raises_at_parse(self, tmp_path):
        from mimarsinan.pipelining.session import parse_deployment_config

        cfg = self._config(
            tmp_path,
            pipeline_mode="phased",
            dp_extra={"weight_quantization": False, "activation_quantization": True},
            pc_extra={},
        )
        with pytest.raises(ValueError, match="bits-driven"):
            parse_deployment_config(cfg)

    def test_legitimate_fp_config_parses(self, tmp_path):
        from mimarsinan.pipelining.session import parse_deployment_config

        cfg = self._config(
            tmp_path,
            pipeline_mode="vanilla",
            dp_extra={
                "weight_quantization": False,
                "activation_quantization": False,
            },
            pc_extra={},
        )
        parsed = parse_deployment_config(cfg)
        assert parsed.pipeline_mode == "vanilla"

    def test_quantized_config_parses(self, tmp_path):
        from mimarsinan.pipelining.session import parse_deployment_config

        cfg = self._config(
            tmp_path,
            pipeline_mode="phased",
            dp_extra={"weight_quantization": True},
            pc_extra={},
        )
        parsed = parse_deployment_config(cfg)
        assert parsed.pipeline_mode == "phased"
