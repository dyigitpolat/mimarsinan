import pytest

from mimarsinan.config_schema import build_flat_pipeline_config
from mimarsinan.config_schema.defaults import get_default_deployment_parameters
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.gui.wizard.config_builder import build_deployment_config_from_state


def test_lif_enables_activation_quantization_preconditioning():
    dp = {"spiking_mode": "lif", "weight_quantization": True, "activation_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is True


def test_ttfs_quantized_enables_activation_quant():
    dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is True


def test_ttfs_cycle_based_finetune_enables_activation_quant():
    # Cycle-accurate TTFS is preconditioned by the activation-quantization chain
    # before TTFS Cycle Fine-Tuning.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_quantization"] is True


def test_ttfs_cycle_based_synchronized_has_no_nevresim():
    # No genuine synchronized-window nevresim backend yet → capability off.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized"}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is False


def test_explicit_enable_of_an_unsupported_backend_is_rejected():
    # An explicit ON for a backend that cannot run the mode is a contract
    # violation (never silently overwritten — that would be a lie).
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized", "enable_nevresim_simulation": True}
    with pytest.raises(ValueError, match="enable_nevresim_simulation"):
        derive_deployment_parameters(dp)


def test_ttfs_cycle_based_cascaded_keeps_nevresim():
    # Cascaded greedy TTFS runs genuinely on nevresim (fire-once-latch policy).
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "cascaded", "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is True


def test_ttfs_cycle_based_default_schedule_keeps_nevresim():
    # Default schedule is cascaded → nevresim stays enabled.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is True


def test_float_weights_vanilla():
    dp = {"weight_quantization": False, "spiking_mode": "ttfs_quantized"}
    derive_deployment_parameters(dp)
    assert dp["pipeline_mode"] == "vanilla"
    assert dp["weight_quantization"] is False
    assert dp["activation_quantization"] is False


def test_config_builder_pipeline_mode_sync():
    cfg = build_deployment_config_from_state({
        "pipeline_mode": "phased",
        "deployment_parameters": {"spiking_mode": "lif", "weight_quantization": True},
    })
    # A declared pipeline_mode is a declarable-derived key: preserved verbatim
    # (dropping it historically made vanilla configs silently load as phased).
    assert cfg["pipeline_mode"] == "phased"
    resolved = build_flat_pipeline_config(
        cfg["deployment_parameters"],
        cfg["platform_constraints"],
        pipeline_mode="phased",
    )
    assert resolved["pipeline_mode"] == "phased"


def test_config_builder_lif_derives_quant_flags():
    cfg = build_deployment_config_from_state({
        "deployment_parameters": {
            "spiking_mode": "lif",
            "activation_quantization": False,
            "weight_quantization": True,
        },
    })
    assert "activation_quantization" not in cfg["deployment_parameters"]
    resolved = build_flat_pipeline_config(
        cfg["deployment_parameters"],
        cfg["platform_constraints"],
        pipeline_mode="phased",
    )
    assert resolved["activation_quantization"] is True


def test_cycle_accurate_lif_forward_always_resolves_true():
    # Recipe-owned (never a knob, no schema default): the derivation writes it
    # for every mode — ON for lif by the recipe, inert-but-resolved for TTFS.
    for mode in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"):
        resolved = build_flat_pipeline_config(
            {"spiking_mode": mode, "weight_quantization": True},
            {},
            pipeline_mode="phased",
        )
        assert resolved["cycle_accurate_lif_forward"] is True, mode
    dp = get_default_deployment_parameters()
    assert "cycle_accurate_lif_forward" not in dp


def test_legacy_ramp_switches_removed():
    """The legacy per-frame ramp and the rejected genuine-gradual ramp switches
    were removed; the value-domain blend ramp is the sole, non-optional path."""
    dp = get_default_deployment_parameters()
    assert "legacy_lif_blend_ramp" not in dp
    assert "genuine_gradual_cascade_ramp" not in dp


def test_simulator_enables_are_recipe_derived_not_defaults():
    """The ConversionPolicy recipe owns the sim enables per mode; the defaults
    dict carries none, and the derivation always writes them."""
    dp = get_default_deployment_parameters()
    assert "enable_nevresim_simulation" not in dp
    derived = {"spiking_mode": "lif"}
    derive_deployment_parameters(derived)
    assert derived["enable_nevresim_simulation"] is True
    assert derived["enable_loihi_simulation"] is True


def test_config_builder_cycle_accurate_default_for_lif():
    cfg = build_deployment_config_from_state({})
    resolved = build_flat_pipeline_config(
        cfg["deployment_parameters"],
        cfg["platform_constraints"],
        pipeline_mode="phased",
    )
    assert resolved["cycle_accurate_lif_forward"] is True


def test_ttfs_genuine_blend_ce_alpha_default():
    # The genuine-blend CE weight survives the SSOT collapse as a registered knob
    # (the recipe writes a mode value over it; the raw default stays 0.3).
    dp = get_default_deployment_parameters()
    assert dp["ttfs_genuine_blend_ce_alpha"] == 0.3


def test_ttfs_genuine_blend_ce_alpha_in_config_keys_set():
    from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
    assert "ttfs_genuine_blend_ce_alpha" in CONFIG_KEYS_SET


# ── ConversionPolicy SSOT folding (the derivation writes the proven recipe) ─────


def test_lif_folds_the_proven_recipe_and_enables_loihi():
    # A config carrying only the hypervolume coordinate (mode + precision) gets the
    # full proven LIF recipe folded in by the policy SSOT.
    dp = {"spiking_mode": "lif", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["optimization_driver"] == "fast"
    assert dp["lif_blend_fast"] is True
    assert dp["fast_ladder_freeze_bn"] is True
    assert dp["kd_ce_alpha"] == 0.5
    assert dp["kd_temperature"] == 4.0
    assert dp["enable_loihi_simulation"] is True
    assert dp["enable_sanafe_simulation"] is True
    assert dp["enable_nevresim_simulation"] is True


def test_ttfs_folds_fast_driver_endpoint_floor_loihi_off():
    dp = {"spiking_mode": "ttfs", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["optimization_driver"] == "fast"
    assert dp["enable_loihi_simulation"] is False  # loihi caps are LIF-only
    assert "lif_blend_fast" not in dp  # the analytical reference carries no mode knobs
    # [5u] bit-parity-lossless ⇒ the endpoint chases the acceptance target,
    # funded at the NAPQ endpoint by the measured wall headroom.
    assert dp["endpoint_target_floor"] == 0.98
    assert dp["wq_endpoint_recovery_steps"] == 16000


def test_bit_parity_every_endpoint_floor_folds_only_for_the_lossless_mode():
    # The bit-parity every-endpoint floor (endpoint_target_floor, read by every
    # run_endpoint_recovery) rides only analytical ttfs; no other mode gets it.
    for mode, schedule in [
        ("lif", None),
        ("ttfs_quantized", None),
        ("ttfs_cycle_based", "cascaded"),
        ("ttfs_cycle_based", "synchronized"),
    ]:
        dp = {"spiking_mode": mode, "weight_quantization": True}
        if schedule is not None:
            dp["ttfs_cycle_schedule"] = schedule
        derive_deployment_parameters(dp)
        assert "endpoint_target_floor" not in dp, (mode, schedule)


def test_generalized_wq_scoped_floor_folds_for_well_conditioned_modes():
    # [5u generalized] lif/sync/cascaded carry the WQ-scoped floor
    # (wq_endpoint_target_floor + a lifted wq budget); ttfs_quantized stays off it.
    for mode, schedule in [
        ("lif", None),
        ("ttfs_cycle_based", "cascaded"),
        ("ttfs_cycle_based", "synchronized"),
    ]:
        dp = {"spiking_mode": mode, "weight_quantization": True}
        if schedule is not None:
            dp["ttfs_cycle_schedule"] = schedule
        derive_deployment_parameters(dp)
        assert dp["wq_endpoint_target_floor"] == 0.98, (mode, schedule)
        assert dp["wq_endpoint_recovery_steps"] == 16000, (mode, schedule)

    dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert "wq_endpoint_target_floor" not in dp
    assert dp["wq_endpoint_recovery_steps"] == 600


def test_ttfs_quantized_folds_full_quantile_decode():
    dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["activation_scale_quantile"] == 1.0
    assert dp["manager_rate_fast_rates"] == [0.25, 0.5, 0.75, 1.0]
    assert dp["manager_rate_fast_steps_per_rate"] == 120
    assert dp["enable_loihi_simulation"] is False


def test_cascaded_folds_genuine_blend_fast():
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "cascaded"}
    derive_deployment_parameters(dp)
    assert dp["optimization_driver"] == "fast"
    assert dp["ttfs_genuine_blend_ramp"] is True
    assert dp["ttfs_genuine_blend_fast"] is True
    # W3 reinvestment: reclaimed eval wall flows to the binding FT endpoint.
    assert dp["endpoint_recovery_steps"] == 600
    assert dp["tuning_full_transform_probe"] is True
    assert dp["enable_loihi_simulation"] is False


def test_synchronized_folds_exact_endpoint_and_disables_nevresim():
    # synchronized rides the ttfs_quantized ladder shape but trains the exact
    # deployed ceil kernel + grid snap as the QAT endpoint (T6, X3 default).
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized"}
    derive_deployment_parameters(dp)
    assert dp["activation_scale_quantile"] == 1.0
    assert dp["manager_rate_fast_rates"] == [0.25, 0.5, 0.75, 1.0]
    assert dp["manager_rate_fast_steps_per_rate"] == 120
    assert dp["optimization_driver"] == "fast"
    assert dp["sync_exact_qat"] is True
    assert dp["endpoint_recovery_steps"] == 600
    # [5v B1] the sync crater levers ride the recipe; ttfs_quantized does NOT
    # get them (its full-quantile decode is the proven green-family shape).
    assert dp["starvation_aware_scale_quantile"] is True
    assert dp["sync_entry_half_step"] is True
    assert dp["sync_hop_staged_install"] is True
    dp_q = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
    derive_deployment_parameters(dp_q)
    assert "starvation_aware_scale_quantile" not in dp_q
    assert "sync_entry_half_step" not in dp_q
    assert "sync_hop_staged_install" not in dp_q
    # the old genuine-QAT knobs are no longer folded for synchronized.
    assert "ttfs_sync_genuine_qat" not in dp
    assert dp["enable_nevresim_simulation"] is False  # no sync-window backend
    assert dp["enable_loihi_simulation"] is False


def test_explicit_loihi_on_a_non_lif_mode_is_rejected():
    # loihi + a non-LIF mode RAISES at assembly, so the derivation rejects the
    # explicit enable loudly (with the remove-the-key remedy) instead of
    # silently shipping a config that contradicts its declaration.
    dp = {"spiking_mode": "ttfs", "weight_quantization": True,
          "enable_loihi_simulation": True}
    with pytest.raises(ValueError, match="enable_loihi_simulation"):
        derive_deployment_parameters(dp)


class TestVehicleUserOff:
    """Round-3 defect 6: a SUPPORTED vehicle defaults ON per the recipe and a
    declared OFF is honored — a legitimate override, never overwritten."""

    def test_user_off_disables_a_supported_vehicle(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "enable_sanafe_simulation": False}
        derive_deployment_parameters(dp)
        assert dp["enable_sanafe_simulation"] is False

    def test_user_off_nevresim_is_honored_for_lif(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "enable_nevresim_simulation": False}
        derive_deployment_parameters(dp)
        assert dp["enable_nevresim_simulation"] is False

    def test_unset_supported_vehicles_default_on(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True}
        derive_deployment_parameters(dp)
        assert dp["enable_nevresim_simulation"] is True
        assert dp["enable_loihi_simulation"] is True
        assert dp["enable_sanafe_simulation"] is True

    def test_explicit_on_of_a_supported_vehicle_is_accepted(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "enable_loihi_simulation": True}
        derive_deployment_parameters(dp)
        assert dp["enable_loihi_simulation"] is True

    def test_explicit_off_of_an_unsupported_vehicle_is_consistent(self):
        # Declaring OFF where the capability is already off is consistent —
        # accepted, and the resolved value stays False.
        dp = {"spiking_mode": "ttfs", "weight_quantization": True,
              "enable_loihi_simulation": False}
        derive_deployment_parameters(dp)
        assert dp["enable_loihi_simulation"] is False


class TestRecipeKnobExplicitWins:
    """Registry-declarable recipe knobs: the recipe is the MODE-AWARE DEFAULT
    and an explicit document value wins (the registry's documented contract —
    'explicit value wins'). Internal recipe constants stay recipe-owned."""

    def test_explicit_wq_endpoint_recovery_steps_wins_over_the_recipe(self):
        # The FAST-respec per-cell cap (tier configs declare 2000) must reach
        # the runtime — the recipe 16000 is only the unset default.
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "wq_endpoint_recovery_steps": 2000}
        derive_deployment_parameters(dp)
        assert dp["wq_endpoint_recovery_steps"] == 2000

    def test_explicit_kd_ce_alpha_wins_over_the_lif_recipe(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "kd_ce_alpha": 0.1}
        derive_deployment_parameters(dp)
        assert dp["kd_ce_alpha"] == 0.1

    def test_unset_registry_knobs_take_the_recipe_value_over_the_generic_default(self):
        # The merged pipeline path carries the GENERIC defaults (kd_ce_alpha
        # 0.3); without an explicit declaration the recipe's mode value must
        # still win over the merged-in generic default.
        resolved = build_flat_pipeline_config(
            {"spiking_mode": "lif", "weight_quantization": True},
            {},
            pipeline_mode="phased",
        )
        assert resolved["kd_ce_alpha"] == 0.5
        assert resolved["wq_endpoint_recovery_steps"] == 16000

    def test_explicit_knob_wins_through_the_merged_pipeline_path(self):
        resolved = build_flat_pipeline_config(
            {"spiking_mode": "lif", "weight_quantization": True,
             "wq_endpoint_recovery_steps": 2000, "kd_ce_alpha": 0.1},
            {},
            pipeline_mode="phased",
        )
        assert resolved["wq_endpoint_recovery_steps"] == 2000
        assert resolved["kd_ce_alpha"] == 0.1

    def test_internal_recipe_constants_stay_recipe_owned(self):
        # Non-registry recipe internals (lif_blend_fast, optimization_driver)
        # are not user keys; the recipe overwrites any injected value.
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "lif_blend_fast": False, "optimization_driver": "controller"}
        derive_deployment_parameters(dp)
        assert dp["lif_blend_fast"] is True
        assert dp["optimization_driver"] == "fast"

    def test_cycle_accurate_lif_forward_is_recipe_owned(self):
        # Correctness mechanism (train/eval bit-exactness): the LIF recipe
        # always folds it ON; an explicit off is overwritten, never stored.
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "cycle_accurate_lif_forward": False}
        derive_deployment_parameters(dp)
        assert dp["cycle_accurate_lif_forward"] is True


def test_vanilla_float_still_gets_driver_and_sim_enables():
    # The mode-level recipe (driver + capability sim-enables) applies even for the
    # float regime; only the quant flags are forced off by the vanilla branch.
    dp = {"spiking_mode": "lif", "weight_quantization": False}
    derive_deployment_parameters(dp)
    assert dp["pipeline_mode"] == "vanilla"
    assert dp["activation_quantization"] is False
    assert dp["optimization_driver"] == "fast"
    assert dp["enable_loihi_simulation"] is True


class TestTtfsFiringValidation:
    def test_explicit_non_ttfs_firing_mode_raises_for_ttfs_modes(self):
        from mimarsinan.config_schema.deployment_derivation import (
            derive_pipeline_runtime_parameters,
        )

        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            dp = {"spiking_mode": mode, "firing_mode": "Default"}
            with pytest.raises(ValueError, match="firing_mode"):
                derive_pipeline_runtime_parameters(dp)

    def test_explicit_non_ttfs_spike_generation_raises_for_ttfs_modes(self):
        from mimarsinan.config_schema.deployment_derivation import (
            derive_pipeline_runtime_parameters,
        )

        dp = {"spiking_mode": "ttfs_quantized", "spike_generation_mode": "Uniform"}
        with pytest.raises(ValueError, match="spike_generation_mode"):
            derive_pipeline_runtime_parameters(dp)

    def test_lif_keeps_custom_firing_mode(self):
        from mimarsinan.config_schema.deployment_derivation import (
            derive_pipeline_runtime_parameters,
        )

        dp = {"spiking_mode": "lif", "firing_mode": "Novena"}
        derive_pipeline_runtime_parameters(dp)
        assert dp["firing_mode"] == "Novena"
