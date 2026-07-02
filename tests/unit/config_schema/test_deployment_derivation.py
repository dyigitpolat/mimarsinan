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


def test_ttfs_cycle_based_synchronized_disables_nevresim():
    # No genuine synchronized-window nevresim backend yet → forced off.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized", "enable_nevresim_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_nevresim_simulation"] is False


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
    assert "pipeline_mode" not in cfg
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


def test_default_cycle_accurate_lif_forward_is_true():
    dp = get_default_deployment_parameters()
    assert dp["cycle_accurate_lif_forward"] is True


def test_legacy_ramp_switches_removed():
    """The legacy per-frame ramp and the rejected genuine-gradual ramp switches
    were removed; the value-domain blend ramp is the sole, non-optional path."""
    dp = get_default_deployment_parameters()
    assert "legacy_lif_blend_ramp" not in dp
    assert "genuine_gradual_cascade_ramp" not in dp


def test_default_enable_nevresim_simulation_is_true():
    dp = get_default_deployment_parameters()
    assert dp["enable_nevresim_simulation"] is True


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


def test_ttfs_folds_fast_driver_no_knobs_loihi_off():
    dp = {"spiking_mode": "ttfs", "weight_quantization": True}
    derive_deployment_parameters(dp)
    assert dp["optimization_driver"] == "fast"
    assert dp["enable_loihi_simulation"] is False  # loihi caps are LIF-only
    assert "lif_blend_fast" not in dp  # the analytical reference carries no knobs


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
    assert dp["ttfs_blend_fast_stabilize_steps"] == 300
    assert dp["tuning_full_transform_probe"] is True
    assert dp["enable_loihi_simulation"] is False


def test_synchronized_folds_ttfs_quantized_floor_recovery_and_disables_nevresim():
    # synchronized floor-collapse: it TRAINS the ttfs_quantized floor recovery
    # (identical recovery knobs) and DEPLOYS the mode-derived ceil kernel + grid-snap.
    dp = {"spiking_mode": "ttfs_cycle_based", "weight_quantization": True,
          "ttfs_cycle_schedule": "synchronized"}
    derive_deployment_parameters(dp)
    assert dp["activation_scale_quantile"] == 1.0
    assert dp["manager_rate_fast_rates"] == [0.25, 0.5, 0.75, 1.0]
    assert dp["manager_rate_fast_steps_per_rate"] == 120
    assert dp["optimization_driver"] == "fast"
    # the old genuine-QAT knobs are no longer folded for synchronized.
    assert "ttfs_sync_genuine_qat" not in dp
    assert dp["enable_nevresim_simulation"] is False  # no sync-window backend
    assert dp["enable_loihi_simulation"] is False


def test_sim_enables_are_capability_authoritative():
    # loihi + a non-LIF mode RAISES at assembly, so the policy MUST override an
    # explicit enable rather than ship an infeasible config.
    dp = {"spiking_mode": "ttfs", "weight_quantization": True,
          "enable_loihi_simulation": True}
    derive_deployment_parameters(dp)
    assert dp["enable_loihi_simulation"] is False


def test_vanilla_float_still_gets_driver_and_sim_enables():
    # The mode-level recipe (driver + capability sim-enables) applies even for the
    # float regime; only the quant flags are forced off by the vanilla branch.
    dp = {"spiking_mode": "lif", "weight_quantization": False}
    derive_deployment_parameters(dp)
    assert dp["pipeline_mode"] == "vanilla"
    assert dp["activation_quantization"] is False
    assert dp["optimization_driver"] == "fast"
    assert dp["enable_loihi_simulation"] is True
