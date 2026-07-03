"""Stage-2 SSOT flag-collapse contract.

The ConversionPolicy recipe is now the SOLE source of the per-mode conversion
recipe: the ~58 research/recipe knobs that used to be user-tunable defaults are
deleted, and the ``_fold_conversion_recipe`` derivation writes the recipe
AUTHORITATIVELY (a user value is overwritten, not honored). This module pins
that contract so the collapse cannot silently regress into a feature-flag island.
"""

import pytest

from mimarsinan.config_schema import build_flat_pipeline_config
from mimarsinan.config_schema.defaults import (
    CONFIG_KEYS_SET,
    DEFAULT_DEPLOYMENT_PARAMETERS,
    get_default_deployment_parameters,
)
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.namespaced_schema import (
    registered_flat_keys,
    unregistered_default_keys,
)

# The full set of knobs collapsed into the ConversionPolicy SSOT (recipe-active),
# the TuningPolicy SSOT (frozen tuning-loop behavior), or deleted as measured-dead
# research levers. None of these is a config key any longer — they are internal
# policy/recipe-table constants or gone entirely.
COLLAPSED_KEYS = frozenset({
    "checkpoint_location",
    "checkpoint_scope",
    "global_budget",
    "k_commit",
    "tuning_use_paired_sensor",
    "fast_ladder_freeze_bn",
    "lif_blend_fast",
    "lif_blend_fast_lr_eta_min",
    "lif_blend_fast_rates",
    "lif_blend_fast_stabilize_steps",
    "lif_blend_fast_steps_per_rate",
    "lif_distmatch",
    "lif_distmatch_bias_eta",
    "lif_distmatch_bias_iters",
    "lif_distmatch_cal_batches",
    "lif_theta_cotrain",
    "optimization_driver",
    "ttfs_blend_fast",
    "ttfs_blend_fast_lr_eta_min",
    "ttfs_blend_fast_rates",
    "ttfs_blend_fast_stabilize_steps",
    "ttfs_blend_fast_steps_per_rate",
    "ttfs_blend_fast_ste_refine",
    "ttfs_boundary_surrogate",
    "ttfs_boundary_surrogate_temp",
    "ttfs_distmatch_bias_eta",
    "ttfs_distmatch_bias_iters",
    "ttfs_distmatch_quantile",
    "ttfs_gain_correction",
    "ttfs_gain_correction_c",
    "ttfs_gain_correction_ramp",
    "ttfs_gain_correction_rule",
    "ttfs_genuine_annealed_ramp",
    "ttfs_genuine_blend_fast",
    "ttfs_genuine_blend_ramp",
    "ttfs_ramp_alpha_max",
    "ttfs_ramp_alpha_min",
    "ttfs_scale_aware_boundaries",
    "ttfs_staircase_ste",
    "ttfs_staircase_ste_fast",
    "ttfs_ste_init_frac",
    "ttfs_ste_mix",
    "ttfs_ste_steps",
    "ttfs_ste_theta_lr",
    "ttfs_ste_w_lr",
    "ttfs_sync_genuine_qat",
    "ttfs_theta_cotrain",
    "tuning_characterization_grid",
    "tuning_enable_characterization",
    "tuning_full_transform_probe",
    "tuning_keepbest_certified",
    "tuning_recipe_recovery",
    "tuning_recovery_check_divisor",
    "tuning_recovery_lr_plateau",
    "tuning_recovery_lr_plateau_factor",
    "tuning_recovery_lr_plateau_reductions",
    "tuning_refind_lr_on_miss",
    "tuning_rollback_cumulative_bound",
    "tuning_rollback_ratchet",
    "tuning_stabilization_bounded",
    "tuning_stabilization_ratio",
    "tuning_target_floor_on_real_target",
    "tuning_tight_plateau",
})

# Knobs the recipe overrides PER MODE but which remain registered config keys
# with a generic default (the recipe writes a mode-specific value over them).
KEPT_GENERAL_KNOBS = frozenset({
    "activation_scale_quantile",
    "kd_ce_alpha",
    "kd_temperature",
    "ttfs_genuine_blend_ce_alpha",
    "cycle_accurate_lif_forward",
})


class TestCollapsedKeysAreGone:
    @pytest.mark.parametrize("key", sorted(COLLAPSED_KEYS))
    def test_absent_from_defaults(self, key):
        assert key not in DEFAULT_DEPLOYMENT_PARAMETERS

    @pytest.mark.parametrize("key", sorted(COLLAPSED_KEYS))
    def test_absent_from_config_keys_set(self, key):
        assert key not in CONFIG_KEYS_SET

    @pytest.mark.parametrize("key", sorted(COLLAPSED_KEYS))
    def test_absent_from_registry(self, key):
        assert key not in registered_flat_keys()

    def test_registry_has_no_unregistered_default_leftovers(self):
        # Symmetric deletion keeps the strangler-fig invariant exact.
        assert unregistered_default_keys() == frozenset()

    def test_general_knobs_survive(self):
        for key in KEPT_GENERAL_KNOBS:
            assert key in DEFAULT_DEPLOYMENT_PARAMETERS, key
            assert key in CONFIG_KEYS_SET, key


class TestFoldIsAuthoritative:
    """A user value for a recipe knob is OVERWRITTEN by the derived recipe."""

    def test_lif_recipe_overrides_injected_user_values(self):
        dp = {
            "spiking_mode": "lif",
            "weight_quantization": True,
            # Adversarial user overrides that the Pure-SSOT fold must overwrite.
            "lif_blend_fast": False,
            "optimization_driver": "controller",
            "kd_ce_alpha": 0.1,
            "fast_ladder_freeze_bn": False,
        }
        derive_deployment_parameters(dp)
        assert dp["optimization_driver"] == "fast"
        assert dp["lif_blend_fast"] is True
        assert dp["fast_ladder_freeze_bn"] is True
        assert dp["kd_ce_alpha"] == 0.5

    def test_cascaded_recipe_writes_genuine_blend_not_stale_proxy(self):
        # The cascaded recipe is the GENUINE blend ramp; a stale proxy override
        # must not survive to conflict with it.
        dp = {
            "spiking_mode": "ttfs_cycle_based",
            "weight_quantization": True,
            "ttfs_cycle_schedule": "cascaded",
            "optimization_driver": "controller",
        }
        derive_deployment_parameters(dp)
        assert dp["optimization_driver"] == "fast"
        assert dp["ttfs_genuine_blend_ramp"] is True
        assert dp["ttfs_genuine_blend_fast"] is True


class TestUnknownKeyHandledCleanly:
    """A config still carrying a deleted knob is ignored, not mis-behaving."""

    def test_deleted_key_passes_through_build_and_roundtrips(self):
        dp = get_default_deployment_parameters()
        dp["spiking_mode"] = "lif"
        dp["weight_quantization"] = True
        dp["ttfs_gain_correction"] = True  # a now-deleted research knob
        resolved = build_flat_pipeline_config(dp, {}, pipeline_mode="phased")
        # The stale knob neither crashes the build nor is silently promoted into
        # the recipe — it simply rides along as an inert pass-through value.
        assert resolved["ttfs_gain_correction"] is True
        assert resolved["optimization_driver"] == "fast"
