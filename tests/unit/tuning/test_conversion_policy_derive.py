"""The ConversionPolicy SSOT table: ``derive(spiking_mode, schedule)`` maps each
deployment mode to its empirically-proven recipe (driver + knob set + the
capability-derived sim-enable set + a special-case marker for the divergences).

These are the collapsed fix-wave findings (``docs/research/findings/*`` in
``mimarsinan_research``): every proven recipe rides the fast ladder; the four
marked rows are the documented divergences from the generic flow. The knob VALUES
here are the SSOT — the per-mode recipe constants are no longer user config keys.
"""

from __future__ import annotations

import pytest

from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_CONTROLLER,
    OPTIMIZATION_DRIVER_FAST,
    ConversionPolicy,
    ConversionRecipe,
)

# WQ demotion knobs ride EVERY mode's recipe (mode-independent, theory 5g-v).
_WQ_KNOBS = {
    "wq_fast_rates": [0.5, 1.0],
    "wq_fast_steps_per_rate": 0,
    "wq_endpoint_recovery_steps": 600,
}

# (spiking_mode, schedule) -> proven recipe knobs (the SSOT constants).
_MODE_KNOBS = {
    ("lif", None): {
        "lif_blend_fast": True,
        "lif_tanneal": True,
        # P1'' budget = dropped inert Clamp/AQ ladders (960) + retired stabilize (600).
        "endpoint_recovery_steps": 1560,
        "cycle_accurate_lif_forward": True,
        "fast_ladder_freeze_bn": True,
        "kd_ce_alpha": 0.5,
        "kd_temperature": 4.0,
    },
    ("ttfs", None): {},
    ("ttfs_quantized", None): {
        "activation_scale_quantile": 1.0,
        "manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
        "manager_rate_fast_steps_per_rate": 120,
    },
    ("ttfs_cycle_based", "cascaded"): {
        "ttfs_genuine_blend_ramp": True,
        "ttfs_genuine_blend_fast": True,
        # P1'' budget = the retired 300-step stabilize (+ freed stall steps at runtime).
        "endpoint_recovery_steps": 300,
        "tuning_full_transform_probe": True,
    },
    # synchronized rides the ttfs_quantized ladder shape but TRAINS the exact
    # deployed composition (ceil kernel + grid snap) as the QAT endpoint (T6);
    # the mapping-time half-step bias compensation is skipped for models so
    # trained (marker-asserted).
    ("ttfs_cycle_based", "synchronized"): {
        "activation_scale_quantile": 1.0,
        "manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
        "manager_rate_fast_steps_per_rate": 120,
        "sync_exact_qat": True,
        # P1'' budget at the sync AQ endpoint (replaces the open-ended stabilize).
        "endpoint_recovery_steps": 600,
    },
}

_EXPECTED_KNOBS = {
    cell: {**_WQ_KNOBS, **knobs} for cell, knobs in _MODE_KNOBS.items()
}

# (spiking_mode, schedule) -> capability-derived sim-enable set (from _BACKEND_CAPS).
_EXPECTED_SIM_ENABLES = {
    ("lif", None): {
        "enable_nevresim_simulation": True,
        "enable_sanafe_simulation": True,
        "enable_loihi_simulation": True,
    },
    ("ttfs", None): {
        "enable_nevresim_simulation": True,
        "enable_sanafe_simulation": True,
        "enable_loihi_simulation": False,
    },
    ("ttfs_quantized", None): {
        "enable_nevresim_simulation": True,
        "enable_sanafe_simulation": True,
        "enable_loihi_simulation": False,
    },
    ("ttfs_cycle_based", "cascaded"): {
        "enable_nevresim_simulation": True,
        "enable_sanafe_simulation": True,
        "enable_loihi_simulation": False,
    },
    ("ttfs_cycle_based", "synchronized"): {
        "enable_nevresim_simulation": False,
        "enable_sanafe_simulation": True,
        "enable_loihi_simulation": False,
    },
}

_EXPECTED_SPECIAL_CASE = {
    ("lif", None): "bn_freeze",
    ("ttfs", None): None,
    ("ttfs_quantized", None): "full_quantile_decode",
    ("ttfs_cycle_based", "cascaded"): "fast_only_never_controller",
    # synchronized trains the exact deployed ceil kernel as the QAT endpoint (T6).
    ("ttfs_cycle_based", "synchronized"): "sync_exact_endpoint",
}

_CELLS = list(_EXPECTED_KNOBS.keys())


class TestDeriveDriver:
    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_every_proven_recipe_rides_the_fast_ladder(self, mode, schedule):
        recipe = ConversionPolicy.derive(mode, schedule)
        assert isinstance(recipe, ConversionRecipe)
        assert recipe.driver == OPTIMIZATION_DRIVER_FAST

    def test_cascaded_is_never_the_controller(self):
        # The controller path collapses on the deep cascade (→ chance); the fast
        # ladder is the only proven survivor. The SSOT can NEVER yield controller here.
        recipe = ConversionPolicy.derive("ttfs_cycle_based", "cascaded")
        assert recipe.driver != OPTIMIZATION_DRIVER_CONTROLLER
        assert recipe.driver == OPTIMIZATION_DRIVER_FAST


class TestDeriveKnobs:
    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_proven_knob_set_is_exact(self, mode, schedule):
        recipe = ConversionPolicy.derive(mode, schedule)
        assert dict(recipe.knobs) == _EXPECTED_KNOBS[(mode, schedule)]

    def test_cascaded_default_schedule_is_the_cascaded_recipe(self):
        # ttfs_cycle_based with no schedule normalizes to cascaded.
        assert dict(ConversionPolicy.derive("ttfs_cycle_based").knobs) == _EXPECTED_KNOBS[
            ("ttfs_cycle_based", "cascaded")
        ]

    def test_ttfs_reference_carries_only_the_wq_demotion(self):
        # The analytical TTFS column is the generic reference (plain fast);
        # only the mode-independent WQ demotion rides it.
        assert dict(ConversionPolicy.derive("ttfs").knobs) == _WQ_KNOBS


class TestDeriveSimEnables:
    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_sim_enable_set_is_capability_derived(self, mode, schedule):
        recipe = ConversionPolicy.derive(mode, schedule)
        assert dict(recipe.sim_enables) == _EXPECTED_SIM_ENABLES[(mode, schedule)]

    def test_loihi_only_for_the_lif_family(self):
        # loihi caps are LIF-only; a non-LIF mode + loihi RAISES at assembly, so the
        # policy must keep loihi off for every TTFS mode.
        for mode, schedule in _CELLS:
            recipe = ConversionPolicy.derive(mode, schedule)
            expected = mode == "lif"
            assert recipe.sim_enables["enable_loihi_simulation"] is expected

    def test_nevresim_off_only_for_synchronized(self):
        # nevresim has no synchronized-window backend; every other mode runs on it.
        for mode, schedule in _CELLS:
            recipe = ConversionPolicy.derive(mode, schedule)
            expected = not (mode == "ttfs_cycle_based" and schedule == "synchronized")
            assert recipe.sim_enables["enable_nevresim_simulation"] is expected

    def test_sanafe_on_for_every_mode(self):
        for mode, schedule in _CELLS:
            recipe = ConversionPolicy.derive(mode, schedule)
            assert recipe.sim_enables["enable_sanafe_simulation"] is True


class TestDeriveSpecialCases:
    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_special_case_marker_matches(self, mode, schedule):
        recipe = ConversionPolicy.derive(mode, schedule)
        assert recipe.special_case == _EXPECTED_SPECIAL_CASE[(mode, schedule)]

    @pytest.mark.parametrize("mode,schedule", _CELLS)
    def test_special_cases_carry_a_rationale(self, mode, schedule):
        # Each divergence from the generic flow is studyable: a non-empty rationale
        # citing the finding. The generic (ttfs) row needs no rationale.
        recipe = ConversionPolicy.derive(mode, schedule)
        if recipe.special_case is None:
            continue_ok = True
            assert continue_ok
        else:
            assert recipe.rationale, f"{mode}/{schedule} special case must cite a finding"

    def test_generic_reference_has_no_special_case(self):
        recipe = ConversionPolicy.derive("ttfs")
        assert recipe.special_case is None


class TestRemovedModeRejected:
    def test_rate_is_rejected_with_migration_hint(self):
        # 'rate' (naive analytical LIF, collapses at SCM) was removed; deriving it
        # must fail loud and point at its replacement, not silently fall through.
        with pytest.raises(ValueError, match="rate.*removed.*use 'lif'"):
            ConversionPolicy.derive("rate")

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown spiking_mode"):
            ConversionPolicy.derive("bogus")


class TestRecipeIsFrozen:
    def test_derived_recipe_is_immutable(self):
        recipe = ConversionPolicy.derive("lif")
        with pytest.raises(Exception):
            recipe.driver = OPTIMIZATION_DRIVER_CONTROLLER  # type: ignore[misc]
