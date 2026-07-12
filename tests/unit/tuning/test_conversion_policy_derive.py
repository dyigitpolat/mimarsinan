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
        "endpoint_recovery_steps": 600,
        "cycle_accurate_lif_forward": True,
        "fast_ladder_freeze_bn": True,
        "kd_ce_alpha": 0.5,
        "kd_temperature": 4.0,
        # [5v B3] the LIF half-step head-start (weight-quant QAT entry fold).
        "lif_half_step_bias": True,
        # [lif_deployment_exactness §7] the exactness ladder above the
        # half-step: C2 membrane-augmented readout (torch-side DIAGNOSTIC
        # only — deployed reads keep the counts decode, ledger §2F.1),
        # C4 per-channel FULL affine fold (calibration-only), C5
        # depth-balancing relays (exact V6 join fix, no-op on gap-free
        # graphs).
        "lif_membrane_readout": True,
        "lif_affine_fold": True,
        "lif_depth_balancing_relays": True,
        # [R5/C3, lossless ledger §2B] per-hop re-timing: the transcode is
        # value-exact (round((c/T)*T) = c) and kills the V3 back-loading on
        # the temporal-A6 FAIL cells at S<=8 (+1.9pp at S=4 chain9, +0.5pp at
        # S=8; nil at S>=16); the latency cost stays mapping-visible.
        "lif_per_hop_retiming": True,
        # [R1/M2, lossless ledger §2C] two-scale WQ grid for LIF: four frozen
        # WQ endpoints prove the residual is grid arithmetic the QAT cannot
        # express (t01_19 -0.57pp WQ residual, t0_05 -0.52pp, t0_03 -0.24pp;
        # wb8 control +1.25pp deployed at S=4). Exact identity, same bits;
        # the resolver capability-gates nobias platforms.
        "wq_two_scale_projection": True,
        # [5u generalized] the WQ-scoped well-conditioned endpoint floor.
        "wq_endpoint_target_floor": 0.98,
        "wq_endpoint_recovery_steps": 16000,
    },
    # [5u] the analytical reference is the bit-parity-lossless family: its
    # endpoint may chase the acceptance target (preservation ≡ stagnation at
    # the float envelope otherwise), funded by the measured wall headroom.
    ("ttfs", None): {
        "endpoint_target_floor": 0.98,
        "wq_endpoint_recovery_steps": 16000,
    },
    ("ttfs_quantized", None): {
        "activation_scale_quantile": 1.0,
        "manager_rate_fast_rates": [0.25, 0.5, 0.75, 1.0],
        "manager_rate_fast_steps_per_rate": 120,
        # [C4] the WQ-scoped well-conditioned floor: the ttfsq proxy→deployed
        # transfer is measured sub-SE (t0_11 +0.0007 / t0_14 −0.0014 / t01_06
        # −0.0010 vs SE 0.0092), so the floor-funded climb survives to hcm.
        "wq_endpoint_target_floor": 0.98,
        "wq_endpoint_recovery_steps": 16000,
    },
    ("ttfs_cycle_based", "cascaded"): {
        "ttfs_genuine_blend_ramp": True,
        "ttfs_genuine_blend_fast": True,
        # T4/P4: multi-segment vehicles walk the converted-prefix frontier
        # (single-segment vehicles keep the blend ramp).
        "ttfs_prefix_ramp": True,
        # [5v B2] single-segment deep chains walk the hop frontier instead.
        "ttfs_hop_prefix_ramp": True,
        # P1'' budget = W3 reinvestment of reclaimed eval wall: the FT endpoint
        # was the only bound binding while still improving (X3 300/300 cutoffs).
        "endpoint_recovery_steps": 600,
        "tuning_full_transform_probe": True,
        # [M2] two-scale WQ projection: the shared max(|w|,|b|) grid is set by
        # the bias on the fc perceptrons and craters the first-crossing forward
        # (wq_cascade_crater_repair.md); weight grid from max|w| alone.
        "wq_two_scale_projection": True,
        # [5u generalized] the WQ-scoped well-conditioned endpoint floor.
        "wq_endpoint_target_floor": 0.98,
        "wq_endpoint_recovery_steps": 16000,
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
        # [5v B1] the two scalar crater levers: the A6-gauge-driven quantile
        # (full-quantile theta deflates where it starves the grid) and the
        # half-step ENTRY fold into bias (trainable; the exact-kernel QAT owns it).
        "starvation_aware_scale_quantile": True,
        "sync_entry_half_step": True,
        # [5v B1(iii)] the hop frontier (arms only on A6-FAIL x deep chains).
        "sync_hop_staged_install": True,
        # [M2] the value-domain forward craters from the same bias-set shared
        # grid (wq_cascade_crater_repair.md §4.4 value-forward control), so the
        # whole ttfs_cycle_based family carries the two-scale projection.
        "wq_two_scale_projection": True,
        # [5u generalized] the WQ-scoped well-conditioned endpoint floor.
        "wq_endpoint_target_floor": 0.98,
        "wq_endpoint_recovery_steps": 16000,
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
    # [5u] the endpoint target floor: the one divergence the lossless row carries.
    ("ttfs", None): "endpoint_target_floor",
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

    def test_ttfs_reference_carries_the_wq_demotion_plus_the_floor(self):
        # The analytical TTFS column rides the plain fast ladder; beyond the
        # mode-independent WQ demotion it carries ONLY the [5u] endpoint floor
        # (lossless ⇒ preservation would otherwise stagnate at the envelope).
        knobs = dict(ConversionPolicy.derive("ttfs").knobs)
        assert knobs == {
            **_WQ_KNOBS,
            "endpoint_target_floor": 0.98,
            "wq_endpoint_recovery_steps": 16000,
        }

    def test_two_scale_projection_scope_is_cycle_based_plus_lif(self):
        # [M2 + R1 scope] the two-scale WQ grid repairs the first-crossing
        # (and the sync value) forward, and — measured-refuting the old
        # "training-recoverable" premise — LIF's WQ residual: four frozen WQ
        # endpoints (entry == exit, divergence_rescued) prove the loss is grid
        # arithmetic no recovery step expresses (t01_19 -0.57pp, t0_05
        # -0.52pp; lossless_refinement_ledger.md §2C, G5). ttfsq/analytic
        # stay on the shared grid for byte-identity.
        for mode, schedule in _CELLS:
            recipe = ConversionPolicy.derive(mode, schedule)
            expected = mode in ("ttfs_cycle_based", "lif")
            assert ("wq_two_scale_projection" in recipe.knobs) is expected, (
                f"{mode}/{schedule}"
            )
            if expected:
                assert recipe.knobs["wq_two_scale_projection"] is True

    def test_floor_rides_only_the_bit_parity_lossless_row(self):
        # [5u scope] near-lossless modes (lif, ttfs_quantized, synchronized) and
        # lossy modes (cascaded) must NOT get the floor: their endpoint gap is
        # real, and the high-water target is already the honest anchor.
        for mode, schedule in _CELLS:
            recipe = ConversionPolicy.derive(mode, schedule)
            expected = mode == "ttfs"
            assert ("endpoint_target_floor" in recipe.knobs) is expected, (
                f"{mode}/{schedule}"
            )


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

    def test_lossless_reference_marks_the_floor_as_its_special_case(self):
        recipe = ConversionPolicy.derive("ttfs")
        assert recipe.special_case == "endpoint_target_floor"
        assert recipe.rationale, "the [5u] divergence must cite its finding"


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
