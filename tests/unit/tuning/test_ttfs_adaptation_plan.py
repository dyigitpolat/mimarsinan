"""Contract-driven resolution of the TTFS adaptation flag-thicket.

``TtfsAdaptationPlan.resolve`` is the SINGLE place that reads the ~24 ``ttfs_*``
adaptation flags and applies the precedence / compatibility rules (blend ⊐ annealed,
STE ⇒ annealed, ste_fast ⊂ ste, proxy_fast excludes the genuine ramps + synchronized,
ste_refine ⊂ proxy_fast, the fast-ladder rung resolution). The tuner reads the
validated plan instead of re-deriving the dispatch across ``_configure``.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan


def _resolve(synchronized=False, **flags):
    return TtfsAdaptationPlan.resolve(dict(flags), synchronized=synchronized)


class TestRampPrecedence:
    def test_default_is_value_domain_proxy(self):
        p = _resolve()
        assert not p.genuine_annealed_ramp
        assert not p.genuine_blend_ramp
        assert not p.staircase_ste
        assert not p.genuine_bare_target_ramp
        assert not p.proxy_fast

    def test_blend_wins_over_annealed(self):
        p = _resolve(ttfs_genuine_annealed_ramp=True, ttfs_genuine_blend_ramp=True)
        assert p.genuine_blend_ramp is True
        assert p.genuine_annealed_ramp is False, "blend wins; annealed forced off"

    def test_ste_forces_annealed_install(self):
        p = _resolve(ttfs_staircase_ste=True)
        assert p.staircase_ste is True
        assert p.genuine_annealed_ramp is True, "STE reuses the annealed cascade-forward install"
        assert p.genuine_bare_target_ramp is True

    def test_ste_excluded_by_blend(self):
        p = _resolve(ttfs_staircase_ste=True, ttfs_genuine_blend_ramp=True)
        assert p.genuine_blend_ramp is True
        assert p.staircase_ste is False

    def test_synchronized_disables_all_genuine_ramps(self):
        p = _resolve(
            synchronized=True,
            ttfs_genuine_annealed_ramp=True, ttfs_genuine_blend_ramp=True,
            ttfs_staircase_ste=True, ttfs_blend_fast=True,
        )
        assert not p.genuine_annealed_ramp
        assert not p.genuine_blend_ramp
        assert not p.staircase_ste
        assert not p.proxy_fast


class TestDriverPrecedence:
    def test_ste_fast_requires_ste(self):
        assert _resolve(ttfs_staircase_ste_fast=True).staircase_ste_fast is False
        assert _resolve(
            ttfs_staircase_ste=True, ttfs_staircase_ste_fast=True,
        ).staircase_ste_fast is True

    def test_genuine_blend_fast_requires_blend(self):
        assert _resolve(ttfs_genuine_blend_fast=True).genuine_blend_fast is False
        assert _resolve(
            ttfs_genuine_blend_ramp=True, ttfs_genuine_blend_fast=True,
        ).genuine_blend_fast is True

    def test_proxy_fast_excludes_bare_target(self):
        # proxy_fast is a value-domain path: inert when a genuine bare-target ramp is on.
        assert _resolve(
            ttfs_blend_fast=True, ttfs_genuine_annealed_ramp=True,
        ).proxy_fast is False
        assert _resolve(ttfs_blend_fast=True).proxy_fast is True

    def test_ste_refine_requires_proxy_fast(self):
        assert _resolve(ttfs_blend_fast_ste_refine=True).staircase_ste_refine is False
        assert _resolve(
            ttfs_blend_fast=True, ttfs_blend_fast_ste_refine=True,
        ).staircase_ste_refine is True


class TestFastLadderResolution:
    def test_ste_fast_single_rung(self):
        p = _resolve(
            ttfs_staircase_ste=True, ttfs_staircase_ste_fast=True, ttfs_ste_steps=777,
        )
        assert p.fast_ladder_enabled is True
        assert p.fast_ladder_rates == [1.0]
        assert p.fast_ladder_steps_per_rate == 777
        assert p.fast_ladder_eta_min_factor == 0.0

    def test_proxy_fast_ladder_floors_eta(self):
        p = _resolve(ttfs_blend_fast=True, ttfs_blend_fast_lr_eta_min=0.25)
        assert p.fast_ladder_enabled is True
        assert p.fast_ladder_rates == [0.5, 0.75, 0.9, 0.97, 1.0]
        assert p.fast_ladder_eta_min_factor == 0.25

    def test_genuine_blend_fast_ladder_no_eta_floor(self):
        p = _resolve(ttfs_genuine_blend_ramp=True, ttfs_genuine_blend_fast=True)
        assert p.fast_ladder_enabled is True
        assert p.fast_ladder_eta_min_factor == 0.0

    def test_no_fast_path_disables_ladder(self):
        assert _resolve().fast_ladder_enabled is False
        assert _resolve(ttfs_genuine_annealed_ramp=True).fast_ladder_enabled is False


class TestComposedTrifecta:
    """The plan composes the orthogonal abstractions: the optimization driver owns
    the fast-ladder rung, the calibration pipeline owns the conversion-health steps.
    The legacy ``fast_ladder_*`` accessors mirror the driver (back-compat for the
    tuner binds)."""

    def test_driver_is_an_optimization_driver(self):
        from mimarsinan.tuning.orchestration.optimization_driver import (
            OptimizationDriver,
        )

        p = _resolve(ttfs_staircase_ste=True, ttfs_staircase_ste_fast=True)
        assert isinstance(p.driver, OptimizationDriver)
        assert p.driver.fast_ladder is True

    def test_fast_ladder_accessors_mirror_driver(self):
        p = _resolve(ttfs_blend_fast=True, ttfs_blend_fast_lr_eta_min=0.3)
        assert p.fast_ladder_enabled == p.driver.fast_ladder
        assert p.fast_ladder_rates == p.driver.fast_ladder_rates
        assert p.fast_ladder_steps_per_rate == p.driver.fast_ladder_steps_per_rate
        assert p.fast_ladder_eta_min_factor == p.driver.fast_ladder_eta_min_factor

    def test_calibration_is_a_calibration_pipeline(self):
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        p = _resolve(ttfs_gain_correction=True, ttfs_theta_cotrain=True)
        assert isinstance(p.calibration, CalibrationPipeline)
        assert p.calibration.gain_cold is True
        assert p.calibration.theta_cotrain is True

    def test_distmatch_driven_by_blend_ramp(self):
        assert _resolve(ttfs_genuine_blend_ramp=True).calibration.distmatch is True
        assert _resolve().calibration.distmatch is False

    def test_calibration_off_under_synchronized(self):
        p = _resolve(
            synchronized=True,
            ttfs_gain_correction=True,
            ttfs_theta_cotrain=True,
            ttfs_boundary_surrogate=True,
            ttfs_genuine_blend_ramp=True,
        )
        assert not p.calibration.gain_active
        assert not p.calibration.theta_cotrain
        assert not p.calibration.boundary_ste
        assert not p.calibration.distmatch, "blend ramp off under synchronized"


class TestNumericPassThrough:
    def test_numeric_params_carried(self):
        p = _resolve(
            ttfs_staircase_ste=True, ttfs_ste_mix=0.3, ttfs_ste_w_lr=1e-3,
            ttfs_ste_theta_lr=2e-2, ttfs_ste_init_frac=0.25,
            ttfs_blend_fast_steps_per_rate=99, ttfs_blend_fast_stabilize_steps=321,
        )
        assert p.ste_mix == 0.3
        assert p.ste_w_lr == 1e-3
        assert p.ste_theta_lr == 2e-2
        assert p.ste_init_frac == 0.25
        assert p.blend_fast_steps_per_rate == 99
        assert p.fast_stabilize_steps == 321

    def test_ste_steps_floored_at_one(self):
        assert _resolve(ttfs_staircase_ste=True, ttfs_ste_steps=0).ste_steps == 1
