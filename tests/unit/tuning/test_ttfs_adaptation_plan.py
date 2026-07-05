"""Contract-driven resolution of the TTFS adaptation flag-thicket.

``TtfsAdaptationPlan.resolve`` is the SINGLE place that reads the ``ttfs_*`` adaptation
flags and applies the precedence / compatibility rules (the genuine blend ramp is
cascaded-only; ``genuine_blend_fast`` requires the blend ramp; the value-domain
``proxy_fast`` excludes the genuine blend ramp and synchronized unless sync-QAT opts it
in; the fast-ladder rung resolution). The tuner reads the validated plan instead of
re-deriving the dispatch across ``_configure``.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan


def _resolve(synchronized=False, optimization_driver=None, **flags):
    return TtfsAdaptationPlan.resolve(
        dict(flags),
        synchronized=synchronized,
        optimization_driver=optimization_driver,
    )


class TestRampPrecedence:
    def test_default_is_value_domain_proxy(self):
        p = _resolve()
        assert not p.genuine_blend_ramp
        assert not p.proxy_fast

    def test_synchronized_disables_the_genuine_blend_ramp(self):
        p = _resolve(
            synchronized=True,
            ttfs_genuine_blend_ramp=True, ttfs_blend_fast=True,
        )
        assert not p.genuine_blend_ramp
        assert not p.proxy_fast

    def test_sync_genuine_qat_does_not_enable_the_cascade_blend_ramp(self):
        p = _resolve(
            synchronized=True,
            ttfs_sync_genuine_qat=True,
            ttfs_genuine_blend_ramp=True,
        )
        assert p.sync_genuine_qat is True
        assert not p.genuine_blend_ramp

    def test_sync_genuine_qat_allows_fast_deployed_staircase_ramp(self):
        p = _resolve(
            synchronized=True,
            optimization_driver="fast",
            ttfs_sync_genuine_qat=True,
            ttfs_blend_fast=True,
        )
        assert p.sync_genuine_qat is True
        assert p.proxy_fast is True
        assert p.fast_ladder_enabled is True

    def test_sync_fast_proxy_stays_default_off_without_sync_qat_flag(self):
        p = _resolve(
            synchronized=True,
            optimization_driver="fast",
            ttfs_blend_fast=True,
        )
        assert p.sync_genuine_qat is False
        assert p.proxy_fast is False
        assert p.fast_ladder_enabled is False


class TestDriverPrecedence:
    def test_genuine_blend_fast_requires_blend(self):
        assert _resolve(ttfs_genuine_blend_fast=True).genuine_blend_fast is False
        assert _resolve(
            ttfs_genuine_blend_ramp=True, ttfs_genuine_blend_fast=True,
        ).genuine_blend_fast is True

    def test_proxy_fast_excludes_the_genuine_blend_ramp(self):
        # proxy_fast is a value-domain path: inert when the genuine blend ramp is on.
        assert _resolve(
            ttfs_blend_fast=True, ttfs_genuine_blend_ramp=True,
        ).proxy_fast is False
        assert _resolve(ttfs_blend_fast=True).proxy_fast is True


class TestFastLadderResolution:
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
        assert _resolve(ttfs_genuine_blend_ramp=True).fast_ladder_enabled is False


class TestComposedTrifecta:
    """The plan composes the orthogonal abstractions: the optimization driver owns
    the fast-ladder rung, the calibration pipeline owns the conversion-health steps.
    The legacy ``fast_ladder_*`` accessors mirror the driver (back-compat for the
    tuner binds)."""

    def test_driver_is_an_optimization_driver(self):
        from mimarsinan.tuning.orchestration.optimization_driver import (
            OptimizationDriver,
        )

        p = _resolve(ttfs_blend_fast=True)
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


class TestOptimizationDriverAxis:
    """EF1 — TTFS reads the pipeline-wide ``optimization_driver`` axis, not just its
    own ``ttfs_*`` flags. The axis is the controller-vs-fast GATE over the fast fork;
    the per-family flag still selects WHICH fast variant (genuine blend / value-domain
    proxy). ``None`` (the back-compat default for direct callers) derives the axis from
    the legacy flags, so the resolution stays byte-identical."""

    def test_none_axis_derives_from_legacy_flags(self):
        # No explicit axis: a TTFS fast flag still selects the fast ladder (the legacy
        # switch feeds the pipeline-wide axis; byte-identical to pre-EF1 resolution).
        p = _resolve(ttfs_blend_fast=True)
        assert p.proxy_fast is True
        assert p.fast_ladder_enabled is True

    def test_controller_axis_forces_every_fast_selector_off(self):
        # An explicit controller axis vetoes the fast path even with a fast flag set.
        p = _resolve(
            optimization_driver="controller",
            ttfs_blend_fast=True,
            ttfs_genuine_blend_ramp=True,
            ttfs_genuine_blend_fast=True,
        )
        assert p.proxy_fast is False
        assert p.genuine_blend_fast is False
        assert p.fast_ladder_enabled is False
        assert p.driver.controller is True

    def test_controller_axis_keeps_ramp_strategy(self):
        # The axis only gates the optimization DRIVER; the genuine ramp strategy
        # (a separate concern) is untouched.
        p = _resolve(
            optimization_driver="controller",
            ttfs_genuine_blend_ramp=True,
            ttfs_genuine_blend_fast=True,
        )
        assert p.genuine_blend_ramp is True
        assert p.genuine_blend_fast is False

    def test_fast_axis_selects_the_flagged_variant(self):
        proxy = _resolve(optimization_driver="fast", ttfs_blend_fast=True)
        assert proxy.proxy_fast is True
        blend = _resolve(
            optimization_driver="fast",
            ttfs_genuine_blend_ramp=True, ttfs_genuine_blend_fast=True,
        )
        assert blend.genuine_blend_fast is True

    def test_fast_axis_without_a_flag_has_no_fast_variant(self):
        # The axis enables fast, but with no per-family fast flag there is no variant
        # to run (the flag selects WHICH fast path) — so the ladder stays disabled.
        p = _resolve(optimization_driver="fast")
        assert p.proxy_fast is False
        assert p.genuine_blend_fast is False
        assert p.fast_ladder_enabled is False


class TestNumericPassThrough:
    def test_numeric_params_carried(self):
        p = _resolve(
            ttfs_blend_fast_steps_per_rate=99, endpoint_recovery_steps=321,
        )
        assert p.blend_fast_steps_per_rate == 99
        assert p.endpoint_recovery_steps == 321
