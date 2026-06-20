"""Composable conversion-health calibration resolution (P2 of the trifecta).

``CalibrationPipeline.resolve`` is the SINGLE place that resolves the four TTFS
calibration steps (gain-correction / theta-cotrain / distmatch / boundary-STE) and
applies their compatibility rules — formerly per-flag guards scattered across the TTFS
tuner ``_configure``: every step cascaded-only, gain-RAMP wins over gain-COLD, and
theta-cotrain is mutually exclusive with the gain RAMP (both manage activation_scale).
The tuner binds the resolved decisions; the precedence lives (and is tested) here.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline


def _resolve(synchronized=False, distmatch_driven=False, **flags):
    return CalibrationPipeline.resolve(
        dict(flags), synchronized=synchronized, distmatch_driven=distmatch_driven,
    )


class TestDefault:
    def test_default_is_all_off(self):
        c = _resolve()
        assert not c.gain_cold
        assert not c.gain_ramp
        assert not c.gain_active
        assert not c.theta_cotrain
        assert not c.distmatch
        assert not c.boundary_ste
        assert c.boundary_surrogate_temp is None


class TestGainCorrection:
    def test_cold_only(self):
        c = _resolve(ttfs_gain_correction=True)
        assert c.gain_cold is True
        assert c.gain_ramp is False
        assert c.gain_active is True

    def test_ramp_only(self):
        c = _resolve(ttfs_gain_correction_ramp=True)
        assert c.gain_ramp is True
        assert c.gain_cold is False
        assert c.gain_active is True

    def test_ramp_wins_over_cold(self):
        c = _resolve(ttfs_gain_correction=True, ttfs_gain_correction_ramp=True)
        assert c.gain_ramp is True
        assert c.gain_cold is False, "the RAMP wins over the COLD trim"

    def test_numeric_params_carried(self):
        c = _resolve(
            ttfs_gain_correction=True,
            ttfs_gain_correction_rule="absolute",
            ttfs_gain_correction_c=2.5,
        )
        assert c.gain_rule == "absolute"
        assert c.gain_c == 2.5


class TestThetaCotrain:
    def test_theta_on(self):
        assert _resolve(ttfs_theta_cotrain=True).theta_cotrain is True

    def test_theta_excluded_by_gain_ramp(self):
        c = _resolve(ttfs_theta_cotrain=True, ttfs_gain_correction_ramp=True)
        assert c.gain_ramp is True
        assert c.theta_cotrain is False, "gain RAMP wins; both manage activation_scale"

    def test_theta_composes_with_gain_cold(self):
        # The COLD trim runs once before node build; theta is per-channel and trained.
        c = _resolve(ttfs_theta_cotrain=True, ttfs_gain_correction=True)
        assert c.gain_cold is True
        assert c.theta_cotrain is True, "theta is compatible_with the cold trim"


class TestBoundarySte:
    def test_default_temp_none(self):
        assert _resolve().boundary_surrogate_temp is None

    def test_flag_sets_temp(self):
        c = _resolve(ttfs_boundary_surrogate=True, ttfs_boundary_surrogate_temp=2.0)
        assert c.boundary_ste is True
        assert c.boundary_surrogate_temp == 2.0

    def test_default_temp_when_flag_on(self):
        c = _resolve(ttfs_boundary_surrogate=True)
        assert c.boundary_surrogate_temp == 1.0


class TestDistmatch:
    def test_distmatch_follows_driver(self):
        assert _resolve(distmatch_driven=False).distmatch is False
        assert _resolve(distmatch_driven=True).distmatch is True

    def test_distmatch_numeric_params_carried(self):
        c = _resolve(
            distmatch_driven=True,
            ttfs_distmatch_quantile=0.95,
            ttfs_distmatch_bias_iters=3,
            ttfs_distmatch_bias_eta=0.5,
        )
        assert c.distmatch_quantile == 0.95
        assert c.distmatch_bias_iters == 3
        assert c.distmatch_bias_eta == 0.5


class TestSynchronizedDisablesAll:
    def test_every_step_off_when_synchronized(self):
        c = _resolve(
            synchronized=True,
            distmatch_driven=True,
            ttfs_gain_correction=True,
            ttfs_gain_correction_ramp=True,
            ttfs_theta_cotrain=True,
            ttfs_boundary_surrogate=True,
        )
        assert not c.gain_cold
        assert not c.gain_ramp
        assert not c.gain_active
        assert not c.theta_cotrain
        assert not c.boundary_ste
        assert c.boundary_surrogate_temp is None
        # distmatch is owned by the ramp; synchronized would not drive it, but the
        # caller passes distmatch_driven=False there — the pipeline records what it is
        # told, mirroring the tuner (blend ramp is forced off under synchronized).
