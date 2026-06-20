"""Composable conversion-health calibration resolution (P2 of the trifecta).

``CalibrationPipeline.for_mode`` is the SINGLE place that resolves the four
calibration steps (gain-correction / theta-cotrain / distmatch / boundary-STE) and
applies their compatibility rules — formerly per-flag guards scattered across the TTFS
tuner ``_configure``: the ENABLE is a (firing × sync) decision owned by the
``SpikingModePolicy``, gain-RAMP wins over gain-COLD, and theta-cotrain is mutually
exclusive with the gain RAMP (both manage activation_scale). The tuner binds the
resolved decisions; the precedence lives (and is tested) here.

E3: the resolution is keyed by the (firing × sync) policy (``for_mode``), not the
``ttfs_*`` names. ``resolve(synchronized=…)`` is the back-compat boolean alias and
must stay bit-exact with the policy-keyed path.
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.spiking_mode_policy import (
    policy_for_spiking_mode,
)
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


# ── E3: the resolution is keyed by the (firing × sync) policy, not ttfs_* ──────

_ALL_STEPS = dict(
    ttfs_gain_correction=True,
    ttfs_gain_correction_ramp=True,
    ttfs_theta_cotrain=True,
    ttfs_boundary_surrogate=True,
)


def _for_mode(mode, schedule=None, distmatch_driven=False, **flags):
    return CalibrationPipeline.for_mode(
        dict(flags),
        mode_policy=policy_for_spiking_mode(mode, schedule),
        distmatch_driven=distmatch_driven,
    )


class TestInert:
    def test_inert_is_all_off(self):
        c = CalibrationPipeline.inert()
        assert not (c.gain_cold or c.gain_ramp or c.gain_active)
        assert not c.theta_cotrain
        assert not c.distmatch
        assert not c.boundary_ste
        assert c.boundary_surrogate_temp is None

    def test_inert_equals_default_for_mode_off_cell(self):
        # A cell that does no conversion-health calibration ignores its ttfs_* flags
        # AND its distmatch driver → the inert pipeline (byte-identical default-off).
        assert _for_mode("lif", distmatch_driven=True, **_ALL_STEPS) == (
            CalibrationPipeline.inert()
        )


class TestForModeKeyedByContract:
    """The ENABLE is the (firing × sync) cell, owned by the SpikingModePolicy."""

    OFF_CELLS = [
        ("lif", None),
        ("rate", None),
        ("ttfs", "cascaded"),
        ("ttfs", "synchronized"),
        ("ttfs_quantized", "cascaded"),
        ("ttfs_cycle_based", "synchronized"),
    ]

    @pytest.mark.parametrize("mode,schedule", OFF_CELLS)
    def test_non_cascade_cells_are_inert_even_with_all_flags(self, mode, schedule):
        # Pipeline-wide availability: every conversion tuner CAN resolve a pipeline;
        # the cells whose policy opts out get the inert one regardless of the flags.
        c = _for_mode(mode, schedule, distmatch_driven=True, **_ALL_STEPS)
        assert c == CalibrationPipeline.inert()

    def test_cascaded_cycle_opts_in(self):
        c = _for_mode("ttfs_cycle_based", "cascaded", ttfs_gain_correction=True)
        assert c.gain_cold is True
        assert c.gain_active is True

    def test_cascaded_cycle_default_schedule_opts_in(self):
        # ttfs_cycle_based defaults to the cascaded schedule → conversion-health on.
        c = _for_mode("ttfs_cycle_based", None, ttfs_theta_cotrain=True)
        assert c.theta_cotrain is True

    def test_cascaded_distmatch_follows_driver(self):
        on = _for_mode("ttfs_cycle_based", "cascaded", distmatch_driven=True)
        off = _for_mode("ttfs_cycle_based", "cascaded", distmatch_driven=False)
        assert on.distmatch is True
        assert off.distmatch is False


class TestResolveAliasMatchesForMode:
    """``resolve(synchronized=…)`` must be bit-exact with the policy-keyed path."""

    FLAG_SETS = [
        {},
        dict(ttfs_gain_correction=True),
        dict(ttfs_gain_correction_ramp=True),
        dict(ttfs_gain_correction=True, ttfs_gain_correction_ramp=True),
        dict(ttfs_theta_cotrain=True, ttfs_gain_correction=True),
        dict(ttfs_theta_cotrain=True, ttfs_gain_correction_ramp=True),
        dict(ttfs_boundary_surrogate=True, ttfs_boundary_surrogate_temp=2.0),
        dict(
            ttfs_gain_correction=True,
            ttfs_gain_correction_rule="absolute",
            ttfs_gain_correction_c=2.5,
            ttfs_distmatch_quantile=0.9,
            ttfs_distmatch_bias_iters=3,
            ttfs_distmatch_bias_eta=0.5,
        ),
        _ALL_STEPS,
    ]

    @pytest.mark.parametrize("synchronized", [False, True])
    @pytest.mark.parametrize("distmatch_driven", [False, True])
    @pytest.mark.parametrize("flags", FLAG_SETS)
    def test_alias_equals_policy_keyed(self, synchronized, distmatch_driven, flags):
        schedule = "synchronized" if synchronized else "cascaded"
        via_alias = CalibrationPipeline.resolve(
            dict(flags), synchronized=synchronized, distmatch_driven=distmatch_driven,
        )
        via_mode = _for_mode(
            "ttfs_cycle_based", schedule,
            distmatch_driven=distmatch_driven, **flags,
        )
        assert via_alias == via_mode

    @pytest.mark.parametrize("synchronized", [False, True])
    def test_synchronized_boolean_maps_to_inert(self, synchronized):
        # The synchronized=True boolean key maps to the inert cell; cascaded does not.
        c = CalibrationPipeline.resolve(
            dict(_ALL_STEPS), synchronized=synchronized, distmatch_driven=True,
        )
        if synchronized:
            assert c == CalibrationPipeline.inert()
        else:
            assert c.gain_active is True
