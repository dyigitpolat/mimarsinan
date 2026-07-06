"""EF3 — the (firing × sync) CalibrationPipeline (E3) is CONSULTED in the live
tuner-selection path, default-off ⇒ byte-identical.

Before EF3, ``contract.calibration_pipeline`` (and the ``DeploymentPlan`` mirror)
existed and was unit-tested, but was consulted NOWHERE in the live pipeline:
``TtfsAdaptationPlan.resolve`` built its calibration via ``CalibrationPipeline.resolve``
raw. EF3 wires the seam — the live TTFS tuner asks the CONTRACT for its calibration
steps through an injected ``calibration_resolver``.

These tests lock that the seam is CONSULTED but INERT by default:

* the plan resolved through the injected contract ``calibration_resolver`` is
  byte-identical to the raw ``CalibrationPipeline.resolve`` it replaces (the contract
  keys on the same (firing × sync) cell);
* an explicit ``optimization_driver=controller`` axis still vetoes the fast fork
  (the axis gate is independent of the calibration seam);
* the LIVE ``TTFSCycleAdaptationTuner._configure`` actually calls the contract seam,
  yet the resolved ``_adaptation_plan`` / ``_calibration`` / fast-ladder decision is
  identical to a tuner built without the wiring observable (goldens unchanged — the
  seam runs the SAME numbers).

(The conversion-policy keystone is no longer a default-off consultation seam — it is
the deterministic ``ConversionPolicy.derive`` SSOT, covered by
``test_conversion_policy_derive.py``.)
"""

from __future__ import annotations

import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline
from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_CONTROLLER,
)
from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
)


# ── (A) the plan-level seam: injected calibration_resolver ────────────────────


class TestPlanCalibrationResolverSeam:
    """``TtfsAdaptationPlan.resolve`` resolves calibration through an INJECTED
    resolver (the contract seam) when one is supplied; the raw resolution is the
    default for direct callers, and the two are byte-identical for any config."""

    @pytest.mark.parametrize(
        "flags",
        [
            {},
            {"ttfs_gain_correction": True},
            {"ttfs_theta_cotrain": True},
            {"ttfs_genuine_blend_ramp": True},
            {"ttfs_gain_correction_ramp": True, "ttfs_theta_cotrain": True},
            {"ttfs_boundary_surrogate": True, "ttfs_boundary_surrogate_temp": 2.0},
        ],
    )
    @pytest.mark.parametrize("synchronized", [False, True])
    def test_injected_resolver_matches_raw_resolution(self, flags, synchronized):
        cfg = dict(flags)
        raw = TtfsAdaptationPlan.resolve(cfg, synchronized=synchronized)

        calls = []

        def resolver(config, *, synchronized, distmatch_driven):
            calls.append((config, synchronized, distmatch_driven))
            return CalibrationPipeline.resolve(
                config, synchronized=synchronized, distmatch_driven=distmatch_driven,
            )

        wired = TtfsAdaptationPlan.resolve(
            cfg, synchronized=synchronized, calibration_resolver=resolver,
        )
        assert calls, "the injected resolver must be consulted"
        # The seam is CONSULTED yet INERT: same calibration, byte-identical plan.
        assert wired.calibration == raw.calibration
        assert wired == raw

    def test_resolver_receives_the_distmatch_driven_decision(self):
        # distmatch is owned by the genuine-blend ramp; the plan computes it and
        # passes it THROUGH to the resolver so the contract keys distmatch correctly.
        seen = {}

        def resolver(config, *, synchronized, distmatch_driven):
            seen["distmatch_driven"] = distmatch_driven
            seen["synchronized"] = synchronized
            return CalibrationPipeline.resolve(
                config, synchronized=synchronized, distmatch_driven=distmatch_driven,
            )

        TtfsAdaptationPlan.resolve(
            {"ttfs_genuine_blend_ramp": True},
            synchronized=False,
            calibration_resolver=resolver,
        )
        assert seen == {"distmatch_driven": True, "synchronized": False}


class TestPlanAxisGate:
    """The explicit ``optimization_driver`` axis gates the fast fork independently of
    the calibration seam (an explicit ``controller`` disables the fast ladder)."""

    def test_explicit_controller_axis_disables_the_fast_ladder(self):
        wired = TtfsAdaptationPlan.resolve(
            {"ttfs_blend_fast": True},
            synchronized=False,
            optimization_driver=OPTIMIZATION_DRIVER_CONTROLLER,
        )
        assert wired.fast_ladder_enabled is False


# ── (B) the LIVE tuner consults the contract seam, inert by default ───────────


def _make_tuner(tmp_path, *, schedule="cascaded", **extra):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    cfg.update(extra)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    # Gain flags require intra-segment depth >= 2 (boundary-dominated guard).
    deep = bool(extra.get("ttfs_gain_correction") or extra.get("ttfs_gain_correction_ramp"))
    tuner = TTFSCycleAdaptationTuner(
        pipeline,
        model=make_tiny_supermodel(hidden_layers=2 if deep else 1),
        target_accuracy=0.5,
        lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )
    return tuner


class TestLiveTunerConsultsTheContractSeam:
    def test_calibration_seam_is_consulted(self, tmp_path, monkeypatch):
        from mimarsinan.chip_simulation import deployment_contract as dc

        calls = []
        original = dc.SpikingDeploymentContract.calibration_pipeline

        def spy(self, config, *, distmatch_driven=False, core=None):
            calls.append(distmatch_driven)
            return original(self, config, distmatch_driven=distmatch_driven, core=core)

        monkeypatch.setattr(
            dc.SpikingDeploymentContract, "calibration_pipeline", spy,
        )
        _make_tuner(tmp_path)
        assert calls, "the live _configure must consult contract.calibration_pipeline"


class TestLiveWiringIsByteIdentical:
    """The wired tuner resolves the SAME plan / calibration / fast-ladder decision
    as the raw resolution — the seam is consulted but runs the same numbers."""

    @pytest.mark.parametrize("schedule", ["cascaded", "synchronized"])
    def test_plan_matches_raw_resolution(self, tmp_path, schedule):
        tuner = _make_tuner(tmp_path, schedule=schedule)
        synchronized = schedule != "cascaded"
        raw = TtfsAdaptationPlan.resolve(
            tuner.pipeline.config, synchronized=synchronized,
        )
        assert tuner._adaptation_plan.calibration == raw.calibration
        assert tuner._adaptation_plan == raw
        assert tuner._calibration == raw.calibration

    def test_cascaded_calibration_flags_still_resolve(self, tmp_path):
        # A cascaded cell that opts into conversion-health still resolves its steps
        # through the contract seam — the wiring did not drop the calibration.
        tuner = _make_tuner(
            tmp_path,
            ttfs_gain_correction=True,
            ttfs_theta_cotrain=True,
        )
        assert tuner._calibration.gain_cold is True
        assert tuner._calibration.theta_cotrain is True

    def test_synchronized_calibration_is_inert(self, tmp_path):
        # The synchronized cell gets the inert pipeline through the seam (byte-identical
        # to the historical synchronized path) even with the ttfs_* flags set.
        tuner = _make_tuner(
            tmp_path,
            schedule="synchronized",
            ttfs_gain_correction=True,
            ttfs_theta_cotrain=True,
            ttfs_boundary_surrogate=True,
        )
        assert tuner._calibration == CalibrationPipeline.inert()
