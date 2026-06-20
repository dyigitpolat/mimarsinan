"""EF3 — the (firing × sync) CalibrationPipeline (E3) + ConversionPolicy keystone
(E4) are CONSULTED in the live tuner-selection path, default-off ⇒ byte-identical.

Before EF3, ``contract.calibration_pipeline`` / ``contract.conversion_policy`` (and
the ``DeploymentPlan`` mirrors) existed and were unit-tested, but were consulted
NOWHERE in the live pipeline: ``TtfsAdaptationPlan.resolve`` built its calibration
via ``CalibrationPipeline.resolve(...)`` raw and never touched the conversion-policy
keystone at all. EF3 wires the seam — the live TTFS tuner asks the CONTRACT for its
calibration steps and (behind the default-off ``conversion_policy`` flag) consults
``ConversionPolicy.propose→characterize→escalate``.

These tests lock that the seam is CONSULTED but INERT by default:

* the plan resolved through the injected contract ``calibration_resolver`` is
  byte-identical to the raw ``CalibrationPipeline.resolve`` it replaces (the contract
  keys on the same (firing × sync) cell);
* an inert (default-off) ``ConversionDecision`` leaves the optimization-driver axis
  exactly as the pipeline-wide ``optimization_driver`` resolved it (no override);
* the LIVE ``TTFSCycleAdaptationTuner._configure`` actually calls both contract
  seams, yet the resolved ``_adaptation_plan`` / ``_calibration`` / fast-ladder
  decision is identical to a tuner built without the wiring observable (goldens
  unchanged — the seam runs the SAME numbers).

The keystone is exposed and consulted, NOT enabled (that is Fix B — a default flip
+ real probes). The escalation BRANCH (enabled + a rejecting characterizer forcing
the controller) is locked at the plan level so Fix B is later just the flip.
"""

from __future__ import annotations

import pytest

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline
from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_CONTROLLER,
    OPTIMIZATION_DRIVER_FAST,
    Characterizer,
    CharacterizationResult,
    ConversionDecision,
    ConversionPolicy,
)
from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
)


# ── (A) the plan-level seam: injected calibration_resolver / conversion_decision ──


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


class TestPlanConversionDecisionSeam:
    """An INERT (default-off) ``ConversionDecision`` does not move the axis; an
    ENABLED escalating decision forces the controller gate (the Fix-B branch)."""

    def test_inert_decision_leaves_the_axis_untouched(self):
        # default-off decision (enabled=False) — the axis stays whatever the
        # pipeline-wide `optimization_driver` resolved (here fast via the flag).
        inert = _inert_decision()
        raw = TtfsAdaptationPlan.resolve(
            {"ttfs_blend_fast": True}, synchronized=False,
        )
        wired = TtfsAdaptationPlan.resolve(
            {"ttfs_blend_fast": True},
            synchronized=False,
            conversion_decision=inert,
        )
        assert raw.proxy_fast is True and raw.fast_ladder_enabled is True
        assert wired == raw, "an inert decision must not change the resolution"

    def test_enabled_controller_decision_vetoes_the_fast_path(self):
        # The enabled propose→confirm→escalate outcome (driver=controller) is the
        # GATE over the fast fork — it vetoes a fast flag exactly like an explicit
        # `optimization_driver=controller` axis (the safe escalation path).
        decision = ConversionDecision(
            enabled=True,
            recipe=_inert_decision().recipe,
            driver=OPTIMIZATION_DRIVER_CONTROLLER,
            characterized=True,
            escalated=True,
            escalation_reason="cold cascade dead",
        )
        wired = TtfsAdaptationPlan.resolve(
            {"ttfs_blend_fast": True},
            synchronized=False,
            optimization_driver=OPTIMIZATION_DRIVER_FAST,
            conversion_decision=decision,
        )
        assert wired.proxy_fast is False
        assert wired.fast_ladder_enabled is False
        assert wired.driver.controller is True

    def test_explicit_axis_still_wins_over_a_missing_decision(self):
        # No decision (None, the default) ⇒ the axis governs exactly as pre-EF3.
        wired = TtfsAdaptationPlan.resolve(
            {"ttfs_blend_fast": True},
            synchronized=False,
            optimization_driver=OPTIMIZATION_DRIVER_CONTROLLER,
        )
        assert wired.fast_ladder_enabled is False


def _inert_decision() -> ConversionDecision:
    return ConversionPolicy.resolve(
        {}, mode_policy=_cascade_policy(),
    )


def _cascade_policy():
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode

    return policy_for_spiking_mode("ttfs_cycle_based", "cascaded")


# ── (B) the LIVE tuner consults the contract seams, inert by default ──────────


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
    tuner = TTFSCycleAdaptationTuner(
        pipeline,
        model=make_tiny_supermodel(),
        target_accuracy=0.5,
        lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )
    return tuner


class TestLiveTunerConsultsTheContractSeams:
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

    def test_conversion_policy_seam_is_consulted(self, tmp_path, monkeypatch):
        from mimarsinan.chip_simulation import deployment_contract as dc

        calls = []
        original = dc.SpikingDeploymentContract.conversion_policy

        def spy(self, config, *, model=None, characterizer=None, core=None):
            decision = original(
                self, config, model=model, characterizer=characterizer, core=core,
            )
            calls.append(decision)
            return decision

        monkeypatch.setattr(
            dc.SpikingDeploymentContract, "conversion_policy", spy,
        )
        tuner = _make_tuner(tmp_path)
        assert calls, "the live _configure must consult contract.conversion_policy"
        # DEFAULT-OFF ⇒ inert: the decision names the CURRENT behavior, nothing runs.
        decision = calls[-1]
        assert decision.enabled is False
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
        assert decision.characterized is False
        assert decision.escalated is False
        # the tuner records the inert decision for provenance but does not act on it.
        assert tuner._conversion_decision is decision

    def test_default_off_does_not_run_the_characterizer(self, tmp_path, monkeypatch):
        # Even a characterizer that would REJECT is never consulted while default-off
        # (the conversion_policy flag is absent) — so behavior is byte-identical.
        from mimarsinan.tuning.tuners import ttfs_cycle_adaptation_tuner as mod

        class _Reject(Characterizer):
            calls = 0

            def characterize(self, *, model, recipe, context=None):
                type(self).calls += 1
                return CharacterizationResult(matches=False, reason="rejected")

        monkeypatch.setattr(
            mod, "_conversion_characterizer_for", lambda tuner: _Reject(),
            raising=False,
        )
        _make_tuner(tmp_path)
        assert _Reject.calls == 0


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
