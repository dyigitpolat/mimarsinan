"""U3 — R2's two-residual S decouple: train the genuine cascade at train_S, deploy
at deploy_S, gated by the ConversionPolicy keystone (default-off ⇒ byte-identical).

The genuine fine-tune is S-NEGATIVE (train-AT-deploy-S collapses the deep cascade), so
R2 splits the TRAINING-forward S (low) from the deployed S (high). The decouple is
driven by the keystone's ``train_s_hint`` threaded through ``TtfsAdaptationPlan``; until
the policy is enabled the hint is ``None`` ⇒ ``_train_T == _T`` ⇒ byte-identical.

Locks:
* the plan carries ``train_s_hint`` only from an ENABLED decision (inert by default);
* the live tuner sets ``_train_T`` from the plan, ``_T`` stays deploy_S;
* the RAMP (training) forward is built at train_S, the FINALIZE (deploy) forward at
  deploy_S — they coincide (byte-identical construction) when train_S == deploy_S.
"""

from __future__ import annotations

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.conversion_policy import (
    ConversionDecision,
    ConversionRecipe,
    OPTIMIZATION_DRIVER_CONTROLLER,
)
from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    StaircaseSteRamp,
    TTFSCycleAdaptationTuner,
)


# ── (A) the plan-level seam: train_s_hint only from an ENABLED decision ────────


def _enabled_decision(train_s_hint):
    recipe = ConversionRecipe(
        name="ttfs_cycle_based/cascaded",
        driver=OPTIMIZATION_DRIVER_CONTROLLER,
        expects_conversion_health=True,
        train_s_hint=train_s_hint,
        deploy_s_hint=32,
    )
    return ConversionDecision(
        enabled=True,
        recipe=recipe,
        driver=OPTIMIZATION_DRIVER_CONTROLLER,
        characterized=True,
        escalated=False,
    )


class TestPlanThreadsTrainSHint:
    def test_default_plan_has_no_train_s_hint(self):
        # No decision (the default-off path) ⇒ no decouple ⇒ byte-identical.
        assert TtfsAdaptationPlan.resolve({}, synchronized=False).train_s_hint is None

    def test_inert_decision_does_not_set_train_s_hint(self):
        from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            policy_for_spiking_mode,
        )

        inert = ConversionPolicy.resolve(
            {}, mode_policy=policy_for_spiking_mode("ttfs_cycle_based", "cascaded"),
        )
        assert inert.enabled is False
        plan = TtfsAdaptationPlan.resolve(
            {}, synchronized=False, conversion_decision=inert,
        )
        assert plan.train_s_hint is None

    def test_enabled_decision_threads_the_train_s_hint(self):
        plan = TtfsAdaptationPlan.resolve(
            {}, synchronized=False, conversion_decision=_enabled_decision(16),
        )
        assert plan.train_s_hint == 16

    def test_enabled_decision_without_a_hint_stays_none(self):
        plan = TtfsAdaptationPlan.resolve(
            {}, synchronized=False, conversion_decision=_enabled_decision(None),
        )
        assert plan.train_s_hint is None


# ── (B) the live tuner: _train_T from the plan, _T stays deploy_S ──────────────


def _make_tuner(tmp_path, *, schedule="cascaded", simulation_steps=32, **extra):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = simulation_steps
    cfg.update(extra)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return TTFSCycleAdaptationTuner(
        pipeline,
        model=make_tiny_supermodel(),
        target_accuracy=0.5,
        lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


class TestDefaultIsByteIdentical:
    def test_train_T_equals_deploy_T_by_default(self, tmp_path):
        # Default-off ⇒ no train_s_hint ⇒ train_S == deploy_S == simulation_steps.
        tuner = _make_tuner(tmp_path, simulation_steps=32)
        assert tuner._T == 32
        assert tuner._train_T == 32

    def test_ramp_and_finalize_forwards_coincide_by_default(self, tmp_path):
        # When train_S == deploy_S the RAMP-forward and FINALIZE-forward are built at
        # the SAME sim-length (byte-identical construction).
        tuner = _make_tuner(tmp_path, simulation_steps=16)
        model = tuner.model
        assert tuner._ramp_forward_for(model).T == tuner._finalize_forward_for(model).T
        assert tuner._ramp_forward_for(model).T == 16


class TestDecoupleTrainsAtTrainSDeploysAtDeployS:
    def test_enabled_keystone_sets_train_T_below_deploy_T(self, tmp_path):
        # conversion_policy enabled (always-matches) ⇒ the cascaded recipe's
        # train_s_hint=16 flows through the plan; deploy_S stays simulation_steps=32.
        tuner = _make_tuner(tmp_path, simulation_steps=32, conversion_policy=True)
        assert tuner._adaptation_plan.train_s_hint == 16
        assert tuner._T == 32, "deploy_S unchanged (parity/mapping read this)"
        assert tuner._train_T == 16, "training forward decoupled to train_S"

    def test_ramp_forward_builds_at_train_S_finalize_at_deploy_S(self, tmp_path):
        # The decisive lock: the RAMP (training) genuine cascade is built at train_S
        # while the FINALIZE (deploy) genuine cascade is built at deploy_S.
        tuner = _make_tuner(tmp_path, simulation_steps=32, conversion_policy=True)
        model = tuner.model
        assert tuner._ramp_forward_for(model).T == 16, "train forward at train_S"
        assert tuner._finalize_forward_for(model).T == 32, "deploy forward at deploy_S"

    def test_staircase_ste_ramp_routes_through_train_S(self, tmp_path):
        # The StaircaseSteRamp/GenuineAnnealedRamp install the TRAINING forward via
        # ramp_forward → _ramp_forward_for, so the STE recipe trains at train_S.
        tuner = _make_tuner(tmp_path, simulation_steps=32, conversion_policy=True)
        fwd = StaircaseSteRamp().ramp_forward(tuner, tuner.model)
        assert fwd.T == 16

    def test_synchronized_has_no_separate_forward(self, tmp_path):
        # Synchronized's deployment IS the class analytical staircase — no segment-spike
        # forward to decouple (both builders return None regardless of train_S).
        tuner = _make_tuner(
            tmp_path, schedule="synchronized", simulation_steps=32,
            conversion_policy=True,
        )
        assert tuner._ramp_forward_for(tuner.model) is None
        assert tuner._finalize_forward_for(tuner.model) is None
