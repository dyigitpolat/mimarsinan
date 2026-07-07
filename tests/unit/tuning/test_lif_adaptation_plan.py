"""Contract-driven resolution of the LIF adaptation flag-thicket (one site).

``LifAdaptationPlan`` mirrors ``TtfsAdaptationPlan``: every knob the LIF tuner
consumes resolves here once — the fast-ladder shape, the realizable T-anneal
decision, the distmatch bundle, theta co-training, and the endpoint budget —
so the tuner's ``_configure`` stays thin and the effective defaults have one
home instead of scattered ``config.get`` literals.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan


def _minimal_cfg(**overrides):
    cfg = {"spiking_mode": "lif", "simulation_steps": 4}
    cfg.update(overrides)
    return cfg


class TestResolveDefaults:
    def test_effective_defaults_match_the_recipe_provenance(self):
        plan = LifAdaptationPlan.resolve(_minimal_cfg())
        assert plan.cycle_accurate is False
        assert plan.blend_fast_rates == [0.25, 0.5, 0.75, 1.0]
        assert plan.blend_fast_steps_per_rate == 120
        assert plan.blend_fast_lr_eta_min == 0.1
        assert plan.tanneal is False
        assert plan.endpoint_recovery_steps == 0
        assert plan.distmatch is False
        assert plan.distmatch_bias_iters == 10
        assert plan.distmatch_bias_eta == 0.5
        assert plan.distmatch_cal_batches == 8
        assert plan.theta_cotrain is False
        assert plan.simulation_steps == 4

    def test_config_values_resolve_with_coercion(self):
        plan = LifAdaptationPlan.resolve(_minimal_cfg(
            cycle_accurate_lif_forward=1,
            lif_blend_fast_rates=["0.5", 1.0],
            lif_blend_fast_steps_per_rate="2",
            lif_distmatch=True,
            lif_distmatch_bias_iters="4",
            lif_theta_cotrain=True,
            endpoint_recovery_steps="600",
        ))
        assert plan.cycle_accurate is True
        assert plan.blend_fast_rates == [0.5, 1.0]
        assert plan.blend_fast_steps_per_rate == 2
        assert plan.distmatch is True and plan.distmatch_bias_iters == 4
        assert plan.theta_cotrain is True
        assert plan.endpoint_recovery_steps == 600


class TestTannealDecision:
    def test_knob_off_is_none(self):
        assert LifAdaptationPlan.resolve(_minimal_cfg()).tanneal_schedule(
            [0.5, 1.0]
        ) is None
        plan = LifAdaptationPlan.resolve(_minimal_cfg(lif_tanneal=False))
        assert plan.tanneal_schedule([0.5, 1.0]) is None

    def test_knob_on_lif_derives_from_simulation_steps(self):
        plan = LifAdaptationPlan.resolve(_minimal_cfg(lif_tanneal=True))
        schedule = plan.tanneal_schedule([0.5, 1.0])
        assert schedule is not None
        assert schedule.target_T == 4
        assert schedule.ladder_rates == (0.5, 1.0)

    def test_knob_on_non_lif_modes_are_none(self):
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            plan = LifAdaptationPlan.resolve(_minimal_cfg(
                spiking_mode=mode, lif_tanneal=True,
            ))
            assert plan.tanneal is False
            assert plan.tanneal_schedule([0.5, 1.0]) is None
