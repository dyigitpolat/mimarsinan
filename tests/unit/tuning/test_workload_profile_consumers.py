"""Consumers read workload registrations; absent profile = the frozen defaults."""

from types import SimpleNamespace

import pytest

from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline
from mimarsinan.tuning.orchestration.install_resolution import chain_gauge_fails
from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan
from mimarsinan.tuning.orchestration.tuning_budget import (
    TuningBudget,
    tuning_budget_from_pipeline,
)
from mimarsinan.tuning.orchestration.tuning_policy import (
    TUNING_POLICY,
    effective_endpoint_floor_lr,
    effective_prefix_stage_lr,
)


class _StubProvider:
    def __init__(self, train=57000, val=3000, bs=128):
        self._train, self._val, self._bs = train, val, bs

    def get_training_set_size(self):
        return self._train

    def get_training_batch_size(self):
        return self._bs

    def get_validation_set_size(self):
        return self._val

    def get_validation_batch_size(self):
        return self._bs


def _pipeline(config=None, provider=None):
    provider = provider or _StubProvider()
    return SimpleNamespace(
        config=dict(config or {}),
        data_provider_factory=SimpleNamespace(create=lambda: provider),
    )


class TestTuningBudgetClamps:
    def test_defaults_keep_the_frozen_cap_and_eval_target(self):
        budget = TuningBudget.from_dataset(
            1_000_000, 100, val_set_size=100_000, val_batch_size=100
        )
        assert budget.max_training_steps == 4000
        assert budget.eval_n_batches == 50  # 5000 target / 100 per batch

    def test_step_cap_epochs_replaces_the_absolute_cap(self):
        budget = TuningBudget.from_dataset(
            1_000_000, 100, val_set_size=100_000, val_batch_size=100,
            tuning_step_cap_epochs=0.5,
        )
        assert budget.max_training_steps == 5000  # 0.5 epoch of 10k steps

    def test_eval_subsample_target_replaces_5000(self):
        budget = TuningBudget.from_dataset(
            1_000_000, 100, val_set_size=100_000, val_batch_size=100,
            eval_subsample_target=20_000,
        )
        assert budget.eval_n_batches == 200

    def test_pipeline_path_reads_the_profile_keys(self):
        pipeline = _pipeline(
            {"eval_subsample_target": 6000, "tuning_step_cap_epochs": 2.0},
            provider=_StubProvider(train=100_000, val=10_000, bs=100),
        )
        budget = tuning_budget_from_pipeline(pipeline)
        # tuning batch = max(16, 100//4) = 25 -> 4000 steps/epoch; cap = 8000.
        assert budget.max_training_steps == 4000  # 1-epoch budget < cap
        assert budget.eval_n_batches == 60  # 6000 / 100 (above the 32-batch floor)

    def test_pipeline_path_defaults_are_bit_identical(self):
        base = tuning_budget_from_pipeline(_pipeline({}))
        keyed = tuning_budget_from_pipeline(
            _pipeline({"eval_subsample_target": None})
        )
        assert base == keyed


class TestEffectiveLrCeilings:
    def test_absent_profile_keeps_the_frozen_policy_values(self):
        assert effective_prefix_stage_lr({}) == TUNING_POLICY.prefix_stage_lr
        assert effective_endpoint_floor_lr({}) == TUNING_POLICY.endpoint_floor_lr

    def test_registered_overrides_win(self):
        assert effective_prefix_stage_lr({"prefix_stage_lr": 5e-4}) == 5e-4
        assert effective_endpoint_floor_lr({"endpoint_floor_lr": 4e-3}) == 4e-3


class TestCalibrationProfileConsumers:
    def test_lif_plan_defaults_unchanged(self):
        plan = LifAdaptationPlan.resolve({"simulation_steps": 8})
        assert plan.distmatch_bias_iters == 10
        assert plan.distmatch_cal_batches == 8

    def test_lif_plan_reads_the_calibration_profile(self):
        plan = LifAdaptationPlan.resolve({
            "simulation_steps": 8,
            "calibration_set_policy": {
                "distmatch_bias_iters": 20, "distmatch_cal_batches": 4,
            },
        })
        assert plan.distmatch_bias_iters == 20
        assert plan.distmatch_cal_batches == 4

    def test_calibration_pipeline_defaults_unchanged(self):
        cal = CalibrationPipeline.resolve(
            {}, synchronized=False, distmatch_driven=True
        )
        assert cal.distmatch_bias_iters == 15
        assert cal.distmatch_cal_batches == 8

    def test_calibration_pipeline_reads_the_profile(self):
        cal = CalibrationPipeline.resolve(
            {"calibration_set_policy": {
                "distmatch_bias_iters": 5, "distmatch_cal_batches": 2,
            }},
            synchronized=False, distmatch_driven=True,
        )
        assert cal.distmatch_bias_iters == 5
        assert cal.distmatch_cal_batches == 2

    def test_explicit_ttfs_key_beats_the_profile(self):
        cal = CalibrationPipeline.resolve(
            {
                "ttfs_distmatch_bias_iters": 7,
                "calibration_set_policy": {"distmatch_bias_iters": 5},
            },
            synchronized=False, distmatch_driven=True,
        )
        assert cal.distmatch_bias_iters == 7


class TestChainGaugeDepthOverride:
    @pytest.mark.parametrize("depth,expected", [(4, False), (5, True), (8, True)])
    def test_default_depth_law_is_six(self, depth, expected):
        assert chain_gauge_fails(
            max_intra_segment_depth=depth, n_segments=1
        ) is expected

    def test_registered_depth_overrides_the_law(self):
        assert chain_gauge_fails(
            max_intra_segment_depth=5, n_segments=1, proven_recovery_depth=9
        ) is False
        assert chain_gauge_fails(
            max_intra_segment_depth=8, n_segments=1, proven_recovery_depth=9
        ) is True
