"""Tests for TuningBudget and tuning_budget_from_pipeline."""

import pytest

from mimarsinan.tuning.tuning_budget import TuningBudget, tuning_budget_from_pipeline
from conftest import MockDataProviderFactory, default_config, MockPipeline


class TestTuningBudget:
    def test_from_dataset_sqrt_scaling(self):
        b = TuningBudget.from_dataset(dataset_size=50000, batch_size=100, budget_scale=1.0)
        # steps_per_epoch = 500; check_interval = sqrt(500) ~ 22
        assert b.check_interval == 22
        assert b.max_training_steps == 500 * 3  # 3x budget
        assert b.validation_steps >= 1

    def test_from_dataset_budget_scale(self):
        b1 = TuningBudget.from_dataset(10000, 100, budget_scale=1.0)
        b2 = TuningBudget.from_dataset(10000, 100, budget_scale=2.0)
        assert b2.max_training_steps >= b1.max_training_steps
        assert b1.check_interval == b2.check_interval

    def test_lr_steps_per_probe_matches_check_interval(self):
        """Each LR probe trains for a full check_interval to reliably detect instability."""
        import math
        b = TuningBudget.from_dataset(50000, 100, budget_scale=1.0)
        assert b.lr_num_probes == max(2, int(math.sqrt(b.check_interval)))
        assert b.lr_steps_per_probe == b.check_interval

    def test_tolerance_probe_steps_equals_check_interval(self):
        b = TuningBudget.from_dataset(50000, 100, budget_scale=1.0)
        assert b.tolerance_probe_steps == b.check_interval

    def test_no_hardcoded_lr_bounds(self):
        """LR fields must scale with dataset, not be clamped to fixed bounds."""
        small = TuningBudget.from_dataset(100, 10, budget_scale=1.0)
        large = TuningBudget.from_dataset(1_000_000, 100, budget_scale=1.0)
        assert small.lr_num_probes < large.lr_num_probes
        assert small.lr_steps_per_probe < large.lr_steps_per_probe

    def test_eval_n_batches_defaults_to_validation_steps_without_val_info(self):
        b = TuningBudget.from_dataset(50000, 100, budget_scale=1.0)
        assert b.eval_n_batches == b.validation_steps

    def test_eval_n_batches_with_small_val_set(self):
        b = TuningBudget.from_dataset(
            50000, 100,
            val_set_size=3000, val_batch_size=128,
        )
        total_val_batches = 3000 // 128  # 23
        assert b.eval_n_batches == max(b.validation_steps, total_val_batches)
        assert b.eval_n_batches >= b.validation_steps

    def test_eval_n_batches_uses_full_val_set(self):
        b = TuningBudget.from_dataset(
            1_000_000, 16,
            val_set_size=64000, val_batch_size=16,
        )
        assert b.eval_n_batches == 64000 // 16  # full validation set

    def test_tuning_budget_from_pipeline_uses_config_scale(self):
        factory = MockDataProviderFactory()
        cfg = default_config()
        cfg["tuning_budget_scale"] = 2.0
        pipe = MockPipeline(config=cfg, data_provider_factory=factory)
        b = tuning_budget_from_pipeline(pipe)
        assert isinstance(b, TuningBudget)
        assert b.max_training_steps >= 1
        assert b.check_interval >= 1

    def test_from_data_provider_populates_eval_n_batches(self):
        factory = MockDataProviderFactory()
        dp = factory.create()
        b = TuningBudget.from_data_provider(dp)
        assert b.eval_n_batches >= b.validation_steps

    def test_3x_budget_multiplier(self):
        b = TuningBudget.from_dataset(10000, 100, budget_scale=1.0)
        spe = 10000 // 100  # 100
        assert b.max_training_steps == 3 * spe
