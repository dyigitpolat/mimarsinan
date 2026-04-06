"""Tests for TuningBudget and tuning_budget_from_pipeline."""

import math
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
        assert b.eval_n_batches >= b.validation_steps

    def test_eval_n_batches_capped_statistically(self):
        b = TuningBudget.from_dataset(
            1_000_000, 16,
            val_set_size=64000, val_batch_size=16,
        )
        total_val_batches = 64000 // 16  # 4000
        # default d=0.05: min_eval_samples=max(256,1600)=1600, stat_batches=100
        assert b.eval_n_batches == 100
        assert b.eval_n_batches < total_val_batches

    def test_eval_n_batches_scales_with_tolerance(self):
        tight = TuningBudget.from_dataset(
            1_000_000, 16,
            val_set_size=64000, val_batch_size=16,
            degradation_tolerance=0.01,
        )
        loose = TuningBudget.from_dataset(
            1_000_000, 16,
            val_set_size=64000, val_batch_size=16,
            degradation_tolerance=0.10,
        )
        assert tight.eval_n_batches > loose.eval_n_batches

    def test_eval_n_batches_never_exceeds_full_val_set(self):
        b = TuningBudget.from_dataset(
            1_000_000, 16,
            val_set_size=64000, val_batch_size=16,
            degradation_tolerance=0.001,
        )
        total_val_batches = 64000 // 16
        assert b.eval_n_batches <= total_val_batches

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

    def test_eval_sample_count_with_val_info(self):
        b = TuningBudget.from_dataset(
            50000, 100,
            val_set_size=10000, val_batch_size=128,
        )
        assert b.eval_sample_count == b.eval_n_batches * 128

    def test_eval_sample_count_without_val_info(self):
        b = TuningBudget.from_dataset(50000, 100, budget_scale=1.0)
        assert b.eval_sample_count == b.eval_n_batches * 100

    def test_accuracy_se_formula(self):
        b = TuningBudget.from_dataset(
            50000, 100,
            val_set_size=10000, val_batch_size=128,
        )
        expected = 0.5 / math.sqrt(b.eval_sample_count)
        assert b.accuracy_se() == pytest.approx(expected)

    def test_accuracy_se_mnist_like(self):
        """MNIST-like: ~10k eval samples -> SE ~0.005."""
        b = TuningBudget.from_dataset(
            60000, 128,
            val_set_size=10000, val_batch_size=128,
        )
        assert 0.003 < b.accuracy_se() < 0.01

    def test_accuracy_se_imagenet_like(self):
        """ImageNet-like with tight tolerance: many eval samples -> small SE."""
        b = TuningBudget.from_dataset(
            1_200_000, 128,
            val_set_size=50000, val_batch_size=128,
            degradation_tolerance=0.01,
        )
        assert 0.001 < b.accuracy_se() < 0.005

    def test_accuracy_se_default_nonzero(self):
        """accuracy_se() is positive even with minimal eval_sample_count."""
        b = TuningBudget(
            max_training_steps=100, check_interval=10,
            validation_steps=1, eval_n_batches=1,
            lr_steps_per_probe=10, lr_num_probes=2,
            tolerance_probe_steps=10, eval_sample_count=0,
        )
        assert b.accuracy_se() == 0.5  # 0.5 / sqrt(1)


class TestLRRangeFinderHeuristic:
    """The LR finder picks the largest non-destructive LR."""

    def test_picks_largest_non_destructive(self):
        from mimarsinan.tuning.learning_rate_explorer import LRRangeFinder

        current_acc = [0.90]
        accs_by_lr = {0.001: 0.90, 0.01: 0.89, 0.1: 0.50}

        def fake_train(self_unused, lr, steps, **kwargs):
            rounded = min(accs_by_lr, key=lambda k: abs(k - lr))
            current_acc[0] = accs_by_lr[rounded]

        finder = LRRangeFinder(
            trainer=type("T", (), {"train_n_steps": fake_train})(),
            clone_state=lambda: current_acc[0],
            restore_state=lambda s: current_acc.__setitem__(0, s),
            lr_min=0.001,
            lr_max=0.1,
            num_probes=3,
            steps_per_probe=1,
            validate_fn=lambda: current_acc[0],
            margin=0.02,
        )
        best = finder.find_best_lr()
        assert best == pytest.approx(0.01)

    def test_fallback_to_best_acc_when_all_destructive(self):
        from mimarsinan.tuning.learning_rate_explorer import LRRangeFinder

        current_acc = [0.90]
        probe_idx = [0]

        def fake_train(self_unused, lr, steps, **kwargs):
            current_acc[0] = 0.80 - probe_idx[0] * 0.05
            probe_idx[0] += 1

        def fake_restore(s):
            current_acc[0] = s

        finder = LRRangeFinder(
            trainer=type("T", (), {"train_n_steps": fake_train})(),
            clone_state=lambda: current_acc[0],
            restore_state=fake_restore,
            lr_min=0.001,
            lr_max=0.1,
            num_probes=3,
            steps_per_probe=1,
            validate_fn=lambda: current_acc[0],
            margin=0.02,
        )
        best = finder.find_best_lr()
        assert best == pytest.approx(0.001)

    def test_anchor_lr_narrows_range(self):
        from mimarsinan.tuning.learning_rate_explorer import find_lr_range_for_trainer
        from unittest.mock import MagicMock

        trainer = MagicMock()
        trainer.train_n_steps = MagicMock()
        pipeline = MagicMock()
        pipeline.config = {"lr_range_min": 1e-5, "lr_range_max": 1e-1}

        budget = MagicMock()
        budget.lr_num_probes = 3
        budget.lr_steps_per_probe = 1
        budget.max_lr_exploration_steps = 100
        budget.accuracy_se.return_value = 0.005

        call_count = [0]
        def fake_validate():
            call_count[0] += 1
            return 0.90

        probed_lrs = []
        orig_train = trainer.train_n_steps
        def capture_lr(lr, steps, **kwargs):
            probed_lrs.append(lr)
        trainer.train_n_steps = capture_lr

        find_lr_range_for_trainer(
            trainer, pipeline, budget,
            validate_fn=fake_validate,
            anchor_lr=0.001,
        )

        for lr in probed_lrs:
            assert lr >= 0.0001 - 1e-9
            assert lr <= 0.01 + 1e-9
