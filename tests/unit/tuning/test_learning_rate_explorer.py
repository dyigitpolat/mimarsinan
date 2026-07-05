"""Tests for validation-accuracy-based LR range finder."""

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.learning_rate_explorer import (
    LRRangeFinder,
    clone_state_for_trainer,
    restore_state_for_trainer,
)
from conftest import MockDataProviderFactory


def _trainer():
    dpf = MockDataProviderFactory()
    dlf = DataLoaderFactory(dpf, num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 4))

    def loss_fn(mod, x, y):
        return nn.CrossEntropyLoss()(mod(x), y)

    return BasicTrainer(model, "cpu", dlf, loss_fn)


class TestLRRangeFinder:
    def test_restore_state_matches_initial_weights(self):
        tr = _trainer()
        w0 = copy.deepcopy(tr.model.state_dict())
        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: clone_state_for_trainer(tr),
            restore_state=lambda s: restore_state_for_trainer(tr, s),
            lr_min=1e-4,
            lr_max=1e-2,
            num_probes=4,
            steps_per_probe=2,
            validate_fn=lambda: tr.validate_n_batches(1),
        )
        lr = finder.find_best_lr()
        # On a noisy tiny fixture the sweep may legitimately refuse (None).
        assert lr is None or 1e-4 <= lr <= 1e-2
        w1 = tr.model.state_dict()
        for k in w0:
            assert torch.allclose(w0[k].cpu(), w1[k].cpu())

    def test_prefers_lr_with_highest_validation_accuracy(self):
        """The finder should pick the LR whose probe gives the highest
        validation accuracy, NOT the lowest training-batch loss."""

        class MockTrainer:
            def __init__(self):
                self._current_lr = 0.0

            def train_n_steps(self, lr, steps, **kwargs):
                self._current_lr = lr

        tr = MockTrainer()

        def mock_validate():
            lr = tr._current_lr
            if lr < 0.001:
                return 0.70
            elif lr < 0.01:
                return 0.85
            else:
                return 0.10

        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-5,
            lr_max=1e-1,
            num_probes=10,
            steps_per_probe=5,
            validate_fn=mock_validate,
        )
        lr = finder.find_best_lr()
        assert lr < 0.01, (
            f"Should pick LR below the destructive region, got {lr}"
        )
        assert lr > 1e-5, f"Should not pick the absolute minimum, got {lr}"

    def test_rejects_destructive_high_lr(self):
        """High LRs that destroy validation accuracy must be avoided, even if
        they would minimise training-batch loss (the old failure mode)."""

        class MockTrainer:
            def __init__(self):
                self._current_lr = 0.0

            def train_n_steps(self, lr, steps, **kwargs):
                self._current_lr = lr

        tr = MockTrainer()

        def mock_validate():
            lr = tr._current_lr
            if lr > 0.01:
                return 0.05
            return 0.75

        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-5,
            lr_max=1e-1,
            num_probes=10,
            steps_per_probe=5,
            validate_fn=mock_validate,
        )
        lr = finder.find_best_lr()
        assert lr <= 0.01, f"Should avoid destructive high LRs, got {lr}"

    def test_num_probes_and_steps_per_probe(self):
        """Verify exact number of train_n_steps calls."""
        calls = []

        class CountingTrainer:
            def train_n_steps(self, lr, steps, **kwargs):
                calls.append((lr, steps))

        tr = CountingTrainer()
        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-4,
            lr_max=1e-1,
            num_probes=5,
            steps_per_probe=3,
            validate_fn=lambda: 0.5,
        )
        finder.find_best_lr()
        assert len(calls) == 5
        for _, steps in calls:
            assert steps == 3

    def test_all_destructive_sweep_refuses(self):
        """When EVERY candidate LR is destructive the finder must REFUSE
        (return None) instead of handing back the least-bad destructive LR
        (W2 fix C: the t0_18 explorer picked 1.12e-2 and recovery committed
        the wreck)."""

        class MockTrainer:
            def __init__(self):
                self._current_lr = 0.0

            def train_n_steps(self, lr, steps, **kwargs):
                self._current_lr = lr

        tr = MockTrainer()

        def mock_validate():
            lr = tr._current_lr
            acc = max(0.01, 0.30 - lr * 100)
            return acc

        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-3,
            lr_max=1e-1,
            num_probes=5,
            steps_per_probe=3,
            validate_fn=mock_validate,
        )
        assert finder.find_best_lr() is None

    def test_all_probes_catastrophic_refuses(self):
        """Baseline 0.80, every probe 0.05: no LR is handed to recovery."""
        call_idx = [0]

        class MockTrainer:
            def train_n_steps(self, lr, steps, **kwargs):
                pass

        tr = MockTrainer()

        def mock_validate():
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                return 0.80  # baseline
            return 0.05  # all probes catastrophic

        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-5,
            lr_max=1e-1,
            num_probes=5,
            steps_per_probe=3,
            validate_fn=mock_validate,
        )
        assert finder.find_best_lr() is None

    def test_coarse_path_all_destructive_refuses(self):
        """The coarse (loss-slope top-K) path shares the refusal contract."""
        call_idx = [0]

        class MockTrainer:
            def train_n_steps(self, lr, steps, **kwargs):
                pass

        tr = MockTrainer()

        def mock_validate():
            idx = call_idx[0]
            call_idx[0] += 1
            return 0.80 if idx == 0 else 0.05

        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: None,
            restore_state=lambda s: None,
            lr_min=1e-5,
            lr_max=1e-1,
            num_probes=5,
            steps_per_probe=3,
            validate_fn=mock_validate,
            coarse_signal=lambda: 1.0,
            coarse_top_k=2,
        )
        assert finder.find_best_lr() is None
