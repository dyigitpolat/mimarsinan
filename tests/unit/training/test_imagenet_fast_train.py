"""Tests for the fast ResNet-50 ImageNet-from-scratch recipe.

Tiny + fast + NO ImageNet. Covers:
  * one-cycle LR schedule shape (linear warmup to peak, cosine decay floor)
  * progressive-resize schedule shape (monotone non-decreasing, hits final)
  * label-smoothing CE numerics vs the closed-form reference
  * the dataloader factory's ffcv-vs-torchvision SELECTION (import mocked)
  * a train-step on a TINY synthetic batch (loss decreases over a few steps)
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# one-cycle LR schedule shape
# --------------------------------------------------------------------------- #
class TestOneCycleSchedule:
    def test_warmup_starts_low_peaks_then_decays(self):
        from mimarsinan.training.imagenet_fast_train import one_cycle_lr_schedule

        total = 100
        warmup = 20
        peak = 1.0
        lrs = [
            one_cycle_lr_schedule(step, total_steps=total, warmup_steps=warmup,
                                  peak_lr=peak, final_lr_frac=0.0)
            for step in range(total)
        ]
        assert len(lrs) == total
        # Step 0 is the warmup floor (well below peak).
        assert lrs[0] < peak * 0.5
        # Peak is reached at the end of warmup.
        peak_idx = max(range(total), key=lambda i: lrs[i])
        assert peak_idx == warmup - 1 or peak_idx == warmup
        assert lrs[peak_idx] == pytest.approx(peak, rel=1e-6)
        # Strictly increasing through warmup.
        for i in range(1, warmup):
            assert lrs[i] >= lrs[i - 1]
        # Non-increasing through the cosine decay.
        for i in range(warmup + 1, total):
            assert lrs[i] <= lrs[i - 1] + 1e-9

    def test_cosine_decay_reaches_floor(self):
        from mimarsinan.training.imagenet_fast_train import one_cycle_lr_schedule

        total, warmup, peak, floor = 50, 10, 2.0, 0.05
        last = one_cycle_lr_schedule(total - 1, total_steps=total, warmup_steps=warmup,
                                     peak_lr=peak, final_lr_frac=floor / peak)
        assert last == pytest.approx(floor, rel=1e-3, abs=1e-3)

    def test_zero_warmup_starts_at_peak(self):
        from mimarsinan.training.imagenet_fast_train import one_cycle_lr_schedule

        lr0 = one_cycle_lr_schedule(0, total_steps=10, warmup_steps=0,
                                    peak_lr=1.0, final_lr_frac=0.0)
        assert lr0 == pytest.approx(1.0, rel=1e-6)

    def test_schedule_matches_closed_form_cosine(self):
        from mimarsinan.training.imagenet_fast_train import one_cycle_lr_schedule

        total, warmup, peak, frac = 40, 8, 0.5, 0.0
        step = 24  # in the decay phase
        progress = (step - warmup) / (total - 1 - warmup)
        expected = peak * (frac + (1 - frac) * 0.5 * (1 + math.cos(math.pi * progress)))
        got = one_cycle_lr_schedule(step, total_steps=total, warmup_steps=warmup,
                                    peak_lr=peak, final_lr_frac=frac)
        assert got == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# progressive resize schedule
# --------------------------------------------------------------------------- #
class TestProgressiveResize:
    def test_monotone_nondecreasing_and_hits_final(self):
        from mimarsinan.training.imagenet_fast_train import progressive_resize_schedule

        sizes = progressive_resize_schedule(
            num_epochs=16, start_size=160, final_size=192, final_epochs=2
        )
        assert len(sizes) == 16
        assert all(s2 >= s1 for s1, s2 in zip(sizes, sizes[1:]))
        assert sizes[0] == 160
        assert sizes[-1] == 192
        # Last `final_epochs` epochs are at the final (eval-matched) size.
        assert sizes[-1] == 192 and sizes[-2] == 192

    def test_all_multiples_of_32(self):
        from mimarsinan.training.imagenet_fast_train import progressive_resize_schedule

        sizes = progressive_resize_schedule(
            num_epochs=10, start_size=128, final_size=224, final_epochs=2
        )
        assert all(s % 32 == 0 for s in sizes)

    def test_single_size_when_start_equals_final(self):
        from mimarsinan.training.imagenet_fast_train import progressive_resize_schedule

        sizes = progressive_resize_schedule(
            num_epochs=5, start_size=192, final_size=192, final_epochs=1
        )
        assert sizes == [192] * 5


# --------------------------------------------------------------------------- #
# label-smoothing cross entropy numerics
# --------------------------------------------------------------------------- #
class TestLabelSmoothingCE:
    def test_matches_closed_form_reference(self):
        from mimarsinan.training.imagenet_fast_train import label_smoothing_cross_entropy

        torch.manual_seed(0)
        n, c = 4, 7
        logits = torch.randn(n, c, dtype=torch.float64)
        targets = torch.tensor([0, 3, 6, 1])
        smoothing = 0.1

        logp = torch.log_softmax(logits, dim=1)
        nll = -logp[torch.arange(n), targets]
        smooth = -logp.mean(dim=1)
        expected = ((1 - smoothing) * nll + smoothing * smooth).mean()

        got = label_smoothing_cross_entropy(logits, targets, smoothing=smoothing)
        assert got.item() == pytest.approx(expected.item(), rel=1e-9)

    def test_zero_smoothing_equals_plain_ce(self):
        from mimarsinan.training.imagenet_fast_train import label_smoothing_cross_entropy

        torch.manual_seed(1)
        logits = torch.randn(5, 10, dtype=torch.float64)
        targets = torch.randint(0, 10, (5,))
        plain = nn.functional.cross_entropy(logits, targets)
        got = label_smoothing_cross_entropy(logits, targets, smoothing=0.0)
        assert got.item() == pytest.approx(plain.item(), rel=1e-9)


# --------------------------------------------------------------------------- #
# dataloader factory: ffcv-vs-torchvision SELECTION (import mocked)
# --------------------------------------------------------------------------- #
class TestDataloaderFactorySelection:
    def test_selects_ffcv_when_importable(self, monkeypatch):
        from mimarsinan.training import imagenet_fast_train as m

        called = {}

        def fake_ffcv(provider, **kw):
            called["path"] = "ffcv"
            return "ffcv-loader"

        def fake_torch(provider, **kw):
            called["path"] = "torch"
            return "torch-loader"

        monkeypatch.setattr(m, "_ffcv_available", lambda: True)
        monkeypatch.setattr(m, "_build_ffcv_loaders", fake_ffcv)
        monkeypatch.setattr(m, "_build_torchvision_loaders", fake_torch)

        out = m.build_imagenet_dataloaders(provider=SimpleNamespace(), batch_size=8)
        assert out == "ffcv-loader"
        assert called["path"] == "ffcv"

    def test_falls_back_to_torchvision_when_ffcv_absent(self, monkeypatch):
        from mimarsinan.training import imagenet_fast_train as m

        called = {}

        def fake_torch(provider, **kw):
            called["path"] = "torch"
            return "torch-loader"

        monkeypatch.setattr(m, "_ffcv_available", lambda: False)
        monkeypatch.setattr(m, "_build_torchvision_loaders", fake_torch)

        out = m.build_imagenet_dataloaders(provider=SimpleNamespace(), batch_size=8)
        assert out == "torch-loader"
        assert called["path"] == "torch"

    def test_prefer_ffcv_false_forces_torchvision_even_if_available(self, monkeypatch):
        from mimarsinan.training import imagenet_fast_train as m

        called = {}

        def fake_ffcv(*a, **k):
            called["p"] = "ffcv"
            return "ffcv-loader"

        def fake_torch(*a, **k):
            called["p"] = "torch"
            return "torch-loader"

        monkeypatch.setattr(m, "_ffcv_available", lambda: True)
        monkeypatch.setattr(m, "_build_ffcv_loaders", fake_ffcv)
        monkeypatch.setattr(m, "_build_torchvision_loaders", fake_torch)

        out = m.build_imagenet_dataloaders(provider=SimpleNamespace(), batch_size=8,
                                           prefer_ffcv=False)
        assert out == "torch-loader"
        assert called["p"] == "torch"

    def test_ffcv_available_uses_guarded_import(self, monkeypatch):
        """_ffcv_available must probe via import, not a module-top hard import."""
        from mimarsinan.training import imagenet_fast_train as m
        import builtins

        real_import = builtins.__import__

        def deny_ffcv(name, *args, **kwargs):
            if name == "ffcv" or name.startswith("ffcv."):
                raise ModuleNotFoundError("No module named 'ffcv'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", deny_ffcv)
        assert m._ffcv_available() is False


# --------------------------------------------------------------------------- #
# train step on a tiny synthetic batch: loss decreases
# --------------------------------------------------------------------------- #
class TestTrainStepLossDecreases:
    def _tiny_model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16),
                             nn.ReLU(), nn.Linear(16, 5))

    def test_loss_decreases_over_steps_cpu(self):
        from mimarsinan.training.imagenet_fast_train import train_step

        torch.manual_seed(0)
        model = self._tiny_model()
        device = torch.device("cpu")
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        x = torch.randn(8, 3, 8, 8)
        y = torch.randint(0, 5, (8,))

        first = None
        last = None
        for i in range(25):
            loss = train_step(model, (x, y), opt, device=device,
                              smoothing=0.0, use_amp=False, channels_last=False)
            if i == 0:
                first = loss
            last = loss
        assert last < first

    def test_train_step_returns_python_float(self):
        from mimarsinan.training.imagenet_fast_train import train_step

        model = self._tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 5, (4,))
        loss = train_step(model, (x, y), opt, device=torch.device("cpu"),
                          smoothing=0.1, use_amp=False, channels_last=False)
        assert isinstance(loss, float)
        assert loss > 0.0


# --------------------------------------------------------------------------- #
# recipe config defaults (super-convergence shape)
# --------------------------------------------------------------------------- #
class TestRecipeConfig:
    def test_default_recipe_is_sgd_super_convergence(self):
        from mimarsinan.training.imagenet_fast_train import FastImageNetRecipe

        r = FastImageNetRecipe()
        assert r.optimizer == "sgd"
        assert 0.0 < r.momentum < 1.0
        assert r.label_smoothing > 0.0
        assert r.use_amp is True
        assert r.channels_last is True
        assert r.peak_lr > 0.0
        assert r.batch_size >= 256
        assert r.final_size >= r.start_size

    def test_resnet50_builder_is_channels_last(self):
        pytest.importorskip("torchvision")
        from mimarsinan.training.imagenet_fast_train import build_resnet50_channels_last

        model = build_resnet50_channels_last(num_classes=1000)
        # fc head matches num_classes
        assert model.fc.out_features == 1000
        # channels-last memory format on a 4D conv weight
        w = model.conv1.weight
        assert w.is_contiguous(memory_format=torch.channels_last)
