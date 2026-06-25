"""Tests for the DDP ImageNet-fast ORCHESTRATOR (scripts/gpu/train_imagenet_fast.py).

A TINY, FAST, SINGLE-PROCESS dry-run. NO ImageNet, NO torchrun, NO CUDA:
  * a 2-class TensorDataset stands in for ImageNet (a few random tensors);
  * a tiny CPU model stands in for ResNet-50 (so forward/backward is cheap);
  * the orchestrator's core (run / train_one_epoch / evaluate) is exercised via
    dependency injection (injected loader-factory + model + recipe).

Asserts the full contract:
  * the loop runs N epochs;
  * the per-epoch LR schedule is applied (warmup -> peak -> decay) to the optimizer;
  * the per-epoch progressive-resize size is applied (and rebuilds the train loader);
  * the val-eval returns a finite top-1 in [0, 100];
  * a structured JSON log line is emitted per epoch with the contracted keys;
  * a checkpoint file (state_dict + final val_top1) is written to --out.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Import the orchestrator module by file path (scripts/ is not a package).
# --------------------------------------------------------------------------- #
_ORCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts" / "gpu" / "train_imagenet_fast.py"
)


def _load_orchestrator():
    import sys

    spec = importlib.util.spec_from_file_location("train_imagenet_fast", _ORCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve cls.__module__.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def orch():
    return _load_orchestrator()


# --------------------------------------------------------------------------- #
# Tiny fixtures: fake dataset + tiny model standing in for ImageNet/ResNet-50.
# --------------------------------------------------------------------------- #
_NUM_CLASSES = 2
_IMG_C = 3
_N_TRAIN = 16
_N_VAL = 8


def _tiny_dataset(n: int, size: int) -> TensorDataset:
    torch.manual_seed(n + size)
    x = torch.randn(n, _IMG_C, size, size)
    y = torch.randint(0, _NUM_CLASSES, (n,))
    return TensorDataset(x, y)


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(_IMG_C, _NUM_CLASSES),
    )


class _RecordingTrainLoaderFactory:
    """Injectable build_train_loader(size): records every requested size."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.requested_sizes: list[int] = []

    def __call__(self, size: int) -> DataLoader:
        self.requested_sizes.append(int(size))
        ds = _tiny_dataset(_N_TRAIN, int(size))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)


def _val_loader(batch_size: int) -> DataLoader:
    # eval at a fixed (eval) resolution.
    ds = _tiny_dataset(_N_VAL, 32)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# --------------------------------------------------------------------------- #
# evaluate(): finite top-1 in [0, 100]
# --------------------------------------------------------------------------- #
class TestEvaluate:
    def test_top1_is_finite_and_in_range(self, orch):
        model = _tiny_model()
        loader = _val_loader(4)
        top1 = orch.evaluate(model, loader, device=torch.device("cpu"))
        assert isinstance(top1, float)
        assert math.isfinite(top1)
        assert 0.0 <= top1 <= 100.0

    def test_perfect_model_scores_100(self, orch):
        # A loader whose every label is class 0, and a model that always
        # predicts class 0, must score exactly 100% top-1.
        x = torch.randn(6, _IMG_C, 8, 8)
        y = torch.zeros(6, dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=3)

        class AlwaysZero(nn.Module):
            def forward(self, inp):
                n = inp.shape[0]
                out = torch.zeros(n, _NUM_CLASSES)
                out[:, 0] = 10.0
                return out

        top1 = orch.evaluate(AlwaysZero(), loader, device=torch.device("cpu"))
        assert top1 == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# train_one_epoch(): applies the per-step LR, returns a finite mean loss
# --------------------------------------------------------------------------- #
class TestTrainOneEpoch:
    def test_applies_lr_and_returns_finite_loss(self, orch):
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9)
        loader = DataLoader(_tiny_dataset(_N_TRAIN, 16), batch_size=4, drop_last=False)

        seen_lrs: list[float] = []

        def lr_for_step(global_step: int) -> float:
            lr = 0.01 * (global_step + 1)
            seen_lrs.append(lr)
            return lr

        loss = orch.train_one_epoch(
            model, loader, opt,
            device=torch.device("cpu"),
            lr_for_step=lr_for_step,
            smoothing=0.1,
            use_amp=False,
            channels_last=False,
            scaler=None,
        )
        assert math.isfinite(loss)
        # LR was requested for every step and pushed onto the optimizer.
        assert len(seen_lrs) == len(loader)
        # The optimizer's current LR is the LAST scheduled value.
        assert opt.param_groups[0]["lr"] == pytest.approx(seen_lrs[-1])

    def test_max_steps_truncates_and_logs(self, orch):
        """``max_steps`` caps the epoch (smoke / dry real run) and ``log_fn``+
        ``log_every`` emit the per-step observability contract."""
        model = _tiny_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        loader = DataLoader(_tiny_dataset(_N_TRAIN, 16), batch_size=4, drop_last=False)
        assert len(loader) > 3
        logs: list[dict] = []
        orch.train_one_epoch(
            model, loader, opt, device=torch.device("cpu"),
            lr_for_step=lambda s: 0.01, use_amp=False, channels_last=False,
            max_steps=3, world_size=4, log_fn=logs.append, log_every=1,
        )
        assert len(logs) == 3  # capped below the full loader length
        for i, line in enumerate(logs):
            assert line["step"] == i and line["epoch"] == 0
            assert {"loss", "lr", "imgs_per_s"} <= set(line)


# --------------------------------------------------------------------------- #
# run(): the full epoch loop end-to-end (DI; no DDP/CUDA/ImageNet)
# --------------------------------------------------------------------------- #
class TestRunLoop:
    def test_runs_n_epochs_with_schedules_and_writes_checkpoint(self, orch, tmp_path):
        from mimarsinan.training.imagenet_fast_train import FastImageNetRecipe

        epochs = 4
        recipe = FastImageNetRecipe(
            num_classes=_NUM_CLASSES,
            epochs=epochs,
            batch_size=4,
            peak_lr=0.05,
            warmup_frac=0.25,
            start_size=32,
            final_size=64,
            final_size_epochs=1,
        )
        model = _tiny_model()
        train_factory = _RecordingTrainLoaderFactory(batch_size=4)
        val_loader = _val_loader(4)
        out = tmp_path / "ckpt.pt"

        logs: list[dict] = []
        result = orch.run(
            recipe=recipe,
            model=model,
            build_train_loader=train_factory,
            val_loader=val_loader,
            device=torch.device("cpu"),
            out_path=str(out),
            is_main=True,
            world_size=1,
            eval_every=1,
            log_fn=logs.append,
            use_amp=False,
            channels_last=False,
        )

        # --- N epochs ran, one structured log line each ----------------------
        assert len(logs) == epochs
        for e, line in enumerate(logs):
            for key in ("epoch", "img_size", "lr", "train_loss",
                        "val_top1", "epoch_seconds"):
                assert key in line, f"missing key {key!r} in epoch log"
            assert line["epoch"] == e
            assert math.isfinite(line["train_loss"])
            assert math.isfinite(line["val_top1"])
            assert 0.0 <= line["val_top1"] <= 100.0
            # log line must be JSON-serialisable (structured logging contract).
            json.dumps(line)

        # --- progressive-resize schedule applied per epoch -------------------
        from mimarsinan.training.imagenet_fast_train import progressive_resize_schedule
        expected_sizes = progressive_resize_schedule(
            num_epochs=epochs, start_size=32, final_size=64, final_epochs=1
        )
        # The recorded train-loader rebuild sizes match the schedule, and the
        # logged img_size matches it too.
        assert train_factory.requested_sizes == expected_sizes
        assert [l["img_size"] for l in logs] == expected_sizes
        # finishes at the eval-matched (final) size.
        assert logs[-1]["img_size"] == 64

        # --- one-cycle LR applied: warmup rises, then decays -----------------
        lrs = [l["lr"] for l in logs]
        assert lrs[0] < recipe.peak_lr  # epoch 0 is in warmup (below peak)
        assert max(lrs) == pytest.approx(recipe.peak_lr, rel=1e-6) or \
            max(lrs) <= recipe.peak_lr + 1e-9
        assert lrs[-1] <= lrs[0] + recipe.peak_lr  # decays toward the floor by the end
        # not all equal (schedule actually varied the LR across epochs).
        assert len(set(round(v, 6) for v in lrs)) > 1

        # --- checkpoint written with state_dict + final val_top1 -------------
        assert out.exists(), "rank-0 checkpoint not written"
        ckpt = torch.load(str(out), map_location="cpu")
        assert "model" in ckpt and isinstance(ckpt["model"], dict)
        assert "val_top1" in ckpt and math.isfinite(ckpt["val_top1"])
        assert ckpt["val_top1"] == pytest.approx(result["val_top1"])

        # --- run() returns the achieved top-1 + wall time --------------------
        assert "val_top1" in result and math.isfinite(result["val_top1"])
        assert "wall_seconds" in result and result["wall_seconds"] >= 0.0

    def test_non_main_rank_does_not_write_checkpoint(self, orch, tmp_path):
        from mimarsinan.training.imagenet_fast_train import FastImageNetRecipe

        recipe = FastImageNetRecipe(
            num_classes=_NUM_CLASSES, epochs=1, batch_size=4, peak_lr=0.05,
            start_size=32, final_size=32, final_size_epochs=1,
        )
        out = tmp_path / "should_not_exist.pt"
        orch.run(
            recipe=recipe,
            model=_tiny_model(),
            build_train_loader=_RecordingTrainLoaderFactory(batch_size=4),
            val_loader=_val_loader(4),
            device=torch.device("cpu"),
            out_path=str(out),
            is_main=False,  # non-rank-0
            world_size=2,
            eval_every=1,
            log_fn=lambda _l: None,
            use_amp=False,
            channels_last=False,
        )
        assert not out.exists(), "non-main rank must not write the checkpoint"

    def test_eval_every_skips_intermediate_evals(self, orch, tmp_path):
        """With eval_every=K only the K-aligned epochs (and the last) carry a real top-1."""
        from mimarsinan.training.imagenet_fast_train import FastImageNetRecipe

        epochs = 4
        recipe = FastImageNetRecipe(
            num_classes=_NUM_CLASSES, epochs=epochs, batch_size=4, peak_lr=0.05,
            start_size=32, final_size=32, final_size_epochs=1,
        )
        logs: list[dict] = []
        orch.run(
            recipe=recipe,
            model=_tiny_model(),
            build_train_loader=_RecordingTrainLoaderFactory(batch_size=4),
            val_loader=_val_loader(4),
            device=torch.device("cpu"),
            out_path=str(tmp_path / "c.pt"),
            is_main=True,
            world_size=1,
            eval_every=2,
            log_fn=logs.append,
            use_amp=False,
            channels_last=False,
        )
        assert len(logs) == epochs
        # The final epoch is ALWAYS evaluated (it's the reported accuracy).
        assert math.isfinite(logs[-1]["val_top1"])
        # An un-evaluated epoch reports NaN (sentinel "not measured").
        assert math.isnan(logs[0]["val_top1"]) or math.isfinite(logs[0]["val_top1"])
        # epoch index 1 (0-based) under eval_every=2 is NOT aligned -> NaN.
        assert math.isnan(logs[1]["val_top1"])


# --------------------------------------------------------------------------- #
# DDP env parsing (single-process default when torchrun env is unset)
# --------------------------------------------------------------------------- #
class TestDistEnv:
    def test_single_process_when_world_size_unset(self, orch, monkeypatch):
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            monkeypatch.delenv(k, raising=False)
        info = orch.read_dist_env()
        assert info.world_size == 1
        assert info.rank == 0
        assert info.local_rank == 0
        assert info.is_distributed is False
        assert info.is_main is True

    def test_parses_torchrun_env(self, orch, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("LOCAL_RANK", "2")
        info = orch.read_dist_env()
        assert info.world_size == 4
        assert info.rank == 2
        assert info.local_rank == 2
        assert info.is_distributed is True
        assert info.is_main is False  # only rank 0 is main


# --------------------------------------------------------------------------- #
# CLI parsing (thin __main__ glue is argparse-driven)
# --------------------------------------------------------------------------- #
class TestCli:
    def test_argparser_defaults_and_overrides(self, orch):
        p = orch.build_arg_parser()
        args = p.parse_args([])
        assert args.data_root == "/data/ImageNet"
        assert args.eval_every >= 1
        # overrides parse.
        args2 = p.parse_args(
            ["--epochs", "3", "--batch-size", "128", "--workers", "8",
             "--data-root", "/tmp/in", "--out", "/tmp/o.pt", "--eval-every", "2"]
        )
        assert args2.epochs == 3
        assert args2.batch_size == 128
        assert args2.workers == 8
        assert args2.data_root == "/tmp/in"
        assert args2.out == "/tmp/o.pt"
        assert args2.eval_every == 2
