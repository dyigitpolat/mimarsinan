"""Accuracy evaluator for NAS that fits a learning curve and extrapolates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory

from .learning_curve import fit_and_extrapolate


@dataclass
class ExtrapolatingAccuracyEvaluator:
    """Fast NAS accuracy evaluator with learning-curve extrapolation."""

    data_provider_factory: DataProviderFactory
    device: torch.device
    lr: float

    num_train_epochs: int = 1
    num_checkpoints: int = 5
    target_epochs: int = 10

    warmup_fraction: float = 0.10
    num_workers: int = 0
    training_batch_size: Optional[int] = None
    seed: int = 0

    def evaluate(self, model) -> float:
        """Train, record learning curve, extrapolate, and return predicted accuracy."""
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        data_loader_factory = DataLoaderFactory(
            self.data_provider_factory, num_workers=int(self.num_workers)
        )
        data_provider = data_loader_factory.create_data_provider()

        pred_mode = data_provider.get_prediction_mode().mode()
        if pred_mode != "classification":
            raise NotImplementedError(
                f"ExtrapolatingAccuracyEvaluator only supports classification (got {pred_mode})"
            )

        loss_fn = data_provider.create_loss()

        train_bs = (
            int(self.training_batch_size)
            if self.training_batch_size is not None
            else int(data_provider.get_training_batch_size())
        )
        val_bs = int(data_provider.get_validation_set_size())

        train_loader = data_loader_factory.create_training_loader(train_bs, data_provider)
        val_loader = data_loader_factory.create_validation_loader(val_bs, data_provider)

        try:
            model = model.to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(self.lr),
                betas=(0.9, 0.99),
                weight_decay=5e-5,
            )

            use_amp = self.device.type == "cuda"
            scaler = GradScaler("cuda", enabled=use_amp)

            steps_per_epoch = max(1, len(train_loader))
            total_steps = steps_per_epoch * int(self.num_train_epochs)
            warmup_steps = max(1, int(total_steps * float(self.warmup_fraction)))

            num_ckpt = max(1, int(self.num_checkpoints))
            checkpoint_interval = max(1, total_steps // num_ckpt)
            checkpoint_steps = set()
            for i in range(1, num_ckpt + 1):
                checkpoint_steps.add(min(i * checkpoint_interval, total_steps))
            checkpoint_steps.add(total_steps)

            curve_t: List[float] = []
            curve_y: List[float] = []

            global_step = 0
            for _epoch in range(int(self.num_train_epochs)):
                model.train()
                for x, y in train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    if global_step < warmup_steps:
                        lr_now = float(self.lr) * float(global_step + 1) / float(warmup_steps)
                    else:
                        lr_now = float(self.lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_now

                    optimizer.zero_grad(set_to_none=True)
                    with autocast("cuda", enabled=use_amp):
                        loss = loss_fn(model, x, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1

                    if global_step in checkpoint_steps:
                        acc = self._validate(model, val_loader)
                        t = float(global_step) / float(steps_per_epoch)
                        curve_t.append(t)
                        curve_y.append(acc)

            t_obs = np.array(curve_t, dtype=np.float64)
            y_obs = np.array(curve_y, dtype=np.float64)
            t_target = float(self.target_epochs)

            if len(t_obs) < 2:
                return float(y_obs[-1]) if len(y_obs) > 0 else 0.0

            extrapolated, _model_name = fit_and_extrapolate(t_obs, y_obs, t_target)
            return float(extrapolated)
        finally:
            shutdown_data_loader(train_loader)
            shutdown_data_loader(val_loader)

    @torch.no_grad()
    def _validate(self, model: torch.nn.Module, val_loader) -> float:
        model.eval()
        correct = 0.0
        total = 0.0
        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, predicted = model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
        model.train()
        return float(correct / total) if total > 0 else 0.0
