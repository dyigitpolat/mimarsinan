from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory


@dataclass
class FastAccuracyEvaluator:
    """
    Fast per-candidate training+validation loop for NAS.

    Schedule (as requested):
    - 1 epoch total over the full training set
    - LR warmup over the first warmup_fraction of batches
    - validate once after the epoch
    """

    data_provider_factory: DataProviderFactory
    device: torch.device
    lr: float
    warmup_fraction: float = 0.10
    num_workers: int = 0
    training_batch_size: Optional[int] = None
    seed: int = 0

    def evaluate(self, model) -> float:
        # Deterministic-ish evaluation (controls shuffle order + any stochastic ops)
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        data_loader_factory = DataLoaderFactory(self.data_provider_factory, num_workers=int(self.num_workers))
        data_provider = data_loader_factory.create_data_provider()

        pred_mode = data_provider.get_prediction_mode().mode()
        if pred_mode != "classification":
            raise NotImplementedError(f"FastAccuracyEvaluator only supports classification (got {pred_mode})")

        loss_fn = data_provider.create_loss()

        train_bs = int(self.training_batch_size) if self.training_batch_size is not None else int(data_provider.get_training_batch_size())
        val_bs = int(data_provider.get_validation_set_size())

        train_loader = data_loader_factory.create_training_loader(train_bs, data_provider)
        val_loader = data_loader_factory.create_validation_loader(val_bs, data_provider)

        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.lr),
            betas=(0.9, 0.99),
            weight_decay=5e-5,
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        total_batches = max(1, len(train_loader))
        warmup_batches = max(1, int(total_batches * float(self.warmup_fraction)))

        # Train 1 epoch
        for step_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # batch-level warmup over first 10% of steps
            if step_idx < warmup_batches:
                lr_now = float(self.lr) * float(step_idx + 1) / float(warmup_batches)
            else:
                lr_now = float(self.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model, x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validate once (full validation set for providers that define it that way)
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                _, predicted = model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())

        return float(correct / total) if total > 0 else 0.0


