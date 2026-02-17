"""
Accuracy evaluator for NAS that fits a learning curve and extrapolates.

Instead of training for one epoch and returning raw accuracy,
this evaluator records accuracy at several checkpoints during training,
fits parametric learning-curve models to the trajectory, and extrapolates
to predict what accuracy would be at a larger epoch budget.

This yields a more informative ranking signal for architecture search
because it estimates the architecture's *potential* rather than its
performance after minimal training.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory


# ---------------------------------------------------------------------------
# Parametric learning-curve models
# ---------------------------------------------------------------------------
# Each model maps (t, *params) -> predicted accuracy.
# `t` is in *epochs* (so 0.2 means 20 % of the first epoch).

def _exp3(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """3-parameter saturating exponential: y = a - b * exp(-c * t)."""
    return a - b * np.exp(-c * t)


def _pow3(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """3-parameter inverse power law: y = a - b * (t + 1)^{-c}."""
    return a - b * np.power(t + 1.0, -c)


def _log2(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """2-parameter log model: y = a * log(1 + b * t)."""
    return a * np.log1p(b * t)


# Registry: (name, function, number_of_params, bounds_lo, bounds_hi)
_CURVE_MODELS = [
    (
        "exp3",
        _exp3,
        3,
        [0.0, 0.0, 1e-6],       # lower bounds for a, b, c
        [1.0, 1.0, 50.0],       # upper bounds for a, b, c
    ),
    (
        "pow3",
        _pow3,
        3,
        [0.0, 0.0, 1e-6],
        [1.0, 1.0, 10.0],
    ),
    (
        "log2",
        _log2,
        2,
        [0.0, 1e-6],
        [1.0, 100.0],
    ),
]


def _fit_and_extrapolate(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_target: float,
) -> Tuple[float, str]:
    """
    Try each curve model, pick the best fit, and extrapolate to *t_target*.

    Returns (extrapolated_accuracy, model_name).
    Falls back to the last observed accuracy if all fits fail.
    """
    best_residual = float("inf")
    best_pred = float(y_obs[-1])   # fallback
    best_name = "fallback"

    for name, func, n_params, lb, ub in _CURVE_MODELS:
        if len(t_obs) < n_params + 1:
            # Not enough data points for this model
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    func,
                    t_obs,
                    y_obs,
                    bounds=(lb, ub),
                    maxfev=5000,
                    p0=None,
                )
            y_fit = func(t_obs, *popt)
            residual = float(np.mean((y_fit - y_obs) ** 2))

            if residual < best_residual:
                best_residual = residual
                pred = float(func(np.array([t_target]), *popt)[0])
                # Clamp to [0, 1] and ensure it's at least as good as last observation
                pred = max(0.0, min(1.0, pred))
                best_pred = pred
                best_name = name
        except (RuntimeError, ValueError, TypeError):
            # curve_fit failed to converge – skip this model
            continue

    return best_pred, best_name


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

@dataclass
class ExtrapolatingAccuracyEvaluator:
    """
    Fast NAS accuracy evaluator with learning-curve extrapolation.

    Schedule
    --------
    1. Train the candidate model for ``num_train_epochs`` epochs.
    2. Every ``total_steps / num_checkpoints`` gradient steps, pause and
       measure validation accuracy.
    3. Fit parametric learning-curve models (exp-saturation, power-law,
       logarithmic) to the recorded (step, accuracy) pairs.
    4. Extrapolate the best-fitting curve to ``target_epochs`` and return
       the predicted accuracy.

    Parameters
    ----------
    num_train_epochs : int
        How many actual training epochs to run (default 1, for speed).
    num_checkpoints : int
        Number of accuracy measurements during training. More checkpoints
        give a better curve fit but cost validation time.
    target_epochs : int
        Epoch count to extrapolate accuracy to.  E.g. if the real pipeline
        trains for 50 epochs, set ``target_epochs=50`` so the evaluator
        predicts that final accuracy.
    """

    data_provider_factory: DataProviderFactory
    device: torch.device
    lr: float

    # Training knobs
    num_train_epochs: int = 1
    num_checkpoints: int = 5

    # Extrapolation target
    target_epochs: int = 10

    # Shared knobs (same as FastAccuracyEvaluator)
    warmup_fraction: float = 0.10
    num_workers: int = 0
    training_batch_size: Optional[int] = None
    seed: int = 0

    def evaluate(self, model) -> float:
        """Train, record learning curve, extrapolate, and return predicted accuracy."""

        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))

        # ---- data setup ------------------------------------------------
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

        model = model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.lr),
            betas=(0.9, 0.99),
            weight_decay=5e-5,
        )

        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ---- training with checkpoints ---------------------------------
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * int(self.num_train_epochs)
        warmup_steps = max(1, int(total_steps * float(self.warmup_fraction)))

        # Decide at which global steps to measure accuracy.
        # We always include the final step; the rest are spaced evenly.
        num_ckpt = max(1, int(self.num_checkpoints))
        checkpoint_interval = max(1, total_steps // num_ckpt)
        checkpoint_steps = set()
        for i in range(1, num_ckpt + 1):
            checkpoint_steps.add(min(i * checkpoint_interval, total_steps))
        checkpoint_steps.add(total_steps)  # always measure at the end

        # (epoch_fraction, accuracy) pairs
        curve_t: List[float] = []
        curve_y: List[float] = []

        global_step = 0
        for epoch in range(int(self.num_train_epochs)):
            model.train()
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # LR warmup
                if global_step < warmup_steps:
                    lr_now = float(self.lr) * float(global_step + 1) / float(warmup_steps)
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

                global_step += 1

                if global_step in checkpoint_steps:
                    acc = self._validate(model, val_loader)
                    t = float(global_step) / float(steps_per_epoch)  # in epochs
                    curve_t.append(t)
                    curve_y.append(acc)

        # ---- curve fitting & extrapolation -----------------------------
        t_obs = np.array(curve_t, dtype=np.float64)
        y_obs = np.array(curve_y, dtype=np.float64)
        t_target = float(self.target_epochs)

        if len(t_obs) < 2:
            # Can't fit a curve with < 2 points; return raw accuracy.
            return float(y_obs[-1]) if len(y_obs) > 0 else 0.0

        extrapolated, model_name = _fit_and_extrapolate(t_obs, y_obs, t_target)

        return float(extrapolated)

    # ---- helpers -------------------------------------------------------

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

