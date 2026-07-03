"""Measurement-only per-rung [MBH] ledger for the shared fast ladder (MBH X1)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from mimarsinan.common.env import mbh_ledger_enabled
from mimarsinan.tuning.orchestration.genuine_probe import (
    eval_forward_over_val,
    genuine_acc_on_clone,
)

_RHO_DENOM_EPS = 1e-12


def emit_fast_rung_ledger(tuner, *, rate, blended_acc):
    """Emit one ``[MBH]`` stdout line for a completed fast-ladder rung.

    Gated by ``MIMARSINAN_MBH_LEDGER`` (env SSOT). Every measurement runs on
    deepcopies inside ``fork_rng`` with the trainer's validation cursor restored,
    so the live model/optimizer/RNG trajectory is bit-identical to flag-OFF.
    """
    if not mbh_ledger_enabled():
        return None
    rung = int(getattr(tuner, "_mbh_rung_index", -1)) + 1
    tuner._mbh_rung_index = rung
    trainer = tuner.trainer
    cursor = getattr(trainer, "_gpu_val_cursor", None)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            full_acc = full_transform_acc_on_clone(tuner)
            rho, grad_norm_t = transfer_alignment(tuner)
    finally:
        if cursor is not None:
            trainer._gpu_val_cursor = cursor
    line = (
        f"[MBH] tuner={type(tuner).__name__} rung={rung} rate={float(rate):.6f} "
        f"blended_acc={float(blended_acc):.6f} full_acc={full_acc:.6f} "
        f"rho={rho:.6f} grad_norm_t={grad_norm_t:.6f}"
    )
    print(line, flush=True)
    return line


def full_transform_acc_on_clone(tuner) -> float:
    """D-hat: the deployed full-transformation (rate 1.0) accuracy, measured on an
    isolated deepcopy over the tuner's eval batches (the probe's sample set)."""
    device = tuner.pipeline.config["device"]
    return genuine_acc_on_clone(
        tuner.model,
        device,
        prepare=lambda clone: None,
        build_forward=tuner._mbh_full_transform_forward,
        evaluate=lambda forward, clone: eval_forward_over_val(
            tuner.trainer, forward, clone, tuner._budget.eval_n_batches, device,
        ),
    )


def transfer_alignment(tuner) -> tuple:
    """(rho, ||g_t||) on one fixed validation batch, on one isolated deepcopy.

    rho = <g_1, g_t> / ||g_t||^2 with plain-CE gradients: g_t under the CURRENT
    blended behavior, g_1 on the same clone forced to the full transformation.
    ||g_t||^2 below 1e-12 yields rho = nan.
    """
    batch = _fixed_validation_batch(tuner.trainer)
    if batch is None:
        return float("nan"), float("nan")
    device = tuner.pipeline.config["device"]
    x, y = (t.to(device) for t in batch)
    clone = copy.deepcopy(tuner.model).to(device)
    grads_t = _ce_parameter_grads(clone, clone, x, y)
    full_forward = tuner._mbh_full_transform_forward(clone)
    grads_1 = _ce_parameter_grads(clone, full_forward, x, y)
    norm_t_sq = sum(float((g * g).sum()) for g in grads_t.values())
    grad_norm_t = math.sqrt(norm_t_sq)
    if norm_t_sq < _RHO_DENOM_EPS:
        return float("nan"), grad_norm_t
    dot = sum(
        float((grads_1[name] * g).sum())
        for name, g in grads_t.items()
        if name in grads_1
    )
    return dot / norm_t_sq, grad_norm_t


def _ce_parameter_grads(clone, forward, x, y) -> dict:
    """Named plain-CE parameter gradients of ``forward`` on the eval-mode clone.

    A gradient-free forward (a bare staircase kernel, zero gradient a.e. — the
    theory's §2a case) yields the empty dict, i.e. an identically-zero gradient.
    """
    clone.eval()
    for param in clone.parameters():
        param.grad = None
    with torch.enable_grad():
        loss = F.cross_entropy(forward(x), y)
        if loss.grad_fn is None:
            return {}
        loss.backward()
    return {
        name: param.grad.detach().clone()
        for name, param in clone.named_parameters()
        if param.grad is not None
    }


def _fixed_validation_batch(trainer):
    """The first cached validation batch — fixed across rungs, cursor untouched."""
    if getattr(trainer, "_gpu_val_cache", None) is None:
        for _ in trainer.iter_validation_batches(0):
            pass
    cache = getattr(trainer, "_gpu_val_cache", None)
    if not cache:
        return None
    return cache[0]
