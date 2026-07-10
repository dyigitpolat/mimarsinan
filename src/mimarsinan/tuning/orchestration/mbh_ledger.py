"""Gate-grade MBH measurement helpers + per-rung [MBH] ledger for the fast ladder."""

from __future__ import annotations

import contextlib
import copy
import math

import torch
import torch.nn.functional as F

from mimarsinan.common.env import mbh_ledger_enabled
from mimarsinan.model_training.basic_trainer_eval import metric_grade_eval
from mimarsinan.tuning.orchestration.genuine_probe import (
    eval_forward_over_val,
    genuine_acc_on_clone,
)

_RHO_DENOM_EPS = 1e-12


def fp32_eval_forward_over_val(trainer, forward_obj, model, n_batches, device) -> float:
    """E0: every MBH read is fp32 — like-for-like with D-hat via the shared
    metric-grade seam, even inside an ambient autocast region."""
    with metric_grade_eval(device):
        return eval_forward_over_val(trainer, forward_obj, model, n_batches, device)


def fp32_deployed_read(tuner) -> float:
    """fp32 accuracy of a tuner's LIVE model (the deployed composition) over its
    eval batches — like-for-like with the gate's D-hat reads."""
    device = tuner.pipeline.config["device"]
    return float(fp32_eval_forward_over_val(
        tuner.trainer, tuner.model, tuner.model,
        tuner._budget.eval_n_batches, device,
    ))


@contextlib.contextmanager
def _measurement_guard(trainer):
    """Isolation for every MBH measurement: fork the RNG (CUDA included) and
    restore the trainer's validation cursor, so the live trajectory is untouched."""
    cursor = getattr(trainer, "_gpu_val_cursor", None)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        if cursor is not None:
            trainer._gpu_val_cursor = cursor


def nonzero_grad_fraction(model) -> float:
    """Fraction of trainable parameter ELEMENTS with nonzero ``.grad`` (A5 reach).

    ``grad is None`` counts as all-zero — severed submodules must lower the
    fraction, not vanish from it. nan when the model has no trainable params.
    """
    nonzero = 0
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        total += param.numel()
        if param.grad is not None:
            nonzero += int(torch.count_nonzero(param.grad))
    if total == 0:
        return float("nan")
    return nonzero / total


def capture_rung_nonzero_grad_fraction(tuner) -> None:
    """Stash the A5 reach gauge from the live ``.grad``s (a rung's first backward).

    Ledger-flag-gated (one pass over the grads; the default gate stays lean);
    consumed by :func:`emit_fast_rung_ledger` via ``_mbh_nonzero_grad_fraction``.
    """
    if mbh_ledger_enabled():
        tuner._mbh_nonzero_grad_fraction = nonzero_grad_fraction(tuner.model)


def rung_measurements(tuner) -> dict:
    """One rung's isolated fp32 reads: blended_fp32, full_acc (D-hat), rho, ||g_t||.

    The gate consumes full_acc only; blended_fp32 and the alignment pair
    (rho, ||g_t||) are verbose diagnostics measured only under the ledger flag
    (nan otherwise — A4 eval consolidation: one gate probe per rung by default).
    """
    with _measurement_guard(tuner.trainer):
        if mbh_ledger_enabled():
            blended_fp32 = blended_acc_fp32(tuner)
            full_acc = full_transform_acc_on_clone(tuner)
            rho, grad_norm_t = transfer_alignment(tuner)
        else:
            blended_fp32 = float("nan")
            full_acc = full_transform_acc_on_clone(tuner)
            rho, grad_norm_t = float("nan"), float("nan")
    return {
        "blended_fp32": blended_fp32,
        "full_acc": full_acc,
        "rho": rho,
        "grad_norm_t": grad_norm_t,
    }


def full_transform_measurement(tuner) -> float:
    """D-hat alone under the same isolation guard (the gate's entry read)."""
    with _measurement_guard(tuner.trainer):
        return full_transform_acc_on_clone(tuner)


def live_model_acc_fp32(tuner) -> float:
    """The LIVE model's fp32 accuracy over the tuner's eval batches, cursor-isolated.

    The installed forward IS the behavior under measurement (e.g. the P4
    k-hybrid), so this is the honest partial-deployment probe for in-place
    calibration loops that mutate the live model.
    """
    device = tuner.pipeline.config["device"]
    with _measurement_guard(tuner.trainer):
        return fp32_eval_forward_over_val(
            tuner.trainer, tuner.model, tuner.model,
            tuner._budget.eval_n_batches, device,
        )


def emit_fast_rung_ledger(tuner, *, rate, blended_acc, measurements=None):
    """Emit one ``[MBH]`` stdout line for a completed fast-ladder rung attempt.

    Prints only under ``MIMARSINAN_MBH_LEDGER`` (verbose diagnostics; the
    default gate's probes run regardless). ``measurements`` reuses a dict
    already produced by ``rung_measurements`` (the gate's path) instead of
    re-measuring.
    """
    if not mbh_ledger_enabled():
        return None
    rung = int(getattr(tuner, "_mbh_rung_index", -1)) + 1
    tuner._mbh_rung_index = rung
    if measurements is None:
        measurements = rung_measurements(tuner)
    nonzero_frac = float(getattr(tuner, "_mbh_nonzero_grad_fraction", float("nan")))
    line = (
        f"[MBH] tuner={type(tuner).__name__} rung={rung} rate={float(rate):.6f} "
        f"blended_acc={float(blended_acc):.6f} "
        f"blended_fp32={float(measurements['blended_fp32']):.6f} "
        f"full_acc={float(measurements['full_acc']):.6f} "
        f"rho={float(measurements['rho']):.6f} "
        f"grad_norm_t={float(measurements['grad_norm_t']):.6f} "
        f"nonzero_grad_frac={nonzero_frac:.6f}"
    )
    print(line, flush=True)
    return line


def blended_acc_fp32(tuner) -> float:
    """The CURRENT blended model's fp32 accuracy on an isolated deepcopy — the
    gate-grade counterpart of the live (autocast) rung probe."""
    device = tuner.pipeline.config["device"]
    return genuine_acc_on_clone(
        tuner.model,
        device,
        prepare=lambda clone: None,
        build_forward=lambda clone: clone,
        evaluate=lambda forward, clone: fp32_eval_forward_over_val(
            tuner.trainer, forward, clone, tuner._budget.eval_n_batches, device,
        ),
    )


def full_transform_acc_on_clone(tuner) -> float:
    """D-hat: the deployed full-transformation (rate 1.0) fp32 accuracy, measured on
    an isolated deepcopy over the tuner's eval batches (the probe's sample set)."""
    device = tuner.pipeline.config["device"]
    return genuine_acc_on_clone(
        tuner.model,
        device,
        prepare=lambda clone: None,
        build_forward=tuner._mbh_full_transform_forward,
        evaluate=lambda forward, clone: fp32_eval_forward_over_val(
            tuner.trainer, forward, clone, tuner._budget.eval_n_batches, device,
        ),
    )


def transfer_alignment(tuner) -> tuple:
    """(rho, ||g_t||) on one fixed validation batch, on one isolated deepcopy.

    rho = <g_1, g_t> / ||g_t||^2 with plain-CE fp32 gradients: g_t under the
    CURRENT blended behavior, g_1 on the same clone forced to the full
    transformation. ||g_t||^2 below 1e-12 yields rho = nan.
    """
    batch = _fixed_validation_batch(tuner.trainer)
    if batch is None:
        return float("nan"), float("nan")
    device = tuner.pipeline.config["device"]
    x, y = (t.to(device) for t in batch)
    clone = copy.deepcopy(tuner.model).to(device)
    grads_t = _ce_parameter_grads(clone, clone, x, y, device)
    full_forward = tuner._mbh_full_transform_forward(clone)
    grads_1 = _ce_parameter_grads(clone, full_forward, x, y, device)
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


def _ce_parameter_grads(clone, forward, x, y, device) -> dict:
    """Named plain-CE fp32 parameter gradients of ``forward`` on the eval-mode clone.

    A gradient-free forward (a bare staircase kernel, zero gradient a.e. — the
    theory's §2a case) yields the empty dict, i.e. an identically-zero gradient.
    """
    clone.eval()
    for param in clone.parameters():
        param.grad = None
    with torch.enable_grad(), metric_grade_eval(device):
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
