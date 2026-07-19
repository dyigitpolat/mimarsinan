"""[spiking_deployment_calculus §15.4] DeployedRiskFinetune: terminal origin-KD minimization of the deployed risk through an injected deployed forward."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DeployedRiskResult:
    entry_genuine: float
    best_genuine: float
    final_genuine: float
    steps_run: int
    resumed_from: int


def _cpu_state(model) -> dict:
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def _atomic_save(payload: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_deployed_risk_finetune(
    model,
    teacher,
    deployed_forward,
    *,
    train_batches,
    eval_genuine,
    steps: int,
    lr: float,
    eval_every: int = 60,
    warmup_frac: float = 0.1,
    kd_alpha: float = 0.5,
    kd_temperature: float = 4.0,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 120,
    stop_after: int | None = None,
) -> DeployedRiskResult:
    """Minimize the DEPLOYED risk: (1-a)*CE + a*KD-to-``teacher`` computed ON
    ``deployed_forward``; warmup+cosine; keep-best on ``eval_genuine`` (the
    entry read seeds the ratchet — the stage never returns worse than entry);
    ``checkpoint_path`` makes the stage own its resume across process windows
    (optimizer/schedule/best state included; batch order is NOT resumed —
    callers hand a freshly shuffled stream per invocation)."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warm = max(1, int(steps * warmup_frac))

    def _lr_lambda(step_index: int) -> float:
        if step_index < warm:
            return (step_index + 1) / warm
        span = max(1, steps - warm)
        progress = min(1.0, (step_index - warm) / span)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    step = 0
    resumed_from = 0
    entry: float | None = None
    best_val = float("-inf")
    best_state: dict | None = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["schedule"])
        step = resumed_from = int(payload["step"])
        entry = float(payload["entry"])
        best_val = float(payload["best_val"])
        best_state = payload["best_model"]

    def _record(value: float) -> None:
        nonlocal best_val, best_state
        if value > best_val:
            best_val = value
            best_state = _cpu_state(model)

    if entry is None:
        model.eval()
        entry = float(eval_genuine())
        _record(entry)

    def _checkpoint() -> None:
        if checkpoint_path is None:
            return
        _atomic_save({
            "step": step, "entry": entry, "best_val": best_val,
            "best_model": best_state, "model": _cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
        }, checkpoint_path)

    ran_this_invocation = 0
    model.train()
    for x, y in train_batches:
        if step >= steps or (stop_after is not None and ran_this_invocation >= stop_after):
            break
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = deployed_forward(x)
        ce = F.cross_entropy(student_logits, y)
        temperature = float(kd_temperature)
        kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        loss = (1.0 - kd_alpha) * ce + kd_alpha * kd
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        schedule.step()
        step += 1
        ran_this_invocation += 1
        if step % eval_every == 0:
            model.eval()
            _record(float(eval_genuine()))
            model.train()
        if checkpoint_path is not None and step % checkpoint_every == 0:
            _checkpoint()

    model.eval()
    last = float(eval_genuine())
    _record(last)
    if best_state is not None and best_val > last:
        model.load_state_dict(best_state)
        final = best_val
    else:
        final = last
    _checkpoint()
    return DeployedRiskResult(
        entry_genuine=float(entry), best_genuine=float(best_val),
        final_genuine=float(final), steps_run=step, resumed_from=resumed_from,
    )
