"""Small shared helpers for pipeline steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def require_lif_spiking_mode(pipeline, step_name: str) -> None:
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    spiking_mode = DeploymentPlan.of(pipeline).spiking_mode
    if spiking_mode != "lif":
        raise ValueError(
            f"{step_name} requires spiking_mode='lif'; got {spiking_mode!r}"
        )


def require_spiking_mode_supported(
    pipeline,
    step_name: str,
    *,
    backend: str,
) -> None:
    from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    plan = DeploymentPlan.of(pipeline)
    policy_for_spiking_mode(
        plan.spiking_mode, plan.ttfs_cycle_schedule
    ).require_backend_supported(backend=backend, context=step_name)


def run_optional_viz(step_name: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except Exception as exc:
        print(f"[{step_name}] Visualization failed (non-fatal): {exc}")


def safe_warmup_forward(model, input_shape, device) -> None:
    try:
        model.eval()
        dummy = torch.zeros((1, *tuple(input_shape)), device=device)
        with torch.no_grad():
            model(dummy)
    except Exception as exc:
        print(f"[safe_warmup_forward] Warmup forward failed (non-fatal): {exc}")
