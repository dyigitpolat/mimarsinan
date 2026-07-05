"""Small shared helpers for pipeline steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.common.best_effort import best_effort
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def require_lif_spiking_mode(pipeline, step_name: str) -> None:
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
    plan = DeploymentPlan.of(pipeline)
    policy_for_spiking_mode(
        plan.spiking_mode, plan.ttfs_cycle_schedule
    ).require_backend_supported(backend=backend, context=step_name)


def run_optional_viz(step_name: str, fn: Callable[[], Any]) -> None:
    with best_effort(f"{step_name} visualization"):
        fn()


def safe_warmup_forward(model, input_shape, device) -> None:
    """Materialize Lazy modules with a throwaway forward.

    The dummy follows the MODEL's own parameter device (builders may leave a
    freshly built model on CPU regardless of the config device — a mismatched
    dummy would silently skip materialization); ``device`` is the param-less
    fallback.
    """
    with best_effort("warmup forward"):
        model.eval()
        param = next(iter(model.parameters()), None)
        target = param.device if param is not None else device
        dummy = torch.zeros((1, *tuple(input_shape)), device=target)
        with torch.no_grad():
            model(dummy)
