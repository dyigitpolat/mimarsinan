"""On-chip parameter-majority validity gate for a mapped IR graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph


class OnchipMajorityError(ValueError):
    """A deployment maps the parameter MINORITY to chip cores (host-majority)."""


@dataclass(frozen=True)
class OnchipParamBreakdown:
    """Logical host/on-chip split of a deployed model's parameters."""

    onchip_params: int
    host_params: int
    total_params: int

    @property
    def fraction(self) -> float:
        if self.total_params <= 0:
            return 0.0
        return self.onchip_params / self.total_params


def _numel(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    if torch.is_tensor(value):
        return int(value.numel())
    return 0


def count_host_params(ir_graph: IRGraph) -> int:
    """Sum of unique host-side ComputeOp parameters (modules deduped by identity).

    A wrapped module's ``parameters()`` already includes bound constant tensors,
    so bare ``bound_tensors`` are only added when the op carries no module.
    """
    seen_modules: set[int] = set()
    total = 0
    for op in ir_graph.get_compute_ops():
        module = op.params.get("module")
        if module is not None and hasattr(module, "parameters"):
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            total += sum(_numel(p) for p in module.parameters())
            continue
        for tensor in op.params.get("bound_tensors") or []:
            total += _numel(tensor)
    return total


def compute_onchip_fraction(
    ir_graph: IRGraph, *, total_params: int
) -> OnchipParamBreakdown:
    """Logical on-chip fraction = (total - host) / total.

    On-chip is the logical unique-parameter remainder, not the raw crossbar
    footprint, since tiling replicates weights across many cores.
    """
    host_params = count_host_params(ir_graph)
    onchip_params = int(total_params) - host_params
    return OnchipParamBreakdown(
        onchip_params=onchip_params,
        host_params=host_params,
        total_params=int(total_params),
    )


def assert_onchip_majority_or_raise(
    ir_graph: IRGraph, *, total_params: int, min_fraction: float = 0.2
) -> OnchipParamBreakdown:
    """Raise :class:`OnchipMajorityError` when the on-chip fraction is below the floor.

    Raises only below the floor; between floor and majority a mapping is
    VALID_FLAGGED (see :func:`onchip_fraction.classify_validity`).
    """
    breakdown = compute_onchip_fraction(ir_graph, total_params=total_params)
    if breakdown.fraction < min_fraction:
        raise OnchipMajorityError(
            "On-chip parameter majority violated: only "
            f"{breakdown.fraction:.2%} of the {breakdown.total_params} deployed "
            f"parameters are placed on chip cores "
            f"(on-chip={breakdown.onchip_params}, host={breakdown.host_params}), "
            f"below the required {min_fraction:.0%} floor. The host-side "
            "ComputeOps (offloaded encoding Linear/Conv, classifier readout, "
            "attention) hold the parameter majority, so this mapping is not a "
            "genuine on-chip deployment."
        )
    return breakdown
