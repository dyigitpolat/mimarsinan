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

    Host-side parameters are the offloaded weights that run OFF the crossbar:
    the encoding Linear/Conv, the classifier readout, MultiheadAttention, etc.
    A wrapped ``nn.Module``'s ``parameters()`` already includes any bound
    constant tensors (``ComputeAdapter`` registers them as parameters), so they
    are only added separately when an op carries bare ``bound_tensors`` with no
    module. The same module instance shared across ops is counted once.
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

    On-chip is defined as ``total_params - host_params`` (the LOGICAL
    unique-parameter remainder), NOT the raw crossbar weight footprint: tiling
    replicates a perceptron's weights across many cores, so the physical core
    count over-counts and would exceed ``total_params``.
    """
    host_params = count_host_params(ir_graph)
    onchip_params = int(total_params) - host_params
    return OnchipParamBreakdown(
        onchip_params=onchip_params,
        host_params=host_params,
        total_params=int(total_params),
    )


def assert_onchip_majority_or_raise(
    ir_graph: IRGraph, *, total_params: int, min_fraction: float = 0.5
) -> OnchipParamBreakdown:
    """Raise :class:`OnchipMajorityError` when the on-chip fraction is below the floor.

    The chip is the deployment vehicle only when the parameter majority is
    physically placed on its cores; a host-majority mapping is not a genuine
    on-chip deployment.
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
