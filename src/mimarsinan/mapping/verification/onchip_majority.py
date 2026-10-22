"""On-chip validity gate for a mapped IR graph; host MACs counted through the forward-MAC SSOT (``module_macs``)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.support.module_macs import module_macs as module_macs
from mimarsinan.mapping.support.host_contributors import (
    describe_host_holders,
    host_contributors_from_ir,
    subsumed_encoder_params,
)
from mimarsinan.mapping.support.scale_wrapper import (
    is_scale_wrapper as is_scale_wrapper,
    unwrap_scale_wrapper as unwrap_scale_wrapper,
)

# Validity threshold SSOT: a mapping whose on-chip params OR ops fraction is below
# DEFAULT_ONCHIP_FLOOR is INVALID (host-majority — an erroneous deployment); the
# MAJORITY threshold separates VALID (>= majority on both) from VALID_FLAGGED.
# Owned here so the pipeline gate, the registry defaults, and the research
# campaign scheduler read ONE source instead of re-hardcoding the literals.
DEFAULT_ONCHIP_FLOOR = 0.20
DEFAULT_ONCHIP_MAJORITY = 0.50


class OnchipMajorityError(ToolchainRefusal, ValueError):
    """A deployment maps the parameter MINORITY to chip cores (host-majority); a refusal by name."""


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


@dataclass(frozen=True)
class OnchipOpsBreakdown:
    """Logical host/on-chip split of a deployed model's forward MACs."""

    onchip_ops: int
    host_ops: int
    total_ops: int

    @property
    def fraction(self) -> float:
        if self.total_ops <= 0:
            return 0.0
        return self.onchip_ops / self.total_ops


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


def _batched(shape):
    return (1, *shape) if shape is not None else None


def count_host_ops(ir_graph: IRGraph) -> int:
    """Sum of forward MACs of the host-side ComputeOps, each one invocation at its declared per-sample shapes
    (``support/module_macs.py``, every output position): a split op (a host Linear or an encoding perceptron per
    column) counts every slice, a subsumed encoding perceptron counts at its op's shapes, and a multi-input op reads
    ``params['input_shapes']``. Reproduces ``estimate_onchip_fraction(metric='macs')``'s host total."""
    total = 0
    for op in ir_graph.get_compute_ops():
        module = op.params.get("module")
        if module is None:
            continue
        perceptron = getattr(module, "perceptron", None)
        unit = perceptron if perceptron is not None else module
        shapes = [_batched(tuple(shape)) for shape in op.params.get("input_shapes") or ()]
        total += module_macs(unit, _batched(op.input_shape), _batched(op.output_shape), shapes)
    return total


def compute_onchip_ops_fraction(
    ir_graph: IRGraph, *, total_ops: int
) -> OnchipOpsBreakdown:
    """Logical on-chip forward-MAC fraction = (total - host) / total.

    On-chip ops are the remainder after the host ComputeOps, never the raw
    crossbar op count: conv weight-bank reuse replicates one logical matmul
    across many cores, so summing per-core MACs over-counts (as tiling does for
    params). ``total_ops`` is the model's forward-MAC total (a model fact, like
    ``sum(model.parameters())`` is for params).
    """
    host_ops = count_host_ops(ir_graph)
    onchip_ops = int(total_ops) - host_ops
    return OnchipOpsBreakdown(
        onchip_ops=onchip_ops,
        host_ops=host_ops,
        total_ops=int(total_ops),
    )


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


#: The deploy-it-anyway escape, owned once. Every floor refusal ends with it,
#: so no remedy sentence repeats it (the two used to collide in one message).
ONCHIP_FLOOR_ESCAPE = (
    "Set deployment_parameters.onchip_majority_gate=false to deploy a "
    "deliberately host-heavy network, or lower onchip_min_fraction."
)


def onchip_placement_remedy(contributors, breakdown: OnchipParamBreakdown) -> str:
    """What ``encoding_layer_placement`` can do about this host majority.

    ``subsume`` runs the encoding layer host-side, ``offload`` maps it on chip —
    so the reachable fraction is today's on-chip params plus whatever the
    subsumed encoders hold. It is an UPPER bound: the negative-boundary
    subsume-forward policy (``negative_value_shift=off``) can host an encoder
    for a reason placement does not control. Placement only; the escape hatches
    are :data:`ONCHIP_FLOOR_ESCAPE`, appended by the refusal itself.
    """
    subsumed = subsumed_encoder_params(contributors)
    if subsumed <= 0 or breakdown.total_params <= 0:
        return (
            "No encoding layer is subsumed here, so encoding_layer_placement "
            "cannot move this host majority; the host units themselves must "
            "shrink."
        )
    reachable = (
        breakdown.onchip_params + subsumed
    ) / breakdown.total_params
    return (
        "encoding_layer_placement='offload' maps the encoding layer on chip "
        "instead of running it host-side, which would raise this to at most "
        f"{reachable:.2%}."
    )


def assert_onchip_majority_or_raise(
    ir_graph: IRGraph, *, total_params: int, min_fraction: float = 0.2
) -> OnchipParamBreakdown:
    """Raise :class:`OnchipMajorityError` when the on-chip fraction is below the floor.

    Raises only below the floor; between floor and majority a mapping is
    VALID_FLAGGED (see :func:`onchip_fraction.classify_validity`). The refusal
    names the host units it actually measured, largest first.
    """
    breakdown = compute_onchip_fraction(ir_graph, total_params=total_params)
    if breakdown.fraction < min_fraction:
        contributors = host_contributors_from_ir(ir_graph)
        raise OnchipMajorityError(
            "On-chip parameter majority violated: only "
            f"{breakdown.fraction:.2%} of the {breakdown.total_params} deployed "
            f"parameters are placed on chip cores "
            f"(on-chip={breakdown.onchip_params}, host={breakdown.host_params}), "
            f"below the required {min_fraction:.0%} floor. "
            f"{describe_host_holders(contributors)}. "
            f"{onchip_placement_remedy(contributors, breakdown)} "
            f"{ONCHIP_FLOOR_ESCAPE}"
        )
    return breakdown
