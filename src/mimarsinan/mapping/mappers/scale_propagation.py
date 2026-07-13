"""Polymorphic per-node scale propagation for the mapper graph."""

from __future__ import annotations

from typing import Any, Callable

import torch

from mimarsinan.mapping.support.scale_broadcast import (
    broadcast_scale_pair as _broadcast_scale_pair,
)


def walk_out_scales(model_repr, visit: Callable[[Any, list, dict], Any]) -> dict:
    """Topological walk of the mapper graph accumulating per-node out-scales.

    visit(node, deps, out_scales) returns this node's out-scale (or None to record nothing); side effects happen inside visit.
    """
    model_repr._ensure_exec_graph()
    out_scales: dict = {}
    for node in model_repr._exec_order:
        deps = model_repr._deps.get(node, [])
        scale = visit(node, deps, out_scales)
        if scale is not None:
            out_scales[node] = scale
    return out_scales


def first_source_scale(deps, out_scales):
    """First source's recorded out-scale (graph order), or ``None``."""
    for d in deps:
        if d in out_scales:
            return out_scales[d]
    return None


def present_source_scales(deps, out_scales) -> list:
    """Recorded out-scales of a node's sources, in dep order."""
    return [out_scales[d] for d in deps if d in out_scales]


def mean_source_scale(deps, out_scales, default):
    """Mean of a node's recorded source out-scales (``default`` if none)."""
    present = present_source_scales(deps, out_scales)
    if not present:
        return default
    return sum(present) / len(present)


def max_source_scale(deps, out_scales, default):
    """Max of a node's recorded source out-scales (``default`` if none).

    The fan-in coverage rule for lane-parallel joins (§6b contract-1): a scalar
    wire scale must cover the widest lane; equal-θ fan-ins reduce to the mean.
    """
    present = present_source_scales(deps, out_scales)
    if not present:
        return default
    return max(float(s) for s in present)


def perceptron_source_out_scale(perceptron) -> torch.Tensor:
    """Per-output-channel out-scale from a perceptron's activation_scale; a per-channel theta is carried verbatim when its length matches output width, else mean-folded."""
    n_out = perceptron.output_channels
    act = perceptron.activation_scale
    if isinstance(act, torch.Tensor) and act.dim() > 0:
        vec = act.detach().to(torch.float32).reshape(-1)
        return (
            vec.clone()
            if vec.numel() == n_out
            else torch.full((n_out,), float(vec.mean()))
        )
    return torch.full((n_out,), float(act))


def assign_per_input_scales(perceptron, source_scales) -> None:
    """Stamp ``per_input_scales`` on a perceptron from its source out-scale vector."""
    in_features = perceptron.input_features
    n_channels = len(source_scales)

    if n_channels == 0:
        return

    if in_features == n_channels:
        perceptron.per_input_scales = source_scales.clone()
        return

    if in_features % n_channels == 0:
        spatial = in_features // n_channels
        perceptron.per_input_scales = source_scales.repeat_interleave(spatial)
        return

    mean_scale = source_scales.mean().item()
    perceptron.per_input_scales = torch.full((in_features,), mean_scale)


def perceptron_per_source_scale(node, deps, out_scales) -> torch.Tensor:
    """Shared ``propagate_source_scale`` body for perceptron-bearing mappers."""
    src = first_source_scale(deps, out_scales)
    if src is not None:
        assign_per_input_scales(node.perceptron, src)
    return perceptron_source_out_scale(node.perceptron)


def arm_compute_op_wrap_slots(model_repr) -> None:
    """Stamp ONLY the ComputeOp wrap slots (``per_source_scales`` /
    ``output_scale``) from the current perceptron thetas, so the deployed
    ScaleNormalizingWrapper and its NF twin agree from the install seam on.

    Never touches ``per_input_scales`` — the weight-fold currency stays owned
    by ``compute_per_source_scales`` at the WQ / mapping seams. A no-op on
    scalar-theta models (uniform scales never trigger the wrap policy)."""

    def visit(node, deps, out_scales):
        perceptron = getattr(node, "perceptron", None)
        if perceptron is not None:
            return perceptron_source_out_scale(perceptron)
        return node.propagate_source_scale(deps, out_scales)

    walk_out_scales(model_repr, visit)


def perceptron_boundary_scale(node, deps, out_scales, default) -> float:
    """Shared ``propagate_boundary_scale`` body for perceptron-bearing mappers.

    The boundary walk is scalar by contract (scale_aware_boundaries): a
    per-channel theta mean-collapses here exactly as in the pure read twin."""
    perceptron = node.perceptron
    in_scale = mean_source_scale(deps, out_scales, default)
    perceptron.set_input_activation_scale(in_scale)
    scale = perceptron.activation_scale
    if isinstance(scale, torch.Tensor) and scale.dim() > 0:
        return float(scale.detach().to(torch.float64).mean())
    return float(scale)


def apply_compute_op_scale_policy(node, source_scales: list) -> torch.Tensor | None:
    """ComputeOp per-source scale policy: wrap heterogeneous/non-uniform fan-in."""
    if not source_scales:
        return None

    normalized = list(source_scales)
    for i in range(1, len(normalized)):
        s_first, s_i = _broadcast_scale_pair(normalized[0], normalized[i])
        normalized[0] = s_first
        normalized[i] = s_i
    target_len = normalized[0].shape[0]
    for i in range(1, len(normalized)):
        if normalized[i].shape[0] != target_len:
            normalized[i] = torch.full(
                (target_len,), normalized[i].mean().item(),
                dtype=normalized[i].dtype,
            )

    needs_wrap = (
        any(_is_per_channel_heterogeneous(s) for s in normalized)
        or (len(normalized) > 1 and not _all_sources_uniform(normalized))
    )

    if not needs_wrap:
        return normalized[0]

    output_scale = node.combine_source_scales(normalized)
    node.per_source_scales = list(normalized)
    node.output_scale = output_scale
    return output_scale


def _is_per_channel_heterogeneous(scale: torch.Tensor) -> bool:
    if scale.numel() <= 1:
        return False
    return not torch.allclose(scale, scale[0].expand_as(scale))


def _all_sources_uniform(scales: list[torch.Tensor]) -> bool:
    first = scales[0]
    return all(
        s.shape == first.shape and torch.allclose(s, first)
        for s in scales[1:]
    )
