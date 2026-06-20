"""Polymorphic per-node scale propagation for the mapper graph.

Both :func:`compute_per_source_scales` (weight-quantization per-input scales) and
:func:`propagate_boundary_input_scales` (TTFS theta-out boundary scales) walk the
mapper exec graph in topological order and, at each node, derive an out-scale from
the node's source out-scales — branching on the *kind* of mapper. This module owns
the single graph walk (:func:`walk_out_scales`) and routes the per-node decision to
a polymorphic ``Mapper`` method, so a new mapper kind overrides ONE method instead
of being added to two (formerly three) ``isinstance`` chains.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from mimarsinan.mapping.support.scale_broadcast import (
    broadcast_scale_pair as _broadcast_scale_pair,
)


def walk_out_scales(model_repr, visit: Callable[[Any, list, dict], Any]) -> dict:
    """Topological walk of the mapper graph, accumulating per-node out-scales.

    ``visit(node, deps, out_scales)`` returns this node's out-scale (or ``None`` to
    record nothing). The returned dict maps node -> out-scale for nodes that
    produced one. Side effects (e.g. assigning ``per_input_scales``) happen inside
    ``visit``; this helper only owns the traversal + accumulation.
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


def perceptron_source_out_scale(perceptron) -> torch.Tensor:
    """Per-output-channel out-scale carried by a perceptron's ``activation_scale``.

    A per-channel theta (e.g. ttfs_theta_cotrain) is carried verbatim when its
    length matches the output width; mean-folded only on a length mismatch.
    """
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

    # Dimension mismatch (e.g. after EinopsRearrange transposes axes):
    # fall back to mean; correct when all sources share the same scale.
    mean_scale = source_scales.mean().item()
    perceptron.per_input_scales = torch.full((in_features,), mean_scale)


def perceptron_per_source_scale(node, deps, out_scales) -> torch.Tensor:
    """Shared ``propagate_source_scale`` body for perceptron-bearing mappers."""
    src = first_source_scale(deps, out_scales)
    if src is not None:
        assign_per_input_scales(node.perceptron, src)
    return perceptron_source_out_scale(node.perceptron)


def perceptron_boundary_scale(node, deps, out_scales, default) -> float:
    """Shared ``propagate_boundary_scale`` body for perceptron-bearing mappers."""
    perceptron = node.perceptron
    in_scale = mean_source_scale(deps, out_scales, default)
    perceptron.set_input_activation_scale(in_scale)
    return float(perceptron.activation_scale)


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
