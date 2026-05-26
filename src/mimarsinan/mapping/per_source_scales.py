"""Per-source input scales for branching architectures.

Traverses the mapper graph and sets ``per_input_scales`` on each
perceptron so that the effective-weight formula uses the correct
activation_scale for each input channel -- even after a concat of
branches with different activation_scales.

For multi-input host-side ``ComputeOpMapper``s with diverging source
scales, this pass stamps ``per_source_scales`` / ``output_scale`` slots
on the mapper.  At IR emission time the mapper wraps its underlying
module in :class:`ScaleNormalizingWrapper` so the rate→absolute→rate
math runs uniformly for *any* multi-input op (not just Add).
"""

import torch

from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    ConcatMapper,
    InputMapper,
)
from mimarsinan.mapping.scale_broadcast import broadcast_scale_pair as _broadcast_scale_pair


def compute_per_source_scales(model_repr):
    """Traverse the mapper graph; set ``per_input_scales`` on each Perceptron.

    After this call every perceptron reachable from *model_repr* will have
    its ``per_input_scales`` attribute set to a 1-D ``torch.Tensor`` of
    shape ``(in_features,)``, where each element is the ``activation_scale``
    of the source feeding that particular input.

    For sequential models every element is identical (equivalent to the old
    scalar ``input_scale``).  For layers after a ``ConcatMapper`` the values
    vary by channel group.
    """
    model_repr._ensure_exec_graph()

    out_scales: dict = {}

    for node in model_repr._exec_order:
        deps = model_repr._deps.get(node, [])

        if isinstance(node, InputMapper):
            out_scales[node] = torch.ones(node.input_shape[0])

        elif hasattr(node, "perceptron"):
            src = _first_source_scales(deps, out_scales)
            if src is not None:
                _assign_per_input_scales(node.perceptron, src)

            n_out = node.perceptron.output_channels
            act_s = float(node.perceptron.activation_scale)
            out_scales[node] = torch.full((n_out,), act_s)

        elif isinstance(node, ConcatMapper):
            parts = [out_scales[d] for d in deps if d in out_scales]
            if parts:
                out_scales[node] = torch.cat(parts)

        elif isinstance(node, ComputeOpMapper):
            parts = [out_scales[d] for d in deps if d in out_scales]
            scale = _apply_compute_op_scale_policy(node, parts)
            if scale is not None:
                out_scales[node] = scale

        else:
            src = _first_source_scales(deps, out_scales)
            if src is not None:
                out_scales[node] = src


def _apply_compute_op_scale_policy(
    node: ComputeOpMapper, source_scales: list[torch.Tensor]
) -> torch.Tensor | None:
    """Decide whether ``node`` needs ``ScaleNormalizingWrapper`` stamping.

    A wrapper is required when **any** input source carries a non-uniform
    per-channel scale vector *or* when multiple sources disagree.  This
    covers two cases under one rule:

    * **Multi-input divergent scales** — e.g. ``Add(a, b)`` with
      ``s_a ≠ s_b``.  The wrapper rescales each input by its source scale
      so the wrapped module operates in absolute units.

    * **Unary input with per-channel heterogeneity** — e.g. a ``LayerNorm``
      downstream of a ``ConcatMapper`` that joins streams of different
      activation scales.  The wrapper rescales per-channel before the
      module computes its mean / variance / etc.

    When every source has a uniform scalar scale that matches across inputs,
    no wrapper is needed and the source scale is propagated unchanged.
    """
    if not source_scales:
        return None

    # Broadcast every source-scale pair to a common length so heterogeneity
    # checks below see comparable vectors.
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
    """True if ``scale`` is a vector with non-uniform entries."""
    if scale.numel() <= 1:
        return False
    return not torch.allclose(scale, scale[0].expand_as(scale))


def _all_sources_uniform(scales: list[torch.Tensor]) -> bool:
    """True if every source scale equals the first one (same shape + values)."""
    first = scales[0]
    return all(
        s.shape == first.shape and torch.allclose(s, first)
        for s in scales[1:]
    )


def _first_source_scales(deps, out_scales):
    """Return output scales of the first dependency that has them."""
    for d in deps:
        if d in out_scales:
            return out_scales[d]
    return None


def _assign_per_input_scales(perceptron, source_scales):
    """Expand per-channel *source_scales* to match *perceptron.input_features*."""
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

    # Dimension mismatch (e.g. after an EinopsRearrange that transposes axes).
    # Fall back to the mean — matches the old group-average behaviour and is
    # correct whenever all sources share the same activation_scale.
    mean_scale = source_scales.mean().item()
    perceptron.per_input_scales = torch.full((in_features,), mean_scale)
