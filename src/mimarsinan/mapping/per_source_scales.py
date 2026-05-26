"""Per-source activation scales for branching architectures.

Sets ``per_input_scales`` on each Perceptron so weight quantization absorbs
heterogeneous channel scales (typically from ``ConcatMapper``).  For
``ComputeOpMapper`` whose inputs carry heterogeneous scales — either across
multiple sources or within one source's channels — populates
``per_source_scales`` / ``output_scale`` slots so emission stamps
:class:`ScaleNormalizingWrapper` around the wrapped module.
"""

import torch

from mimarsinan.mapping.mapping_utils import (
    ComputeOpMapper,
    ConcatMapper,
    InputMapper,
)
from mimarsinan.mapping.scale_broadcast import broadcast_scale_pair as _broadcast_scale_pair


def compute_per_source_scales(model_repr):
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


def _first_source_scales(deps, out_scales):
    for d in deps:
        if d in out_scales:
            return out_scales[d]
    return None


def _assign_per_input_scales(perceptron, source_scales):
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
