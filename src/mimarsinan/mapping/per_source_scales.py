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

        elif (
            isinstance(node, ComputeOpMapper)
            and len(node.get_source_mappers()) >= 2
        ):
            parts = [out_scales[d] for d in deps if d in out_scales]
            out_scales[node] = _apply_multi_input_scale_policy(node, parts)

        else:
            src = _first_source_scales(deps, out_scales)
            if src is not None:
                out_scales[node] = src


def _apply_multi_input_scale_policy(
    node: ComputeOpMapper, source_scales: list[torch.Tensor]
) -> torch.Tensor:
    """Handle source-scale propagation for a multi-input ``ComputeOpMapper``.

    When source scales diverge, write ``per_source_scales`` and
    ``output_scale`` on the mapper; ``ComputeOpMapper._maybe_wrap_for_scales``
    stamps the :class:`ScaleNormalizingWrapper` around the underlying
    module at IR-emission time.  When sources agree, the wrapper is
    skipped and the bare module is emitted.
    """
    if len(source_scales) < 2:
        return source_scales[0] if source_scales else torch.ones(1)

    # Broadcast every pair to a common length (matches the historical
    # AddMapper pair-broadcast on length-mismatched scales).
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

    coherent = all(torch.allclose(normalized[0], s) for s in normalized[1:])
    if coherent:
        return normalized[0]

    output_scale = node.combine_source_scales(normalized)
    node.per_source_scales = list(normalized)
    node.output_scale = output_scale
    return output_scale


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
