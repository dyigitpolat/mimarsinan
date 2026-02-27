"""Per-source input scales for branching architectures.

Traverses the mapper graph and sets ``per_input_scales`` on each
perceptron so that the effective-weight formula uses the correct
activation_scale for each input channel -- even after a concat of
branches with different activation_scales.
"""

import torch

from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ConcatMapper,
    AddMapper,
)


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

        elif isinstance(node, AddMapper):
            parts = [out_scales[d] for d in deps if d in out_scales]
            if len(parts) >= 2:
                out_scales[node] = (parts[0] + parts[1]) / 2.0
            elif parts:
                out_scales[node] = parts[0]

        else:
            src = _first_source_scales(deps, out_scales)
            if src is not None:
                out_scales[node] = src


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
