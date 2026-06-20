"""Promote per-perceptron ``activation_scale`` (theta) to a per-output-channel
trainable Parameter so a deployed-cascade fine-tune co-optimises the firing-gain
(theta) WITH the weights — the key lever of the near-lossless cascaded-TTFS recipe
(``docs/research_artifacts_for_cascaded_ttfs_tuning/51_near_lossless_recipe.md``).

``perceptron.set_activation_scale`` only copies ``.data`` into the existing
(non-trainable) parameter, so installing a NEW trainable per-channel param requires
REBINDING ``activation_scale`` on the perceptron AND every activation node that
references it; otherwise the optimiser trains a tensor the forward never reads.
Encoding/segment-entry layers are left fixed (their scale is pinned by the
input-encoding contract — see ``gain_correction``/``scale_aware_boundaries``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _scale_carrying_nodes(perceptron):
    """Activation modules under ``perceptron`` (excluding the perceptron itself) that
    carry an ``activation_scale`` — e.g. ``TTFSActivation`` and a blend target."""
    for module in perceptron.modules():
        if module is perceptron:
            continue
        if isinstance(getattr(module, "activation_scale", None), torch.Tensor):
            yield module


def promote_activation_scale_per_channel(model, *, skip_encoding: bool = True) -> list:
    """Rebind each (non-encoding) perceptron's ``activation_scale`` to a
    per-output-channel ``requires_grad`` Parameter (seeded from its current value),
    on the perceptron AND every node that references it. Returns the new params.

    Idempotent: a second call re-syncs node references to the perceptron's param."""
    params = []
    for perceptron in model.get_perceptrons():
        if skip_encoding and getattr(perceptron, "is_encoding_layer", False):
            continue
        scale = perceptron.activation_scale.detach()
        out_dim = int(perceptron.layer.weight.shape[0])
        vec = (
            scale * torch.ones(out_dim, dtype=scale.dtype, device=scale.device)
            if scale.dim() == 0
            else scale.clone()
        )
        param = nn.Parameter(vec.contiguous(), requires_grad=True)
        perceptron.activation_scale = param
        for node in _scale_carrying_nodes(perceptron):
            node.activation_scale = param
        params.append(param)
    return params
