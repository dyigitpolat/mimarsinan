"""Per-channel trainable theta (activation_scale) promotion for the deployed cascade.

The near-lossless cascaded-TTFS recipe co-trains theta (the per-neuron firing-gain
threshold) WITH the weights through the genuine cascade. ``set_activation_scale``
only copies ``.data`` into the existing (non-trainable) parameter, so installing a
NEW trainable per-channel param requires REBINDING ``activation_scale`` on the
perceptron AND every activation node that references it — otherwise the optimiser
trains a tensor the forward never reads. These tests pin that contract.
"""

from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_fixtures import build_cascade_flow  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.spiking.theta_cotrain import (  # noqa: E402
    promote_activation_scale_per_channel,
)


def _flow():
    flow, calib = build_cascade_flow(depth=3, width=6, in_dim=6, out_dim=4, S=4, seed=0)
    return flow, calib


def test_promotes_non_encoding_to_per_channel_trainable():
    flow, _ = _flow()
    params = promote_activation_scale_per_channel(flow)
    assert len(params) >= 1
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        s = p.activation_scale
        assert s.requires_grad, "co-trained theta must require grad"
        assert s.dim() == 1 and s.numel() == p.layer.weight.shape[0], (
            "theta must be per-output-channel"
        )


def test_encoding_layer_theta_left_fixed():
    flow, _ = _flow()
    before = {
        id(p): (p.activation_scale.detach().clone(), p.activation_scale.requires_grad)
        for p in flow.get_perceptrons()
        if getattr(p, "is_encoding_layer", False)
    }
    promote_activation_scale_per_channel(flow)
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            scale, req = before[id(p)]
            assert p.activation_scale.requires_grad == req
            torch.testing.assert_close(p.activation_scale.detach(), scale)


def test_node_references_same_param_object():
    """The activation node (and any nested target) must point at the SAME promoted
    param so its forward reads what the optimiser trains."""
    flow, _ = _flow()
    promote_activation_scale_per_channel(flow)
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        for m in p.modules():
            if isinstance(m, TTFSActivation):
                assert m.activation_scale is p.activation_scale, (
                    "node theta must be the same object as the perceptron's"
                )


def test_theta_in_model_parameters_deduplicated():
    flow, _ = _flow()
    params = promote_activation_scale_per_channel(flow)
    model_param_ids = {id(p) for p in flow.parameters()}
    for t in params:
        assert id(t) in model_param_ids, "promoted theta must be a model parameter"
    # the same object registered on perceptron + node must not double-count
    thetas_in_model = [p for p in flow.parameters() if id(p) in {id(t) for t in params}]
    assert len(thetas_in_model) == len(params)


def test_gradient_reaches_theta_through_genuine_cascade():
    flow, calib = _flow()
    params = promote_activation_scale_per_channel(flow)
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    drv = TTFSSegmentForward(flow.get_mapper_repr(), 4)
    logits = drv(calib.double())
    logits.sum().backward()
    assert any(t.grad is not None and torch.any(t.grad != 0) for t in params), (
        "co-trained theta must receive a nonzero gradient from the genuine cascade"
    )


def test_per_source_scales_handles_per_channel_theta():
    """Deployment robustness: compute_per_source_scales (weight-quant step) must NOT
    crash on per-channel theta — it used float(activation_scale) (scalar-only)."""
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales

    flow, _ = _flow()
    promote_activation_scale_per_channel(flow)
    compute_per_source_scales(flow.get_mapper_repr())  # was: ValueError on per-channel theta


def test_idempotent_promotion_keeps_single_param():
    """Re-promoting (e.g. after a node rebuild) must not leave the node pointing at a
    stale param: a second call re-syncs node and perceptron to one object."""
    flow, _ = _flow()
    promote_activation_scale_per_channel(flow)
    promote_activation_scale_per_channel(flow)
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        for m in p.modules():
            if isinstance(m, TTFSActivation):
                assert m.activation_scale is p.activation_scale
