"""Normalization fusion must be behavior-preserving under the cascade forward.

The cascaded TTFS segment walk feeds each interior core ``norm(W s_t + b)`` per
cycle; the additive constant of that pre-activation is the *effective* bias
``(b - mean)*u + beta``, not the raw ``layer.bias``. Subtracting the raw bias
poured ``fused_b - b`` into the ramp every cycle, so Normalization Fusion (an
analytical transformation) changed the genuine forward's behavior — the
2026-06-07/08 cascaded+offload incident (QV 0.9468 -> NF 0.9199).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron


def _calibrate_activation_scales(flow):
    """Set each perceptron's activation_scale to its observed max pre-activation
    so cascade values land strictly inside the (0, 1) staircase (a 1.0 scale on
    an untrained model saturates the clamp and hides bias errors)."""
    records = {}
    hooks = []
    for p in flow.get_perceptrons():
        def _record(mod, inp, out, p=p):
            records[id(p)] = max(records.get(id(p), 1e-3), float(out.detach().abs().max()))
        target = p.normalization if not isinstance(p.normalization, nn.Identity) else p.layer
        hooks.append(target.register_forward_hook(_record))
    with torch.no_grad():
        flow(torch.rand(8, 1, 28, 28))
    for h in hooks:
        h.remove()
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(records[id(p)]))


def _ttfs_flow(placement: str, T: int = 8, bias: bool = True):
    torch.manual_seed(0)
    m = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    )
    # Non-trivial BN statistics/affine so fusion actually changes the bias.
    with torch.no_grad():
        m.train()
        for _ in range(3):
            m(torch.rand(8, 1, 28, 28))
        m.patch_bn.weight.copy_(torch.rand_like(m.patch_bn.weight) + 0.5)
        m.patch_bn.bias.copy_(torch.randn_like(m.patch_bn.bias))
    m.eval()
    flow = convert_torch_model(
        m, input_shape=(1, 28, 28), num_classes=10,
        encoding_layer_placement=placement,
    )
    _calibrate_activation_scales(flow)
    for p in flow.get_perceptrons():
        if not bias and not isinstance(p.normalization, nn.Identity):
            p.layer.bias = None
        p.set_activation(TTFSActivation(
            T=T,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    flow.get_mapper_repr().assign_perceptron_indices()
    # The pipeline freezes norm statistics before the cascade ever runs
    # (FrozenStatsNormalization); eval() models that frozen-affine contract.
    return flow.double().eval()


def _cascade_node_values(flow, x, T=8):
    """(output, {perceptron_name: decoded value}) under the genuine cascade walk."""
    driver = TTFSSegmentForward(flow.get_mapper_repr(), T)
    with torch.no_grad():
        out, recorder = driver.forward_with_node_values(x)
    values = {
        node.perceptron.name: v
        for node, v in recorder.items()
        if getattr(node, "perceptron", None) is not None
    }
    return out, values


def _fused(flow):
    fused = copy.deepcopy(flow)
    for p in fused.get_perceptrons():
        fuse_into_perceptron(p, device="cpu")
    return fused


def _assert_fusion_invariant(placement: str, bias: bool = True):
    flow = _ttfs_flow(placement, bias=bias)
    torch.manual_seed(7)
    x = torch.rand(4, 1, 28, 28, dtype=torch.float64)
    out_pre, values_pre = _cascade_node_values(flow, x)
    out_post, values_post = _cascade_node_values(_fused(flow), x)
    assert values_pre.keys() == values_post.keys() and values_pre
    for name in values_pre:
        torch.testing.assert_close(
            values_post[name], values_pre[name], atol=1e-9, rtol=0.0,
            msg=lambda m, name=name: f"node {name} changed under fusion:\n{m}",
        )
    torch.testing.assert_close(out_post, out_pre, atol=1e-9, rtol=0.0)


class TestFusionInvariance:
    def test_offload_cascade_forward_invariant_under_fusion(self):
        """Offload: the BN'd patch_embed runs as an interior cascade core; the
        genuine forward (every node's decoded value) must not change when the
        norm is fused into the layer."""
        _assert_fusion_invariant("offload")

    def test_subsume_cascade_forward_invariant_under_fusion(self):
        """Subsume control: patch_embed is the value-domain encoding entry."""
        _assert_fusion_invariant("subsume")

    def test_offload_biasless_normed_layer_invariant_under_fusion(self):
        """A biasless layer under BN still has a nonzero effective bias
        ((0 - mean)*u + beta); the walk must charge it."""
        _assert_fusion_invariant("offload", bias=False)


class TestDriveTimeBiasContract:
    def test_stored_bias_reference_restored_after_drive(self):
        """The policy installs the effective bias for the walk and restores the
        raw ``layer.bias`` reference afterwards (the picklable stored contract)."""
        flow = _ttfs_flow("offload")
        x = torch.rand(2, 1, 28, 28, dtype=torch.float64)
        _ = _cascade_node_values(flow, x)
        for p in flow.get_perceptrons():
            for mod in p.modules():
                if isinstance(mod, TTFSActivation):
                    assert mod._bias is p.layer.bias

    def test_gradient_reaches_norm_affine_params(self):
        """Training through the cascade must update the norm's affine params via
        the effective-bias path (the deployed fused-bias semantics)."""
        flow = _ttfs_flow("offload")
        driver = TTFSSegmentForward(flow.get_mapper_repr(), 8)
        x = torch.rand(2, 1, 28, 28, dtype=torch.float64)
        out = driver(x)
        out.sum().backward()
        normed = [p for p in flow.get_perceptrons()
                  if not isinstance(p.normalization, nn.Identity)]
        assert normed
        for p in normed:
            g = p.normalization.bias.grad
            assert g is not None and torch.isfinite(g).all()
