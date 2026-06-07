"""Layer replacement must re-sync TTFSActivation bias references.

``TTFSActivation`` holds the owning perceptron's ``layer.bias`` (subtracted
from pre-activations, re-added as a per-cycle ramp). Steps that REPLACE
``perceptron.layer`` (normalization fusion, bring_back_bias) orphan that
reference — the 2026-06-07 offload-cascaded incident: the driver subtracted a
stale pre-fusion bias, pouring ``b_new − b_old`` into the ramp every cycle.
"""

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import (
    TTFSActivation,
    refresh_perceptron_bias_references,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron


def _perceptron_with_ttfs(normalization):
    torch.manual_seed(0)
    p = Perceptron(6, 8, normalization=normalization, base_activation_name="ReLU")
    activation = TTFSActivation(
        T=4,
        activation_scale=p.activation_scale,
        input_scale=p.input_activation_scale,
        bias=p.layer.bias,
    )
    p.base_activation = activation
    p.activation = activation
    return p, activation


class TestRefreshHelper:
    def test_refresh_repoints_every_ttfs_activation(self):
        p, activation = _perceptron_with_ttfs(nn.Identity())
        p.layer = nn.Linear(8, 6)
        assert activation._bias is not p.layer.bias
        refresh_perceptron_bias_references(p)
        assert activation._bias is p.layer.bias

    def test_refresh_handles_biasless_layer(self):
        p, activation = _perceptron_with_ttfs(nn.Identity())
        p.layer = nn.Linear(8, 6, bias=False)
        refresh_perceptron_bias_references(p)
        assert activation._bias is None


class TestFusionRefreshesBias:
    def test_fuse_into_perceptron_repoints_ttfs_bias(self):
        p, activation = _perceptron_with_ttfs(nn.BatchNorm1d(6))
        # give the BN non-trivial statistics so fusion changes the bias
        p.train()
        with torch.no_grad():
            for _ in range(4):
                p(torch.randn(16, 8))
        p.eval()
        fuse_into_perceptron(p, device="cpu")
        assert activation._bias is p.layer.bias, (
            "fusion replaced perceptron.layer; the TTFSActivation must be "
            "re-pointed at the fused bias"
        )
        # and the held values are the fused ones
        np.testing.assert_array_equal(
            activation._bias.detach().numpy(), p.layer.bias.detach().numpy(),
        )

    def test_fusion_noop_for_identity_norm_keeps_reference(self):
        p, activation = _perceptron_with_ttfs(nn.Identity())
        fuse_into_perceptron(p, device="cpu")
        assert activation._bias is p.layer.bias
