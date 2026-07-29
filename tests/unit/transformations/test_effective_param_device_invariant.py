"""W0.7 — the effective-parameter device/dtype invariant.

Every input to an effective-parameter expression must be materialized on the
perceptron's OWN parameter device and dtype. A bias-free layer (a BN-paired
convolution, as in the CIFAR VGG-8 / ResNet-20 vehicles) has no ``layer.bias``,
so its additive term is fabricated; fabricating it with a bare ``torch.zeros``
put it on CPU/float32 and blew up the mapping step of every CUDA converted
vehicle with::

    RuntimeError: Expected all tensors to be on the same device, but found at
    least two devices, cuda:0 and cpu!

reached through ``conv2d_mapper._map_to_ir -> get_effective_bias``.

The CPU suite reproduces the precondition two ways without a GPU:
  * the ``meta`` device — mixing meta with CPU raises the same class of error
    ("Tensor on device meta is not on the expected device cpu!");
  * float64 — a bare float32 create silently downcasts the effective bias of a
    float64 bias-free perceptron (0-dim scalars do not promote a 1-dim tensor).
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.scale_propagation import (
    assign_per_input_scales,
    perceptron_source_out_scale,
)
from mimarsinan.models.perceptron_mixer.perceptron import (
    Perceptron,
    effective_preactivation_bias,
    layer_bias_or_zeros,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)


def _bias_free_perceptron(out=4, inp=6):
    return Perceptron(out, inp, bias=False, normalization=nn.BatchNorm1d(out))


class TestLayerBiasOrZeros:
    def test_returns_the_live_bias_parameter_when_there_is_one(self):
        p = Perceptron(4, 6, bias=True, normalization=nn.Identity())
        assert layer_bias_or_zeros(p) is p.layer.bias

    def test_bias_free_zero_follows_the_layer_device(self):
        p = _bias_free_perceptron().to("meta")
        zero = layer_bias_or_zeros(p)
        assert zero.device == p.layer.weight.device
        assert zero.shape == (p.layer.weight.shape[0],)

    def test_bias_free_zero_follows_the_layer_dtype(self):
        p = _bias_free_perceptron().double()
        assert layer_bias_or_zeros(p).dtype is torch.float64


class TestGetEffectiveBiasDeviceInvariant:
    """The crash site: ``((layer_bias - mean) * u + beta) / activation_scale``."""

    def test_bias_free_normalized_perceptron_maps_off_cpu(self):
        """Pre-fix this raised the cuda/cpu mismatch (meta/cpu on this suite)."""
        p = _bias_free_perceptron().to("meta")
        eff = PerceptronTransformer().get_effective_bias(p)
        assert eff.device == p.layer.weight.device

    def test_bias_free_unnormalized_perceptron_maps_off_cpu(self):
        p = Perceptron(4, 6, bias=False, normalization=nn.Identity()).to("meta")
        eff = PerceptronTransformer().get_effective_bias(p)
        assert eff.device == p.layer.weight.device

    def test_biased_perceptron_maps_off_cpu(self):
        p = Perceptron(4, 6, bias=True, normalization=nn.BatchNorm1d(4)).to("meta")
        eff = PerceptronTransformer().get_effective_bias(p)
        assert eff.device == p.layer.weight.device

    def test_bias_free_perceptron_keeps_the_layer_dtype(self):
        """A 0-dim float64 ``activation_scale`` does not promote a 1-dim float32
        zero (wrapped-scalar promotion), so a bare create silently downcast the
        whole effective bias of a float64 bias-free perceptron."""
        p = Perceptron(4, 6, bias=False, normalization=nn.Identity()).double()
        assert PerceptronTransformer().get_effective_bias(p).dtype is torch.float64

    def test_bias_free_normalized_perceptron_keeps_the_layer_dtype(self):
        p = _bias_free_perceptron().double()
        assert PerceptronTransformer().get_effective_bias(p).dtype is torch.float64

    def test_bias_free_value_is_unchanged_on_the_cpu_default_path(self):
        """Byte-identical default: on a CPU float32 perceptron the fix is a no-op."""
        p = _bias_free_perceptron()
        with torch.no_grad():
            p.normalization.running_mean.normal_()
            p.normalization.running_var.uniform_(0.5, 2.0)
            p.normalization.weight.normal_()
            p.normalization.bias.normal_()
        pt = PerceptronTransformer()
        u = p.normalization.weight / torch.sqrt(
            p.normalization.running_var + p.normalization.eps
        )
        expected = (
            (torch.zeros(4) - p.normalization.running_mean) * u + p.normalization.bias
        ) / p.activation_scale
        torch.testing.assert_close(pt.get_effective_bias(p), expected.detach())


class TestEffectivePreactivationBiasDeviceInvariant:
    def test_bias_free_normalized_perceptron_stays_on_its_device(self):
        p = _bias_free_perceptron().to("meta")
        assert effective_preactivation_bias(p).device == p.layer.weight.device

    def test_bias_free_unnormalized_perceptron_still_reports_no_bias(self):
        """Contract preserved: no normalization and no bias means no additive term."""
        p = Perceptron(4, 6, bias=False, normalization=nn.Identity())
        assert effective_preactivation_bias(p) is None


class TestPerInputScalesDeviceInvariant:
    """``per_input_scales`` is folded into effective weights, so the scale
    vector must be born on the perceptron's device too."""

    def test_scalar_activation_scale_out_scale_follows_the_perceptron(self):
        p = Perceptron(4, 6, normalization=nn.Identity()).to("meta")
        assert perceptron_source_out_scale(p).device == p.layer.weight.device

    def test_mean_folded_per_channel_theta_follows_the_perceptron(self):
        p = Perceptron(4, 6, normalization=nn.Identity())
        p.activation_scale = nn.Parameter(torch.rand(7) + 0.5, requires_grad=False)
        p = p.to("meta")
        assert perceptron_source_out_scale(p).device == p.layer.weight.device

    def test_mean_fallback_stamp_follows_the_source_scales(self):
        p = Perceptron(4, 6, normalization=nn.Identity()).to("meta")
        source = torch.full((4,), 0.5, device="meta")
        assign_per_input_scales(p, source)  # 6 inputs % 4 channels != 0 -> mean fold
        assert p.per_input_scales is not None
        assert p.per_input_scales.device == source.device

    def test_scalar_out_scale_value_is_unchanged_on_cpu(self):
        p = Perceptron(4, 6, normalization=nn.Identity())
        p.set_activation_scale(2.5)
        torch.testing.assert_close(
            perceptron_source_out_scale(p), torch.full((4,), 2.5)
        )
