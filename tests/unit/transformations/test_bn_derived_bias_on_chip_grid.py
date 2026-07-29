"""The BN-derived effective bias of a BIAS-FREE perceptron must land on the
chip grid (W0.9).

A BN-paired convolution (``nn.Conv2d(..., bias=False)`` + ``nn.BatchNorm2d``)
is the standard CIFAR vehicle idiom -- ``CifarVGG8`` and ``CifarResNet20`` are
built entirely from it. Such a perceptron still HAS an effective bias: the whole
of it is derived from the normalization affine,
``b_eff = ((0 - mean) * u + beta) / activation_scale``, and the mappers export
exactly that value (``PerceptronTransformer.get_effective_bias``).

Before W0.9 the weight-quantization projection silently skipped it -- there was
no raw ``layer.bias`` Parameter to invert into -- so the deployed bias was a
function of UNQUANTIZED BN statistics and landed off the integer lattice, which
``assert_effective_parameters_on_chip_grid`` (correctly) refuses.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
)
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

BITS = 8
Q_MAX = (2 ** (BITS - 1)) - 1


def _trained_bn(num_features, *, spatial=False, seed=0):
    """A BatchNorm with non-trivial running stats and a non-trivial affine."""
    torch.manual_seed(seed)
    bn = nn.BatchNorm2d(num_features) if spatial else nn.BatchNorm1d(num_features)
    bn.train()
    with torch.no_grad():
        shape = (32, num_features, 3, 3) if spatial else (32, num_features)
        bn(torch.randn(*shape) * 2.5 + 1.25)
        bn.weight.data = torch.randn(num_features).abs() + 0.4
        bn.bias.data = torch.randn(num_features) * 1.5
    bn.eval()
    return bn


def _bias_free_fc_perceptron(out_channels=8, in_features=6, seed=0):
    torch.manual_seed(seed)
    p = Perceptron(out_channels, in_features, bias=False,
                   normalization=_trained_bn(out_channels, seed=seed))
    p.set_activation_scale(1.0)
    assert p.layer.bias is None
    return p


def _bias_free_conv_perceptron(out_channels=6, in_channels=4, seed=1):
    """A BN-paired conv perceptron in the packaged (im2col patch-Linear) form
    ``Conv2DPerceptronMapper`` builds -- the exact VGG-8 / ResNet-20 idiom."""
    torch.manual_seed(seed)
    p = Perceptron(out_channels, in_channels * 3 * 3, bias=False,
                   normalization=_trained_bn(out_channels, seed=seed))
    p.output_channel_axis = 1
    p.set_activation_scale(1.0)
    assert p.layer.bias is None
    return p


def _quantize(perceptron, *, two_scale=False, rate=1.0):
    NormalizationAwarePerceptronQuantization(
        bits=BITS, device="cpu", rate=rate, two_scale=two_scale,
    ).transform(perceptron)


class TestBiasFreeBNPerceptronReachesTheChipGrid:
    """The real WQ projection must put the DERIVED bias on the chip grid."""

    @pytest.mark.parametrize("two_scale", [False, True])
    def test_fc_perceptron_passes_the_verification_gate(self, two_scale):
        p = _bias_free_fc_perceptron()
        _quantize(p, two_scale=two_scale)
        assert_effective_parameters_on_chip_grid(p, Q_MAX)

    @pytest.mark.parametrize("two_scale", [False, True])
    def test_conv_perceptron_passes_the_verification_gate(self, two_scale):
        p = _bias_free_conv_perceptron()
        _quantize(p, two_scale=two_scale)
        assert_effective_parameters_on_chip_grid(p, Q_MAX)

    def test_derived_bias_actually_moves_onto_the_lattice(self):
        """Regression guard: pre-fix the projection left the bias BYTE-IDENTICAL
        (an off-lattice float), which is what the gate caught."""
        p = _bias_free_fc_perceptron()
        transformer = PerceptronTransformer()
        before = transformer.get_effective_bias(p).clone()
        _quantize(p)
        after = transformer.get_effective_bias(p)
        scaled = after * p.bias_scale
        assert torch.allclose(scaled, torch.round(scaled), atol=1e-4)
        assert not torch.allclose(before, after, atol=1e-6), (
            "the float bias was already on the lattice; the fixture is not "
            "exercising the projection"
        )


class TestProjectionIsRealizedInTheForwardPass:
    """The on-chip value must be what the model actually computes."""

    def test_normalization_output_matches_the_quantized_effective_bias(self):
        p = _bias_free_fc_perceptron()
        _quantize(p)
        expected = PerceptronTransformer().get_effective_bias(p) * p.activation_scale
        with torch.no_grad():
            # additive constant of normalization(layer(x)) at x = 0
            actual = p.normalization(p.layer(torch.zeros(2, p.input_features)))[0]
        assert torch.allclose(actual, expected, atol=1e-5), (
            "the deployed bias and the forward pass disagree"
        )

    def test_effective_weight_is_untouched_by_the_bias_write(self):
        """beta enters the fold additively: writing it must not move the weight
        grid the projection just installed."""
        p = _bias_free_fc_perceptron()
        transformer = PerceptronTransformer()
        _quantize(p)
        w_before = transformer.get_effective_weight(p).clone()
        transformer.apply_effective_bias_transform(p, lambda b: b + 3.0)
        assert torch.allclose(transformer.get_effective_weight(p), w_before, atol=0.0)

    def test_quantization_error_stays_within_one_grid_step(self):
        """No silent drift: the grid projection ROUNDS, it does not relocate.

        Measured past the (function-preserving, now also reachable on bias-free
        perceptrons) constant-OFF saturation canonicalization, which is a
        separate, deliberate transform.
        """
        from mimarsinan.transformations.perceptron.bias_saturation import (
            clip_off_saturated_effective_bias,
        )

        p = _bias_free_fc_perceptron()
        transformer = PerceptronTransformer()
        clip_off_saturated_effective_bias(p, transformer=transformer)
        before = transformer.get_effective_bias(p).clone()
        _quantize(p)
        after = transformer.get_effective_bias(p)
        step = 1.0 / float(p.bias_scale)
        assert float((after - before).abs().max()) <= 0.5 * step + 1e-6

    def test_arbitrary_effective_bias_is_realized_exactly(self):
        p = _bias_free_fc_perceptron()
        transformer = PerceptronTransformer()
        target = torch.linspace(-2.0, 2.0, p.output_channels)
        transformer.apply_effective_bias_transform(p, lambda _: target)
        assert torch.allclose(transformer.get_effective_bias(p), target, atol=1e-5)


class TestPruningContractSurvives:
    def test_pruned_rows_keep_an_exactly_zero_effective_bias(self):
        p = _bias_free_fc_perceptron()
        mask = torch.zeros(p.output_channels, dtype=torch.bool)
        mask[1] = True
        mask[4] = True
        row_mask = mask.clone()
        p.layer.register_buffer("prune_mask", mask.unsqueeze(-1).expand_as(p.layer.weight).clone())
        p.normalization.register_buffer("_prune_row_mask", row_mask)
        with torch.no_grad():
            p.layer.weight.data[mask] = 0.0
            p.normalization.running_mean.data[row_mask] = 0.0
            p.normalization.bias.data[row_mask] = 0.0

        _quantize(p)

        eff_b = PerceptronTransformer().get_effective_bias(p)
        assert float(eff_b[row_mask].abs().max()) == 0.0
        assert float(p.normalization.bias.data[row_mask].abs().max()) == 0.0
        assert_effective_parameters_on_chip_grid(p, Q_MAX)


class TestNoRealizationSeamFailsLoud:
    """Where the derived bias genuinely cannot be written, say so -- never
    silently drop the projection (that is the defect this unit repairs)."""

    def test_identity_normalization_no_op_transform_is_allowed(self):
        p = Perceptron(4, 6, bias=False, normalization=nn.Identity())
        p.set_activation_scale(2.0)
        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b * 2)
        assert p.layer.bias is None
        assert torch.allclose(
            PerceptronTransformer().get_effective_bias(p), torch.zeros(4)
        )

    def test_identity_normalization_moving_transform_raises(self):
        p = Perceptron(4, 6, bias=False, normalization=nn.Identity())
        p.set_activation_scale(1.0)
        with pytest.raises(RuntimeError, match="no seam"):
            PerceptronTransformer().apply_effective_bias_transform(
                p, lambda b: b + 1.0
            )

    def test_writability_is_declared_per_seam(self):
        identity = Perceptron(4, 6, bias=False, normalization=nn.Identity())
        assert PerceptronTransformer.normalization_bias_seam(identity) is None
        assert not PerceptronTransformer.effective_bias_is_writable(identity)

        bn_paired = _bias_free_fc_perceptron()
        assert (
            PerceptronTransformer.normalization_bias_seam(bn_paired)
            is bn_paired.normalization.bias
        )
        assert PerceptronTransformer.effective_bias_is_writable(bn_paired)

        with_bias = Perceptron(4, 6, bias=True, normalization=nn.Identity())
        assert PerceptronTransformer.effective_bias_is_writable(with_bias)


class TestRealVehicleLayerClass:
    """End-to-end over the converted-model seam, on the VGG-8 / ResNet-20 idiom."""

    class _BiasFreeConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(8)
            self.act1 = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(8)
            self.act2 = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8 * 4 * 4, 5)

        def forward(self, x):
            x = self.pool(self.act1(self.bn1(self.conv1(x))))
            x = self.pool(self.act2(self.bn2(self.conv2(x))))
            return self.fc(self.flatten(x))

    def _converted_perceptrons(self):
        from mimarsinan.torch_mapping.converter import convert_torch_model

        torch.manual_seed(3)
        model = self._BiasFreeConvNet()
        model.train()
        with torch.no_grad():
            model(torch.randn(16, 3, 16, 16) * 2.0 + 0.5)
            for bn in (model.bn1, model.bn2):
                bn.bias.data = torch.randn(8) * 1.5
                bn.weight.data = torch.randn(8).abs() + 0.4
        model.eval()
        flow = convert_torch_model(model, (3, 16, 16), 5, device="cpu")
        return flow.get_perceptrons()

    def test_vehicle_class_is_bias_free_and_bn_paired(self):
        perceptrons = self._converted_perceptrons()
        bias_free_bn = [
            p for p in perceptrons
            if p.layer.bias is None
            and not isinstance(p.normalization, nn.Identity)
        ]
        assert bias_free_bn, (
            "fixture no longer reproduces the bias-free BN-paired vehicle class"
        )

    def test_every_converted_perceptron_passes_the_gate_after_quantization(self):
        perceptrons = self._converted_perceptrons()
        for p in perceptrons:
            _quantize(p)
        for p in perceptrons:
            assert_effective_parameters_on_chip_grid(p, Q_MAX)
