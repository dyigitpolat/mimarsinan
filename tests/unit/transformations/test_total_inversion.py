"""Total effective->raw inversion at the PerceptronTransformer SSOT (W2 fix A).

Degenerate BN channels (|u| = |gamma/sigma| below epsilon, or raw-write
amplification beyond bound) must realize effective-bias deltas through the
normalization affine beta instead of dividing by u; raw weights on such
channels stay unchanged. Healthy channels stay bit-identical to the classic
inversion, and the non-finite guard keeps failing loud.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

SHIFT = 0.03125  # the LIF half-step delta (0.5 / Tq at Tq=16) from the t0_03 crater


def _bn_perceptron(u_values, out=None, inp=4, act_scale=1.0, seed=0):
    """Perceptron with BatchNorm1d whose fold factors u = gamma/sqrt(var+eps) are exact."""
    torch.manual_seed(seed)
    u = torch.tensor(u_values, dtype=torch.float32)
    out = out or u.numel()
    bn = nn.BatchNorm1d(out)
    with torch.no_grad():
        bn.running_var.fill_(0.0)  # u = weight / sqrt(eps)
        bn.weight.copy_(u * torch.sqrt(torch.tensor(bn.eps)))
        bn.running_mean.copy_(torch.linspace(-0.2, 0.3, out))
        bn.bias.copy_(torch.linspace(0.1, 0.5, out))
    p = Perceptron(out, inp, normalization=bn)
    p.set_activation_scale(act_scale)
    with torch.no_grad():
        p.layer.bias.copy_(torch.linspace(-0.4, 0.6, out))
    return p


def _classic_inverted_bias(p, transform):
    """The pre-fix raw-bias inversion: ((target*act - beta)/u) + mean."""
    t = PerceptronTransformer()
    u, beta, mean = t._get_u_beta_mean(p.normalization)
    target = transform(t.get_effective_bias(p))
    return ((target * p.activation_scale - beta) / u) + mean


class TestDegenerateBiasRouting:
    @pytest.mark.parametrize("dead_u", [1.13e-17, 2.572e-29])
    def test_dead_channel_bias_delta_is_bounded_and_exact(self, dead_u):
        """The crater class: |u| ~ 1e-17 / 1e-29 channels took 5.6e15 / 2.4e27
        finite raw writes; the total inversion must keep raw params bounded and
        the effective-domain result exact."""
        p = _bn_perceptron([0.8, dead_u, 1.2, dead_u], act_scale=2.018)
        raw_before = p.layer.bias.data.clone()
        transformer = PerceptronTransformer()
        entry_effective = transformer.get_effective_bias(p).clone()

        transformer.apply_effective_bias_transform(p, lambda b: b + SHIFT)

        raw_after = p.layer.bias.data
        assert torch.isfinite(raw_after).all()
        assert raw_after.abs().max() < 1e3, (
            f"raw bias must stay bounded, got {raw_after.abs().max():.3e}"
        )
        # Dead channels keep their raw bias; the delta lives in beta.
        torch.testing.assert_close(raw_after[1], raw_before[1])
        torch.testing.assert_close(raw_after[3], raw_before[3])
        # Effective-domain behavior is exact on every channel.
        torch.testing.assert_close(
            transformer.get_effective_bias(p), entry_effective + SHIFT,
            atol=1e-6, rtol=1e-5,
        )

    def test_crater_numbers_no_longer_amplify(self):
        """The exact t0_03 fingerprint: gamma=3.6e-20, var=1.8e-40, act=2.018;
        the classic inversion writes ~5.58e15 into the raw bias."""
        bn = nn.BatchNorm1d(2)
        with torch.no_grad():
            bn.running_var.copy_(torch.tensor([1.0, 1.8e-40]))
            bn.weight.copy_(torch.tensor([1.0, 3.6e-20]))
            bn.running_mean.zero_()
            bn.bias.copy_(torch.tensor([0.05, 0.02]))
        p = Perceptron(2, 4, normalization=bn)
        p.set_activation_scale(2.018)
        with torch.no_grad():
            p.layer.bias.copy_(torch.tensor([0.1, -0.2]))
        classic = _classic_inverted_bias(p, lambda b: b + SHIFT)
        assert classic[1].abs() > 1e14  # the defect this fix removes

        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b + SHIFT)
        assert p.layer.bias.data.abs().max() < 1e3

    def test_amplification_bound_routes_marginally_dead_channels(self):
        """|u| = 8e-6 passes a bare eps=1e-6 but takes a ~7.9e3x amplified raw
        write for the shift delta; the amplification bound must route it."""
        p = _bn_perceptron([1.0, 8e-6], act_scale=2.0)
        raw_before = p.layer.bias.data.clone()
        transformer = PerceptronTransformer()
        entry_effective = transformer.get_effective_bias(p).clone()

        transformer.apply_effective_bias_transform(p, lambda b: b + SHIFT)

        torch.testing.assert_close(p.layer.bias.data[1], raw_before[1])
        assert p.layer.bias.data.abs().max() < 1e3
        torch.testing.assert_close(
            transformer.get_effective_bias(p), entry_effective + SHIFT,
            atol=1e-6, rtol=1e-5,
        )

    def test_beta_write_realizes_target_exactly_through_normalization(self):
        p = _bn_perceptron([1e-17, 1e-17], act_scale=1.5)
        transformer = PerceptronTransformer()
        target = transformer.get_effective_bias(p) + SHIFT
        transformer.apply_effective_bias_transform(p, lambda b: b + SHIFT)
        torch.testing.assert_close(
            transformer.get_effective_bias(p), target, atol=1e-7, rtol=1e-6
        )


class TestHealthyChannelsBitIdentical:
    def test_bias_transform_matches_classic_inversion_bitwise(self):
        p = _bn_perceptron([0.7, 1.3, 0.05, 2.4])
        expected = _classic_inverted_bias(p, lambda b: b + SHIFT)
        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b + SHIFT)
        assert torch.equal(p.layer.bias.data, expected)

    def test_mixed_tensor_healthy_positions_bitwise_dead_positions_kept(self):
        p = _bn_perceptron([0.9, 1e-20, 1.1, 1e-20])
        raw_before = p.layer.bias.data.clone()
        expected_healthy = _classic_inverted_bias(p, lambda b: b + SHIFT)
        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b + SHIFT)
        raw_after = p.layer.bias.data
        assert torch.equal(raw_after[0], expected_healthy[0])
        assert torch.equal(raw_after[2], expected_healthy[2])
        assert torch.equal(raw_after[1], raw_before[1])
        assert torch.equal(raw_after[3], raw_before[3])

    def test_healthy_beta_untouched(self):
        p = _bn_perceptron([0.7, 1.3])
        beta_before = p.normalization.bias.data.clone()
        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b + SHIFT)
        assert torch.equal(p.normalization.bias.data, beta_before)

    def test_identity_transform_healthy_matches_classic_bitwise(self):
        p = _bn_perceptron([0.6, 1.7, 0.02])
        expected = _classic_inverted_bias(p, lambda b: b)
        PerceptronTransformer().apply_effective_bias_transform(p, lambda b: b)
        assert torch.equal(p.layer.bias.data, expected)


class TestWeightTransform:
    def test_dead_channels_keep_raw_weights(self):
        p = _bn_perceptron([0.8, 1e-18, 1.2])
        w_before = p.layer.weight.data.clone()
        transformer = PerceptronTransformer()
        transformer.apply_effective_weight_transform(p, lambda w: w * 0.5)
        w_after = p.layer.weight.data
        assert torch.isfinite(w_after).all()
        assert torch.equal(w_after[1], w_before[1])
        # Dead rows stay effectively ~0 through the fold.
        assert transformer.get_effective_weight(p)[1].abs().max() < 1e-6

    def test_healthy_rows_bitwise_match_classic_inversion(self):
        p = _bn_perceptron([0.8, 1e-18, 1.2])
        t = PerceptronTransformer()
        u, _, _ = t._get_u_beta_mean(p.normalization)
        eff = t.get_effective_weight(p)
        expected = ((eff * 0.5) * p.activation_scale / 1.0) / u.unsqueeze(-1)
        t.apply_effective_weight_transform(p, lambda w: w * 0.5)
        w_after = p.layer.weight.data
        assert torch.equal(w_after[0], expected[0])
        assert torch.equal(w_after[2], expected[2])


class TestFailLoudGuardRemains:
    def test_nonfinite_target_on_live_channel_still_raises(self):
        p = _bn_perceptron([0.8, 1.2])

        def poison(b):
            out = b.clone()
            out[0] = float("nan")
            return out

        with pytest.raises(RuntimeError, match="non-finite"):
            PerceptronTransformer().apply_effective_bias_transform(p, poison)
