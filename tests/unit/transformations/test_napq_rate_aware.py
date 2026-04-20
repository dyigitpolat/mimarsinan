"""Tests for rate-aware ``NormalizationAwarePerceptronQuantization``.

Before this fix, ``transform()`` ignored the ``rate`` argument and always
produced full-step quantization. Combined with the random-mask mix in
``PerceptronTransformTuner._mix_params``, partial rates produced incoherent
hybrid networks whose accuracy was catastrophically worse than either
endpoint. The tuner therefore fast-failed on every partial rate and only
ever recovered via the forced rate=1.0 one-shot inside ``_after_run``.

The fix makes quantization itself rate-aware by linearly interpolating
between the FP value and the quantized value in weight-value space:

    rescaled = rate * q(param) + (1 - rate) * param

This produces a continuous path from the FP model (rate=0) to the fully
quantized model (rate=1), and at every intermediate rate the model is a
coherent small perturbation of the FP model.
"""

import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)


def _make_perceptron(seed=0):
    torch.manual_seed(seed)
    p = Perceptron(4, 8, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    # Non-trivial but small effective weights so the fully-quantized model
    # rounds into a meaningful distribution of quanta.
    with torch.no_grad():
        p.layer.weight.data.mul_(0.3)
        if p.layer.bias is not None:
            p.layer.bias.data.mul_(0.3)
    return p


class TestRateAwareQuantization:
    """At rate=0 the transform is identity; at rate=1 it is full
    quantization; at intermediate rates it is a linear interpolation in
    weight-value space."""

    def test_rate_zero_is_identity(self):
        p = _make_perceptron(seed=0)
        orig_w = p.layer.weight.data.clone()
        orig_b = None if p.layer.bias is None else p.layer.bias.data.clone()

        napq = NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=0.0)
        napq.transform(p)

        assert torch.allclose(p.layer.weight.data, orig_w, atol=1e-6), (
            "rate=0 must leave effective weights unchanged"
        )
        if orig_b is not None:
            assert torch.allclose(p.layer.bias.data, orig_b, atol=1e-6), (
                "rate=0 must leave effective bias unchanged"
            )

    def test_rate_one_is_full_quantization(self):
        """rate=1 must produce exactly the same result as the legacy
        full-quantization path, i.e. effective weights lie on the
        quantization grid defined by ``parameter_scale``."""
        p = _make_perceptron(seed=1)

        napq = NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0)
        napq.transform(p)

        pt = PerceptronTransformer()
        eff_w = pt.get_effective_weight(p)
        eff_b = pt.get_effective_bias(p)
        scale = float(p.parameter_scale)

        # rate=1 output * scale must be integer-valued.
        assert torch.allclose(eff_w * scale, torch.round(eff_w * scale), atol=1e-3), (
            "rate=1 must produce integer-scaled effective weights"
        )
        assert torch.allclose(eff_b * scale, torch.round(eff_b * scale), atol=1e-3), (
            "rate=1 must produce integer-scaled effective bias"
        )

    def test_rate_half_is_midpoint(self):
        """At rate=0.5 the effective weight should equal the midpoint of
        the FP and the fully-quantized effective weight."""
        # Capture the FP effective weight.
        p_fp = _make_perceptron(seed=2)
        pt = PerceptronTransformer()
        eff_fp = pt.get_effective_weight(p_fp).clone()

        # Capture the fully-quantized effective weight with a fresh copy.
        p_q = _make_perceptron(seed=2)
        NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0).transform(p_q)
        eff_q = pt.get_effective_weight(p_q).clone()

        # Now the half-rate transform.
        p_half = _make_perceptron(seed=2)
        NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=0.5).transform(p_half)
        eff_half = pt.get_effective_weight(p_half)

        expected = 0.5 * eff_q + 0.5 * eff_fp
        assert torch.allclose(eff_half, expected, atol=1e-4), (
            "rate=0.5 must linearly interpolate between FP and quantized "
            f"effective weights.\nmax|diff|={torch.max(torch.abs(eff_half - expected)).item()}"
        )

    def test_partial_rate_is_smooth_perturbation(self):
        """At small rates, the effective weight is a small perturbation of
        the FP effective weight, bounded by ``rate * max_quantization_err``."""
        p_fp = _make_perceptron(seed=3)
        pt = PerceptronTransformer()
        eff_fp = pt.get_effective_weight(p_fp).clone()

        p_q = _make_perceptron(seed=3)
        NormalizationAwarePerceptronQuantization(bits=8, device="cpu", rate=1.0).transform(p_q)
        eff_q = pt.get_effective_weight(p_q).clone()
        max_q_err = torch.max(torch.abs(eff_q - eff_fp)).item()

        for rate in [0.1, 0.25, 0.5, 0.75]:
            p = _make_perceptron(seed=3)
            NormalizationAwarePerceptronQuantization(
                bits=8, device="cpu", rate=rate
            ).transform(p)
            eff = pt.get_effective_weight(p)

            actual_err = torch.max(torch.abs(eff - eff_fp)).item()
            allowed = rate * max_q_err + 1e-5
            assert actual_err <= allowed, (
                f"At rate={rate}, perturbation {actual_err:.5f} exceeds "
                f"rate * max_q_err = {allowed:.5f}"
            )

    def test_parameter_scale_always_set_to_full_scale(self):
        """The ``parameter_scale`` recorded on the perceptron must always
        reflect the full-range scale (derived from p_max at rate=1), not
        a rate-scaled variant. Downstream IR mapping uses this scale."""
        p_fp = _make_perceptron(seed=4)
        p_q = _make_perceptron(seed=4)
        NormalizationAwarePerceptronQuantization(
            bits=8, device="cpu", rate=1.0
        ).transform(p_q)
        full_scale = float(p_q.parameter_scale)

        for rate in [0.1, 0.5, 0.9]:
            p = _make_perceptron(seed=4)
            NormalizationAwarePerceptronQuantization(
                bits=8, device="cpu", rate=rate
            ).transform(p)
            assert float(p.parameter_scale) == full_scale, (
                "parameter_scale must match the full-range scale "
                "regardless of rate (downstream IR mapping depends on it)."
            )

    def test_default_rate_is_one(self):
        """``rate=1.0`` must remain the constructor default, so every
        existing call site that does not specify a rate gets full
        quantization unchanged."""
        import inspect

        sig = inspect.signature(NormalizationAwarePerceptronQuantization.__init__)
        default = sig.parameters["rate"].default
        assert default == 1.0, (
            f"Default rate must remain 1.0 for backwards compat, got {default}"
        )


class TestRateInterpolationIsMonotone:
    """The per-weight distance from the FP value should be monotone
    non-decreasing in ``rate`` (it equals ``rate * |q(w) - w|``)."""

    def test_distance_from_fp_grows_with_rate(self):
        p_fp = _make_perceptron(seed=5)
        pt = PerceptronTransformer()
        eff_fp = pt.get_effective_weight(p_fp).clone()

        prev = -1.0
        for rate in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            p = _make_perceptron(seed=5)
            NormalizationAwarePerceptronQuantization(
                bits=8, device="cpu", rate=rate
            ).transform(p)
            eff = pt.get_effective_weight(p)
            dist = torch.mean(torch.abs(eff - eff_fp)).item()
            assert dist >= prev - 1e-7, (
                f"Distance from FP must be monotone in rate; at rate={rate}, "
                f"dist={dist} < previous {prev}"
            )
            prev = dist
