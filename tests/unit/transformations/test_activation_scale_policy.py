"""Tests for the selectable activation-scale calibration policies.

Covers the new Rueckauer-style percentile-norm baseline alongside the existing
count-quantile default, and the byte-identical guarantee for the default path.
"""

import torch
import pytest

from mimarsinan.transformations.activation_scale_policy import (
    ActivationScalePolicy,
    MaxNormPolicy,
    PercentileNormPolicy,
    CountQuantilePolicy,
    make_activation_scale_policy,
    DEFAULT_ACTIVATION_SCALE_POLICY,
)
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
    scale_from_activations,
    DEFAULT_SCALE_QUANTILE,
    MIN_SCALE,
    PRUNED_THRESHOLD,
)


def _toy_activations_with_outliers():
    """1000 bulk activations in [0, 1] plus a handful of large outliers."""
    bulk = torch.linspace(0.0, 1.0, steps=1000, dtype=torch.float32)
    outliers = torch.tensor([50.0, 80.0, 100.0], dtype=torch.float32)
    return torch.cat([bulk, outliers])


class TestPercentileNormPolicy:
    def test_p100_recovers_the_literal_max(self):
        """p=100 is the textbook max-norm baseline: scale == max activation."""
        acts = _toy_activations_with_outliers()
        policy = PercentileNormPolicy(percentile=100.0)
        scale = policy.scale(acts)
        assert scale == pytest.approx(float(acts.max().item()))
        assert scale == pytest.approx(100.0)

    def test_robust_percentile_ignores_outliers(self):
        """p=99 robust-norm clamps the 3 outliers (0.3% of 1003 samples).

        With p=99 we discard the top 1% (~10 samples), which subsumes all 3
        outliers, so the scale lands in the [0, 1] bulk -- not on an outlier.
        """
        acts = _toy_activations_with_outliers()
        policy = PercentileNormPolicy(percentile=99.0)
        scale = policy.scale(acts)
        expected = torch.quantile(
            acts, 0.99, interpolation="higher"
        ).item()
        assert scale == pytest.approx(expected)
        # Robust scale is FAR below the max -- the outliers are clamped out.
        assert scale < 5.0
        assert scale <= 1.0
        assert scale < float(acts.max().item())

    def test_percentile_999_is_the_exact_quantile(self):
        """p=99.9 scale == the 99.9th percentile of the full distribution."""
        acts = _toy_activations_with_outliers()
        scale = PercentileNormPolicy(percentile=99.9).scale(acts)
        expected = torch.quantile(acts, 0.999, interpolation="higher").item()
        assert scale == pytest.approx(expected)
        # Still strictly below the literal max (clamps the very top sample).
        assert scale < float(acts.max().item())

    def test_percentile_is_monotone_in_p(self):
        acts = _toy_activations_with_outliers()
        s90 = PercentileNormPolicy(percentile=90.0).scale(acts)
        s999 = PercentileNormPolicy(percentile=99.9).scale(acts)
        s100 = PercentileNormPolicy(percentile=100.0).scale(acts)
        assert s90 <= s999 <= s100

    def test_percentile_over_full_distribution_not_positive_only(self):
        """Rueckauer robust-norm uses the WHOLE distribution, including zeros."""
        acts = torch.cat(
            [torch.zeros(900, dtype=torch.float32), torch.linspace(0.1, 1.0, 100)]
        )
        # 90% of samples are exactly zero -> the 50th percentile is 0.
        policy = PercentileNormPolicy(percentile=50.0)
        scale = policy.scale(acts)
        # min_scale floor keeps it strictly positive.
        assert scale == pytest.approx(MIN_SCALE)

    def test_min_scale_floor(self):
        acts = torch.zeros(100, dtype=torch.float32)
        policy = PercentileNormPolicy(percentile=99.9, min_scale=1e-3)
        assert policy.scale(acts) == pytest.approx(1e-3)

    def test_empty_returns_unit_scale(self):
        policy = PercentileNormPolicy(percentile=99.9)
        assert policy.scale(torch.empty(0)) == pytest.approx(1.0)


class TestMaxNormPolicy:
    def test_scale_is_the_max(self):
        acts = _toy_activations_with_outliers()
        assert MaxNormPolicy().scale(acts) == pytest.approx(100.0)

    def test_max_norm_equals_percentile_100(self):
        acts = _toy_activations_with_outliers()
        assert MaxNormPolicy().scale(acts) == pytest.approx(
            PercentileNormPolicy(percentile=100.0).scale(acts)
        )


class TestCountQuantilePolicyMatchesLegacy:
    """The DEFAULT policy must be byte-identical to scale_from_activations."""

    @pytest.mark.parametrize(
        "acts",
        [
            _toy_activations_with_outliers(),
            torch.cat([torch.zeros(500), torch.rand(500)]),
            torch.rand(1234) * 7.0,
            torch.empty(0),
            torch.zeros(50),
            torch.tensor([1e-12, 2e-12, 3e-12]),  # all below pruned threshold
        ],
    )
    @pytest.mark.parametrize("quantile", [0.5, 0.9, DEFAULT_SCALE_QUANTILE, 1.0])
    def test_byte_identical_to_legacy(self, acts, quantile):
        legacy = scale_from_activations(
            acts, quantile=quantile, min_scale=MIN_SCALE
        )
        policy = CountQuantilePolicy(quantile=quantile, min_scale=MIN_SCALE)
        new = policy.scale(acts)
        # Bit-exact: same torch ops, same args.
        assert new == legacy or (new != new and legacy != legacy)

    def test_default_policy_is_count_quantile(self):
        assert DEFAULT_ACTIVATION_SCALE_POLICY == "count_quantile"

    def test_default_factory_matches_legacy_default(self):
        acts = _toy_activations_with_outliers()
        policy = make_activation_scale_policy(DEFAULT_ACTIVATION_SCALE_POLICY)
        legacy = scale_from_activations(
            acts, quantile=DEFAULT_SCALE_QUANTILE, min_scale=MIN_SCALE
        )
        assert policy.scale(acts) == legacy


class TestFactory:
    def test_selects_percentile(self):
        policy = make_activation_scale_policy("percentile_norm", percentile=99.9)
        assert isinstance(policy, PercentileNormPolicy)
        assert policy.percentile == pytest.approx(99.9)

    def test_selects_max(self):
        assert isinstance(make_activation_scale_policy("max_norm"), MaxNormPolicy)

    def test_selects_count_quantile(self):
        policy = make_activation_scale_policy("count_quantile", quantile=0.95)
        assert isinstance(policy, CountQuantilePolicy)
        assert policy.quantile == pytest.approx(0.95)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            make_activation_scale_policy("nonexistent")

    def test_all_policies_are_activation_scale_policies(self):
        for name in ("count_quantile", "percentile_norm", "max_norm"):
            assert isinstance(
                make_activation_scale_policy(name), ActivationScalePolicy
            )


class TestOutlierContract:
    """The whole point of robust-norm: outliers must not inflate the scale."""

    def test_single_huge_outlier_dominates_max_not_percentile(self):
        acts = torch.cat(
            [torch.full((9999,), 0.5), torch.tensor([1000.0])]
        )
        max_scale = MaxNormPolicy().scale(acts)
        robust_scale = PercentileNormPolicy(percentile=99.0).scale(acts)
        assert max_scale == pytest.approx(1000.0)
        # 1 outlier in 10000 is the top 0.01% -> clamped at p=99.
        assert robust_scale == pytest.approx(0.5)
        assert robust_scale < max_scale / 100.0
