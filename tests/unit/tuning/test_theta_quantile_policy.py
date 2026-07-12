"""[R4/S1] S-aware theta-quantile policy: the calibration quantile follows the grid.

The sync memo's measured optimum (sync_deployment_exactness.md §3.3): the right
theta loading is an (S, marginal)-dependent quantile that FALLS as the grid
coarsens — S=4 hops selected 0.90–0.995 (representative 0.95), S=8 ~0.99,
S=16 mostly 0.995–1.0. The policy maps grid levels to a quantile and only ever
DEFLATES the configured base (never inflates a mode's proven choice).
"""

import math

import pytest
import torch

from mimarsinan.tuning.orchestration.theta_quantile_policy import (
    S_AWARE_QUANTILE_ANCHORS,
    effective_theta_quantile,
    s_aware_quantile_for_levels,
)


class TestAnchors:
    def test_measured_anchor_values(self):
        assert s_aware_quantile_for_levels(4) == pytest.approx(0.95)
        assert s_aware_quantile_for_levels(8) == pytest.approx(0.99)
        assert s_aware_quantile_for_levels(16) == pytest.approx(0.995)
        assert s_aware_quantile_for_levels(32) == pytest.approx(1.0)

    def test_anchor_table_is_the_ssot(self):
        for levels, quantile in S_AWARE_QUANTILE_ANCHORS.items():
            assert s_aware_quantile_for_levels(levels) == pytest.approx(quantile)

    def test_below_lowest_anchor_clamps_to_lowest(self):
        assert s_aware_quantile_for_levels(2) == pytest.approx(0.95)

    def test_above_highest_anchor_clamps_to_highest(self):
        assert s_aware_quantile_for_levels(64) == pytest.approx(1.0)
        assert s_aware_quantile_for_levels(1024) == pytest.approx(1.0)

    def test_interpolation_between_anchors_is_log2(self):
        # Halfway between S=8 (0.99) and S=16 (0.995) in log2 space.
        expected = 0.99 + (0.995 - 0.99) * (math.log2(11) - 3.0)
        assert s_aware_quantile_for_levels(11) == pytest.approx(expected)

    def test_monotone_nondecreasing_in_levels(self):
        values = [s_aware_quantile_for_levels(s) for s in range(2, 65)]
        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_invalid_levels_fail_loud(self):
        with pytest.raises(ValueError):
            s_aware_quantile_for_levels(0)
        with pytest.raises(ValueError):
            s_aware_quantile_for_levels(-4)


class TestEffectiveQuantile:
    """The policy only deflates: effective = min(base, policy(levels))."""

    def test_low_s_deflates_the_lif_base(self):
        assert effective_theta_quantile(4, 0.99) == pytest.approx(0.95)

    def test_low_s_deflates_the_sync_full_quantile(self):
        assert effective_theta_quantile(4, 1.0) == pytest.approx(0.95)
        assert effective_theta_quantile(8, 1.0) == pytest.approx(0.99)

    def test_high_s_never_inflates_the_base(self):
        # policy(16) = 0.995 > lif base 0.99: the base wins.
        assert effective_theta_quantile(16, 0.99) == pytest.approx(0.99)
        assert effective_theta_quantile(32, 0.99) == pytest.approx(0.99)

    def test_continuous_modes_keep_the_base(self):
        assert effective_theta_quantile(None, 0.99) == pytest.approx(0.99)
        assert effective_theta_quantile(None, 1.0) == pytest.approx(1.0)


class TestMaterialThetaReduction:
    """At S=4 the chosen theta must sit materially below the q0.99 theta on a
    heavy-tailed (ReLU-like) activation marginal — the whole point of the lever."""

    def test_s4_theta_materially_below_q99_on_lognormal(self):
        from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
            scale_from_activations,
        )

        torch.manual_seed(0)
        # Heavy-tailed positives: lognormal with sigma=1.5 (decades of spread,
        # the mixer-hop regime measured in the memos).
        flat = torch.distributions.LogNormal(0.0, 1.5).sample((20000,))

        q_s4 = effective_theta_quantile(4, 0.99)
        theta_s4 = scale_from_activations(flat, quantile=q_s4)
        theta_q99 = scale_from_activations(flat, quantile=0.99)

        assert q_s4 == pytest.approx(0.95)
        assert theta_s4 < 0.9 * theta_q99, (
            f"S=4 theta {theta_s4} is not materially below the q0.99 theta {theta_q99}"
        )
