"""[calculus §17.13] the activation-scale policy: θ as the argmin of deployed
distortion D(θ) = clipping + resolution, not a fixed quantile.

Measured motivation (t2_04, T=32): the 0.99-quantile θ costs 4.04pp at the AQ
staircase install; the per-layer optimum costs 0.65pp.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.transformations.activation_scale_policy import (
    deployed_distortion,
    optimal_theta,
)


def test_distortion_is_the_two_named_terms():
    """D(θ) = E[(a−θ)²·1(a>θ)] + (θ/L)²/12·P(a≤θ) — computed, not approximated."""
    acts = torch.tensor([0.5, 1.0, 4.0])
    theta, levels = 1.0, 32
    clip = ((4.0 - 1.0) ** 2) / 3.0
    resolution = (theta / levels) ** 2 / 12.0 * (2.0 / 3.0)
    assert deployed_distortion(acts, theta, levels) == pytest.approx(
        clip + resolution
    )


def test_optimum_is_interior_and_beats_both_extremes():
    """The whole point: clipping and resolution pull opposite ways, so the
    argmin is interior — a tiny θ clips catastrophically, a huge θ coarsens."""
    torch.manual_seed(0)
    acts = torch.randn(20000).abs()
    levels = 32
    t = optimal_theta(acts, levels)
    assert t > 0
    d_opt = deployed_distortion(acts, t, levels)
    assert d_opt < deployed_distortion(acts, t * 0.25, levels)
    assert d_opt < deployed_distortion(acts, t * 8.0, levels)


def test_heavy_tail_moves_the_optimum_above_the_quantile():
    """The measured t2_04 shape: with a heavy tail the 0.99 quantile clips far
    too aggressively, so θ* sits above it."""
    torch.manual_seed(0)
    bulk = torch.rand(20000) * 0.3
    tail = torch.rand(200) * 6.0 + 2.0
    acts = torch.cat([bulk, tail])
    q99 = float(torch.quantile(acts, 0.99))
    assert optimal_theta(acts, 32) > q99


def test_more_levels_admit_a_larger_theta():
    """Resolution cost scales as (θ/L)²: a finer grid tolerates more range."""
    torch.manual_seed(0)
    acts = torch.randn(20000).abs()
    assert optimal_theta(acts, 128) >= optimal_theta(acts, 8)


def test_weights_shift_the_optimum_but_never_the_shape():
    """A scalar Fisher weight rescales D uniformly, so the argmin is unchanged
    — the weighting only matters ACROSS units, which is where it is applied."""
    torch.manual_seed(0)
    acts = torch.randn(5000).abs()
    assert optimal_theta(acts, 32, weight=1.0) == pytest.approx(
        optimal_theta(acts, 32, weight=7.5)
    )


def test_degenerate_inputs_fail_safe_not_loud():
    """Pruned/dead units are normal, not errors: fall back to the floor."""
    assert optimal_theta(torch.zeros(100), 32) == pytest.approx(1.0)
    assert optimal_theta(torch.tensor([]), 32) == pytest.approx(1.0)


def test_optimum_is_deterministic():
    torch.manual_seed(0)
    acts = torch.randn(10000).abs()
    assert optimal_theta(acts, 32) == optimal_theta(acts, 32)


class TestRegisteredPolicy:
    """The argmin ships as a policy in the EXISTING registry, not a new seam."""

    def _heavy_tailed(self):
        torch.manual_seed(0)
        return torch.cat([torch.rand(20000) * 0.3, torch.rand(200) * 6.0 + 2.0])

    def test_registry_exposes_min_distortion(self):
        from mimarsinan.transformations.activation_scale_policy import (
            make_activation_scale_policy,
        )

        acts = self._heavy_tailed()
        policy = make_activation_scale_policy("min_distortion", levels=32)
        assert policy.scale(acts) == pytest.approx(optimal_theta(acts, 32))

    def test_it_beats_the_default_quantile_on_the_measured_shape(self):
        from mimarsinan.transformations.activation_scale_policy import (
            make_activation_scale_policy,
        )

        acts = self._heavy_tailed()
        q = make_activation_scale_policy("count_quantile").scale(acts)
        d = make_activation_scale_policy("min_distortion", levels=32).scale(acts)
        assert d > q
        assert deployed_distortion(acts, d, 32) < deployed_distortion(acts, q, 32)

    def test_levels_are_required_not_guessed(self):
        from mimarsinan.transformations.activation_scale_policy import (
            make_activation_scale_policy,
        )

        with pytest.raises(TypeError):
            make_activation_scale_policy("min_distortion")

    def test_default_registry_policy_unchanged(self):
        from mimarsinan.transformations.activation_scale_policy import (
            DEFAULT_ACTIVATION_SCALE_POLICY,
        )

        assert DEFAULT_ACTIVATION_SCALE_POLICY == "count_quantile"
