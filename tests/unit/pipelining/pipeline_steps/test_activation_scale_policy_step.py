"""[§17.13] the analysis step's scale policy: quantile (default, byte-identical)
or the deployed-distortion argmin, with levels from the value_grid_levels SSOT."""

from __future__ import annotations

import pytest
import torch

from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
    resolve_scale_for_samples,
)
from mimarsinan.transformations.activation_scale_policy import (
    DEFAULT_SCALE_QUANTILE,
    optimal_theta,
)


def _heavy_tailed():
    torch.manual_seed(0)
    return torch.cat([torch.rand(20000) * 0.3, torch.rand(200) * 6.0 + 2.0])


def test_default_is_the_quantile_and_byte_identical():
    from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
        scale_from_activations,
    )

    acts = _heavy_tailed()
    assert resolve_scale_for_samples(
        acts, policy="count_quantile", quantile=DEFAULT_SCALE_QUANTILE, levels=32,
    ) == pytest.approx(scale_from_activations(acts, quantile=DEFAULT_SCALE_QUANTILE))


def test_min_distortion_uses_the_argmin_at_the_deployed_grid():
    acts = _heavy_tailed()
    assert resolve_scale_for_samples(
        acts, policy="min_distortion", quantile=DEFAULT_SCALE_QUANTILE, levels=32,
    ) == pytest.approx(optimal_theta(acts, 32))


def test_min_distortion_needs_a_grid_and_says_so():
    """A continuous deployment (analytical ttfs => levels None) has no
    resolution term, so the argmin is undefined: fail loud, never silently
    fall back to the heuristic this policy replaces."""
    with pytest.raises(ValueError, match="min_distortion"):
        resolve_scale_for_samples(
            _heavy_tailed(), policy="min_distortion",
            quantile=DEFAULT_SCALE_QUANTILE, levels=None,
        )


def test_quantile_policy_ignores_missing_levels():
    acts = _heavy_tailed()
    assert resolve_scale_for_samples(
        acts, policy="count_quantile", quantile=0.99, levels=None,
    ) > 0.0


def test_knob_registered_default_preserves_behavior():
    from mimarsinan.config_schema.registry import effective_value

    assert effective_value({}, "activation_scale_policy") == "count_quantile"
    assert effective_value(
        {"activation_scale_policy": "min_distortion"}, "activation_scale_policy",
    ) == "min_distortion"
