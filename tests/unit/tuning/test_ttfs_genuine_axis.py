"""TTFSGenuineAxis: blend ramp + annealed surrogate-alpha schedule.

The genuine cascade ramp walks the same ANN->TTFS blend rate as ``TTFSAxis`` AND
anneals the spike surrogate sharpness smooth->sharp on a geometric schedule
``alpha_min * (alpha_max/alpha_min)**r`` so the deployment dynamics are exact at
rate 1 (alpha_max) while intermediate reps stay well-conditioned.
"""

from __future__ import annotations

import math

import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.tuning.axes.blend_axis import BlendAxis, TTFSGenuineAxis


class _Blend(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = 0.0


class _Perceptron(nn.Module):
    def __init__(self, S=8):
        super().__init__()
        self.base_activation = _Blend()
        self.act = TTFSActivation(T=S, activation_scale=1.0)


class _Model(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.perceptrons = nn.ModuleList(_Perceptron() for _ in range(n))

    def get_perceptrons(self):
        return list(self.perceptrons)


def test_genuine_axis_is_a_blend_axis():
    assert isinstance(TTFSGenuineAxis(), BlendAxis)
    assert TTFSGenuineAxis().name == "ttfs_genuine"


def test_alpha_for_rate_endpoints_and_monotonicity():
    axis = TTFSGenuineAxis()
    axis.attach(_Model(), None, {})  # defaults: alpha_min=0.5, alpha_max=2.0
    assert math.isclose(axis._alpha_for_rate(0.0), 0.5)
    assert math.isclose(axis._alpha_for_rate(1.0), 2.0)
    rates = [i / 10.0 for i in range(11)]
    alphas = [axis._alpha_for_rate(r) for r in rates]
    assert all(b > a for a, b in zip(alphas, alphas[1:]))


def test_alpha_for_rate_reads_config_overrides():
    axis = TTFSGenuineAxis()
    axis.attach(_Model(), None, {"ttfs_ramp_alpha_min": 1.0, "ttfs_ramp_alpha_max": 16.0})
    assert math.isclose(axis._alpha_for_rate(0.0), 1.0)
    assert math.isclose(axis._alpha_for_rate(1.0), 16.0)
    # geometric midpoint
    assert math.isclose(axis._alpha_for_rate(0.5), math.sqrt(1.0 * 16.0))


def test_set_rate_pushes_blend_rate_and_scheduled_alpha():
    model = _Model(n=3)
    axis = TTFSGenuineAxis()
    axis.attach(model, None, {})

    axis.set_rate(0.0)
    assert all(p.base_activation.rate == 0.0 for p in model.get_perceptrons())
    nodes = [m for m in model.modules() if isinstance(m, TTFSActivation)]
    assert all(node.surrogate_alpha == 0.5 for node in nodes)

    axis.set_rate(1.0)
    assert all(p.base_activation.rate == 1.0 for p in model.get_perceptrons())
    assert all(node.surrogate_alpha == 2.0 for node in nodes)

    axis.set_rate(0.5)
    assert all(p.base_activation.rate == 0.5 for p in model.get_perceptrons())
    assert all(math.isclose(node.surrogate_alpha, math.sqrt(0.5 * 2.0)) for node in nodes)
