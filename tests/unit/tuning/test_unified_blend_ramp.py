"""Unified value-domain blend-ramp contract (shared by LIF + TTFS tuners).

The golden ramp is the per-perceptron ``BlendActivation`` evaluated in the
*value domain* (the plain class forward, no per-frame / cascade instance
forward):

    BlendActivation(v) = (1 - r)·ReLU(v) + r·OnChipAct(v)

- ``r = 0`` reproduces the continuous teacher **bit-for-bit** (non-destructive).
- ``r = 1`` is the pointwise on-chip composition (LIF rate / TTFS staircase).

These tests pin those endpoints, and document the latent LIF leak: when the
cycle-accurate cross-layer forward is installed *during the ramp*, the blend is
applied per spike-frame, so ``r = 0`` is a biased rate-coded approximation —
NOT the continuous teacher. The unified ramp keeps the cascade forward out of
the ramp (deferred to finalize), which is what these tests protect.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LeakyGradReLU, LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import BlendActivation


class _TwoLayerBlend(nn.Module):
    """Two cascaded linear+activation stages with per-stage blends.

    ``old`` is a shared continuous ReLU; ``target`` is the on-chip activation.
    The plain ``forward`` is the value-domain ramp; ``reference_*`` recompute the
    same graph with a fixed activation so endpoint equality is meaningful (two
    cascaded nonlinearities, not just the ``BlendActivation`` short-circuit).
    """

    def __init__(self, target_kind: str, T: int, scale: float, d=5, h=4, o=3):
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(d, h)
        self.l2 = nn.Linear(h, o)
        self.old1, self.old2 = LeakyGradReLU(), LeakyGradReLU()
        self.t1 = self._make_target(target_kind, T, scale)
        self.t2 = self._make_target(target_kind, T, scale)
        self.act1 = BlendActivation(self.old1, self.t1, 0.0, target_type=target_kind)
        self.act2 = BlendActivation(self.old2, self.t2, 0.0, target_type=target_kind)

    @staticmethod
    def _make_target(kind: str, T: int, scale: float) -> nn.Module:
        if kind == "LIF":
            return LIFActivation(T=T, activation_scale=torch.tensor(scale),
                                 thresholding_mode="<=")
        return TTFSActivation(T=T, activation_scale=torch.tensor(scale),
                              input_scale=torch.tensor(scale), thresholding_mode="<=")

    def set_rate(self, rate: float) -> None:
        self.act1.rate = float(rate)
        self.act2.rate = float(rate)

    def forward(self, x):
        return self.act2(self.l2(self.act1(self.l1(x))))

    def reference(self, x, act1, act2):
        return act2(self.l2(act1(self.l1(x))))


def _inputs():
    torch.manual_seed(1)
    return torch.randn(6, 5)


@pytest.mark.parametrize("kind", ["LIF", "TTFS"])
def test_value_domain_rate0_is_continuous_teacher(kind):
    """r=0 value-domain forward == the continuous (ReLU) reference, bit-exact."""
    model = _TwoLayerBlend(kind, T=8, scale=1.5).eval()
    x = _inputs()
    model.set_rate(0.0)
    with torch.no_grad():
        got = model(x)
        ref = model.reference(x, model.old1, model.old2)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["LIF", "TTFS"])
def test_value_domain_rate1_is_onchip_composition(kind):
    """r=1 value-domain forward == the pointwise on-chip composition, bit-exact."""
    model = _TwoLayerBlend(kind, T=8, scale=1.5).eval()
    x = _inputs()
    model.set_rate(1.0)
    with torch.no_grad():
        got = model(x)
        ref = model.reference(x, model.t1, model.t2)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["LIF", "TTFS"])
def test_value_domain_blend_is_monotone_interpolation(kind):
    """Intermediate rate is the exact linear interpolation of the endpoints."""
    model = _TwoLayerBlend(kind, T=8, scale=1.5).eval()
    x = _inputs()
    with torch.no_grad():
        model.set_rate(0.0)
        lo = model(x)
        model.set_rate(1.0)
        hi = model(x)
        # Per-layer linearity holds at the activation; check the first stage,
        # whose output blends exactly (downstream is nonlinear in the blend).
        pre1 = model.l1(x)
        model.set_rate(0.3)
        blended = model.act1(pre1)
        expected = 0.7 * model.old1(pre1) + 0.3 * model.t1(pre1)
    torch.testing.assert_close(blended, expected, rtol=0, atol=0)
    assert not torch.allclose(lo, hi)


class _LifModel(nn.Module):
    def get_perceptrons(self):
        return []


def _lif_ramp_stub(*, cycle_accurate: bool):
    return types.SimpleNamespace(
        model=_LifModel(), _T=8, _cycle_accurate=cycle_accurate,
        _per_hop_retiming=False,
    )


class TestLifRampSelection:
    """LIF always ramps in the value domain (no instance ``_ramp_forward``
    override — it inherits the base ``None``); finalize installs the
    chip-aligned forward when cycle-accurate, and nothing otherwise."""

    def test_lif_inherits_value_domain_ramp(self):
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
            KDBlendAdaptationTuner,
        )
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        # No override: the base class returns None (value-domain ramp).
        assert LIFAdaptationTuner._ramp_forward is KDBlendAdaptationTuner._ramp_forward

    @pytest.mark.parametrize("cycle_accurate", [True, False])
    def test_finalize_forward_matches_cycle_accurate(self, cycle_accurate):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            LIFAdaptationTuner, _ChipAlignedNFForward,
        )
        stub = _lif_ramp_stub(cycle_accurate=cycle_accurate)
        # The per-tuner builder is now ``_finalize_forward_for(model)`` (shared by
        # the finalize install and the genuine probe); ``_finalize_forward``
        # delegates to it with ``self.model``.
        fwd = LIFAdaptationTuner._finalize_forward_for(stub, stub.model)
        if cycle_accurate:
            assert isinstance(fwd, _ChipAlignedNFForward)
            assert fwd.model is stub.model
        else:
            assert fwd is None
