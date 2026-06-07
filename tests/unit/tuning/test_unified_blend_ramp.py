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


def test_lif_cycle_accurate_ramp_rate0_leaks_off_continuous():
    """LEAK REGRESSION: the legacy ``_CycleAccurateForward`` ramp applies the
    blend per spike-frame, so at r=0 the output is ``mean_t ReLU(W s_t + b)`` —
    a biased rate-coded value, NOT the continuous teacher. The unified ramp
    (value domain) does reproduce the teacher; this test documents the gap the
    refactor closes and guards against silently routing the ramp through the
    cascade forward again."""
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import _CycleAccurateForward

    model = _TwoLayerBlend("LIF", T=8, scale=2.0).eval()
    # Non-negative inputs so the continuous ReLU path is active (worst case for
    # the convex-Jensen rate-coding bias).
    x = _inputs().abs()
    model.set_rate(0.0)

    with torch.no_grad():
        continuous = model(x)                      # value-domain r=0 == teacher
        cycle_accurate = _CycleAccurateForward(model, T=8)(x)

    # Value-domain r=0 is the teacher (sanity, bit-exact).
    torch.testing.assert_close(continuous, model.reference(x, model.old1, model.old2),
                               rtol=0, atol=0)
    # Cycle-accurate r=0 is demonstrably NOT the teacher (the leak).
    assert not torch.allclose(cycle_accurate, continuous, atol=1e-3), (
        "legacy cycle-accurate ramp unexpectedly reproduced the continuous "
        "teacher at r=0 — the per-frame-blend leak this test guards is gone"
    )


class _LifModel(nn.Module):
    def get_perceptrons(self):
        return []


def _lif_ramp_stub(*, cycle_accurate: bool, legacy: bool):
    return types.SimpleNamespace(
        model=_LifModel(), _T=8,
        _cycle_accurate=cycle_accurate, _legacy_blend_ramp=legacy,
    )


class TestLifRampSelection:
    """The LIF tuner picks its ramp/finalize forwards by flag. Legacy per-frame
    ramp is opt-in (default-on until verified); the unified value-domain ramp
    installs no ramp forward. Finalize always installs the chip-aligned forward
    when cycle-accurate, regardless of which ramp ran."""

    def test_legacy_cycle_accurate_installs_ramp_forward(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            LIFAdaptationTuner, _CycleAccurateForward,
        )
        stub = _lif_ramp_stub(cycle_accurate=True, legacy=True)
        assert isinstance(LIFAdaptationTuner._ramp_forward(stub), _CycleAccurateForward)

    def test_unified_cycle_accurate_installs_no_ramp_forward(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
        stub = _lif_ramp_stub(cycle_accurate=True, legacy=False)
        assert LIFAdaptationTuner._ramp_forward(stub) is None

    def test_non_cycle_accurate_installs_no_ramp_forward(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
        for legacy in (True, False):
            stub = _lif_ramp_stub(cycle_accurate=False, legacy=legacy)
            assert LIFAdaptationTuner._ramp_forward(stub) is None

    @pytest.mark.parametrize("legacy", [True, False])
    def test_finalize_forward_is_chip_aligned_when_cycle_accurate(self, legacy):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
            LIFAdaptationTuner, _ChipAlignedNFForward,
        )
        stub = _lif_ramp_stub(cycle_accurate=True, legacy=legacy)
        assert isinstance(
            LIFAdaptationTuner._finalize_forward(stub), _ChipAlignedNFForward
        )

    def test_finalize_forward_none_when_not_cycle_accurate(self):
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
        stub = _lif_ramp_stub(cycle_accurate=False, legacy=True)
        assert LIFAdaptationTuner._finalize_forward(stub) is None
