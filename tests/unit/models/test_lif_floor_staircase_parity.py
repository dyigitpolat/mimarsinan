"""spikingjelly-IFNode vs floor-staircase BIT parity (MBH T-annealing, theory [A] §4a).

The MBH T-annealing family (``tuning/orchestration/mbh_tanneal.py``) claims every
rung is a genuine deployable staircase at that rung's T. This suite pins the exact
convention, constant drive ``c = x/theta`` swept over integer and non-integer
``T*c`` including the tie points ``c = k/T``:

- inclusive ``<=`` (subtractive reset, ties fire): the rate forward is BITWISE
  ``floor(T*clamp(c,0,1))/T * theta`` for every power-of-two T;
- strict ``<`` (the chip default, ties fire one cycle later): BITWISE
  ``(ceil(T*c)-1)/T * theta`` for c in the open unit interval, saturating to a
  full-rate ``theta`` only for ``c`` strictly above 1 (at ``c == 1`` exactly the
  tie loses one fire: ``(T-1)/T * theta``);
- non-power-of-two T: float accumulation (IFNode adds ``c`` per step) vs the
  single-multiply ``floor(c*T)`` drift apart by AT MOST one spike (1/T), and only
  in the immediate neighbourhood of exact grid ties. The T-anneal accounts for
  this by snapping intermediate rungs to powers of two, where parity is exact.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.wire_semantics import floor_staircase

POW2_TS = [2, 4, 8, 16, 32, 64, 128]
TIE_EPS = 2.0**-16


def _drive_sweep(T: int) -> torch.Tensor:
    """Constant-drive values c: every exact tie k/T (k = 0..T+2), near-tie
    offsets, seeded random interiors, and the negative/saturated tails."""
    ks = torch.arange(0, T + 3, dtype=torch.float32)
    ties = ks / T
    generator = torch.Generator().manual_seed(0)
    interiors = torch.rand(256, generator=generator)
    tails = torch.tensor([-1.0, -0.25, 1.0, 1.5, 2.7])
    return torch.cat([ties, ties + TIE_EPS, ties - TIE_EPS, interiors, tails])


def _theta32(theta: float) -> torch.Tensor:
    """The node's own scale arithmetic: fp32 tensor, clamped like ``safe_scale``."""
    return torch.tensor(float(theta), dtype=torch.float32).clamp(min=1e-12)


def _lif_rate_forward(c: torch.Tensor, theta: float, T: int, mode: str) -> torch.Tensor:
    act = LIFActivation(
        T=T, activation_scale=torch.tensor(float(theta)), thresholding_mode=mode,
    )
    return act((c * _theta32(theta)).unsqueeze(0)).squeeze(0)


def _inclusive_reference(c: torch.Tensor, theta: float, T: int) -> torch.Tensor:
    # Mirror the node's normalization bit-for-bit: x = c*theta, drive = x/theta.
    safe = _theta32(theta)
    drive = (c * safe) / safe
    return floor_staircase(drive.clamp(0.0, 1.0), T) * safe


def _strict_reference(c: torch.Tensor, theta: float, T: int) -> torch.Tensor:
    # No upper pre-clamp: drive strictly above threshold fires EVERY cycle, so
    # saturation to full theta is reached for c > 1 while c == 1 loses one fire.
    safe = _theta32(theta)
    drive = (c * safe) / safe
    raw = T * drive.clamp(min=0.0)
    k = torch.where(raw > 0, torch.ceil(raw) - 1.0, torch.zeros_like(raw))
    return k.clamp(max=T) / T * safe


class TestPow2BitParity:
    @pytest.mark.parametrize("theta", [1.0, 3.7])
    @pytest.mark.parametrize("T", POW2_TS)
    def test_inclusive_rate_forward_is_bitwise_floor_staircase(self, T, theta):
        c = _drive_sweep(T)
        y = _lif_rate_forward(c, theta, T, "<=")
        ref = _inclusive_reference(c, theta, T)
        assert torch.equal(y, ref), (
            f"IFNode '<=' diverged from floor(T*c)/T at T={T}, theta={theta}: "
            f"{int((y != ref).sum())} points, max |diff| {float((y - ref).abs().max())}"
        )

    @pytest.mark.parametrize("theta", [1.0, 3.7])
    @pytest.mark.parametrize("T", POW2_TS)
    def test_strict_rate_forward_is_bitwise_strict_staircase(self, T, theta):
        c = _drive_sweep(T)
        y = _lif_rate_forward(c, theta, T, "<")
        ref = _strict_reference(c, theta, T)
        assert torch.equal(y, ref), (
            f"IFNode '<' diverged from (ceil(T*c)-1)/T at T={T}, theta={theta}: "
            f"{int((y != ref).sum())} points, max |diff| {float((y - ref).abs().max())}"
        )


class TestTieConvention:
    """The exact tie behaviour the anneal relies on, spelled out point-wise."""

    def test_inclusive_ties_fire(self):
        T = 8
        ties = torch.arange(0, T + 1, dtype=torch.float32) / T
        y = _lif_rate_forward(ties, 1.0, T, "<=")
        torch.testing.assert_close(y, ties, atol=0.0, rtol=0.0)

    def test_strict_ties_fire_one_cycle_later(self):
        T = 8
        ks = torch.arange(1, T + 1, dtype=torch.float32)
        y = _lif_rate_forward(ks / T, 1.0, T, "<")
        torch.testing.assert_close(y, (ks - 1) / T, atol=0.0, rtol=0.0)

    def test_strict_saturation_is_discontinuous_at_theta(self):
        T, theta = 8, 1.0
        at_theta = _lif_rate_forward(torch.tensor([1.0]), theta, T, "<")
        above_theta = _lif_rate_forward(torch.tensor([1.0 + 2.0**-16]), theta, T, "<")
        assert float(at_theta) == (T - 1) / T * theta
        assert float(above_theta) == theta


class TestNonPow2TieDrift:
    """Documented convention: at non-power-of-two T the IFNode accumulation and
    the single-multiply floor can disagree at exact grid ties (c = k/T is not
    float-representable), by exactly one spike and only near ties. This is WHY
    the T-anneal snaps intermediate rungs to powers of two."""

    @pytest.mark.parametrize("mode", ["<=", "<"])
    @pytest.mark.parametrize("T", [10, 51])
    def test_drift_is_at_most_one_spike_and_only_near_ties(self, T, mode):
        c = _drive_sweep(T)
        y = _lif_rate_forward(c, 1.0, T, mode)
        ref = (
            _inclusive_reference(c, 1.0, T) if mode == "<="
            else _strict_reference(c, 1.0, T)
        )
        diff = (y - ref).abs()
        assert float(diff.max()) <= 1.0 / T + 1e-7, "tie drift exceeded one spike"
        raw = T * c.clamp(0.0, 1.0)
        near_tie = (raw - raw.round()).abs() <= 1e-3
        off_tie = ~near_tie & (c > 0) & (c < 1)
        assert torch.equal(y[off_tie], ref[off_tie]), (
            "divergence outside the tie neighbourhood: the drift is not a pure "
            "tie-convention effect"
        )

    def test_non_pow2_inclusive_ties_do_drift(self):
        # The positive control: the drift exists (T=10 loses the c=0.7 tie fire),
        # so the pow2 snap in the anneal is load-bearing, not decorative.
        y = _lif_rate_forward(torch.tensor([0.7]), 1.0, 10, "<=")
        ref = _inclusive_reference(torch.tensor([0.7]), 1.0, 10)
        assert float(ref) == pytest.approx(0.7)
        assert float(y) == pytest.approx(0.6)


class TestThetaDecodeScaleConsistency:
    """y = mean_t(spikes) * theta stays in [0, theta] at every T, and values on
    the coarse target grid are fixed points of every finer pow2 rung — the anneal
    refines resolution only, never the scale."""

    @pytest.mark.parametrize("mode", ["<=", "<"])
    @pytest.mark.parametrize("theta", [0.5, 1.0, 3.7])
    def test_output_range_is_scale_bounded_across_T(self, theta, mode):
        theta_fp32 = float(_theta32(theta))
        x = torch.linspace(-theta, 2.0 * theta, 197).unsqueeze(0)
        for T in (4, 8, 16, 32):
            act = LIFActivation(
                T=T, activation_scale=torch.tensor(theta), thresholding_mode=mode,
            )
            y = act(x)
            assert float(y.min()) >= 0.0
            assert float(y.max()) <= theta_fp32

    def test_target_grid_values_are_fixed_points_of_finer_rungs(self):
        target_T, theta = 4, 1.0
        grid = torch.arange(0, target_T + 1, dtype=torch.float32) / target_T
        for T in (32, 16, 8, 4):
            y = _lif_rate_forward(grid, theta, T, "<=")
            torch.testing.assert_close(y, grid * theta, atol=0.0, rtol=0.0)
