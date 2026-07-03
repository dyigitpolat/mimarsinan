"""[MBH X2b] LIF T-annealing realizable family — opt-in T1/P1' experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mimarsinan.models.nn.activations import ChipInputQuantizer, LIFActivation
from mimarsinan.tuning.axes.blend_axis import BlendAxis
from mimarsinan.tuning.orchestration.ramp_strategy import ValueDomainProxyRamp
from mimarsinan.tuning.perceptron_rate import set_blend_rate

DEFAULT_START_T = 32
HIGH_TARGET_START_FACTOR = 4


def tanneal_start_T(target_T: int) -> int:
    """First-rung T: 32 for small targets, 4x for targets already >= 32."""
    target = int(target_T)
    start = (
        HIGH_TARGET_START_FACTOR * target
        if target >= DEFAULT_START_T
        else DEFAULT_START_T
    )
    return max(start, target)


@dataclass(frozen=True)
class TAnnealSchedule:
    """The rate -> simulation-step-count map of the realizable LIF T-anneal.

    Rungs descend a geometric ladder from ``tanneal_start_T`` to ``target_T``;
    intermediate rungs snap to powers of two (IFNode <-> floor-staircase tie
    parity is bit-exact there — ``test_lif_floor_staircase_parity``) and the
    full rate lands on ``target_T`` EXACTLY (P1' endpoint exactness).
    """

    target_T: int
    ladder_rates: tuple[float, ...]

    def __post_init__(self):
        if int(self.target_T) < 1:
            raise ValueError(f"target_T must be >= 1; got {self.target_T}")
        rates = self.ladder_rates
        if not rates:
            raise ValueError("ladder_rates must be non-empty")
        if any(b <= a for a, b in zip(rates, rates[1:])):
            raise ValueError(f"ladder_rates must be strictly increasing; got {rates}")
        if abs(rates[-1] - 1.0) > 1e-9:
            raise ValueError(f"ladder_rates must end at 1.0; got {rates}")

    @property
    def start_T(self) -> int:
        return tanneal_start_T(self.target_T)

    @property
    def rung_Ts(self) -> tuple[int, ...]:
        return tuple(self.T_for_rate(rate) for rate in self.ladder_rates)

    def T_for_rate(self, rate: float) -> int:
        """The rung's T for a ladder (or gate-midpoint) rate, monotone in rate."""
        r = float(rate)
        target = int(self.target_T)
        if r >= 1.0 - 1e-9:
            return target
        n = len(self.ladder_rates)
        if n <= 1 or self.start_T == target:
            return target
        frac = self._ladder_position(r) / (n - 1)
        log2_T = (1.0 - frac) * math.log2(self.start_T) + frac * math.log2(target)
        T = 2 ** round(log2_T)
        return int(min(max(T, target), self.start_T))

    def _ladder_position(self, r: float) -> float:
        """Fractional rung index of ``r`` by piecewise-linear interpolation over
        the ladder rates (gate midpoints land between rungs; below-ladder clamps
        to the first rung)."""
        rates = self.ladder_rates
        if r <= rates[0]:
            return 0.0
        for i in range(len(rates) - 1):
            lo, hi = rates[i], rates[i + 1]
            if r <= hi:
                return i + (r - lo) / (hi - lo)
        return float(len(rates) - 1)


def apply_simulation_steps(model, T: int) -> None:
    """Set the rung's T on every LIF node and chip input quantizer under ``model``."""
    for module in model.modules():
        if isinstance(module, (LIFActivation, ChipInputQuantizer)):
            module.T = int(T)


def _current_simulation_steps(model) -> int | None:
    for module in model.modules():
        if isinstance(module, (LIFActivation, ChipInputQuantizer)):
            return int(module.T)
    return None


class LIFTAnnealAxis(BlendAxis):
    """Realizable-family axis: blend pinned fully at 1.0; the rate anneals T.

    Every rate is a genuine deployable LIF network at ``schedule.T_for_rate(rate)``
    — never an old/target output mixture.
    """

    name = "lif_tanneal"

    def __init__(self, schedule: TAnnealSchedule):
        super().__init__()
        self._schedule = schedule

    def set_rate(self, alpha: float) -> None:
        set_blend_rate(self._model, 1.0)
        apply_simulation_steps(self._model, self._schedule.T_for_rate(alpha))

    def get_extra_state(self):
        return {
            "blend_rates": [
                p.base_activation.rate for p in self._model.get_perceptrons()
            ],
            "simulation_steps": _current_simulation_steps(self._model),
        }

    def set_extra_state(self, extra) -> None:
        for perceptron, rate in zip(
            self._model.get_perceptrons(), extra["blend_rates"],
        ):
            perceptron.base_activation.rate = float(rate)
        steps = extra["simulation_steps"]
        if steps is not None:
            apply_simulation_steps(self._model, int(steps))

    def descriptor(self) -> str:
        return f"{self.name}(target_T={self._schedule.target_T})"


class TAnnealRealizableRamp(ValueDomainProxyRamp):
    """Ramp strategy whose only deviation from the value-domain recipe is the
    axis: same blend module, KD loss, and forwards (equal-budget T1 comparison)."""

    def __init__(self, schedule: TAnnealSchedule):
        self._schedule = schedule

    def make_axis(self, tuner) -> LIFTAnnealAxis:
        return LIFTAnnealAxis(self._schedule)
