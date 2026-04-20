"""Optional per-perceptron rate schedule for smooth adaptation.

The orchestration loop (``SmoothAdaptationTuner``) drives a single scalar
``rate`` from 0.0 to 1.0. Historically tuners applied that rate uniformly
to every perceptron, which is simple but penalizes layers with very
different sensitivities — highly-sensitive layers get pushed too hard
while tolerant layers are sandbagged.

This module adds an **opt-in** per-perceptron mapping: the scalar stays
the same, but each perceptron can lag or lead relative to the scalar.
Two invariants are enforced:

1. **Start invariant**: at scalar rate ``0.0``, every perceptron's
   effective rate is ``0.0`` (no transformation anywhere).
2. **Endpoint invariant**: at scalar rate ``1.0``, every perceptron's
   effective rate is ``1.0`` (full transformation everywhere — guaranteed
   gradient flow toward the fully-adapted fixed point).

Default behaviour (``config["per_layer_rate_schedule"]`` missing or
``False``) is uniform rate application — this preserves the existing
behaviour exactly.

This is intentionally a simple LINEAR warping; it is NOT a full
differentiable annealing scheme. The plan (see
``TUNING_FUTURE_WORK.md``) identifies STE/LSQ/DSQ as the principled
follow-up; this is the minimal "use the same scalar, but make it per-
layer aware" fix that keeps the existing tuner loop intact.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def uniform_rate_fn(rate: float) -> Callable[[object], float]:
    """Trivial schedule: every perceptron gets the same scalar rate."""
    r = _clamp01(rate)

    def _uniform(_perceptron):
        return r

    return _uniform


class LinearPerLayerSchedule:
    """Piecewise-linear per-perceptron rate schedule.

    Given a scalar ``rate`` in [0, 1] and per-perceptron sensitivities
    ``s_i`` in [0, 1] (higher = more sensitive), returns a rate function
    ``r_i(rate)`` such that:

    - ``r_i(0.0) == 0.0``  for all ``i``                 (start invariant)
    - ``r_i(1.0) == 1.0``  for all ``i``                 (endpoint invariant)
    - When ``0 < rate < 1``, a less-sensitive layer (``s_i = 0``) gets a
      higher effective rate than a highly-sensitive layer (``s_i = 1``).

    The specific warping is a linear blend controlled by a spread
    parameter ``α``:

        r_i(rate) = clamp01(rate + α * (1 - rate) * (0.5 - s_i))

    At ``rate = 0`` the term ``α * (1 - rate) * (0.5 - s_i)`` could push
    ``r_i`` below 0 when ``s_i > 0.5``; the ``clamp01`` handles that, and
    the ``(1 - rate)`` prefactor guarantees the endpoint invariant (at
    ``rate = 1`` the correction term is exactly 0).

    However, for the start invariant, we also need ``r_i(0) >= 0`` for
    all layers. The correction term ``α * 1 * (0.5 - s_i)`` is positive
    for low-sensitivity layers and negative for high-sensitivity ones;
    without the floor, low-sensitivity layers would start **above** 0.
    This is actually desirable as "head start", but it violates the
    strict start invariant. To be safe, we apply an additional
    "rate-gated" factor so at rate 0 every layer is exactly 0:

        r_i(rate) = clamp01(rate + α * rate * (1 - rate) * (0.5 - s_i) * 4)

    The extra ``* 4`` scales the peak correction (at rate=0.5) so the
    amplitude of the lead/lag remains meaningful.
    """

    def __init__(
        self,
        perceptrons: Iterable[object],
        sensitivities: Dict[str, float],
        spread: float = 0.5,
    ):
        self.perceptrons = list(perceptrons)
        self._sens: Dict[int, float] = {}
        for p in self.perceptrons:
            key = getattr(p, "name", None) or id(p)
            s = sensitivities.get(key, 0.5) if isinstance(key, str) else 0.5
            self._sens[id(p)] = _clamp01(s)
        self.spread = float(spread)

    def rate_fn(self, scalar_rate: float) -> Callable[[object], float]:
        r = _clamp01(scalar_rate)
        spread = self.spread

        def _per_layer(perceptron):
            s = self._sens.get(id(perceptron), 0.5)
            correction = spread * r * (1.0 - r) * (0.5 - s) * 4.0
            return _clamp01(r + correction)

        return _per_layer


def build_per_layer_schedule(
    config: Dict,
    perceptrons: Iterable[object],
    sensitivities: Optional[Dict[str, float]],
) -> Callable[[float], Callable[[object], float]]:
    """Build a rate-fn factory from pipeline config.

    Returns a callable that takes a scalar rate in [0, 1] and returns a
    per-perceptron rate function ``(perceptron) -> float``.

    Default (``per_layer_rate_schedule`` missing/False): returns a
    factory that ignores the perceptron argument and returns the scalar
    uniformly — exactly equivalent to the legacy behaviour.

    Opt-in (``per_layer_rate_schedule = True`` and ``sensitivities``
    provided): returns a :class:`LinearPerLayerSchedule`. Opt-in without
    sensitivities falls back to uniform so a misconfigured run does not
    silently behave strangely.
    """
    opt_in = bool(config.get("per_layer_rate_schedule", False))
    if not opt_in or sensitivities is None:
        def _factory(rate):
            return uniform_rate_fn(rate)
        return _factory

    schedule = LinearPerLayerSchedule(list(perceptrons), sensitivities)

    def _factory(rate):
        return schedule.rate_fn(rate)

    return _factory
