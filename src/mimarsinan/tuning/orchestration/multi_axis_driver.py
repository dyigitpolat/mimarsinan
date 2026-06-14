"""Interleaved multi-axis continuation over value-domain axes (research-grade).

Opt-in via ``config["interleave_axes"]`` (default off). The single-axis
``SmoothAdaptationTuner`` ramp advances ONE transformation rate 0->1; this
driver advances a *vector* of rates over several value-domain axes at once,
stepping the least-sensitive axis first, reusing the per-axis bisection idea
(propose committed+step; accept if the attempt holds, else shrink and retreat
to the committed edge) until every axis reaches its feasible edge.

HARD CONTRACT: interleaving is **value-domain only**. The blend / LIF / TTFS
families swap in genuine deployed dynamics at a ``_finalize`` / forward-install
seam (see ``KDBlendAdaptationTuner`` and subclasses); interleaving their rates
would reorder that finalize against other axes and break NF<->SCM parity. The
driver therefore rejects any non-value-domain axis up front
(``assert_value_domain_axes``) and never touches a finalize / forward-install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

_RATE_EPS = 1e-9
_FULL_RATE = 1.0


def _is_value_domain(axis) -> bool:
    """A plain callable is value-domain by construction; a descriptor must say so.

    Anything that declares a non-value ``domain`` (blend/lif/ttfs) or sets
    ``is_value_domain`` falsy is rejected — those advance a deployment forward
    at a finalize seam that interleaving must never reorder.
    """
    domain = getattr(axis, "domain", None)
    if domain is not None and str(domain) != "value":
        return False
    flag = getattr(axis, "is_value_domain", None)
    if flag is not None:
        return bool(flag)
    return callable(axis) or callable(getattr(axis, "attempt", None))


def assert_value_domain_axes(axes: Sequence) -> None:
    """Reject any axis that is not value-domain (blend/LIF/TTFS finalize-bearing).

    Interleaving must NEVER touch a deployment forward-install / finalize seam.
    """
    offenders = [
        getattr(a, "name", repr(a)) for a in axes if not _is_value_domain(a)
    ]
    if offenders:
        raise ValueError(
            "interleave_axes is value-domain only; refusing non-value-domain "
            f"axes {offenders}. Blend/LIF/TTFS axes carry a finalize / "
            "forward-install seam that interleaving must never reorder."
        )


def _axis_attempt(axis) -> Callable[[float], float]:
    """Resolve an axis to its ``attempt(target) -> committed_rate`` callable."""
    attempt = getattr(axis, "attempt", None)
    if callable(attempt):
        return attempt
    if callable(axis):
        return axis
    raise TypeError(f"axis {axis!r} is neither callable nor has a callable .attempt")


@dataclass
class ValueAxisSpec:
    """A value-domain axis: a name + an ``attempt(target) -> committed`` callable.

    ``domain == "value"`` is the guard token interleaving requires; this is the
    self-contained descriptor the driver consumes when a caller does not pass a
    full ``AdaptationAxis``.
    """

    name: str
    attempt: Callable[[float], float]
    sensitivity_hint: Optional[float] = None
    domain: str = "value"
    is_value_domain: bool = True


@dataclass
class _AxisState:
    """Per-axis bisection bookkeeping for one continuation run."""

    name: str
    attempt: Callable[[float], float]
    committed: float = 0.0
    step: float = 0.5
    stalled: bool = False
    trace: List[float] = field(default_factory=list)

    def headroom(self) -> float:
        return max(0.0, _FULL_RATE - self.committed)


AxisLike = Union[ValueAxisSpec, Callable[[float], float], object]


class VectorRateScheduler:
    """Advance a committed *vector* over multiple value-domain axes to 1.0.

    Per-axis bisection (the single-axis ``RateScheduler`` idea): propose
    ``committed + step``; accept (advance, grow the step) when the attempt holds
    within ``eps``, else retreat to the returned committed rate and halve the
    step. Each round the **least-sensitive** axis (most remaining headroom with
    the largest live step) advances first; an axis whose step shrinks below
    ``min_step`` is *saturated* at its feasible edge and skipped thereafter.
    """

    def __init__(
        self,
        attempts: Sequence[Callable[[float], float]],
        *,
        names: Optional[Sequence[str]] = None,
        committed: Optional[Sequence[float]] = None,
        initial_step: float = 0.5,
        growth: float = 1.5,
        min_step: float = 1e-3,
        eps: float = _RATE_EPS,
    ):
        attempts = list(attempts)
        if not attempts:
            raise ValueError("VectorRateScheduler requires at least one axis attempt")
        if names is None:
            names = [f"axis{i}" for i in range(len(attempts))]
        if committed is None:
            committed = [0.0] * len(attempts)
        if not (len(names) == len(committed) == len(attempts)):
            raise ValueError("names, committed, attempts must be equal length")

        self._initial_step = float(initial_step)
        self._growth = float(growth)
        self._min_step = float(min_step)
        self._eps = float(eps)
        self._states = [
            _AxisState(
                name=str(n),
                attempt=a,
                committed=float(c),
                step=self._initial_step,
            )
            for n, a, c in zip(names, attempts, committed)
        ]

    @property
    def committed_vector(self) -> List[float]:
        return [s.committed for s in self._states]

    @property
    def traces(self) -> List[List[float]]:
        return [list(s.trace) for s in self._states]

    def _active(self) -> List[_AxisState]:
        return [
            s
            for s in self._states
            if not s.stalled and s.headroom() > self._eps
        ]

    def _least_sensitive(self, active: Sequence[_AxisState]) -> _AxisState:
        """Pick the axis that should advance first: the one *least* resisting.

        Resistance is read off the live bisection state — an axis with more
        headroom and a larger surviving step has shown the least rollback so
        far, so it is the least-sensitive and goes first.
        """
        return max(active, key=lambda s: (s.headroom(), s.step))

    def _advance_once(self, state: _AxisState) -> None:
        target = min(state.committed + state.step, _FULL_RATE)
        result = state.attempt(target)
        result = state.committed if result is None else float(result)

        if result < target - self._eps:
            state.committed = max(state.committed, result)
            state.step *= 0.5
            if state.step < self._min_step:
                state.stalled = True
        else:
            state.committed = target
            remaining = state.headroom()
            if remaining <= self._eps:
                state.stalled = True
            else:
                state.step = min(state.step * self._growth, remaining)
        state.trace.append(state.committed)

    def run(self, max_rounds: Optional[int] = None) -> List[float]:
        """Interleave axis advances until every axis reaches its feasible edge."""
        rounds = 0
        cap = max_rounds if max_rounds is not None else self._default_round_cap()
        while True:
            active = self._active()
            if not active:
                break
            if rounds >= cap:
                break
            self._advance_once(self._least_sensitive(active))
            rounds += 1
        return self.committed_vector

    def _default_round_cap(self) -> int:
        per_axis = max(4, int(2.0 / max(self._min_step, 1e-6)))
        return per_axis * len(self._states)


class MultiAxisDriver:
    """Drive a ``VectorRateScheduler`` over value-domain axes (opt-in).

    Guards that every axis is value-domain (never blend/LIF/TTFS) before running,
    so interleaving can never touch a deployment finalize / forward-install seam.
    """

    def __init__(
        self,
        axes: Sequence[AxisLike],
        *,
        initial_step: float = 0.5,
        growth: float = 1.5,
        min_step: float = 1e-3,
        committed: Optional[Sequence[float]] = None,
    ):
        axes = list(axes)
        if not axes:
            raise ValueError("MultiAxisDriver requires at least one axis")
        assert_value_domain_axes(axes)
        self._axes = axes
        self._scheduler = VectorRateScheduler(
            attempts=[_axis_attempt(a) for a in axes],
            names=[getattr(a, "name", f"axis{i}") for i, a in enumerate(axes)],
            committed=committed,
            initial_step=initial_step,
            growth=growth,
            min_step=min_step,
        )

    @property
    def scheduler(self) -> VectorRateScheduler:
        return self._scheduler

    @property
    def committed_vector(self) -> List[float]:
        return self._scheduler.committed_vector

    @property
    def traces(self) -> List[List[float]]:
        return self._scheduler.traces

    def run(self, max_rounds: Optional[int] = None) -> List[float]:
        return self._scheduler.run(max_rounds=max_rounds)
