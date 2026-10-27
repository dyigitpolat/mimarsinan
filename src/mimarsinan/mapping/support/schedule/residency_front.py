"""[E85/E297] The residency planner: phase fit, the exact minimal rotating region r* and the live-set carry front;
[F3 round] r* bisected on the exact feasibility of a plan (``residency_front_bounds.plan_exists``), the floor's carry
probed when a step is first read, each next step searched outward from the last step's r + 1 (``StepWalk._next``).
The reference fronts are ``residency_front_reference``'s."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.support.residency_front_bounds import plan_exists
from mimarsinan.mapping.support.residency_front_reference import (
    front_mincarry_np as front_mincarry_np, front_mincarry_py as front_mincarry_py,
)
from mimarsinan.mapping.support.residency_front_store import (
    carries_below, front_mincarry_below, front_mincarry_store, operand_bound, planned_store,
)

#: Phases beyond the fit the planner may use: Q in [Q_fit, Q_fit + Q_SPAN] (law.md Q_hi).
Q_SPAN = 8
#: Unit count from which the store front runs (``residency_front_store``) [W03]. Whole step functions, one cold call
#: each (RA/11_wave2/explore/w03/walls/front_walls.jsonl): below 30 units pure Python 5 ms against 21 ms per call;
#: from 100 units the store wins (DenseNet-BC-100 at C 144: 0.64 s against 1.78 s vectorized and 6.4 s Python;
#: t0_03's 640 tile units at C 162: 14.2 s against 41.2 s vectorized). The vectorized front (E296) stays a branch.
STORE_MIN_UNITS = 100

FrontFn = Callable[..., Optional[Tuple[int, int]]]
FRONTS: Dict[str, FrontFn] = {"python": front_mincarry_py, "vectorized": front_mincarry_np,
                              "store": front_mincarry_store}


def select_front(unit_count: int) -> FrontFn:
    """The size dispatch: the pure-Python front below ``STORE_MIN_UNITS`` units, else the store front."""
    return FRONTS["store" if unit_count >= STORE_MIN_UNITS else "python"]


def q_fit(sizes: Sequence[int], C: int, forced: Sequence[int] = ()) -> Optional[int]:
    """Fewest contiguous phases within C, a phase never spanning a forced cut (greedy is optimal); None if a
    unit exceeds C."""
    if max(sizes) > C:
        return None
    cuts = set(int(p) for p in forced)
    q, acc = 1, 0
    for index, s in enumerate(sizes):
        if index in cuts or acc + s > C:
            q, acc = q + 1, s
        else:
            acc += s
    return q


@dataclass(frozen=True)
class ResidencyStepFunction:
    """``x9_0_v3.plan_steps``: the steps (r, carry, Q) from the smallest feasible r, the stateless plan. [W03] The
    steps are found on demand (``step``; ``steps`` walks the whole function), each by bisection on r."""

    stateless: Tuple[int, int]
    q_fit: int
    F: int
    fluid_r: float
    C: int
    qmax: int
    forced: Tuple[int, ...] = ()
    walk: Optional["StepWalk"] = field(default=None, compare=False, repr=False)

    @property
    def steps(self) -> Tuple[Tuple[int, int, int], ...]:
        assert self.walk is not None
        return self.walk.every()

    def step(self, index: int) -> Optional[Tuple[int, int, int]]:
        """Step ``index`` of the function (0: the floor), or None past its last step."""
        assert self.walk is not None
        return self.walk.upto(int(index))

    def at(self, r: int) -> Optional[Tuple[int, int, int]]:
        """[W03] ``(r, carry, Q)`` of the minimal plan at ``r``, a step or not (None below r*)."""
        assert self.walk is not None
        got = self.walk.probe(int(r))
        return None if got is None else (int(r), int(got[0]), int(got[1]))

    def as_ra_dict(self) -> dict:
        """The RA function's own return value, for equality against it."""
        return dict(steps=list(self.steps), stateless=self.stateless, q_fit=self.q_fit, F=self.F,
                    fluid_r=self.fluid_r)


class StepWalk:
    """The steps of mincarry(r) from r*, each the smallest r whose carry falls below the last step's, probed once
    per r (feasibility and the carry are monotone in r). ``first`` is the floor step, or [F3 round] its r alone,
    probed when a step is first read (the stateless plan reads none)."""

    def __init__(self, probe: Callable[[int], Optional[Tuple[int, int]]], C: int, first: Any, done: bool = False,
                 below: Optional[Callable[[int, int], Optional[Tuple[int, int]]]] = None,
                 decide: Optional[Callable[[int, int], bool]] = None):
        self.probe, self.C, self.done = probe, int(C), done
        self.below = below or self._below
        self.decide = decide or (lambda r, ceiling: self.below(r, ceiling) is not None)
        self.found: List[Tuple[int, int, int]] = [] if isinstance(first, int) else [first]
        self.first_r = int(first) if isinstance(first, int) else None

    def _start(self) -> None:
        if not self.found and self.first_r is not None:
            self.found.append((self.first_r, *self._at(self.first_r)))

    def _below(self, r: int, ceiling: int) -> Optional[Tuple[int, int]]:
        """``probe(r)`` where its carry is below ``ceiling``, else None (a step's decision, [F3 round])."""
        got = self.probe(r)
        return got if got is not None and got[0] < ceiling else None

    def _at(self, r: int) -> Tuple[int, int]:
        got = self.probe(r)
        assert got is not None, "probed below r*"
        return got

    def _next(self) -> None:
        """The smallest r above the last step's whose carry falls below it, searched outward from r + 1 [F3 round]:
        the next r's exact probe first (the step, where the function steps at every r), then strides doubling from
        r + 2 on the decisions, then a bisection inside the bracket; the same r as a bisection over (r, C]."""
        r, carry = self.found[-1][0], self.found[-1][1]
        if r >= self.C or self.below(self.C, carry) is None:
            self.done = True
            return
        got = self.below(r + 1, carry)
        if got is not None:
            self.found.append((r + 1, *got))
            return
        lo, hi, stride = r + 2, self.C, 1
        while lo < hi:  # the decision holds at hi and fails below lo
            point = min(lo + stride - 1, hi - 1)
            if self.decide(point, carry):
                hi = point
                break
            lo, stride = point + 1, 2 * stride
        while lo < hi:
            mid = (lo + hi) // 2
            if self.decide(mid, carry):
                hi = mid
            else:
                lo = mid + 1
        got = self.below(lo, carry)
        assert got is not None, "the step's r carries no less"
        self.found.append((lo, *got))

    def upto(self, index: int) -> Optional[Tuple[int, int, int]]:
        self._start()
        while len(self.found) <= index and not self.done:
            self._next()
        return self.found[index] if index < len(self.found) else None

    def every(self) -> Tuple[Tuple[int, int, int], ...]:
        self._start()
        while not self.done:
            self._next()
        return tuple(self.found)


def binding_cap(sizes: Sequence[int], C: int, forced: Sequence[int], qmax: int,
                cap: Optional[int]) -> Optional[int]:
    """[W03] A per-segment phase cap where it can bind: some segment (between forced cuts) could hold more phases
    than ``cap`` (its units, and at most ``qmax`` less the other segments' fewest); None where no plan exceeds it."""
    if cap is None:
        return None
    bounds = [0, *sorted(set(int(p) for p in forced)), len(sizes)]
    fewest = [q_fit(sizes[a:b], C) or 0 for a, b in zip(bounds, bounds[1:])]
    most = [min(b - a, int(qmax) - (sum(fewest) - k)) for (a, b), k in zip(zip(bounds, bounds[1:]), fewest)]
    return int(cap) if max(most) > int(cap) else None


def plan_steps(sizes, live, C, *, forced=(), front: Optional[FrontFn] = None,
               segment_cap: Optional[int] = None, operand=()) -> Optional[ResidencyStepFunction]:
    """The step function of mincarry(r), non-increasing in r; None if no plan exists (a unit exceeds C, or a segment
    needs more phases than ``segment_cap``). A cap that can bind plans on the store front with it [W03]. [F3]
    ``operand``: per unit its operand tiles, never pinned, so a phase at r holds at most r of them (a fit with
    operand tiles is planned by the fronts: its closed form, r* = 0, pins every tile)."""
    sizes, operand = tuple(int(s) for s in sizes), tuple(int(o) for o in operand)
    live = tuple(int(v) for v in live)
    forced = tuple(sorted(int(p) for p in forced))
    F = sum(sizes)
    qf = q_fit(sizes, C, forced)
    if qf is None:
        return None
    qmax = qf + Q_SPAN
    cap = binding_cap(sizes, C, forced, qmax, segment_cap)
    if cap is not None and max(q_fit(sizes[a:b], C) or 0 for a, b in _segments(len(sizes), forced)) > cap:
        return None
    if F <= C and all(0 < p < len(sizes) for p in forced) and not any(operand):
        return _fit(sizes, live, int(C), qf, qmax, forced)
    memo: Dict[int, Optional[Tuple[int, int]]] = {}
    run, held = front or select_front(len(sizes)), ({"operand": operand} if any(operand) else {})

    def f(r: int) -> Optional[Tuple[int, int]]:
        if r not in memo:
            memo[r] = (run(sizes, live, C, r, qmax, forced, **held) if cap is None else
                       front_mincarry_store(sizes, live, C, r, qmax, forced, segment_cap=cap, **held))
        return memo[r]

    def fits(r: int) -> bool:  # [F3 round] the exact feasibility of the default fronts, without a front
        if front is None and cap is None:
            return plan_exists(sizes, C, r, qmax, forced, operand_bound(operand))
        return f(r) is not None

    known_floor: Dict[int, int] = {}  # per probed r, a carry its minimal plan is known not to be below

    def below(r: int, ceiling: int) -> Optional[Tuple[int, int]]:  # [F3 round] a walk step's decision
        if r in memo or front is not None or cap is not None or len(sizes) < STORE_MIN_UNITS:
            got = f(r)
            return got if got is not None and got[0] < ceiling else None
        if known_floor.get(r, -1) >= ceiling:
            return None
        got = front_mincarry_below(sizes, live, C, r, qmax, forced, ceiling, operand=operand)
        if got is None:
            known_floor[r] = max(known_floor.get(r, 0), int(ceiling))
            return None
        memo[r] = got
        return got

    def decide(r: int, ceiling: int) -> bool:  # [F3 round] a walk step's search decision: the bound may decide it
        if r in memo or known_floor.get(r, -1) >= ceiling or front is not None or cap is not None \
                or len(sizes) < STORE_MIN_UNITS:
            return below(r, ceiling) is not None
        verdict = carries_below(sizes, live, C, r, qmax, forced, ceiling, operand=operand)
        if verdict is False:
            known_floor[r] = max(known_floor.get(r, 0), int(ceiling))
        return below(r, ceiling) is not None if verdict is None else verdict

    lo = max(0, math.ceil((F - C) / (qmax - 1)))
    stateless = f(C)
    if stateless is None:
        return None
    a, b = (lo, lo) if fits(lo) else (lo + 1, C)  # smallest feasible r (feasibility monotone in r)
    while a < b:
        mid = (a + b) // 2
        if not fits(mid):
            a = mid + 1
        else:
            b = mid
    return ResidencyStepFunction(stateless=stateless, q_fit=qf, F=F, fluid_r=(F - C) / max(1, qf - 1), C=int(C),
                                 qmax=qmax, forced=forced, walk=StepWalk(f, int(C), a, below=below, decide=decide))


def _segments(n: int, forced: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = [0, *sorted(set(int(p) for p in forced)), n]
    return list(zip(bounds, bounds[1:]))


def _fit(sizes, live, C, qf, qmax, forced) -> ResidencyStepFunction:
    """[W03] The step function of a fit (F <= C), in closed form: every r admits one phase per segment, which pays only
    the forced cuts' carry and runs the fewest phases (``q_fit``), so r* = 0 and the function is that one step."""
    carry = sum(2 * live[p - 1] for p in set(forced))
    return ResidencyStepFunction(stateless=(carry, qf), q_fit=qf, F=sum(sizes), fluid_r=(sum(sizes) - C) / max(1, qf - 1),
                                 C=C, qmax=qmax, forced=forced,
                                 walk=StepWalk(lambda r: (carry, qf), C, (0, carry, qf), done=True))


def phase_partition(sizes, live, C, r, qmax, forced=(), *, segment_cap: Optional[int] = None,
                    operand=()) -> Optional[Tuple[Tuple[int, int], ...]]:
    """The phases ``[(start, end), ...]`` of the plan the front answers at ``r`` (its minimal (carry, Q)), from the
    store front with ``plan_steps``' cap and the pointer front's tie order ([F3] ``operand`` as ``plan_steps``')."""
    sizes = tuple(int(s) for s in sizes)
    cap = binding_cap(sizes, C, forced, qmax, segment_cap)
    store = planned_store(sizes, tuple(int(v) for v in live), C, r, qmax, forced, segment_cap=cap, operand=operand)
    return None if store is None else store.path()
