"""[W04] Plane parity: every point the shape-only plane treats as feasible builds on the executed plane, and every
point both refuse is refused by the same name; a built scheduled point runs the passes the shape-only plane planned
on the builder's segments, and the shape-only footprint is the executed one.

The vehicles and geometries are the small reproductions of the divergence classes the census found on the roster
(RA/11_wave2/pack/plane_parity.md): a flat program on exactly the shape-only footprint refused by the executed packer
(the layout packer pre-split wide cores into independent fragments where the hard-core packer fuses them), planned
passes the hard-core packer could not place (bank affinity and coalescing groups unseen by the layout packer), bank
residency compositions whose executed placements diverged across passes, and refusals of one point under two names.
"""

from __future__ import annotations

import math

import pytest

from . import planes as P

PAIRS = [("conv_odd", 64, 16), ("conv_odd", 32, 32), ("pw", 32, 16), ("pw", 16, 8), ("pw", 64, 16),
         ("res", 64, 16), ("res", 16, 16), ("wide_fc", 16, 16)]
SMALL = (1, 2, 3, 4, 6, 8)


def _budgets(F: int):
    return sorted({F, *(max(1, math.ceil(F / d)) for d in (2, 4, 8, 16)), *SMALL})


@pytest.fixture(scope="module")
def reps():
    return {name: P.representation(name) for name in {p[0] for p in PAIRS}}


@pytest.mark.parametrize("name,a,n", PAIRS, ids=[f"{p[0]}-{p[1]}x{p[2]}" for p in PAIRS])
def test_the_shape_only_footprint_is_the_executed_one(reps, name, a, n):
    rep = reps[name]
    layout, specs = P.walk(rep, a, n)
    bound = sum(-(-int(s.input_count) // a) * -(-int(s.output_count) // n) for s in specs)
    executed = P.executed_answer(rep, a, n, bound, scheduled=False)
    assert executed.ok
    assert P.footprint(specs, a, n) == executed.cores


@pytest.mark.parametrize("name,a,n", PAIRS, ids=[f"{p[0]}-{p[1]}x{p[2]}" for p in PAIRS])
def test_every_point_agrees_on_both_planes(reps, name, a, n):
    rep = reps[name]
    layout, specs = P.walk(rep, a, n)
    F = P.footprint(specs, a, n)
    points = [(F, False)] + [(count, True) for count in _budgets(F)]
    disagreements = []
    for count, scheduled in points:
        shape = P.shape_answer(layout, specs, a, n, count, scheduled=scheduled)
        executed = P.executed_answer(rep, a, n, count, scheduled=scheduled)
        if shape.ok != executed.ok or (shape.ok and scheduled and shape.passes != executed.passes) or (
                not shape.ok and shape.refusal != executed.refusal):
            disagreements.append((count, scheduled, shape, executed))
    assert not disagreements


def test_a_budget_below_one_window_is_refused_by_one_name_on_both_planes(reps):
    rep = reps["wide_fc"]
    layout, specs = P.walk(rep, 16, 16)
    for scheduled in (False, True):
        shape = P.shape_answer(layout, specs, 16, 16, 1, scheduled=scheduled)
        executed = P.executed_answer(rep, 16, 16, 1, scheduled=scheduled)
        assert (shape.ok, executed.ok) == (False, False)
        assert shape.refusal == executed.refusal == "CoreCapacityRefusal"
