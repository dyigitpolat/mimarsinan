"""CL3 (a)-(c): G5 acceptance 7 at relative error 0 on the E85 pricing fixture, per (member, C, K) of the fixture.

The fixture (``residency_fixture.load_document("e85_price")``) holds ``x9_0_v3.price`` on the estimator's own member
sets (CIFAR ResNet-18 width and R-P members on the COMPASS geometry, VGG-16 width members on the ISAAC-like
geometry), each on its chip grid, at K in {1, 16, 64}, under every energy crossbar technology row at each bracket
end, on both arms (residency and stateless).

(a) The toolchain step function (``residency_steps``) equals ``plan_steps`` step for step; a ``price`` of None (a
    layer exceeds C) is a toolchain refusal by name; F <= C gives the resident census.
(b) Under a test-only physics that transcribes the row's per-byte constants (``tech_physics``: the commit
    ``e_program_per_byte`` = e_byte with ``e_dma_per_byte`` 0, so the programmed byte costs e_byte; the carry read
    out at ``e_readout_per_byte`` = 2 c_byte with the inject at ``e_dma_per_byte`` 0, so the carry over both
    directions costs c_byte per byte), the argmin over the toolchain steps of the toolchain-priced writes plus carry
    (``step_choice.cheapest_step``) reproduces ``price``'s W, carry and Q. Where two steps price exactly equal in
    rational arithmetic, ``price`` breaks the tie by the rounding of its float sum with comp0 and the toolchain by
    its rule (the first step, the smaller r): both choices must then be exact minima, and each such tie is listed.
(c) The write price (``price_write_terms``) equals W x TB x e_byte / K and the carry price (the carry components)
    equals carry_el x cb x c_byte, each the correctly rounded value of the exact rational price (relative error 0
    to it), as ``price``'s own float terms are. The compute term comp0 is excluded by name: it prices from the RA
    cost model and is the same for every step of one (member, C), so it never moves the argmin.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Tuple

import pytest

from mimarsinan.deployment_record.cost.absolute.step_choice import cheapest_step, price_step
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.schedule.residency_units import (
    plan_period_census,
    residency_steps,
    stateless_step,
)

from .residency_fixture import load_document

#: Every 9th (member, C) row in the fast suite (25 of 225, all three member sets); the slow test runs all.
FAST_STRIDE = 9


@lru_cache(maxsize=None)
def _document() -> dict:
    return load_document("e85_price")


def tech_physics(e_byte: float, c_byte: float):
    """The test-only override that transcribes one RA ``Tech`` row at one bracket end (``x9_0_v3.py:343``)."""
    constants = {"e_program_per_byte": e_byte, "e_dma_per_byte": 0.0, "e_readout_per_byte": 2.0 * c_byte}
    return profile_from_dict({
        "format_version": 1, "name": "x9_tech_override", "display_name": "x9_0_v3 Tech (test only)",
        "description_file": "t.md", "validity": {"measurement_kind": "projection"},
        "constants": {k: {"nominal": v, "unit": "J", "evidence_kind": "estimated",
                          "note": "test-only transcription of an x9_0_v3 Tech row (CL3 acceptance (b))"}
                      for k, v in constants.items()},
    })


def _member(index: int) -> dict:
    return _document()["members"][index]


def _censuses(member: dict, C: int):
    """(the plan, the census of every step of its step function, the stateless census), stated by full tiles."""
    tiles, live = tuple(member["tiles"]), tuple(member["live"])
    axons, neurons, cell_bits = member["geom"]
    plan = residency_steps(tiles, live, C)
    census = lambda step: plan_period_census(  # noqa: E731
        tiles, live, C, step, bytes_of_cells=lambda cells: cells * cell_bits // 8, tile_cells=axons * neurons,
        bytes_per_element=member["carry_bits"] // 8)
    return plan, [census(step) for step in plan.steps], census(stateless_step(plan))


def _facts(census) -> Tuple[int, int, int]:
    """(W written tiles, carried bytes over both directions, Q) of a tile-stated census."""
    return (sum(p.cores - p.resident_cores for p in census.passes), census.carry_out_bytes + census.carry_in_bytes,
            len(census.passes))


def _exact(constant: float, multiplicand: int, divisor: int = 1, scale: int = 1) -> float:
    """The exact rational price rounded once to a double."""
    return float(Fraction(constant) * multiplicand * scale / divisor)


def _exact_costs(candidates, price: dict, TB: int) -> List[Fraction]:
    """W x TB x e_byte / K + carry x c_byte of every candidate census, in exact rational arithmetic."""
    out = []
    for census in candidates:
        W, carry, _Q = _facts(census)
        out.append(Fraction(price["e_byte"]) * W * TB / int(price["K"]) + Fraction(price["c_byte"]) * carry)
    return out


def check_row(row: dict, solutions: Dict[Tuple[str, int], dict], ties: List[tuple]) -> int:
    """Every clause of (a)-(c) on one (member, C) row of the fixture; returns the priced (arm, K, row) checks and
    appends to ``ties`` every cell where ``price`` and the toolchain pick different steps of equal exact price."""
    member = _member(row["member"])
    tiles, live, C, TB = tuple(member["tiles"]), tuple(member["live"]), row["C"], member["tile_bytes"]
    if row["plan"] is None:
        with pytest.raises(ResidencyPlanError):
            residency_steps(tiles, live, C)
        assert all(p["residency"] is None and p["stateless"] is None for p in row["prices"])
        return 0
    plan, steps, stateless = _censuses(member, C)
    expected = row["plan"]
    assert plan.as_ra_dict() == dict(steps=[tuple(s) for s in expected["steps"]], stateless=tuple(expected["stateless"]),
                                     q_fit=expected["q_fit"], F=expected["F"], fluid_r=expected["fluid_r"])
    if plan.F <= C:
        assert [_facts(c) for c in steps] == [(0, 0, 1)] and all(p.resident for p in steps[0].passes)
    checks = 0
    for price in row["prices"]:
        physics = tech_physics(price["e_byte"], price["c_byte"])
        solutions.setdefault((price["tech"], price["end"]), {
            "e_byte": price["e_byte"], "c_byte": price["c_byte"],
            "override": {"e_program_per_byte": price["e_byte"], "e_dma_per_byte": 0.0,
                         "e_readout_per_byte": 2.0 * price["c_byte"]}})
        policy = WritePolicy(batch_k=int(price["K"]))
        for arm, candidates in (("residency", steps), ("stateless", [stateless])):
            ra = price[arm]
            index = cheapest_step(candidates, physics, policy)
            facts = [_facts(c) for c in candidates]
            ra_facts = (ra["W"], int(ra["carry_bytes"]), ra["Q"])
            assert ra_facts in facts, (member["label"], C, price, arm)
            exact = _exact_costs(candidates, price, TB)
            assert exact[index] == min(exact) == exact[facts.index(ra_facts)]
            if facts[index] != ra_facts:
                ties.append((member["set"], member["label"], C, price["tech"], price["end"], price["K"], arm,
                             facts[index], ra_facts))
            if ra["mode"] == "resident":
                assert facts[index] == ra_facts == (0, 0, 1)
            for census in {facts[index]: candidates[index], ra_facts: candidates[facts.index(ra_facts)]}.values():
                W, carry, _Q = _facts(census)
                priced = price_step(census, physics, policy)
                assert priced.write.nominal == _exact(price["e_byte"], W * TB, price["K"], 1000)
                assert priced.carry.nominal == _exact(price["c_byte"], carry, 1, 1000)
            assert ra["prog"] == _exact(price["e_byte"], ra["W"] * TB, price["K"])
            assert ra["car"] == _exact(price["c_byte"], int(ra["carry_bytes"]))
            checks += 1
    return checks


def _rows(stride: int) -> List[dict]:
    return _document()["rows"][::stride]


def test_the_fixture_covers_refusals_fits_and_multi_pass_plans():
    rows = _document()["rows"]
    modes = {(p["residency"] or {"mode": "refused"})["mode"].split()[0] for r in rows for p in r["prices"]}
    assert modes == {"refused", "resident", "mux"}
    assert {m["set"] for m in _document()["members"]} == {"R18 R-W, COMPASS", "R18 R-P, COMPASS",
                                                           "VGG-16 R-W, ISAAC-like"}
    assert {r["member"] for r in _rows(FAST_STRIDE)} >= {0, 5, 10}


@pytest.mark.parametrize("index", range(0, len(_rows(FAST_STRIDE))))
def test_acceptance_7_on_a_fast_subset(index):
    assert check_row(_rows(FAST_STRIDE)[index], {}, []) >= 0


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_acceptance_7_on_the_whole_fixture():
    solutions: Dict[Tuple[str, int], dict] = {}
    ties: List[tuple] = []
    checks = sum(check_row(row, solutions, ties) for row in _rows(1))
    rows = _document()["rows"]
    print(f"[CL3 a-c] {len(rows)} (member, C) rows, {sum(r['plan'] is None for r in rows)} refused by name, "
          f"{checks} (arm, K, technology, end) cells: W, carry and Q equal on {checks - len(ties)}, an exact tie of "
          f"two steps on {len(ties)}; every price at relative error 0")
    for tie in ties:
        print(f"[CL3 b] exact tie {tie[:7]}: toolchain (first step) {tie[7]}, price (float sum with comp0) {tie[8]}")
    for (tech, end), solution in sorted(solutions.items()):
        print(f"[CL3 b] override for {tech} (end {end}): {solution['override']} "
              f"(e_byte {solution['e_byte']!r}, c_byte {solution['c_byte']!r})")
    assert checks == sum(2 * len(r["prices"]) for r in rows if r["plan"] is not None)
