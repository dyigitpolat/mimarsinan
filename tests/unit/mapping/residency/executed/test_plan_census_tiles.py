"""[integration of W03 with pack's physical tiles] The plan census counts physical tiles, as the record and the
candidate planes count them (pack: a fused core counts its members). Since pack's W04 class 1 the layout packer keeps
a wide spec whole on one fused core, so a composition's packed core can occupy several tiles; the plan census counted
packed cores, and the certification read t0_03 x1.15 under index 1 as "pass 11 cores 27 against 18" (record 27
tiles, plan 18 fused cores, one packing). Each pass's cores and kept cores are its packed cores' tiles."""

from __future__ import annotations

from mimarsinan.mapping.support.schedule.residency_units import PlanPeriodFacts, facts_period_census


def _facts(r: int) -> PlanPeriodFacts:
    """Two passes on a budget of 4 tiles: pass 0 one fused core of two tiles (cells 200) and one plain core (cells
    50); pass 1 one plain core (cells 80). r = 4 pins nothing; r = 1 pins pass 0's fused core (largest payload)."""
    return PlanPeriodFacts(r=r, budget=4, phases=((0, 1), (1, 2)), pass_tiles=(3, 1), pass_pinned=(0, 0),
                           boundary_elements=(0,), pass_cores=(((200, 2), (50, 1)), ((80, 1),)),
                           pass_resident_cores=(frozenset(), frozenset()) if r == 4 else (frozenset({0}), frozenset()))


def _census(r: int):
    return facts_period_census(_facts(r), bytes_of_cells=int, tile_cells=0, bytes_per_element=1)


def test_a_pass_states_its_packed_cores_tiles():
    census = _census(4)
    assert [p.cores for p in census.passes] == [3, 1]
    assert [p.resident_cores for p in census.passes] == [0, 0]
    assert [p.written_cells_backing_store for p in census.passes] == [250, 80]


def test_a_pinned_fused_core_keeps_all_its_tiles():
    census = _census(1)
    assert [p.cores for p in census.passes] == [3, 1]
    assert [p.resident_cores for p in census.passes] == [2, 0]
    assert [p.written_cells_backing_store for p in census.passes] == [50, 80]
