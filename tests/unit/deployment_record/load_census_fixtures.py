"""[U-ENERGY] What the existing tests hand the one load law, and what a pricing without a load census refuses.

Every load of a deployment (its first load, the reprogramming per round) is priced by one law over its census
(``cost/absolute/walls/loads.py``). A test that prices a program load hands the pricer a census
(``written_program``: passes that the period census rewrites whole each round, since two or more passes write, each
core written once by the first load). A test that prices a view stating no pass structure reads each load figure
refused by the law's named reason and the rest of the pricing as before (``steady``). Every constant is a test
constant."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute.walls.loads import LOAD_FIGURES, NO_LOAD_CENSUS
from mimarsinan.deployment_record.quantities.planes.loads import LoadInputs, load_inputs

#: Every load figure of the one law: each load's wall, its energy and that energy's static leg.
LOAD_TERMS = tuple(name for _kind, *names in LOAD_FIGURES for name in names)


def steady(refusals: Mapping[str, str]) -> Dict[str, str]:
    """A pricing given no load census: each load figure is refused by the law's named reason (the view states no pass
    structure, and the law prices a census, never a scalar); returns the other refusals, the steady pricing's own."""
    for name in LOAD_TERMS:
        assert refusals.get(name) == NO_LOAD_CENSUS, (name, refusals.get(name))
    return {name: why for name, why in refusals.items() if name not in LOAD_TERMS}


def gated_loads() -> list:
    """The load figures a description's external gate refuses beside its headlines where the pricing is given no
    load census: each by the law's own reason, in the law's order (``walls.loads.load_refusals``)."""
    return [(name, NO_LOAD_CENSUS) for name in LOAD_TERMS]


def law_energy(census: Any, physics: Any, *, without: Sequence[str] = ()) -> Any:
    """The write law's energy over one load's census at K = 1, serial (the load law's write part), with the
    description's rows ``without`` left out (a difference against it isolates their part)."""
    from dataclasses import replace

    from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs

    if without:
        physics = replace(physics, constants={k: v for k, v in physics.constants.items() if k not in without})
    return price_write_inputs(WriteInputs.of(census.writes), physics, lifetime=False).energy


def written_program(*passes: Sequence[int], weight_bits: int = 8, entries: Optional[Sequence[Sequence[int]]] = None,
                    axons: int = 256, neurons: int = 256, granularity: Optional[str] = None, label: str = "test",
                    **platform: Any) -> LoadInputs:
    """The load censuses of a program of two or more written passes, each given by its cores' cells (``entries``: per
    pass, each core's connectivity entries). Its period census rewrites every pass each round, its first load writes
    every core once, so each load's payload is the cells' bytes at ``weight_bits`` (a cell per weight). The chip
    declares one core type of ``axons`` x ``neurons`` (with its ``write_granularity`` where given: a per-unit write
    time needs it); a core's extent is its cells over at most ``neurons`` columns."""
    assert len(passes) >= 2, "a lone written pass is never rewritten per round"

    def extent(cells: int) -> tuple:
        columns = max(1, min(cells, neurons))
        return (max(1, -(-cells // columns)), columns)

    rows = tuple(PassRow(segment_index=i, resident=False, core_cells=tuple(cells), marks=frozenset(),
                         core_extents=tuple(extent(c) for c in cells), compute_steps=3,
                         core_entries=() if entries is None else tuple(entries[i]))
                 for i, cells in enumerate(passes))
    core = {"max_axons": axons, "max_neurons": neurons, "count": max(len(p) for p in passes)}
    if granularity is not None:
        core["write_granularity"] = granularity
    declared = {"cores": [core], "weight_bits": weight_bits, **platform}
    return load_inputs(ProgramPasses(rows=rows), declared, weight_bits=weight_bits, label=label)
