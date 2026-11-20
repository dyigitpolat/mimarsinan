"""[U-BRANGE, U-BRANGE2] One range rule for every calibrated grid: the range of least squared error of the grid.

A calibrated grid (a host to chip boundary's entry grid, a perceptron's readout grid, a lane's grid, a bank's payload
grid; on the spiking route a LIF staircase's theta, an entry encode's gauge, a signed seam's window) reads the values
at its site and writes them onto ``levels`` levels of step ``span / levels`` above its low end, rounding half away from
the low end and saturating at both ends. Its range is the one at which that grid departs least from the values it
reads: the squared error of the site's own grid, the rounding inside the range and the clamp beyond it both counted,
summed over the calibration batches. [U-BRANGE2 (c)] The search covers the whole interval up to the calibration
maximum: ``RANGE_CANDIDATES_PER_OCTAVE`` geometrically spaced spans per octave over ``RANGE_OCTAVES`` octaves, the
maximum itself included; a window with a low end below 0 (a signed seam) takes its low end from the same candidates
on the negative side. A value is weighed by its error, never by its rarity, so the rule clamps where the error minimum
says so, at any bit width and at any number of levels.

[U-BRANGE2, second form] The search is exact at any width with a transient bounded independent of candidates times
levels (``range_search``). A site reads its values from a source: a module's forward input or output (``RangeSite``),
or any pass the caller runs (``least_squares_recorded``), every value read, none sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mimarsinan.mapping.support.range_search import (
    RANGE_CANDIDATES,
    RANGE_CANDIDATES_PER_OCTAVE,
    RANGE_OCTAVES,
    RangeCensus,
    RangeSearchError,
    SiteSpec,
    search_ranges,
)

__all__ = ["RANGE_CANDIDATES", "RANGE_CANDIDATES_PER_OCTAVE", "RANGE_OCTAVES", "RANGE_RULE",
           "RangeCensus", "RangeSearchError", "RangeSite", "least_squares_pooled", "least_squares_ranges",
           "least_squares_recorded", "positions_above"]

#: The rule in words, as the step lines and the reports print it.
RANGE_RULE = ("least squared error of the site's own grid over the calibration batches (rounding and clamp counted), "
              f"{RANGE_CANDIDATES} geometric candidates over {RANGE_OCTAVES} octaves up to the calibration maximum")

Record = Callable[[int, torch.Tensor], None]


@dataclass(frozen=True)
class RangeSite:
    """One calibrated range read from a module: ``module`` is the module whose forward input (``reads="input"``, its
    pre-hook) or output (``reads="output"``) the grid reads, ``levels`` the grid's levels above its low end
    (``2^(b-1) - 1`` on a signed grid's magnitudes, ``2^b - 1`` on an unsigned lane grid, ``T`` on a LIF staircase),
    ``magnitudes`` the values the grid covers from that tensor (``|x|`` unless the site says otherwise), ``signed``
    whether the grid's low end is searched too (a window below 0, the values themselves rather than magnitudes)."""

    module: nn.Module
    levels: int
    magnitudes: Callable[[torch.Tensor], torch.Tensor] = torch.abs
    reads: str = "input"
    signed: bool = False


def _run_hooked(model: nn.Module, sites: Sequence[RangeSite], record: Record, batches: Sequence,
                only: Optional[Sequence[int]] = None) -> None:
    handles = []
    for i in (range(len(sites)) if only is None else only):
        site = sites[i]

        def hook(module, args, output: Any = None, i=i, site=site):
            value = output if site.reads == "output" else args[0]
            record(i, site.magnitudes(value.detach()))

        if site.reads == "output":
            handles.append(site.module.register_forward_hook(lambda m, a, o, h=hook: h(m, a, o)))
        else:
            handles.append(site.module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()


def least_squares_recorded(run_pass: Callable[[Record, Optional[Sequence[int]]], None],
                           specs: Sequence[SiteSpec], *, full: bool = False) -> List[RangeCensus]:
    """Each site's range by the rule from a source the caller runs: ``run_pass(record, only)`` runs one pass over the
    calibration data and calls ``record(i, values)`` with every value site ``i`` reads (all of them, never a sample),
    for the sites in ``only`` (every site when None); it is called two or three times and must read the same values in
    the same calls each time (a call with no values is no call). ``full``: the full search (``range_search``)."""
    return search_ranges(run_pass, specs, full=full)


def least_squares_ranges(model: nn.Module, sites: Sequence[RangeSite], batches: Iterable, *,
                         full: bool = False) -> List[RangeCensus]:
    """Each module site's range by the rule (module docstring) and its clamp census. ``full``: the full search."""
    batches = list(batches)
    return search_ranges(lambda record, only: _run_hooked(model, sites, record, batches, only),
                         [SiteSpec(int(s.levels), signed=s.signed) for s in sites], full=full)


def least_squares_pooled(model: nn.Module, sites: Sequence[RangeSite], unions: Sequence[Sequence[int]],
                         batches: Iterable, *, full: bool = False) -> List[RangeCensus]:
    """[U-BRANGE2, the review's m-3] Each module site's census by the rule where sites read together are one site:
    each of ``unions`` (site indices, disjoint) whose members share one grid's levels and form is one range site, its
    values pooled over every member (each member's record a call of the union), and every member takes the union's
    census; every other site is its own. A union of members on different grids stays apart (the law solvers refuse
    such a union by name)."""
    batches = list(batches)
    group_of = list(range(len(sites)))
    members: Dict[int, List[int]] = {i: [i] for i in range(len(sites))}
    for union in unions:
        union = sorted(set(int(i) for i in union))
        forms = {(int(sites[i].levels), bool(sites[i].signed)) for i in union}
        if len(union) > 1 and len(forms) == 1:
            for i in union:
                members.pop(i, None)
                group_of[i] = union[0]
            members[union[0]] = union
    order = sorted(members)
    index = {g: k for k, g in enumerate(order)}

    def run_pass(record: Record, only: Optional[Sequence[int]]) -> None:
        wanted = None if only is None else [i for k in only for i in members[order[k]]]
        _run_hooked(model, sites, lambda i, v: record(index[group_of[i]], v), batches, wanted)

    pooled = search_ranges(run_pass, [SiteSpec(int(sites[g].levels), signed=sites[g].signed) for g in order],
                           full=full)
    return [pooled[index[group_of[i]]] for i in range(len(sites))]


def positions_above(run_pass: Callable[[Record, Optional[Sequence[int]]], None], ranges: Sequence[Tuple[float, float]],
                    ) -> List[Dict[str, int]]:
    """The census of systematic clamps: per site, the positions (an index within one sample: every axis but the first)
    whose value lies above the range's top (or below its low end) on every calibration sample, and the samples read.
    ``ranges`` holds each site's ``(low, top)``; the source's values carry the samples on their first axis."""
    always: List[Optional[torch.Tensor]] = [None] * len(ranges)
    always_low: List[Optional[torch.Tensor]] = [None] * len(ranges)
    samples = [0] * len(ranges)

    def record(i: int, values: torch.Tensor) -> None:
        low, top = ranges[i]
        v = values.detach()
        if v.numel() == 0:  # an empty call is no call: no sample read
            return
        v = v.reshape(v.shape[0], -1) if v.dim() > 1 else v.reshape(1, -1)
        above = (v > top).all(dim=0).cpu()
        below = (v < low).all(dim=0).cpu()
        seen_above, seen_below = always[i], always_low[i]
        always[i] = above if seen_above is None else (seen_above & above)
        always_low[i] = below if seen_below is None else (seen_below & below)
        samples[i] += int(v.shape[0])

    run_pass(record, None)
    return [{"positions_above": int(a.sum()) if a is not None else 0,
             "positions_below": int(b.sum()) if b is not None else 0,
             "positions": int(a.numel()) if a is not None else 0, "samples": n}
            for a, b, n in zip(always, always_low, samples)]
