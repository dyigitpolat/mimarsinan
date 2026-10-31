"""[U-PP P-1] The candidate plane's pass count is the pass structure's that supplies its latency steps (SD-13, SDB-B2).

The candidate view read ``pass_count`` (and ``sync_count``) off whatever layout statistics its caller handed it, while
its latency steps, programming census and carry come from the NoC fragments (the shared pass planner on the builder's
segments). A caller that packs the statistics without the walk's segments (the baseline rerun's candidate plane, the
plane-parity gate) plans them on the depth segments, so on a program whose two segmentations differ the candidate's
pass count contradicted its own schedule: SD-13 measured 18 against 21 passes and stages on ResNet-18 at the PIMCOMP
geometry (117 of 120 rows). A residual block with a projection shortcut is that program reduced: the census reads the
fragments' schedule census, which is the sealed schedule record's, at the footprint and on a constrained budget.
"""

from __future__ import annotations

import math

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.support.residency_rules import spec_runs
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats

from . import planes as P

A, N = 16, 16


@pytest.fixture(scope="module")
def proj():
    rep = P.representation("proj")
    layout, specs = P.walk(rep, A, N)
    return rep, layout, specs, P.footprint(specs, A, N)


def _budgets(F: int):
    return {"unbounded": F, "constrained": math.ceil(F / 4)}


@pytest.mark.parametrize("budget", ["unbounded", "constrained"])
def test_candidate_pass_count_equals_the_shape_only_passes_and_the_executed_stages(proj, budget):
    rep, layout, specs, F = proj
    count = _budgets(F)[budget]
    shape = P.shape_answer(layout, specs, A, N, count, scheduled=True)
    executed = P.executed_answer(rep, A, N, count, scheduled=True)
    assert shape.ok and executed.ok, (shape, executed)
    assert shape.census[0] == shape.passes == executed.passes
    assert shape.census == executed.census


@pytest.mark.parametrize("budget", ["unbounded", "constrained"])
def test_the_vehicle_is_the_divergence_class(proj, budget):
    """Not vacuous: the statistics as the gate packs them (no runs) plan another pass count than the builder's
    segments do, so a candidate that read them would contradict its record."""
    _rep, layout, specs, F = proj
    count = _budgets(F)[budget]
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=True,
                            max_schedule_passes=P.PASSES)
    types = [LayoutHardCoreType(A, N, int(count))]
    stats, _error = compute_mapping_stats(specs, types, **caps.layout_kwargs())
    frag = collect_noc_fragments(softcores=specs, core_types=types, census=None, runs=spec_runs(layout),
                                 **caps.layout_kwargs())
    assert stats.schedule_pass_count != len(frag.pass_placements)


def test_the_fragments_state_the_schedule_census_of_their_program(proj):
    """The fragments' census by the record's rule: the planner's passes per builder segment when scheduled (passes,
    passes less segments), (0, 0) for the single pool (no scheduling: the stages carry no pass index)."""
    rep, layout, specs, F = proj
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=True,
                            max_schedule_passes=P.PASSES)
    types = [LayoutHardCoreType(A, N, int(F))]
    frag = collect_noc_fragments(softcores=specs, core_types=types, census=None, runs=spec_runs(layout),
                                 **caps.layout_kwargs())
    assert frag.schedule_census == P.executed_answer(rep, A, N, F, scheduled=True).census
    flat = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=False,
                            max_schedule_passes=P.PASSES)
    single = collect_noc_fragments(softcores=specs, core_types=types, census=None, **flat.layout_kwargs())
    assert single.schedule_census == (0, 0) == P.executed_answer(rep, A, N, F, scheduled=False).census
