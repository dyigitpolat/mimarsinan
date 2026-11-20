"""[U-LIF2 W11, the review's p-3] The latency rule's unit under a pass cut: D is read per execution stage (pass and
segment) from that stage's own wiring, a slice an earlier pass carried read as a stage input.

A fused chain of three cores cut into two passes runs two execution stages: under integrate then emit their depths are
1 and 2 and the program runs 31 + 47 = 78 cycles at T = 15, where the uncut stage of depth 3 runs 63; under firing
while integrating the per-cycle rule prices each pass at T + L + 1. Red on the executor side's tree: the depth reader
does not exist there."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

FWI, ITE = "fire_while_integrating", "integrate_then_emit"
T = 15


def _chain(passes):
    from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
    from mimarsinan.mapping.noc.wire_census import LayoutWireCensus

    softcores = [LayoutSoftCoreSpec(input_count=8, output_count=6, latency_tag=tag, segment_id=0) for tag in (0, 1, 2)]
    census = LayoutWireCensus(pair_wires={(0, 1): 6, (1, 2): 6}, input_wires=(8, 0, 0), on_wires=(0, 0, 0))
    placements = [[(i, i) for i in members] for members in passes]
    return softcores, SimpleNamespace(pass_placements=placements, census=census, runs=None,
                                      pass_programs=[SimpleNamespace(wavefront=False) for _ in passes])


@pytest.mark.parametrize("passes,depths,fwi,ite", [
    ([[0, 1, 2]], (3,), 18, 63),
    ([[0], [1, 2]], (1, 2), 33, 78),
    ([[0, 1], [2]], (2, 1), 33, 78),
])
def test_p3_a_fused_stage_cut_into_passes_reads_its_depth_per_pass(passes, depths, fwi, ite):
    from mimarsinan.search.problems.joint.candidate_fragments import candidate_latency_steps, stage_semantics_of

    depth_of = getattr(importlib.import_module("mimarsinan.mapping.noc.stages"), "execution_stage_depths")
    softcores, noc = _chain(passes)
    assert depth_of(softcores, noc.pass_placements, retimed=False, census=noc.census) == depths
    priced = {value: candidate_latency_steps(softcores, noc, stage_semantics_of("lif", None, retimed=False,
                                                                                window_firing=value), T)
              for value in (FWI, ITE)}
    assert priced == {FWI: fwi, ITE: ite}
    assert ite == sum((d + 1) * T + d for d in depths)
