"""[R3] The NoC model prices the placement that RUNS, not the planned one.

The runner maps EACH execution stage from tile 0 — a re-timed program runs
one small remapped chip per depth level. The twin priced the pass's
co-resident placement instead: 113.6 modeled hops on the loihi MLP where the
executed program measured a TRUE 0 (three level stages of 2/4/1 cores, each
inside one tile).
"""

from __future__ import annotations

import pytest

from mimarsinan.chip_simulation.sanafe.noc_estimate import estimate_noc
from mimarsinan.mapping.noc import execution_stage_placements
from mimarsinan.mapping.noc.fragments import LayoutNocFragments
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus


class _Spec:
    def __init__(self, segment, level):
        self.segment_id = segment
        self.latency_tag = level
        self.output_count = 4


#: Seven softcores in one pass: levels 0/1/2 hold 2/4/1 of them — the loihi
#: MLP's shape. Pass placement spreads them over hardcores 0..6 (2 tiles at 4
#: cores/tile); each LEVEL alone fits one tile.
SOFTCORES = [_Spec(0, 0), _Spec(0, 0), _Spec(0, 1), _Spec(0, 1),
             _Spec(0, 1), _Spec(0, 1), _Spec(0, 2)]
PASS_PLACEMENTS = (tuple((i, i) for i in range(7)),)
#: Level-1 softcore 2 reads level-0 softcore 0 — cross-LEVEL, host-mediated
#: under re-timing; softcores 2→3 share level 1 (a same-stage wire); 0→5
#: crosses both a level AND, under the fused co-resident placement, a tile.
WIRES = {(0, 2): 5, (2, 3): 3, (0, 5): 4}


def _estimate(*, retimed):
    stages = execution_stage_placements(
        SOFTCORES, PASS_PLACEMENTS, retimed=retimed)
    fragments = LayoutNocFragments(
        pass_placements=PASS_PLACEMENTS,
        census=LayoutWireCensus(
            pair_wires=dict(WIRES),
            input_wires=(0,) * 7, on_wires=(0,) * 7,
        ),
        stage_placements=stages,
    )
    return estimate_noc(
        fragments=fragments, cores_per_tile=4, mesh_height=1,
        activity_factor=1.0, timesteps=2,
        stage_placements=fragments.stage_placements,
    )


class TestTheExecutedPlacement:
    def test_a_retimed_program_prices_no_cross_tile_hops(self):
        """Each level remaps from tile 0 and fits inside it: the cross-level
        wire re-enters as carried input, the same-level wire stays intra-tile
        — zero hops, matching the measured truth."""
        estimate = _estimate(retimed=True)
        assert estimate.total_hops == 0.0
        assert estimate.inter_tile_packets == 0.0
        # the cross-level wire is carried re-entry; the same-level wire is mesh
        assert estimate.total_packets == pytest.approx((5 + 3 + 4) * 1.0 * 2)

    def test_a_fused_program_still_prices_the_co_resident_crossing(self):
        """Without re-timing the pass IS the execution stage: hardcore 4+ sits
        on tile 1 and cross-tile wires cost hops — the pricing a fused
        program's chip really pays."""
        estimate = _estimate(retimed=False)
        assert estimate.total_hops > 0.0

    def test_stage_local_reindexing_starts_every_stage_at_core_zero(self):
        stages = execution_stage_placements(
            SOFTCORES, PASS_PLACEMENTS, retimed=True)
        assert [sorted({hc for _, hc in stage}) for stage in stages] == [
            [0, 1], [0, 1, 2, 3], [0],
        ]
