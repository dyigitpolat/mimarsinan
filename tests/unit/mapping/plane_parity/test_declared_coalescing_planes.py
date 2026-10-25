"""[integration of the exploration unit with pack's W04 class 2] Every shape-only consumer the exploration added reads
the declared coalescing permission: a softcore wider than every declared core fuses cores only where the chip declares
coalescing, as the hard-core packer does, and is refused by name (the capacity refusal both planes name) where it does
not. The exploration's verify found the regime map pricing 1,239 as-declared spiking points that no build places
(a 145-axon conv softcore on Loihi's and ODIN's 128-axon cores): fused on the shape-only plane, refused by the builder.
The witness here is that softcore (16 channels of 3 x 3 taps and a bias row, 16 outputs) on 128 x 16 cores."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.introspection import scaling_ladder as ladder
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal, refusal_name_of_message
from mimarsinan.mapping.packing.residency_tiles import tile_groups
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import unscheduled_pack_holds
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats

AXONS, NEURONS, COUNT = 128, 16, 8
CORES = [LayoutHardCoreType(AXONS, NEURONS, COUNT)]


def _specs():
    """Two layers: the 145-axon conv softcore and a 16-axon reader of it."""
    return [LayoutSoftCoreSpec(input_count=145, output_count=16, segment_id=0, latency_tag=0, perceptron_index=0,
                               name="conv"),
            LayoutSoftCoreSpec(input_count=17, output_count=10, segment_id=0, latency_tag=1, perceptron_index=1,
                               name="head")]


def _kwargs(coalescing: bool, scheduling: bool = True):
    return dict(allow_scheduling=scheduling, allow_neuron_splitting=False, allow_coalescing=coalescing,
                max_schedule_passes=8)


class TestWithoutDeclaredCoalescing:
    def test_the_layout_packer_refuses_by_name(self):
        pack = pack_layout(softcores=_specs(), core_types=CORES, allow_coalescing=False)
        assert not pack.feasible and refusal_name_of_message(pack.error or "") == "CoreCapacityRefusal"

    @pytest.mark.parametrize("scheduling", [False, True])
    def test_the_mapping_stats_refuse_by_name(self, scheduling):
        stats, error = compute_mapping_stats(_specs(), CORES, **_kwargs(False, scheduling))
        assert not stats.feasible and "CoreCapacityRefusal" in (error or "")

    def test_the_noc_fragments_refuse_by_name(self):
        with pytest.raises(CoreCapacityRefusal):
            collect_noc_fragments(softcores=_specs(), core_types=CORES, census=None, **_kwargs(False))

    def test_the_residency_tile_groups_refuse_by_name(self):
        with pytest.raises(ResidencyPlanError, match="CoreCapacityRefusal") as caught:
            tile_groups(_specs(), CORES, allow_coalescing=False, allow_splitting=False)
        assert isinstance(caught.value, ToolchainRefusal)

    def test_the_fit_rule_sees_no_unscheduled_pack(self):
        pack = pack_layout(softcores=_specs(), core_types=CORES, allow_coalescing=False)
        assert not unscheduled_pack_holds(pack, CORES)

    def test_the_ladder_footprint_refuses_by_name(self):
        platform = {"cores": [{"max_axons": AXONS, "max_neurons": NEURONS, "count": COUNT}],
                    "allow_coalescing": False, "allow_neuron_splitting": False, "allow_scheduling": True}
        with pytest.raises(ladder.LadderGeometryError, match="CoreCapacityRefusal"):
            ladder.packed_tiles(SimpleNamespace(softcores=tuple(_specs())), platform)


class TestWithDeclaredCoalescing:
    def test_every_plane_fuses_the_wide_softcore_on_two_cores(self):
        pack = pack_layout(softcores=_specs(), core_types=CORES, allow_coalescing=True)
        assert pack.feasible
        fused = [s for s in pack.used_core_snapshots if s.used_axons == 145]
        assert len(fused) == 1 and fused[0].axons_per_core == 2 * AXONS and fused[0].tiles == 2
        stats, error = compute_mapping_stats(_specs(), CORES, **_kwargs(True))
        assert error is None and stats.feasible
        frag = collect_noc_fragments(softcores=_specs(), core_types=CORES, census=None, **_kwargs(True))
        assert sum(p.tiles for p in frag.pass_programs) == 3
        groups = tile_groups(_specs(), CORES, allow_coalescing=True, allow_splitting=False)
        assert sum(t for _, cores in groups for _, t in cores) == 3
        platform = {"cores": [{"max_axons": AXONS, "max_neurons": NEURONS, "count": COUNT}],
                    "allow_coalescing": True, "allow_neuron_splitting": False, "allow_scheduling": True}
        assert ladder.packed_tiles(SimpleNamespace(softcores=tuple(_specs())), platform) == 3
