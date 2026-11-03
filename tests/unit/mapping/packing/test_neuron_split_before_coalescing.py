"""[WI-5f] Neuron split before axon coalescing: a soft core wider than every core AND holding more neurons than every
type that can coalesce its axons packs as neuron windows, each on a fused core of ceil(axons / A) cores.

Before the rule the engine refused such a core ("does not fit any unused type even with coalescing"): the fusion
needs one type with enough neurons, the split into an unused core needs one type with enough axons. The rule runs
only where the engine refused, so every program that packed before packs as before (the reference engine carries
the same branch, ``reference_engine.py``, and the equivalence tests compare whole traces).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.core_packing import canonical_is_mapping_possible, greedy_pack_softcores
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.support.schedule.schedule_split import capacity_bound

from unit.chip_simulation.value_run.integer_programs import assert_exact, build_integer_program, integer_inputs
from unit.mapping.packing.test_engine_equivalence import Hard, Soft, _fuse, _place, _split


def _pack(softs, hards, *, split=True, fuse=True):
    trace, used = [], []
    greedy_pack_softcores(softcores=softs, used_hardcores=used, unused_hardcores=hards,
                          is_mapping_possible=canonical_is_mapping_possible, place=_place(trace),
                          fuse_hardcores=_fuse if fuse else None, split_softcore=_split if split else None)
    return trace, used


class TestTheEngineSplitsNeuronsBeforeCoalescing:
    def test_a_wide_core_with_many_neurons_packs_as_fused_windows(self):
        hards = [Hard(8, 8, f"h{i}") for i in range(12)]
        trace, used = _pack([Soft(20, 12, "w")], hards)
        # 12 neurons on 8-neuron cores: windows of 8 and 4; 20 axons on 8-axon cores: 3 cores fused per window.
        assert [(h.get_input_count(), h.get_output_count()) for h in used] == [(24, 8), (24, 8)]
        assert sorted(n for _idx, _hard, _soft, _a, n in trace) == [4, 8]
        assert all(a == 20 for _idx, _hard, _soft, a, _n in trace)
        assert len(hards) == 6

    def test_the_widest_coalescing_type_cuts_the_windows(self):
        hards = [Hard(8, 4, f"a{i}") for i in range(6)] + [Hard(8, 8, f"b{i}") for i in range(6)]
        trace, used = _pack([Soft(20, 12, "w")], hards)
        assert used[0].get_output_count() == 8                       # the 8-neuron type, not the 4-neuron one

    def test_without_splitting_the_core_is_refused_by_the_same_message(self):
        with pytest.raises(RuntimeError, match="does not fit any unused type even with coalescing"):
            _pack([Soft(20, 12, "w")], [Hard(8, 8, f"h{i}") for i in range(12)], split=False)

    def test_without_coalescing_the_core_is_refused(self):
        with pytest.raises(RuntimeError, match="No more hard cores available"):
            _pack([Soft(20, 12, "w")], [Hard(8, 8, f"h{i}") for i in range(12)], fuse=False)

    def test_too_few_cores_to_coalesce_a_window_is_refused(self):
        with pytest.raises(RuntimeError, match="does not fit any unused type even with coalescing"):
            _pack([Soft(20, 12, "w")], [Hard(8, 8, f"h{i}") for i in range(2)])


class TestTheShapeOnlyPlane:
    def test_pack_layout_places_the_core_as_fused_windows_under_coalescing(self):
        """[W04] The layout packer keeps the wide core whole and fuses its windows, as the hard-core packer does;
        without declared coalescing it refuses the core, as the hard-core packer does."""
        spec = LayoutSoftCoreSpec(input_count=20, output_count=12)
        result = pack_layout(softcores=[spec], core_types=[LayoutHardCoreType(8, 8, 12)],
                             allow_neuron_splitting=True, allow_coalescing=True)
        assert result.feasible and result.split_fragment_count == 1
        assert [s.axons_per_core for s in result.used_core_snapshots] == [24, 24]
        refused = pack_layout(softcores=[spec], core_types=[LayoutHardCoreType(8, 8, 12)],
                              allow_neuron_splitting=True, allow_coalescing=False)
        assert not refused.feasible and "CoreCapacityRefusal" in (refused.error or "")

    def test_the_capacity_bound_counts_windows_times_fused_cores(self):
        bound = capacity_bound([LayoutSoftCoreSpec(input_count=20, output_count=12)],
                               [LayoutHardCoreType(8, 8, 12)], allow_coalescing=True, allow_splitting=True)
        assert bound is not None and bound.per_softcore == (6,)
        unfused = capacity_bound([LayoutSoftCoreSpec(input_count=20, output_count=12)],
                                 [LayoutHardCoreType(8, 8, 12)], allow_coalescing=False, allow_splitting=True)
        assert unfused is not None and unfused.per_softcore[0] > 12


def _wide_cnn(seed: int = 0) -> nn.Module:
    """A conv of 3 x 3 x 6 = 54 inputs into 20 channels, then one of 180 inputs into 20: wide and many-neuron
    on 16 x 8 cores; the classifier is output-tiled by the IR mapper."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(6, 20, 3, padding=1), nn.ReLU(), nn.Conv2d(20, 20, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(20, 5),
    ).eval()


SHAPE = (6, 4, 4)


@pytest.fixture(scope="module")
def program():
    """The wide CNN as an integer value program (G2's law: every output an exact code, so atol 0 is meaningful)."""
    return build_integer_program(_wide_cnn(), SHAPE, num_classes=5, max_axons=16, max_neurons=8,
                                 allow_coalescing=True)


def _run(hybrid, x):
    with torch.no_grad():
        return ValueHybridCoreFlow(hybrid, dtype=torch.float64, device="cpu")(x.double())


class TestTheExecutedPlane:
    def test_the_vehicle_is_wide_and_many_neuron(self, program):
        banked = [c for c in program.ir.get_neural_cores() if c.weight_bank_id is not None]
        assert banked and all(c.get_input_count() > 16 and c.get_output_count() > 8 for c in banked)

    @pytest.mark.parametrize("passes", [None, 64], ids=["pool", "scheduled"])
    def test_it_packs_and_equals_the_identity_program_and_the_golden_at_atol_0(self, program, passes):
        hybrid = program.packed_mapping(axons=16, neurons=8, count=4000 if passes is None else 400,
                                        split=True, coalescing=True, passes=passes)
        fused = [hc for st in hybrid.stages if st.kind == "neural" for hc in st.hard_core_mapping.cores
                 if hc.fused_component_axons]
        assert fused and all(hc.get_output_count() == 8 for hc in fused)
        x = integer_inputs(program, 32)
        got = _run(hybrid, x)
        assert_exact(got, _run(program.identity().hybrid_mapping, x))
        codes, steps = run_golden(hybrid, x)
        assert_exact(got, torch.as_tensor(codes * steps))

    def test_fragments_keep_their_bank_and_contiguous_neuron_windows(self, program):
        hybrid = program.packed_mapping(axons=16, neurons=8, count=4000, split=True, coalescing=True)
        windows = {}
        for st in hybrid.stages:
            if st.kind != "neural":
                continue
            for placements in st.hard_core_mapping.soft_core_placements_per_hard_core:
                for p in placements:
                    if p.get("split_group_id") is not None and p.get("weight_bank_id") is not None:
                        windows.setdefault(p["ir_node_id"], []).append(tuple(p["bank_neuron_range"]))
        assert windows
        for ranges in windows.values():
            ranges = sorted(ranges)
            assert ranges[0][0] == 0 and ranges[-1][1] == 20
            assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


class TestLifSemantics:
    """A LIF conv chain on 32 x 16 cores: the second conv reads 54 axons into 20 neurons, wide and many-neuron, so it
    packs only by the neuron split onto fused cores; the pool, the scheduled and the position-multiplexed programs
    equal the identity program, outputs and per-node spike counts."""

    CORES = [{"max_axons": 32, "max_neurons": 16, "count": 400, "has_bias": True}]

    def test_every_program_equals_the_identity_program(self):
        from mimarsinan.certification.count_alignment import certify_twin_flow_counts
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_hybrid_hard_core_mapping, build_identity_hybrid_mapping)
        from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
        from unit.mapping.position_multiplexing import vehicles as V

        ir = V.lif_ir(V.lif_conv_chain(widths=(6, 20)), 32, 16, coalescing=True)
        assert any(c.get_input_count() > 32 and c.get_output_count() > 16 for c in ir.get_neural_cores())
        identity = V.lif_flow(build_identity_hybrid_mapping(ir_graph=ir))
        pool = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=self.CORES, strategy=MappingStrategy.resolve(
            ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True)))
        programs = [pool] + [V.build(ir, self.CORES, m, allow_coalescing=True, allow_neuron_splitting=True)
                             for m in (False, True)]
        assert any(hc.fused_component_axons for st in pool.stages if st.kind == "neural"
                   for hc in st.hard_core_mapping.cores)
        for batch in range(2):
            x = torch.rand(4, *V.LIF_SHAPE, generator=torch.Generator().manual_seed(batch))
            with torch.no_grad():
                want = identity(x)
                for hybrid in programs:
                    assert torch.equal(V.lif_flow(hybrid)(x), want)
            assert 0 < float(want.mean()) < V.T
            for hybrid in programs:
                cert, report, _ = certify_twin_flow_counts(ir, identity, V.lif_flow(hybrid), x, backend="hcm",
                                                           discipline="streaming", reference_discipline="streaming")
                assert cert.passed and cert.neuron_windows_compared > 0, (cert.summary(), report)
