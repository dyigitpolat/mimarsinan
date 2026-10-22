"""U1 pass projection (plan §21 item 1): every DAG-monotone cut the scheduler makes runs equal to the identity run.

A pass is the fused segment's execution restricted to its cores, every cross-pass read served as the fused trace
serves it (``u1_design.md`` §5.1): R1 pins (a pass runs its cores at the fused ALAP latencies where its own timing
would move an edge), R2 delivery delays (a carried slice replayed ``d = Lc - 1 - Lp`` cycles late), R3 a carry store
scoped to the producing schedule segment, R4 per-slice input kinds (in the fused stage's channel positions), R5 level
units under per-hop re-timing, R6 dependency-ordered value walks. U1 keeps the capacity split. Two program classes
outside the random-DAG family refuse by name at build (``TestTheBuildRefusalClasses``): a windowed-LIF segment that
streams (coalescing or psum groups keep it whole) cut with a carried wire, and a hard core that co-packs soft cores the
fused segment runs at two latencies (instances of one layer read at two depths), single pass included.

The witnesses are the re-verifiers' vehicles, each inexact or crashing before (``G5/runs/u1_projection/``): the skew
and shift vehicles (pins, delays), the fuzz seeds of the leak (191, 23, 144, 192), the mixed-input and windowed-crash
seeds, the multi-source in-pass seed 71, the order-effect seeds 28 and 61, the identity-own crash seed 4, and the
segment-id seed 2."""

from __future__ import annotations

import pickle

import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing import hybrid_build_pool, hybrid_build_scheduled
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import PassTimingError
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage

from . import runs as R
from . import vehicles as V


def _fuzz(seed: int):
    return V.ir_of(V.random_rep(seed))


class TestTimingMovingCutsRunExact:
    """R1 and R2: the cuts the round-2 predicate refused (skew3 C 2-3, shift_probe and shift_streamed C 3-6,
    late_skip C 2-3, deep_skip C 2) and the inexact ones it admitted build as the capacity split cuts them and run
    equal to the identity run under all four deployed runs."""

    @pytest.mark.parametrize("semantics", R.ALL_RUNS, ids=R.RUN_IDS)
    @pytest.mark.parametrize("kind,count", [("skew3", 2), ("skew3", 3), ("shift_probe", 3), ("shift_probe", 5),
                                            ("shift_streamed", 3), ("shift_streamed", 5), ("late_skip", 2),
                                            ("deep_skip", 2)])
    def test_the_cut_builds_and_runs_equal_to_the_identity_run(self, kind, count, semantics):
        ir = V.ir_of(V.branchy_repr(kind))
        u1 = V.build(ir, count, None, retimed_level_stages=R.retimed(semantics))
        assert R.multi_pass(u1, ir) and carried_outputs_by_stage(u1.stages)
        R.assert_runs_equal_to_the_identity_run(ir, count, semantics, mapping=u1)

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.CASCADED_TTFS], ids=["streamed_lif", "cascaded_ttfs"])
    @pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
    def test_a_consumer_of_two_in_pass_producers_runs_equal(self, count, semantics):
        """Fuzz seed 71: only the second producer's relative latency moves when the pass runs alone."""
        R.assert_runs_equal_to_the_identity_run(_fuzz(71), count, semantics)


class TestTheCarryStoreIsScopedToItsSegment:
    """R3: a producer carried within one schedule segment and also read by a LATER segment reaches that segment
    through its count, as the fused program delivers it; the run-wide train buffer replayed its raster (seed 191 C 2:
    12 of 72 outputs differed under streamed LIF with the predicate reading 0)."""

    @pytest.mark.parametrize("seed,count", [(191, 2), (23, 2), (23, 3), (144, 2), (192, 2)])
    def test_the_leak_witnesses_run_equal_under_streamed_lif(self, seed, count):
        R.assert_runs_equal_to_the_identity_run(_fuzz(seed), count, R.STREAMED_LIF)


class TestEachInputSliceIsEncodedByItsKind:
    """R4: a pass that replays a carried raster and reads a host-side input encodes each slice by its own kind (the
    raster replayed, the host value encoded as the fused stage encodes it); it crashed at run time before."""

    def test_a_mixed_input_pass_runs_equal_under_streamed_lif(self):
        R.assert_runs_equal_to_the_identity_run(V.ir_of(V.branchy_repr("edge_class")), 12, R.STREAMED_LIF)

    @pytest.mark.parametrize("seed,count", [(6, 2), (44, 2), (46, 3), (67, 2), (192, 3), (144, 3)])
    def test_the_fuzz_crash_witnesses_run_equal_under_streamed_lif(self, seed, count):
        R.assert_runs_equal_to_the_identity_run(_fuzz(seed), count, R.STREAMED_LIF)

    def test_a_fused_stage_reading_a_lif_host_train_and_an_uncached_input_runs(self):
        """The identity-own crash (seed 4 C 2): the identity run itself raised on a fused stage that reads a
        LIF-perceptron host train and an uncached host value; each slice is now encoded by its kind, as the chip
        regenerates every slice from its value."""
        R.assert_runs_equal_to_the_identity_run(_fuzz(4), 2, R.STREAMED_LIF)


class TestLevelUnits:
    """R5: under per-hop re-timing every pass of a multi-level fused segment executes the fused levels it holds as
    count re-encoded level units, a single-level pass included; it streamed before and mixed cached host trains with
    an unpublished carried producer (seed 23 C 2: 'cached spike trains for some inputs')."""

    @pytest.mark.parametrize("seed,count", [(23, 2), (23, 3), (144, 2)])
    def test_the_windowed_crash_witnesses_run_equal(self, seed, count):
        R.assert_runs_equal_to_the_identity_run(_fuzz(seed), count, R.WINDOWED_LIF)

    def test_every_pass_of_a_multi_level_segment_holds_one_unit_per_fused_level(self):
        ir = V.ir_of(V.chain_repr(seed=0))
        u1 = V.build(ir, 2, None, retimed_level_stages=True)
        identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
        level_of = {int(p["ir_node_id"]): depth
                    for stage in V.neural_passes(identity) for depth, level in enumerate(stage.retimed_level_stages)
                    for placements in level.hard_core_mapping.soft_core_placements_per_hard_core for p in placements}
        assert len(V.neural_passes(u1)) > 1
        for stage in V.neural_passes(u1):
            assert stage.retimed_level_stages
            for unit in stage.retimed_level_stages:
                held = {level_of[int(p["ir_node_id"])]
                        for placements in unit.hard_core_mapping.soft_core_placements_per_hard_core for p in placements}
                assert len(held) == 1


class TestDependencyOrderedWalks:
    """R6: the analytical TTFS walk orders cores by dependency, so a run after any ChipLatency run of the same
    mapping (which raises a producer above a shallower consumer, ALAP) evaluates no consumer before its producer
    (seeds 28 and 61 C 4); without the IR latency pass a packed segment no longer runs in packing order."""

    @pytest.mark.parametrize("seed", [28, 61])
    def test_analytical_ttfs_after_a_streamed_run_of_the_same_mapping(self, seed):
        ir = _fuzz(seed)
        u1 = V.build(ir, 4, None)
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            R.flow(u1, **R.STREAMED_LIF)(x)                                 # ChipLatency writes ALAP latencies
        R.assert_runs_equal_to_the_identity_run(ir, 4, R.ANALYTICAL_TTFS, mapping=u1)

    def test_without_the_ir_latency_pass_a_packed_segment_runs_equal_to_the_identity_run(self):
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS,
                       max_neurons=V.MAX_NEURONS).map(V.branchy_repr("skew3"))
        R.assert_runs_equal_to_the_identity_run(ir, 5, R.ANALYTICAL_TTFS, mapping=V.build(ir, 5, None))


class TestPositionDependentEncodes:
    """R4 with spike_phase_dither: an external slice is encoded at the channel the fused stage (under re-timing, the
    identity's level stage) encodes it at; the dither comb rotates by the stage-input channel index
    (``split_level`` C 1: 3 of 72 outputs differed under streamed LIF with the slice at its own offset; seeds 5, 16
    and 152: windowed LIF with dither differed, the level units reading their slices at their own offsets)."""

    def test_split_level_with_phase_dither_under_streamed_lif(self):
        R.assert_runs_equal_to_the_identity_run(V.ir_of(V.split_level_repr()), 1, R.STREAMED_LIF, phase_dither=True)

    @pytest.mark.parametrize("seed,count", [(5, 2), (16, 3), (152, 2)])
    def test_fuzz_witnesses_with_phase_dither_under_windowed_lif(self, seed, count):
        R.assert_runs_equal_to_the_identity_run(_fuzz(seed), count, R.WINDOWED_LIF, phase_dither=True)

    @pytest.mark.parametrize("make,count,semantics", [(lambda: V.ir_of(V.split_level_repr()), 1, R.STREAMED_LIF),
                                                      (lambda: _fuzz(16), 3, R.WINDOWED_LIF)],
                             ids=["split_level", "seed16"])
    def test_a_mapping_whose_flow_ran_keeps_its_positions_through_a_pickle_round_trip(self, make, count, semantics):
        """The fused positions are memoized on the mapping by stage position, not identity: a mapping pickled after
        its flow ran (split_level C 1: 5 outputs differed with identity keys; seed 16 C 3: 3) runs equal."""
        ir = make()
        u1 = V.build(ir, count, None, retimed_level_stages=R.retimed(semantics))
        with torch.no_grad():
            R.flow(u1, **semantics, phase_dither=True)(torch.rand(6, V.IN, generator=torch.Generator().manual_seed(9)))
        R.assert_runs_equal_to_the_identity_run(ir, count, semantics, mapping=pickle.loads(pickle.dumps(u1)),
                                                phase_dither=True)


class TestOneSegmentationForThePlanner:
    """W9 (review round 2): the builder keeps the base's plan, its capacity split cutting each IR-run segment at the
    specs' depth segments, and every plane plans those passes given the walk's runs (``test_plane_segmentation_parity``
    holds the stats plane and the key-absent sweep). Fuzz seed 2 holds one five-core IR-run segment with depth
    segments 0, 1 and 2: it runs as three passes, each cut exact (the W9 control of round 1: 2,494 multi-pass programs
    at 600 x 6, every one equal to its identity run)."""

    @pytest.mark.parametrize("semantics", R.ALL_RUNS, ids=R.RUN_IDS)
    def test_the_depth_segment_cuts_of_one_ir_run_segment_run_exact(self, semantics):
        ir = _fuzz(2)
        u1 = V.build(ir, 5, None, retimed_level_stages=R.retimed(semantics))
        assert len(V.neural_passes(u1)) == 3 and len(V.neural_passes(build_identity_hybrid_mapping(ir_graph=ir))) == 1
        R.assert_runs_equal_to_the_identity_run(ir, 5, semantics, mapping=u1)

    @pytest.mark.parametrize("kind", ["moved_to_output", "edge_class", "intra_to_cross", "cores_only",
                                      "deep_mismatch", "shift_streamed"])
    @pytest.mark.parametrize("count", [2, 3, 5, 8])
    def test_the_census_plane_plans_the_builders_passes(self, kind, count):
        """With the walk's program graph the census plane groups the specs by the builder's IR-run segments."""
        rep = V.branchy_repr(kind)
        specs, census, program = V.layout_plane(rep)
        ir = V.ir_of(rep)
        assert V.fragment_pass_layers(V.fragments(specs, census, count, program=program), specs) == \
            V.built_pass_layers(V.build(ir, count, None), ir)


class TestAPlannedPassThatDoesNotPack:
    """Where the planner's residency classes (a unique fallback per spec) and the packer's (the source perceptron's)
    differ, a planned pass may not pack: fuzz seed 38 C 2 and 3, among the base's 29 fuzz build failures at 600 x 6
    (an unnamed RuntimeError there), refuse by name. Only the packer's capacity failure is renamed: any other error
    of the flush propagates as it is raised (review round 2)."""

    @pytest.mark.parametrize("count", [2, 3])
    def test_it_refuses_by_name(self, count):
        with pytest.raises(hybrid_build_scheduled.PlannedPassPackingRefusal, match="does not pack") as caught:
            V.build(_fuzz(38), count, None)
        assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, RuntimeError)
        assert "No more hard cores available" in str(caught.value)

    @pytest.mark.parametrize("error", [RuntimeError("an unrelated defect"), NotImplementedError("unwired"),
                                       RecursionError("deep")], ids=["runtime", "not_implemented", "recursion"])
    def test_any_other_error_of_the_flush_propagates_unwrapped(self, monkeypatch, error):
        def flush(**kwargs):
            raise error
        monkeypatch.setattr(hybrid_build_scheduled, "_flush_scheduled_segment", flush)
        with pytest.raises(type(error)) as caught:
            V.build(_fuzz(0), 3, None)
        assert caught.value is error

    def test_an_error_the_packer_wraps_propagates_unwrapped(self, monkeypatch):
        """``hybrid_segment`` re-raises every RuntimeError of the packer as "Hard-core packing failed"; a recursion
        error inside the packer keeps that wrapper and is not a capacity failure."""
        from mimarsinan.mapping.packing.softcore import HardCoreMapping

        def deep(self, *args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")
        monkeypatch.setattr(HardCoreMapping, "map", deep)
        with pytest.raises(RuntimeError, match="Hard-core packing failed") as caught:
            V.build(_fuzz(0), 3, None)
        assert not isinstance(caught.value, hybrid_build_scheduled.PlannedPassPackingRefusal)
        assert isinstance(caught.value.__cause__, RecursionError)


def _lenient_levels(original):
    def levels(label, cores, chunks):
        try:
            return original(label, cores, chunks)
        except PassTimingError:
            return None
    return levels


class TestTheBuildRefusalClasses:
    """Two program classes the random-DAG family cannot produce refuse by name at build, each load-bearing: with the
    refusal removed the run differs from the identity run (the review's ``rv_coalesce2`` and ``probe_copack``)."""

    COALESCE = dict(allow_coalescing=True, allow_neuron_splitting=True)

    def test_a_streaming_windowed_segment_cut_with_a_carried_wire_refuses(self):
        """Coalescing family seed 3 C 2 (fan-in up to 40 on 25-axon cores): psum groups keep the fused segment whole
        (one level), so its passes cannot run as level units, and a count transfer cannot carry its raster."""
        ir = V.ir_of(V.wide_rep(3), coalescing=True)
        with pytest.raises(PassTimingError, match="streams under per-hop re-timing"):
            V.build(ir, 2, None, retimed_level_stages=True, **self.COALESCE)

    def test_without_that_refusal_the_windowed_run_differs(self, monkeypatch):
        ir = V.ir_of(V.wide_rep(3), coalescing=True)
        monkeypatch.setattr(hybrid_build_scheduled, "_pass_levels", _lenient_levels(hybrid_build_scheduled._pass_levels))
        u1 = V.build(ir, 2, None, retimed_level_stages=True, **self.COALESCE)
        identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            assert not torch.equal(R.flow(u1, **R.WINDOWED_LIF)(x), R.flow(identity, **R.WINDOWED_LIF)(x))

    @pytest.mark.parametrize("count,scheduled", [(2, True), (8, True), (8, False)],
                             ids=["multi_pass", "single_pass", "single_pool"])
    def test_co_packed_instances_the_fused_segment_runs_at_two_latencies_refuse(self, count, scheduled):
        """A token-wise FC whose token 0 feeds a three-layer chain and token 1 the output layer: tokens 0 and 1 share
        an IR depth tier and a bank, so the packer co-packs them, and ChipLatency would start the shallow token at the
        deep reader's latency minus one; the scheduled build (multi- and single-pass) and the single pool refuse."""
        ir = V.ir_of(V.token_repr())
        build = V.build if scheduled else _single_pool
        with pytest.raises(PassTimingError, match="one hard core starts at one latency"):
            build(ir, count, None)

    def test_a_producer_co_packed_with_its_consumer_refuses(self):
        """An IR whose latency tags let the packer co-pack a producer with its consumer (both tagged 0, not as the IR
        latency pass tags them): the hard core would read its own output a cycle late (``test_softcore_traceability``'s
        two-core IR, which builds with the latency pass's tags now)."""
        import numpy as np
        from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore

        first = NeuralCore(id=0, name="h", core_matrix=np.full((5, 4), 0.1, dtype=np.float32), latency=0,
                           input_sources=np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)]),
                           perceptron_index=0)
        second = NeuralCore(id=1, name="o", core_matrix=np.full((5, 2), 0.1, dtype=np.float32), latency=0,
                            input_sources=np.array([IRSource(0, i) for i in range(4)] + [IRSource(-3, 0)]),
                            perceptron_index=0)
        ir = IRGraph(nodes=[first, second], output_sources=np.array([IRSource(1, 0), IRSource(1, 1)]))
        with pytest.raises(PassTimingError, match="one hard core starts at one latency"):
            _single_pool(ir, 5, None)
        second.latency = 1
        assert len(_single_pool(ir, 5, None).stages) == 1

    def test_without_that_refusal_the_single_pool_differs(self, monkeypatch):
        """The base's silent difference: the default single pool of the token vehicle at C 8, run unprojected."""
        ir = V.ir_of(V.token_repr())
        monkeypatch.setattr(hybrid_build_pool, "project_passes", lambda stages: None)
        packed = _single_pool(ir, 8, None)
        identity = build_identity_hybrid_mapping(ir_graph=ir)
        x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0))
        for semantics in (R.STREAMED_LIF, R.CASCADED_TTFS):
            with torch.no_grad():
                assert not torch.equal(R.flow(packed, **semantics)(x), R.flow(identity, **semantics)(x))


def _single_pool(ir, count, residency_plan):
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy

    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=V.cores(count),
                                          strategy=MappingStrategy.resolve(ChipCapabilities()))
