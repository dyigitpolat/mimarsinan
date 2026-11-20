"""The pass projection's mechanism (plan §21 item 1): the latency SSOT, the build-time stamps, the replay SSOT, the
per-slice encode and the plan-time refusals by name. Behaviour against the identity run is ``test_pass_projection``."""

from __future__ import annotations

import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.simulation_runner import emit as nevresim_emit
from mimarsinan.chip_simulation.simulation_runner.carry import assemble_input_train
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES, ChipLatency
from mimarsinan.mapping.latency.core_graph import alap_latencies, longest_paths, topological_order
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.noc import execution_stage_latencies
from mimarsinan.mapping.noc.fragments import delayed_carry_wires, walk_runs
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import (
    REPLAY_DELAYS, PassTimingError, _core_latencies, _delay, carried_nodes, late_stage_input_readers,
)
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid import carry as C

from unit.chip_simulation import nevresim_trees as NT

from . import runs as R
from . import vehicles as V

KINDS = ["skew3", "skip", "late_skip", "deep_skip", "shift_probe", "shift_streamed", "edge_class", "moved_to_output",
         "wide_branches", "cores_only", "deep_mismatch"]


class _StandInSource:
    def __init__(self, core: int, is_input: bool = False) -> None:
        self.core_, self.neuron_, self.is_input_, self.is_off_, self.is_always_on_ = core, 0, is_input, False, False


class _StandInCore:
    def __init__(self, sources) -> None:
        self.axon_sources = [_StandInSource(-2, True)] + [_StandInSource(s) for s in sources]
        self.latency = None

    def get_core_matrix(self):
        return np.ones((len(self.axon_sources), 1))


class _StandInStage:
    """One-neuron cores reading one stage input and their sources, every weight live (ChipLatency's facts)."""

    def __init__(self, sources) -> None:
        self.cores = [_StandInCore(s) for s in sources]
        self.output_sources = [_StandInSource(i) for i in range(len(sources))]


def _identity_sources(stage):
    cores = stage.hard_core_mapping.cores
    return [sorted({int(s.core_) for s in core.axon_sources
                    if not (s.is_off_ or s.is_always_on_ or s.is_input_)}) for core in cores]


class TestTheLatencySsot:
    """``alap_latencies`` is ChipLatency's fixed point in O(V + E): equal per core on the identity stages of every
    vehicle and on U1's packed passes (the quadratic stand-in it replaces: 13.7 s at 8000 cores)."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_it_equals_chip_latency_on_the_identity_and_the_packed_passes(self, kind):
        ir = V.ir_of(V.branchy_repr(kind))
        for mapping in (build_identity_hybrid_mapping(ir_graph=ir), V.build(ir, 3, None)):
            for stage in V.neural_passes(mapping):
                want = _identity_sources(stage)
                if getattr(stage.hard_core_mapping, PINNED_LATENCIES, None) is not None:
                    continue
                ChipLatency(stage.hard_core_mapping).calculate()
                assert alap_latencies(want) == [int(c.latency or 0) for c in stage.hard_core_mapping.cores]

    def test_a_deep_residual_chain_equals_chip_latency_and_is_linear(self):
        def residual(n):
            return [[] if i < 2 else [i - 1, i - 2] for i in range(n)]

        stand_in = _StandInStage(residual(300))
        ChipLatency(stand_in).calculate()
        assert alap_latencies(residual(300)) == [c.latency for c in stand_in.cores]
        start = time.perf_counter()
        latency = alap_latencies(residual(50000))
        assert time.perf_counter() - start < 5.0
        assert latency[-1] == longest_paths(residual(50000))[-1] == 50000 - 2

    def test_a_producer_is_raised_to_its_deepest_consumer_minus_one(self):
        # 0 -> 1 -> 2 and 0 -> 2: 0 raised to 1 by its deepest consumer; a shallow sink is never raised
        assert alap_latencies([[], [0], [0, 1]]) == [1, 1, 2]
        assert alap_latencies([[], [0], [1], [0]]) == [0, 1, 2, 1]

    def test_a_cycle_refuses(self):
        with pytest.raises(ValueError, match="cycle"):
            topological_order([[1], [0]])


class TestPins:
    """R1: ChipLatency honours a pass's pinned latencies (one SSOT for every backend); a pass is pinned only where its
    own timing moves an edge, so a chain carries no pin."""

    def test_chip_latency_returns_the_pins(self):
        ir = V.ir_of(V.branchy_repr("skew3"))
        pinned = [s for s in V.neural_passes(V.build(ir, 2, None))
                  if getattr(s.hard_core_mapping, PINNED_LATENCIES, None) is not None]
        assert pinned
        hcm = pinned[0].hard_core_mapping
        pins = getattr(hcm, PINNED_LATENCIES)
        assert ChipLatency(hcm).calculate() == max(pins) + 1
        assert [c.latency for c in hcm.cores] == pins and min(pins) == 0

    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_a_chain_carries_no_pin_and_no_delay(self, count):
        u1 = V.build(V.ir_of(V.chain_repr(seed=0)), count, None)
        assert len(V.neural_passes(u1)) > 1
        for stage in V.neural_passes(u1):
            assert getattr(stage.hard_core_mapping, PINNED_LATENCIES, None) is None
            assert getattr(stage, REPLAY_DELAYS, None) is None

    def test_co_packed_soft_cores_at_two_fused_latencies_refuse_by_name(self):
        class _Placed:
            cores = [object()]

            @staticmethod
            def nodes(h):
                return {1, 2}

        with pytest.raises(PassTimingError, match="one hard core starts at one latency") as caught:
            _core_latencies(_Placed(), {1: 0, 2: 1}, "pass")
        assert isinstance(caught.value, ToolchainRefusal)

    def test_a_positive_delivery_delay_refuses_by_name(self):
        """A reader at fused latency 3 of a carried producer at 0 would read it 2 cycles early (d = 3 - 1 - 0); under
        ALAP no delay is positive, and a hold-last replay is not designed, so the build refuses."""
        source = SimpleNamespace(is_input_=True, is_off_=False, is_always_on_=False, neuron_=0)

        class _Placed:
            cores = [SimpleNamespace(axon_sources=[source])]

            @staticmethod
            def nodes(h):
                return {9}

        stage = SimpleNamespace(name="pass", input_map=[SimpleNamespace(node_id=7, offset=0, size=1)])
        with pytest.raises(PassTimingError, match="cycles early") as caught:
            _delay(stage, _Placed(), {9: 3, 7: 0}, {7})
        assert isinstance(caught.value, ToolchainRefusal)
        _delay(stage, _Placed(), {9: 1, 7: 0}, {7})          # d = 0: no slice, no delay
        assert getattr(stage, REPLAY_DELAYS, None) is None


class TestDeliveryDelays:
    """R2: one input slice per (carried producer, delay), every delay negative (ALAP), the readers repointed."""

    def test_shift_probe_reads_its_shifted_producer_from_its_own_slice(self):
        u1 = V.build(V.ir_of(V.branchy_repr("shift_probe")), 3, None)
        delayed = [(stage, k, d) for stage in V.neural_passes(u1)
                   for k, d in (getattr(stage, REPLAY_DELAYS, None) or {}).items()]
        assert delayed and all(d < 0 for _, _, d in delayed)
        for stage, _k, _d in delayed:
            nodes = [int(s.node_id) for s in stage.input_map]
            delays = getattr(stage, REPLAY_DELAYS)
            keys = [(int(s.node_id), delays.get(k, 0)) for k, s in enumerate(stage.input_map)]
            assert len(keys) == len(set(keys)) and len(nodes) >= len(set(nodes))

    @pytest.mark.parametrize("make,count", [(lambda: V.branchy_repr("late_skip"), 2), (lambda: V.random_rep(0), 3)])
    def test_the_census_plane_counts_the_builders_extra_slices(self, make, count):
        """late_skip C 2 adds one delayed slice and fuzz seed 0 C 3 two, on both planes."""
        rep = make()
        specs, census, program = V.layout_plane(rep)
        u1 = V.build(V.ir_of(rep), count, None)
        built = sum(len(s.input_map) - len({int(x.node_id) for x in s.input_map}) for s in V.neural_passes(u1))
        assert built > 0
        frag = V.fragments(specs, census, count, program=program)
        assert len(delayed_carry_wires(specs, frag, walk_runs(program, len(specs)))) == built

    def test_the_boundary_in_term_counts_late_skips_delayed_slice_at_its_producers_width(self):
        """[W10] The census plane's extra boundary-in slice of late_skip C 2 is the builder's: one more slice of the
        producer B (8 wide) into the pass that reads it twice (the integration edit counts it in ``pass_carry``)."""
        rep = V.branchy_repr("late_skip")
        specs, census, program = V.layout_plane(rep)
        frag = V.fragments(specs, census, 2, program=program)
        u1 = V.build(V.ir_of(rep), 2, None)
        built = [int(s.input_map[k].size) for s in V.neural_passes(u1) for k in (getattr(s, REPLAY_DELAYS, None) or {})]
        assert [width for _, width, _ in delayed_carry_wires(specs, frag, walk_runs(program, len(specs)))] == built == [8]


class TestTheReplaySsot:
    def test_a_negative_delay_shifts_late_and_drops_the_tail(self):
        raster = np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2)          # (N, T, size)
        out = C.replay_raster(raster, -1, time_axis=1)
        assert out[:, 0].tolist() == [[0, 0], [0, 0]]
        assert out[:, 1:].tolist() == raster[:, :2].tolist()
        torch_raster = torch.arange(6.0).reshape(3, 1, 2)                    # (T, B, size)
        assert C.replay_raster(torch_raster, -2, time_axis=0)[2].tolist() == [[0.0, 1.0]]
        assert C.replay_raster(raster, 0, time_axis=1) is raster

    def test_a_positive_delay_refuses_by_name(self):
        with pytest.raises(C.PassReplayRefusal):
            C.replay_raster(np.zeros((3, 1)), 1, time_axis=0)

    def test_the_store_is_scoped_by_schedule_segment(self):
        class _Stage:
            def __init__(self, segment):
                self.schedule_segment_index = segment

        store: dict = {}
        C.carry_scope(store, _Stage(0), publish=True)[7] = "raster"
        assert C.carry_scope(store, _Stage(0)) == {7: "raster"} and C.carry_scope(store, _Stage(1)) == {}

    def test_a_re_encoded_delayed_slice_shifts_and_a_published_one_replays(self):
        """``replay_input_slices``, every backend's twin of the HCM's ``replay_carried_slices``: a slice with a raster
        in the scope replays it at its delay, a delayed slice without one (COLLAPSE) shifts its re-encode."""
        class _Slice:
            def __init__(self, node, offset):
                self.node_id, self.offset, self.size = node, offset, 1

        stage = SimpleNamespace(input_map=[_Slice(7, 0), _Slice(8, 1)], replay_delays={0: -1, 1: -2})
        encoded = np.array([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=np.float32)       # (N, size, T)
        C.replay_input_slices(encoded, stage, {7: np.array([[0], [1], [0], [1]], dtype=np.float32)}, 4)
        assert encoded[0].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]

    def test_rows_move_late_into_a_cores_local_time(self):
        window = np.arange(2 * 1 * 6, dtype=np.float32).reshape(2, 1, 6)
        window[:, :, 3:] = 0
        moved = C.shift_rows_to_core_time(window, [(1, 2)], 2, 3)
        assert moved[0].tolist() == window[0].tolist() and moved[1, 0].tolist() == [0, 0, 6, 7, 8, 0]
        assert C.shift_rows_to_core_time(window, [(1, 2)], 0, 3) is window

    def test_backends_replay_a_delayed_slice_late(self):
        class _Slice:
            def __init__(self, node, offset, size):
                self.node_id, self.offset, self.size = node, offset, size

        class _Stage:
            input_map = [_Slice(7, 0, 1)]
            replay_delays = {0: -1}

        encoded = np.zeros((1, 1, 3), dtype=np.float32)
        assert C.apply_carried_input(encoded, _Stage(), {7: np.array([[1], [0], [1]], dtype=np.float32)})
        assert encoded[0, 0].tolist() == [0, 1, 0]
        encoded[0, 0] = [1, 0, 0]
        C.shift_delayed_slices(encoded, _Stage(), 3)
        assert encoded[0, 0].tolist() == [0, 1, 0]


def _streamed(**enabled):
    return {"spiking_family": "lif", "spiking_variant": "streamed", "spike_generation_mode": "Uniform", **enabled}


class TestTheChipClock:
    """[U1 W6, review rounds 1-2, W33] Every backend reads a stage-input row in the reading core's local time: SANA-FE
    and Lava shift it host-side, nevresim's generated simulator reads its input train at each core's local cycle. A
    backend not declared core-local refuses by name, at plan time and before compiling, a program with a stage-input
    reader above latency 0 in the units it executes (a pass, or its level units under per-hop re-timing)."""

    def test_a_carried_slice_read_above_latency_zero_is_found(self):
        u1 = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        late = {k: late_stage_input_readers(s, carried_nodes(u1.stages, s)) for k, s in enumerate(u1.stages)
                if s.kind == "neural"}
        assert late[4] == [(0, 1)] and sum(map(len, late.values())) == 1
        skew = V.build(V.ir_of(V.branchy_repr("skew3")), 3, None)
        assert not any(late_stage_input_readers(s, carried_nodes(skew.stages, s)) for s in V.neural_passes(skew))

    def test_every_declared_backend_admits_a_late_reader_at_plan_time(self):
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        assert C.run_pass_transfer(_streamed(enable_nevresim_simulation=True, enable_sanafe_simulation=True,
                                             enable_loihi_simulation=True), hybrid_mapping=late) == VERBATIM
        for backend in ("nevresim", "sanafe"):
            C.require_backend_replay(late, backend, transfer=COLLAPSE, spike_mode="TTFS",
                                     spiking_mode="ttfs_cycle_based")

    def test_late_readers_are_judged_on_the_units_a_backend_executes(self, monkeypatch):
        """[Review round 2] Under per-hop re-timing (windowed lif, COLLAPSE: the deployed lifsync) a backend executes a
        pass's level units, whose cores all start at latency 0: chain C 3's pass 4 reads its carried slice at latency 1
        as a pass, at 0 in its units; the same cut run streamed, unit-less, reads it at 1. An undeclared backend is
        refused only the second."""
        monkeypatch.setitem(C._BACKEND_ENABLE_KEYS, "future_backend", "enable_future_backend")
        retimed = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None, retimed_level_stages=True)
        reader = retimed.stages[4]
        assert late_stage_input_readers(reader, carried_nodes(retimed.stages, reader)) == [(0, 1)]
        assert reader.retimed_level_stages and C.executed_late_input_readers(reader) == []
        C.require_backend_replay(retimed, "future_backend", transfer=COLLAPSE, spike_mode="Uniform", spiking_mode="lif")
        streamed = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        assert (0, 1) in C.executed_late_input_readers(streamed.stages[4])
        with pytest.raises(C.ChipClockReplayRefusal):
            C.require_backend_replay(streamed, "future_backend", transfer=COLLAPSE, spike_mode="TTFS",
                                     spiking_mode="ttfs_cycle_based")

    def test_an_undeclared_clock_refuses_before_nevresim_compiles(self, monkeypatch, tmp_path):
        """The refusal is wired into nevresim's emit: on a nevresim tree without W33's core-local read (base 90af7d2,
        ``nevresim_trees.base_tree``), a late reader refuses before any segment compiles."""
        monkeypatch.setattr(NevresimDriver, "nevresim_path", str(NT.base_tree(tmp_path)))
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        runner = SimpleNamespace(pass_transfer=VERBATIM, spike_generation_mode="Uniform", spiking_mode="lif")
        with pytest.raises(C.ChipClockReplayRefusal):
            nevresim_emit.prepare_all_segments(runner, late)       # refuses before it reads test data or compiles

    def test_nevresim_assembles_a_delayed_slice_late_and_a_late_readers_slice_verbatim(self):
        """Fuzz seed 58 C 2: a carried slice delayed by 1 cycle, read at latency 0; chain C 3: read at latency 1, in
        stage time (the generated simulator moves it into the reader's local time)."""
        u1 = V.build(V.ir_of(V.random_rep(58)), 2, None)
        stage = next(s for s in V.neural_passes(u1) if getattr(s, REPLAY_DELAYS, None))
        (k, delay), = getattr(stage, REPLAY_DELAYS).items()
        node, width = int(stage.input_map[k].node_id), int(stage.input_map[k].size)
        raster = (np.arange(2 * V.T * width).reshape(2, V.T, width) % 3 == 0).astype(np.float64)
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        train = assemble_input_train(stage, np.zeros((2, size)), {stage.schedule_segment_index: {node: raster}},
                                     spike_generation_mode="Uniform", T=V.T)
        offset = int(stage.input_map[k].offset)
        assert delay < 0 and np.array_equal(train[:, :, offset:offset + width],
                                            C.replay_raster(raster, delay, time_axis=1))
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        reader = late.stages[4]
        nodes = carried_nodes(late.stages, reader)
        store = {reader.schedule_segment_index: {n: np.ones((1, V.T, 64)) for n in nodes}}
        train = assemble_input_train(reader, np.zeros((1, 64)), store, spike_generation_mode="Uniform", T=V.T)
        carried = [(int(s.offset), int(s.size)) for s in reader.input_map if int(s.node_id) in nodes]
        assert carried and all(train[:, :, lo:lo + n].min() == 1 for lo, n in carried)

    def test_sanafe_presents_every_stage_input_row_in_its_readers_local_time(self):
        """[Review round 2] A carried slice (node 7) and a host-encoded one (node 3) alike: the input neuron feeding a
        core that starts at latency 2 replays its row 2 cycles late; a core at latency 0 keeps the encode."""
        from mimarsinan.chip_simulation.sanafe.runner.carry import present_inputs_in_core_time

        class _Neuron:
            def __init__(self):
                self.attrs = None

            def set_attributes(self, model_attributes):
                self.attrs = model_attributes

        src = lambda k: SimpleNamespace(neuron_=k)
        hcm = SimpleNamespace(cores=[SimpleNamespace(latency=0, axon_sources=[src(0)]),
                                     SimpleNamespace(latency=2, axon_sources=[src(0), src(1)])])
        neurons = {(0, 0): _Neuron(), (1, 0): _Neuron(), (1, 1): _Neuron()}
        encoded = np.array([[[1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]], dtype=np.float32)
        present_inputs_in_core_time(neurons, hcm, encoded)
        assert neurons[(0, 0)].attrs is None                                         # on time
        assert neurons[(1, 0)].attrs["spikes"] == [0, 0, 1, 0, 1, 0]                 # the carried row
        assert neurons[(1, 1)].attrs["spikes"] == [0, 0, 1, 1, 0, 0]                 # the host-encoded row


class _Stop(Exception):
    pass


def _sanafe_ready() -> None:
    """Skip where SANA-FE or the mimarsinan plugins it loads are not built (``scripts/bootstrap_sanafe.sh``)."""
    pytest.importorskip("sanafe")
    from mimarsinan.chip_simulation.sanafe.arch_synth import spec

    try:
        spec._resolve_plugins("loihi")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


class TestTheChipPaths:
    """Each chip path applies the projection where it runs: SANA-FE's cascaded TTFS branch shifts the delayed slices,
    Lava reads the carry store of the stage's own schedule segment and gives a late reader its local-time window."""

    def test_sanafe_cascaded_ttfs_shifts_the_delayed_slices_of_the_stage_it_runs(self, monkeypatch):
        _sanafe_ready()
        from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
        from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
        from mimarsinan.chip_simulation.sanafe.runner import neural_stage

        u1 = V.build(V.ir_of(V.branchy_repr("shift_probe")), 3, None)
        delayed = {id(s) for s in V.neural_passes(u1) if getattr(s, REPLAY_DELAYS, None)}
        shifted = []
        real = neural_stage.shift_delayed_slices

        def spy(encoded, stage, T, skip=()):
            before = encoded.copy()
            real(encoded, stage, T, skip)
            shifted.append((id(stage), before, encoded.copy(), stage))

        monkeypatch.setattr(neural_stage, "shift_delayed_slices", spy)
        behavior = NeuralBehaviorConfig(spiking_mode="ttfs_cycle_based", firing_mode="TTFS", thresholding_mode="<=",
                                        spike_generation_mode="TTFS")
        runner = SanafeRunner(mapping=u1, simulation_length=V.T, behavior=behavior, ttfs_cycle_schedule="cascaded",
                              pass_transfer=COLLAPSE)
        runner.run(np.random.default_rng(0).random((1, V.IN)), sample_index=0)
        runs = [(before, after, stage) for key, before, after, stage in shifted if key in delayed]
        assert runs
        for before, after, stage in runs:
            for k, d in getattr(stage, REPLAY_DELAYS).items():
                lo, hi = int(stage.input_map[k].offset), int(stage.input_map[k].offset) + int(stage.input_map[k].size)
                assert np.array_equal(after[:, lo:hi, :V.T], C.replay_raster(before[:, lo:hi, :V.T], d, time_axis=2))

    @pytest.mark.parametrize("kind,count", [("chain", 3), ("late_skip", 2), ("seed58", 2), ("skew3", 2),
                                            ("skew3", None), ("shift_probe", None), ("shift_probe", 3)])
    def test_sanafe_records_equal_the_hcm_records(self, kind, count):
        """The pipeline's SANA-FE parity gate (``compare_records`` on the HCM and SANA-FE records): chain C 3 (pass 4's
        core 0 starts at latency 1 and reads the carried slice), late_skip C 2 (a carried slice read late and a
        delayed one) and seed 58 C 2 (a delayed slice read at latency 0): every record field equal, segment and core
        inputs included (the cores read the slice in their local time, and the record counts what they read).
        [Review round 2] A host-encoded slice read by an ALAP-raised core too: skew3 C 2 and the identity programs
        (``count`` None) of skew3 and shift_probe, which differed from the HCM on the base."""
        _sanafe_ready()
        from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
        from mimarsinan.chip_simulation.recording.compare import compare_records
        from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

        rep = V.chain_repr(seed=0) if kind == "chain" else V.random_rep(58) if kind == "seed58" else \
            V.branchy_repr(kind)
        ir = R.mapped(rep, R.STREAMED_LIF)  # [U-TGAUGE5] as the mapping builds it: the HCM reference reads the window
        u1 = build_identity_hybrid_mapping(ir_graph=ir) if count is None else V.build(ir, count, None)
        flow = R.flow(u1, **R.STREAMED_LIF, spike_mode="Uniform")
        behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                        spike_generation_mode="Uniform")
        runner = SanafeRunner(mapping=u1, simulation_length=V.T, behavior=behavior, pass_transfer=VERBATIM)
        for b in range(2):
            x = torch.rand(1, V.IN, generator=torch.Generator().manual_seed(b))
            with torch.no_grad():
                _, ref = flow.forward_with_recording(x, sample_index=0)
            got = runner.run(x.numpy().astype(np.float64), sample_index=0).to_hcm_subset()
            assert ref.segments and compare_records(ref, got) == []

    @staticmethod
    def _lava_host(mapping):
        from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
        from mimarsinan.chip_simulation.lava_loihi.segment_runner import LavaSegmentMixin

        class _Host(LavaSegmentMixin):
            T = V.T
            _behavior = NeuralBehaviorConfig(spiking_mode="lif", firing_mode="Default", thresholding_mode="<=",
                                             spike_generation_mode="Uniform")
            _simulation_step_timeout_s = 60.0

        host = _Host()
        host.mapping = mapping
        return host

    def test_lava_replays_the_carry_store_of_the_stages_own_segment(self, monkeypatch):
        from mimarsinan.chip_simulation.lava_loihi import segment_runner

        u1 = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        stage = u1.stages[4]
        store = {stage.schedule_segment_index: {"own": 1}, (stage.schedule_segment_index or 0) + 1: {"other": 2}}
        seen = []

        def spy(encoded, stage, scope, T):
            seen.append(scope)
            raise _Stop

        monkeypatch.setattr(segment_runner, "replay_input_slices", spy)
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        with pytest.raises(_Stop):
            self._lava_host(u1)._run_neural_segment_scheduled(stage.hard_core_mapping, np.zeros((1, size)),
                                                              stage=stage, state_buffer_trains=store)
        assert seen == [store[stage.schedule_segment_index]]

    def test_lava_gives_a_late_reader_its_local_time_window(self, monkeypatch):
        """chain C 3: core 0 of stage 4 starts at latency 1 and reads the carried slice; its active window
        ``[1, 1 + T)`` reads the slice's first T cycles."""
        from mimarsinan.chip_simulation.lava_loihi import segment_runner

        u1 = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        stage = u1.stages[4]
        node = next(iter(carried_nodes(u1.stages, stage)))
        width = next(int(s.size) for s in stage.input_map if int(s.node_id) == node)
        raster = (np.arange(V.T * width).reshape(1, V.T, width) % 2).astype(np.float64)
        windows = {}
        real = segment_runner.assemble_core_active_input

        def spy(core, *, core_idx, latency, seg_input_logical, **kwargs):
            windows[core_idx] = (latency, seg_input_logical)
            return real(core, core_idx=core_idx, latency=latency, seg_input_logical=seg_input_logical, **kwargs)

        def no_lava(wave_tasks, pool=None):           # each core's (weights, threshold, bias, input) -> zero output
            return {i: np.zeros((task[0].shape[0], task[3].shape[1])) for i, task in wave_tasks.items()}

        host = self._lava_host(u1)
        monkeypatch.setattr(segment_runner, "assemble_core_active_input", spy)
        monkeypatch.setattr(host, "_dispatch_wave", no_lava)
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        host._run_neural_segment_scheduled(stage.hard_core_mapping, np.zeros((1, size)), stage=stage,
                                           state_buffer_trains={stage.schedule_segment_index: {node: raster}})
        late = [(i, lat, window) for i, (lat, window) in windows.items() if lat > 0]
        offset = next(int(s.offset) for s in stage.input_map if int(s.node_id) == node)
        assert late and all(np.array_equal(window[offset:offset + width, 0, lat:lat + V.T], raster[0].T)
                            for _, lat, window in late)


    def test_lava_gives_a_late_reader_of_a_host_slice_its_local_time_window(self, monkeypatch):
        """[Review round 2] skew3's identity program raises a reader of the host-encoded stage input above latency 0
        (ALAP); its window ``[L, L + T)`` reads the encode's first T cycles, as the HCM reads a stage input."""
        from mimarsinan.chip_simulation.lava_loihi import segment_runner

        identity = build_identity_hybrid_mapping(ir_graph=V.ir_of(V.branchy_repr("skew3")))
        stage = V.neural_passes(identity)[0]
        cores = stage.hard_core_mapping.cores
        windows = {}
        real = segment_runner.assemble_core_active_input

        def spy(core, *, core_idx, latency, seg_input_logical, **kwargs):
            windows[core_idx] = (latency, seg_input_logical)
            return real(core, core_idx=core_idx, latency=latency, seg_input_logical=seg_input_logical, **kwargs)

        host = self._lava_host(identity)
        monkeypatch.setattr(segment_runner, "assemble_core_active_input", spy)
        monkeypatch.setattr(host, "_dispatch_wave", lambda wave_tasks, pool=None: {
            i: np.zeros((task[0].shape[0], task[3].shape[1])) for i, task in wave_tasks.items()})
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        rates = np.random.default_rng(0).random((1, size))
        encoded = host._behavior.encode_segment_input(rates, V.T)[0]
        host._run_neural_segment_scheduled(stage.hard_core_mapping, rates, stage=stage, state_buffer_trains={})
        late = [c for c, (lat, _) in windows.items()
                if lat > 0 and any(sp.kind == "input" for sp in cores[c].get_axon_source_spans())]
        assert late
        for c in late:
            lat, window = windows[c]
            assert np.array_equal(window[:, 0, lat:lat + V.T], encoded)

    def test_lava_records_the_input_it_replays(self, monkeypatch):
        """Seed 58 C 2: the segment record counts the delayed slice as replayed (its last cycle dropped), as the HCM's
        record does; it counted the plain encode before (3 record differences against the HCM)."""
        from mimarsinan.chip_simulation.lava_loihi import segment_runner
        from mimarsinan.chip_simulation.recording.records import SegmentSpikeRecord

        u1 = V.build(V.ir_of(V.random_rep(58)), 2, None)
        stage = next(s for s in V.neural_passes(u1) if getattr(s, REPLAY_DELAYS, None))
        (k, delay), = getattr(stage, REPLAY_DELAYS).items()
        node, lo = int(stage.input_map[k].node_id), int(stage.input_map[k].offset)
        hi = lo + int(stage.input_map[k].size)
        raster = np.ones((1, V.T, hi - lo))
        host = self._lava_host(u1)
        monkeypatch.setattr(host, "_dispatch_wave", lambda wave_tasks, pool=None: {
            i: np.zeros((task[0].shape[0], task[3].shape[1])) for i, task in wave_tasks.items()})
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        record = SegmentSpikeRecord(stage_index=0, stage_name=stage.name, schedule_segment_index=0,
                                    schedule_pass_index=1, seg_input_rates=np.zeros((1, size)),
                                    seg_input_spike_count=np.zeros(0), seg_output_spike_count=np.zeros(0))
        host._run_neural_segment_scheduled(stage.hard_core_mapping, np.zeros((1, size)), stage=stage,
                                           recorder_seg=record,
                                           state_buffer_trains={stage.schedule_segment_index: {node: raster}})
        assert record.seg_input_spike_count[lo:hi].tolist() == [V.T + delay] * (hi - lo)


class TestPlanTimeRefusals:
    def test_stochastic_spike_generation_on_a_verbatim_multi_pass_run(self):
        u1 = V.build(V.ir_of(V.random_rep(6)), 2, None)
        assert any(C.stochastic_replay_conflicts(u1, s) for s in V.neural_passes(u1))
        with pytest.raises(C.PassReplayRefusal, match="Stochastic") as caught:
            R.flow(u1, spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM,
                   spike_mode="Stochastic")
        assert isinstance(caught.value, ToolchainRefusal)
        R.flow(u1, spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=COLLAPSE, spike_mode="Stochastic")
        single = V.build(V.ir_of(V.random_rep(6)), 500, None)
        R.flow(single, spiking_mode="lif", cycle_accurate_lif_forward=True, pass_transfer=VERBATIM,
               spike_mode="Stochastic")

    def test_stochastic_generation_refuses_at_plan_time_and_before_nevresim_compiles(self):
        """[Review round 1] The refusal fires where the mapping and the run's declaration meet: ``run_pass_transfer``
        with the built program (the hard-core mapping step), and nevresim's emit before any segment compiles."""
        u1 = V.build(V.ir_of(V.random_rep(6)), 2, None)
        with pytest.raises(C.PassReplayRefusal, match="Stochastic"):
            C.run_pass_transfer(_streamed(spike_generation_mode="Stochastic"), hybrid_mapping=u1)
        runner = SimpleNamespace(pass_transfer=VERBATIM, spike_generation_mode="Stochastic", spiking_mode="lif")
        with pytest.raises(C.PassReplayRefusal, match="Stochastic"):
            nevresim_emit.prepare_all_segments(runner, u1)

    def test_the_hard_core_mapping_step_refuses_a_stochastic_multi_pass_program_before_any_simulation(self,
                                                                                                    monkeypatch):
        """[Review round 2; integration edit IE-2, report 23.6] The step holds the built program and the run's
        declaration before it simulates anything (``run_pass_transfer(config, hybrid_mapping=hybrid_mapping)``), so a
        streamed Stochastic multi-pass run refuses there."""
        from mimarsinan.pipelining.pipeline_steps.mapping import hard_core_mapping_step as step_module

        class _Reached(Exception):
            pass

        def reached(*args, **kwargs):
            raise _Reached("the step went past its plan-time refusals")

        u1 = V.build(V.ir_of(V.random_rep(6)), 2, None)
        monkeypatch.setattr(step_module, "load_hybrid_mapping_for_step", lambda pipeline, step: u1)
        monkeypatch.setattr(step_module, "enforce_declared_buffer_capacity", reached)
        step = object.__new__(step_module.HardCoreMappingStep)
        step.pipeline = SimpleNamespace(config=_streamed(spike_generation_mode="Stochastic", simulation_steps=V.T))
        step.get_entry = lambda key: None
        with pytest.raises(C.PassReplayRefusal, match="Stochastic"):
            step.process()

    def test_a_streamed_run_with_a_backend_that_cannot_replay(self, monkeypatch):
        monkeypatch.setitem(C._BACKEND_ENABLE_KEYS, "future_backend", "enable_future_backend")
        with pytest.raises(C.PassReplayRefusal, match="future_backend"):
            C.run_pass_transfer({"spiking_family": "lif", "spiking_variant": "streamed",
                                 "enable_future_backend": True})
        assert C.run_pass_transfer({"spiking_family": "lif", "spiking_variant": "synchronized",
                                    "enable_future_backend": True}) == COLLAPSE

    def test_a_delayed_slice_on_every_backend(self, monkeypatch):
        """[W33] nevresim compiles a cycle-based COLLAPSE unit that reads a delayed slice in SpikeTrain mode and reads
        it late, as SANA-FE shifts it host-side; the analytical TTFS reads values, where a delay moves nothing; a
        backend not declared core-local refuses it by name. Fuzz seed 58 C 2: a delayed slice read at latency 0."""
        monkeypatch.setitem(C._BACKEND_ENABLE_KEYS, "future_backend", "enable_future_backend")
        u1 = V.build(V.ir_of(V.random_rep(58)), 2, None)
        assert any(getattr(s, REPLAY_DELAYS, None) for s in V.neural_passes(u1))
        kwargs = dict(spike_mode="TTFS", spiking_mode="ttfs_cycle_based")
        for backend in ("nevresim", "sanafe"):
            C.require_backend_replay(u1, backend, transfer=COLLAPSE, **kwargs)
        with pytest.raises(C.ChipClockReplayRefusal, match="delayed slices"):
            C.require_backend_replay(u1, "future_backend", transfer=COLLAPSE, **kwargs)
        C.require_backend_replay(u1, "nevresim", transfer=VERBATIM, spike_mode="Uniform", spiking_mode="lif")
        C.require_backend_replay(u1, "future_backend", transfer=COLLAPSE, spike_mode="TTFS", spiking_mode="ttfs")


class TestSingleSpikeCarry:
    """Cascaded TTFS under VERBATIM records the single-spike raster in the reference loop (it refused before) and the
    replay runs equal to the identity run."""

    @pytest.mark.parametrize("kind,count", [("skew3", 2), ("shift_probe", 3), ("late_skip", 2)])
    def test_cascaded_ttfs_carries_its_raster(self, kind, count):
        R.assert_runs_equal_to_the_identity_run(R.mapped(V.branchy_repr(kind), R.CASCADED_TTFS), count,
                                                dict(R.CASCADED_TTFS, pass_transfer=VERBATIM))

    def test_the_relay_free_vehicle_is_refused_by_name(self):
        """[U-TGAUGE5] This file's refusal test: the relay-free skew vehicle cut at C 2, read by the cascaded
        per-cycle executor, is refused by name (the window rule)."""
        R.assert_refused_at_the_executor(V.build(V.ir_of(V.branchy_repr("skew3")), 2, None),
                                         dict(R.CASCADED_TTFS, pass_transfer=VERBATIM))


class TestTheCensusPlanePricesPinnedSpans:
    """W10: with the walk's wire census, a pass the projection pins is priced at its pinned latency span; a chain
    keeps the latency-tag span."""

    def test_a_pinned_pass_spans_its_fused_latencies(self):
        """skew3 C 3: the builder pins pass 0 (L1 raised to its reader L4's latency minus one, span 1); the latency
        tags span 2."""
        rep = V.branchy_repr("skew3")
        specs, census, _ = V.layout_plane(rep)
        frag = V.fragments(specs, census, 3)
        pins = getattr(V.neural_passes(V.build(V.ir_of(rep), 3, None))[0].hard_core_mapping, PINNED_LATENCIES)
        assert execution_stage_latencies(specs, frag.pass_placements, retimed=False) == (2, 1)
        assert execution_stage_latencies(specs, frag.pass_placements, retimed=False, census=census) == (max(pins), 1)

    def test_a_chain_prices_as_before(self):
        rep = V.chain_repr(seed=0)
        specs, census, _ = V.layout_plane(rep)
        frag = V.fragments(specs, census, 3)
        assert execution_stage_latencies(specs, frag.pass_placements, retimed=False, census=census) == \
            execution_stage_latencies(specs, frag.pass_placements, retimed=False)


class TestTheWalkRuns:
    """W9: the walk's program graph gives each spec its IR-run segment (the builder's ``partition_ir_graph``); an
    entry point without the graph groups the specs by their depth segments."""

    @pytest.mark.parametrize("kind", ["edge_class", "intra_to_cross", "moved_to_output", "two_hosts"])
    def test_the_walk_runs_are_the_ir_runs(self, kind):
        rep = V.branchy_repr(kind)
        specs, _census, program = V.layout_plane(rep)
        runs = walk_runs(program, len(specs))
        ir = V.ir_of(rep)
        run_of = {int(n.id): k for k, seg in enumerate(s for s in partition_ir_graph(ir) if isinstance(s, NeuralSegment))
                  for n in seg.nodes}
        assert runs is not None and len(set(runs)) == len(set(run_of.values()))
        assert walk_runs(None, len(specs)) is None
