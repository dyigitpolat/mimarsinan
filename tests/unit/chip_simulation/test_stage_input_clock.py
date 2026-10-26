"""[W33] The stage-input timing rule: every backend reads a stage-input row in the reading core's local time.

A core whose window starts at latency L reads cycle t of its stage input at global cycle L + t, a delayed slice |d|
cycles later still (R2), as the fused segment and the HCM (``lif_step``, ``packed_cycle``) read it. SANA-FE and Lava
shift the rows host-side; nevresim reads its T-cycle input train at each core's local cycle in the generated simulator
(``nevresim/include/simulator/spike_generation/stage_input_train.hpp``) and compiles every unit whose input replays a
carried raster or a delayed slice in SpikeTrain mode, its train assembled host-side. A backend not declared to read in
core-local time refuses by name every program with a stage-input reader above latency 0, carried or host-encoded.
The compiled behaviour is ``tests/integration/test_nevresim_stage_input_clock.py``."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_single_spike_train
from mimarsinan.chip_simulation.recording._spike_encoding import encode_segment_input
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.packing.pass_timing import REPLAY_DELAYS, carried_nodes, executed_core_latencies
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid import carry as C

from unit.mapping.residency.executed import vehicles as V


def _late_readers(stage, nodes=None):
    """``(core, latency)`` per core that reads a stage-input row (of ``nodes``, or any) above executed latency 0."""
    rows = {}
    for s in stage.input_map:
        rows.update({i: int(s.node_id) for i in range(int(s.offset), int(s.offset) + int(s.size))})
    hcm = stage.hard_core_mapping
    return [(h, lat) for h, (core, lat) in enumerate(zip(hcm.cores, executed_core_latencies(hcm))) if lat > 0 and any(
        getattr(src, "is_input_", False) and not getattr(src, "is_off_", False)
        and not getattr(src, "is_always_on_", False) and (nodes is None or rows.get(int(src.neuron_)) in nodes)
        for src in core.axon_sources)]


def _streamed(**enabled):
    return {"spiking_family": "lif", "spiking_variant": "streamed", "spike_generation_mode": "Uniform", **enabled}


def _delayed_pass():
    """Fuzz seed 58 at 2 cores: one carried slice delivered 1 cycle late, read at latency 0."""
    u1 = V.build(V.ir_of(V.random_rep(58)), 2, None)
    stage = next(s for s in V.neural_passes(u1) if getattr(s, REPLAY_DELAYS, None))
    return u1, stage


class TestTheDeclaration:
    def test_every_backend_reads_a_stage_input_in_core_local_time(self):
        assert set(C._BACKEND_ENABLE_KEYS) | {"hcm"} <= C.CORE_LOCAL_REPLAY_BACKENDS

    def test_nevresim_admits_a_carried_slice_read_above_latency_zero(self):
        """Chain C 3: pass 4 reads its carried slice at latency 1 (refused by name before W33)."""
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        reader = late.stages[4]
        assert _late_readers(reader, carried_nodes(late.stages, reader)) == [(0, 1)]
        assert C.run_pass_transfer(_streamed(enable_nevresim_simulation=True), hybrid_mapping=late) == VERBATIM
        C.require_backend_replay(late, "nevresim", transfer=COLLAPSE, spike_mode="TTFS",
                                 spiking_mode="ttfs_cycle_based")

    def test_nevresim_admits_a_delayed_slice_under_collapse(self):
        u1, _ = _delayed_pass()
        C.require_backend_replay(u1, "nevresim", transfer=COLLAPSE, spike_mode="TTFS", spiking_mode="ttfs_cycle_based")

    def test_a_backend_not_declared_core_local_refuses_every_late_reader_by_name(self, monkeypatch):
        """The guard reads every stage-input row, carried or host-encoded: skew3's identity program raises a reader of
        a host-encoded slice above latency 0 (no carried slice exists there), and chain C 3 has a carried one."""
        monkeypatch.setitem(C._BACKEND_ENABLE_KEYS, "future_backend", "enable_future_backend")
        monkeypatch.setattr(C, "VERBATIM_BACKENDS", C.VERBATIM_BACKENDS | {"future_backend"})   # it replays rasters
        identity = build_identity_hybrid_mapping(ir_graph=V.ir_of(V.branchy_repr("skew3")))
        host_late = [late for s in V.neural_passes(identity) for late in _late_readers(s)]
        assert host_late and not any(carried_nodes(identity.stages, s) for s in V.neural_passes(identity))
        with pytest.raises(C.ChipClockReplayRefusal, match="future_backend reads a stage input at the global cycle"):
            C.require_backend_replay(identity, "future_backend", transfer=VERBATIM, spike_mode="Uniform",
                                     spiking_mode="lif")
        with pytest.raises(C.ChipClockReplayRefusal, match="future_backend"):
            C.run_pass_transfer(_streamed(enable_future_backend=True), hybrid_mapping=identity)
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        with pytest.raises(C.ChipClockReplayRefusal):
            C.require_backend_replay(late, "future_backend", transfer=COLLAPSE, spike_mode="TTFS",
                                     spiking_mode="ttfs_cycle_based")
        u1, _ = _delayed_pass()
        with pytest.raises(C.ChipClockReplayRefusal, match="delayed slices"):
            C.require_backend_replay(u1, "future_backend", transfer=COLLAPSE, spike_mode="TTFS",
                                     spiking_mode="ttfs_cycle_based")
        # an analytical TTFS value is read whenever its core runs: no clock refusal on any backend
        C.require_backend_replay(identity, "future_backend", transfer=COLLAPSE, spike_mode="TTFS", spiking_mode="ttfs")


class TestTheInputTrain:
    """nevresim's SpikeTrain-mode input, assembled host-side as the HCM's ``replay_carried_slices`` builds it."""

    def test_a_delayed_slice_is_read_late_without_a_carried_raster(self):
        """COLLAPSE: no raster is published; the slice's re-encode is read at its delivery delay."""
        from mimarsinan.chip_simulation.simulation_runner.carry import assemble_input_train
        _, stage = _delayed_pass()
        (k, delay), = getattr(stage, REPLAY_DELAYS).items()
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        seg_input = np.random.default_rng(0).random((2, size))
        train = assemble_input_train(stage, seg_input, {}, spike_generation_mode="Uniform", T=V.T)
        encoded = np.transpose(encode_segment_input(seg_input, V.T, "Uniform"), (0, 2, 1))
        lo, hi = int(stage.input_map[k].offset), int(stage.input_map[k].offset) + int(stage.input_map[k].size)
        assert delay < 0 and train.shape == (2, V.T, size)
        assert np.array_equal(train[:, :, lo:hi], C.replay_raster(encoded[:, :, lo:hi], delay, time_axis=1))
        rest = [i for i in range(size) if not lo <= i < hi]
        assert np.array_equal(train[:, :, rest], encoded[:, :, rest])

    def test_a_ttfs_input_is_its_single_spike_train(self):
        from mimarsinan.chip_simulation.simulation_runner.carry import assemble_input_train
        _, stage = _delayed_pass()
        (k, delay), = getattr(stage, REPLAY_DELAYS).items()
        size = max(int(s.offset) + int(s.size) for s in stage.input_map)
        seg_input = np.random.default_rng(1).random((3, size))
        train = assemble_input_train(stage, seg_input, {}, spike_generation_mode="TTFS", T=V.T)
        encoded = np.transpose(ttfs_single_spike_train(seg_input, V.T), (0, 2, 1))
        lo, hi = int(stage.input_map[k].offset), int(stage.input_map[k].offset) + int(stage.input_map[k].size)
        assert np.array_equal(train[:, :, lo:hi], C.replay_raster(encoded[:, :, lo:hi], delay, time_axis=1))
        assert train.sum(axis=1).max() <= 1

    def test_a_carried_raster_read_by_a_late_core_is_assembled_verbatim(self):
        """Chain C 3's reader at latency 1: the train holds the producer's raster as published; the generated
        simulator, not the host, moves it into the reader's local time."""
        from mimarsinan.chip_simulation.simulation_runner.carry import assemble_input_train
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        reader = late.stages[4]
        nodes = carried_nodes(late.stages, reader)
        widths = {int(s.node_id): int(s.size) for s in reader.input_map}
        rng = np.random.default_rng(2)
        store = {reader.schedule_segment_index: {n: (rng.random((1, V.T, widths[n])) < 0.5).astype(np.float64)
                                                 for n in nodes}}
        size = max(int(s.offset) + int(s.size) for s in reader.input_map)
        train = assemble_input_train(reader, np.zeros((1, size)), store, spike_generation_mode="Uniform", T=V.T)
        for s in reader.input_map:
            if int(s.node_id) in nodes:
                lo, hi = int(s.offset), int(s.offset) + int(s.size)
                assert np.array_equal(train[:, :, lo:hi], store[reader.schedule_segment_index][int(s.node_id)])


class TestTheSegmentInputMode:
    """Which units nevresim compiles in SpikeTrain input mode: every unit whose input the host must time."""

    def test_a_carried_consumer_and_a_delayed_collapse_unit_take_spike_trains(self):
        from mimarsinan.chip_simulation.simulation_runner.emit import segment_input_mode
        late = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None)
        reader = late.stages[4]
        nodes = carried_nodes(late.stages, reader)
        assert segment_input_mode(reader, nodes, transfer=VERBATIM, spiking_mode="lif") == "SpikeTrain"
        _, stage = _delayed_pass()
        assert segment_input_mode(stage, set(), transfer=COLLAPSE, spiking_mode="ttfs_cycle_based") == "SpikeTrain"

    def test_every_other_unit_encodes_on_chip(self):
        from mimarsinan.chip_simulation.simulation_runner.emit import segment_input_mode
        _, stage = _delayed_pass()
        assert segment_input_mode(stage, set(), transfer=COLLAPSE, spiking_mode="ttfs") is None
        identity = build_identity_hybrid_mapping(ir_graph=V.ir_of(V.branchy_repr("skew3")))
        for s in V.neural_passes(identity):
            assert segment_input_mode(s, set(), transfer=VERBATIM, spiking_mode="lif") is None
        retimed = V.build(V.ir_of(V.chain_repr(seed=0)), 3, None, retimed_level_stages=True)
        for s in V.neural_passes(retimed):
            for unit in s.retimed_level_stages or ():
                assert segment_input_mode(unit, set(), transfer=COLLAPSE, spiking_mode="lif") is None
