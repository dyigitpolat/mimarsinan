"""SanafeRunner orchestration.

The runner is the bridge between a ``HybridHardCoreMapping`` and SANA-FE's
``SpikingChip.sim()``.  Most of the hard work (arch/network synthesis) is
delegated to dedicated modules already covered by their own unit tests;
this file pins the orchestration:

* state-buffer threading between neural / compute stages,
* per-stage spike-count → rate conversion (spike_count / T),
* extraction of a ``SanafeSegmentRecord`` from a ``chip.sim()`` results
  dict + the runner's own per-axon span walk.

Real SANA-FE is mocked here.  A separate slow-marker test file exercises
the actual SANA-FE Python package for single-core parity.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe import runner as runner_mod
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
)
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


# ---------------------------------------------------------------------------
# Fake SANA-FE module + tiny fixture-builders
# ---------------------------------------------------------------------------


class _FakeNeuron:
    def __init__(self, group, idx):
        self.group = group
        self.idx = idx
        self.attributes: dict = {}
        self.mapped_core = None
        self.connections: list = []

    def set_attributes(self, model_attributes=None, log_spikes=False,
                       log_potential=False):
        if model_attributes:
            self.attributes.update(model_attributes)

    def map_to_core(self, core):
        self.mapped_core = core

    def connect_to_neuron(self, dst, attrs):
        self.connections.append((dst, dict(attrs)))


class _FakeGroup:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.neurons = [_FakeNeuron(self, i) for i in range(size)]

    def __getitem__(self, i): return self.neurons[i]

    def __iter__(self): return iter(self.neurons)

    def __len__(self): return self.size


class _FakeNetwork:
    def __init__(self):
        self.groups = []

    def create_neuron_group(self, name, size, model_attributes=None):
        g = _FakeGroup(name, size)
        self.groups.append(g)
        return g


class _FakeChip:
    """Returns a programmed result dict; lets tests pin spikes / energy."""

    _next_result: dict | None = None

    def __init__(self, arch):
        self.arch = arch
        self.last_sim = None

    def load(self, net):
        self.net = net

    def sim(self, T, spike_trace=False, potential_trace=False,
            perf_trace=None, message_trace=None,
            input_spikes=None):
        self.last_sim = {
            "T": T, "spike_trace": spike_trace,
            "potential_trace": potential_trace,
            "input_spikes": input_spikes,
        }
        if _FakeChip._next_result is None:
            raise AssertionError(
                "Test forgot to seed _FakeChip._next_result before chip.sim()"
            )
        return _FakeChip._next_result


def _fake_sanafe_module():
    return SimpleNamespace(SpikingChip=_FakeChip)


def _fake_arch(*tile_core_counts):
    tiles = []
    for t, n in enumerate(tile_core_counts):
        cores = [SimpleNamespace(_t=t, _c=c) for c in range(n)]
        tiles.append(SimpleNamespace(cores=cores))
    return SimpleNamespace(tiles=tiles)


def _fake_hard_core(*, axons=2, neurons=2, threshold=1.0,
                    axon_sources=None, core_matrix=None,
                    hardware_bias=None, latency=0):
    if core_matrix is None:
        core_matrix = np.zeros((axons, neurons), dtype=np.float32)
    if axon_sources is None:
        axon_sources = [SpikeSource(-1, i, True, False, False)
                        for i in range(axons)]
    return SimpleNamespace(
        axons_per_core=axons, neurons_per_core=neurons,
        available_axons=0, available_neurons=0,
        threshold=threshold,
        core_matrix=core_matrix,
        axon_sources=list(axon_sources),
        hardware_bias=hardware_bias,
        latency=latency,
        get_axon_source_spans=lambda: [],
    )


def _fake_hcm(*cores):
    return SimpleNamespace(cores=list(cores))


def _fake_stage(kind, *, name="stage", hcm=None, compute_op=None,
                input_map=None, output_map=None,
                schedule_segment_index=None, schedule_pass_index=None):
    return SimpleNamespace(
        kind=kind, name=name,
        hard_core_mapping=hcm, compute_op=compute_op,
        input_map=input_map or [],
        output_map=output_map or [],
        schedule_segment_index=schedule_segment_index,
        schedule_pass_index=schedule_pass_index,
    )


def _fake_mapping(*stages, output_sources=None):
    return SimpleNamespace(
        stages=list(stages),
        get_neural_segments=lambda: [s.hard_core_mapping for s in stages
                                     if s.kind == "neural" and s.hard_core_mapping],
        get_compute_ops=lambda: [s.compute_op for s in stages
                                 if s.kind == "compute" and s.compute_op],
        output_sources=output_sources if output_sources is not None
                       else np.array([], dtype=object),
        node_activation_scales={},
        node_input_activation_scales={},
    )


def _seg_io_slice(node_id, offset, size):
    return SimpleNamespace(node_id=node_id, offset=offset, size=size)


def _patch_sanafe_stack(monkeypatch, *, fake_arch=None, fake_net_builder=None):
    """Patch _sanafe + build_architecture + build_network_for_segment.

    The runner's orchestration is exercised; the synthesis modules are
    tested independently in their own files.
    """
    monkeypatch.setattr(runner_mod, "_sanafe", _fake_sanafe_module)
    if fake_arch is None:
        fake_arch = _fake_arch(2)
    monkeypatch.setattr(runner_mod, "build_architecture",
                        lambda spec, custom_arch_path=None: fake_arch)
    if fake_net_builder is None:
        def default_builder(arch, hcm, **kw):
            net = _FakeNetwork()
            core_to_group = {
                i: net.create_neuron_group(f"core{i}",
                                           hcm.cores[i].neurons_per_core)
                for i in range(len(hcm.cores))
            }
            input_group = net.create_neuron_group("input", kw["seg_in_size"])
            return net, core_to_group, input_group, None
        fake_net_builder = default_builder
    monkeypatch.setattr(runner_mod, "build_network_for_segment", fake_net_builder)
    return fake_arch


def _seed_chip_result(**overrides):
    """Default chip.sim() result; tests override what they care about."""
    result = {
        "timestep_start": 0,
        "timesteps_executed": 8,
        "sim_time": 1.6e-6,
        "spikes": 0,
        "packets_sent": 0,
        "neurons_updated": 0,
        "neurons_fired": 0,
        "energy": {"total": 1.0, "synapse": 0.4, "dendrite": 0.1,
                   "soma": 0.3, "network": 0.2},
        # per-neuron spike counts in group order (input group last)
        "group_spike_counts": {},
    }
    result.update(overrides)
    _FakeChip._next_result = result


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


def test_runner_init_stores_config_and_does_not_import_sanafe(monkeypatch):
    """Constructing the runner must NOT touch SANA-FE — it stays lazy."""
    sentinel = []
    monkeypatch.setattr(runner_mod, "_sanafe",
                        lambda: sentinel.append("touched"))
    SanafeRunner(
        mapping=_fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core()))),
        simulation_length=8,
    )
    assert sentinel == []


def test_runner_rejects_non_lif_spiking_mode(monkeypatch):
    monkeypatch.setattr(runner_mod, "_sanafe", _fake_sanafe_module)
    mapping = _fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core())))
    with pytest.raises(ValueError, match="lif"):
        SanafeRunner(mapping=mapping, simulation_length=8, spiking_mode="ttfs")


def test_runner_rejects_unknown_arch_preset(monkeypatch):
    monkeypatch.setattr(runner_mod, "_sanafe", _fake_sanafe_module)
    mapping = _fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core())))
    with pytest.raises(ValueError, match="preset"):
        SanafeRunner(mapping=mapping, simulation_length=8,
                     arch_preset="silicon-dreams")


# ---------------------------------------------------------------------------
# Single neural stage — returns a populated SanafeRunRecord
# ---------------------------------------------------------------------------


def test_run_returns_sanafe_run_record_for_single_neural_stage(monkeypatch):
    core = _fake_hard_core(axons=2, neurons=3, latency=0)
    hcm = _fake_hcm(core)
    stage = _fake_stage(
        "neural", name="s0", hcm=hcm,
        input_map=[_seg_io_slice(node_id=-2, offset=0, size=2)],
        output_map=[_seg_io_slice(node_id=0, offset=0, size=3)],
    )
    mapping = _fake_mapping(
        stage,
        output_sources=np.asarray(
            [SimpleNamespace(is_off=lambda: False,
                             is_input=lambda: False,
                             is_always_on=lambda: False,
                             node_id=0, index=i) for i in range(3)],
            dtype=object,
        ),
    )
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        spikes=20, packets_sent=7, neurons_updated=24, neurons_fired=4,
        group_spike_counts={"core0": np.asarray([4, 8, 0], dtype=np.int64)},
    )

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[0.5, 1.0]], dtype=np.float32), sample_index=0)

    assert isinstance(rec, SanafeRunRecord)
    assert rec.sample_index == 0
    assert rec.T == 8
    assert set(rec.segments.keys()) == {0}
    seg = rec.segments[0]
    assert isinstance(seg, SanafeSegmentRecord)
    assert seg.stage_index == 0
    assert seg.stage_name == "s0"
    assert seg.spikes == 20
    assert seg.packets_sent == 7
    assert seg.energy.total_j == pytest.approx(1.0)
    # per-core spikes_fired comes from group_spike_counts sum
    assert seg.per_core[0].spikes_fired == 12
    np.testing.assert_array_equal(
        seg.per_core[0].output_spike_count,
        np.asarray([4, 8, 0], dtype=np.int64),
    )


def test_run_threads_chip_sim_with_simulation_length(monkeypatch):
    core = _fake_hard_core(axons=2, neurons=2)
    stage = _fake_stage("neural", hcm=_fake_hcm(core),
                        input_map=[_seg_io_slice(-2, 0, 2)],
                        output_map=[_seg_io_slice(0, 0, 2)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        group_spike_counts={"core0": np.asarray([0, 0], dtype=np.int64)},
    )
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=32)
    runner.run(np.asarray([[0.0, 0.0]], dtype=np.float32), sample_index=0)
    # The fake chip stored its last sim call's T.
    chip = runner._last_chip   # set by the runner for white-box test access
    assert chip.last_sim["T"] == 32


def test_run_propagates_arch_preset_to_record(monkeypatch):
    stage = _fake_stage("neural", hcm=_fake_hcm(_fake_hard_core()),
                        input_map=[_seg_io_slice(-2, 0, 2)],
                        output_map=[_seg_io_slice(0, 0, 2)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        group_spike_counts={"core0": np.asarray([0, 0], dtype=np.int64)},
    )
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=8,
                          arch_preset="truenorth")
    rec = runner.run(np.asarray([[0.0, 0.0]], dtype=np.float32), sample_index=4)
    assert rec.arch_preset == "truenorth"
    assert rec.sample_index == 4


# ---------------------------------------------------------------------------
# Hybrid mapping (neural + compute) — state-buffer threading
# ---------------------------------------------------------------------------


def test_run_executes_compute_stage_via_hybrid_execution(monkeypatch):
    """A pure-compute stage runs ``execute_compute_op_numpy`` and stores in buffer."""
    op = SimpleNamespace(id=42)
    stage_compute = _fake_stage("compute", name="op", compute_op=op,
                                input_map=[], output_map=[])
    mapping = _fake_mapping(
        stage_compute,
        output_sources=np.array(
            [SimpleNamespace(is_off=lambda: False,
                             is_input=lambda: False,
                             is_always_on=lambda: False,
                             node_id=42, index=0)],
            dtype=object,
        ),
    )
    _patch_sanafe_stack(monkeypatch)

    called = {}
    def fake_compute(op_arg, original_input, state_buffer, *, in_scale, out_scale):
        called["op"] = op_arg
        called["state_buffer"] = dict(state_buffer)
        return np.asarray([[3.0]], dtype=np.float32)
    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy", fake_compute)

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)

    assert called["op"] is op
    # The compute_op's output is now in compute_outputs[op.id].
    assert 42 in rec.compute_outputs
    np.testing.assert_array_equal(rec.compute_outputs[42],
                                  np.asarray([[3.0]], dtype=np.float32))


def test_run_walks_stages_in_order(monkeypatch):
    """Neural → compute → neural order is preserved end-to-end."""
    core = _fake_hard_core(axons=1, neurons=1)
    s1 = _fake_stage("neural", name="s1", hcm=_fake_hcm(core),
                     input_map=[_seg_io_slice(-2, 0, 1)],
                     output_map=[_seg_io_slice(0, 0, 1)])
    op = SimpleNamespace(id=99)
    s2 = _fake_stage("compute", name="op", compute_op=op,
                     input_map=[], output_map=[])
    s3 = _fake_stage("neural", name="s3", hcm=_fake_hcm(core),
                     input_map=[_seg_io_slice(99, 0, 1)],
                     output_map=[_seg_io_slice(1, 0, 1)])
    mapping = _fake_mapping(s1, s2, s3,
                            output_sources=np.array(
                                [SimpleNamespace(is_off=lambda: False,
                                                 is_input=lambda: False,
                                                 is_always_on=lambda: False,
                                                 node_id=1, index=0)],
                                dtype=object,
                            ))
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        group_spike_counts={"core0": np.asarray([1], dtype=np.int64)},
    )
    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy",
                        lambda op, orig, buf, **kw: np.asarray([[0.5]], dtype=np.float32))

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)

    # Both neural stages produced segment records (keyed by stage_index).
    assert sorted(rec.segments.keys()) == [0, 2]
    assert rec.segments[0].stage_name == "s1"
    assert rec.segments[2].stage_name == "s3"
    # The compute stage contributed to compute_outputs.
    assert 99 in rec.compute_outputs


# ---------------------------------------------------------------------------
# Aggregation across segments
# ---------------------------------------------------------------------------


def test_run_aggregates_total_energy_spikes_packets_across_segments(monkeypatch):
    core = _fake_hard_core(axons=1, neurons=1)
    stages = [
        _fake_stage("neural", name=f"s{i}", hcm=_fake_hcm(core),
                    input_map=[_seg_io_slice(-2, 0, 1)],
                    output_map=[_seg_io_slice(i, 0, 1)])
        for i in range(2)
    ]
    mapping = _fake_mapping(*stages,
                            output_sources=np.array(
                                [SimpleNamespace(is_off=lambda: False,
                                                 is_input=lambda: False,
                                                 is_always_on=lambda: False,
                                                 node_id=1, index=0)],
                                dtype=object,
                            ))
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        spikes=5, packets_sent=2, neurons_updated=4, neurons_fired=1,
        energy={"total": 2.5, "synapse": 1.0, "dendrite": 0.25,
                "soma": 0.75, "network": 0.5},
        group_spike_counts={"core0": np.asarray([1], dtype=np.int64)},
    )

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)

    assert rec.total_spikes == 10        # 5 + 5
    assert rec.total_packets == 4         # 2 + 2
    assert rec.aggregate_energy.total_j == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Per-core input spike count derivation (host-side)
# ---------------------------------------------------------------------------


def test_run_derives_per_core_input_spike_count_from_input_axons(monkeypatch):
    """Input-kind axons feed each core; per-axon counts come from the rate-encoded train."""
    core = _fake_hard_core(
        axons=2, neurons=1,
        axon_sources=[
            SpikeSource(-1, 0, True, False, False),   # input[0]
            SpikeSource(-1, 1, True, False, False),   # input[1]
        ],
    )
    stage = _fake_stage("neural", hcm=_fake_hcm(core),
                        input_map=[_seg_io_slice(-2, 0, 2)],
                        output_map=[_seg_io_slice(0, 0, 1)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        group_spike_counts={"core0": np.asarray([0], dtype=np.int64)},
    )
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=8)
    # rate 0.5 → 4 spikes / cycle 8; rate 1.0 → 8 spikes
    rec = runner.run(np.asarray([[0.5, 1.0]], dtype=np.float32), sample_index=0)
    core_rec = rec.segments[0].per_core[0]
    assert core_rec.input_spike_count.tolist() == [4, 8]


def test_run_per_core_input_count_always_on_axon_counts_T(monkeypatch):
    core = _fake_hard_core(
        axons=1, neurons=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],   # always_on
    )
    stage = _fake_stage("neural", hcm=_fake_hcm(core),
                        input_map=[_seg_io_slice(-2, 0, 1)],
                        output_map=[_seg_io_slice(0, 0, 1)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        group_spike_counts={"core0": np.asarray([0], dtype=np.int64)},
    )
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=12)
    rec = runner.run(np.asarray([[0.0]], dtype=np.float32), sample_index=0)
    assert rec.segments[0].per_core[0].input_spike_count.tolist() == [12]
    assert rec.segments[0].per_core[0].n_always_on_axons == 1
