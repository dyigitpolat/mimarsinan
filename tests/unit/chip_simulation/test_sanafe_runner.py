"""SanafeRunner orchestration.

The runner is the bridge between a ``HybridHardCoreMapping`` and SANA-FE's
``SpikingChip.sim()``.  Most of the hard work (arch/network synthesis) is
delegated to dedicated modules already covered by their own unit tests;
this file pins the orchestration:

* state-buffer threading between neural / compute stages,
* spike-trace → per-group / per-core spike-count reduction
  (``_spike_trace_to_group_counts``),
* extraction of a ``SanafeSegmentRecord`` from a ``chip.sim()`` results
  dict + the runner's own per-axon span walk.

Real SANA-FE is mocked here.  A separate slow-marker test file exercises
the actual SANA-FE Python package for end-to-end parity on MNIST.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe import runner as runner_mod
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeRunRecord,
    SanafeSegmentRecord,
)
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.chip_simulation.sanafe.analysis import (
    _read_ttfs_core_activations,
    _spike_trace_to_group_counts,
    _ttfs_potential_trace_group_names,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


# ---------------------------------------------------------------------------
# Fake SANA-FE module + tiny fixture-builders
# ---------------------------------------------------------------------------


class _FakeNeuron:
    def __init__(self, group, idx):
        self.group = group
        self.idx = idx
        self.model_attributes: dict = {}
        self.soma_hw_name = None
        self.mapped_core = None
        self.connections: list = []

    def set_attributes(self, soma_hw_name=None, default_synapse_hw_name=None,
                       dendrite_hw_name=None, log_spikes=None, log_potential=None,
                       model_attributes=None, soma_attributes=None,
                       dendrite_attributes=None):
        if soma_hw_name is not None:
            self.soma_hw_name = soma_hw_name
        if model_attributes:
            self.model_attributes = dict(model_attributes)

    def map_to_core(self, core):
        self.mapped_core = core

    def connect_to_neuron(self, dst, attrs):
        self.connections.append((dst, dict(attrs)))


class _FakeGroup:
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.neurons = [_FakeNeuron(self, i) for i in range(size)]

    def get_name(self): return self.name      # mirror sanafe.NeuronGroup.get_name

    def __iter__(self): return iter(self.neurons)
    def __getitem__(self, i): return self.neurons[i]
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
        self.last_sim: dict | None = None

    def load(self, net):
        self.net = net

    def sim(self, timesteps=1, timing_model="detailed",
            processing_threads=0, scheduler_threads=0,
            spike_trace=None, potential_trace=None,
            perf_trace=None, message_trace=None,
            write_trace_headers=True):
        self.last_sim = {
            "timesteps": timesteps, "spike_trace": spike_trace,
            "potential_trace": potential_trace, "message_trace": message_trace,
        }
        if _FakeChip._next_result is None:
            raise AssertionError(
                "Test forgot to seed _FakeChip._next_result before chip.sim()"
            )
        return _FakeChip._next_result


def _fake_sanafe_module():
    return SimpleNamespace(SpikingChip=_FakeChip, Network=_FakeNetwork)


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
        axon_sources = [SpikeSource(-1, i, True, False, False) for i in range(axons)]
    # Match real-HardCore semantics: ``available_axons`` complements the
    # number of live axon_sources so ``axons_per_core - available_axons``
    # is the live count.
    available_axons = max(0, axons - len(axon_sources))
    return SimpleNamespace(
        axons_per_core=axons, neurons_per_core=neurons,
        available_axons=available_axons, available_neurons=0,
        threshold=threshold,
        core_matrix=core_matrix,
        axon_sources=list(axon_sources),
        hardware_bias=hardware_bias,
        latency=latency,
    )


def _fake_hcm(*cores):
    out = []
    if cores:
        out.append(SpikeSource(0, 0, is_input=False, is_off=False))
    return SimpleNamespace(
        cores=list(cores),
        output_sources=np.array(out, dtype=object),
    )


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


def _patch_sanafe_stack(monkeypatch, *, fake_arch=None):
    """Patch the runner's SANA-FE seams: _sanafe + arch + network synth.

    The synth modules have their own unit tests; here we substitute a
    fake builder that exercises the runner's data flow without touching
    real SANA-FE.
    """
    monkeypatch.setattr(runner_mod, "_sanafe", _fake_sanafe_module)
    if fake_arch is None:
        fake_arch = _fake_arch(2)

    def _fake_derive_arch_spec(mapping, *, preset_name, cores_per_tile=0):
        from mimarsinan.chip_simulation.sanafe.arch_synth import ArchSpec
        from mimarsinan.chip_simulation.sanafe.presets import PRESETS

        return ArchSpec(
            name=f"fake_{preset_name}",
            n_tiles=1,
            n_cores_per_tile=[len(mapping.get_neural_segments()[0].cores)],
            axons_per_core=4,
            neurons_per_core=4,
            preset=PRESETS[preset_name],
            dendrite_plugin_path="/fake/dendrite.so",
            soma_plugin_path="/fake/soma.so",
            ttfs_continuous_plugin_path="/fake/ttfs_cont.so",
            ttfs_quantized_plugin_path="/fake/ttfs_q.so",
        )

    monkeypatch.setattr(runner_mod, "derive_arch_spec", _fake_derive_arch_spec)
    monkeypatch.setattr(
        runner_mod, "build_architecture",
        lambda spec, custom_arch_path=None, thresholding_mode="<=",
               simulation_length=1: fake_arch,
    )

    def default_builder(arch, hcm, **kw):  # accepts spiking_mode, simulation_length, ...
        net = _FakeNetwork()
        core_to_group = {
            i: net.create_neuron_group(f"core{i}", hcm.cores[i].neurons_per_core)
            for i in range(len(hcm.cores))
        }
        # New net_synth API: returns (net, core_to_group, core_input_neurons,
        # core_always_on_neurons).  Synthesise per-(core,axon) input neurons
        # from the HCM directly so the runner's spike-train injection has
        # something to drive.
        core_input_neurons = {}
        core_always_on_neurons = {}
        for ci, core in enumerate(hcm.cores):
            for a, src in enumerate(getattr(core, "axon_sources", [])):
                if getattr(src, "is_off_", False):
                    continue
                if getattr(src, "is_input_", False):
                    g = net.create_neuron_group(f"core{ci}_in_{a}", 1)
                    core_input_neurons[(ci, a)] = g[0]
                elif getattr(src, "is_always_on_", False):
                    if ci not in core_always_on_neurons:
                        g = net.create_neuron_group(f"core{ci}_on", 1)
                        core_always_on_neurons[ci] = g[0]
        return net, core_to_group, core_input_neurons, core_always_on_neurons

    monkeypatch.setattr(runner_mod, "build_network_for_segment", default_builder)
    monkeypatch.setattr(runner_mod, "set_input_spike_trains",
                        lambda core_input_neurons, hcm, encoded: None)
    monkeypatch.setattr(runner_mod, "set_always_on_spike_trains",
                        lambda core_always_on_neurons, T, **kw: None)
    monkeypatch.setattr(runner_mod, "set_ttfs_input_spike_trains",
                        lambda *args, **kw: None)
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
        # SANA-FE-shaped spike_trace: list-per-timestep of "group.idx" strings.
        "spike_trace": [],
    }
    result.update(overrides)
    _FakeChip._next_result = result


# ---------------------------------------------------------------------------
# spike_trace decoder unit
# ---------------------------------------------------------------------------


def test_spike_trace_decoder_tallies_group_index_strings():
    trace = [
        ["core0.0", "core0.1"],
        [],
        ["core0.0", "core1.0", "core0.0"],
        ["in.2"],
    ]
    counts, skipped = _spike_trace_to_group_counts(
        trace, group_sizes={"core0": 3, "core1": 2, "in": 4},
    )
    assert skipped == 0
    assert counts["core0"].tolist() == [3, 1, 0]
    assert counts["core1"].tolist() == [1, 0]
    assert counts["in"].tolist() == [0, 0, 1, 0]


def test_spike_trace_decoder_ignores_unknown_groups_and_indices():
    trace = [["ghost.0", "core0.9", "core0.0", "junk"]]
    counts, skipped = _spike_trace_to_group_counts(trace, group_sizes={"core0": 3})
    assert counts["core0"].tolist() == [1, 0, 0]
    assert skipped == 3


# ---------------------------------------------------------------------------
# Init / config
# ---------------------------------------------------------------------------


def test_runner_init_stores_config_and_does_not_import_sanafe(monkeypatch):
    sentinel = []
    monkeypatch.setattr(runner_mod, "_sanafe",
                        lambda: sentinel.append("touched"))
    SanafeRunner(
        mapping=_fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core()))),
        simulation_length=8,
    )
    assert sentinel == []


def test_runner_accepts_ttfs_quantized_spiking_mode(monkeypatch):
    mapping = _fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core())))
    monkeypatch.setattr(runner_mod, "_sanafe",
                        lambda: (_ for _ in ()).throw(AssertionError("no import")))
    runner = SanafeRunner(
        mapping=mapping, simulation_length=8,
        spiking_mode="ttfs_quantized", firing_mode="TTFS",
    )
    assert runner.spiking_mode == "ttfs_quantized"


def test_runner_rejects_unknown_arch_preset():
    mapping = _fake_mapping(_fake_stage("neural", hcm=_fake_hcm(_fake_hard_core())))
    with pytest.raises(ValueError, match="preset"):
        SanafeRunner(mapping=mapping, simulation_length=8,
                     arch_preset="silicon-dreams")


# ---------------------------------------------------------------------------
# Single neural stage
# ---------------------------------------------------------------------------


def test_run_returns_sanafe_run_record_for_single_neural_stage(monkeypatch):
    core = _fake_hard_core(axons=2, neurons=3, latency=0)
    stage = _fake_stage(
        "neural", name="s0", hcm=_fake_hcm(core),
        input_map=[_seg_io_slice(node_id=-2, offset=0, size=2)],
        output_map=[_seg_io_slice(node_id=0, offset=0, size=3)],
    )
    mapping = _fake_mapping(stage)
    _patch_sanafe_stack(monkeypatch)
    # core0 fires neuron 0 four times, neuron 1 eight times over 8 cycles.
    trace = ([["core0.0", "core0.1"]] * 4) + ([["core0.1"]] * 4)
    _seed_chip_result(spikes=12, packets_sent=7, neurons_fired=12,
                      spike_trace=trace)

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
    assert seg.energy.total_j == pytest.approx(1.0)
    assert seg.spikes == 12
    assert seg.per_core[0].output_spike_count.tolist() == [4, 8, 0]
    assert seg.per_core[0].spikes_fired == 12


def test_run_threads_chip_sim_with_extended_simulation_length(monkeypatch):
    """Runner pads ``chip.sim`` to ``T + max_core_latency + 1`` cycles.

    The +1 accounts for SANA-FE's input→synapse pipeline delay (an input
    neuron's spike at sim_time t reaches its consumer's synapse at t+1)
    and the ``max_core_latency`` extension lets multi-depth cascades
    flush through within the simulation.  Here ``latency=0`` for the
    single fake core, so we expect ``T + 0 + 1 = 33`` ticks.
    """
    stage = _fake_stage(
        "neural", hcm=_fake_hcm(_fake_hard_core(axons=2, neurons=2)),
        input_map=[_seg_io_slice(-2, 0, 2)],
        output_map=[_seg_io_slice(0, 0, 2)],
    )
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[[] for _ in range(33)])
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=32)
    runner.run(np.asarray([[0.0, 0.0]], dtype=np.float32), sample_index=0)
    chip = runner._last_chip
    assert chip.last_sim["timesteps"] == 33


def test_run_propagates_arch_preset_to_record(monkeypatch):
    stage = _fake_stage(
        "neural", hcm=_fake_hcm(_fake_hard_core()),
        input_map=[_seg_io_slice(-2, 0, 2)],
        output_map=[_seg_io_slice(0, 0, 2)],
    )
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[[] for _ in range(8)])
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=8,
                          arch_preset="truenorth")
    rec = runner.run(np.asarray([[0.0, 0.0]], dtype=np.float32), sample_index=4)
    assert rec.arch_preset == "truenorth"
    assert rec.sample_index == 4


# ---------------------------------------------------------------------------
# Hybrid mapping
# ---------------------------------------------------------------------------


def test_run_executes_compute_stage_via_hybrid_execution(monkeypatch):
    op = SimpleNamespace(id=42)
    stage_compute = _fake_stage("compute", name="op", compute_op=op,
                                input_map=[], output_map=[])
    mapping = _fake_mapping(stage_compute)
    _patch_sanafe_stack(monkeypatch)

    called = {}
    def fake_compute(op_arg, original_input, state_buffer, *,
                     in_scale, out_scale, dtype=np.float32):
        # Accept ``dtype`` — the runner passes ``dtype=np.float64`` so
        # compute-op arithmetic matches HCM's ``_COMPUTE_DTYPE``.
        called["op"] = op_arg
        called["dtype"] = dtype
        return np.asarray([[3.0]], dtype=dtype)
    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy", fake_compute)

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)

    assert called["op"] is op
    assert called["dtype"] == np.float64
    assert 42 in rec.compute_outputs
    np.testing.assert_array_equal(rec.compute_outputs[42],
                                  np.asarray([[3.0]], dtype=np.float64))


def test_run_lif_compute_stage_does_not_apply_ttfs_scales(monkeypatch):
    op = SimpleNamespace(id=42)
    stage_compute = _fake_stage("compute", name="op", compute_op=op,
                                input_map=[], output_map=[])
    mapping = _fake_mapping(stage_compute)
    mapping.node_input_activation_scales = {42: 7.0}
    mapping.node_activation_scales = {42: 11.0}
    _patch_sanafe_stack(monkeypatch)

    called = {}

    def fake_compute(op_arg, original_input, state_buffer, *,
                     in_scale, out_scale, dtype=np.float32):
        called["in_scale"] = in_scale
        called["out_scale"] = out_scale
        return np.asarray([[3.0]], dtype=dtype)

    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy", fake_compute)

    runner = SanafeRunner(mapping=mapping, simulation_length=8, spiking_mode="lif")
    runner.run(np.asarray([[1.0, 2.0]], dtype=np.float32), sample_index=0)

    assert called["in_scale"] == 1.0
    assert called["out_scale"] == 1.0


def test_run_ttfs_compute_stage_records_op_id_after_neural(monkeypatch):
    """Regression: TTFS compute path must bind op.id for compute_outputs."""
    core = _fake_hard_core(axons=1, neurons=1)
    s1 = _fake_stage(
        "neural", name="s1", hcm=_fake_hcm(core),
        input_map=[_seg_io_slice(-2, 0, 1)],
        output_map=[_seg_io_slice(0, 0, 1)],
    )
    op = SimpleNamespace(id=77)
    s2 = _fake_stage("compute", name="op", compute_op=op, input_map=[], output_map=[])
    mapping = _fake_mapping(
        s1, s2, output_sources=np.array([], dtype=object),
    )
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[["core0.0"]] * 8)

    contract_called = {}

    def fake_contract(mapping, stage, state_buffer, sample_input):
        contract_called["stage"] = stage
        out = np.asarray([[0.25]], dtype=np.float64)
        return SimpleNamespace(op_id=77, output=out)

    monkeypatch.setattr(
        "mimarsinan.chip_simulation.ttfs.ttfs_executor.run_ttfs_contract_compute_stage",
        fake_contract,
    )
    monkeypatch.setattr(runner_mod, "is_ttfs_spiking_mode", lambda _mode: True)

    runner = SanafeRunner(
        mapping=mapping,
        simulation_length=8,
        spiking_mode="ttfs",
        firing_mode="TTFS",
    )
    rec = runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)
    assert contract_called["stage"] is s2
    assert 77 in rec.compute_outputs
    np.testing.assert_array_equal(
        rec.compute_outputs[77], np.asarray([[0.25]], dtype=np.float64),
    )


def test_run_walks_stages_in_order(monkeypatch):
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
    mapping = _fake_mapping(s1, s2, s3)
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[["core0.0"]] * 8)
    monkeypatch.setattr(runner_mod, "execute_compute_op_numpy",
                        lambda op, orig, buf, **kw: np.asarray([[0.5]], dtype=np.float32))

    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)
    assert sorted(rec.segments.keys()) == [0, 2]
    assert rec.segments[0].stage_name == "s1"
    assert rec.segments[2].stage_name == "s3"
    assert 99 in rec.compute_outputs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_run_aggregates_total_energy_spikes_packets_across_segments(monkeypatch):
    core = _fake_hard_core(axons=1, neurons=1)
    stages = [
        _fake_stage("neural", name=f"s{i}", hcm=_fake_hcm(core),
                    input_map=[_seg_io_slice(-2, 0, 1)],
                    output_map=[_seg_io_slice(i, 0, 1)])
        for i in range(2)
    ]
    mapping = _fake_mapping(*stages)
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(
        spikes=5, packets_sent=2,
        energy={"total": 2.5, "synapse": 1.0, "dendrite": 0.25,
                "soma": 0.75, "network": 0.5},
        spike_trace=[["core0.0"]] * 8,
    )
    runner = SanafeRunner(mapping=mapping, simulation_length=8)
    rec = runner.run(np.asarray([[1.0]], dtype=np.float32), sample_index=0)
    assert rec.total_spikes == 10
    assert rec.total_packets == 4
    assert rec.aggregate_energy.total_j == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Per-core input spike count
# ---------------------------------------------------------------------------


def test_run_derives_per_core_input_spike_count_from_input_axons(monkeypatch):
    core = _fake_hard_core(
        axons=2, neurons=1,
        axon_sources=[
            SpikeSource(-1, 0, True, False, False),
            SpikeSource(-1, 1, True, False, False),
        ],
    )
    stage = _fake_stage("neural", hcm=_fake_hcm(core),
                        input_map=[_seg_io_slice(-2, 0, 2)],
                        output_map=[_seg_io_slice(0, 0, 1)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[[] for _ in range(8)])
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=8)
    rec = runner.run(np.asarray([[0.5, 1.0]], dtype=np.float32), sample_index=0)
    core_rec = rec.segments[0].per_core[0]
    assert core_rec.input_spike_count.tolist() == [4, 8]


def test_ttfs_potential_trace_group_order_is_lexicographic():
    chip = SimpleNamespace(mapped_neuron_groups={
        "core10": [object()],
        "core2": [object(), object()],
        "core0": [object()],
        "core0_in": [object(), object()],
    })
    assert _ttfs_potential_trace_group_names(chip) == ["core0", "core10", "core2"]


def test_read_ttfs_core_activations_slices_potential_trace():
    chip = SimpleNamespace(mapped_neuron_groups={
        "core0": [object(), object()],
        "core1": [object()],
    })
    results = {"potential_trace": [[0.1, 0.2, 0.3]]}
    out = _read_ttfs_core_activations(chip, 1, 1, results)
    np.testing.assert_allclose(out, [0.3])


def test_run_per_core_input_count_always_on_axon_counts_T(monkeypatch):
    core = _fake_hard_core(
        axons=1, neurons=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],
    )
    stage = _fake_stage("neural", hcm=_fake_hcm(core),
                        input_map=[_seg_io_slice(-2, 0, 1)],
                        output_map=[_seg_io_slice(0, 0, 1)])
    _patch_sanafe_stack(monkeypatch)
    _seed_chip_result(spike_trace=[[] for _ in range(12)])
    runner = SanafeRunner(mapping=_fake_mapping(stage), simulation_length=12)
    rec = runner.run(np.asarray([[0.0]], dtype=np.float32), sample_index=0)
    assert rec.segments[0].per_core[0].input_spike_count.tolist() == [12]
    assert rec.segments[0].per_core[0].n_always_on_axons == 1
