"""SANA-FE Network synthesis from one ``HardCoreMapping`` neural segment.

Hardware-faithful mapping pinned here:

* one LIF group per HardCore on its corresponding SANA-FE core,
* per-axon input/always-on neurons live on the **same** SANA-FE core
  as the consuming HardCore (using local ``inputs[axon_idx]`` soma slots),
* cross-core axons wire directly from the upstream HardCore's LIF
  neuron — no global input host,
* per-axon spike-train injection looks up the logical input index from
  ``core.axon_sources[axon_idx].neuron_``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe import net_synth
from mimarsinan.chip_simulation.sanafe.net_synth import (
    build_network_for_segment,
    set_always_on_spike_trains,
    set_input_spike_trains,
)
from mimarsinan.chip_simulation.sanafe.presets import (
    SOMA_INPUT_RANGE_NAME, SOMA_LIF_NAME, SYNAPSE_NAME,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


# ---------------------------------------------------------------------------
# Fake SANA-FE module
# ---------------------------------------------------------------------------


class _FakeNeuron:
    def __init__(self, group, index):
        self.group = group
        self.index = index
        self.model_attributes: dict = {}
        self.soma_hw_name: str | None = None
        self.default_synapse_hw_name: str | None = None
        self.mapped_core = None
        self.connections: list[tuple[_FakeNeuron, dict]] = []

    def set_attributes(self, soma_hw_name=None, default_synapse_hw_name=None,
                       dendrite_hw_name=None, log_spikes=None, log_potential=None,
                       model_attributes=None, soma_attributes=None,
                       dendrite_attributes=None):
        if soma_hw_name is not None:
            self.soma_hw_name = soma_hw_name
        if default_synapse_hw_name is not None:
            self.default_synapse_hw_name = default_synapse_hw_name
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

    def __iter__(self): return iter(self.neurons)
    def __getitem__(self, i): return self.neurons[i]
    def __len__(self): return self.size


class _FakeNetwork:
    def __init__(self):
        self.groups: list[_FakeGroup] = []

    def create_neuron_group(self, name, size, model_attributes=None):
        g = _FakeGroup(name, size)
        self.groups.append(g)
        return g


def _fake_sanafe_module():
    return SimpleNamespace(Network=_FakeNetwork)


def _fake_arch(*tile_core_counts: int):
    tiles = []
    for t, n in enumerate(tile_core_counts):
        cores = [SimpleNamespace(_t=t, _c=c) for c in range(n)]
        tiles.append(SimpleNamespace(cores=cores))
    return SimpleNamespace(tiles=tiles)


def _fake_hard_core(*, axons_per_core, neurons_per_core, available_axons=0,
                    available_neurons=0, threshold=1.0, hardware_bias=None,
                    core_matrix=None, axon_sources=None):
    if core_matrix is None:
        core_matrix = np.zeros((axons_per_core, neurons_per_core), dtype=np.float32)
    if axon_sources is None:
        axon_sources = []
    return SimpleNamespace(
        axons_per_core=axons_per_core,
        neurons_per_core=neurons_per_core,
        available_axons=available_axons,
        available_neurons=available_neurons,
        threshold=threshold,
        hardware_bias=hardware_bias,
        core_matrix=core_matrix,
        axon_sources=list(axon_sources),
    )


def _fake_hcm(*cores): return SimpleNamespace(cores=list(cores))


# ---------------------------------------------------------------------------
# LIF group placement: one per HardCore on its consuming SANA-FE core
# ---------------------------------------------------------------------------


def test_build_network_creates_one_lif_group_per_hard_core(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2)
    core_a = _fake_hard_core(axons_per_core=2, neurons_per_core=3)
    core_b = _fake_hard_core(axons_per_core=2, neurons_per_core=1)

    _, core_to_group, _, _ = build_network_for_segment(
        arch, _fake_hcm(core_a, core_b),
        tile_offset=0, core_offset=0,
    )

    assert len(core_to_group) == 2
    assert core_to_group[0].size == 3
    assert core_to_group[1].size == 1


def test_build_network_lif_neurons_map_to_their_hardcore_sanafe_core(monkeypatch):
    """Each HardCore's LIF group lands on the SANA-FE core matching its index."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(3)
    cores = [_fake_hard_core(axons_per_core=1, neurons_per_core=1) for _ in range(3)]
    _, c2g, _, _ = build_network_for_segment(
        arch, _fake_hcm(*cores),
        tile_offset=0, core_offset=0,
    )
    for hc_idx, group in c2g.items():
        for n in group:
            assert n.mapped_core is arch.tiles[0].cores[hc_idx]


def test_build_network_lif_soma_hw_and_synapse_hw_are_named(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    _, c2g, _, _ = build_network_for_segment(
        _fake_arch(1),
        _fake_hcm(_fake_hard_core(axons_per_core=1, neurons_per_core=2)),
        tile_offset=0, core_offset=0,
    )
    for n in c2g[0]:
        assert n.soma_hw_name == SOMA_LIF_NAME
        assert n.default_synapse_hw_name == SYNAPSE_NAME


def test_build_network_lif_threshold_and_bias_are_set_per_neuron(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    bias = np.asarray([0.5, -0.25, 1.0], dtype=np.float32)
    core = _fake_hard_core(axons_per_core=1, neurons_per_core=3, threshold=1.5,
                           hardware_bias=bias)
    _, c2g, _, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    for i, n in enumerate(c2g[0]):
        assert n.model_attributes["threshold"] == 1.5
        assert n.model_attributes["bias"] == pytest.approx(float(bias[i]))


def test_build_network_no_bias_attribute_when_hardware_bias_is_none(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(axons_per_core=1, neurons_per_core=2, hardware_bias=None)
    _, c2g, _, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    for n in c2g[0]:
        assert "bias" not in n.model_attributes


# ---------------------------------------------------------------------------
# Per-axon input neurons live on the consuming core
# ---------------------------------------------------------------------------


def test_build_network_input_neuron_per_input_axon_on_same_core(monkeypatch):
    """Each ``is_input_`` axon yields one input neuron on the consuming core."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=3, neurons_per_core=2,
        axon_sources=[
            SpikeSource(-1, 5, True, False, False),   # input[5] at axon 0
            SpikeSource(-1, 1, True, False, False),   # input[1] at axon 1
            SpikeSource(-1, 0, False, True, False),   # off
        ],
        core_matrix=np.asarray([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        arch, _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    assert set(ci.keys()) == {(0, 0), (0, 1)}
    # Each input neuron is on the same SANA-FE core as core 0.
    expected_core = arch.tiles[0].cores[0]
    for neuron in ci.values():
        assert neuron.mapped_core is expected_core
    # Soma hw uses the LOCAL axon index, not the logical input index.
    assert ci[(0, 0)].soma_hw_name == f"{SOMA_INPUT_RANGE_NAME}[0]"
    assert ci[(0, 1)].soma_hw_name == f"{SOMA_INPUT_RANGE_NAME}[1]"


def test_build_network_no_input_neurons_when_no_input_axons(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    _, _, ci, ao = build_network_for_segment(
        _fake_arch(1),
        _fake_hcm(_fake_hard_core(axons_per_core=1, neurons_per_core=1)),
        tile_offset=0, core_offset=0,
    )
    assert ci == {}
    assert ao == {}


def test_build_network_two_hard_cores_each_with_own_input_neurons(monkeypatch):
    """Two HardCores reading the same logical input index get separate neurons."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2)
    core_a = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(-1, 7, True, False, False)],
        core_matrix=np.asarray([[1.0]], dtype=np.float32),
    )
    core_b = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(-1, 7, True, False, False)],   # same logical input
        core_matrix=np.asarray([[1.0]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        arch, _fake_hcm(core_a, core_b),
        tile_offset=0, core_offset=0,
    )
    assert (0, 0) in ci and (1, 0) in ci
    assert ci[(0, 0)].mapped_core is arch.tiles[0].cores[0]
    assert ci[(1, 0)].mapped_core is arch.tiles[0].cores[1]
    assert ci[(0, 0)] is not ci[(1, 0)]


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def test_build_network_input_axon_synapse_carries_weight_and_hw_name(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=2,
        axon_sources=[SpikeSource(-1, 3, True, False, False)],
        core_matrix=np.asarray([[1.5, 2.5]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    src = ci[(0, 0)]
    conns = [(c.index, w) for c, w in src.connections if c.group is c2g[0]]
    assert len(conns) == 2
    weights_by_idx = {idx: attrs for idx, attrs in conns}
    for attrs in weights_by_idx.values():
        assert attrs["synapse_hw_name"] == SYNAPSE_NAME
    assert weights_by_idx[0]["weight"] == pytest.approx(1.5)
    assert weights_by_idx[1]["weight"] == pytest.approx(2.5)


def test_build_network_cross_core_axon_wires_from_source_core_neuron(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    src_core = _fake_hard_core(axons_per_core=1, neurons_per_core=2)
    dst_core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 1, False, False, False)],   # core 0, neuron 1
        core_matrix=np.asarray([[3.0]], dtype=np.float32),
    )
    _, c2g, _, _ = build_network_for_segment(
        _fake_arch(2), _fake_hcm(src_core, dst_core),
        tile_offset=0, core_offset=0,
    )
    src_neuron = c2g[0][1]
    assert any(
        c.group is c2g[1] and c.index == 0 and w["weight"] == pytest.approx(3.0)
        for c, w in src_neuron.connections
    )


def test_build_network_off_axons_are_skipped(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=2, neurons_per_core=1,
        axon_sources=[
            SpikeSource(-1, 0, False, True, False),     # off
            SpikeSource(-1, 0, True, False, False),     # input[0]
        ],
        core_matrix=np.asarray([[99.0], [1.0]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    # Only one input neuron, only one synapse, only the input[0] weight.
    assert set(ci.keys()) == {(0, 1)}
    weights = [w["weight"] for c, w in ci[(0, 1)].connections if c.group is c2g[0]]
    assert weights == [pytest.approx(1.0)]


def test_build_network_always_on_axon_wires_from_local_always_on_neuron(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],
        core_matrix=np.asarray([[7.0]], dtype=np.float32),
    )
    _, c2g, _, ao = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    assert 0 in ao
    weights = [w["weight"] for c, w in ao[0].connections if c.group is c2g[0]]
    assert weights == [pytest.approx(7.0)]


def test_build_network_duplicate_axons_to_same_source_accumulate_weights(monkeypatch):
    """Two axons reading the same logical input → one synapse, summed weight."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=2, neurons_per_core=1,
        axon_sources=[
            SpikeSource(-1, 0, True, False, False),
            SpikeSource(-1, 0, True, False, False),
        ],
        core_matrix=np.asarray([[2.0], [3.0]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    # Two separate input neurons (one per axon), each with its own synapse.
    src_a, src_b = ci[(0, 0)], ci[(0, 1)]
    w_a = [w["weight"] for c, w in src_a.connections if c.group is c2g[0]]
    w_b = [w["weight"] for c, w in src_b.connections if c.group is c2g[0]]
    assert w_a == [pytest.approx(2.0)]
    assert w_b == [pytest.approx(3.0)]


def test_build_network_zero_weights_dont_create_synapses(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=2,
        axon_sources=[SpikeSource(-1, 0, True, False, False)],
        core_matrix=np.asarray([[0.0, 4.0]], dtype=np.float32),
    )
    _, c2g, ci, _ = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    targets = [(c.index, w["weight"]) for c, w in ci[(0, 0)].connections if c.group is c2g[0]]
    assert targets == [(1, pytest.approx(4.0))]


# ---------------------------------------------------------------------------
# Per-sample spike-train injection
# ---------------------------------------------------------------------------


def test_set_input_spike_trains_uses_logical_input_index(monkeypatch):
    """The runner-side spike feed must look up encoded[0, k, :] where k = src.neuron_."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=2, neurons_per_core=1,
        axon_sources=[
            SpikeSource(-1, 5, True, False, False),   # axon 0 reads input[5]
            SpikeSource(-1, 2, True, False, False),   # axon 1 reads input[2]
        ],
        core_matrix=np.ones((2, 1), dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, _, ci, _ = build_network_for_segment(
        _fake_arch(1), hcm,
        tile_offset=0, core_offset=0,
    )
    encoded = np.zeros((1, 8, 4), dtype=np.float32)
    encoded[0, 5, :] = [1, 0, 1, 0]
    encoded[0, 2, :] = [0, 1, 1, 0]
    set_input_spike_trains(ci, hcm, encoded)
    assert ci[(0, 0)].model_attributes["spikes"] == [1, 0, 1, 0]
    assert ci[(0, 1)].model_attributes["spikes"] == [0, 1, 1, 0]


def test_set_always_on_spike_trains_fires_every_cycle(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],
        core_matrix=np.asarray([[1.0]], dtype=np.float32),
    )
    _, _, _, ao = build_network_for_segment(
        _fake_arch(1), _fake_hcm(core),
        tile_offset=0, core_offset=0,
    )
    set_always_on_spike_trains(ao, T=5)
    assert ao[0].model_attributes["spikes"] == [1, 1, 1, 1, 1]
