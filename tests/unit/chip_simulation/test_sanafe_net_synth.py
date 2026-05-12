"""SANA-FE Network synthesis from one ``HardCoreMapping`` neural segment.

``build_network_for_segment`` walks every HardCore in a segment and:

  * creates one SANA-FE neuron group per core,
  * sets per-neuron attributes from ``HardCore.threshold`` and (when present)
    ``HardCore.hardware_bias``,
  * resolves each axon source (input / always-on / off / another core) and
    wires weighted synapses from the source neuron(s) into this core,
  * maps every neuron to the SANA-FE core at the position the spec assigned.

The tests use a lightweight fake ``sanafe`` module so they run without
SANA-FE installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mimarsinan.chip_simulation.sanafe import net_synth
from mimarsinan.chip_simulation.sanafe.net_synth import build_network_for_segment
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


# ---------------------------------------------------------------------------
# Fake SANA-FE module
# ---------------------------------------------------------------------------


class _FakeNeuron:
    def __init__(self, group, index):
        self.group = group
        self.index = index
        self.attributes: dict = {}
        self.connections: list[tuple[_FakeNeuron, dict]] = []
        self.mapped_core = None

    def connect_to_neuron(self, dst, attrs):
        self.connections.append((dst, dict(attrs)))

    def set_attributes(self, model_attributes=None, log_spikes=False,
                       log_potential=False):
        if model_attributes:
            self.attributes.update(model_attributes)
        self.attributes.setdefault("_log_spikes", log_spikes)
        self.attributes.setdefault("_log_potential", log_potential)

    def map_to_core(self, core):
        self.mapped_core = core


class _FakeGroup:
    def __init__(self, name, size, model_attributes=None):
        self.name = name
        self.size = size
        self.model_attributes = dict(model_attributes or {})
        self.neurons = [_FakeNeuron(self, i) for i in range(size)]

    def __iter__(self):
        return iter(self.neurons)

    def __getitem__(self, i):
        return self.neurons[i]

    def __len__(self):
        return self.size


class _FakeNetwork:
    def __init__(self):
        self.groups: list[_FakeGroup] = []

    def create_neuron_group(self, name, size, model_attributes=None):
        g = _FakeGroup(name, size, model_attributes=model_attributes)
        self.groups.append(g)
        return g


def _fake_sanafe_module():
    """Return an object that looks like the public ``sanafe`` module surface."""
    return SimpleNamespace(Network=_FakeNetwork)


def _fake_arch(*tile_core_counts: int):
    """Architecture-shaped object: tiles[t].cores[c] indexable."""
    tiles = []
    for t, n in enumerate(tile_core_counts):
        cores = []
        for c in range(n):
            cores.append(SimpleNamespace(_tile_index=t, _core_index=c))
        tiles.append(SimpleNamespace(cores=cores))
    return SimpleNamespace(tiles=tiles)


# ---------------------------------------------------------------------------
# Fake HardCore / HardCoreMapping
# ---------------------------------------------------------------------------


def _fake_hard_core(
    *,
    axons_per_core: int,
    neurons_per_core: int,
    available_axons: int = 0,
    available_neurons: int = 0,
    threshold: float = 1.0,
    hardware_bias: np.ndarray | None = None,
    core_matrix: np.ndarray | None = None,
    axon_sources: list | None = None,
):
    """Build a HardCore-shaped fake exposing the fields net_synth queries."""
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


def _fake_hcm(*cores) -> SimpleNamespace:
    return SimpleNamespace(cores=list(cores))


# ---------------------------------------------------------------------------
# Group-creation tests
# ---------------------------------------------------------------------------


def test_build_network_creates_neuron_group_per_hard_core(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2)

    core_a = _fake_hard_core(axons_per_core=2, neurons_per_core=3)
    core_b = _fake_hard_core(axons_per_core=2, neurons_per_core=1)
    hcm = _fake_hcm(core_a, core_b)

    net, core_to_group, input_group, always_on_group = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=4,
    )

    assert len(core_to_group) == 2
    assert all(g in net.groups for g in core_to_group.values())
    assert core_to_group[0].size == 3   # used neuron count for core_a
    assert core_to_group[1].size == 1


def test_build_network_input_neuron_group_size_matches_seg_in_size(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    hcm = _fake_hcm(_fake_hard_core(axons_per_core=2, neurons_per_core=1))

    _, _, input_group, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=5,
    )
    assert input_group is not None
    assert input_group.size == 5


def test_build_network_always_on_group_omitted_when_no_always_on_axons(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    hcm = _fake_hcm(_fake_hard_core(axons_per_core=2, neurons_per_core=1))
    _, _, _, always_on = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=2,
    )
    assert always_on is None


def test_build_network_always_on_group_size_one_when_used(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],   # always_on axon
        core_matrix=np.asarray([[2.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, _, _, always_on = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    assert always_on is not None
    assert always_on.size == 1


# ---------------------------------------------------------------------------
# Per-neuron attribute tests
# ---------------------------------------------------------------------------


def test_build_network_threshold_per_neuron_from_core_threshold(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2)
    core_a = _fake_hard_core(axons_per_core=1, neurons_per_core=2, threshold=1.5)
    core_b = _fake_hard_core(axons_per_core=1, neurons_per_core=2, threshold=4.0)
    hcm = _fake_hcm(core_a, core_b)

    _, core_to_group, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    for n in core_to_group[0].neurons:
        assert n.attributes["threshold"] == 1.5
    for n in core_to_group[1].neurons:
        assert n.attributes["threshold"] == 4.0


def test_build_network_hardware_bias_propagated(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    bias = np.asarray([0.5, -0.25, 1.0], dtype=np.float32)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=3, hardware_bias=bias,
    )
    hcm = _fake_hcm(core)
    _, core_to_group, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    for i, n in enumerate(core_to_group[0].neurons):
        assert n.attributes.get("bias") == pytest.approx(float(bias[i]))


def test_build_network_no_bias_attribute_when_hardware_bias_is_none(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(axons_per_core=1, neurons_per_core=2, hardware_bias=None)
    hcm = _fake_hcm(core)
    _, core_to_group, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    for n in core_to_group[0].neurons:
        assert "bias" not in n.attributes


# ---------------------------------------------------------------------------
# Mapping tests
# ---------------------------------------------------------------------------


def test_build_network_maps_neurons_to_correct_cores(monkeypatch):
    """Each neuron's ``map_to_core`` lands on tile/core determined by tile_offset
    and the per-segment cores_per_tile packing."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(3, 3)         # 2 tiles, each 3 cores
    cores = [
        _fake_hard_core(axons_per_core=1, neurons_per_core=1) for _ in range(4)
    ]
    hcm = _fake_hcm(*cores)
    _, core_to_group, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
        cores_per_tile=3,
    )
    # First 3 cores land on tile 0 cores 0..2; 4th lands on tile 1 core 0.
    assert core_to_group[0].neurons[0].mapped_core is arch.tiles[0].cores[0]
    assert core_to_group[1].neurons[0].mapped_core is arch.tiles[0].cores[1]
    assert core_to_group[2].neurons[0].mapped_core is arch.tiles[0].cores[2]
    assert core_to_group[3].neurons[0].mapped_core is arch.tiles[1].cores[0]


def test_build_network_tile_offset_skips_leading_tiles(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2, 2)
    core = _fake_hard_core(axons_per_core=1, neurons_per_core=1)
    hcm = _fake_hcm(core)
    _, c2g, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=1, core_offset=0, seg_in_size=1,
    )
    assert c2g[0].neurons[0].mapped_core is arch.tiles[1].cores[0]


# ---------------------------------------------------------------------------
# Axon-source resolution tests
# ---------------------------------------------------------------------------


def test_build_network_input_axon_wires_from_input_group(monkeypatch):
    """An ``is_input_`` axon wires the input neuron group's neuron[k] to this core."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=2,
        axon_sources=[SpikeSource(-1, 3, True, False, False)],   # input[3]
        core_matrix=np.asarray([[1.5, 2.5]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, c2g, input_group, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=4,
    )
    # Input neuron 3 has two outgoing synapses (one per destination neuron).
    src = input_group[3]
    targets = {(c.group is c2g[0], c.index): w["weight"] for c, w in src.connections}
    assert targets == {(True, 0): pytest.approx(1.5), (True, 1): pytest.approx(2.5)}


def test_build_network_core_axon_wires_from_source_core_neuron(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(2)
    src_core = _fake_hard_core(axons_per_core=1, neurons_per_core=2)
    dst_core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 1, False, False, False)],   # core 0 neuron 1
        core_matrix=np.asarray([[3.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(src_core, dst_core)
    _, c2g, _, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    src_neuron = c2g[0][1]
    assert any(
        c.group is c2g[1] and c.index == 0 and w["weight"] == pytest.approx(3.0)
        for c, w in src_neuron.connections
    )


def test_build_network_off_axons_are_skipped(monkeypatch):
    """``is_off_`` axons emit no synapses regardless of core_matrix value."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=2, neurons_per_core=1,
        axon_sources=[
            SpikeSource(-1, 0, False, True, False),       # off
            SpikeSource(-1, 0, True, False, False),       # input[0]
        ],
        core_matrix=np.asarray([[99.0], [1.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, c2g, input_group, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    # Only input[0] should connect to core 0 neuron 0; the off axon contributes nothing.
    conns = input_group[0].connections
    weights = [c[1]["weight"] for c in conns if c[0].group is c2g[0]]
    assert weights == [pytest.approx(1.0)]


def test_build_network_always_on_axon_wires_from_always_on_neuron(monkeypatch):
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=1,
        axon_sources=[SpikeSource(0, 0, False, False, True)],
        core_matrix=np.asarray([[7.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, c2g, _, always_on = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    assert always_on is not None
    on_neuron = always_on[0]
    weights = [w["weight"] for c, w in on_neuron.connections if c.group is c2g[0]]
    assert weights == [pytest.approx(7.0)]


def test_build_network_duplicate_axons_accumulate_weights(monkeypatch):
    """Two axons reading the same source produce one synapse whose weight sums."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=2, neurons_per_core=1,
        axon_sources=[
            SpikeSource(-1, 0, True, False, False),   # input[0]
            SpikeSource(-1, 0, True, False, False),   # input[0] again
        ],
        core_matrix=np.asarray([[2.0], [3.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, c2g, input_group, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    src = input_group[0]
    conns_to_dst = [w["weight"] for c, w in src.connections if c.group is c2g[0]]
    # Exactly one synapse whose weight is the SUM of the two axon entries.
    assert len(conns_to_dst) == 1
    assert conns_to_dst[0] == pytest.approx(5.0)


def test_build_network_zero_weights_dont_create_synapses(monkeypatch):
    """A zero entry in ``core_matrix`` must not produce a wasteful synapse."""
    monkeypatch.setattr(net_synth, "_sanafe", _fake_sanafe_module)
    arch = _fake_arch(1)
    core = _fake_hard_core(
        axons_per_core=1, neurons_per_core=2,
        axon_sources=[SpikeSource(-1, 0, True, False, False)],
        core_matrix=np.asarray([[0.0, 4.0]], dtype=np.float32),
    )
    hcm = _fake_hcm(core)
    _, c2g, input_group, _ = build_network_for_segment(
        arch, hcm, tile_offset=0, core_offset=0, seg_in_size=1,
    )
    conns = input_group[0].connections
    # Only neuron 1 should receive a synapse — neuron 0 had weight 0.
    targets = [(c.index, w["weight"]) for c, w in conns if c.group is c2g[0]]
    assert targets == [(1, pytest.approx(4.0))]
