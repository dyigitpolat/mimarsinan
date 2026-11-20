"""[U-PCH] The chip backends carry per-neuron thresholds (theta_c = s_c) or refuse them by name.

Lava's subtractive LIF process takes a per-neuron threshold override and runs spike for spike as the reference
executor's cycle with each neuron's own threshold; SANA-FE's network synthesis gives each soma its own neuron's
threshold; nevresim's chip model holds a threshold per neuron and emits each; the TTFS executor, which holds one
register per core, refuses a per-neuron core by name. Programs come from the per-neuron Weight Quantization and Soft
Core Mapping sequence of ``unit.mapping.test_per_neuron_weight_scales``.
"""

from __future__ import annotations

from types import SimpleNamespace

from spiking_points import point_of_mode

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.mapping.support.neuron_thresholds import PerNeuronThresholdRefusal, neuron_thresholds
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire

from unit.mapping.test_per_neuron_weight_scales import GEOMETRIES, _program


def _per_neuron_stage_mapping():
    program = _program("pool_downsample", 0, register=True, geometry=GEOMETRIES[1])
    for stage in program.hybrid.stages:
        if stage.kind == "neural":
            mapping = stage.hard_core_mapping
            if any(np.ndim(c.threshold) == 1 and len(set(np.asarray(c.threshold).tolist())) > 1
                   for c in mapping.cores):
                return mapping
    raise AssertionError("the program holds a stage of cores with unequal per-neuron thresholds")


def test_nevresim_emits_each_neurons_threshold():
    from mimarsinan.mapping.export.chip_export import hard_cores_to_chip

    mapping = _per_neuron_stage_mapping()
    axons = max(int(c.axons_per_core) for c in mapping.cores)
    neurons = max(int(c.neurons_per_core) for c in mapping.cores)
    chip = hard_cores_to_chip(64, mapping, axons, neurons, 0, int, threshold_type=int)
    for core, emitted in zip(mapping.cores, chip.cores):
        used = int(core.neurons_per_core - core.available_neurons)
        want = neuron_thresholds(core.threshold, used, int(core.neurons_per_core))
        assert [n.thresh for n in emitted.neurons[:used]] == want.tolist()


def test_the_ttfs_executor_refuses_a_per_neuron_core_by_name():
    from mimarsinan.chip_simulation.ttfs.segment_arrays import segment_ttfs_arrays_from_mapping

    with pytest.raises(PerNeuronThresholdRefusal, match="TTFS"):
        segment_ttfs_arrays_from_mapping(_per_neuron_stage_mapping())


class _FakeNeuron:
    def __init__(self):
        self.attributes = {}

    def set_attributes(self, **kwargs):
        self.attributes.update(kwargs)

    def map_to_core(self, core):
        self.core = core

    def connect_to_neuron(self, dst, attrs):
        pass


class _FakeNetwork:
    def __init__(self):
        self.groups = {}

    def create_neuron_group(self, name, count, model_attributes=None):
        self.groups[name] = [_FakeNeuron() for _ in range(int(count))]
        return self.groups[name]


def test_sanafe_gives_each_soma_its_own_neurons_threshold(monkeypatch):
    import mimarsinan.chip_simulation.sanafe.net_synth as net_synth
    from mimarsinan.chip_simulation.sanafe.net_synth.build import build_network_for_segment

    monkeypatch.setattr(net_synth, "_sanafe", lambda: SimpleNamespace(Network=_FakeNetwork))
    mapping = _per_neuron_stage_mapping()
    arch = SimpleNamespace(tiles=[SimpleNamespace(cores=[object() for _ in range(len(mapping.cores))])])
    net, core_to_group, _, _ = build_network_for_segment(arch, mapping, tile_offset=0, core_offset=0,
                                                         cores_per_tile=len(mapping.cores), simulation_length=4,
                                                         point=point_of_mode("lif", "cascaded"))
    checked = 0
    for core_idx, group in core_to_group.items():
        core = mapping.cores[core_idx]
        want = neuron_thresholds(core.threshold, len(group), int(core.neurons_per_core))
        got = [n.attributes["model_attributes"]["threshold"] for n in group]
        assert got == want.tolist()
        checked += 1
    assert checked > 0


def _reference(weights: np.ndarray, thetas: np.ndarray, spikes: np.ndarray, behavior) -> np.ndarray:
    """The reference executor's cycle with each neuron's own threshold: ``(N, n_in, T)`` -> ``(n_out, N*T)``."""
    n_samples, _, T = spikes.shape
    w = torch.tensor(weights, dtype=torch.float64)
    theta = torch.tensor(thetas, dtype=torch.float64)
    out = np.zeros((weights.shape[0], n_samples * T), dtype=np.float64)
    for i in range(n_samples):
        memb = torch.zeros((1, weights.shape[0]), dtype=torch.float64)
        for cycle in range(T):
            inp = torch.tensor(spikes[i, :, cycle], dtype=torch.float64).reshape(1, -1)
            fired = lif_core_contribute_and_fire(memb, w, inp, theta, hw_bias=None,
                                                 thresholding_mode=behavior.thresholding_mode,
                                                 firing_mode=behavior.firing_mode, output_dtype=torch.float64)
            out[:, i * T + cycle] = fired.reshape(-1).numpy()
    return out


@pytest.mark.slow
def test_lava_runs_each_neuron_on_its_own_threshold(lava_graph_slot):
    from mimarsinan.chip_simulation.lava_loihi import _uniform_rate_encode
    from mimarsinan.chip_simulation.lava_loihi.core_worker import _WaveCoreHost

    behavior = NeuralBehaviorConfig.for_lava({"spiking_mode": "lif", "firing_mode": "Default",
                                              "spike_generation_mode": "Uniform"})
    weights = np.array([[1.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0]], dtype=np.float32)
    thetas = np.array([2.0, 5.0, 3.0, 7.0])
    rates = np.array([[1.0, 0.5, 0.25], [0.5, 0.75, 1.0]], dtype=np.float32)
    T = 8
    spikes = _uniform_rate_encode(rates, T)
    packed = spikes.transpose(1, 0, 2).reshape(weights.shape[1], -1)
    actual = np.asarray(_WaveCoreHost(T, behavior)._run_core_lava(
        weights=weights, threshold=thetas, hardware_bias=None, input_spikes=packed), dtype=np.float64)
    expected = _reference(weights.astype(np.float64), thetas, spikes.astype(np.float64), behavior)
    np.testing.assert_array_equal(actual, expected)
    one_register = _reference(weights.astype(np.float64), np.full(4, 2.0), spikes.astype(np.float64), behavior)
    assert not np.array_equal(expected, one_register), "the fixture's neurons read their own thresholds"
