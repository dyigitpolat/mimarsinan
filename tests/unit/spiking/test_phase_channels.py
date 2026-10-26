"""[E1] The phase-dithered Uniform encode keys every channel's comb rotation on its IR neuron at unbounded geometry.

A stage-input channel is a neuron of an IR node; its channel at unbounded geometry is the neuron's index in its
perceptron's output (the node's ``perceptron_output_slice`` start plus its column before compaction), a host op's or
the raw input's output index otherwise. So no cost-only decision (the IR's tiling at the declared geometry, its node ids
and order, packing, pass cuts) moves a dithered train. Before E1 the rotation was keyed on the channel's position in
the executed unit's stage input, whose layout orders external nodes by IR node id: the X3a R4 arm found 26 of 100
admitted windowed-LIF deployments differing from the identity program at unbounded geometry (closure fix round), all
26 from the relabelled node ids (``RA/11_wave2/e1/diag/r4_root_cause.jsonl``)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from mimarsinan.certification.randomized.arms import ARMS
from mimarsinan.certification.randomized.cost_draws import platform_constraints
from mimarsinan.certification.randomized.runner import (
    EQUAL, _run, compare, deployment, instance_for, reference, run_instance, targeted_declaration)
from mimarsinan.chip_simulation.recording.spike_modes import to_uniform_spikes, uniform_phase_offsets
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.segment_boundary import encode_segment_input
from mimarsinan.spiking.spike_trains import uniform_spike_train

#: The closure fix round's R4 counterexamples (``closure/fix_round/gates/screen/R4.jsonl``), reproduced on the
#: exactness tree at 216 of 216 seeds (``e1/repro/screen_exact_tree``).
R4_COUNTEREXAMPLES = (7, 25, 28, 51, 53, 62, 63, 76, 79, 82, 85, 90, 92, 93, 102, 103, 105, 108, 110, 111, 114, 151,
                      155, 167, 194, 213)
T = 16


def _dithered_arm():
    """The R4 arm's draws with the dithered encode on both programs and nothing switched off."""
    return replace(ARMS["R4"], id="R4/keyed", switch=None)


def _runs(seed: int, *, relabel: bool = True):
    arm = _dithered_arm()
    instance = instance_for(seed, arm)
    constraints = platform_constraints(targeted_declaration(seed, arm), hardware_bias=instance.hardware_bias)
    ref = reference(instance, arm)
    ir, mapping = deployment(instance, constraints, seed, arm, relabel=relabel)
    with torch.no_grad():
        nodes = _run(mapping, instance, arm)
    return compare(ref.ir, ref.nodes, ir, nodes)


@pytest.mark.parametrize("seed", R4_COUNTEREXAMPLES)
def test_a_dithered_deployment_equals_the_identity_program_at_unbounded_geometry(seed):
    """The 26 counterexamples (windowed LIF, the IR at the declared geometry, relabelled): every neuron equal at
    atol 0 to the identity program of the IR at unbounded geometry, on the arm's inputs."""
    windows, mismatches, first = _runs(seed)
    assert windows > 0 and mismatches == 0, (seed, mismatches, first)


@pytest.mark.parametrize("seed", R4_COUNTEREXAMPLES[:6])
def test_the_r4_arms_paired_control_runs_equal(seed):
    assert run_instance(seed, "R4", control=True).outcome == EQUAL


def _tiled_ir(*, keyed: bool = True, pruned=None) -> IRGraph:
    """Input (4) -> perceptron 0 at 6 neurons, tiled 4 + 2 (slices (0, 4) and (4, 6)) -> perceptron 1 reading the second
    tile only, so the second tile sits at offset 0 of the windowed level unit that reads it."""
    rng = np.random.default_rng(3)
    x = np.array([IRSource(-2, i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
    w0 = np.round(rng.normal(size=(5, 4)) * 3)
    w1 = np.round(rng.normal(size=(5, 2)) * 3)
    t1 = np.array([IRSource(1, j) for j in range(2)] + [IRSource(-3, 0)], dtype=object)
    w2 = np.round(rng.normal(size=(3, 3)) * 3)
    nodes = [NeuralCore(id=0, name="p0_t0", input_sources=x, core_matrix=w0, threshold=3.0, perceptron_index=0,
                        perceptron_output_slice=(0, 4)),
             NeuralCore(id=1, name="p0_t1", input_sources=x, core_matrix=w1, threshold=3.0,
                        perceptron_index=0 if keyed else None, perceptron_output_slice=(4, 6)),
             NeuralCore(id=2, name="p1", input_sources=t1, core_matrix=w2, threshold=2.0, perceptron_index=1,
                        perceptron_output_slice=(0, 3))]
    if pruned is not None:
        nodes[1].pre_pruning_col_mask = list(pruned)
    ir = IRGraph(nodes=nodes, output_sources=np.array([IRSource(0, j) for j in range(4)] +
                                                      [IRSource(2, j) for j in range(3)], dtype=object))
    IRLatency(ir).calculate()
    return ir


def _level_reading(mapping, node: int):
    for stage in mapping.stages:
        for unit in getattr(stage, "retimed_level_stages", None) or ():
            if [int(s.node_id) for s in unit.input_map] == [node]:
                return unit
    raise AssertionError(f"no level unit reads node {node} alone")


def _encode(mapping, unit, rates):
    config = BoundaryConfig(simulation_length=T, spiking_mode="lif", cycle_accurate=True, phase_dither=True)
    return encode_segment_input(unit, rates, {}, config=config, hybrid_mapping=mapping, T=T,
                                batch_size=rates.shape[0], device=torch.device("cpu"))


def _comb(rates, channels):
    offsets = uniform_phase_offsets(int(max(channels)) + 1, T)[torch.as_tensor(channels)]
    return torch.stack([to_uniform_spikes(rates.clamp(0.0, 1.0), c, T, phase_offsets=offsets)
                        for c in range(T)]).to(torch.float64)


def test_a_tile_channel_is_rotated_by_its_perceptron_channel():
    """The second tile's neurons are perceptron 0's channels 4 and 5: their combs rotate by those channels, not by
    their position 0 and 1 in the level unit's stage input."""
    ir = _tiled_ir()
    mapping = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    unit = _level_reading(mapping, 1)
    rates = torch.rand(5, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 0.8 + 0.1
    got = _encode(mapping, unit, rates)
    assert torch.equal(got, _comb(rates, [4, 5]))
    assert not torch.equal(got, _comb(rates, [0, 1]))


def test_a_compacted_tile_keeps_its_perceptron_channels():
    """The second tile compacted from three columns (its middle one pruned): its two columns are channels 4 and 6."""
    ir = _tiled_ir()
    node = ir.nodes[1]
    node.perceptron_output_slice = (4, 7)
    node.pre_pruning_col_mask = [False, True, False]
    mapping = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    rates = torch.rand(3, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.8 + 0.1
    assert torch.equal(_encode(mapping, _level_reading(mapping, 1), rates), _comb(rates, [4, 6]))


def test_a_dithered_encode_of_a_neuron_without_an_unbounded_channel_refuses_by_name():
    """A node with no perceptron identity (a relay, an inserted core) has no channel at unbounded geometry: a dithered
    encode of it refuses by name instead of keying on where a cost-only decision put it."""
    ir = _tiled_ir(keyed=False)
    mapping = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    rates = torch.rand(2, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    with pytest.raises(ToolchainRefusal, match="channel at unbounded geometry"):
        _encode(mapping, _level_reading(mapping, 1), rates)


def test_a_channel_at_its_own_position_encodes_as_before():
    """Where a channel's position is its channel (a producer's own output, the NF twin's and the host boundary's
    encode), the keyed rotation is the position rotation bit for bit."""
    rates = torch.rand(3, 7, 40, generator=torch.Generator().manual_seed(4))
    assert torch.equal(uniform_spike_train(rates, T, phase_dither=True),
                       uniform_spike_train(rates, T, phase_dither=True, channels=torch.arange(40)))


def test_the_tiled_and_the_unbounded_program_key_every_neuron_alike():
    """X3a seed 7 (three perceptrons tiled at the declared geometry, the IR relabelled): every neuron of the tiled
    program carries the channel the unbounded program carries for the same perceptron output."""
    from mimarsinan.mapping.packing.phase_channels import channels_of

    arm = _dithered_arm()
    instance = instance_for(7, arm)
    constraints = platform_constraints(targeted_declaration(7, arm), hardware_bias=instance.hardware_bias)
    ref = reference(instance, arm)
    ir, mapping = deployment(instance, constraints, 7, arm)

    def by_output(graph, program):
        out = {}
        for n in graph.get_neural_cores():
            width = n.get_core_matrix(graph).shape[1]
            for k, c in enumerate(channels_of(program, int(n.id), width).tolist()):
                out[(int(n.perceptron_index), int(n.perceptron_output_slice[0]) + k)] = c
        return out

    tiled, unbounded = by_output(ir, mapping), by_output(ref.ir, ref.mapping)
    assert tiled == unbounded and all(c == channel for (_p, channel), c in tiled.items())
    assert any(int(n.perceptron_output_slice[0]) > 0 for n in ir.get_neural_cores())
