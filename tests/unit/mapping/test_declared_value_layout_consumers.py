"""[U-PCH3] The consumers of a per-node value outside the static check, each reading it by its holder's declared layout
(``mapping.support.value_layouts``), one test each.

- A depth-balancing relay takes its producer's per-channel activation scale at the producer's channels of the relayed
  columns (a tile past channel 0 once gave it the vector's first entries).
- Weight Quantization reads a per-channel weight and bias scale at each core's channels without writing them back, so
  every node keeps its declared layout (a node with bias rows once kept a per-column bias scale).
- The ODIN emission bound refuses a per-neuron threshold by name and keeps its own refusal of a boolean or non-numeric
  one (the verify's m-8).
- The GUI snapshot's threshold range is each group's own registers' extremes (the verify's m-2).
- The arming reads a one-entry weight scale as one grid per perceptron (the verify's m-10).
- The decode and seam readers never read a neural producer's per-channel vector by position: a neural producer has no
  wire divisor and a host stage collapses a per-channel entry by its mean (the premise of the verify's m-7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping

from unit.mapping.residency.executed import vehicles as V


def _channel_scales(rep):
    for p in rep.get_perceptrons():
        p.set_activation_scale(torch.tensor([1.0 + 0.125 * c for c in range(int(p.layer.weight.shape[0]))]))
    return rep


def test_a_depth_relay_reads_its_producers_per_channel_scale_at_the_producers_channels():
    from mimarsinan.mapping.latency.depth_balancing import RELAY_MARKER, insert_depth_balancing_relays

    checked = 0
    for seed in range(12):
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=4).map(
            _channel_scales(V.random_rep(seed)))
        insert_depth_balancing_relays(ir, thresholding_mode="<=")
        by_id = {int(n.id): n for n in ir.nodes}
        for relay in ir.get_neural_cores():
            sources = list(np.asarray(relay.input_sources).flatten())
            producer = by_id.get(int(sources[0].node_id)) if getattr(relay, RELAY_MARKER, False) else None
            if not isinstance(producer, NeuralCore) or producer.perceptron_output_slice is None:
                continue
            start = int(producer.perceptron_output_slice[0])
            vector = torch.as_tensor(producer.activation_scale).reshape(-1)
            want = vector if vector.numel() == 1 else vector[[start + int(s.index) for s in sources]]
            assert torch.equal(torch.as_tensor(relay.activation_scale).reshape(-1), want), (seed, relay.name)
            checked += int(start > 0)
    assert checked, "no relay of a producer tile past channel 0 among the seeds"


def _tiles_with_bias_rows():
    """A perceptron of 8 output channels on two nodes of 4 (channels 0 to 3 and 4 to 7), its bias on always-on rows
    (no hardware bias), projected per neuron: ``s_c = 40 + c``, ``bias_scale_c = s_c / 2``."""
    rng = np.random.default_rng(3)
    scale = torch.tensor([40.0 + c for c in range(8)])
    nodes = []
    for index, (lo, hi) in enumerate(((0, 4), (4, 8))):
        sources = np.array([IRSource(-2, i) for i in range(3)] + [IRSource(-3, 0)], dtype=object)
        nodes.append(NeuralCore(id=index, name=f"tile{index}", input_sources=sources,
                                core_matrix=rng.normal(size=(4, 4)) * 0.01, perceptron_index=0,
                                perceptron_output_slice=(lo, hi), parameter_scale=scale.clone(),
                                bias_scale=scale.clone() / 2.0, latency=0))
    ir = IRGraph(nodes=nodes, output_sources=np.array([IRSource(n.id, j) for n in nodes for j in range(4)],
                                                      dtype=object))
    return ir, scale


def test_weight_quantization_reads_per_channel_scales_at_each_cores_channels_and_never_writes_them_back():
    ir, scale = _tiles_with_bias_rows()
    quantize_ir_graph(ir, 8, weight_quantization=True)
    for node in ir.get_neural_cores():
        lo, hi = node.perceptron_output_slice
        assert np.array_equal(np.asarray(node.threshold), scale[lo:hi].numpy())  # theta_c = s_c at its channels
        assert float(node.parameter_scale) == 1.0
        assert torch.equal(torch.as_tensor(node.bias_scale), scale / 2.0)  # its declared layout: per channel, whole


def test_the_odin_emission_bound_keeps_its_own_refusal_of_a_boolean_or_non_numeric_threshold():
    from mimarsinan.mapping.export.odin.feasibility import OdinFeasibilityError, propagate_emission_bounds

    from unit.mapping.test_odin_exporter import _two_core_mapping

    for held in (True, "seven"):
        mapping = _two_core_mapping()
        mapping.cores[0].threshold = held
        with pytest.raises(OdinFeasibilityError, match="threshold"):
            propagate_emission_bounds(mapping, ceiling=255)


def test_the_gui_snapshot_gives_each_group_its_own_registers_range():
    """Each node of 3 columns starts at a channel ``c`` with ``c mod 3 == 0``, whose scale (50) is neither extreme of
    its perceptron's (40 and 70), so a range read off each core's first register reads another range."""
    from mimarsinan.gui.snapshot.ir_graph import snapshot_ir_graph

    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=3).map(V.random_rep(0))
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        width = max(int(n.perceptron_output_slice[1]) for n in ir.get_neural_cores()
                    if n.perceptron_index == owner.perceptron_index)
        owner.parameter_scale = torch.tensor([(50.0, 70.0, 40.0)[c % 3] for c in range(width)])
    quantize_ir_graph(ir, 8, weight_quantization=True)
    registers = {int(n.id): np.atleast_1d(np.asarray(n.threshold, dtype=np.float64)) for n in ir.get_neural_cores()}
    summary, _ = snapshot_ir_graph(ir)
    groups = [g for g in summary["groups"] if g.get("threshold_range") is not None]
    assert groups
    off_first = 0
    for group in groups:
        mine = np.concatenate([registers[i] for i in group["node_ids"] if i in registers])
        firsts = [float(registers[i][0]) for i in group["node_ids"] if i in registers]
        assert group["threshold_range"] == [float(mine.min()), float(mine.max())], group["node_ids"]
        off_first += int([min(firsts), max(firsts)] != [float(mine.min()), float(mine.max())])
    assert off_first, "the fixture puts some group's extreme off every core's first register"


def test_the_arming_reads_a_one_entry_weight_scale_as_one_grid():
    from mimarsinan.models.nn.lif_kernels import NeuronLattice, integral_lattice
    from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice

    assert integral_lattice(torch.tensor([7.0])) == 7  # one entry: every neuron's grid
    assert isinstance(integral_lattice(torch.tensor([3.0, 5.0])), NeuronLattice)  # one per neuron
    rep = V.random_rep(0)
    perceptron = next(p for p in rep.get_perceptrons()  # a core perceptron (an encoder's lattice has its own rule)
                      if int(p.layer.weight.shape[0]) > 1 and not getattr(p, "is_encoding_layer", False))
    perceptron.set_parameter_scale(torch.tensor([7.0]))
    from mimarsinan.models.perceptron_mixer.membrane_lattice import record_membrane_lattice_kind

    record_membrane_lattice_kind(perceptron)  # [U-POST2] the projection's record; the arming derives the lattice
    arm_integer_membrane_lattice(rep)
    from mimarsinan.spiking.lif_utils import unwrap_lif_activation

    lattice = unwrap_lif_activation(perceptron.activation).if_node.lattice_scale
    assert not isinstance(lattice, NeuronLattice) and float(lattice) > 0


def test_no_decode_or_seam_reader_reads_a_neural_producers_vector_by_position():
    from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import resolve_stage_compute_scales
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph as quantize
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.spiking.segment_boundary import boundary_normalization_scales

    found = 0
    for seed in range(8):
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=4).map(
            _channel_scales(V.random_rep(seed)))
        for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
            owner.parameter_scale = torch.tensor(0.0)
        quantize(ir, 8, weight_quantization=True)
        IRLatency(ir).calculate()
        program = V.build(ir, 4, None)
        neural = {int(n.id) for n in ir.get_neural_cores()}
        assert not set(boundary_normalization_scales(program)) & neural, seed
        for stage in program.stages:
            if stage.kind == "compute" and stage.compute_op is not None:
                found += 1
                scales = resolve_stage_compute_scales(program, int(stage.compute_op.id), op=stage.compute_op)
                assert all(isinstance(s, float) for s in scales), seed
    assert found, "no host stage among the seeds"
