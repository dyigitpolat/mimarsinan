"""Vehicles of position multiplexing: an integer value CNN (G2's law) and a seeded LIF conv chain, built on both
planes; one helper builds the replicated (undeclared) and the multiplexed program of either."""

from __future__ import annotations

from typing import List

import torch

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

from unit.chip_simulation.value_run.integer_programs import _randomize_bn, _ResNet, build_integer_program

PASSES = 4096
T = 8
LIF_SHAPE = (2, 5, 5)


def integer_resnet(shape=(3, 8, 8)):
    """A two-stage CIFAR ResNet (8 and 16 channels, one block each, a projection shortcut) as an integer program."""
    torch.manual_seed(13)
    return build_integer_program(_randomize_bn(_ResNet((8, 16), blocks=1)), shape, num_classes=10)


def _lif_conv(source, cin: int, cout: int, *, encoding: bool = False) -> Conv2DPerceptronMapper:
    mapper = Conv2DPerceptronMapper(source, cin, cout, 3, padding=1, use_batchnorm=False)
    perceptron = mapper.perceptron
    perceptron.base_activation = LIFActivation(T=T, activation_scale=torch.tensor(1.0))
    perceptron.activation = perceptron.base_activation
    with torch.no_grad():
        perceptron.layer.weight.copy_((torch.rand_like(perceptron.layer.weight) - 0.5) * 1.2)
        if perceptron.layer.bias is not None:
            perceptron.layer.bias.fill_(0.15)
    if encoding:
        perceptron.is_encoding_layer = True
        perceptron.use_cycle_accurate_trains = True
    return mapper


def lif_conv_chain(seed: int = 0, widths=(6, 6)) -> ModelRepresentation:
    """Encode on the host, then one LIF conv per width on chip: one segment of len(widths) dependency levels."""
    torch.manual_seed(seed)
    node = _lif_conv(InputMapper(LIF_SHAPE), LIF_SHAPE[0], 4, encoding=True)
    cin = 4
    for width in widths:
        node = _lif_conv(node, cin, width)
        cin = width
    rep = ModelRepresentation(node)
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


def lif_ir(rep: ModelRepresentation, max_axons: int, max_neurons: int, *, coalescing: bool = False):
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=max_axons, max_neurons=max_neurons,
                   allow_coalescing=coalescing).map(rep)
    IRLatency(ir).calculate()
    return ir


def lif_flow(hybrid, shape=LIF_SHAPE) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(shape, hybrid, simulation_length=T, spiking_mode="lif",
                                 cycle_accurate_lif_forward=True, thresholding_mode="<=")


def caps(multiplexed: bool, **kwargs) -> ChipCapabilities:
    return ChipCapabilities(allow_scheduling=True, max_schedule_passes=PASSES,
                            position_multiplexing=True if multiplexed else None, **kwargs)


def build(ir, cores: List[dict], multiplexed: bool, **kwargs):
    capabilities = caps(multiplexed, **kwargs)
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores, strategy=MappingStrategy.resolve(capabilities),
                                          max_schedule_passes=PASSES)


def ir_runs(ir) -> List[int]:
    """Per layout spec, the builder segment (``partition_ir_graph``) its IR core runs in: the walk's ``runs``."""
    runs = [0] * len(ir.layout_softcores)
    neural_segments = [s for s in partition_ir_graph(ir) if isinstance(s, NeuralSegment)]
    for ordinal, segment in enumerate(neural_segments):
        for node in segment.nodes:
            runs[int(node.layout_softcore_index)] = ordinal
    return runs


def fragments(ir, cores: List[dict], multiplexed: bool, **kwargs):
    """The shape-only plane over the IR's own layout specs, grouped by the builder's segments."""
    types = [LayoutHardCoreType(int(c["max_axons"]), int(c["max_neurons"]), int(c["count"])) for c in cores]
    return collect_noc_fragments(softcores=list(ir.layout_softcores), core_types=types, census=None,
                                 runs=ir_runs(ir), **caps(multiplexed, **kwargs).layout_kwargs())


def neural(hybrid) -> list:
    return [s for s in hybrid.stages if s.kind == "neural"]


def built_banks(hybrid, ir) -> List[tuple]:
    """Per executed pass, the sorted (perceptron, bank) of the IR cores it holds."""
    node = {int(n.id): n for n in ir.get_neural_cores()}
    out = []
    for stage in neural(hybrid):
        ids = {int(p["ir_node_id"]) for ps in stage.hard_core_mapping.soft_core_placements_per_hard_core for p in ps}
        out.append(tuple(sorted({(node[i].perceptron_index, node[i].weight_bank_id) for i in ids},
                                key=lambda t: (t[0], -1 if t[1] is None else t[1]))))
    return out


def planned_banks(frag, specs) -> List[tuple]:
    return [tuple(sorted({(specs[i].perceptron_index, specs[i].bank_id) for i, _ in placements},
                         key=lambda t: (t[0], -1 if t[1] is None else t[1])))
            for placements in frag.pass_placements]
