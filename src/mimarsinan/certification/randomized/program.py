"""An X3a instance as the toolchain maps it: mapper graph, integer IR at a declared geometry, IR relabelling."""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, FrozenSet, Optional

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.certification.randomized.instance import HOST, INPUT, VALUE, Instance, Layer
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.core_residency import ALL_SINGLETON_NAMES, residency_key
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

#: The weight-quantization bound the IR is mapped under; the integer weights already lie on the grid.
_Q_MAX = 127.0


class HalfSum(nn.Module):
    """The host residual add: the sum of the two equal halves of its input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return x[..., :half] + x[..., half:]


_HOST_MODULES = {"identity": nn.Identity, "relu": nn.ReLU, "sum": HalfSum}


@dataclasses.dataclass
class MappedNetwork:
    """The mapper graph and the perceptron index of every neural layer (``theta`` keyed by that index)."""

    representation: ModelRepresentation
    perceptron_of: Dict[str, int]
    theta_of: Dict[int, int]


def _perceptron(width: int, fan_in: int, timesteps: int, weights, bias, *, dtype=torch.float64) -> Perceptron:
    """A perceptron holding the integer weights exactly (float64 holds every integer below 2^53)."""
    p = Perceptron(width, fan_in, normalization=nn.Identity())
    p.base_activation = LIFActivation(T=timesteps, activation_scale=torch.tensor(1.0))
    p.activation = p.base_activation
    p.layer.to(dtype)
    with torch.no_grad():
        p.layer.weight.copy_(torch.as_tensor(weights, dtype=dtype).reshape(width, fan_in))
        assert p.layer.bias is not None
        p.layer.bias.copy_(torch.as_tensor(bias, dtype=dtype).reshape(width))
    return p


def mapper_graph(instance: Instance) -> MappedNetwork:
    """The mapper graph of ``instance``; a spiking instance reads its input through the identity encoder."""
    net, T = instance.network, instance.timesteps
    nodes: Dict[str, object] = {}
    raw = InputMapper((net.input_width,))
    if instance.semantics == VALUE:
        nodes[INPUT] = raw
    else:
        eye = np.eye(net.input_width, dtype=np.float32)
        encoder = _perceptron(net.input_width, net.input_width, T, eye, np.zeros(net.input_width),
                              dtype=torch.float32)
        setattr(encoder, "use_cycle_accurate_trains", True)
        nodes[INPUT] = PerceptronMapper(raw, encoder)
    neural: Dict[str, PerceptronMapper] = {}
    for layer in net.layers:
        source = _joined(nodes, layer)
        if layer.kind == HOST:
            nodes[layer.name] = ComputeOpMapper(
                source, _HOST_MODULES[layer.op](), input_shape=(net.fan_in(layer),),
                output_shape=(layer.width,), name=f"host_{layer.name}")
            continue
        mapper = PerceptronMapper(source, _perceptron(layer.width, net.fan_in(layer), T, layer.weights, layer.bias))
        nodes[layer.name] = neural[layer.name] = mapper
    sinks = [nodes[name] for name in net.outputs]
    out = sinks[0] if len(sinks) == 1 else ConcatMapper(sinks, dim=1, rank=2)
    rep = ModelRepresentation(out)
    mark_encoding_layers(rep, packaging=MVM_PACKAGING if instance.semantics == VALUE else SPIKING_PACKAGING)
    rep.assign_perceptron_indices()
    perceptron_of = {name: int(getattr(m, "perceptron_index")) for name, m in neural.items()}
    theta_of = {perceptron_of[layer.name]: int(layer.theta) for layer in net.neural_layers()}
    return MappedNetwork(rep, perceptron_of, theta_of)


def _joined(nodes: Dict[str, object], layer: Layer):
    sources = [nodes[s] for s in layer.sources]
    return sources[0] if len(sources) == 1 else ConcatMapper(sources, dim=1, rank=2)


def integer_ir(mapped: MappedNetwork, *, max_axons: Optional[int], max_neurons: Optional[int],
               allow_coalescing: bool, hardware_bias: bool,
               constrained: FrozenSet[str] = ALL_SINGLETON_NAMES) -> IRGraph:
    """The IR as the Soft Core Mapping step maps it, its weights on the integer grid with each layer's theta.

    The weights are integers already, so the step's weight quantization is the identity on them and only the
    threshold it stamps is set here; the residency classes the layout finalizer keyed on the pre-stamp
    threshold are re-keyed on the stamped values (the finalizer's own key over ``constrained``), then the IR
    latency pass runs.
    """
    ir = IRMapping(q_max=_Q_MAX, firing_mode="Default", max_axons=max_axons, max_neurons=max_neurons,
                   allow_coalescing=allow_coalescing, hardware_bias=hardware_bias).map(mapped.representation)
    for node in ir.get_neural_cores():
        if node.perceptron_index is None:
            continue
        node.threshold = float(mapped.theta_of.get(int(node.perceptron_index), 1))
    _rekey_residency_classes(ir, constrained)
    IRLatency(ir).calculate()
    return ir


def _rekey_residency_classes(ir: IRGraph, constrained: FrozenSet[str]) -> None:
    specs = list(getattr(ir, "layout_softcores", None) or ())
    if not specs:
        return
    interned: Dict[object, int] = {}
    for node in ir.get_neural_cores():
        index = getattr(node, "layout_softcore_index", None)
        if index is None or not 0 <= int(index) < len(specs):
            continue
        key = residency_key(node, constrained=constrained, weight_banks=ir.weight_banks)
        specs[int(index)] = dataclasses.replace(specs[int(index)],
                                                residency_class_id=interned.setdefault(key, len(interned)))
    ir.layout_softcores = specs


def relabel(ir: IRGraph, seed: int) -> Dict[int, int]:
    """A consistent permutation of node ids and of node order (a random linear extension), in place.

    Returns the id map; every source, the output sources and the node order are rewritten together, so
    the relabelled graph is the same program.
    """
    rng = random.Random(int(seed))
    ids = [int(n.id) for n in ir.nodes]
    shuffled = list(ids)
    rng.shuffle(shuffled)
    id_map = dict(zip(ids, shuffled))
    order = _random_linear_extension(ir, rng)
    for node in ir.nodes:
        node.input_sources = _remap_sources(node.input_sources, id_map)
        node.id = id_map[int(node.id)]
    ir.output_sources = _remap_sources(ir.output_sources, id_map)
    ir.nodes = [ir.nodes[i] for i in order]
    return id_map


def _remap_sources(sources: np.ndarray, id_map: Dict[int, int]) -> np.ndarray:
    flat = [IRSource(node_id=id_map[int(s.node_id)], index=int(s.index))
            if isinstance(s, IRSource) and int(s.node_id) in id_map else s for s in sources.flatten()]
    return np.array(flat, dtype=object).reshape(sources.shape)


def _random_linear_extension(ir: IRGraph, rng: random.Random):
    position = {int(n.id): i for i, n in enumerate(ir.nodes)}
    preds = {i: {position[int(s.node_id)] for s in n.input_sources.flatten()
                 if isinstance(s, IRSource) and int(s.node_id) in position}
             for i, n in enumerate(ir.nodes)}
    placed: list = []
    done: set = set()
    while len(placed) < len(ir.nodes):
        ready = [i for i in range(len(ir.nodes)) if i not in done and preds[i] <= done]
        pick = rng.choice(ready)
        placed.append(pick)
        done.add(pick)
    return placed


def neural_core_count(ir: IRGraph) -> int:
    return sum(1 for n in ir.nodes if isinstance(n, NeuralCore))
