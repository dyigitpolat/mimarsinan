"""An X3a instance as the toolchain maps it: mapper graph, integer IR at a declared geometry, IR relabelling."""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.certification.randomized.instance import HOST, INPUT, VALUE, Instance, Layer
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.export.law_grids import data_sources
from mimarsinan.mapping.export.law_solver import ReadoutLawSpec, solve_ir_laws
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
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

#: The weight-quantization bound the IR is mapped under; the integer weights already lie on the grid.
_Q_MAX = 127.0
#: The value regime's integer readout law (G2's E283, from CL1): a 16-bit activation grid whose range is its levels
#: (step 1, so a code is the integer it carries), the identity ADC (32 bits, at least every X3a accumulator's width:
#: fan-in 64 x |w| 8 x 32,767 needs 25) and a 16-bit multiplier (M0 = 2^15, shift 15: the ratio 1 exactly).
VALUE_ACTIVATION_BITS = 16
VALUE_LAW: Mapping[str, Any] = {
    "activation_bits": VALUE_ACTIVATION_BITS,
    "readout_law": {"adc_bits": 32, "range": "fixed", "range_value": float(grid_levels(VALUE_ACTIVATION_BITS)),
                    "mult_bits": 16},
}


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


def mapper_graph(instance: Instance, encoding_placement: str = "subsume", positions: int = 1) -> MappedNetwork:
    """The mapper graph of ``instance``; a spiking instance reads its input through the identity encoder, on the host
    (``subsume``) or on chip (``offload``, a first segment reading the raw input). [thru:w4] ``positions`` P > 1 applies
    the network at P positions, its weights shared (an input of P rows of ``input_width``): every neural layer is then a
    weight bank of P instances, as a 1 x 1 convolution maps, the form position multiplexing plans."""
    net, T = instance.network, instance.timesteps
    nodes: Dict[str, object] = {}
    lead = () if positions == 1 else (int(positions),)
    raw = InputMapper(lead + (net.input_width,))
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
        source = _joined(nodes, layer, lead)
        if layer.kind == HOST:
            nodes[layer.name] = ComputeOpMapper(
                source, _HOST_MODULES[layer.op](), input_shape=lead + (net.fan_in(layer),),
                output_shape=lead + (layer.width,), name=f"host_{layer.name}")
            continue
        mapper = PerceptronMapper(source, _perceptron(layer.width, net.fan_in(layer), T, layer.weights, layer.bias))
        nodes[layer.name] = neural[layer.name] = mapper
    sinks = [nodes[name] for name in net.outputs]
    out = sinks[0] if len(sinks) == 1 else ConcatMapper(sinks, dim=len(lead) + 1, rank=len(lead) + 2)
    rep = ModelRepresentation(out)
    mark_encoding_layers(rep, placement=encoding_placement,
                         packaging=MVM_PACKAGING if instance.semantics == VALUE else SPIKING_PACKAGING)
    rep.assign_perceptron_indices()
    perceptron_of = {name: int(getattr(m, "perceptron_index")) for name, m in neural.items()}
    theta_of = {perceptron_of[layer.name]: int(layer.theta) for layer in net.neural_layers()}
    return MappedNetwork(rep, perceptron_of, theta_of)


def _joined(nodes: Dict[str, object], layer: Layer, lead: Tuple[int, ...] = ()):
    sources = [nodes[s] for s in layer.sources]
    return sources[0] if len(sources) == 1 else ConcatMapper(sources, dim=len(lead) + 1, rank=len(lead) + 2)


def integer_ir(mapped: MappedNetwork, *, max_axons: Optional[int], max_neurons: Optional[int],
               allow_coalescing: bool, hardware_bias: bool,
               constrained: FrozenSet[str] = ALL_SINGLETON_NAMES,
               law: Optional[Mapping[str, Any]] = None) -> IRGraph:
    """The IR as the Soft Core Mapping step maps it, its weights on the integer grid with each layer's theta.

    The weights are integers already, so the step's weight quantization is the identity on them and only the
    threshold it stamps is set here; the residency classes the layout finalizer keyed on the pre-stamp
    threshold are re-keyed on the stamped values (the finalizer's own key over ``constrained``), then the IR
    latency pass runs. ``law`` (``activation_bits`` and ``readout_law``) installs the integer readout law as
    the step's hook does (:func:`install_integer_law`).
    """
    ir = IRMapping(q_max=_Q_MAX, firing_mode="Default", max_axons=max_axons, max_neurons=max_neurons,
                   allow_coalescing=allow_coalescing, hardware_bias=hardware_bias).map(mapped.representation)
    for node in ir.get_neural_cores():
        if node.perceptron_index is None:
            continue
        node.threshold = float(mapped.theta_of.get(int(node.perceptron_index), 1))
    _rekey_residency_classes(ir, constrained)
    IRLatency(ir).calculate()
    if law is not None:
        install_integer_law(ir, law)
    return ir


def law_levels(law: Mapping[str, Any]) -> int:
    """The positive codes of the law's activation grid: every value the integer program carries lies in +-it."""
    return grid_levels(int(law["activation_bits"]))


def install_integer_law(ir: IRGraph, law: Mapping[str, Any]) -> None:
    """G2's integer readout law on every core (E283), solved as the Soft Core Mapping hook solves it
    (``law_solver.install_readout_laws``) on a program whose codes are its integers: every perceptron's output
    range is ``readout_law.range_value`` (the grid's levels, so the output step is 1) and every core reading a
    host value or the raw input carries the boundary grid of step 1. Refuses by name what the solver refuses (a
    partial-sum core, a core mixing host and core sources, a law past the 2^53 bound)."""
    bits = int(law["activation_bits"])
    spec = ReadoutLawSpec.from_declaration(law["readout_law"], activation_bits=bits, weight_bits=8)
    span = float(law["readout_law"]["range_value"])
    by_id = {int(n.id): n for n in ir.nodes}
    for node in ir.get_neural_cores():
        if any(not isinstance(by_id.get(int(s.node_id)), NeuralCore) for s in data_sources(node)):
            node.boundary_grid = BoundaryGrid(scale=span, bits=bits)
    ranges = {int(n.perceptron_index): span for n in ir.get_neural_cores() if n.perceptron_index is not None}
    for core_id, core_law in solve_ir_laws(ir, ranges, spec).items():
        setattr(by_id[core_id], "readout_law", core_law)


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


def golden_relation(mapping, x, nodes: Mapping[int, torch.Tensor]) -> Tuple[int, int]:
    """``(windows, mismatches)``: a law-carrying program's run (``nodes``, its output at key -1) on the host input
    ``x`` against G2's int64 golden (``run_golden``, whose ``evaluate`` is its codes) at atol 0, on every program
    output a core computes (every neural layer is one; an unconsumed host op is no core's output, so no law grid)."""
    neural = {int(sl.node_id) for s in mapping.stages if s.kind == "neural" for sl in s.output_map}
    sources = list(mapping.output_sources.flatten())
    keep = [i for i, src in enumerate(sources) if isinstance(src, IRSource) and int(src.node_id) in neural]
    view = SimpleNamespace(stages=mapping.stages, output_sources=np.array([sources[i] for i in keep], dtype=object))
    codes, steps = run_golden(view, np.asarray(x, dtype=np.float64))
    want = torch.from_numpy(codes * steps)
    got = nodes[-1][:, keep].double()
    return int(want.numel()), int((got != want).sum())
