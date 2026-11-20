"""[U-TGAUGE4] Generated layouts for the placement rule's property test, in both walk families.

A tree is drawn from a seed: one encoding perceptron over a token input, producers of random cascade depth (chains of
one to three perceptrons over the encoding, or the encoding itself) at random gauges (one value, or one per channel),
and a random tree of the toolchain's layout mappers over them (a concatenation along either axis, a flatten, an axis
move, a gather, a stack). The tree is read as the output, by a core, by a host operation, or by a host operation then
a core. Each program is built by the toolchain's own sequence: the timing family's cascaded pair (the gauge
establishment, the IR and the identity program, ``ttfs_cycle_based`` cascaded), the count family's Soft Core Mapping
sequence (``host_gather_forms.scm_sequence``, with the declared depth-balancing relays). ``read`` compares the walk's
output with the deployed executor's read decoded at its program's output gauge, element by element, or reports the
toolchain's refusal by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import (
    ConcatMapper, InputMapper, PermuteMapper, ReshapeMapper, StackMapper, SubscriptMapper)
from mimarsinan.mapping.mapping_utils import ModelRepresentation, PerceptronMapper
from mimarsinan.mapping.support.gauge_placement import GaugePlacementError
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

TOKENS, FEATURES = 2, 4
#: Refusals by name: the toolchain declines the construction and says why.
NAMED_REFUSALS: Tuple[type, ...] = (ToolchainRefusal, GaugePlacementError, NotImplementedError)


class GraphFlow(PerceptronFlow):
    """A flow around a hand-built mapper graph."""

    def __init__(self, mapper_repr):
        super().__init__("cpu")
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation

    def forward(self, x):
        return self._mapper_repr(x)


class HostAffine(nn.Module):
    """A host operation that does not commute with a gauge (``x * a + b``)."""

    def forward(self, x):
        return x * 0.7 + 0.3


@dataclass
class Tree:
    seed: int
    text: str
    repr_: Any
    perceptrons: List[Any]
    gauges: List[Any]
    shape: tuple = (TOKENS, FEATURES)
    reader: str = "out"
    depths: List[int] = field(default_factory=list)


def _part_shape(node) -> tuple:
    return node._u4_shape


def _tag(node, shape):
    node._u4_shape = tuple(int(d) for d in shape)
    return node


def draw_tree(seed: int, *, per_channel: bool = True) -> Tree:
    """One generated tree (no data: the weights and gauges are drawn from ``seed``)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perceptrons: List[Any] = []
    gauges: List[Any] = []

    def perceptron(src, out, inp, name):
        p = Perceptron(out, inp, name=name)
        with torch.no_grad():
            p.layer.weight.uniform_(-0.6, 0.9)
            p.layer.bias.uniform_(0.0, 0.15)
        perceptrons.append(p)
        if per_channel and rng.random() < 0.25 and out > 1:
            gauges.append([float(v) for v in rng.uniform(0.4, 2.5, out)])
        else:
            gauges.append(float(rng.uniform(0.4, 2.5)))
        return PerceptronMapper(src, p)

    width = int(rng.integers(3, 6))
    inp = _tag(InputMapper((TOKENS, FEATURES)), (TOKENS, FEATURES))
    enc = _tag(perceptron(inp, width, FEATURES, "encoding"), (TOKENS, width))
    leaves, depths = [], []
    for i in range(int(rng.integers(2, 5))):
        depth = int(rng.choice([0, 1, 1, 2, 2, 3])) if i else int(rng.integers(1, 4))
        node, w = enc, width
        for j in range(depth):
            out = int(rng.integers(2, 5))
            node = _tag(perceptron(node, out, w, f"p{i}_{j}"), (TOKENS, out))
            w = out
        leaves.append(node)
        depths.append(depth)
    text: List[str] = []

    def combine(parts):
        """Join ``parts`` (each of one rank) into one node by a random layout."""
        while len(parts) > 1:
            a, b = parts.pop(int(rng.integers(len(parts)))), parts.pop(int(rng.integers(len(parts))))
            sa, sb = _part_shape(a), _part_shape(b)
            if len(sa) != len(sb):
                a, b = flatten(a), flatten(b)
                sa, sb = _part_shape(a), _part_shape(b)
            choices = []
            if all(x == y for x, y in zip(sa[:-1], sb[:-1])):
                choices.append("cat_last")
            if len(sa) >= 2 and sa[1:] == sb[1:]:
                choices.append("cat_first")
            if sa == sb and min(sa) >= 2:
                choices.append("stack")
            if not choices:
                a, b = flatten(a), flatten(b)
                sa, sb = _part_shape(a), _part_shape(b)
                choices = ["cat_last"]
            op = str(rng.choice(choices))
            rank = len(sa) + 1
            if op == "cat_last":
                node = _tag(ConcatMapper([a, b], dim=rank - 1, rank=rank), sa[:-1] + (sa[-1] + sb[-1],))
            elif op == "cat_first":
                node = _tag(ConcatMapper([a, b], dim=1, rank=rank), (sa[0] + sb[0],) + sa[1:])
            else:
                node = _tag(StackMapper([a, b]), (2,) + sa)
            text.append(op)
            parts.append(move(node))
        return parts[0]

    def flatten(node):
        shape = _part_shape(node)
        if len(shape) == 1:
            return node
        text.append("flatten")
        return _tag(ReshapeMapper(node, (int(np.prod(shape)),)), (int(np.prod(shape)),))

    def move(node):
        """A random element movement of one part: none, an axis move, a gather, or a flatten."""
        shape = _part_shape(node)
        r = rng.random()
        if len(shape) == 2 and r < 0.25:
            text.append("axis_move")
            return _tag(PermuteMapper(node, (0, 2, 1)), (shape[1], shape[0]))
        if len(shape) >= 2 and shape[0] >= 2 and r < 0.4:
            text.append("gather")
            return _tag(SubscriptMapper(node, int(rng.integers(shape[0]))), shape[1:])
        if len(shape) == 3 and r < 0.6:
            return flatten(node)
        return node

    root = combine([move(leaf) for leaf in leaves])
    dead = _dead_producers(root, leaves)
    if dead:  # a gather that drops every element of a producer leaves a dead core: it joins the output instead
        text.append(f"keep{len(dead)}")
        root = combine([flatten(root)] + [flatten(leaf) for leaf in dead])
    reader = str(rng.choice(["out", "out", "core", "host", "host_core"]))
    flat = flatten(root)
    n = _part_shape(flat)[0]
    if reader == "out":
        top = flat if rng.random() < 0.5 else root
    elif reader == "core":
        top = _tag(perceptron(flat, 3, n, "reader"), (3,))
    else:
        host = _tag(ComputeOpMapper(flat, HostAffine(), name="host"), (n,))
        top = host if reader == "host" else _tag(perceptron(host, 3, n, "reader"), (3,))
    text.append(reader)
    return Tree(seed, " ".join(text), ModelRepresentation(top), perceptrons, gauges, reader=reader, depths=depths)


def _dead_producers(root, leaves) -> list:
    """The leaves none of whose elements ``root`` holds (read from the identity program's IR sources)."""
    from mimarsinan.mapping.map_model_to_ir import mapper_source_table

    table = mapper_source_table(ModelRepresentation(root))
    live = set(table.sources[id(root)][0].tolist())
    return [leaf for leaf in leaves if not live & set(int(i) for i in table.sources[id(leaf)][0] if i >= 0)]


def refused_by_name(exc: BaseException) -> bool:
    return isinstance(exc, NAMED_REFUSALS) and bool(str(exc))


TIMING_S = 8
SAMPLES = 8


def _set_gauges(tree: Tree, *, encoding_at_one: bool) -> None:
    for p, g in zip(tree.perceptrons, tree.gauges):
        value = 1.0 if (encoding_at_one and getattr(p, "is_encoding_layer", False)) else g
        p.set_activation_scale(torch.as_tensor(value, dtype=torch.float32).clone())


def timing_program(tree: Tree, S: int = TIMING_S):
    """The timing family's cascaded pair on ``tree`` (the sequence of ``test_cascaded_boundary_transcoding._build_pair``
    at the drawn gauges, the encoding perceptron at the input's gauge as the calibration pins it): ``(walk, executor)``."""
    from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.latency.ir import IRLatency
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
    from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
    from mimarsinan.transformations.quantization_bounds import quantization_bounds

    repr_ = tree.repr_
    flow = GraphFlow(repr_).eval()
    with torch.no_grad():
        flow(torch.rand(2, *tree.shape))
    mark_encoding_layers(repr_)
    _set_gauges(tree, encoding_at_one=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    for p in tree.perceptrons:
        p.set_activation(TTFSActivation(T=S, activation_scale=p.activation_scale, input_scale=p.input_activation_scale,
                                        bias=p.layer.bias, thresholding_mode="<=",
                                        encoding=getattr(p, "is_encoding_layer", False)))
    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)
    _, q_max = quantization_bounds(8)
    ir = IRMapping(q_max=q_max, firing_mode="TTFS", max_axons=2048, max_neurons=2048).map(repr_)
    quantize_ir_graph(ir, 8, weight_quantization=False)
    IRLatency(ir).calculate()
    executor = build_identity_spiking_flow(tree.shape, ir, S, None, "TTFS", "TTFS", thresholding_mode="<=",
                                           spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded").eval()
    for p in tree.perceptrons:  # the walk reads the float64 twin, as the pair's builder hands it over
        p.double()
    return TTFSSegmentForward(repr_, S), executor


def count_program(tree: Tree, semantics: str, **pruning):
    """The count family's program on ``tree`` (``host_gather_forms``: LIF nodes at the drawn thresholds, the Soft Core
    Mapping sequence with the declared depth-balancing relays; ``pruning``: ``scm_sequence``'s)."""
    import unit.spiking.host_gather_forms as hg
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

    repr_ = tree.repr_
    flow = GraphFlow(repr_).eval()
    with torch.no_grad():
        flow(torch.rand(2, *tree.shape))
    mark_encoding_layers(repr_)
    _set_gauges(tree, encoding_at_one=False)
    for p in tree.perceptrons:
        lif = LIFActivation(T=hg.T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    _r, ir, hybrid = hg.scm_sequence(flow, tree.perceptrons, wq=False, retimed=hg.SEMANTICS[semantics][0],
                                     relays=True, **pruning)
    thetas = [float(torch.as_tensor(p.activation_scale).max()) for p in tree.perceptrons]
    return hg.Program(tree.text, tree.shape, flow, repr_, ir, hybrid, tree.perceptrons, thetas, False)


def read(seed: int, family: str) -> dict:
    """One generated tree read by both twins: ``{"seed", "tree", "max_abs"}`` where compared, ``{"refused": name}``
    where the toolchain declines the construction by name. Any other error propagates."""
    from mimarsinan.chip_simulation.hybrid_run.output_decode import executor_output_values

    tree = draw_tree(seed)
    rec: dict = {"seed": seed, "tree": tree.text}
    try:
        if family == "timing":
            walk, executor = timing_program(tree)
            torch.manual_seed(7000 + seed)
            x = torch.rand(SAMPLES, *tree.shape, dtype=torch.float64)
            with torch.no_grad():
                ours = walk(x).double().reshape(SAMPLES, -1)
                deployed = executor_output_values(executor, executor(x)).double().reshape(SAMPLES, -1)
        else:
            import unit.spiking.host_gather_forms as hg
            from mimarsinan.mapping.support.host_kernel_evaluation import twin_as_program
            from mimarsinan.models.nn.lif_kernels import measurement_plane
            from mimarsinan.spiking.segment_forward import SegmentForwardDriver

            semantics = sorted(hg.SEMANTICS)[seed % len(hg.SEMANTICS)]
            program = count_program(tree, semantics)
            x = hg.samples(program, SAMPLES, seed)

            def walk_of(prog, xs):
                with torch.no_grad(), measurement_plane():
                    return SegmentForwardDriver(prog.repr_, hg.T, hg.SEMANTICS[semantics][2]())(xs)

            ours = twin_as_program(program.hybrid, walk_of, program, x).double().reshape(SAMPLES, -1)
            flow = hg.deployed_flow(program, semantics)
            with torch.no_grad():
                deployed = executor_output_values(flow, flow(x)).double().reshape(SAMPLES, -1)
            rec["semantics"] = semantics
    except Exception as exc:  # noqa: BLE001 - a named refusal is a result; any other error is re-raised
        if refused_by_name(exc):
            rec["refused"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return rec
        raise
    rec["max_abs"] = float((ours - deployed).abs().max())
    rec["elements"] = int(ours.numel())
    return rec
