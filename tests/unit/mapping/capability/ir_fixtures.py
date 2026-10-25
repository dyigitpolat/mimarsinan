"""Hand-built IR graphs with hand-countable censuses (no conversion involved)."""

from __future__ import annotations

import operator
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.capability import census as census_module
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.support.compute_modules import ComputeAdapter

INPUT = -2


def sources(pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    return np.array([IRSource(node_id=int(n), index=int(i)) for n, i in pairs], dtype=object)


def outputs(node_id: int, count: int, start: int = 0) -> list:
    return [(node_id, i) for i in range(start, start + count)]


def core(node_id: int, srcs: Sequence[Tuple[int, int]], n_out: int) -> NeuralCore:
    return NeuralCore(
        id=node_id, name=f"core{node_id}", input_sources=sources(srcs),
        core_matrix=np.zeros((len(srcs), n_out)),
    )


def op(node_id: int, module: nn.Module, srcs: Sequence[Tuple[int, int]], *, output_shape,
       input_shape=None, input_shapes=None) -> ComputeOp:
    params: dict = {"module": module}
    if input_shape is not None:
        params["input_shape"] = tuple(input_shape)
    if input_shapes is not None:
        params["input_shapes"] = [tuple(s) for s in input_shapes]
    label = getattr(module, "display_name", None)
    return ComputeOp(
        id=node_id, name=f"op{node_id}", input_sources=sources(srcs),
        op_type=label if isinstance(label, str) else type(module).__name__, params=params,
        input_shape=tuple(input_shape) if input_shape is not None else None,
        output_shape=tuple(output_shape),
    )


def graph(nodes, output_pairs) -> IRGraph:
    return IRGraph(nodes=list(nodes), output_sources=sources(output_pairs))


def lane_graph() -> IRGraph:
    """A(core) B(ReLU) C(affine) D(core) E(add C, D) F(core) G(GELU) H(ReLU) I(core, 2 logits); 4-wide input.

    Hand census. C0: host ops B, C, E, G, H; in order [A] B C [D] E [F] G H [I] = 4 neural / 5 host; boundary
    entry 4 (x into A), carry out 12 (A to B, D to E, F to G), carry in 8 (E to F, H to I), readout 2 = 26.
    C1 (L A E P C W): B, C, E on chip, G (needs T) and H (its producer G is host) stay host: 2 / 2, boundary
    4 + 4 + 4 + 2 = 14, lane ops 12. C5 (adds T): every op on chip, 1 / 0, boundary 6, lane ops 16 (G is a T
    kind, not a lane kind).
    """
    scale = torch.full((4,), 2.0)
    nodes = [
        core(0, outputs(INPUT, 4), 4),
        op(1, nn.ReLU(), outputs(0, 4), input_shape=(4,), output_shape=(4,)),
        op(2, ComputeAdapter(operator.mul, bound_tensors=(scale,)), outputs(1, 4), input_shape=(4,),
           output_shape=(4,)),
        core(3, outputs(0, 4), 4),
        op(4, ComputeAdapter(operator.add), outputs(2, 4) + outputs(3, 4), input_shapes=((4,), (4,)),
           output_shape=(4,)),
        core(5, outputs(4, 4), 4),
        op(6, nn.GELU(), outputs(5, 4), input_shape=(4,), output_shape=(4,)),
        op(7, nn.ReLU(), outputs(6, 4), input_shape=(4,), output_shape=(4,)),
        core(8, outputs(7, 4), 2),
    ]
    return graph(nodes, outputs(8, 2))


class Contraction(nn.Module):
    """Stand-in for G1's ``mapping/support/contraction.py`` payload, the §5.1 interface G7 reads (duck-typed); the MAC
    SSOT counts it by its ``forward_macs()`` declaration, as it counts G1's payload."""

    def __init__(self, einsum_spec: str, operand_roles, operand_shapes):
        super().__init__()
        self.einsum_spec = einsum_spec
        self.operand_roles = tuple(operand_roles)
        self.operand_shapes = tuple(tuple(s) for s in operand_shapes)

    def forward(self, a, b):
        return torch.einsum("b" + self.einsum_spec.replace(",", ",b").replace("->", "->b"), a, b)

    def forward_macs(self) -> int:
        (m, k), (_, n) = self.operand_shapes
        return m * k * n


def contraction_graph(roles=("activation", "activation")) -> IRGraph:
    """Q(core, 2x4) and K(core, 4x3) from an 8-wide input, P = Q @ K (Contraction, 2x3), O(core, 2 logits).

    Hand census. C0 and C2: P on the host, [Q K] P [O] = 2 / 1; boundary entry 8, carry out 20 (Q and K into
    P), carry in 6, readout 2 = 36; no runtime write. C3, on-chip write-back: 1 / 0, boundary 10, 12 written
    cells (the stationary K, 4 x 3) per inference. C3, host payload write: one host op (the payload write of K
    before P), [Q K] W [P O] = 2 / 1, boundary 8 + 12 out + 12 in + 2 = 34, the same 12 written cells.
    """
    nodes = [
        core(0, outputs(INPUT, 8), 8),
        core(1, outputs(INPUT, 8), 12),
        op(2, Contraction("ik,kj->ij", roles, ((2, 4), (4, 3))), outputs(0, 8) + outputs(1, 12),
           input_shapes=((2, 4), (4, 3)), output_shape=(2, 3)),
        core(3, outputs(2, 6), 2),
    ]
    return graph(nodes, outputs(3, 2))


def grouped_graph() -> IRGraph:
    """A(core, 8 from input) G(grouped 1x1 conv, 2 groups, on 2x2x2) B(core, 2 logits).

    Hand census. Cells 8 x 8 + 8 x 2 = 80, every source a value, so the core MAC floor is 80. G has a MAC rule:
    2 output channels x 4 positions x 1 input channel per group x a 1 x 1 kernel = 8 MACs; it is a host op until
    C2 declares class G, then on chip.
    """
    grouped = nn.Conv2d(2, 2, 1, groups=2)
    return graph([core(0, outputs(INPUT, 8), 8),
                  op(1, grouped, outputs(0, 8), input_shape=(2, 2, 2), output_shape=(2, 2, 2)),
                  core(2, outputs(1, 8), 2)], outputs(2, 2))


def grouped_conv3d_graph() -> IRGraph:
    """A(core, 8 from input) G(grouped 1x1x1 Conv3d, 2 groups, on 2x2x2x1) B(core, 2 logits).

    G is the stated-boundary kind (class G, "grouped convolution on the host until N5"); the MAC SSOT counts it at
    every output position, 2 output channels x 4 positions x 1 input channel per group = 8 MACs (with
    ``seed_mac_gap(monkeypatch, "Conv3d")`` it counts 0, and the stated boundary has no host MACs to state). Core
    MAC floor 8 x 8 + 8 x 2 = 80.
    """
    grouped = nn.Conv3d(2, 2, 1, groups=2)
    return graph([core(0, outputs(INPUT, 8), 8),
                  op(1, grouped, outputs(0, 8), input_shape=(2, 2, 2, 1), output_shape=(2, 2, 2, 1)),
                  core(2, outputs(1, 8), 2)], outputs(2, 2))


def convtranspose_graph() -> IRGraph:
    """A(core, 8 from input) T(ConvTranspose2d 2->2, k2 s2, on 2x2x2 to 2x4x4) B(core, 2 logits).

    T is a static-linear operator (class S): host at C0 and C1, on chip from C2. The MAC SSOT counts it in its
    scatter form, every input element times every kernel weight of its group: 2 x 2 x 2 input elements x 2 output
    channels x a 2 x 2 kernel = 64 MACs, on whichever side runs it.
    """
    up = nn.ConvTranspose2d(2, 2, 2, stride=2)
    return graph([core(0, outputs(INPUT, 8), 8),
                  op(1, up, outputs(0, 8), input_shape=(2, 2, 2), output_shape=(2, 4, 4)),
                  core(2, outputs(1, 32), 2)], outputs(2, 2))


def bilinear_upsample_graph() -> IRGraph:
    """A(core, 8 from input) U(bilinear Upsample x2, on 2x2x2 to 2x4x4) B(core, 2 logits).

    U is a static-linear operator (class S: each output a fixed weighted sum of its input neighbours): host at C0
    and C1, on chip from C2. The MAC SSOT has no interpolation rule, so it counts U as 0 MACs on whichever side
    runs it: a MAC-carrying product the SSOT cannot count.
    """
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    return graph([core(0, outputs(INPUT, 8), 8),
                  op(1, up, outputs(0, 8), input_shape=(2, 2, 2), output_shape=(2, 4, 4)),
                  core(2, outputs(1, 32), 2)], outputs(2, 2))


def adapter_product_graph() -> IRGraph:
    """The shape of ``contraction_graph`` with P = Q @ K a generic host op of ``torch.matmul`` (a ``ComputeAdapter``,
    no ``forward_macs`` declaration): a dynamic product (class D). The MAC SSOT counts it in functional form since
    Swin's W10; a test of the census's reaction to a product it cannot count seeds the gap (``seed_mac_gap``)."""
    nodes = [
        core(0, outputs(INPUT, 8), 8),
        core(1, outputs(INPUT, 8), 12),
        op(2, ComputeAdapter(torch.matmul), outputs(0, 8) + outputs(1, 12), input_shapes=((2, 4), (4, 3)),
           output_shape=(2, 3)),
        core(3, outputs(2, 6), 2),
    ]
    return graph(nodes, outputs(3, 2))


def seed_mac_gap(monkeypatch, op_type: str) -> None:
    """The census's MAC SSOT (``count_host_ops``, read through ``mapping.capability.census``) with no rule for
    ``op_type``: every op of that type counts 0 MACs, every other op its SSOT count. Since integration every grouped
    convolution has a MAC rule, so the stated-boundary kind reaches the uncounted-product path only through this
    seeded gap; the census's reaction to a gap is the behaviour under test, never which ops have one."""
    real = census_module.count_host_ops

    def without_rule(ir_graph: IRGraph) -> int:
        return sum(0 if str(node.op_type) == op_type else real(IRGraph(nodes=[node], output_sources=sources(())))
                   for node in ir_graph.get_compute_ops())

    monkeypatch.setattr(census_module, "count_host_ops", without_rule)


def shared_readout_graph() -> IRGraph:
    """A(core, 4 from input) B(ReLU of A) C(core of B, 2 logits); the network output reads A and C.

    Hand census at C0: entry 4, carry out 4 (A to B), carry in 4 (B to C), readout 2: A's outputs leave the chip
    once although both B and the output read them, so 14 elements, not 18. At C1 B joins the chip: entry 4,
    readout 6 (A's 4 and C's 2), 10 elements.
    """
    nodes = [
        core(0, outputs(INPUT, 4), 4),
        op(1, nn.ReLU(), outputs(0, 4), input_shape=(4,), output_shape=(4,)),
        core(2, outputs(1, 4), 2),
    ]
    return graph(nodes, outputs(0, 4) + outputs(2, 2))


def interleaved_branches_graph() -> IRGraph:
    """Two branches emitted interleaved: A(core) B(ReLU of A) C(core from input) D(ReLU of C) E(core of B and D).

    Hand census at C0: in order [A] B [C] D [E] = 3 neural / 2 host segments; by dependency A and C share the
    first neural segment (neither waits for a host op) and E is the second: 2 neural segments, one host slot.
    """
    nodes = [
        core(0, outputs(INPUT, 4), 4),
        op(1, nn.ReLU(), outputs(0, 4), input_shape=(4,), output_shape=(4,)),
        core(2, outputs(INPUT, 4), 4),
        op(3, nn.ReLU(), outputs(2, 4), input_shape=(4,), output_shape=(4,)),
        core(4, outputs(1, 4) + outputs(3, 4), 2),
    ]
    return graph(nodes, outputs(4, 2))


def biased_padded_core_graph() -> IRGraph:
    """One core of 4 input axons, one padding axon (off) and one bias axon (always on), 3 outputs.

    Cells 6 x 3 = 18; the core MAC floor counts the 4 x 3 cells fed by a value.
    """
    srcs = outputs(INPUT, 4) + [(-1, 0), (-3, 0)]
    return graph([core(0, srcs, 3)], outputs(0, 3))


def independent_boundary_elements(ir_graph: IRGraph) -> int:
    """C0 boundary census written independently of the toolchain census: distinct elements crossing chip and host."""
    on_chip = {node.id: isinstance(node, NeuralCore) for node in ir_graph.nodes}
    chip_to_host, host_to_chip, input_to_chip = set(), set(), set()
    for node in ir_graph.nodes:
        for source in node.input_sources.flatten():
            producer = int(source.node_id)
            if on_chip[node.id] and producer == INPUT:
                input_to_chip.add(int(source.index))
            elif producer >= 0 and on_chip[node.id] != on_chip[producer]:
                (host_to_chip if on_chip[node.id] else chip_to_host).add((producer, int(source.index)))
    for source in ir_graph.output_sources.flatten():
        if int(source.node_id) >= 0 and on_chip[int(source.node_id)]:
            chip_to_host.add((int(source.node_id), int(source.index)))
    return len(chip_to_host) + len(host_to_chip) + len(input_to_chip)
