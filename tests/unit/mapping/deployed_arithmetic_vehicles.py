"""[U-DAR] Constructed value programs on which the fold's arithmetic is observable (diagnosis.md section 4.5, V1 to V4;
diagnosis_verify.md section 7, V5).

Every vehicle is built the way ``constant_vehicles`` builds its programs: each core carries an always-on axon row as its
constant carrier. The fold runs as Soft Core Mapping runs it (``prune_ir_graph``, cascade, constant folding full), in
the domain the spiking mode names; the folded program and the unpruned one then run through the deployed read's value
executor (``ValueHybridCoreFlow(build_identity_hybrid_mapping(g), "cpu", float64)``) on the same 16 inputs (zero and 15
integer inputs, seeded).

- V1 core column: a bias-only core (one always-on row, weight 1, threshold 3) emits CONST(1/3); the head folds it.
- V2 host op: a bias-only core emits CONST(0.25), a parameterless GELU host op maps it to GELU(0.25), the head folds it.
- V3 class flip: V1 with the head's two logits a 2^-40 margin apart in the deployed read's arithmetic.
- V4 zero channel: a bias-only core whose carrier weight is 1 + 2^-30 and whose bias is -1 (its line is 2^-30 exactly,
  0 in float32); the head's row reading it dies if the line descends CONST(0).
- V5 plane: a host op reads the always-on source and hosts Linear(1, 1, weight 0.25 - 2^-20) and a lattice-armed
  ``LIFActivation`` (T 4, membrane lattice 8): no spike with a continuous membrane, one spike (rate 1/4) on the lattice,
  which every deployed host evaluator opens (``measurement_plane``).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.pruning.ir_pruning_helpers import _boundary_policy_exemptions
from mimarsinan.models.nn.activations.lif import LIFActivation

#: The deployed read's value of V1's bias line: 1/3 in float64 (0x1.5555555555555p-2).
ONE_THIRD = 1.0 / 3.0
#: V3's margin between the head's logits in the deployed read's arithmetic.
MARGIN = math.ldexp(1.0, -40)
#: V5's host weight: float32-exact, 2^-20 below a quarter (the lattice snaps it to 1/4).
V5_WEIGHT = 0.25 - math.ldexp(1.0, -20)


def srcs(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _data_core(nid: int, latency: int = 0) -> NeuralCore:
    # Two model inputs and the always-on row; dyadic weights (every product and sum exact in float64).
    return NeuralCore(id=nid, name=f"data{nid}", input_sources=srcs([(-2, 0), (-2, 1), (-3, 0)]),
                      core_matrix=np.array([[0.5, -0.25], [0.25, 0.5], [0.125, 0.0625]], dtype=np.float64),
                      threshold=1.0, latency=latency)


def _head(nid: int, line: tuple, data: int, matrix) -> NeuralCore:
    return NeuralCore(id=nid, name="head", input_sources=srcs([line, (data, 0), (data, 1), (-3, 0)]),
                      core_matrix=np.asarray(matrix, dtype=np.float64), threshold=1.0, latency=1)


_HEAD = [[0.75, -0.5], [0.5, 0.25], [-0.25, 0.5], [0.0625, 0.125]]


def v1_core_column() -> IRGraph:
    bias_core = NeuralCore(id=0, name="bias_core", input_sources=srcs([(-3, 0)]),
                           core_matrix=np.array([[1.0]], dtype=np.float64), threshold=3.0, latency=0)
    return IRGraph(nodes=[bias_core, _data_core(1), _head(2, (0, 0), 1, _HEAD)], output_sources=srcs([(2, 0), (2, 1)]))


def v2_host_op() -> IRGraph:
    bias_core = NeuralCore(id=0, name="bias_core", input_sources=srcs([(-3, 0)]),
                           core_matrix=np.array([[0.25]], dtype=np.float64), threshold=1.0, latency=0)
    act = ComputeOp(id=1, name="act", input_sources=srcs([(0, 0)]), op_type="GELU",
                    params={"module": nn.GELU().eval(), "input_shape": (1,)}, input_shape=(1,), output_shape=(1,))
    return IRGraph(nodes=[bias_core, act, _data_core(2), _head(3, (1, 0), 2, _HEAD)],
                   output_sources=srcs([(3, 0), (3, 1)]))


def v3_class_flip() -> IRGraph:
    """logit0 = -line + (1/3 + 2^-40); logit1 = 0: class 0 by 2^-40 in the deployed read's arithmetic."""
    bias_core = NeuralCore(id=0, name="bias_core", input_sources=srcs([(-3, 0)]),
                           core_matrix=np.array([[1.0]], dtype=np.float64), threshold=3.0, latency=0)
    head = _head(2, (0, 0), 1, [[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ONE_THIRD + MARGIN, 0.0]])
    return IRGraph(nodes=[bias_core, _data_core(1), head], output_sources=srcs([(2, 0), (2, 1)]))


def v4_zero_channel() -> IRGraph:
    """The bias core's line: (1 + 2^-30) * 1 + (-1) = 2^-30 exactly, 0 in float32 (1 + 2^-30 rounds to 1)."""
    bias_core = NeuralCore(id=0, name="bias_core", input_sources=srcs([(-3, 0)]),
                           core_matrix=np.array([[1.0 + math.ldexp(1.0, -30)]], dtype=np.float64),
                           hardware_bias=np.array([-1.0], dtype=np.float64), threshold=1.0, latency=0)
    head = _head(2, (0, 0), 1, [[-2.0 ** 31, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    return IRGraph(nodes=[bias_core, _data_core(1), head], output_sources=srcs([(2, 0), (2, 1)]))


def v5_host_module() -> nn.Module:
    lin = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(V5_WEIGHT)
    lif = LIFActivation(T=4, activation_scale=1.0)
    lif.set_membrane_lattice(8.0)
    return nn.Sequential(lin, lif).eval()


def v5_host_op(nid: int = 0) -> ComputeOp:
    return ComputeOp(id=nid, name="host_perceptron", input_sources=srcs([(-3, 0)]), op_type="host_perceptron",
                     params={"module": v5_host_module(), "input_shape": (1,)}, input_shape=(1,), output_shape=(1,))


def v5_plane() -> IRGraph:
    data = NeuralCore(id=1, name="data", input_sources=srcs([(-2, 0), (-2, 1), (-3, 0)]),
                      core_matrix=np.array([[0.0625, 0.0625], [0.0625, 0.0625], [0.0, 0.0]], dtype=np.float64),
                      threshold=1.0, latency=0)
    head = NeuralCore(id=2, name="head", input_sources=srcs([(0, 0), (1, 0), (1, 1), (-3, 0)]),
                      core_matrix=np.array([[4.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 0.5]], dtype=np.float64),
                      threshold=1.0, latency=1)
    return IRGraph(nodes=[v5_host_op(0), data, head], output_sources=srcs([(2, 0), (2, 1)]))


VEHICLES = {"V1": v1_core_column, "V2": v2_host_op, "V3": v3_class_flip, "V4": v4_zero_channel, "V5": v5_plane}


@dataclass
class FoldReading:
    """One vehicle folded in one domain, against the unpruned program through the deployed read's value executor."""

    lattice: Dict[str, float]
    folded_rows: Dict[int, Dict[int, float]]
    max_abs_delta: float
    classes_differing: int
    outputs_differing: int
    neural_cores: int
    head_rows: int
    core_names: tuple = ()


def deployed_inputs() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.cat([torch.zeros(1, 2, dtype=torch.float64),
                      torch.randint(-8, 9, (15, 2), generator=gen).to(torch.float64)])


def fold_and_read(graph: IRGraph, spiking_mode: str) -> FoldReading:
    """Fold ``graph`` as Soft Core Mapping does and read it against the unpruned program."""
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    analysis = compute_global_pruned_sets(copy.deepcopy(graph), exempt_rows_per_node=exempt_rows,
                                          exempt_cols_per_node=exempt_cols, mode="cascade",
                                          elimination_constant_folding="full", spiking_mode=spiking_mode)
    lattice = {f"{k[0]}:{k[1]}": float(v) for k, v in analysis.constant_folds.lattice.snapshot().items()}
    folded = {int(n): {int(r): float(c) for r, c in rows.items()}
              for n, rows in analysis.constant_folds.folded_rows.items()}
    pruned = prune_ir_graph(copy.deepcopy(graph), spiking_mode=spiking_mode, elimination_propagation="cascade",
                            elimination_constant_folding="full", simulation_steps=1)
    ref = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=copy.deepcopy(graph)), device="cpu",
                              dtype=torch.float64)
    cand = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=pruned), device="cpu", dtype=torch.float64)
    x = deployed_inputs()
    with torch.no_grad():
        want, got = ref(x), cand(x)
    delta = (got - want).abs()
    head_rows = [len(n.input_sources.flatten()) for n in pruned.nodes if getattr(n, "name", "") == "head"]
    return FoldReading(lattice=lattice, folded_rows=folded, max_abs_delta=float(delta.max()),
                       classes_differing=int((want.argmax(1) != got.argmax(1)).sum()),
                       outputs_differing=int((delta > 0).sum()), neural_cores=len(pruned.get_neural_cores()),
                       head_rows=head_rows[0] if head_rows else 0,
                       core_names=tuple(sorted(n.name for n in pruned.get_neural_cores())))
