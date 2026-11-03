"""The integer certificate of an executed program on two executors, with a per-operator activity census.

Executors of the executed (multi-pass) program: the fp64 hybrid executor (``ValueHybridCoreFlow``) and the int64
golden (``run_golden`` keeping its node outputs, every core's law in exact int64). References: the converted model with its readout laws
(NF) and the identity mapping of the same IR (SCM). Edges, each (mismatches, compared) at atol 0: NF against SCM, the
executor against SCM, SCM against the golden on the identity mapping, the executor against the golden, the golden on
the executed program against the golden on the identity mapping. The census reads both executors node by node (every
node output the program materializes: segment outputs and host ops), counts nonzero outputs per executor and
mismatches between them, and groups nodes per operator; a certificate whose observed operators all carry nonzero
outputs on both executors is not vacuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.deployment_record.design_point.placement import operator_groups, operator_label
from mimarsinan.deployment_record.design_point.schema import CORE, HOST
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping

FLOW, GOLDEN = "value_flow", "int64_golden"
EXECUTORS: Tuple[Dict[str, str], ...] = (
    {"name": FLOW, "role": "executor",
     "description": "the fp64 hybrid executor (ValueHybridCoreFlow) on the executed multi-pass program"},
    {"name": GOLDEN, "role": "executor",
     "description": "the int64 golden (run_golden): every core's law in exact int64 on the executed program, host "
                    "ops in fp64 on grid values"},
    {"name": "nf", "role": "reference",
     "description": "the converted model with its readout laws, in fp64"},
    {"name": "scm", "role": "reference",
     "description": "the identity mapping of the same IR (one core per IR core), run by the fp64 executor"},
)


@dataclass(frozen=True)
class Edge:
    name: str
    between: Tuple[str, str]
    mismatches: int
    compared: int

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "between": list(self.between), "mismatches": self.mismatches,
                "compared": self.compared}


@dataclass
class OperatorActivity:
    operator: str
    placement: str
    outputs: int = 0
    nonzero: Dict[str, int] = field(default_factory=lambda: {FLOW: 0, GOLDEN: 0})
    mismatches: int = 0
    nodes: int = 0
    nodes_observed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"operator": self.operator, "placement": self.placement, "outputs": self.outputs,
                "nonzero": dict(self.nonzero), "mismatches": self.mismatches, "nodes": self.nodes,
                "nodes_observed": self.nodes_observed}


@dataclass
class ProgramCertificate:
    inputs: int
    edges: List[Edge]
    activity: List[OperatorActivity]

    @property
    def exact(self) -> bool:
        return all(e.mismatches == 0 for e in self.edges) and all(a.mismatches == 0 for a in self.activity)

    @property
    def observed(self) -> List[OperatorActivity]:
        return [a for a in self.activity if a.nodes_observed]

    @property
    def nonvacuous(self) -> bool:
        seen = self.observed
        return bool(seen) and all(a.outputs and all(a.nonzero.values()) for a in seen)

    def to_dict(self, *, atol: float = 0.0, notes: Sequence[str] = ()) -> Dict[str, Any]:
        return {"atol": atol, "inputs": self.inputs, "edges": [e.to_dict() for e in self.edges],
                "exact": self.exact, "nonvacuous": self.nonvacuous, "operators_observed": len(self.observed),
                "operators": len(self.activity), "notes": list(notes)}


def _edge(name: str, between: Tuple[str, str], a: Any, b: Any) -> Edge:
    b_t = torch.as_tensor(b).double()
    a_t = torch.as_tensor(a).double().reshape(b_t.shape)
    return Edge(name, between, int((a_t != b_t).sum()), int(b_t.numel()))


@dataclass(frozen=True)
class GoldenRun:
    """The golden's output codes and steps on a program, with every node output it materialized."""

    output_codes: np.ndarray
    output_steps: np.ndarray
    codes: Mapping[int, np.ndarray]
    values: Mapping[int, np.ndarray]


def golden_run(program: Any, x_int: torch.Tensor) -> GoldenRun:
    keep: Dict[str, Dict[int, np.ndarray]] = {}
    codes, steps = run_golden(program, x_int, keep=keep)
    return GoldenRun(codes, steps, keep["codes"], keep["values"])


def _census(ir_graph: Any, state: Mapping[int, torch.Tensor], golden: GoldenRun,
            names: Optional[Mapping[int, str]]) -> List[OperatorActivity]:
    out: List[OperatorActivity] = []
    for key, nodes in operator_groups(ir_graph).items():
        placement = CORE if isinstance(nodes[0], NeuralCore) else HOST
        row = OperatorActivity(operator_label(key, nodes, names), placement, nodes=len(nodes))
        for node in nodes:
            nid = int(node.id)
            if nid not in golden.values or nid not in state:
                continue
            reference = np.asarray(golden.values[nid], dtype=np.float64)
            got = state[nid].detach().double().cpu().numpy().reshape(reference.shape)
            row.nodes_observed += 1
            row.outputs += int(reference.size)
            row.mismatches += int(np.count_nonzero(got != reference))
            row.nonzero[FLOW] += int(np.count_nonzero(got))
            row.nonzero[GOLDEN] += int(np.count_nonzero(golden.codes[nid] if nid in golden.codes else reference))
        out.append(row)
    return out


def certify_program(nf: Any, ir_graph: Any, hybrid: Any, x_int: torch.Tensor, *,
                    names: Optional[Mapping[int, str]] = None) -> ProgramCertificate:
    """The edges and the per-operator census of ``hybrid`` (built from ``ir_graph``) on integer inputs ``x_int``."""
    x = x_int.double()
    identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
    with torch.no_grad():
        nf_out = nf.double()(x)
        scm = ValueHybridCoreFlow(identity, device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype)(x)
        executed, state, _flat = ValueHybridCoreFlow(
            hybrid, device=VALUE_ARITHMETIC.device, dtype=VALUE_ARITHMETIC.core_dtype).run_retaining_state(x)
    reference = golden_run(identity, x_int)
    golden = golden_run(hybrid, x_int)
    edges = [
        _edge("nf_vs_scm", ("nf", "scm"), nf_out, scm),
        _edge("executor_vs_scm", (FLOW, "scm"), executed, scm),
        _edge("scm_vs_golden_identity", ("scm", f"{GOLDEN}:identity"), scm,
              reference.output_codes * reference.output_steps),
        _edge("executor_vs_golden", (FLOW, GOLDEN), executed, golden.output_codes * golden.output_steps),
        _edge("golden_vs_golden_identity", (GOLDEN, f"{GOLDEN}:identity"), golden.output_codes,
              reference.output_codes),
    ]
    return ProgramCertificate(inputs=int(x_int.shape[0]), edges=edges,
                              activity=_census(ir_graph, state, golden, names))
