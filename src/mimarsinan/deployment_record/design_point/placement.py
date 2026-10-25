"""Per-operator placement of an executed program, with the reason of each place.

An operator is a perceptron (every IR core carrying its ``perceptron_index``: one weight bank over its output
positions, or its partial sums) or one host op. A core-placed operator's role is ``static`` (its matrix is written
from the model) or ``operand`` (its matrix is a run-time payload written once per inference: its weight bank is one
of the program's ``dynamic_bank_bindings``, the dynamic-operand lowering's binding table, which this tree's
programs do not carry). The reason is the requirement table's row for the operator's kind (``mapping/capability``), so the
placement and the coverage census read one table. Passes are the executed program's neural passes that hold any of
the operator's cores; a host op names its host stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.deployment_record.design_point.schema import CORE, HOST, HOST_ROLE, OPERAND, STATIC, WIRING
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds
from mimarsinan.mapping.capability.requirements import WIRING as WIRING_PLACEMENT, requirement_of
from mimarsinan.mapping.ir import ComputeOp, NeuralCore

#: The binding table of the dynamic-operand lowering: weight bank id -> its run-time payload binding.
DYNAMIC_BANKS = "dynamic_bank_bindings"
OPERAND_REASON = ("operand core: its matrix is a run-time payload written once per inference and the other operand "
                  "streamed through it")
OperatorKey = Tuple[str, int]


def operator_key(node: Any) -> OperatorKey:
    """A perceptron's IR cores share one key; every other node is its own operator."""
    index = getattr(node, "perceptron_index", None)
    if isinstance(node, NeuralCore) and index is not None:
        return ("perceptron", int(index))
    return ("node", int(node.id))


def operator_label(key: OperatorKey, nodes: List[Any], names: Optional[Mapping[int, str]] = None) -> str:
    """The perceptron's name where the caller states it, else the first node's name (a host op's own name)."""
    if key[0] == "perceptron" and names is not None and key[1] in names:
        return str(names[key[1]])
    return str(nodes[0].name)


def operator_groups(ir_graph: Any) -> Dict[OperatorKey, List[Any]]:
    """IR nodes per operator, in IR order."""
    groups: Dict[OperatorKey, List[Any]] = {}
    for node in ir_graph.nodes:
        groups.setdefault(operator_key(node), []).append(node)
    return groups


def node_passes(hybrid: Any) -> Tuple[Dict[int, set], Dict[int, int]]:
    """({IR core id: neural pass positions holding it}, {host op id: its stage position})."""
    passes: Dict[int, set] = {}
    host: Dict[int, int] = {}
    q = 0
    for position, stage in enumerate(hybrid.stages):
        if stage.kind == "neural":
            for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
                for placement in placements:
                    passes.setdefault(int(placement["ir_node_id"]), set()).add(q)
            q += 1
        elif stage.kind == "compute" and stage.compute_op is not None:
            host[int(stage.compute_op.id)] = position
    return passes, host


def _elements(node: Any) -> int:
    if isinstance(node, NeuralCore):
        return int(node.get_output_count())
    shape = getattr(node, "output_shape", None)
    return int(np.prod(shape)) if shape else 0


def operator_placement(ir_graph: Any, hybrid: Any, *, names: Optional[Mapping[int, str]] = None
                       ) -> List[Dict[str, Any]]:
    """One row per operator: kind, placement, role, reason (the requirement table's), IR nodes and passes."""
    kinds = ir_op_kinds(ir_graph)
    passes, host = node_passes(hybrid)
    dynamic = set(getattr(ir_graph, DYNAMIC_BANKS, None) or getattr(hybrid, DYNAMIC_BANKS, None) or ())
    rows: List[Dict[str, Any]] = []
    for key, nodes in operator_groups(ir_graph).items():
        first = nodes[0]
        kind = kinds[first.id]
        requirement = requirement_of(kind)
        row: Dict[str, Any] = {"operator": operator_label(key, nodes, names), "kind": kind,
                               "capability_classes": "".join(sorted(requirement.classes)),
                               "ir_nodes": len(nodes), "output_elements": sum(_elements(n) for n in nodes)}
        if isinstance(first, NeuralCore):
            runtime = any(getattr(n, "weight_bank_id", None) in dynamic for n in nodes)
            row.update(placement=CORE, role=OPERAND if runtime else STATIC,
                       reason=OPERAND_REASON if runtime else requirement.reason,
                       passes=sorted({q for n in nodes for q in passes.get(int(n.id), ())}),
                       as_mapped_cells=sum(int(n.get_input_count()) * int(n.get_output_count()) for n in nodes))
        else:
            assert isinstance(first, ComputeOp)
            row.update(placement=HOST, role=WIRING if requirement.placement == WIRING_PLACEMENT else HOST_ROLE,
                       reason=requirement.reason, passes=[], host_stage=host.get(int(first.id)))
            if requirement.host_boundary:
                row["host_boundary"] = requirement.host_boundary
        rows.append(row)
    return rows


def operator_of_nodes(ir_graph: Any, names: Optional[Mapping[int, str]] = None) -> Dict[int, str]:
    """{IR node id: its operator's label}."""
    out: Dict[int, str] = {}
    for key, nodes in operator_groups(ir_graph).items():
        label = operator_label(key, nodes, names)
        out.update({int(n.id): label for n in nodes})
    return out
