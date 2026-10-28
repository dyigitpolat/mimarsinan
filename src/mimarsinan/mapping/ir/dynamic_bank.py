"""Dynamic operand banks (N2): a weight bank whose matrix is a payload a host op writes once per inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, TypeVar

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal

_P = TypeVar("_P")


class OperandBindingError(ToolchainRefusal, ValueError):
    """A dynamic-bank binding that does not describe a program: a bank, a shape or an order that disagrees with
    the graph, a spiking contract, or a pruned operand bank."""


@dataclass(frozen=True)
class DynamicBankBinding:
    """Bank ``bank_id``'s matrix is ``payload_shape`` (axons, neurons) of the output of host op
    ``source_node_id``, flat from ``payload_offset``, in units of ``payload_step`` (None: the values themselves,
    a float program; a grid step: the payload's integer codes, set where the readout law is solved).
    ``transpose``: the payload is the transpose of the traced operand it carries; ``batch_index``: the index of
    the product's batch axes (the head) the bank serves."""

    bank_id: int
    source_node_id: int
    payload_shape: Tuple[int, int]
    transpose: bool
    batch_index: Tuple[int, ...]
    payload_bits: int
    payload_offset: int = 0
    payload_step: Optional[float] = None

    @property
    def cells(self) -> int:
        return int(self.payload_shape[0]) * int(self.payload_shape[1])


def dynamic_bank_ids(ir_graph: Any) -> FrozenSet[int]:
    """The banks of ``ir_graph`` bound to a payload."""
    return frozenset(int(b) for b in getattr(ir_graph, "dynamic_bank_bindings", {}) or {})


def operand_cells_by_core(hard_core_mapping: Any, banks: Any) -> List[int]:
    """Per hard core of one built segment, the cells of its bound-bank regions (0 for a static core): the payload the
    producer writes every inference (N2), the cores the pinned rule never pins (F3)."""
    return [sum(int(p["axons"]) * int(p["neurons"]) for p in placements
                if p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in banks)
            for placements in hard_core_mapping.soft_core_placements_per_hard_core]


def operand_only_by_core(hard_core_mapping: Any, banks: Any) -> List[bool]:
    """[U-C C-6] Per hard core of one built segment, whether every placement reads a bound bank: the core holds no
    static program (a core with no placement holds none either way, and is not operand-only)."""
    return [bool(placements) and all(p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in banks
                                     for p in placements)
            for placements in hard_core_mapping.soft_core_placements_per_hard_core]


def payload_sources(ir_graph: Any) -> Dict[int, int]:
    """{node id: the host op that writes its payload} for every instance of a bound bank. An instance reads it (the
    value executor materializes it beside the stage's slices, ``stage_payload_sources``), so a program graph and its
    depth segments count it among the instance's producers (F3)."""
    bindings = getattr(ir_graph, "dynamic_bank_bindings", None) or {}
    return {int(node.id): int(bindings[int(node.weight_bank_id)].source_node_id) for node in ir_graph.nodes
            if getattr(node, "weight_bank_id", None) is not None and int(node.weight_bank_id) in bindings}


def bound_instances(ir_graph: Any) -> Dict[int, List[Any]]:
    """{bank id: the NeuralCore instances reading it} for every bound bank."""
    banks = dynamic_bank_ids(ir_graph)
    out: Dict[int, List[Any]] = {bank: [] for bank in banks}
    for node in ir_graph.nodes:
        bank = getattr(node, "weight_bank_id", None)
        if bank is not None and int(bank) in banks:
            out[int(bank)].append(node)
    return out


def stage_bound_banks(stage: Any, bindings: Mapping[int, Any]) -> List[int]:
    """The bound banks a neural stage's placements read, sorted (empty without bindings or a mapping)."""
    hcm = getattr(stage, "hard_core_mapping", None)
    if not bindings or hcm is None:
        return []
    return sorted({int(p["weight_bank_id"]) for placements in hcm.soft_core_placements_per_hard_core
                   for p in placements if p.get("weight_bank_id") is not None and int(p["weight_bank_id"]) in bindings})


def stage_payload_sources(stage: Any, bindings: Mapping[int, Any]) -> List[int]:
    """The host ops whose payload writes a neural stage reads (its bound banks' sources), sorted: the value executor
    materializes them from the state buffer when the stage runs, so they are the stage's reads beside its slices."""
    return sorted({int(bindings[bank].source_node_id) for bank in stage_bound_banks(stage, bindings)})


def carry_bindings(program: _P, ir_graph: Any) -> _P:
    """The program built from ``ir_graph`` carries its dynamic bank bindings (declared-only: absent where there are
    none), so its executors and censuses read the same table; returns ``program``."""
    bindings = getattr(ir_graph, "dynamic_bank_bindings", None)
    if bindings:
        setattr(program, "dynamic_bank_bindings", dict(bindings))
    return program


def _pruned(node: Any) -> bool:
    return any(bool(np.any(np.asarray(mask, dtype=bool))) for mask in
               (getattr(node, "pruned_row_mask", None), getattr(node, "pruned_col_mask", None)) if mask is not None)


def validate_bindings(ir_graph: Any, *, packaging: Any) -> None:
    """Refuse by name every binding that does not describe ``ir_graph`` under ``packaging``."""
    bindings = getattr(ir_graph, "dynamic_bank_bindings", {}) or {}
    if not bindings:
        return
    if not packaging.is_value_domain:
        raise OperandBindingError(
            f"{len(bindings)} dynamic operand bank(s) under a spiking contract: a payload written at run time has no "
            "window-exact spiking form, so dynamic operands run in the value domain only")
    order = {int(node.id): position for position, node in enumerate(ir_graph.nodes)}
    instances = bound_instances(ir_graph)
    for key, binding in bindings.items():
        if int(key) != int(binding.bank_id):
            raise OperandBindingError(f"the binding of bank {binding.bank_id} is keyed under bank {key}: one "
                                      "binding per bank, under its own id")
        bank = ir_graph.weight_banks.get(int(key))
        if bank is None:
            raise OperandBindingError(f"bank {key} is bound to a payload but the graph holds no such weight bank")
        if tuple(np.asarray(bank.core_matrix).shape) != tuple(binding.payload_shape):
            raise OperandBindingError(f"bank {key}: the payload shape {tuple(binding.payload_shape)} is not the "
                                      f"bank's shape {tuple(np.asarray(bank.core_matrix).shape)}")
        source = next((n for n in ir_graph.nodes if int(n.id) == int(binding.source_node_id)), None)
        if source is None or not hasattr(source, "op_type"):
            raise OperandBindingError(f"bank {key}: its payload source {binding.source_node_id} is not a host op of "
                                      "the graph")
        for node in instances[int(key)]:
            if order[int(node.id)] < order[int(source.id)]:
                raise OperandBindingError(f"bank {key}: its payload write {source.name!r} does not precede its "
                                          f"instance {node.name!r} in graph order")
            if _pruned(node):
                raise OperandBindingError(f"bank {key}: instance {node.name!r} is pruned; pruning does not read "
                                          "payload sparsity, so a pruned operand bank is refused")
