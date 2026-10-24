"""IR emission of a lowered product: one OperandWrite host op, one bound bank per batch index, one core instance
per streamed vector (N2 design §2.4), on either mapping walk (the full IR or the shape-only layout)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

import numpy as np

from mimarsinan.mapping.ir.dynamic_bank import DynamicBankBinding
from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.support.operand_form import B_STATIONARY

#: The op type an OperandWrite host op carries in the IR (G7's op-kind table reads it).
OPERAND_WRITE_OP_TYPE = "OperandWrite"


def _first_node_id(sources) -> int:
    return int(np.asarray(sources, dtype=object).reshape(-1)[0].node_id)


def _hoist(ir_mapping, write_id: int, producers: set) -> None:
    """Schedule the payload write right after the last node it reads, so the products of one block share a
    neural segment wherever no host op sits between them (the ReLU-attention block)."""
    nodes = getattr(ir_mapping, "nodes", None)
    if nodes is None or not nodes or int(nodes[-1].id) != write_id:
        return
    write = nodes.pop()
    last = max((i for i, node in enumerate(nodes) if int(node.id) in producers), default=len(nodes) - 1)
    nodes.insert(last + 1, write)


def bindings_of(ir_mapping) -> Dict[int, DynamicBankBinding]:
    """The dynamic bank bindings a mapping walk has emitted so far (created on first use)."""
    return ir_mapping.__dict__.setdefault("dynamic_bank_bindings", {})


def emit_operand_product(mapper: Any, ir_mapping: Any, stream: np.ndarray, payload: np.ndarray) -> np.ndarray:
    """Emit ``mapper``'s banks; returns the instances' output sources, (banks, instances, neurons)."""
    banks, axons, neurons = (int(d) for d in payload.shape)
    instances = int(stream.shape[1])
    write_sources = np.asarray(payload, dtype=object).reshape(-1)
    shape = (banks, axons, neurons)
    write = ir_mapping.add_compute_op(
        input_sources=write_sources, op_type=OPERAND_WRITE_OP_TYPE,
        params={"module": mapper.write, "input_shape": shape}, input_shape=shape, output_shape=shape,
        name=f"{mapper.name}_operand_write",
    )
    write_id = _first_node_id(write)
    producers = {int(s.node_id) for s in write_sources if int(s.node_id) >= 0}
    _hoist(ir_mapping, write_id, producers)
    ordering = getattr(ir_mapping, "_node_input_node_ids", None)
    nodes = getattr(ir_mapping, "nodes", None)
    theta = 1.0 / mapper.scale
    out = np.empty((banks, instances, neurons), dtype=object)
    for j, operand in enumerate(mapper.banks):
        bank = ir_mapping.register_weight_bank(
            weights=np.full((neurons, axons), np.nan), biases=None, perceptron_index=operand.perceptron_index)
        for i in range(instances):
            result = ir_mapping.add_shared_neural_core(
                input_sources=np.asarray(stream[j, i], dtype=object), weight_bank_id=bank, has_bias=False,
                name=f"{operand.name}_v{i}", perceptron_index=operand.perceptron_index,
            )
            node_id = _first_node_id(result)
            specs = getattr(ir_mapping, "layout_softcores", None)
            if specs:
                specs[-1] = replace(specs[-1], core_role=OPERAND_ROLE)
            if ordering is not None:
                ordering.setdefault(node_id, set()).add(write_id)
            if nodes is not None:
                nodes[-1].threshold = theta
                nodes[-1].core_role = OPERAND_ROLE
            out[j, i] = np.asarray(result, dtype=object).reshape(-1)
        bindings_of(ir_mapping)[bank] = DynamicBankBinding(
            bank_id=bank, source_node_id=write_id, payload_shape=(axons, neurons),
            transpose=mapper.lowering.stationary != B_STATIONARY, batch_index=operand.batch_index,
            payload_bits=mapper.payload_bits, payload_offset=j * axons * neurons,
        )
    return out
