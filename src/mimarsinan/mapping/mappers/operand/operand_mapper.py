"""OperandContractionMapper: a product of two activations lowered onto dynamic operand banks (N2 design §2.4)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.operand.emission import emit_operand_product
from mimarsinan.mapping.mappers.operand.followers import RELU, Follower
from mimarsinan.mapping.mappers.operand.operand_perceptron import OperandPerceptron
from mimarsinan.mapping.support.operand_form import B_STATIONARY, Lowering, bank_indices
from mimarsinan.mapping.support.operand_write import OperandWrite
from mimarsinan.models.nn.activations.integer_law import IntegerLawRangeError, apply_law, grid_codes


class OperandContractionMapper(Mapper):
    """``C = A B`` per bank: one ``OperandPerceptron`` per index of the batch axes, the stationary operand written
    by one ``OperandWrite`` host op, the other streamed through the bank's core instances. Each operand reads
    its producer's output through a flat index map (``gather_chain``); absorbed followers ride in the bank's law."""

    name: str
    reason: str
    payload_bits: int
    followers: List[Follower]
    operand_roots: Tuple[int, ...]
    write: OperandWrite

    def __init__(self, *, name: str, lowering: Lowering, reason: str, roots: Sequence[Mapper],
                 operand_roots: Tuple[int, int], operand_maps: Tuple[np.ndarray, np.ndarray],
                 root_shapes: Sequence[Tuple[int, ...]], operand_shapes: Sequence[Tuple[int, ...]],
                 output_shape: Tuple[int, ...], payload_bits: int, resnap: Sequence[bool],
                 host_fed: Tuple[bool, bool]) -> None:
        super().__init__()
        self.name = str(name)
        self.lowering, self.reason = lowering, str(reason)
        self._sources_list: List[Mapper] = list(roots)
        self.operand_roots = tuple(int(i) for i in operand_roots)
        self.root_shapes = [tuple(int(d) for d in s) for s in root_shapes]
        self.operand_shapes = [tuple(int(d) for d in s) for s in operand_shapes]
        self.output_shape = tuple(int(d) for d in output_shape)
        self.payload_bits = int(payload_bits)
        self.followers = []
        for role, index in enumerate(operand_maps):
            self.register_buffer(f"_map_{role}", torch.as_tensor(np.asarray(index, dtype=np.int64)), persistent=False)
        streamed_role = 0 if lowering.stationary == B_STATIONARY else 1
        self.operand_perceptrons = nn.ModuleList(
            OperandPerceptron(f"{self.name}_bank{'_'.join(map(str, bank)) or '0'}", batch_index=bank,
                              payload_bits=payload_bits, resnap=bool(resnap[j]),
                              streamed_host_fed=bool(host_fed[streamed_role]))
            for j, bank in enumerate(bank_indices(lowering.form)))
        quantizers = [op.payload_quantizer for op in self.banks]
        if any(q is not None for q in quantizers) and not all(q is not None for q in quantizers):
            raise ValueError(f"{self.name}: every bank of one product re-snaps its payload, or none does")
        grids = [q for q in quantizers if q is not None]
        self.write = OperandWrite(banks=lowering.form.banks, payload_shape=(lowering.axons, lowering.neurons),
                                  payload_quantizers=grids if len(grids) == len(quantizers) else None)

    def get_source_mappers(self):
        return [m for m in self._sources_list if m is not None]

    @property
    def banks(self) -> List[OperandPerceptron]:
        """The bank perceptrons, one per index of the product's batch axes, in row-major order."""
        return [cast(OperandPerceptron, op) for op in self.operand_perceptrons]

    @property
    def follower_kinds(self) -> Tuple[str, ...]:
        return tuple(f.kind for f in self.followers)

    @property
    def scale(self) -> float:
        """The product of the absorbed constant factors (1.0 without one)."""
        out = 1.0
        for follower in self.followers:
            if follower.kind != RELU:
                out *= follower.factor
        return out

    @property
    def relu(self) -> bool:
        return RELU in self.follower_kinds

    def operand_map(self, role: int) -> np.ndarray:
        return cast(torch.Tensor, getattr(self, f"_map_{role}")).cpu().numpy()

    def assign_operand_indices(self, start: int) -> int:
        """Give each bank perceptron the next perceptron index (after every static perceptron's)."""
        for offset, op in enumerate(self.banks):
            op.perceptron_index = start + offset
        return start + len(self.banks)

    def streamed_producer_nodes(self, ir_graph) -> set:
        """IR node ids whose outputs the bank instances stream (the producer of the streamed operand)."""
        banks = {b for b, binding in ir_graph.dynamic_bank_bindings.items()
                 if binding.batch_index in {op.batch_index for op in self.banks}}
        indices = {op.perceptron_index for op in self.banks}
        out = set()
        for node in ir_graph.get_neural_cores():
            if node.weight_bank_id in banks and node.perceptron_index in indices:
                out |= {int(s.node_id) for s in node.input_sources.flatten() if int(s.node_id) >= 0}
        return out

    # Model side --------------------------------------------------------------------------------------------

    def _gather(self, inputs, role: int) -> torch.Tensor:
        source = inputs[self.operand_roots[role]]
        flat = source.reshape(source.shape[0], -1)
        index = cast(torch.Tensor, getattr(self, f"_map_{role}"))
        return flat[:, index].reshape(source.shape[0], *self.operand_shapes[role])

    def _canonical(self, x: torch.Tensor, role: int) -> torch.Tensor:
        form = self.lowering.form
        perm = form.a_perm if role == 0 else form.b_perm
        tail = (form.m, form.k) if role == 0 else (form.k, form.n)
        return x.permute(0, *[1 + p for p in perm]).reshape(x.shape[0], form.banks, *tail)

    def _float_bank(self, op: OperandPerceptron, product: torch.Tensor) -> torch.Tensor:
        y = product
        for follower in self.followers:
            y = follower.apply(y)
        return op.readout(y)

    def _integer_bank(self, op: OperandPerceptron, product, stream, payload) -> torch.Tensor:
        stamp = op.stamp
        assert stamp is not None
        device = product.device
        cs = grid_codes(stream, torch.tensor(stamp.input_step, dtype=torch.float64, device=device), where=op.name)
        cp = grid_codes(payload, torch.tensor(stamp.payload_step, dtype=torch.float64, device=device), where=op.name)
        acc = cs @ cp if self.lowering.stationary == B_STATIONARY else cp @ cs
        if not bool(torch.equal(acc, torch.round(acc))):
            raise IntegerLawRangeError(f"{op.name}: the fp64 operand accumulator is not integral")
        law = stamp.on(device)
        shape = (1, -1) if self.lowering.stationary == B_STATIONARY else (-1, 1)
        per = {k: v.reshape(shape) for k, v in law.items()}
        codes = apply_law(acc, m0=per["m0"], bias=per["bias"], shift=per["shift"], lo=per["lo"], hi=per["hi"],
                          adc_shift=per["adc_shift"], adc_levels=per["adc_levels"])
        surrogate = self._float_bank(op, product) if torch.is_grad_enabled() else product
        return (codes * per["output_step"]).to(product.dtype) + (surrogate - surrogate.detach())

    def _forward_impl(self, x):
        inputs = (x,) if len(self._sources_list) == 1 else tuple(x)
        a, b = self._canonical(self._gather(inputs, 0), 0), self._canonical(self._gather(inputs, 1), 1)
        b_stationary = self.lowering.stationary == B_STATIONARY
        banks = []
        for j, op in enumerate(self.banks):
            stream = op.streamed(a[:, j] if b_stationary else b[:, j])
            payload = op.payload(b[:, j] if b_stationary else a[:, j])
            product = stream @ payload if b_stationary else payload @ stream
            banks.append(self._float_bank(op, product) if op.stamp is None
                         else self._integer_bank(op, product, stream, payload))
        form = self.lowering.form
        out = torch.stack(banks, dim=1).reshape(a.shape[0], *form.batch, form.m, form.n)
        return out.permute(0, *[1 + p for p in form.out_perm])

    # IR side ----------------------------------------------------------------------------------------------

    def ir_operands(self, a_ir: np.ndarray, b_ir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(streamed vectors (banks, instances, k), payload (banks, k, neurons)) of the canonical operands."""
        if self.lowering.stationary == B_STATIONARY:
            return a_ir, b_ir
        return np.swapaxes(b_ir, -1, -2), np.swapaxes(a_ir, -1, -2)

    def canonical_ir(self, roots_ir: Sequence, role: int) -> np.ndarray:
        form = self.lowering.form
        flat = np.asarray(roots_ir[self.operand_roots[role]], dtype=object).reshape(-1)
        operand = flat[self.operand_map(role)].reshape(self.operand_shapes[role])
        perm = form.a_perm if role == 0 else form.b_perm
        tail = (form.m, form.k) if role == 0 else (form.k, form.n)
        return operand.transpose(perm).reshape(form.banks, *tail)

    def _map_to_ir(self, ir_mapping):
        roots_ir = [source.map_to_ir(ir_mapping) for source in self._sources_list]
        stream, payload = self.ir_operands(self.canonical_ir(roots_ir, 0), self.canonical_ir(roots_ir, 1))
        out = emit_operand_product(self, ir_mapping, stream, payload)
        form = self.lowering.form
        if self.lowering.stationary != B_STATIONARY:
            out = np.swapaxes(out, -1, -2)
        return out.reshape(*form.batch, form.m, form.n).transpose(form.out_perm)


def operand_mappers(model_repr) -> List[OperandContractionMapper]:
    return [m for m in model_repr.execution_order() if isinstance(m, OperandContractionMapper)]


def operand_perceptrons(model_repr) -> List[OperandPerceptron]:
    """Every bank perceptron of the graph, in execution order."""
    return [op for m in operand_mappers(model_repr) for op in m.banks]


def operand_entry_perceptrons(model_repr) -> List[OperandPerceptron]:
    """The bank perceptrons whose streamed operand is host-fed: each takes its own entry grid."""
    return [op for op in operand_perceptrons(model_repr) if op.streamed_host_fed]


def operand_mapper_of(model_repr, perceptron_index: int) -> Optional[OperandContractionMapper]:
    for mapper in operand_mappers(model_repr):
        if any(op.perceptron_index == perceptron_index for op in mapper.banks):
            return mapper
    return None
