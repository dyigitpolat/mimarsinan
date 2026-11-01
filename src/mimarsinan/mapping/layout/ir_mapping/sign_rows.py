"""[U-AX] The physical rows a layout walk records under per-axon weight sign: the build's rule, on the build's grid.

Under per-axon weight sign (``platform_constraints.PER_AXON``) the build splits each mixed-sign row of a soft core into
a W+ and a W- row and keeps an all-zero or single-sign row as one (``sign_split.physical_row_counts``, the rule
``compaction.deployed_row_counts`` reads), on the integer codes the quantizer installs. A walk that holds the network's
weights is given the grid the build reads as a census (``sign_row_census``, the walk's ``sign_rows``): the walk records
each core's matrix as ``IRMapping`` builds it (the transposed weights above the parameter-encoded always-on rows; a
bank once, each instance by its rows and column slice) and, after the walk, the census counts each core's rows by that
rule on that grid: the quantizer's rule for weights with no trained scale (one grid per perceptron at
floor(q_max / peak), the peak over every core and bank of the perceptron), run by ``quantize_ir_graph`` itself on a
scratch graph of the recorded matrices. On a network trained on its integer lattice (NAPQ's install, every parameter
within q_max steps of its scale) that grid's step is no coarser than the trained one, so every nonzero code keeps its
sign and the rows are the trained build's.

A walk without the census, or one holding a core whose weights it cannot read (a shape-only placeholder, a non-finite
payload, a core a lane fusion inserted), keeps two rows per input: an upper bound, stated by name
(``layout_types.PhysicalRows``). The walk's callers construct the census, so the layout package imports none of this.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import IRGraph, WeightBank
from mimarsinan.mapping.layout.layout_types import ROWS_DEPLOYED, ROWS_UPPER_BOUND, PhysicalRows
from mimarsinan.mapping.packing.softcore.sign_split import physical_row_counts
from mimarsinan.mapping.support.bias_rows import core_matrix_with_bias_rows


def sign_row_census(rows_per_input: int, weight_bits: Any) -> Optional["SignRowCensus"]:
    """The census of a walk that holds the network's weights: on a per-axon pool (``rows_per_input`` above 1) with the
    declared ``weight_bits``, else None (a per-synapse walk needs none; without the grid the walk states its bound)."""
    return SignRowCensus(int(weight_bits)) if int(rows_per_input) > 1 and weight_bits is not None else None


def _array(value: Any) -> Optional[np.ndarray]:
    """``value`` as a finite numeric array (a tensor's values), else None (a placeholder or a payload)."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        return None
    return value if np.isfinite(value).all() else None


def _identity(value: Any) -> Hashable:
    """The storage a tensor or array reads (a view of the same values keys the same matrix), else the object."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))
    if isinstance(value, np.ndarray):
        return ("array", value.__array_interface__["data"][0], value.shape, value.strides, value.dtype.str)
    return ("object", id(value))


def core_matrix(weights: Any, biases: Any, *, hardware_bias: bool, bias_rows: int) -> Optional[np.ndarray]:
    """The (axons, neurons) matrix ``IRMapping`` builds for ``weights`` (outputs x inputs) and ``biases``: the
    transposed weights, above the ``bias_rows`` always-on rows where the bias rides axons; None where a part is not
    readable values."""
    w = _array(weights)
    if w is None or w.ndim != 2:
        return None
    if biases is None or hardware_bias:
        return np.ascontiguousarray(w.T, dtype=float)
    b = _array(biases)
    return None if b is None else core_matrix_with_bias_rows(w.T, b.flatten(), bias_rows)


class SignRowCensus:
    """The matrices one per-axon walk records (a core's own, a bank's once) and each emitted core's deployed rows."""

    def __init__(self, weight_bits: int):
        self.weight_bits = int(weight_bits)
        self._matrices: Dict[Hashable, Tuple[np.ndarray, Optional[int]]] = {}
        self._owned: Dict[Hashable, Hashable] = {}
        self._held: List[Any] = []  # the recorded tensors, alive for the walk, so their storage keys them
        self._nodes: Dict[int, Tuple[Hashable, int, Optional[Tuple[int, int]]]] = {}
        self.unread: Optional[str] = None

    def _unread(self, name: Any, why: str) -> None:
        if self.unread is None:
            self.unread = f"core {name!r}: {why}, so the walk keeps two rows per input, an upper bound"

    def owned(self, node_id: int, weights: Any, biases: Any, *, hardware_bias: bool, bias_rows: int,
              perceptron_index: Optional[int], rows: int, name: Any) -> None:
        """A core holding its own weights; one matrix per weight storage, however many cores it serves."""
        key = (_identity(weights), _identity(biases), int(bias_rows), bool(hardware_bias))
        if key not in self._owned:
            matrix = core_matrix(weights, biases, hardware_bias=hardware_bias, bias_rows=bias_rows)
            if matrix is None:
                self._unread(name, "the walk holds no readable weights for it")
                return
            self._owned[key] = ("own", len(self._owned))
            self._matrices[self._owned[key]] = (matrix, perceptron_index)
            self._held += [weights, biases]
        self._nodes[int(node_id)] = (self._owned[key], int(rows), None)

    def bank(self, bank_id: int, weights: Any, biases: Any, *, hardware_bias: bool, bias_rows: int,
             perceptron_index: Optional[int]) -> None:
        """A weight bank's matrix, once; its instances read their rows of it."""
        matrix = core_matrix(weights, biases, hardware_bias=hardware_bias, bias_rows=bias_rows)
        if matrix is not None:
            self._matrices[("bank", int(bank_id))] = (matrix, perceptron_index)

    def instance(self, node_id: int, bank_id: int, *, rows: int, columns: Optional[Tuple[int, int]],
                 name: Any) -> None:
        """A bank-backed core: its first ``rows`` bank rows (as the build's deployed rows read them) over
        ``columns``."""
        if ("bank", int(bank_id)) not in self._matrices:
            self._unread(name, f"its weight bank {bank_id} holds no readable weights")
            return
        span = None if columns is None else (int(columns[0]), int(columns[1]))
        self._nodes[int(node_id)] = (("bank", int(bank_id)), int(rows), span)

    def settle(self, mapping: Any) -> PhysicalRows:
        """After the walk: rewrite each softcore's ``input_count`` to its core's deployed rows plus its fused-add
        operand rows (``deployed``), or, where any core is unread, leave the walk's two rows per input (the bound)."""
        index = mapping._node_id_to_softcore_idx
        rows = self._rows({n: mapping.layout_softcores[sc].name for n, sc in index.items()})
        self._matrices, self._owned, self._held = {}, {}, []  # the walk's weights are no longer held
        if rows is None:
            return PhysicalRows(ROWS_UPPER_BOUND, self.unread)
        for node_id, sc in index.items():
            spec = mapping.layout_softcores[sc]
            mapping.layout_softcores[sc] = replace(spec, input_count=rows[node_id] + int(spec.operand_axons or 0))
        return PhysicalRows(ROWS_DEPLOYED)

    def _rows(self, names: Dict[int, Any]) -> Optional[Dict[int, int]]:
        """Per node of ``names``, its deployed rows on the grid; None (``unread`` says why) if any is unread."""
        for n in names:
            if int(n) not in self._nodes:
                self._unread(names[n], "the walk recorded no weights for it (a core the walk inserted)")
        if self.unread is not None:
            return None
        codes = self._codes()
        counted: Dict[Tuple[Hashable, int, Optional[Tuple[int, int]]], int] = {}
        out: Dict[int, int] = {}
        for n in names:
            entry = self._nodes[int(n)]
            if entry not in counted:
                key, rows, columns = entry
                matrix = codes[key]
                if matrix.shape[0] < rows:
                    self._unread(names[n], f"its matrix holds {matrix.shape[0]} rows for {rows} inputs")
                    return None
                matrix = matrix[:rows] if columns is None else matrix[:rows, columns[0]:columns[1]]
                counted[entry] = int(physical_row_counts(matrix).sum())
            out[int(n)] = counted[entry]
        return out

    def _codes(self) -> Dict[Hashable, np.ndarray]:
        """Every recorded matrix on the build's grid: ``quantize_ir_graph`` at ``weight_bits`` on a scratch graph
        holding each matrix as a bank of its perceptron with no trained scale (the quantizer's own rule and peaks)."""
        keys = list(self._matrices)
        banks = {i: WeightBank(id=i, core_matrix=self._matrices[k][0], parameter_scale=torch.tensor(0.0),
                               perceptron_index=self._matrices[k][1]) for i, k in enumerate(keys)}
        quantize_ir_graph(IRGraph(nodes=[], output_sources=np.array([], dtype=object), weight_banks=banks),
                          self.weight_bits, weight_quantization=True)
        return {k: np.asarray(banks[i].core_matrix) for i, k in enumerate(keys)}
