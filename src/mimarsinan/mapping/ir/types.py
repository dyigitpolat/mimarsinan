"""Unified IR: NeuralCore crossbars, ComputeOps, and shared WeightBanks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mimarsinan.mapping.ir.gather_plan import gather_plan_for
from mimarsinan.mapping.ir.source import (
    decode_ir_sources,
    encode_ir_sources,
)
from mimarsinan.mapping.support.device_placement import preserved_module_placement

from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid

if TYPE_CHECKING:
    from mimarsinan.mapping.ir.graph import IRGraph


@dataclass
class IRNode(ABC):
    """Base class for all IR nodes."""
    id: int
    name: str
    input_sources: np.ndarray

    def __getstate__(self) -> dict:
        """Pickle wiring columnar (~8 B/axon vs ~27.7 as objects)."""
        state = dict(self.__dict__)
        state["input_sources"] = encode_ir_sources(state.get("input_sources"))
        return state

    def __setstate__(self, state: dict) -> None:
        """Decode columnar wiring; legacy object arrays pass through."""
        decoded = dict(state)
        decoded["input_sources"] = decode_ir_sources(decoded.get("input_sources"))
        for key, value in decoded.items():
            object.__setattr__(self, key, value)

    @abstractmethod
    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute this node during simulation."""
        pass

    def gather_inputs(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
        dtype: "torch.dtype | None" = None,
    ) -> torch.Tensor:
        """Gather inputs from sources into a 1D tensor (override for special cases).

        Runs a weakly-cached :class:`GatherPlan` (sources are static); bit-equal
        to the reference walk kept in ``mapping.ir.gather_plan``. ``dtype=None``
        keeps the default-dtype buffer; the value-domain fp64 path widens it.
        """
        return gather_plan_for(self).gather(input_tensor, buffers, dtype=dtype)


@dataclass
class NeuralCore(IRNode):
    """Crossbar neural core: owned core_matrix or shared WeightBank reference."""
    core_matrix: np.ndarray | None = None
    threshold: float = 1.0
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    # [mvm AQ] the realized host->chip value grid; None = float boundary.
    boundary_grid: "BoundaryGrid | None" = None
    latency: int | None = None

    weight_bank_id: int | None = None
    weight_row_slice: tuple[int, int] | None = None

    perceptron_index: int | None = None
    perceptron_output_slice: tuple[int, int] | None = None
    # Token/position column for per-column emission (map_fc 2-D recursion,
    # conv positions); flat output placement = column * channels + slice.
    perceptron_output_column: int | None = None
    perceptron_input_slice: tuple[int, int] | None = None

    psum_group_id: int | None = None
    psum_role: str | None = None
    coalescing_group_id: int | None = None
    coalescing_role: str | None = None
    normalization_type: str | None = None
    activation_type: str | None = None

    hardware_bias: np.ndarray | None = None
    # Two-scale WQ bias grid (parameter_scale / integer r); None == shared grid.
    bias_scale: torch.Tensor | None = None

    layout_softcore_index: int | None = None

    pre_pruning_heatmap: "np.ndarray | None" = None
    pre_pruning_row_mask: list | None = None
    pre_pruning_col_mask: list | None = None
    pruned_row_mask: list | None = None
    pruned_col_mask: list | None = None

    def has_weight_bank(self) -> bool:
        return self.weight_bank_id is not None

    def get_core_matrix(self, graph: "IRGraph | None" = None) -> np.ndarray:
        """Return effective core matrix (resolve weight bank via graph when needed)."""
        if self.core_matrix is not None:
            return self.core_matrix

        if self.weight_bank_id is None:
            raise ValueError(
                f"NeuralCore {self.name} (id={self.id}) has neither an owned "
                f"core_matrix nor a weight_bank_id to resolve one from."
            )

        if graph is None:
            raise ValueError(
                f"NeuralCore {self.name} (id={self.id}) references weight_bank_id="
                f"{self.weight_bank_id} but no IRGraph was provided for resolution."
            )

        bank = graph.get_weight_bank(self.weight_bank_id)
        if bank is None:
            raise KeyError(
                f"NeuralCore {self.name} references weight_bank_id={self.weight_bank_id} "
                f"which does not exist in the graph."
            )

        if self.weight_row_slice is None:
            return bank.core_matrix
        start, end = self.weight_row_slice
        return bank.column_slice(int(start), int(end))


    def resolve_pre_pruning_heatmap(
        self, graph: "IRGraph | None" = None
    ) -> "np.ndarray | None":
        """Pre-compaction weights for GUI heatmaps: owned field, or a view of
        the bank's shared snapshot (never per-core storage)."""
        if self.pre_pruning_heatmap is not None:
            return self.pre_pruning_heatmap
        if self.weight_bank_id is None or graph is None:
            return None
        bank = graph.get_weight_bank(self.weight_bank_id)
        snap = getattr(bank, "pre_pruning_snapshot", None) if bank else None
        if snap is None:
            return None
        if self.weight_row_slice is None:
            return snap
        start, end = self.weight_row_slice
        return snap[:, int(start):int(end)]

    def get_input_count(self) -> int:
        return int(len(self.input_sources.flatten()))

    def get_output_count(self) -> int:
        if self.core_matrix is not None:
            return self.core_matrix.shape[1]
        if self.weight_row_slice is not None:
            return self.weight_row_slice[1] - self.weight_row_slice[0]
        raise ValueError(
            f"Cannot determine output count for bank-backed core {self.name} "
            f"without a concrete core_matrix or weight_row_slice."
        )

    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute crossbar computation: W @ x + activation."""
        inputs = self.gather_inputs(input_tensor, buffers)

        mat = self.core_matrix
        if mat is None:
            raise RuntimeError(
                f"NeuralCore {self.name}: cannot execute without a resolved "
                f"core_matrix.  Call get_core_matrix(graph) first or use "
                f"SpikingHybridCoreFlow which resolves weight banks."
            )
        weight = torch.tensor(
            mat.T,
            dtype=torch.float32,
            device=input_tensor.device,
        )
        out = torch.matmul(inputs, weight.T)

        if self.hardware_bias is not None:
            bias = torch.tensor(
                self.hardware_bias, dtype=torch.float32, device=input_tensor.device
            )
            out = out + bias

        out = F.relu(out)
        out = torch.clamp(out, 0.0, self.activation_scale.item())

        return out


@dataclass
class ComputeOp(IRNode):
    """Non-neural compute op (pooling, norm, attention, etc.)."""
    op_type: str  # Free-form display label; ``params["module"]`` is authoritative.
    params: Dict[str, Any] = field(default_factory=dict)

    input_shape: Tuple[int, ...] | None = None
    output_shape: Tuple[int, ...] | None = None

    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        x = self._gather_structured_input(input_tensor, buffers)
        return self._exec_module(x)

    def execute_on_gathered(self, flat_input: torch.Tensor) -> torch.Tensor:
        """DEPLOYMENT seam: places the host module on the buffer device."""
        return self._exec_module(flat_input)

    def probe_on_gathered(self, flat_input: torch.Tensor) -> torch.Tensor:
        """ANALYSIS seam: same execution, borrowed placement restored exactly.

        ``params["module"]`` is a live reference into the model's mapper graph,
        not a copy. A probe on a fabricated tensor would otherwise half-migrate
        the model — which stranded a host classifier on CPU mid-run on cuda:0.
        """
        with preserved_module_placement(self.params.get("module")):
            return self._exec_module(flat_input)

    def _gather_structured_input(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        return self.gather_inputs(input_tensor, buffers)

    def _exec_module(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat ``(B, N)`` per ``input_shape(s)``, run module, flatten back."""
        module = self.params["module"]
        if hasattr(module, "to"):
            module.to(x.device)
        input_shapes = self.params.get("input_shapes")
        module_kwargs = self.params.get("module_kwargs", {}) or {}
        output_index = self.params.get("output_index")
        if input_shapes is not None:
            inputs = []
            offset = 0
            for shape in input_shapes:
                size = 1
                for dim in shape:
                    size *= dim
                next_offset = offset + size
                inputs.append(x[:, offset:next_offset].view(x.shape[0], *shape))
                offset = next_offset
            call_args = tuple(inputs)
        else:
            input_shape = self.params.get("input_shape")
            if input_shape is not None:
                x = x.view(x.shape[0], *input_shape)
            call_args = (x,)
        with torch.no_grad():
            out = module(*call_args, **module_kwargs)
        # An armed ScaleNormalizingWrapper OWNS output_index (it must select
        # the tuple element before scaling); re-selecting here sliced the
        # BATCH on attention ops [calculus §16.12].
        if output_index is not None and getattr(module, "output_index", None) is None:
            out = out[output_index]
        # Explicit feature size keeps the reshape well-defined for empty batches (-1 is ambiguous at 0 elements).
        features = 1
        for dim in out.shape[1:]:
            features *= dim
        return out.reshape(out.shape[0], features)


