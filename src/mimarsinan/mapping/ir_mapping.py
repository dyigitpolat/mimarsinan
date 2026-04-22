"""IRMapping: full-weight mapping that produces an ``IRGraph`` with concrete
``NeuralCore`` / ``ComputeOp`` / ``WeightBank`` nodes.

All structural decisions (tiling mode, psum decomposition, coalescing,
bias-axon counting, shared-bank wiring) live in the base class
``LayoutIRMapping``.  This subclass only attaches weight material and builds
the graph, guaranteeing the emitted softcore shapes are byte-identical to
what the wizard / architecture-search path predicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRNode,
    IRSource,
    NeuralCore,
    WeightBank,
    spike_source_to_ir_source,
)
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping


class IRMapping(LayoutIRMapping):
    """Unified IR mapping.  Inherits all tiling / dispatch logic from
    ``LayoutIRMapping`` and overrides the emission hooks to additionally
    construct concrete ``IRGraph`` nodes.
    """

    def __init__(
        self,
        q_max: float = 1.0,
        firing_mode: str = "Default",
        max_axons: int | None = None,
        max_neurons: int | None = None,
        allow_coalescing: bool = False,
        hardware_bias: bool = False,
    ):
        super().__init__(
            max_axons=max_axons,
            max_neurons=max_neurons,
            allow_coalescing=allow_coalescing,
            hardware_bias=hardware_bias,
        )

        assert firing_mode in ("Default", "Novena", "TTFS"), \
            f"Invalid firing_mode: {firing_mode!r}"

        self.q_max = q_max
        self.firing_mode = firing_mode

        self.nodes: List[IRNode] = []
        self._weight_banks: Dict[int, WeightBank] = {}

    # ------------------------------------------------------------------
    # Public mapping entry point
    # ------------------------------------------------------------------

    def map(self, model_representation) -> IRGraph:
        output_sources = super().map(model_representation)
        return IRGraph(
            nodes=self.nodes.copy(),
            output_sources=output_sources,
            weight_banks=dict(self._weight_banks),
        )

    # ------------------------------------------------------------------
    # Source conversion (SpikeSource / IRSource compatibility)
    # ------------------------------------------------------------------

    def _convert_sources(self, sources: np.ndarray) -> np.ndarray:
        """Convert a SpikeSource or IRSource array to an IRSource array.

        Fast path: if every element is already an IRSource we return a shallow
        copy of the input (object refs, not new IRSource instances).
        """
        arr = np.asarray(sources, dtype=object)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return np.empty(arr.shape, dtype=object)

        first = flat[0]
        if isinstance(first, IRSource):
            return arr

        result = np.empty(flat.shape, dtype=object)
        for i in range(flat.size):
            src = flat[i]
            if isinstance(src, IRSource):
                result[i] = src
            elif hasattr(src, "is_input_"):
                result[i] = spike_source_to_ir_source(src)
            else:
                raise TypeError(f"Unknown source type: {type(src)}")
        return result.reshape(arr.shape)

    def _to_numpy(self, tensor_or_array) -> np.ndarray:
        if isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        return tensor_or_array.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Emission hooks — construct real IR nodes in addition to shape tracking
    # ------------------------------------------------------------------

    def add_compute_op(
        self,
        input_sources: np.ndarray,
        op_type: str,
        params: Dict[str, Any],
        input_shape: Tuple[int, ...] | None = None,
        output_shape: Tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> np.ndarray:
        ir_input_sources = self._convert_sources(input_sources)
        result = super().add_compute_op(
            input_sources=ir_input_sources,
            op_type=op_type,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            name=name,
        )
        node_id = int(result.flatten()[0].node_id)
        self.nodes.append(ComputeOp(
            id=node_id,
            name=name or f"compute_{op_type}_{node_id}",
            input_sources=ir_input_sources,
            op_type=op_type,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
        ))
        return result

    def register_weight_bank(
        self,
        weights: np.ndarray | torch.Tensor,
        biases: np.ndarray | torch.Tensor | None = None,
        activation_scale: torch.Tensor = torch.tensor(1.0),
        parameter_scale: torch.Tensor = torch.tensor(1.0),
        input_activation_scale: torch.Tensor = torch.tensor(1.0),
        perceptron_index: int | None = None,
    ) -> int:
        bank_id = super().register_weight_bank(
            weights=weights,
            biases=biases,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            perceptron_index=perceptron_index,
        )

        w = self._to_numpy(weights)
        in_features = w.shape[1]
        out_features = w.shape[0]
        core_matrix = np.zeros((in_features, out_features), dtype=float)
        core_matrix[:, :] = w.T

        bank_hw_bias = None
        if biases is not None:
            b = self._to_numpy(biases).flatten()
            if self.hardware_bias:
                bank_hw_bias = b
            else:
                core_matrix = np.zeros((in_features + 1, out_features), dtype=float)
                core_matrix[:in_features, :] = w.T
                core_matrix[-1, :] = b

        self._weight_banks[bank_id] = WeightBank(
            id=bank_id,
            core_matrix=core_matrix,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            perceptron_index=perceptron_index,
            hardware_bias=bank_hw_bias,
        )
        return bank_id

    def add_neural_core(
        self,
        *,
        input_sources: np.ndarray,
        weights: Any,
        biases: Any = None,
        activation_scale: torch.Tensor = torch.tensor(1.0),
        parameter_scale: torch.Tensor = torch.tensor(1.0),
        input_activation_scale: torch.Tensor = torch.tensor(1.0),
        name: str | None = None,
        normalization_type: str | None = None,
        activation_type: str | None = None,
        perceptron_index: int | None = None,
        perceptron_input_slice: tuple[int, int] | None = None,
        perceptron_output_slice: tuple[int, int] | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
    ) -> np.ndarray:
        ir_input_sources = self._convert_sources(np.asarray(input_sources, dtype=object))

        result = super().add_neural_core(
            input_sources=ir_input_sources,
            weights=weights,
            biases=biases,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            name=name,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            perceptron_input_slice=perceptron_input_slice,
            perceptron_output_slice=perceptron_output_slice,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )
        node_id = int(result.flatten()[0].node_id)

        # Psum partials receive the un-clamped tile slice from the base class
        # (avoids clamp cost in shape-only path); materialise pos/neg here.
        w_np = self._to_numpy(weights) if weights is not None else None
        if psum_role == "partial_pos" and w_np is not None:
            w_np = np.clip(w_np, a_min=0, a_max=None)
        elif psum_role == "partial_neg" and w_np is not None:
            w_np = np.clip(-w_np, a_min=0, a_max=None)

        ir_input_list = list(ir_input_sources.flatten())
        in_features = len(ir_input_list)

        hardware_bias_arr: np.ndarray | None = None
        if w_np is None:
            # Safety net — the base class still emitted a shape; this path is
            # not exercised by the current mappers but kept for robustness.
            out_features = int(result.flatten().shape[0])
            core_matrix = np.zeros((in_features, out_features), dtype=float)
        else:
            out_features = w_np.shape[0]
            if biases is not None:
                if self.hardware_bias:
                    core_matrix = np.ascontiguousarray(w_np.T, dtype=float)
                    hardware_bias_arr = self._to_numpy(biases).flatten()
                else:
                    core_matrix = np.empty((in_features + 1, out_features), dtype=float)
                    core_matrix[:in_features, :] = w_np.T
                    core_matrix[-1, :] = self._to_numpy(biases).flatten()
                    ir_input_list.append(IRSource(node_id=-3, index=0))
            else:
                core_matrix = np.ascontiguousarray(w_np.T, dtype=float)

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list, dtype=object),
            core_matrix=core_matrix,
            hardware_bias=hardware_bias_arr,
            threshold=1.0,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            perceptron_input_slice=perceptron_input_slice,
            perceptron_output_slice=perceptron_output_slice,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )
        self.nodes.append(neural_core)
        return result

    def add_shared_neural_core(
        self,
        *,
        input_sources: np.ndarray,
        weight_bank_id: int,
        has_bias: bool = True,
        weight_row_slice: tuple[int, int] | None = None,
        name: str | None = None,
        normalization_type: str | None = None,
        activation_type: str | None = None,
        perceptron_index: int | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
    ) -> np.ndarray:
        ir_input_sources = self._convert_sources(np.asarray(input_sources, dtype=object))

        result = super().add_shared_neural_core(
            input_sources=ir_input_sources,
            weight_bank_id=weight_bank_id,
            has_bias=has_bias,
            weight_row_slice=weight_row_slice,
            name=name,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )
        node_id = int(result.flatten()[0].node_id)
        bank = self._weight_banks[weight_bank_id]

        ir_input_list = list(ir_input_sources.flatten())

        out_features = bank.core_matrix.shape[1]
        if weight_row_slice is None:
            weight_row_slice = (0, out_features)

        node_hw_bias: np.ndarray | None = None
        if has_bias:
            if bank.hardware_bias is not None:
                start, end = weight_row_slice
                node_hw_bias = bank.hardware_bias[start:end]
            else:
                ir_input_list.append(IRSource(node_id=-3, index=0))

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list, dtype=object),
            core_matrix=None,
            hardware_bias=node_hw_bias,
            threshold=1.0,
            activation_scale=bank.activation_scale,
            parameter_scale=bank.parameter_scale,
            input_activation_scale=bank.input_activation_scale,
            weight_bank_id=weight_bank_id,
            weight_row_slice=weight_row_slice,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )
        self.nodes.append(neural_core)
        return result


def map_model_to_ir(
    model_representation,
    q_max: float = 1.0,
    firing_mode: str = "Default",
    max_axons: int | None = None,
    max_neurons: int | None = None,
    allow_coalescing: bool = False,
    hardware_bias: bool = False,
) -> IRGraph:
    """Convenience wrapper around ``IRMapping.map``."""
    return IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    ).map(model_representation)
