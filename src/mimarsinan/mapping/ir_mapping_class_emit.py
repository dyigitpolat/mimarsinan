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


class IRMappingEmitMixin:

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
            ir_input_sources = self._convert_sources(input_sources)

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
            result = np.asarray(result, dtype=object)
            node_id = int(result.flat[0].node_id)

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
            ir_input_sources = self._convert_sources(input_sources)

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
            result = np.asarray(result, dtype=object)
            node_id = int(result.flat[0].node_id)
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

