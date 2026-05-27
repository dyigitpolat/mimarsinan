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


class IRMappingCore(LayoutIRMapping):
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

    # Public mapping entry point

    def map(self, model_representation) -> IRGraph:
        output_sources = super().map(model_representation)
        # Materialise the final output sources -- IRGraph and downstream
        # consumers (hybrid_hardcore_mapping, simulators) treat
        # ``output_sources`` as a real numpy object array with ``.flat``,
        # mutation, etc.  The shape-only ``LayoutSourceView`` lives only
        # inside the mapper chain.
        output_sources = np.asarray(output_sources, dtype=object)
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                sc_idx = self._node_id_to_softcore_idx.get(node.id)
                if sc_idx is not None:
                    node.layout_softcore_index = sc_idx
        return IRGraph(
            nodes=self.nodes.copy(),
            output_sources=output_sources,
            weight_banks=dict(self._weight_banks),
            layout_softcores=list(self.layout_softcores),
        )

    # Source conversion (SpikeSource / IRSource compatibility)

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

    # Emission hooks — construct real IR nodes in addition to shape tracking

    def add_compute_op(
        self,
        input_sources,
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
        # Materialise once per emission so the IRMapping path keeps producing
        # real IRSource arrays end-to-end; downstream mappers and the IR
        # graph never see a ``LayoutSourceView``.
        result = np.asarray(result, dtype=object)
        node_id = int(result.flat[0].node_id)
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
