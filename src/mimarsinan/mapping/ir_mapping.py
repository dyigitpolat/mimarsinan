"""
IRMapping: Unified mapping that produces IRGraph with both NeuralCore and ComputeOp nodes.

This extends the SoftCoreMapping concept to handle non-neural operations (pooling, etc.)
in addition to perceptron-based layers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


class IRMapping:
    """
    Unified mapping that produces an IRGraph containing both:
    - NeuralCore: For perceptron/linear layers
    - ComputeOp: For non-neural operations (pooling, etc.)

    The mapping maintains the execution order and source references
    between all node types.
    """

    def __init__(
        self,
        q_max: float = 1.0,
        firing_mode: str = "Default",
        max_axons: int | None = None,
        max_neurons: int | None = None,
        allow_axon_tiling: bool = False,
    ):
        self.nodes: List[IRNode] = []
        self.output_sources: np.ndarray = np.array([])

        self.q_max = q_max
        self.firing_mode = firing_mode
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.allow_axon_tiling = bool(allow_axon_tiling)

        assert firing_mode in ("Default", "Novena", "TTFS"), \
            f"Invalid firing_mode: {firing_mode!r}"

        self._next_node_id = 0
        self._psum_group_counter = 0
        self._next_bank_id = 0
        self._weight_banks: Dict[int, WeightBank] = {}

    def _allocate_node_id(self) -> int:
        """Allocate a unique node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def register_weight_bank(
        self,
        weights: np.ndarray | torch.Tensor,
        biases: np.ndarray | torch.Tensor | None = None,
        activation_scale: torch.Tensor = torch.tensor(1.0),
        parameter_scale: torch.Tensor = torch.tensor(1.0),
        input_activation_scale: torch.Tensor = torch.tensor(1.0),
    ) -> int:
        """Register a shared weight bank and return its ID.

        The bank stores a ``core_matrix`` in the standard IR layout
        ``(axons, neurons)``.  If ``biases`` is provided the matrix
        includes an extra "always-on" axon row for the bias, matching
        the convention of ``add_neural_core``.
        """
        bank_id = self._next_bank_id
        self._next_bank_id += 1

        w = self._to_numpy(weights)  # (out_features, in_features)
        in_features = w.shape[1]
        out_features = w.shape[0]

        if biases is not None:
            b = self._to_numpy(biases).flatten()
            core_matrix = np.zeros((in_features + 1, out_features), dtype=float)
            core_matrix[:in_features, :] = w.T
            core_matrix[-1, :] = b
        else:
            core_matrix = np.zeros((in_features, out_features), dtype=float)
            core_matrix[:, :] = w.T

        self._weight_banks[bank_id] = WeightBank(
            id=bank_id,
            core_matrix=core_matrix,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
        )
        return bank_id

    def map(self, model_representation) -> IRGraph:
        """
        Map a ModelRepresentation to an IRGraph.

        This traverses the mapper graph and produces IR nodes for each mapper.
        """
        output_sources = model_representation.map_to_ir(self)
        self.output_sources = output_sources

        return IRGraph(
            nodes=self.nodes.copy(),
            output_sources=output_sources,
            weight_banks=dict(self._weight_banks),
        )

    def add_compute_op(
        self,
        input_sources: np.ndarray,
        op_type: str,
        params: Dict[str, Any],
        input_shape: Tuple[int, ...] | None = None,
        output_shape: Tuple[int, ...] | None = None,
        name: str | None = None,
    ) -> np.ndarray:
        """
        Add a ComputeOp node to the graph.

        Args:
            input_sources: Array of IRSource (or SpikeSource for compat) for inputs.
            op_type: Type of operation ("max_pool2d", "avg_pool2d", etc.).
            params: Operation parameters.
            input_shape: Shape of input excluding batch (e.g., (C, H, W)).
            output_shape: Shape of output excluding batch.
            name: Optional name for debugging.

        Returns:
            Array of IRSource pointing to this node's outputs.
        """
        node_id = self._allocate_node_id()

        # Convert input sources to IRSource if needed
        ir_input_sources = self._convert_sources(input_sources)

        # Compute output size
        if output_shape is not None:
            output_size = 1
            for d in output_shape:
                output_size *= d
        else:
            output_size = len(ir_input_sources.flatten())

        compute_op = ComputeOp(
            id=node_id,
            name=name or f"compute_{op_type}_{node_id}",
            input_sources=ir_input_sources,
            op_type=op_type,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        self.nodes.append(compute_op)

        # Create output sources pointing to this node
        output_sources = np.array([
            IRSource(node_id=node_id, index=i) for i in range(output_size)
        ])

        if output_shape is not None:
            output_sources = output_sources.reshape(output_shape)

        return output_sources

    def add_neural_core(
        self,
        input_sources: np.ndarray,
        weights: np.ndarray,
        biases: np.ndarray | None = None,
        activation_scale: torch.Tensor = torch.tensor(1.0),
        parameter_scale: torch.Tensor = torch.tensor(1.0),
        input_activation_scale: torch.Tensor = torch.tensor(1.0),
        name: str | None = None,
        normalization_type: str | None = None,
        activation_type: str | None = None,
    ) -> np.ndarray:
        """
        Add a NeuralCore node to the graph.

        This is the core operation for perceptron-based layers.

        Args:
            input_sources: Array of sources for the core's axons.
            weights: Weight matrix (out_features, in_features).
            biases: Optional bias vector.
            activation_scale: Activation scale for quantization.
            parameter_scale: Parameter scale for quantization.
            input_activation_scale: Input activation scale.
            name: Optional name for debugging.

        Returns:
            Array of IRSource pointing to this core's outputs.
        """
        node_id = self._allocate_node_id()

        # Convert input sources
        ir_input_sources = self._convert_sources(input_sources)
        ir_input_list = list(ir_input_sources.flatten())

        # Build core matrix (axons x neurons)
        in_features = len(ir_input_list)
        out_features = weights.shape[0]

        if biases is not None:
            # Add bias row
            core_matrix = np.zeros((in_features + 1, out_features), dtype=float)
            core_matrix[:in_features, :] = self._to_numpy(weights).T
            core_matrix[-1, :] = self._to_numpy(biases).flatten()
            # Add always-on source for bias
            ir_input_list.append(IRSource(node_id=-3, index=0))
        else:
            core_matrix = np.zeros((in_features, out_features), dtype=float)
            core_matrix[:, :] = self._to_numpy(weights).T

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list),
            core_matrix=core_matrix,
            threshold=1.0,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            normalization_type=normalization_type,
            activation_type=activation_type,
        )
        self.nodes.append(neural_core)

        # Create output sources
        output_sources = np.array([
            IRSource(node_id=node_id, index=i) for i in range(out_features)
        ])

        return output_sources

    def add_shared_neural_core(
        self,
        input_sources: np.ndarray,
        weight_bank_id: int,
        has_bias: bool = True,
        weight_row_slice: tuple[int, int] | None = None,
        name: str | None = None,
        normalization_type: str | None = None,
        activation_type: str | None = None,
    ) -> np.ndarray:
        """Add a NeuralCore that references a shared ``WeightBank``.

        No weight matrix is copied; only the bank ID, input wiring, and
        an optional output-channel slice are stored.

        Args:
            input_sources: Patch sources (without bias axon — it is added
                automatically when ``has_bias`` is True).
            weight_bank_id: ID returned by ``register_weight_bank``.
            has_bias: Whether the bank's core_matrix includes a bias row
                (an always-on source is appended to ``input_sources``).
            weight_row_slice: ``(start, end)`` slice along the neuron
                (column) axis of the bank's core_matrix.  ``None`` means
                use the full bank.
            name: Optional debug name.

        Returns:
            Array of ``IRSource`` pointing to this core's outputs.
        """
        node_id = self._allocate_node_id()

        bank = self._weight_banks[weight_bank_id]

        ir_input_sources = self._convert_sources(input_sources)
        ir_input_list = list(ir_input_sources.flatten())

        if has_bias:
            ir_input_list.append(IRSource(node_id=-3, index=0))

        out_features = bank.core_matrix.shape[1]
        if weight_row_slice is None:
            weight_row_slice = (0, out_features)
        else:
            out_features = weight_row_slice[1] - weight_row_slice[0]

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list),
            core_matrix=None,
            threshold=1.0,
            activation_scale=bank.activation_scale,
            parameter_scale=bank.parameter_scale,
            input_activation_scale=bank.input_activation_scale,
            weight_bank_id=weight_bank_id,
            weight_row_slice=weight_row_slice,
            normalization_type=normalization_type,
            activation_type=activation_type,
        )
        self.nodes.append(neural_core)

        output_sources = np.array([
            IRSource(node_id=node_id, index=i) for i in range(out_features)
        ])
        return output_sources

    def map_fc(
        self,
        input_tensor_sources: np.ndarray,
        output_shape: np.ndarray,
        fc_weights: np.ndarray | torch.Tensor,
        fc_biases: np.ndarray | torch.Tensor | None = None,
        activation_scale: torch.Tensor = torch.tensor(1.0),
        parameter_scale: torch.Tensor = torch.tensor(1.0),
        input_activation_scale: torch.Tensor = torch.tensor(1.0),
        name: str | None = None,
        normalization_type: str | None = None,
        activation_type: str | None = None,
    ) -> np.ndarray:
        """
        Map a fully-connected layer to IR nodes.

        This handles:
        - Simple FC (single NeuralCore)
        - Output-channel tiling (multiple NeuralCores)
        - Axon tiling with partial sums (if allow_axon_tiling is True)
        """
        fc_weights = self._to_numpy(fc_weights)
        if fc_biases is not None:
            fc_biases = self._to_numpy(fc_biases)

        out_features = fc_weights.shape[0]
        in_features = fc_weights.shape[1]
        src_arr = np.array(input_tensor_sources, dtype=object)

        # Support mapping a *batch* of independent FC applications:
        # input sources shape (in_features, core_count) -> output sources shape (out_features, core_count)
        if src_arr.ndim == 2:
            # Allow callers to pass transposed: (core_count, in_features)
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                src_arr = src_arr.T

            if src_arr.shape[0] != in_features:
                raise ValueError(
                    f"IRMapping.map_fc: input sources first dim must match in_features "
                    f"({src_arr.shape} vs in_features={in_features})"
                )

            core_count = int(src_arr.shape[1])
            outputs = []
            for i in range(core_count):
                col_sources = np.array(src_arr[:, i], dtype=object).flatten()
                outputs.append(
                    self.map_fc(
                        col_sources,
                        np.array([out_features, 1]),
                        fc_weights,
                        fc_biases,
                        activation_scale,
                        parameter_scale,
                        input_activation_scale,
                        name=(f"{name}_col{i}" if name else None),
                        normalization_type=normalization_type,
                        activation_type=activation_type,
                    ).flatten()
                )

            out = np.stack(outputs, axis=1)  # (out_features, core_count)
            return out.reshape(tuple(output_shape))

        input_count = src_arr.flatten().shape[0]

        # Check axon limits
        if self.max_axons is not None and in_features > self.max_axons - 1:
            if not self.allow_axon_tiling:
                raise ValueError(
                    f"FC layer requires {in_features} axons but max is {self.max_axons - 1}. "
                    f"Enable allow_axon_tiling for partial sum decomposition."
                )
            return self._map_fc_with_psum(
                input_tensor_sources,
                fc_weights,
                fc_biases,
                activation_scale,
                parameter_scale,
                input_activation_scale,
                name,
                normalization_type,
                activation_type,
            )

        # Check neuron limits and tile output channels if needed
        if self.max_neurons is not None and out_features > self.max_neurons:
            return self._map_fc_output_tiled(
                input_tensor_sources,
                fc_weights,
                fc_biases,
                activation_scale,
                parameter_scale,
                input_activation_scale,
                name,
                normalization_type,
                activation_type,
            )

        # Simple case: single core
        return self.add_neural_core(
            input_sources=input_tensor_sources.T if len(input_tensor_sources.shape) > 1 else input_tensor_sources,
            weights=fc_weights,
            biases=fc_biases,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            name=name,
            normalization_type=normalization_type,
            activation_type=activation_type,
        )

    def _map_fc_output_tiled(
        self,
        input_sources: np.ndarray,
        weights: np.ndarray,
        biases: np.ndarray | None,
        activation_scale: torch.Tensor,
        parameter_scale: torch.Tensor,
        input_activation_scale: torch.Tensor,
        name: str | None,
        normalization_type: str | None,
        activation_type: str | None,
    ) -> np.ndarray:
        """Tile output channels across multiple cores."""
        out_features = weights.shape[0]
        chunk_size = self.max_neurons

        output_sources_list = []
        start = 0

        while start < out_features:
            end = min(start + chunk_size, out_features)
            tile_weights = weights[start:end, :]
            tile_biases = biases[start:end] if biases is not None else None

            tile_sources = self.add_neural_core(
                input_sources=input_sources.flatten(),
                weights=tile_weights,
                biases=tile_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=f"{name}_tile_{start}_{end}" if name else None,
                normalization_type=normalization_type,
                activation_type=activation_type,
            )
            output_sources_list.append(tile_sources)
            start = end

        return np.concatenate(output_sources_list)

    def _map_fc_with_psum(
        self,
        input_sources: np.ndarray,
        weights: np.ndarray,
        biases: np.ndarray | None,
        activation_scale: torch.Tensor,
        parameter_scale: torch.Tensor,
        input_activation_scale: torch.Tensor,
        name: str | None,
        normalization_type: str | None,
        activation_type: str | None,
    ) -> np.ndarray:
        """
        Map FC with axon tiling using partial sums.

        Strategy:
        1. Split inputs into tiles that fit max_axons
        2. Create pos/neg partial sum cores for each tile
        3. Create accumulator cores that sum partials and add bias
        """
        out_features = weights.shape[0]
        in_features = weights.shape[1]

        # Tile size for axons (leave room for bias in accumulator)
        tile_size = self.max_axons - 1
        tile_count = (in_features + tile_size - 1) // tile_size

        psum_group_id = self._psum_group_counter
        self._psum_group_counter += 1

        # Also tile output features if needed
        out_chunk = self.max_neurons if self.max_neurons else out_features
        mapped = []

        for out_start in range(0, out_features, out_chunk):
            out_end = min(out_start + out_chunk, out_features)
            out_block = out_end - out_start

            partial_sources = []

            # Create partial sum cores for each input tile
            for tile_idx in range(tile_count):
                in_start = tile_idx * tile_size
                in_end = min(in_start + tile_size, in_features)

                tile_inputs = input_sources.flatten()[in_start:in_end]
                tile_weights = weights[out_start:out_end, in_start:in_end]

                # Separate positive and negative weights
                pos_weights = np.maximum(tile_weights, 0)
                neg_weights = np.minimum(tile_weights, 0)

                # Positive partial
                pos_sources = self.add_neural_core(
                    input_sources=tile_inputs,
                    weights=pos_weights,
                    biases=None,
                    activation_scale=torch.tensor(float("inf")),  # No clamping
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=f"{name}_psum_pos_t{tile_idx}_o{out_start}" if name else None,
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                )

                # Negative partial (absolute values)
                neg_sources = self.add_neural_core(
                    input_sources=tile_inputs,
                    weights=-neg_weights,  # Make positive
                    biases=None,
                    activation_scale=torch.tensor(float("inf")),
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=f"{name}_psum_neg_t{tile_idx}_o{out_start}" if name else None,
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                )

                partial_sources.append((pos_sources, neg_sources))

            # Create accumulator core
            # Accumulator inputs: all pos partials, then all neg partials
            acc_inputs = []
            for pos, neg in partial_sources:
                acc_inputs.extend(pos.flatten().tolist())
            for pos, neg in partial_sources:
                acc_inputs.extend(neg.flatten().tolist())

            # Accumulator weights: +1 for pos, -1 for neg
            ps = float(parameter_scale.item() if hasattr(parameter_scale, "item") else parameter_scale)
            unit = 1.0 / ps

            acc_in_count = len(acc_inputs)
            acc_weights = np.zeros((out_block, acc_in_count), dtype=float)

            pos_offset = 0
            neg_offset = tile_count * out_block

            for t_idx in range(tile_count):
                for n in range(out_block):
                    acc_weights[n, pos_offset + t_idx * out_block + n] = unit
                    acc_weights[n, neg_offset + t_idx * out_block + n] = -unit

            tile_biases = biases[out_start:out_end] if biases is not None else None

            acc_sources = self.add_neural_core(
                input_sources=np.array(acc_inputs),
                weights=acc_weights,
                biases=tile_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=f"{name}_psum_accum_o{out_start}" if name else None,
                normalization_type=normalization_type,
                activation_type=activation_type,
            )

            mapped.append(acc_sources)

        return np.concatenate(mapped)

    def _convert_sources(self, sources: np.ndarray) -> np.ndarray:
        """Convert SpikeSource or IRSource array to IRSource array."""
        flat = sources.flatten()
        result = []

        for src in flat:
            if isinstance(src, IRSource):
                result.append(src)
            elif hasattr(src, "is_input_"):  # SpikeSource
                result.append(spike_source_to_ir_source(src))
            else:
                raise TypeError(f"Unknown source type: {type(src)}")

        return np.array(result).reshape(sources.shape)

    def _to_numpy(self, tensor_or_array) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        return tensor_or_array.detach().cpu().numpy()


def map_model_to_ir(
    model_representation,
    q_max: float = 1.0,
    firing_mode: str = "Default",
    max_axons: int | None = None,
    max_neurons: int | None = None,
    allow_axon_tiling: bool = False,
) -> IRGraph:
    """
    Convenience function to map a model representation to an IRGraph.

    Args:
        model_representation: ModelRepresentation from mapping_utils.
        q_max: Quantization maximum.
        firing_mode: Firing mode ("Default" or "Novena").
        max_axons: Maximum axons per core.
        max_neurons: Maximum neurons per core.
        allow_axon_tiling: Whether to allow axon tiling with partial sums.

    Returns:
        IRGraph containing both NeuralCore and ComputeOp nodes.
    """
    ir_mapping = IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_axon_tiling=allow_axon_tiling,
    )
    return ir_mapping.map(model_representation)



