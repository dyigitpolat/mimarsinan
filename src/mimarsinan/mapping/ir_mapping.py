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
from mimarsinan.mapping.mapping_structure import (
    compute_core_input_count,
    compute_fc_tiling_mode,
    compute_psum_params,
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
        allow_core_coalescing: bool = False,
        hardware_bias: bool = False,
    ):
        self.nodes: List[IRNode] = []
        self.output_sources: np.ndarray = np.array([])

        self.q_max = q_max
        self.firing_mode = firing_mode
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.allow_core_coalescing = bool(allow_core_coalescing)
        self.hardware_bias = bool(hardware_bias)

        assert firing_mode in ("Default", "Novena", "TTFS"), \
            f"Invalid firing_mode: {firing_mode!r}"

        self._next_node_id = 0
        self._coalescing_group_counter = 0
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
        perceptron_index: int | None = None,
    ) -> int:
        """Register a shared weight bank and return its ID.

        When ``self.hardware_bias`` is True and biases are provided, the bias
        is stored in ``WeightBank.hardware_bias`` (a plain vector) and the
        core_matrix contains weights only — no extra always-on axon row.
        Each NeuralCore that later references this bank via
        ``add_shared_neural_core`` will receive a copy of that bias as its
        own ``NeuralCore.hardware_bias``.

        When ``self.hardware_bias`` is False (legacy), biases are appended as
        the last row of core_matrix and an always-on source is wired in.

        perceptron_index: Optional index into model.get_perceptrons() for pruning provenance.
        """
        bank_id = self._next_bank_id
        self._next_bank_id += 1

        w = self._to_numpy(weights)  # (out_features, in_features)
        in_features = w.shape[1]
        out_features = w.shape[0]

        core_matrix = np.zeros((in_features, out_features), dtype=float)
        core_matrix[:, :] = w.T

        bank_hw_bias = None
        if biases is not None:
            b = self._to_numpy(biases).flatten()
            if self.hardware_bias:
                # Hardware-bias mode: keep core_matrix weights-only; store bias separately.
                bank_hw_bias = b
            else:
                # Legacy always-on axon row mode.
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
        perceptron_index: int | None = None,
        perceptron_output_slice: tuple[int, int] | None = None,
        perceptron_input_slice: tuple[int, int] | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
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

        hardware_bias_arr = None
        if biases is not None:
            if self.hardware_bias:
                # Hardware-bias mode: bias in dedicated register, no axon slot consumed
                core_matrix = np.zeros((in_features, out_features), dtype=float)
                core_matrix[:, :] = self._to_numpy(weights).T
                hardware_bias_arr = self._to_numpy(biases).flatten()
            else:
                # Legacy always-on mode: bias occupies the last axon row
                core_matrix = np.zeros((in_features + 1, out_features), dtype=float)
                core_matrix[:in_features, :] = self._to_numpy(weights).T
                core_matrix[-1, :] = self._to_numpy(biases).flatten()
                ir_input_list.append(IRSource(node_id=-3, index=0))
        else:
            core_matrix = np.zeros((in_features, out_features), dtype=float)
            core_matrix[:, :] = self._to_numpy(weights).T

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list),
            core_matrix=core_matrix,
            hardware_bias=hardware_bias_arr,
            threshold=1.0,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            perceptron_output_slice=perceptron_output_slice,
            perceptron_input_slice=perceptron_input_slice,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
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
        perceptron_index: int | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
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

        out_features = bank.core_matrix.shape[1]
        if weight_row_slice is None:
            weight_row_slice = (0, out_features)
        else:
            out_features = weight_row_slice[1] - weight_row_slice[0]

        # Determine bias mode.  When the bank carries a hardware_bias vector
        # (set by register_weight_bank when hardware_bias=True) we copy (a
        # slice of) it onto the NeuralCore and do NOT add an always-on axon.
        # Otherwise fall back to the legacy always-on axon row.
        node_hw_bias = None
        if has_bias:
            if bank.hardware_bias is not None:
                start, end = weight_row_slice
                node_hw_bias = bank.hardware_bias[start:end]
            else:
                ir_input_list.append(IRSource(node_id=-3, index=0))

        neural_core = NeuralCore(
            id=node_id,
            name=name or f"neural_core_{node_id}",
            input_sources=np.array(ir_input_list),
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
        perceptron_index: int | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
    ) -> np.ndarray:
        """
        Map a fully-connected layer to IR nodes.

        Creates one NeuralCore per FC layer (or per output tile when
        out_features exceeds max_neurons).  Cores may be wider than
        max_axons — hardware packing (HardCoreMapping) handles axon
        constraints via core fusion.

        ``allow_core_coalescing`` only controls whether coalescing
        metadata (group_id / role) is attached to wide cores.
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
                        perceptron_index=perceptron_index,
                        psum_group_id=psum_group_id,
                        psum_role=psum_role,
                        coalescing_group_id=coalescing_group_id,
                        coalescing_role=coalescing_role,
                    ).flatten()
                )

            out = np.stack(outputs, axis=1)  # (out_features, core_count)
            return out.reshape(tuple(output_shape))

        has_bias = fc_biases is not None
        mode = compute_fc_tiling_mode(
            in_features, out_features,
            self.max_axons, self.max_neurons,
            has_bias, self.hardware_bias, self.allow_core_coalescing,
        )

        if mode == "coalescing":
            if coalescing_group_id is None:
                coalescing_group_id = self._coalescing_group_counter
                self._coalescing_group_counter += 1
                coalescing_role = "master"
            # Coalescing is just metadata; still need output tiling if neurons exceed limit.
        elif mode == "psum":
            return self._map_fc_with_psum(
                input_sources=input_tensor_sources.flatten(),
                weights=fc_weights,
                biases=fc_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=name,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
            )

        if mode == "output_tiled" or (mode == "coalescing" and self.max_neurons is not None and out_features > self.max_neurons):
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
                perceptron_index,
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
            )

        # Single core (fits max_axons, or coalescing enabled for wide core).
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
            perceptron_index=perceptron_index,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
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
        perceptron_index: int | None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
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
                perceptron_index=perceptron_index,
                perceptron_output_slice=(start, end),
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
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
        perceptron_index: int | None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
    ) -> np.ndarray:
        """Map FC with axon tiling using partial sums.

        Mirrors the psum strategy in ``SoftCoreMapping.map_fc`` (soft_core_mapper.py
        lines 179-275):

        1. Split input axons into tiles that fit ``max_axons``.
        2. For each tile × output block: create pos and neg partial-sum NeuralCores
           (``w_pos = clamp(w, min=0)``, ``w_neg = clamp(-w, min=0)``).
        3. Create accumulator NeuralCores that gather pos/neg partials, subtract,
           and add bias.

        The accumulator weight matrix is a sparse identity-like matrix with
        ``+1/parameter_scale`` for pos partials and ``-1/parameter_scale`` for neg.
        """
        in_features = weights.shape[1]
        out_features = weights.shape[0]

        pp = compute_psum_params(
            in_features, out_features,
            int(self.max_axons), self.max_neurons,
            biases is not None, self.hardware_bias,
        )

        src_arr = input_sources.flatten()
        group_id = self._psum_group_counter
        self._psum_group_counter += 1

        all_output_sources: list[np.ndarray] = []
        a = 0
        while a < out_features:
            b = min(out_features, a + pp.out_block_size)
            block = b - a

            w_block = weights[a:b, :]  # (block, in_features)
            b_block = biases[a:b] if biases is not None else None

            # -- partial cores --
            partial_pos_sources: list[np.ndarray] = []
            partial_neg_sources: list[np.ndarray] = []
            for t_idx, (ta, tb) in enumerate(pp.tile_slices):
                w_tile = w_block[:, ta:tb]
                if not torch.is_tensor(w_tile):
                    w_tile = torch.as_tensor(w_tile, dtype=torch.float32)
                w_pos = torch.clamp(w_tile, min=0).numpy()
                w_neg = torch.clamp(-w_tile, min=0).numpy()

                tile_src = src_arr[ta:tb]

                pos_out = self.add_neural_core(
                    input_sources=tile_src,
                    weights=w_pos,
                    biases=None,
                    activation_scale=activation_scale,
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=f"{name}_psum_pos_g{group_id}_t{t_idx}_o{a}_{b}" if name else None,
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                    perceptron_index=perceptron_index,
                    perceptron_input_slice=(ta, tb),
                    perceptron_output_slice=(a, b),
                    psum_group_id=group_id,
                    psum_role="partial_pos",
                )
                neg_out = self.add_neural_core(
                    input_sources=tile_src,
                    weights=w_neg,
                    biases=None,
                    activation_scale=activation_scale,
                    parameter_scale=parameter_scale,
                    input_activation_scale=input_activation_scale,
                    name=f"{name}_psum_neg_g{group_id}_t{t_idx}_o{a}_{b}" if name else None,
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                    perceptron_index=perceptron_index,
                    perceptron_input_slice=(ta, tb),
                    perceptron_output_slice=(a, b),
                    psum_group_id=group_id,
                    psum_role="partial_neg",
                )
                partial_pos_sources.append(pos_out)  # each (block,)
                partial_neg_sources.append(neg_out)

            # -- accumulator core --
            # Gather partial outputs: [pos_t0_n0..nB, pos_t1_n0..nB, ..., neg_t0_n0..nB, ...]
            acc_input_list: list[IRSource] = []
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_input_list.append(partial_pos_sources[t_idx][n])
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_input_list.append(partial_neg_sources[t_idx][n])

            ps = parameter_scale.item() if hasattr(parameter_scale, "item") else float(parameter_scale)
            unit = 1.0 / float(ps)
            acc_axons = 2 * pp.tile_count * block
            acc_w = np.zeros((block, acc_axons), dtype=float)
            pos_off = 0
            neg_off = pp.tile_count * block
            for t_idx in range(pp.tile_count):
                for n in range(block):
                    acc_w[n, pos_off + t_idx * block + n] = unit
                    acc_w[n, neg_off + t_idx * block + n] = -unit

            acc_out = self.add_neural_core(
                input_sources=np.array(acc_input_list),
                weights=acc_w,
                biases=b_block,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=f"{name}_psum_accum_g{group_id}_o{a}_{b}" if name else None,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_output_slice=(a, b),
                psum_group_id=group_id,
                psum_role="accum",
            )
            all_output_sources.append(acc_out)
            a = b

        return np.concatenate(all_output_sources)

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
    allow_core_coalescing: bool = False,
    hardware_bias: bool = False,
) -> IRGraph:
    """
    Convenience function to map a model representation to an IRGraph.

    Args:
        model_representation: ModelRepresentation from mapping_utils.
        q_max: Quantization maximum.
        firing_mode: Firing mode ("Default" or "Novena").
        max_axons: Maximum axons per core.
        max_neurons: Maximum neurons per core.
        allow_core_coalescing: Whether to allow expanding core widths.
        hardware_bias: Whether to use dedicated bias registers.

    Returns:
        IRGraph containing both NeuralCore and ComputeOp nodes.
    """
    ir_mapping = IRMapping(
        q_max=q_max,
        firing_mode=firing_mode,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_core_coalescing=allow_core_coalescing,
        hardware_bias=hardware_bias,
    )
    return ir_mapping.map(model_representation)



