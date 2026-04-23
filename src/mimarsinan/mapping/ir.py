"""
Unified Intermediate Representation (IR) for mimarsinan.

This IR supports both:
1. NeuralCore: Crossbar-based neural computation (matrix multiply + threshold)
2. ComputeOp: Non-neural operations (pooling, element-wise ops, etc.)

The IR enables simulation and hardware mapping of networks that mix
perceptron-based layers with non-neural operations.

Weight banks allow convolution-style layers to share a single weight matrix
across many NeuralCores (one per spatial position) without duplicating the
matrix in memory.  See ``WeightBank``, ``NeuralCore.weight_bank_id``, and
``IRGraph.weight_banks``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _broadcast_scale_to_dim(scale: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Expand a 1-D scale vector to *target_dim* using repeat or mean-fill."""
    if scale.shape[0] == target_dim:
        return scale
    if target_dim % scale.shape[0] == 0:
        return scale.repeat_interleave(target_dim // scale.shape[0])
    return torch.full((target_dim,), scale.mean().item(), dtype=scale.dtype, device=scale.device)


# ---------------------------------------------------------------------------
# WeightBank: Shared weight storage for conv-style layers
# ---------------------------------------------------------------------------
@dataclass
class WeightBank:
    """
    A single stored weight matrix (and optional bias) shared by multiple
    NeuralCores.

    Convolution layers map to many cores that differ only in their input
    wiring (receptive field position) but share the same kernel weights.
    Storing one ``WeightBank`` per conv layer and letting each core
    *reference* it (via ``NeuralCore.weight_bank_id``) avoids duplicating
    the kernel ``h_out * w_out`` times in the IR and in simulation.

    The ``core_matrix`` stored here has the same layout as
    ``NeuralCore.core_matrix``: shape ``(axons, neurons)`` — weights only,
    no bias row.  When ``hardware_bias`` is set, each NeuralCore that
    references this bank receives a copy (or slice) of that array as its
    own ``hardware_bias`` at construction time.
    """
    id: int
    core_matrix: np.ndarray  # (axons, neurons) — weights only, no bias row
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    # Pruning provenance: index into model.get_perceptrons() when this bank backs a single perceptron
    perceptron_index: int | None = None
    # Hardware-bias mode: bias vector shared across all cores that reference this bank.
    # When set, add_shared_neural_core copies (a slice of) this into NeuralCore.hardware_bias
    # instead of appending an always-on axon row.
    hardware_bias: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Source: Where an input to a node comes from
# ---------------------------------------------------------------------------
@dataclass
class IRSource:
    """
    Describes where an input element comes from:
    - node_id >= 0: output from another node
    - node_id == -1: always off (zero)
    - node_id == -2: from the original input tensor
    - node_id == -3: always on (constant 1)
    """
    node_id: int
    index: int  # Which output index from that node

    def is_off(self) -> bool:
        return self.node_id == -1

    def is_input(self) -> bool:
        return self.node_id == -2

    def is_always_on(self) -> bool:
        return self.node_id == -3


# ---------------------------------------------------------------------------
# Base IR Node
# ---------------------------------------------------------------------------
@dataclass
class IRNode(ABC):
    """Base class for all IR nodes."""
    id: int
    name: str
    input_sources: np.ndarray  # Array of IRSource objects (shape depends on node type)

    @abstractmethod
    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute this node during simulation.

        Args:
            input_tensor: The original input to the network (flattened).
            buffers: Dict mapping node_id -> output tensor from that node.

        Returns:
            Output tensor for this node.
        """
        pass

    def gather_inputs(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Gather inputs from sources into a tensor suitable for this node.

        Default: 1D gather for simple sources. Override for special cases.
        """
        batch_size = input_tensor.shape[0]
        sources = self.input_sources.flatten()
        result = torch.zeros(batch_size, len(sources), device=input_tensor.device)

        for idx, src in enumerate(sources):
            if src.is_off():
                continue  # result[:, idx] already zero
            elif src.is_input():
                result[:, idx] = input_tensor[:, src.index]
            elif src.is_always_on():
                result[:, idx] = 1.0
            else:
                result[:, idx] = buffers[src.node_id][:, src.index]

        return result


# ---------------------------------------------------------------------------
# NeuralCore: Crossbar-based computation
# ---------------------------------------------------------------------------
@dataclass
class NeuralCore(IRNode):
    """
    A crossbar-based neural core.

    Computes: activation(matmul(core_matrix, inputs))

    This is the hardware-mappable primitive for spiking neural network chips.

    **Weight ownership modes (mutually exclusive):**

    1. *Owned weights* (default / FC layers): ``core_matrix`` is a concrete
       ``np.ndarray`` and ``weight_bank_id`` is ``None``.
    2. *Shared weights* (conv layers): ``weight_bank_id`` references a
       ``WeightBank`` stored on the owning ``IRGraph``.  ``core_matrix``
       is ``None``; use ``get_core_matrix(graph)`` to resolve the actual
       matrix.  Optional ``weight_row_slice`` restricts to a subset of
       the bank's rows (output-channel tiling).
    """
    core_matrix: np.ndarray | None = None  # (axons, neurons); None when using a bank
    threshold: float = 1.0
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    latency: int | None = None

    # Shared-weight support
    weight_bank_id: int | None = None
    weight_row_slice: tuple[int, int] | None = None  # (start, end) neuron slice into the bank

    # Pruning provenance: index into model.get_perceptrons(); slice of perceptron output dim for tiled FC
    perceptron_index: int | None = None
    perceptron_output_slice: tuple[int, int] | None = None  # (start, end) for owned tiled cores
    perceptron_input_slice: tuple[int, int] | None = None  # (start, end) for axon dimension (e.g. psum tiles)

    # Metadata for debugging/visualization
    psum_group_id: int | None = None
    psum_role: str | None = None  # "partial_pos", "partial_neg", "accum"
    coalescing_group_id: int | None = None
    coalescing_role: str | None = None  # "master", "slave"
    normalization_type: str | None = None
    activation_type: str | None = None

    # Hardware-bias mode: bias stored in a dedicated register, not as an always-on axon row.
    # When set, core_matrix has shape (in_features, out_features) — no extra bias row.
    # When None, bias is encoded as the last axon row wired to IRSource(-3, 0) (legacy mode).
    hardware_bias: np.ndarray | None = None

    # Pre-pruning snapshot for GUI (set by ir_pruning before compacting, opt-in via store_heatmap)
    pre_pruning_heatmap: "np.ndarray | None" = None  # full matrix (axons, neurons) float32 ndarray for soft-core viz; None by default
    pre_pruning_row_mask: list | None = None  # pre-compaction row mask for GUI red markings (same length as pre_pruning_heatmap rows)
    pre_pruning_col_mask: list | None = None  # pre-compaction col mask for GUI red markings (same length as pre_pruning_heatmap cols)
    pruned_row_mask: list | None = None  # bool per row (True = pruned); post-compaction length for soft-core conversion
    pruned_col_mask: list | None = None  # bool per column (True = pruned); post-compaction length for soft-core conversion

    # ------------------------------------------------------------------
    # Weight resolution
    # ------------------------------------------------------------------
    def has_weight_bank(self) -> bool:
        return self.weight_bank_id is not None

    def get_core_matrix(self, graph: "IRGraph | None" = None) -> np.ndarray:
        """Return the effective core matrix, resolving weight-bank references.

        For owned-weight cores this simply returns ``self.core_matrix``.
        For shared-weight cores ``graph`` must be provided so the bank can
        be looked up.
        """
        if self.core_matrix is not None:
            return self.core_matrix

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

        mat = bank.core_matrix
        if self.weight_row_slice is not None:
            start, end = self.weight_row_slice
            mat = mat[:, start:end]
        return mat

    # ------------------------------------------------------------------

    def get_input_count(self) -> int:
        return int(len(self.input_sources.flatten()))

    def get_output_count(self) -> int:
        if self.core_matrix is not None:
            return self.core_matrix.shape[1]
        # Bank-backed: output count from the slice or the full bank
        if self.weight_row_slice is not None:
            return self.weight_row_slice[1] - self.weight_row_slice[0]
        # Fallback — should not normally be reached without a graph
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
                f"SpikingUnifiedCoreFlow which resolves weight banks."
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


# ---------------------------------------------------------------------------
# ComputeOp: Non-neural operations
# ---------------------------------------------------------------------------
@dataclass
class ComputeOp(IRNode):
    """
    A non-neural compute operation (pooling, element-wise, reshape, etc.).

    These operations are executed on the chip's auxiliary compute units,
    not on crossbar cores.
    """
    op_type: str  # "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "flatten", etc.
    params: Dict[str, Any] = field(default_factory=dict)

    # Shape info for reshaping inputs if needed
    input_shape: Tuple[int, ...] | None = None
    output_shape: Tuple[int, ...] | None = None

    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute the non-neural operation (gather + reshape + dispatch)."""
        x = self._gather_structured_input(input_tensor, buffers)
        return self._dispatch(x)

    def execute_on_gathered(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Execute on a pre-gathered flat input tensor ``(B, N)``.

        Each ``_exec_*`` method is responsible for its own reshaping via
        ``self.input_shape`` when spatial dimensions are needed.
        """
        return self._dispatch(flat_input)

    def _dispatch(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_type == "max_pool2d":
            return self._exec_max_pool2d(x)
        elif self.op_type == "avg_pool2d":
            return self._exec_avg_pool2d(x)
        elif self.op_type == "adaptive_avg_pool2d":
            return self._exec_adaptive_avg_pool2d(x)
        elif self.op_type == "flatten":
            return self._exec_flatten(x)
        elif self.op_type == "identity":
            return x.view(x.shape[0], -1)
        elif self.op_type == "layer_norm":
            return self._exec_layer_norm(x)
        elif self.op_type == "gelu":
            return self._exec_gelu(x)
        elif self.op_type == "multi_head_attention":
            return self._exec_multi_head_attention(x)
        elif self.op_type == "add_constant":
            return self._exec_add_constant(x)
        elif self.op_type == "concat_constant":
            return self._exec_concat_constant(x)
        elif self.op_type == "select":
            return self._exec_select(x)
        elif self.op_type == "add":
            return self._exec_add(x)
        elif self.op_type == "mean":
            return self._exec_mean(x)
        elif self.op_type == "dropout":
            return self._exec_dropout(x)
        elif self.op_type == "linear":
            return self._exec_linear(x)
        elif self.op_type == "module":
            return self._exec_module(x)
        else:
            raise NotImplementedError(f"ComputeOp: unsupported op_type '{self.op_type}'")

    def _gather_structured_input(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Gather inputs into a flat ``(B, N)`` tensor.

        Each ``_exec_*`` method reshapes internally when spatial dims are needed.
        """
        return self.gather_inputs(input_tensor, buffers)  # (B, N)

    def _exec_max_pool2d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], *self.input_shape)
        kernel_size = self.params.get("kernel_size", 2)
        stride = self.params.get("stride", kernel_size)
        padding = self.params.get("padding", 0)
        y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        return y.view(y.shape[0], -1)

    def _exec_avg_pool2d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], *self.input_shape)
        kernel_size = self.params.get("kernel_size", 2)
        stride = self.params.get("stride", kernel_size)
        padding = self.params.get("padding", 0)
        y = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        return y.view(y.shape[0], -1)

    def _exec_adaptive_avg_pool2d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], *self.input_shape)
        output_size = self.params.get("output_size", (1, 1))
        y = F.adaptive_avg_pool2d(x, output_size)
        return y.view(y.shape[0], -1)

    def _exec_flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    # ---- ViT / Transformer ComputeOps ---------------------------------

    def _exec_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm across the last dimension(s)."""
        if self.input_shape is not None and len(self.input_shape) >= 2:
            x = x.view(x.shape[0], *self.input_shape)
        weight = torch.tensor(self.params["weight"], dtype=x.dtype, device=x.device)
        bias = torch.tensor(self.params["bias"], dtype=x.dtype, device=x.device)
        eps = self.params.get("eps", 1e-5)
        normalized_shape = self.params["normalized_shape"]
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
        return y.view(y.shape[0], -1)

    def _exec_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Element-wise GELU."""
        y = F.gelu(x)
        return y.view(y.shape[0], -1)

    def _exec_multi_head_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled dot-product multi-head self-attention.

        Expects input laid out as [Q; K; V] concatenated along the flat dim,
        i.e. x shape is (B, 3*S*D).  ``input_shape`` should be ``(3, S, D)``.
        """
        S = self.params["seq_len"]
        D = self.params["d_model"]
        H = self.params["num_heads"]
        d_k = D // H

        B = x.shape[0]
        x = x.view(B, 3, S, D)
        Q, K, V = x[:, 0], x[:, 1], x[:, 2]  # each (B, S, D)

        Q = Q.view(B, S, H, d_k).transpose(1, 2)  # (B, H, S, d_k)
        K = K.view(B, S, H, d_k).transpose(1, 2)
        V = V.view(B, S, H, d_k).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, S, d_k)

        out = out.transpose(1, 2).contiguous().view(B, S * D)
        return out

    def _exec_add_constant(self, x: torch.Tensor) -> torch.Tensor:
        """Element-wise addition with a stored constant."""
        const = torch.tensor(
            self.params["constant"], dtype=x.dtype, device=x.device
        )
        return (x + const).view(x.shape[0], -1)

    def _exec_concat_constant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend / append a constant vector along dim-0 of the token sequence.

        ``input_shape`` = ``(S, D)``, constant shape = ``(1, D)``.
        Result shape: ``(S+1, D)`` → flattened to ``(B, (S+1)*D)``.
        """
        S, D = self.input_shape
        B = x.shape[0]
        x_seq = x.view(B, S, D)
        const = torch.tensor(
            self.params["constant"], dtype=x.dtype, device=x.device
        ).view(1, 1, D).expand(B, -1, -1)
        dim = self.params.get("dim", 0)
        if dim == 0:
            y = torch.cat([const, x_seq], dim=1)
        else:
            y = torch.cat([x_seq, const], dim=1)
        return y.view(B, -1)

    def _exec_select(self, x: torch.Tensor) -> torch.Tensor:
        """Select a single index along the first non-batch dimension."""
        index = self.params.get("index", 0)
        if self.input_shape is not None and len(self.input_shape) >= 2:
            B = x.shape[0]
            x = x.view(B, *self.input_shape)
            y = x[:, index]  # (B, D)
            return y.view(B, -1)
        return x

    def _exec_add(self, x: torch.Tensor) -> torch.Tensor:
        """
        Element-wise addition of two source tensors.

        Input is the concatenation of A and B, so x has shape (B, 2*N).
        When scale_a / scale_b are present (set by compute_per_source_scales
        when the two branches have different activation scales), the output is
        scale_a * a + scale_b * b = (A_full + B_full) / s_combined.
        """
        half = self.params["half_size"]
        a = x[:, :half]
        b = x[:, half:]
        scale_a = self.params.get("scale_a", None)
        scale_b = self.params.get("scale_b", None)
        if scale_a is not None and scale_b is not None:
            sa = _broadcast_scale_to_dim(
                torch.tensor(scale_a, dtype=x.dtype, device=x.device), half
            )
            sb = _broadcast_scale_to_dim(
                torch.tensor(scale_b, dtype=x.dtype, device=x.device), half
            )
            return sa.unsqueeze(0) * a + sb.unsqueeze(0) * b
        return a + b

    def _exec_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-reduce along a dimension.

        Input has shape (B, num_groups * group_size).  We reshape to
        (B, num_groups, group_size) and take the mean along dim=1,
        producing (B, group_size).
        """
        group_size = self.params["group_size"]
        num_groups = self.params["num_groups"]
        x = x[:, :num_groups * group_size].view(x.shape[0], num_groups, group_size)
        return x.mean(dim=1)

    def _exec_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Identity at inference (dropout is training-only)."""
        return x.view(x.shape[0], -1)

    def _exec_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Host-side linear (matmul + bias). No activation — preserves negatives."""
        weight = torch.tensor(self.params["weight"], dtype=x.dtype, device=x.device)
        out = torch.matmul(x, weight.T)
        if "bias" in self.params and self.params["bias"] is not None:
            bias = torch.tensor(self.params["bias"], dtype=x.dtype, device=x.device)
            out = out + bias
        return out.reshape(out.shape[0], -1)

    def _exec_module(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a generic PyTorch module stored in params.

        Reshapes flat (B, N) input to the module's expected shape,
        runs the module, and flattens the output back.  Ensures the
        module lives on the same device as the input (handles
        deserialized-from-pickle modules that land on CPU).
        """
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
        if output_index is not None:
            out = out[output_index]
        return out.reshape(out.shape[0], -1)


# ---------------------------------------------------------------------------
# IRGraph: The complete IR representation
# ---------------------------------------------------------------------------
@dataclass
class IRGraph:
    """
    Unified IR graph containing both neural cores and compute operations.

    The graph is topologically sorted for execution order.

    ``weight_banks`` stores shared weight matrices referenced by
    bank-backed ``NeuralCore`` nodes (see ``WeightBank``).
    """
    nodes: List[IRNode]
    output_sources: np.ndarray  # Array of IRSource for final outputs
    weight_banks: Dict[int, WeightBank] = field(default_factory=dict)

    def __getattr__(self, name: str):
        # Backward compat: old pickles lack weight_banks
        if name == "weight_banks":
            self.weight_banks = {}
            return self.weight_banks
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def get_neural_cores(self) -> List[NeuralCore]:
        """Return all neural core nodes."""
        return [n for n in self.nodes if isinstance(n, NeuralCore)]

    def get_compute_ops(self) -> List[ComputeOp]:
        """Return all compute operation nodes."""
        return [n for n in self.nodes if isinstance(n, ComputeOp)]

    def get_node_by_id(self, node_id: int) -> IRNode | None:
        """Look up a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_weight_bank(self, bank_id: int) -> WeightBank | None:
        """Look up a weight bank by its ID."""
        return self.weight_banks.get(bank_id)

    def resolve_core_matrix(self, core: NeuralCore) -> np.ndarray:
        """Convenience: resolve the effective core matrix for any NeuralCore."""
        return core.get_core_matrix(self)

    def validate(self) -> List[str]:
        """
        Validate the IR graph for consistency.

        Returns a list of error messages (empty if valid).
        """
        errors = []
        node_ids = {n.id for n in self.nodes}

        for node in self.nodes:
            for src in node.input_sources.flatten():
                if not isinstance(src, IRSource):
                    continue
                if src.node_id >= 0 and src.node_id not in node_ids:
                    errors.append(
                        f"Node {node.id} ({node.name}) references non-existent node {src.node_id}"
                    )

        for src in self.output_sources.flatten():
            if not isinstance(src, IRSource):
                continue
            if src.node_id >= 0 and src.node_id not in node_ids:
                errors.append(f"Output references non-existent node {src.node_id}")

        # Validate weight bank references
        for node in self.nodes:
            if isinstance(node, NeuralCore) and node.weight_bank_id is not None:
                if node.weight_bank_id not in self.weight_banks:
                    errors.append(
                        f"NeuralCore {node.id} ({node.name}) references "
                        f"weight_bank_id={node.weight_bank_id} which does not exist"
                    )

        return errors


# ---------------------------------------------------------------------------
# Conversion utilities: SoftCore/SpikeSource <-> IRNode/IRSource
# ---------------------------------------------------------------------------
def spike_source_to_ir_source(spike_source, core_id_offset: int = 0) -> IRSource:
    """
    Convert a SpikeSource (from mapping_utils) to an IRSource.

    Args:
        spike_source: SpikeSource object from the old mapping system.
        core_id_offset: Offset to add to core IDs (for combining multiple mappings).
    """
    if spike_source.is_off_:
        return IRSource(node_id=-1, index=0)
    elif spike_source.is_input_:
        return IRSource(node_id=-2, index=spike_source.neuron_)
    elif spike_source.is_always_on_:
        return IRSource(node_id=-3, index=0)
    else:
        return IRSource(node_id=spike_source.core_ + core_id_offset, index=spike_source.neuron_)


def soft_core_to_neural_core(soft_core, core_id_offset: int = 0) -> NeuralCore:
    """
    Convert a SoftCore (from softcore_mapping) to a NeuralCore.

    Args:
        soft_core: SoftCore object from the old mapping system.
        core_id_offset: Offset to add to source core IDs.
    """
    # Convert axon sources to IRSource objects
    ir_sources = np.array([
        spike_source_to_ir_source(s, core_id_offset)
        for s in soft_core.axon_sources
    ])

    return NeuralCore(
        id=soft_core.id + core_id_offset,
        name=soft_core.name or f"core_{soft_core.id}",
        input_sources=ir_sources,
        core_matrix=soft_core.core_matrix,
        threshold=soft_core.threshold,
        activation_scale=soft_core.activation_scale,
        parameter_scale=soft_core.parameter_scale,
        input_activation_scale=soft_core.input_activation_scale,
        latency=soft_core.latency,
        psum_group_id=soft_core.psum_group_id,
        psum_role=soft_core.psum_role,
        coalescing_group_id=soft_core.coalescing_group_id,
        coalescing_role=soft_core.coalescing_role,
    )


def soft_core_mapping_to_ir_graph(soft_core_mapping) -> IRGraph:
    """
    Convert a SoftCoreMapping to an IRGraph.

    This is a bridge for migrating existing code to the new IR.
    """
    nodes = []
    for soft_core in soft_core_mapping.cores:
        nodes.append(soft_core_to_neural_core(soft_core))

    output_sources = np.array([
        spike_source_to_ir_source(s) for s in soft_core_mapping.output_sources
    ])

    return IRGraph(nodes=nodes, output_sources=output_sources)


def ir_source_to_spike_source(ir_source: IRSource):
    """Convert an IRSource to a SpikeSource."""
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    
    if ir_source.is_off():
        return SpikeSource(-1, 0, is_input=False, is_off=True)
    elif ir_source.is_input():
        return SpikeSource(-2, ir_source.index, is_input=True, is_off=False)
    elif ir_source.is_always_on():
        return SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True)
    else:
        return SpikeSource(ir_source.node_id, ir_source.index, is_input=False, is_off=False)


def neural_core_to_soft_core(neural_core: NeuralCore, graph: IRGraph | None = None):
    """Convert a NeuralCore to a SoftCore.

    For bank-backed cores ``graph`` must be provided so the weight matrix
    can be materialized from the referenced ``WeightBank``.

    When the node has pruning masks (pruned_row_mask, pruned_col_mask) and the
    matrix shape matches their lengths, they are attached to the SoftCore so
    compaction uses the pruning maps instead of parameter values.
    """
    from mimarsinan.mapping.softcore_mapping import SoftCore

    axon_sources = [
        ir_source_to_spike_source(src) for src in neural_core.input_sources.flatten()
    ]

    core_matrix = neural_core.get_core_matrix(graph)

    pruned_row_mask = getattr(neural_core, "pruned_row_mask", None)
    pruned_col_mask = getattr(neural_core, "pruned_col_mask", None)
    # Only attach masks when they match current matrix (full pre-compaction layout)
    if pruned_row_mask is not None and pruned_col_mask is not None:
        if len(pruned_row_mask) != core_matrix.shape[0] or len(pruned_col_mask) != core_matrix.shape[1]:
            raise ValueError(
                f"neural_core_to_soft_core: pruning mask length mismatch for node_id={neural_core.id}: "
                f"core_matrix.shape={core_matrix.shape}, pruned_row_mask len={len(pruned_row_mask)}, "
                f"pruned_col_mask len={len(pruned_col_mask)}. Masks must match matrix shape (fix in ir_pruning)."
            )

    pi = getattr(neural_core, "perceptron_index", None)

    # Bank provenance.  When the IR node is bank-backed we preserve the
    # bank_id + slice so the HCM GPU sim can share a single resident
    # tensor per bank (instead of uploading one copy per position).
    # ``core_matrix`` is still materialised above for CPU consumers
    # (packer blit, compaction, codegen) — the two paths stay
    # consistent at construction time.
    bank_axon_slice = None
    bank_neuron_slice = None
    bank_includes_bias_row = False
    if neural_core.has_weight_bank() and graph is not None:
        bank = graph.get_weight_bank(neural_core.weight_bank_id)
        bank_in, bank_out = bank.core_matrix.shape
        # ``weight_row_slice`` in the IR slices the bank's COLUMN (output)
        # axis despite its name — matches ``(start, end)`` into
        # ``bank.core_matrix[:, start:end]``.
        wrs = neural_core.weight_row_slice
        bank_neuron_slice = (
            (int(wrs[0]), int(wrs[1])) if wrs is not None else (0, bank_out)
        )
        # The last axon source is always-on when this core is bias-
        # augmented in the legacy non-``hardware_bias`` path (the bank
        # matrix then has an extra final row holding the bias).
        last_is_always_on = (
            len(axon_sources) > 0
            and getattr(axon_sources[-1], "is_always_on_", False)
        )
        bank_includes_bias_row = bool(
            last_is_always_on and bank.hardware_bias is None and bank_in > 0
        )
        bank_axon_slice = (0, bank_in)

    soft = SoftCore(
        core_matrix=core_matrix,
        axon_sources=axon_sources,
        id=neural_core.id,
        activation_scale=neural_core.activation_scale,
        parameter_scale=neural_core.parameter_scale,
        input_activation_scale=neural_core.input_activation_scale,
        name=neural_core.name,
        psum_group_id=neural_core.psum_group_id,
        psum_role=neural_core.psum_role,
        coalescing_group_id=neural_core.coalescing_group_id,
        coalescing_role=neural_core.coalescing_role,
        threshold_group_id=int(pi) if pi is not None else None,
        weight_bank_id=(
            int(neural_core.weight_bank_id)
            if neural_core.has_weight_bank() else None
        ),
        bank_axon_slice=bank_axon_slice,
        bank_neuron_slice=bank_neuron_slice,
        bank_includes_bias_row=bank_includes_bias_row,
    )
    if pruned_row_mask is not None and pruned_col_mask is not None:
        soft.pruned_row_mask = pruned_row_mask
        soft.pruned_col_mask = pruned_col_mask
    # Pass hardware_bias through (no always-on row needed).
    if neural_core.hardware_bias is not None:
        n_neurons = core_matrix.shape[1]
        if len(neural_core.hardware_bias) != n_neurons:
            raise ValueError(
                f"neural_core_to_soft_core: hardware_bias length ({len(neural_core.hardware_bias)}) "
                f"does not match core_matrix neuron count ({n_neurons}) for node_id={neural_core.id}. "
                f"hardware_bias must be pruned alongside core_matrix columns in ir_pruning."
            )
        soft.hardware_bias = neural_core.hardware_bias.copy()
    return soft


def ir_graph_to_soft_core_mapping(ir_graph: IRGraph):
    """
    Convert an IRGraph to a SoftCoreMapping.
    
    NOTE: This only works for neural-only graphs (no ComputeOp nodes).
    Raises ValueError if the graph contains ComputeOps.

    Bank-backed NeuralCores are materialized (the shared weight matrix is
    copied into each SoftCore) so downstream packing and codegen see one
    concrete matrix per core.
    """
    from mimarsinan.mapping.soft_core_mapper import SoftCoreMapping
    
    compute_ops = ir_graph.get_compute_ops()
    if compute_ops:
        raise ValueError(
            f"Cannot convert IRGraph to SoftCoreMapping: graph contains {len(compute_ops)} "
            f"ComputeOp nodes. Use SpikingUnifiedCoreFlow / SpikingHybridCoreFlow for simulation instead."
        )
    
    soft_core_mapping = SoftCoreMapping()

    # Expose raw bank matrices so the downstream HardCoreMapping / HCM
    # simulator can share one GPU tensor per bank across all placements.
    soft_core_mapping.weight_banks = {
        bid: bank.core_matrix
        for bid, bank in (ir_graph.weight_banks or {}).items()
    }

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            soft_core = neural_core_to_soft_core(node, graph=ir_graph)
            soft_core.threshold = node.threshold
            soft_core.latency = node.latency
            soft_core_mapping.cores.append(soft_core)

    soft_core_mapping.output_sources = [
        ir_source_to_spike_source(src) for src in ir_graph.output_sources.flatten()
    ]

    return soft_core_mapping


