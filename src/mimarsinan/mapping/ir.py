"""
Unified Intermediate Representation (IR) for mimarsinan.

This IR supports both:
1. NeuralCore: Crossbar-based neural computation (matrix multiply + threshold)
2. ComputeOp: Non-neural operations (pooling, element-wise ops, etc.)

The IR enables simulation and hardware mapping of networks that mix
perceptron-based layers with non-neural operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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
    """
    core_matrix: np.ndarray  # shape: (axons, neurons)
    threshold: float = 1.0
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    latency: int | None = None

    # Metadata for debugging/visualization
    psum_group_id: int | None = None
    psum_role: str | None = None  # "partial_pos", "partial_neg", "accum"

    def get_input_count(self) -> int:
        return self.core_matrix.shape[0]

    def get_output_count(self) -> int:
        return self.core_matrix.shape[1]

    def execute(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Execute crossbar computation: W @ x + activation."""
        inputs = self.gather_inputs(input_tensor, buffers)

        # Core matrix stored as (axons, neurons), we need (neurons, axons) for matmul
        weight = torch.tensor(
            self.core_matrix.T,
            dtype=torch.float32,
            device=input_tensor.device,
        )
        # inputs: (batch, axons), weight: (neurons, axons)
        # output: (batch, neurons)
        out = torch.matmul(inputs, weight.T)

        # Apply ReLU-style activation (can be customized via activation_scale)
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
        elif self.op_type == "dropout":
            return self._exec_dropout(x)
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
        """
        half = self.params["half_size"]
        a = x[:, :half]
        b = x[:, half:]
        return a + b

    def _exec_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Identity at inference (dropout is training-only)."""
        return x.view(x.shape[0], -1)


# ---------------------------------------------------------------------------
# IRGraph: The complete IR representation
# ---------------------------------------------------------------------------
@dataclass
class IRGraph:
    """
    Unified IR graph containing both neural cores and compute operations.

    The graph is topologically sorted for execution order.
    """
    nodes: List[IRNode]
    output_sources: np.ndarray  # Array of IRSource for final outputs

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


def neural_core_to_soft_core(neural_core: NeuralCore):
    """Convert a NeuralCore to a SoftCore."""
    from mimarsinan.mapping.softcore_mapping import SoftCore
    
    axon_sources = [
        ir_source_to_spike_source(src) for src in neural_core.input_sources.flatten()
    ]
    
    return SoftCore(
        core_matrix=neural_core.core_matrix,
        axon_sources=axon_sources,
        id=neural_core.id,
        activation_scale=neural_core.activation_scale,
        parameter_scale=neural_core.parameter_scale,
        input_activation_scale=neural_core.input_activation_scale,
        name=neural_core.name,
        psum_group_id=neural_core.psum_group_id,
        psum_role=neural_core.psum_role,
    )


def ir_graph_to_soft_core_mapping(ir_graph: IRGraph):
    """
    Convert an IRGraph to a SoftCoreMapping.
    
    NOTE: This only works for neural-only graphs (no ComputeOp nodes).
    Raises ValueError if the graph contains ComputeOps.
    """
    from mimarsinan.mapping.mapping_utils import SoftCoreMapping
    
    compute_ops = ir_graph.get_compute_ops()
    if compute_ops:
        raise ValueError(
            f"Cannot convert IRGraph to SoftCoreMapping: graph contains {len(compute_ops)} "
            f"ComputeOp nodes. Use SpikingUnifiedCoreFlow / SpikingHybridCoreFlow for simulation instead."
        )
    
    # Create an empty SoftCoreMapping and populate it
    soft_core_mapping = SoftCoreMapping()
    
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            soft_core = neural_core_to_soft_core(node)
            soft_core.threshold = node.threshold
            soft_core.latency = node.latency
            soft_core_mapping.cores.append(soft_core)
    
    # Convert output sources
    soft_core_mapping.output_sources = [
        ir_source_to_spike_source(src) for src in ir_graph.output_sources.flatten()
    ]
    
    return soft_core_mapping


