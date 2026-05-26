"""Unified IR: NeuralCore crossbars, ComputeOps, and shared WeightBanks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class WeightBank:
    """Shared weight matrix (and optional bias) referenced by multiple NeuralCores."""
    id: int
    core_matrix: np.ndarray  # (axons, neurons) — weights only, no bias row
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    perceptron_index: int | None = None
    hardware_bias: np.ndarray | None = None


@dataclass
class IRSource:
    """Input source: node output, off (-1), network input (-2), or always-on (-3)."""
    node_id: int
    index: int  # Which output index from that node

    def is_off(self) -> bool:
        return self.node_id == -1

    def is_input(self) -> bool:
        return self.node_id == -2

    def is_always_on(self) -> bool:
        return self.node_id == -3


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
        """Execute this node during simulation."""
        pass

    def gather_inputs(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Gather inputs from sources into a 1D tensor (override for special cases)."""
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


@dataclass
class NeuralCore(IRNode):
    """Crossbar neural core: owned core_matrix or shared WeightBank reference."""
    core_matrix: np.ndarray | None = None  # (axons, neurons); None when using a bank
    threshold: float = 1.0
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    latency: int | None = None

    weight_bank_id: int | None = None
    weight_row_slice: tuple[int, int] | None = None

    perceptron_index: int | None = None
    perceptron_output_slice: tuple[int, int] | None = None
    perceptron_input_slice: tuple[int, int] | None = None

    psum_group_id: int | None = None
    psum_role: str | None = None  # "partial_pos", "partial_neg", "accum"
    coalescing_group_id: int | None = None
    coalescing_role: str | None = None  # "master", "slave"
    normalization_type: str | None = None
    activation_type: str | None = None

    hardware_bias: np.ndarray | None = None

    # Index into ``IRGraph.layout_softcores`` when produced by ``IRMapping``.
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


@dataclass
class ComputeOp(IRNode):
    """Non-neural compute op (pooling, norm, attention, etc.)."""
    op_type: str  # "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "flatten", etc.
    params: Dict[str, Any] = field(default_factory=dict)

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
        """Execute on pre-gathered flat input ``(B, N)``."""
        return self._dispatch(flat_input)

    def _dispatch(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the wrapped host-side ``nn.Module``.

        All non-neural ComputeOps share one dispatch path: ``params["module"]``
        is the ``nn.Module`` to run, gathered ``(B, N)`` input is reshaped
        through ``input_shape`` / ``input_shapes`` first.  ``op_type`` is a
        free-form display label (typically ``type(module).__name__``) — it
        has no role in dispatch.
        """
        return self._exec_module(x)

    def _gather_structured_input(
        self,
        input_tensor: torch.Tensor,
        buffers: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Gather inputs into flat ``(B, N)``."""
        return self.gather_inputs(input_tensor, buffers)  # (B, N)

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


@dataclass
class IRGraph:
    """IR graph of NeuralCores and ComputeOps with shared WeightBanks."""
    nodes: List[IRNode]
    output_sources: np.ndarray  # Array of IRSource for final outputs
    weight_banks: Dict[int, WeightBank] = field(default_factory=dict)
    layout_softcores: List[Any] = field(default_factory=list)

    def __getattr__(self, name: str):
        # Backward compat: old pickles lack weight_banks / layout_softcores
        if name == "weight_banks":
            self.weight_banks = {}
            return self.weight_banks
        if name == "layout_softcores":
            self.layout_softcores = []
            return self.layout_softcores
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

    def remove_nodes(self, node_ids: Iterable[int]) -> None:
        """Delete nodes from the graph; rewire dangling references to OFF.

        This is the canonical whole-node deletion API. It enforces the
        following invariants and raises :class:`ValueError` when any of
        them would be violated:

        - At least one entry of ``output_sources`` must remain pointing at
          a surviving NeuralCore (otherwise the graph has no live output).
        - Every member of a non-trivial ``psum_group_id`` group is removed
          atomically (all-or-none) -- the partial-positive and
          partial-negative halves of a partial-sum group cannot be split.
        - Every member of a non-trivial ``coalescing_group_id`` group is
          removed atomically (master and slaves come together).
        - Every requested ``node_id`` must currently exist in the graph.

        Side effects on success:

        - Each removed node disappears from ``self.nodes``.
        - Every input_source on every surviving node, plus every entry of
          ``output_sources``, that referenced a removed node is rewired to
          ``IRSource(node_id=-1, index=0)`` (off). The shape of
          ``output_sources`` is preserved (entries are rewritten, not
          dropped) so downstream sinks keep their slot indexing.
        - Weight banks that have no remaining NeuralCore consumers are
          removed from ``self.weight_banks``.

        Args:
            node_ids: Ids of NeuralCore / ComputeOp nodes to delete.
                Duplicates and an empty iterable are tolerated.
        """
        ids_to_remove = {int(nid) for nid in node_ids}
        if not ids_to_remove:
            return

        existing_ids = {n.id for n in self.nodes}
        unknown = ids_to_remove - existing_ids
        if unknown:
            raise ValueError(
                f"IRGraph.remove_nodes: unknown node ids: {sorted(unknown)} "
                f"(not in self.nodes)"
            )

        self._enforce_atomic_group_removal(
            ids_to_remove, attr="psum_group_id", label="psum",
        )
        self._enforce_atomic_group_removal(
            ids_to_remove, attr="coalescing_group_id", label="coalescing",
        )

        if self.output_sources is not None and self.output_sources.size:
            survives = [
                isinstance(s, IRSource)
                and (s.node_id < 0 or s.node_id not in ids_to_remove)
                for s in self.output_sources.flatten()
            ]
            if not any(survives):
                raise ValueError(
                    "IRGraph.remove_nodes: would empty output_sources "
                    "(every live output target is in the removal set)"
                )

        off_source = IRSource(node_id=-1, index=0)

        def _rewire(arr: np.ndarray) -> np.ndarray:
            flat = arr.flatten()
            for i, src in enumerate(flat):
                if (
                    isinstance(src, IRSource)
                    and src.node_id >= 0
                    and src.node_id in ids_to_remove
                ):
                    flat[i] = IRSource(node_id=-1, index=0)
            return flat.reshape(arr.shape)

        self.nodes = [n for n in self.nodes if n.id not in ids_to_remove]

        for node in self.nodes:
            if hasattr(node, "input_sources") and node.input_sources is not None:
                node.input_sources = _rewire(node.input_sources)

        if self.output_sources is not None and self.output_sources.size:
            self.output_sources = _rewire(self.output_sources)

        self._cleanup_orphan_weight_banks()
        _ = off_source  # kept for readability; flat path uses fresh instance

    def _enforce_atomic_group_removal(
        self,
        ids_to_remove: "set[int]",
        *,
        attr: str,
        label: str,
    ) -> None:
        """Reject removals that split a non-trivial group along ``attr``.

        A group is *non-trivial* when at least two distinct nodes share the
        same non-``None`` value for ``attr``.  When some -- but not all --
        of those members are scheduled for removal, raise ``ValueError``;
        the caller must include the entire group or none of it.
        """
        groups: Dict[int, List[int]] = {}
        for n in self.nodes:
            gid = getattr(n, attr, None)
            if gid is None:
                continue
            groups.setdefault(int(gid), []).append(n.id)
        for gid, members in groups.items():
            if len(members) < 2:
                continue
            in_set = [m for m in members if m in ids_to_remove]
            if 0 < len(in_set) < len(members):
                missing = sorted(set(members) - set(in_set))
                raise ValueError(
                    f"IRGraph.remove_nodes: cannot split {label} group "
                    f"id={gid}; members {sorted(members)} are atomic. "
                    f"Removing {sorted(in_set)} would leave {missing} "
                    "without their group partners."
                )

    def _cleanup_orphan_weight_banks(self) -> None:
        """Drop weight banks no longer referenced by any NeuralCore."""
        banks = getattr(self, "weight_banks", None)
        if not banks:
            return
        referenced = {
            n.weight_bank_id
            for n in self.nodes
            if isinstance(n, NeuralCore) and n.weight_bank_id is not None
        }
        for bank_id in list(banks.keys()):
            if bank_id not in referenced:
                banks.pop(bank_id, None)

    def validate(self) -> List[str]:
        """Return validation errors (empty if valid)."""
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

        for node in self.nodes:
            if isinstance(node, NeuralCore) and node.weight_bank_id is not None:
                if node.weight_bank_id not in self.weight_banks:
                    errors.append(
                        f"NeuralCore {node.id} ({node.name}) references "
                        f"weight_bank_id={node.weight_bank_id} which does not exist"
                    )

        return errors


def spike_source_to_ir_source(spike_source, core_id_offset: int = 0) -> IRSource:
    """Convert SpikeSource to IRSource."""
    if spike_source.is_off_:
        return IRSource(node_id=-1, index=0)
    elif spike_source.is_input_:
        return IRSource(node_id=-2, index=spike_source.neuron_)
    elif spike_source.is_always_on_:
        return IRSource(node_id=-3, index=0)
    else:
        return IRSource(node_id=spike_source.core_ + core_id_offset, index=spike_source.neuron_)


def soft_core_to_neural_core(soft_core, core_id_offset: int = 0) -> NeuralCore:
    """Convert SoftCore to NeuralCore."""
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
    """Convert SoftCoreMapping to IRGraph."""
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
    """Convert NeuralCore to SoftCore (graph required for bank-backed cores)."""
    from mimarsinan.mapping.softcore_mapping import SoftCore

    axon_sources = [
        ir_source_to_spike_source(src) for src in neural_core.input_sources.flatten()
    ]

    core_matrix = neural_core.get_core_matrix(graph)

    pruned_row_mask = getattr(neural_core, "pruned_row_mask", None)
    pruned_col_mask = getattr(neural_core, "pruned_col_mask", None)
    if pruned_row_mask is not None and pruned_col_mask is not None:
        if len(pruned_row_mask) != core_matrix.shape[0] or len(pruned_col_mask) != core_matrix.shape[1]:
            raise ValueError(
                f"neural_core_to_soft_core: pruning mask length mismatch for node_id={neural_core.id}: "
                f"core_matrix.shape={core_matrix.shape}, pruned_row_mask len={len(pruned_row_mask)}, "
                f"pruned_col_mask len={len(pruned_col_mask)}. Masks must match matrix shape (fix in ir_pruning)."
            )

    pi = getattr(neural_core, "perceptron_index", None)

    bank_axon_slice = None
    bank_neuron_slice = None
    bank_includes_bias_row = False
    if neural_core.has_weight_bank() and graph is not None:
        bank = graph.get_weight_bank(neural_core.weight_bank_id)
        bank_in, bank_out = bank.core_matrix.shape
        wrs = neural_core.weight_row_slice
        bank_neuron_slice = (
            (int(wrs[0]), int(wrs[1])) if wrs is not None else (0, bank_out)
        )
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
    """Convert neural-only IRGraph to SoftCoreMapping."""
    from mimarsinan.mapping.soft_core_mapper import SoftCoreMapping
    
    compute_ops = ir_graph.get_compute_ops()
    if compute_ops:
        raise ValueError(
            f"Cannot convert IRGraph to SoftCoreMapping: graph contains {len(compute_ops)} "
            f"ComputeOp nodes. Use SpikingUnifiedCoreFlow / SpikingHybridCoreFlow for simulation instead."
        )
    
    soft_core_mapping = SoftCoreMapping()

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


