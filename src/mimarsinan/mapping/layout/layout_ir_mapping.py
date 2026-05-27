from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.layout.layout_source_view_ops import (
    concat_source_views,
    node_ids_of,
    stack_source_views,
    total_size,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.layout.layout_ir_mapping_fc import _LayoutIRMappingFC
from mimarsinan.mapping.layout.layout_ir_mapping_finalize import _LayoutIRMappingFinalize
from mimarsinan.mapping.platform.mapping_structure import compute_core_input_count


@dataclass
class LayoutIRMapping(_LayoutIRMappingFinalize, _LayoutIRMappingFC):
    """Shape-only mapping backend — the single source of truth for all
    softcore emission decisions (tiling mode, psum decomposition, coalescing,
    bias-axon counting, shared-bank wiring).

    Both the wizard/architecture-search flow and the real ``IRMapping`` build
    on top of this class.  Shape-only callers use it directly; ``IRMapping``
    subclasses it and overrides the emission hooks (``add_neural_core``,
    ``add_shared_neural_core``, ``register_weight_bank``, ``add_compute_op``)
    to additionally attach weight material and build an ``IRGraph``.

    Threshold groups are assigned from ``perceptron_index``: all softcores
    produced by a single perceptron (output-tiling, psum fragments, conv
    positions) share one integer group id, matching the real packer's
    behavior for shared-weight cores (they end up with identical quantization
    scales and thus identical float thresholds within 10% tolerance).

    Softcores without a ``perceptron_index`` (e.g. synthesised accumulator
    cores) fall back to a unique per-node id so they are never misjudged as
    "shareable" with unrelated cores.
    """

    max_axons: Optional[int]
    max_neurons: Optional[int]
    allow_coalescing: bool = False
    hardware_bias: bool = False

    def __post_init__(self):
        self.max_axons = int(self.max_axons) if self.max_axons is not None else None
        self.max_neurons = int(self.max_neurons) if self.max_neurons is not None else None
        self.allow_coalescing = bool(self.allow_coalescing)
        self.hardware_bias = bool(self.hardware_bias)

        self._next_node_id = 0
        self._coalescing_group_counter = 0
        self._psum_group_counter = 0
        self._next_bank_id = 0

        self.layout_softcores: List[LayoutSoftCoreSpec] = []
        self.output_sources: np.ndarray = np.array([])
        self.host_side_segment_count: int = 0
        self.layout_preview: Dict[str, Any] | None = None

        # Dependency tracking for latency / segment computation
        self._node_input_node_ids: Dict[int, Set[int]] = {}
        self._node_id_to_softcore_idx: Dict[int, int] = {}
        self._node_is_neural: Dict[int, bool] = {}
        self._sc_idx_to_perceptron_index: Dict[int, Optional[int]] = {}

        # Shared weight-bank shape registry
        # bank_id -> (in_features_with_bias, out_features)
        self._layout_weight_banks: Dict[int, Tuple[int, int]] = {}
        self._sc_idx_to_bank_id: Dict[int, int] = {}

    # Node id allocation

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    # Dependency / shape helpers

    def _extract_input_node_ids(self, input_sources) -> Set[int]:
        """Set of producer node ids feeding ``input_sources``.

        Fast path: a ``LayoutSourceView`` carries the producer set as metadata
        and we avoid scanning per-cell.
        """
        return node_ids_of(input_sources)

    def _emit_softcore_record(
        self,
        *,
        node_id: int,
        input_count: int,
        output_count: int,
        name: Optional[str],
        input_sources,
        perceptron_index: Optional[int],
    ) -> LayoutSourceView:
        """Record a softcore shape under ``node_id`` and return a 1-D
        ``LayoutSourceView`` over its outputs."""
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)
        self._node_is_neural[node_id] = True

        sc_idx = len(self.layout_softcores)
        self._node_id_to_softcore_idx[node_id] = sc_idx
        self._sc_idx_to_perceptron_index[sc_idx] = (
            int(perceptron_index) if perceptron_index is not None else None
        )

        # threshold_group_id / latency_tag / segment_id are finalised later
        self.layout_softcores.append(
            LayoutSoftCoreSpec(
                input_count=int(input_count),
                output_count=int(output_count),
                threshold_group_id=0,
                latency_tag=None,
                name=name,
            )
        )
        return LayoutSourceView.from_producer(
            producer_node_id=node_id,
            shape=(int(output_count),),
        )

    # Public mapping protocol (called by Mapper.map_to_ir)

    def map(self, model_representation) -> np.ndarray:
        """Run ``map_to_ir`` over the mapper graph and finalise metadata.

        Returns the graph's output sources.  ``IRMapping`` subclasses this to
        wrap the result in an ``IRGraph``.
        """
        output_sources = model_representation.map_to_ir(self)
        self.output_sources = output_sources
        self._finalize_softcores()
        return output_sources

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """Back-compat entry point for shape-only callers (wizard / search)."""
        self.map(model_representation)
        return list(self.layout_softcores)

    def add_compute_op(
        self,
        input_sources,
        op_type: str,
        params=None,
        input_shape=None,
        output_shape=None,
        name: Optional[str] = None,
    ) -> LayoutSourceView:
        """Record a compute-op's dependency structure and return a
        ``LayoutSourceView`` over its outputs.  Shape-only: no ComputeOp
        node is built."""
        if output_shape is not None:
            output_size = 1
            for d in output_shape:
                output_size *= d
            view_shape: Tuple[int, ...] = tuple(int(d) for d in output_shape)
        else:
            output_size = total_size(input_sources)
            view_shape = (output_size,)

        node_id = self._alloc_node_id()
        self._node_input_node_ids[node_id] = self._extract_input_node_ids(input_sources)
        self._node_is_neural[node_id] = False

        return LayoutSourceView.from_producer(
            producer_node_id=node_id,
            shape=view_shape,
        )

    def register_weight_bank(
        self,
        weights: Any,
        biases: Any = None,
        activation_scale=None,
        parameter_scale=None,
        input_activation_scale=None,
        perceptron_index: Optional[int] = None,
    ) -> int:
        """Register a shared weight bank (shape only) and return its ID."""
        bank_id = self._next_bank_id
        self._next_bank_id += 1

        w_shape = getattr(weights, "shape", None)
        if w_shape is not None:
            out_features = int(w_shape[0])
            in_features = int(w_shape[1]) if len(w_shape) > 1 else 1
        else:
            out_features = 1
            in_features = 1

        has_bias = biases is not None
        in_features_with_bias = compute_core_input_count(
            in_features, has_bias, self.hardware_bias
        )
        self._layout_weight_banks[bank_id] = (in_features_with_bias, out_features)
        return bank_id

    def add_neural_core(
        self,
        *,
        input_sources,
        weights: Any,
        biases: Any = None,
        activation_scale: Any = None,
        parameter_scale: Any = None,
        input_activation_scale: Any = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        perceptron_input_slice: Optional[Tuple[int, int]] = None,
        perceptron_output_slice: Optional[Tuple[int, int]] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> LayoutSourceView:
        """Emit a single neural softcore (owned weights).

        Base implementation records shape only.  ``IRMapping`` overrides to
        additionally construct a concrete ``NeuralCore`` node.
        """
        in_features = total_size(input_sources)
        out_features = int(getattr(weights, "shape", [0])[0])
        has_bias = biases is not None
        in_count = compute_core_input_count(
            in_features, has_bias, self.hardware_bias
        )

        node_id = self._alloc_node_id()
        return self._emit_softcore_record(
            node_id=node_id,
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=input_sources,
            perceptron_index=perceptron_index,
        )

    def add_shared_neural_core(
        self,
        *,
        input_sources,
        weight_bank_id: int,
        has_bias: bool = True,
        weight_row_slice: Optional[Tuple[int, int]] = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> LayoutSourceView:
        """Emit a bank-backed neural softcore (one conv position).

        Base implementation records shape only.  ``IRMapping`` overrides to
        also build the concrete ``NeuralCore`` referencing the bank.
        """
        bank_shape = self._layout_weight_banks.get(weight_bank_id)
        if bank_shape is None:
            raise ValueError(f"Unknown weight_bank_id={weight_bank_id}")
        _in_features_with_bias, bank_out_features = bank_shape

        in_count = compute_core_input_count(
            total_size(input_sources), has_bias, self.hardware_bias
        )

        if weight_row_slice is not None:
            out_features = weight_row_slice[1] - weight_row_slice[0]
        else:
            out_features = bank_out_features

        node_id = self._alloc_node_id()
        sc_idx = len(self.layout_softcores)
        result = self._emit_softcore_record(
            node_id=node_id,
            input_count=in_count,
            output_count=out_features,
            name=name,
            input_sources=input_sources,
            perceptron_index=perceptron_index,
        )
        self._sc_idx_to_bank_id[sc_idx] = weight_bank_id
        return result

    # Shared FC tiling dispatch
