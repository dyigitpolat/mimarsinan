from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.layout.layout_source_view_ops import (
    node_ids_of,
    total_size,
)
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.layout.ir_mapping import (
    _LayoutIRMappingBanks,
    _LayoutIRMappingFC,
    _LayoutIRMappingFinalize,
)
from mimarsinan.mapping.noc.wire_census import record_emission_census
from mimarsinan.mapping.platform.mapping_structure import compute_core_input_count


@dataclass
class LayoutIRMapping(_LayoutIRMappingFinalize, _LayoutIRMappingFC, _LayoutIRMappingBanks):
    """Shape-only mapping backend: the single source of truth for softcore
    emission decisions (tiling mode, psum decomposition, coalescing, bias-axon
    counting, shared-bank wiring). ``IRMapping`` subclasses it to attach weights."""

    max_axons: Optional[int]
    max_neurons: Optional[int]
    allow_coalescing: bool = False
    hardware_bias: bool = False
    bias_row_splitting: bool = False
    onchip_residual_merge: bool = False
    # Wire-census collection MATERIALISES every deferred input view (the very
    # cost LayoutSourceView defers), so it is opt-in: only a walk whose caller
    # needs NoC traffic pays it.
    collect_wire_census: bool = False

    def __post_init__(self):
        self.max_axons = int(self.max_axons) if self.max_axons is not None else None
        self.max_neurons = int(self.max_neurons) if self.max_neurons is not None else None
        self.allow_coalescing = bool(self.allow_coalescing)
        self.hardware_bias = bool(self.hardware_bias)
        self.bias_row_splitting = bool(self.bias_row_splitting)
        self.onchip_residual_merge = bool(self.onchip_residual_merge)

        self._next_node_id = 0
        self._coalescing_group_counter = 0
        self._psum_group_counter = 0
        self._next_bank_id = 0

        self.layout_softcores: List[LayoutSoftCoreSpec] = []
        self.output_sources: np.ndarray = np.array([])
        self.host_side_segment_count: int = 0
        self.layout_preview: Dict[str, Any] | None = None

        self._node_input_node_ids: Dict[int, Set[int]] = {}
        self._node_wire_census: Dict[int, Any] = {}
        self._node_id_to_softcore_idx: Dict[int, int] = {}
        self._node_is_neural: Dict[int, bool] = {}
        self._node_host_labels: Dict[int, Dict[str, str]] = {}
        self._sc_idx_to_perceptron_index: Dict[int, Optional[int]] = {}

        self._layout_weight_banks: Dict[int, Tuple[int, int]] = {}
        self._layout_bank_bias_rows: Dict[int, int] = {}
        self._sc_idx_to_bank_id: Dict[int, int] = {}

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _extract_input_node_ids(self, input_sources) -> Set[int]:
        """Set of producer node ids feeding ``input_sources``."""
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
        if self.collect_wire_census:
            record_emission_census(
                self._node_wire_census, node_id, input_sources, input_count)
        self._node_is_neural[node_id] = True

        sc_idx = len(self.layout_softcores)
        self._node_id_to_softcore_idx[node_id] = sc_idx
        self._sc_idx_to_perceptron_index[sc_idx] = (
            int(perceptron_index) if perceptron_index is not None else None
        )

        self.layout_softcores.append(
            LayoutSoftCoreSpec(
                input_count=int(input_count),
                output_count=int(output_count),
                residency_class_id=0,
                latency_tag=None,
                name=name,
            )
        )
        return LayoutSourceView.from_producer(
            producer_node_id=node_id,
            shape=(int(output_count),),
        )

    def map(self, model_representation) -> np.ndarray:
        """Run ``map_to_ir`` over the mapper graph and finalise metadata.

        ``onchip_residual_merge`` on lowers each param-free residual add to an
        on-chip signed-IF merge core (Tier-1); off keeps the byte-identical host add.
        """
        if self.onchip_residual_merge:
            from mimarsinan.mapping.support.residual_merge import (
                lower_residual_adds_to_onchip_merge,
            )

            lower_residual_adds_to_onchip_merge(model_representation)
        output_sources = model_representation.map_to_ir(self)
        self.output_sources = output_sources
        self._finalize_softcores()
        return output_sources

    def collect_layout_softcores(self, model_representation) -> List[LayoutSoftCoreSpec]:
        """Back-compat entry point for shape-only callers (wizard / search).

        Side-effect-free on the model: the walk-scoped IR memos are cleared so
        estimate/telemetry callers (GUI step snapshot) cannot pin un-picklable
        LayoutSourceView closures on a live pipeline model.
        """
        try:
            self.map(model_representation)
        finally:
            clear = getattr(model_representation, "clear_ir_caches", None)
            if callable(clear):
                clear()
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
        self._node_host_labels[node_id] = {
            "name": str(name) if name else str(op_type or "host"),
            "op_type": str(op_type or ""),
        }

        return LayoutSourceView.from_producer(
            producer_node_id=node_id,
            shape=view_shape,
        )

    def add_neural_core(
        self,
        *,
        input_sources,
        weights: Any,
        biases: Any = None,
        activation_scale: Any = None,
        parameter_scale: Any = None,
        input_activation_scale: Any = None,
        bias_scale: Any = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        perceptron_input_slice: Optional[Tuple[int, int]] = None,
        perceptron_output_slice: Optional[Tuple[int, int]] = None,
        perceptron_output_column: Optional[int] = None,
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
            in_features, has_bias, self.hardware_bias,
            self.param_encoded_bias_rows(bias_scale, parameter_scale, name=name),
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
