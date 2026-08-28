"""Shape-only weight-bank registration and bank-backed softcore emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.layout.layout_source_view_ops import total_size
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
    compute_core_input_count,
)
from mimarsinan.mapping.support.bias_rows import param_encoded_bias_rows


class _LayoutIRMappingBanks:
    """Weight banks: registration, the bias-row view, and bank-backed cores.

    A bank's always-on row count is decided ONCE, at registration, and every
    core that references the bank reads it back — the shape-only walk and the
    weight-attaching mapper cannot disagree about the bank's geometry.
    """

    if TYPE_CHECKING:
        _next_bank_id: int
        _layout_weight_banks: Dict[int, Tuple[int, int]]
        _layout_bank_bias_rows: Dict[int, int]
        _sc_idx_to_bank_id: Dict[int, int]
        hardware_bias: bool
        max_axons: Optional[int]
        max_neurons: Optional[int]
        allow_coalescing: bool
        layout_softcores: list
        def _alloc_node_id(self) -> int: ...
        _emit_softcore_record: Callable[..., LayoutSourceView]


    def register_weight_bank(
        self,
        weights: Any,
        biases: Any = None,
        activation_scale=None,
        parameter_scale=None,
        input_activation_scale=None,
        perceptron_index: Optional[int] = None,
        bias_scale: Any = None,
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
        bias_rows = self.param_encoded_bias_rows(
            bias_scale, parameter_scale, name=f"weight bank {bank_id}"
        )
        in_features_with_bias = compute_core_input_count(
            in_features, has_bias, self.hardware_bias, bias_rows
        )
        self._layout_weight_banks[bank_id] = (in_features_with_bias, out_features)
        self._layout_bank_bias_rows[bank_id] = bias_rows
        return bank_id

    def param_encoded_bias_rows(
        self, bias_scale: Any, parameter_scale: Any, *, name: Any = None
    ) -> int:
        """Always-on rows one perceptron's parameter-encoded bias occupies.

        Read from the installed grids, never plumbed: the shape-only walk, the
        weight-attaching mapper and the packed chip all recover the same k.
        """
        return param_encoded_bias_rows(
            bias_scale, parameter_scale,
            hardware_bias=self.hardware_bias, name=name,
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
        perceptron_output_slice: Optional[Tuple[int, int]] = None,
        perceptron_output_column: Optional[int] = None,
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
            total_size(input_sources), has_bias, self.hardware_bias,
            self._layout_bank_bias_rows.get(weight_bank_id, 1),
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

    def _fc_bank_tiles(self, in_features, out_features, has_bias, bias_rows=1):
        """Bank-shareable output tiling for a multi-instance FC; ``None`` = the
        owned per-column path (wide-fan-in coalescing columns stay owned)."""
        strategy = MappingStrategy.resolve(ChipCapabilities(
            max_axons=self.max_axons, max_neurons=self.max_neurons,
            hardware_bias=self.hardware_bias,
            allow_coalescing=self.allow_coalescing))
        mode = strategy.tiling_mode(in_features, out_features, has_bias, bias_rows)
        if mode == "single":
            return [(0, out_features)]
        if mode == "output_tiled":
            assert self.max_neurons is not None
            chunk = int(self.max_neurons)
            return [(s, min(s + chunk, out_features))
                    for s in range(0, out_features, chunk)]
        return None
