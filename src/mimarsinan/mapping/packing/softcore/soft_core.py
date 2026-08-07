"""SoftCore container for a mapped neural IR node."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.packing.softcore.matrix_placement import (
    materialize_compacted,
)


class SoftCore:
    def __init__(
        self,
        core_matrix,
        axon_sources,
        id,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
        *,
        name: str | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
        residency_class_id: int | None = None,
        weight_bank_id: int | None = None,
        bank_axon_slice: tuple[int, int] | None = None,
        bank_neuron_slice: tuple[int, int] | None = None,
        bank_includes_bias_row: bool = False,
        boundary_grid=None,
    ):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources
        self.boundary_grid = boundary_grid

        self.id = id
        self.input_activation_scale = input_activation_scale
        self.activation_scale = activation_scale
        self.parameter_scale = parameter_scale
        self.threshold = 1.0

        # Pack-time: share hardcore only when residency_class_id matches; None → unique group.
        self.residency_class_id = residency_class_id

        self.name = name
        self.psum_group_id = psum_group_id
        self.psum_role = psum_role
        self.coalescing_group_id = coalescing_group_id
        self.coalescing_role = coalescing_role

        self.weight_bank_id = weight_bank_id
        self.bank_axon_slice = bank_axon_slice
        self.bank_neuron_slice = bank_neuron_slice
        self.bank_includes_bias_row = bool(bank_includes_bias_row)

        # Compaction descriptor: when set, the dense matrix is
        # ``float64(compact_source_matrix)[np.ix_(rows, cols)]`` resolved on
        # read — the source stays the SHARED bank payload, never a copy.
        self.compact_source_matrix = None
        self.compact_keep_rows: "np.ndarray | None" = None
        self.compact_keep_cols: "np.ndarray | None" = None

        self.hardware_bias: np.ndarray | None = None

        self.latency: int | None = None
        self._axon_source_spans = None

        self.neuron_offset_in_original = 0
        self.split_group_id = None
        self.split_fragment_index = None
        self.split_original_neurons = None

        self.perceptron_index: int | None = None
        self.perceptron_output_slice: tuple[int, int] | None = None
        self.pruned_row_mask: list | None = None
        self.pruned_col_mask: list | None = None

    def get_input_count(self):
        return len(self.axon_sources)

    def get_output_count(self):
        if self.core_matrix is not None:
            return self.core_matrix.shape[-1]
        if self.compact_keep_cols is not None:
            return len(self.compact_keep_cols)
        raise ValueError(
            f"SoftCore id={self.id}: no core_matrix and no compaction "
            f"descriptor to size the output count from."
        )

    def has_core_matrix(self) -> bool:
        return self.core_matrix is not None or self.compact_keep_rows is not None

    def get_core_matrix(self):
        """The dense weight matrix; compacted cores materialize TRANSIENTLY
        (byte-identical to the eager ``np.ix_`` recipe), owned/bank-shared
        matrices return the stored object unchanged."""
        if self.core_matrix is not None:
            return self.core_matrix
        if self.compact_keep_rows is not None:
            return materialize_compacted(
                self.compact_source_matrix,
                self.compact_keep_rows,
                self.compact_keep_cols,
            )
        raise ValueError(
            f"SoftCore id={self.id}: no core_matrix and no compaction "
            f"descriptor to resolve one from."
        )

    def get_axon_source_spans(self):
        """Cached range-compressed axon_sources; invalidate if axon_sources mutate."""
        if self._axon_source_spans is None:
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans
