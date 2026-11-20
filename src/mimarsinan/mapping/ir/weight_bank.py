"""Shared weight-bank storage: one matrix, many NeuralCore instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Mapping

import numpy as np
import torch

from mimarsinan.mapping.support.value_layouts import ValueLayout, layouts

__all__ = ["WeightBank"]


@dataclass
class WeightBank:
    """Shared weight matrix (and optional bias) referenced by multiple NeuralCores."""
    #: [U-PCH3] What each per-bank value's entries index (``mapping.support.value_layouts``), as its nodes' are.
    VALUE_LAYOUTS: ClassVar[Mapping[str, ValueLayout]] = layouts(
        PER_NEURON=("hardware_bias",), PER_CHANNEL=("activation_scale", "parameter_scale", "bias_scale"),
        PER_INPUT_ROW=("input_activation_scale",), WHOLE=("weight_bits",))
    id: int
    core_matrix: np.ndarray  # (axons, neurons) — weights only, no bias row
    activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    parameter_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    input_activation_scale: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    perceptron_index: int | None = None
    hardware_bias: np.ndarray | None = None
    # Two-scale WQ bias grid (parameter_scale / integer r); None == shared grid.
    bias_scale: torch.Tensor | None = None
    # ONE pre-compaction weights snapshot per bank (dtype-preserving); every
    # instance's GUI heatmap is a slice of this, never a per-core copy.
    pre_pruning_snapshot: "np.ndarray | None" = None
    # Per-range view memo (id-invalidated): every instance of a (bank, range)
    # shares ONE ndarray object so pickle memoization stores the payload once.
    _column_views: dict = field(default_factory=dict, repr=False, compare=False)
    _column_views_base: int | None = field(default=None, repr=False, compare=False)

    def column_slice(self, start: int, end: int) -> np.ndarray:
        """The (start, end) column view — full range returns the array itself."""
        if start == 0 and end == self.core_matrix.shape[1]:
            return self.core_matrix
        if self._column_views_base != id(self.core_matrix):
            self._column_views = {}
            self._column_views_base = id(self.core_matrix)
        view = self._column_views.get((start, end))
        if view is None:
            view = self.core_matrix[:, start:end]
            self._column_views[(start, end)] = view
        return view
