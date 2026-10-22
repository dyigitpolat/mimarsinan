"""Leaf decomposition under ``decompose_leaf_modules``: attention and encoder leaves replaced before tracing
(``registry.py``, ``attention.py``), the recurrent leaves lowered at conversion (``recurrence.py``), and the probe's
native reference of a decomposed program (``reference.py``)."""

from mimarsinan.torch_mapping.leaf_decomposition.reference import decomposition_reference
from mimarsinan.torch_mapping.leaf_decomposition.registry import (
    ActivationMaskError,
    Decomposition,
    decompose_leaves,
    refuse_activation_mask,
)

__all__ = ["ActivationMaskError", "Decomposition", "decompose_leaves", "decomposition_reference",
           "refuse_activation_mask"]
