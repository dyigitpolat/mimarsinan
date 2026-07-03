"""IRMapping: full-weight mapping producing an IRGraph of NeuralCore/ComputeOp/WeightBank nodes."""

from __future__ import annotations

from mimarsinan.mapping.ir_mapping_class_base import IRMappingCore
from mimarsinan.mapping.ir_mapping_class_emit import IRMappingEmitMixin



class IRMapping(IRMappingEmitMixin, IRMappingCore):
    """Unified IR mapping with concrete graph emission."""
