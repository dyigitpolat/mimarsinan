"""IRMapping: full-weight mapping that produces an ``IRGraph`` with concrete
``NeuralCore`` / ``ComputeOp`` / ``WeightBank`` nodes.

All structural decisions (tiling mode, psum decomposition, coalescing,
bias-axon counting, shared-bank wiring) live in the base class
``LayoutIRMapping``.  This subclass only attaches weight material and builds
the graph, guaranteeing the emitted softcore shapes are byte-identical to
what the wizard / architecture-search path predicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import (
    ComputeOp,
    IRGraph,
    IRNode,
    IRSource,
    NeuralCore,
    WeightBank,
    spike_source_to_ir_source,
)
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping


from mimarsinan.mapping.ir_mapping_class_base import IRMappingCore
from mimarsinan.mapping.ir_mapping_class_emit import IRMappingEmitMixin

class IRMapping(IRMappingEmitMixin, IRMappingCore):
    """Unified IR mapping with concrete graph emission."""
