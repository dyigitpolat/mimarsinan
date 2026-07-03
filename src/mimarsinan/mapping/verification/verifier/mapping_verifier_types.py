from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec

@dataclass
class MappingVerificationResult:
    """Result of a soft-core mapping verification pass."""

    feasible: bool
    softcores: List[LayoutSoftCoreSpec]

    num_neural_cores: int
    max_input_size: int
    max_output_size: int
    total_area: int

    host_side_segment_count: int = 0

    layout_preview: Optional[Dict[str, Any]] = None

    error: Optional[str] = None

