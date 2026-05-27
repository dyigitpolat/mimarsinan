from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class MappingVerificationResult:
    """Result of a soft-core mapping verification pass."""

    feasible: bool
    softcores: List[LayoutSoftCoreSpec]

    # Summary statistics
    num_neural_cores: int
    max_input_size: int
    max_output_size: int
    total_area: int

    # Layout-derived count of host-side compute segments between / around
    # neural segments. Used by the wizard as the "Sync Barriers" metric.
    host_side_segment_count: int = 0

    # Compact layout-only flow summary for the wizard miniview.
    layout_preview: Optional[Dict[str, Any]] = None

    # Optional error message when feasible=False
    error: Optional[str] = None

