from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class HardwareSuggestion:
    """Suggested ``platform_constraints``-ready core_types list and metadata."""

    core_types: List[Dict[str, Any]]
    total_cores: int
    rationale: str
    num_passes: int = 1
    estimated_latency_multiplier: float = 1.0

