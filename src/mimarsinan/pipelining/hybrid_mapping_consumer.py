"""Shared helpers for pipeline steps that consume cached hybrid mappings."""

from __future__ import annotations

from typing import Any

from mimarsinan.pipelining.pipeline_helpers import require_spiking_mode_supported
from mimarsinan.pipelining.simulation_factory import build_hybrid_mapping_for_pipeline


def load_hybrid_mapping_for_step(pipeline: Any, *, rebuild: bool = False) -> Any:
    """Return ``hybrid_mapping`` from cache or build via ``simulation_factory``."""
    if not rebuild:
        try:
            return pipeline.get_entry("hybrid_mapping")
        except KeyError:
            pass
    return build_hybrid_mapping_for_pipeline(pipeline)


def require_lif_for_backend(pipeline: Any, step_name: str, *, backend: str) -> None:
    require_spiking_mode_supported(pipeline, step_name, backend=backend)
