"""Shared helpers for pipeline steps that consume cached hybrid mappings."""

from __future__ import annotations

from typing import Any

from mimarsinan.pipelining.core.engine.pipeline_helpers import require_spiking_mode_supported
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

_HYBRID_MAPPING_CACHE_KEY = "hybrid_mapping"


def load_hybrid_mapping_for_step(
    pipeline: Any,
    step: Any,
    *,
    rebuild: bool = False,
    cache_key: str = _HYBRID_MAPPING_CACHE_KEY,
) -> Any:
    """Return the cached hybrid mapping, or build it from the step's IR graph and platform constraints.

    A cached mapping is trusted only when its ``source_ir_build_token`` matches the
    step's current ``ir_graph`` (a resumed run that regenerated the graph must not
    reuse the previous run's packed mapping).
    """
    ir_graph = step.get_entry("ir_graph")
    if not rebuild:
        cached = pipeline.cache.get(cache_key)
        if cached is not None and (
            getattr(cached, "source_ir_build_token", None)
            == getattr(ir_graph, "build_token", None)
        ):
            return cached

    platform_constraints = step.get_entry("platform_constraints_resolved")
    hybrid_mapping = build_hybrid_mapping_for_pipeline(
        ir_graph,
        platform_constraints,
        pipeline_config=pipeline.config,
    )
    pipeline.cache.add(cache_key, hybrid_mapping, "pickle")
    return hybrid_mapping


def require_lif_for_backend(pipeline: Any, step_name: str, *, backend: str) -> None:
    require_spiking_mode_supported(pipeline, step_name, backend=backend)
