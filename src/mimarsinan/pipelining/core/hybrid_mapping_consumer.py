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
        cached = cached_hybrid_mapping(pipeline, step, cache_key=cache_key)
        if cached is not None:
            return cached

    platform_constraints = step.get_entry("platform_constraints_resolved")
    hybrid_mapping = build_hybrid_mapping_for_pipeline(
        ir_graph,
        platform_constraints,
        pipeline_config=pipeline.config,
    )
    pipeline.cache.add(cache_key, hybrid_mapping, "pickle")
    return hybrid_mapping


def cached_hybrid_mapping(pipeline: Any, step: Any, *, cache_key: str = _HYBRID_MAPPING_CACHE_KEY) -> Any:
    """The cached hybrid mapping ``load_hybrid_mapping_for_step`` would return without building, or None: the one
    whose ``source_ir_build_token`` is the step's current ``ir_graph``'s. [W15] The Hard Core Mapping step prices the
    static check against a compile wall only when the mapping was built."""
    cache = getattr(pipeline, "cache", None)
    if cache is None:                       # a pipeline without a cache holds no mapping to load
        return None
    ir_graph = step.get_entry("ir_graph")
    cached = cache.get(cache_key)
    if cached is not None and (
        getattr(cached, "source_ir_build_token", None) == getattr(ir_graph, "build_token", None)
    ):
        return cached
    return None


def require_lif_for_backend(pipeline: Any, step_name: str, *, backend: str) -> None:
    require_spiking_mode_supported(pipeline, step_name, backend=backend)
