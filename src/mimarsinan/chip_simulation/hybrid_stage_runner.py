"""Shared hybrid stage loop for nevresim, Lava, and SANA-FE backends."""

from __future__ import annotations

from typing import Any, Callable, Dict


def run_hybrid_stages(
    mapping,
    state_buffer: Dict[int, Any],
    *,
    on_neural,
    on_compute,
    finalize,
) -> Any:
    """Iterate ``mapping.stages``; neural/compute callbacks mutate ``state_buffer``."""
    for stage in mapping.stages:
        if stage.kind == "neural":
            on_neural(stage, state_buffer)
        elif stage.kind == "compute":
            on_compute(stage, state_buffer)
    return finalize(state_buffer)
