"""Shared hybrid stage loop for nevresim, Lava, and SANA-FE backends."""

from __future__ import annotations

from typing import Any, Callable, Dict


def run_hybrid_stages(
    mapping,
    state_buffer: Dict[int, Any],
    *,
    on_neural,
    on_compute,
    finalize=None,
    on_unknown=None,
) -> Any:
    """Iterate ``mapping.stages``; callbacks receive ``(stage_index, stage, state_buffer)``."""
    for stage_index, stage in enumerate(mapping.stages):
        if stage.kind == "neural":
            on_neural(stage_index, stage, state_buffer)
        elif stage.kind == "compute":
            on_compute(stage_index, stage, state_buffer)
        elif on_unknown is not None:
            on_unknown(stage_index, stage, state_buffer)
        else:
            raise ValueError(f"Unknown hybrid stage kind: {stage.kind!r}")
    if finalize is not None:
        return finalize(state_buffer)
    return state_buffer
