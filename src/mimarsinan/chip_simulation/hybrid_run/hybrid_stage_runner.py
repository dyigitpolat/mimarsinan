"""Shared hybrid stage loop for nevresim, Lava, SANA-FE, and HCM backends."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class HybridStageContext:
    """Per-stage context passed to hybrid stage callbacks."""

    stage_index: int
    stage: Any
    state_buffer: Dict[int, Any]
    remaining: Optional[Dict[int, int]] = None
    state_buffer_spikes: Optional[Dict[int, Any]] = None
    recorder: Any = None


def run_hybrid_stages(
    mapping,
    state_buffer: Dict[int, Any],
    *,
    on_neural,
    on_compute,
    after_neural=None,
    after_compute=None,
    finalize=None,
    on_unknown=None,
    context_factory: Optional[Callable[[int, Any, Dict[int, Any]], HybridStageContext]] = None,
) -> Any:
    """Iterate ``mapping.stages`` with optional post-stage hooks.

    Callbacks may accept either ``HybridStageContext`` or the legacy
    ``(stage_index, stage, state_buffer)`` triple for backward compatibility.
    """
    for stage_index, stage in enumerate(mapping.stages):
        if context_factory is not None:
            ctx = context_factory(stage_index, stage, state_buffer)
        else:
            ctx = HybridStageContext(
                stage_index=stage_index,
                stage=stage,
                state_buffer=state_buffer,
            )

        if stage.kind == "neural":
            _invoke_cb(on_neural, ctx)
            if after_neural is not None:
                _invoke_cb(after_neural, ctx)
        elif stage.kind == "compute":
            _invoke_cb(on_compute, ctx)
            if after_compute is not None:
                _invoke_cb(after_compute, ctx)
        elif on_unknown is not None:
            _invoke_cb(on_unknown, ctx)
        else:
            raise ValueError(f"Unknown hybrid stage kind: {stage.kind!r}")
    if finalize is not None:
        return finalize(state_buffer)
    return state_buffer


def _invoke_cb(cb: Callable, ctx: HybridStageContext) -> None:
    sig = inspect.signature(cb)
    params = list(sig.parameters.values())
    if not params:
        cb()
        return
    first = params[0].name
    if first in ("ctx", "context"):
        cb(ctx)
    else:
        cb(ctx.stage_index, ctx.stage, ctx.state_buffer)
