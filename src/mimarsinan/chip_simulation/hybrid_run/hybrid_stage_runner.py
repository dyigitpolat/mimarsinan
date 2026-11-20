"""Shared hybrid stage loop for nevresim, Lava, SANA-FE, and HCM backends."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from mimarsinan.common.refusal import ToolchainRefusal


class StageOrderError(ToolchainRefusal, AssertionError):
    """A stage reads a node no earlier stage produced: the program does not run in dependency order (H9)."""


def require_inputs_produced(stage, state_buffer: Dict[int, Any]) -> None:
    """[theorem box H9] every node a stage reads is the raw input or an earlier stage's stored output."""
    if stage.kind == "neural":
        reads = {int(s.node_id) for s in getattr(stage, "input_map", None) or ()}
    elif stage.kind == "compute" and getattr(getattr(stage, "compute_op", None), "input_sources", None) is not None:
        sources = stage.compute_op.input_sources
        flat = sources.flatten() if hasattr(sources, "flatten") else sources
        reads = {int(src.node_id) for src in flat if getattr(src, "node_id", None) is not None}
    else:
        return
    missing = sorted(n for n in reads if n >= 0 and n not in state_buffer)
    if missing:
        raise StageOrderError(
            f"stage {getattr(stage, 'name', '?')!r} reads node(s) {missing[:8]} that no earlier stage produced; a program "
            f"runs its stages in dependency order (a pass after every pass it reads)")


@dataclass
class HybridStageContext:
    """Per-stage context passed to hybrid stage callbacks."""

    stage_index: int
    stage: Any
    state_buffer: Dict[int, Any]
    remaining: Optional[Dict[int, int]] = None
    state_buffer_spikes: Optional[Dict[int, Any]] = None
    recorder: Any = None


def execution_neural_stages(stage) -> list:
    """The neural stages a backend EXECUTES for one program stage.

    [C3 fused] a re-timed fused stage carries per-depth-level stages — each is
    run through the backend's ordinary per-stage path (assemble → boundary
    chain → encode → run → decode → store, real node-id I/O) — otherwise the
    stage itself executes directly.
    """
    levels = getattr(stage, "retimed_level_stages", None)
    return list(levels) if levels else [stage]


def enumerate_execution_stages(stages):
    """Yield ``(index, program_stage, exec_stage)`` for every EXECUTED unit,
    with the same index arithmetic ``run_hybrid_stages`` uses (the fused
    bookkeeping context of a re-timed stage consumes one index). Reference-
    driven runners iterate THIS so their indices match recorded ones."""
    index = 0
    for stage in stages:
        levels = (
            getattr(stage, "retimed_level_stages", None)
            if stage.kind == "neural" else None
        )
        if levels:
            for level_stage in levels:
                yield index, stage, level_stage
                index += 1
            index += 1
        else:
            yield index, stage, stage
            index += 1


def _pop_intermediate_level_outputs(stage, levels, state_buffer) -> None:
    """Drop level outputs consumed only inside the fused stage (never read
    downstream: they are not in the fused stage's output map)."""
    fused_output_ids = {int(s.node_id) for s in stage.output_map}
    for level_stage in levels:
        for s in level_stage.output_map:
            if int(s.node_id) not in fused_output_ids:
                state_buffer.pop(int(s.node_id), None)


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
    stage_timer=None,
    timed_samples: Optional[int] = None,
) -> Any:
    """Iterate ``mapping.stages`` with optional post-stage hooks.

    Callbacks may accept either ``HybridStageContext`` or the legacy
    ``(stage_index, stage, state_buffer)`` triple for backward compatibility.
    Stage indices enumerate EXECUTION units (level stages included), so both
    certification twins see identical per-unit ordinals.

    ``stage_timer`` (opt-in, default ``None`` — byte-identical behavior): a
    ``stage_timing.StageTimer`` that wraps every ``on_compute`` invocation to
    accumulate the measured host-op wall. Neural segments are untimed — the
    chip simulator measures those. [U-HOST] ``timed_samples``: the samples this
    traversal carries, stated by the runner that drives it (each timed call's
    count); a timed call without it refuses by name (``stage_timing``).
    """

    def _ctx(index: int, stage) -> HybridStageContext:
        if context_factory is not None:
            return context_factory(index, stage, state_buffer)
        return HybridStageContext(
            stage_index=index, stage=stage, state_buffer=state_buffer,
        )

    index = 0
    for stage in mapping.stages:
        if stage.kind == "neural":
            levels = getattr(stage, "retimed_level_stages", None)
            if levels:
                # The level stages ARE the neural execution; the fused stage
                # keeps program-level bookkeeping (input decref) and its
                # outputs are stored by the levels under the same node ids.
                for level_stage in levels:
                    require_inputs_produced(level_stage, state_buffer)
                    _invoke_cb(on_neural, _ctx(index, level_stage))
                    index += 1
                if after_neural is not None:
                    _invoke_cb(after_neural, _ctx(index, stage))
                index += 1
                _pop_intermediate_level_outputs(stage, levels, state_buffer)
                continue
            ctx = _ctx(index, stage)
            index += 1
            require_inputs_produced(stage, state_buffer)
            _invoke_cb(on_neural, ctx)
            if after_neural is not None:
                _invoke_cb(after_neural, ctx)
        elif stage.kind == "compute":
            ctx = _ctx(index, stage)
            index += 1
            require_inputs_produced(stage, state_buffer)
            if stage_timer is not None:
                with stage_timer.time_compute_stage(ctx.stage_index, stage.name, timed_samples):
                    _invoke_cb(on_compute, ctx)
            else:
                _invoke_cb(on_compute, ctx)
            if after_compute is not None:
                _invoke_cb(after_compute, ctx)
        else:
            ctx = _ctx(index, stage)
            index += 1
            if on_unknown is not None:
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
