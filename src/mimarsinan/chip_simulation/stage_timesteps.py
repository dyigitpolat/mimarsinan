"""The executed-window rule: how many timesteps a neural stage actually runs.

ONE home, because two readers need it and they must agree: the SANA-FE
runner (which sizes the simulation) and the candidate quantity extraction
(which prices the wall before any simulation exists). A stage does NOT run
``T`` steps — it runs ``T`` plus the latency its cores sit at plus one cycle
for input delivery, and a retimed program executes one stage per depth
level, so a program's step count is a SUM over stages.

The candidate previously used ``timesteps x neural_segment_count``, a second
formula for the same quantity: measured on the sealed MLP study run it said
4 where the record measured 15, under-charging the wall 3.75x and with it
every term that multiplies latency (static energy, e2e, throughput).
"""

from __future__ import annotations

from typing import Any, FrozenSet, Iterable, Optional, Sequence, Tuple

from mimarsinan.chip_simulation.window_firing_axis import INTEGRATE_THEN_EMIT


def integrate_then_emit_cycles(depth: int, timesteps: int) -> int:
    """[U-LIF2 W11] A stage of dependency depth D runs (D + 1) T + D cycles under integrate then emit (each level emits
    its window while the next integrates, one delivery cycle a level); a stage of no core, none. The one rule the gated
    executor runs (``gated_window.gated_window_plan``) and every reader of a stage's cycles prices."""
    depth, T = int(depth), int(timesteps)
    return (depth + 1) * T + depth if depth > 0 else 0


def executed_stage_timesteps(
    *,
    timesteps: int,
    max_latency: int,
    is_cycle: bool,
    is_cascade: bool,
    latency_group_count: Optional[int] = None,
    chip_latency: Optional[int] = None,
    integrates_then_emits: bool = False,
    depth: Optional[int] = None,
) -> int:
    """Timesteps ONE neural stage executes under its firing semantics.

    ``is_cycle`` (synchronized TTFS) runs a whole window per latency group;
    ``is_cascade`` spans the full chip latency; everything else (LIF, the
    default) runs the window plus the stage's own latency. The trailing +1
    on both non-cycle branches is the one-cycle input-delivery delay.

    [U-LIF2 W11] A synchronized LIF stage that integrates then emits runs ``integrate_then_emit_cycles`` of its
    dependency depth ``depth``, read from the stage's own wiring (never from the latency tags), which it requires.
    Firing while integrating keeps T + L + 1, ``max_latency`` being the latency the stage's executor loops on (its
    chip latency less one, ``ChipLatency``: the stage's greatest executed core latency, from cycle 0).
    """
    T = int(timesteps)
    if integrates_then_emits:
        if is_cycle or is_cascade:
            raise ValueError("integrate then emit is a synchronized LIF firing; a TTFS stage never takes it")
        if depth is None:
            raise ValueError(
                "a stage that integrates then emits runs (D + 1) T + D cycles from its dependency depth D, read from "
                "the stage's wiring, which was not supplied")
        return integrate_then_emit_cycles(depth, T)
    if is_cycle:
        if latency_group_count is None:
            raise ValueError(
                "cycle-based TTFS sizes its window from the latency_group_count, "
                "which was not supplied")
        return (int(latency_group_count) + 1) * T
    if is_cascade:
        if chip_latency is None:
            raise ValueError(
                "cascaded TTFS sizes its window from the whole chip_latency "
                "(not the max core latency), which was not supplied")
        return T + int(chip_latency) + 1
    return T + int(max_latency) + 1


def program_latency_steps(
    *,
    stage_max_latencies: Sequence[int],
    timesteps: int,
    is_cycle: bool,
    is_cascade: bool,
    latency_group_count: Optional[int] = None,
    chip_latency: Optional[int] = None,
    integrates_then_emits: bool = False,
    stage_depths: Optional[Sequence[int]] = None,
) -> int:
    """Σ over the program's execution stages of their executed windows.

    ``stage_max_latencies`` is one entry per stage — for a retimed program,
    one per depth LEVEL of each segment (and per pass, when the schedule cuts
    the segment), holding the greatest core latency inside that stage.
    [U-LIF2 W11] ``stage_depths``, one entry per stage in the same order, holds each stage's dependency depth from its
    wiring; integrate then emit requires it.
    """
    depths: Sequence[Optional[int]] = (
        [None] * len(stage_max_latencies) if stage_depths is None else list(stage_depths))
    if len(depths) != len(stage_max_latencies):
        raise ValueError(f"{len(depths)} stage depths for {len(stage_max_latencies)} stages")
    return sum(
        executed_stage_timesteps(
            timesteps=timesteps, max_latency=latency,
            is_cycle=is_cycle, is_cascade=is_cascade,
            latency_group_count=latency_group_count,
            chip_latency=chip_latency,
            integrates_then_emits=integrates_then_emits, depth=depth,
        )
        for latency, depth in zip(stage_max_latencies, depths)
    )


def per_cycle_spans(spans: Sequence[int], depths: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """[U-LIF2 W11, the review's p-1] The latency a synchronized LIF stage's per-cycle executor loops on: at least the
    stage's wiring depth less one (its chip latency enforces each consumer a cycle after its producer, while the
    shiftable-core alignment can raise a producer into its consumer's tag and shrink the tag span), else the tag
    span."""
    if depths is None:
        return tuple(int(s) for s in spans)
    return tuple(max(int(s), int(d) - 1) for s, d in zip(spans, depths))


def semantics_program_steps(semantics: Any, spans: Sequence[int], depths: Optional[Sequence[int]],
                            timesteps: Optional[int]) -> Optional[int]:
    """A planned program's steps under the run's firing semantics (``StageSemantics``, duck-typed): per stage its tag
    span and [U-LIF2 W11] its wiring depth (None without the wire census). A stage that integrates then emits runs
    (D + 1) T + D cycles, unstated without its depth; a synchronized LIF stage that fires while integrating runs its
    executor's loop (``per_cycle_spans``). None without a window."""
    sync = getattr(semantics, "window_firing", None) is not None
    ite = bool(getattr(semantics, "integrates_then_emits", False))
    if timesteps is None or (ite and depths is None):
        return None
    return program_latency_steps(
        stage_max_latencies=per_cycle_spans(spans, depths) if sync else spans, timesteps=int(timesteps),
        is_cycle=semantics.is_cycle, is_cascade=semantics.is_cascade, latency_group_count=semantics.latency_group_count,
        chip_latency=semantics.chip_latency, integrates_then_emits=ite, stage_depths=depths if ite else None)


def sealed_stage_steps(*, timesteps: int, window_firing: Optional[str], depth: Optional[int],
                       delays: Sequence[Optional[int]]) -> Optional[int]:
    """[U-LIF2 W11] A synchronized LIF stage's cycles read off its sealed program, for a record no simulator timed:
    integrate then emit runs (D + 1) T + D from the stage's sealed dependency depth; firing while integrating runs
    T + L + 1, L the greatest stated core latency (the latency the stage's executor loops on from cycle 0, the review's
    p-1). None where the record states too little: no window firing (away from the synchronized LIF variant), no depth
    (a format-1 record), a core with no latency."""
    if window_firing is None:
        return None
    if window_firing == INTEGRATE_THEN_EMIT:
        return None if depth is None else integrate_then_emit_cycles(depth, timesteps)
    stated = [int(d) for d in delays if d is not None]
    if not delays or len(stated) != len(delays):
        return None
    return executed_stage_timesteps(timesteps=timesteps, max_latency=max(stated), is_cycle=False, is_cascade=False)


def sealed_program_steps(segments: Sequence[Any], timesteps: int, window_firing: Optional[str]) -> Optional[int]:
    """[U-LIF2 W11] A synchronized LIF program's compute steps from its own sealed stages (``SegmentRecord``s, each by
    ``sealed_stage_steps``), for a run no simulator timed; None when any stage states too little."""
    steps = [sealed_stage_steps(timesteps=int(timesteps), window_firing=window_firing, depth=segment.dependency_depth,
                                delays=[core.static_delay_levels for core in segment.cores]) for segment in segments]
    return None if not steps or any(s is None for s in steps) else sum(int(s or 0) for s in steps)


def window_start(level: int, timesteps: int) -> int:
    """[U-LIF2 W9, r6] The cycle a core of dependency ``level`` opens its own window on under integrate then emit:
    l (T + 1), each level a window and a delivery cycle after the one below (the gate's starts)."""
    return int(level) * (int(timesteps) + 1)


def pre_window_emissions(reads: Sequence[Iterable[int]],
                         levels: Sequence[Optional[int]]) -> Tuple[FrozenSet[int], ...]:
    """[U-LIF2 r6, m-2] Per core of an execution stage that integrates then emits, the levels of its stage producers
    whose emission reaches it before its own window opens: a producer of level p emits on the T input-free cycles
    after its window, arriving on [window_start(p + 1), window_start(p + 1) + T), inside the window of a consumer one
    level above and before the window of a consumer two or more levels above (a wire that skips a level without a
    relay). A core with no level (never stepped) integrates none and is no producer. ``reads[i]``: core i's producers
    in the stage. The one reading of a stage's wiring the gate (``window_arrival_cycles``) and the cost rule
    (``neuron_cycles_per_window``) take, free of the window length."""
    return tuple(frozenset() if level is None else frozenset(
        int(levels[p]) for p in producers if levels[p] is not None and int(levels[p]) <= int(level) - 2)  # type: ignore
        for producers, level in zip(reads, levels))


def window_arrival_cycles(reads: Sequence[Iterable[int]], levels: Sequence[Optional[int]],
                          timesteps: int) -> Tuple[FrozenSet[int], ...]:
    """[U-LIF2 r6, m-2] Per core, the cycles before its own window on which a producer's emission arrives, the only
    pre-window cycles whose charge is not zero (the gate integrates them): T cycles for each level of
    ``pre_window_emissions``."""
    T = int(timesteps)
    return tuple(frozenset(cycle for p in early for cycle in range(window_start(p + 1, T), window_start(p + 1, T) + T))
                 for early in pre_window_emissions(reads, levels))


def stage_window_arrivals(neurons: Sequence[int], reads: Sequence[Iterable[int]],
                          levels: Sequence[Optional[int]]) -> int:
    """[U-LIF2 r6, m-2] The producer windows an execution stage's neurons integrate before their own windows,
    neuron-weighted (each core's neurons times its count of ``pre_window_emissions``): free of the window length, so a
    program states it once (the record's ``SegmentRecord.window_arrivals``)."""
    return sum(int(n) * len(early) for n, early in zip(neurons, pre_window_emissions(reads, levels)))


def neuron_cycles_per_window(timesteps: int, window_firing: Optional[str], *, neurons: Optional[float] = None,
                             window_arrivals: int = 0) -> float:
    """[U-LIF2 W12] The cycles a neuron runs per inference window, the multiplicand of the neuron-update and leak
    energy terms: T, or for a synchronized LIF neuron that integrates then emits 2T (T integrating, T emitting).
    [U-LIF2 r6, m-2] Under integrate then emit each core also integrates the arrival cycles before its own window, T
    for each producer window it takes early (``stage_window_arrivals``, read from each stage's wiring by the gate's own
    ``pre_window_emissions``): over ``neurons`` driven neurons the mean is T (2 + window_arrivals / neurons), so
    ``neurons`` x the value prices each core at 2T plus its arrival cycles. A program whose stages take no producer
    window early (every in-stage wire spans one level, as on every re-timed stage) reads 2T."""
    T = int(timesteps)
    if window_firing != INTEGRATE_THEN_EMIT:
        return T
    if not window_arrivals:
        return 2 * T
    if not neurons:
        raise ValueError(f"{window_arrivals} producer windows integrated early over no driven neuron: a record that "
                         "states its arrivals states the neurons they price")
    return T * (2 + int(window_arrivals) / float(neurons))
