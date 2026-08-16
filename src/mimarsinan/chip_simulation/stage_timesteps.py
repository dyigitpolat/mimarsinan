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

from typing import Optional, Sequence


def executed_stage_timesteps(
    *,
    timesteps: int,
    max_latency: int,
    is_cycle: bool,
    is_cascade: bool,
    latency_group_count: Optional[int] = None,
    chip_latency: Optional[int] = None,
) -> int:
    """Timesteps ONE neural stage executes under its firing semantics.

    ``is_cycle`` (synchronized TTFS) runs a whole window per latency group;
    ``is_cascade`` spans the full chip latency; everything else (LIF, the
    default) runs the window plus the stage's own latency. The trailing +1
    on both non-cycle branches is the one-cycle input-delivery delay.
    """
    T = int(timesteps)
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
) -> int:
    """Σ over the program's execution stages of their executed windows.

    ``stage_max_latencies`` is one entry per stage — for a retimed program,
    one per depth LEVEL of each segment (and per pass, when the schedule cuts
    the segment), holding the greatest core latency inside that stage.
    """
    return sum(
        executed_stage_timesteps(
            timesteps=timesteps, max_latency=latency,
            is_cycle=is_cycle, is_cascade=is_cascade,
            latency_group_count=latency_group_count,
            chip_latency=chip_latency,
        )
        for latency in stage_max_latencies
    )
