"""[U-LIF2 W9] The gated cycle-level executor of integrate then emit, stage-flat (the packed layout).

A synchronized LIF hop that integrates then emits runs on the per-cycle LIF physics with its fire gate
(``LIFCyclePolicy.integrate`` and ``emit``): the load-bearing executor of the value. The count executor
(``sync_counts.run_neural_segment_counts``) is its closed form, equal to it on every neuron window and never
load-bearing (the owner's law of 2026-07-22).

The gate of one execution stage (a pass of a segment, or one level stage of a re-timed hop). A core's level is 0 when it
reads no stepped core of the stage, else one more than the deepest stepped stage core it reads: read from the stage's own
wiring (its axon spans), never from the latency tags, and a stage input (a slice another pass carried included) is a
stage input. A core of level l starts at s = l (T + 1): it integrates every arrival from the stage's start until cycle
s + T, takes its stage input, its always-on axons and its per-cycle bias on its own T cycles [s, s + T), then fires and
subtracts on the T input-free cycles [s + T, s + 2T); a consumer reads its producers' fires of the previous cycle, so a
producer's emission arrives on [s_p + T + 1, s_p + 2T + 1), inside every consumer's integration. A stage of depth D runs
(D + 1) T + D cycles. The membrane guard precharges v0 theta, as the closed form adds it; the phase dither moves arrivals
inside the integration window only, so it is inert."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import torch

from mimarsinan.chip_simulation.stage_timesteps import integrate_then_emit_cycles, window_arrival_cycles, window_start
from mimarsinan.chip_simulation.stage_wiring import dependency_levels
from mimarsinan.models.spiking.hybrid.executors.packed_stage import build_packed_stage
from mimarsinan.models.spiking.hybrid.membrane_readout import stash_membrane_readout_correction
from mimarsinan.models.spiking.serial.refusals import (
    refuse_cycle_atomic,
    refuse_saturating_membrane,
    refuse_zero_reset_window_count,
)
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE

GATED_EXECUTOR = "the gated integrate-then-emit executor"


@dataclass(frozen=True)
class GatedWindowPlan:
    """One execution stage's gate: each core's level and start cycle (``None`` for a core never stepped), the stage's
    depth and the cycles it runs; [U-LIF2 r6] each core's stepped in-stage producers (the reads it levels by)."""

    levels: Tuple[Optional[int], ...]
    starts: Tuple[Optional[int], ...]
    depth: int
    cycles: int
    reads: Tuple[Tuple[int, ...], ...] = ()


def gated_window_cycles(depth: int, T: int) -> int:
    """A stage of dependency depth D runs (D + 1) T + D cycles under integrate then emit; a stage of no core, none: the
    one rule every reader of a stage's cycles prices (``stage_timesteps.integrate_then_emit_cycles``)."""
    return integrate_then_emit_cycles(depth, T)


def stage_core_reads(seg: dict) -> Tuple[Tuple[int, ...], ...]:
    """Each core's stepped in-stage producers, from the stage's own core reads (its axon spans of kind ``core``)."""
    cores = seg["cores"]
    stepped = [c.latency is not None for c in cores]
    return tuple(tuple(sorted({int(sp.src_core) for sp in spans if sp.kind == "core" and 0 <= int(sp.src_core) <
                               len(cores) and stepped[int(sp.src_core)]})) for spans in seg["axon_spans"])


def stage_dependency_levels(seg: dict) -> Tuple[Optional[int], ...]:
    """Each stepped core's dependency level in its execution stage, from the stage's own core reads, by the one
    leveling (``stage_wiring.dependency_levels``, the R6 walk order, which refuses a cycle of reads by name); a core
    never stepped (no latency) has none and is no producer. The dependency order walks every producer before its
    consumers; a walk that reached a consumer first (only the R6 ablation's stored-latency walk) reads that producer
    as not yet run, as every walk of the base tree read an unwritten buffer."""
    cores = seg["cores"]
    return dependency_levels(stage_core_reads(seg), [int(c.latency or 0) for c in cores],
                             [c.latency is not None for c in cores])


def gated_window_plan(seg: dict, T: int) -> GatedWindowPlan:
    """The stage's gate at window ``T``, memoized on the segment tensors (static per stage)."""
    key = f"gated_window_plan_T{int(T)}"
    plan = seg.get(key)
    if plan is None:
        levels = stage_dependency_levels(seg)
        depth = 1 + max((lv for lv in levels if lv is not None), default=-1)
        starts = tuple(None if lv is None else window_start(lv, T) for lv in levels)
        plan = GatedWindowPlan(levels=levels, starts=starts, depth=depth, cycles=gated_window_cycles(depth, T),
                               reads=stage_core_reads(seg))
        seg[key] = plan
    return plan


def require_gated_window_admissible(soma_law, firing_mode) -> None:
    """Refuse by name what an input-free emission window cannot run: a per-event comparison, a zero reset (its
    emission window emits at most one spike, never the window's count) and a register that clamps (it loses the
    window charge it clamps). A signed register that asserts its rails runs, its own assertion the guard."""
    refuse_cycle_atomic(soma_law, mechanism=GATED_EXECUTOR, theorem=(
        "its emission window fires each neuron at most once a cycle on the whole window's membrane."))
    if soma_law is not None and not soma_law.asserts_no_saturation:
        refuse_saturating_membrane(soma_law, mechanism=GATED_EXECUTOR, identity=(
            "it holds the window's whole charge before it emits, and a register that clamps loses what it clamps."))
    refuse_zero_reset_window_count(firing_mode, mechanism=GATED_EXECUTOR)


def _bucket_arrivals(seg: dict, packed, plan: GatedWindowPlan, T: int) -> List[FrozenSet[int]]:
    """Per bucket, the cycles before its own window on which a producer's emission arrives (the only pre-window
    cycles whose charge is not zero): [U-LIF2 r6] its cores' ``stage_timesteps.window_arrival_cycles``, the rule the
    cost rule prices (``neuron_cycles_per_window``)."""
    per_core = window_arrival_cycles(plan.reads, plan.levels, T)
    return [frozenset().union(*(per_core[core] for core in bucket.core_indices)) for bucket in packed.buckets]


def run_neural_segment_gated(
    flow, input_spike_train, *, seg, stage, T, batch_size, device, policy,
    readout_corrections=None, neuron_count_recorder: Optional[dict] = None,
) -> torch.Tensor:
    """The gated packed executor: cores bucketed by (start, axons, neurons), each cycle's active buckets gathered
    against the previous cycle's fires, then integrated (fire gate closed) or emitted; returns the segment's output
    counts ``(B, out)``. ``neuron_count_recorder`` receives each stepped core's window counts ``(B, n)``."""
    require_gated_window_admissible(getattr(flow, "soma_law", None), getattr(flow, "firing_mode", None))
    T = int(T)
    plan = gated_window_plan(seg, T)
    if seg.get("packed_gated_window") != T:   # the layout's starts are the window's: rebuilt for another one
        seg["packed_gated"] = build_packed_stage(seg, device, starts=plan.starts)
        seg["gated_arrivals"] = _bucket_arrivals(seg, seg["packed_gated"], plan, T)
        seg["packed_gated_window"] = T
    packed, arrivals = seg["packed_gated"], seg["gated_arrivals"]
    width = max(packed.total_neurons, 1)
    state = policy.make_state(batch_size, width, device, COMPUTE_DTYPE)
    membrane_init = float(getattr(flow, "lif_membrane_init", 0.0))
    if membrane_init and packed.total_neurons:
        state["memb"] += membrane_init * packed.theta_flat
    fires = torch.zeros(batch_size, width, device=device, dtype=COMPUTE_DTYPE)
    counts = torch.zeros_like(fires)
    zeros_in = torch.zeros(batch_size, input_spike_train.shape[2], device=device, dtype=COMPUTE_DTYPE)
    train = input_spike_train.to(COMPUTE_DTYPE)

    for cycle in range(plan.cycles):
        staged, closing = [], []
        for bucket, arrive in zip(packed.buckets, arrivals):
            start = bucket.latency
            if cycle >= start + 2 * T:
                if cycle == start + 2 * T:
                    closing.append(bucket)
                continue
            if cycle >= start + T:
                staged.append((bucket, None, False))
                continue
            own = cycle >= start
            if not own and cycle not in arrive:
                continue
            signals = torch.zeros(batch_size, len(bucket.core_indices) * bucket.n_axons, device=device,
                                  dtype=COMPUTE_DTYPE)
            if own and bucket.on_dst is not None:
                signals.index_fill_(1, bucket.on_dst, 1.0)
            if own and bucket.inp_dst is not None and bucket.inp_src is not None:
                local = cycle - start
                src = train[local] if local < train.shape[0] else zeros_in
                signals.index_copy_(1, bucket.inp_dst, src.index_select(1, bucket.inp_src))
            if bucket.buf_dst is not None and bucket.buf_src is not None:
                signals.index_copy_(1, bucket.buf_dst, fires.index_select(1, bucket.buf_src))
            staged.append((bucket, signals, own))
        for bucket, signals, own in staged:
            n0, n1 = bucket.neuron_start, bucket.neuron_end
            slice_state = {k: v[:, n0:n1] for k, v in state.items()}
            if signals is None:
                out = policy.emit(slice_state, packed.theta_flat[n0:n1], thresholding_mode=flow.thresholding_mode,
                                  output_dtype=COMPUTE_DTYPE)
                fires[:, n0:n1] = out
                counts[:, n0:n1] += out
                continue
            group = len(bucket.core_indices)
            charge = torch.einsum("gna,bga->bgn", bucket.weights, signals.reshape(batch_size, group, bucket.n_axons))
            if own and bucket.bias is not None:
                charge = charge + bucket.bias
            policy.integrate(slice_state, charge.reshape(batch_size, -1))
        for bucket in closing:
            fires[:, bucket.neuron_start:bucket.neuron_end] = 0.0

    cores = seg["cores"]
    if neuron_count_recorder is not None:
        for i, off in packed.neuron_offset.items():
            neuron_count_recorder[i] = counts[:, off:off + max(int(cores[i].neurons_per_core
                                                                    - cores[i].available_neurons), 1)].detach()
    output_counts = torch.zeros(batch_size, len(seg["output_sources"]), device=device, dtype=COMPUTE_DTYPE)
    input_total = train[:T].sum(dim=0)
    for sp in seg["output_spans"]:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
        elif sp.kind == "input":
            output_counts[:, d0:d1] = input_total[:, int(sp.src_start):int(sp.src_end)]
        elif sp.kind != "off" and packed.neuron_offset.get(int(sp.src_core)) is not None:
            off = packed.neuron_offset[int(sp.src_core)]
            output_counts[:, d0:d1] = counts[:, off + int(sp.src_start):off + int(sp.src_end)]
    if readout_corrections is not None and getattr(flow, "membrane_readout", False):
        states = []
        for i, c in enumerate(cores):
            n = max(int(c.neurons_per_core - c.available_neurons), 1)
            off = packed.neuron_offset.get(i)
            states.append({"memb": state["memb"][:, off:off + n] if off is not None
                           else torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE)})
        stash_membrane_readout_correction(
            flow, seg=seg, stage=stage, output_counts=output_counts, output_spans=seg["output_spans"],
            neuron_states=states, thresholds=seg["thresholds"], single_spike=False,
            readout_corrections=readout_corrections)
    return output_counts
