"""[calculus §16] the synchronized count-domain HCM executor: [U-LIF2 W9] the closed form of integrate then emit."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.chip_simulation.value_run.value_execution import dependency_order
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.serial.refusals import (
    refuse_cycle_atomic,
    refuse_saturating_membrane,
    refuse_zero_reset_window_count,
)
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.models.spiking.wire_semantics import lif_window_count


def integrates_then_emits(flow) -> bool:
    """[U-LIF2 W9] Whether ``flow`` executes integrate then emit (its two-window flag at LIF): its executor of record is
    the gated cycle-level executor (``gated_reference.run_integrate_then_emit``), recording or not. One predicate for the
    executed path (``_run_neural_segment_rate``) and for the input it reads (``_encode_segment_input``): a window-count
    reader reads each slice's count, so a pass's delivery delay, a shift in time, does not apply to its input."""
    policy = cycle_neuron_policy(flow.spiking_mode, flow.ttfs_cycle_schedule, flow.firing_mode,
                                 soma_law=getattr(flow, "soma_law", DEFAULT_SOMA_LAW))
    return (bool(getattr(flow, "lif_execution_synchronized", False)) and flow.spiking_mode == "lif"
            and not policy.single_spike_io)


def runs_closed_form(flow, *, recording: bool) -> bool:
    """[U-LIF2 W9] Whether ``run_neural_segment_counts``, the closed form of integrate then emit, computes a segment of
    ``flow``: only where it is asked for by its name (``window_count_closed_form``, the certification's analytic
    gauge), never for a spike record, and never as the executor of record (the owner's law of 2026-07-22)."""
    return integrates_then_emits(flow) and bool(getattr(flow, "window_count_closed_form", False)) and not recording


def run_neural_segment_counts(
    flow, input_spike_train, *, seg, T, batch_size, device, soma_law,
    neuron_count_recorder: dict | None = None,
) -> torch.Tensor:
    """The closed form of integrate then emit: per core, memb = W·counts +
    bias·T (+ V0·θ) and the emitted count is the staircase's count of that charge — bit-equal
    to two-window execution for ANY arrival (§16 theorem), and latch-correct
    for level-gapped consumers (counts have no windows). [U-LIF2 W9] Equal to the
    gated cycle-level executor (``gated_window``) on every neuron window, never load-bearing.

    That theorem's hypotheses are the subtractive reset (a zero-reset
    emission window emits at most one spike, never the window's count) on an
    UNBOUNDED accumulator with at most one spike per cycle, so the reset and
    both soma axes can deny it — and this executor then refuses instead of
    approximating.
    """
    refuse_cycle_atomic(
        soma_law, mechanism="the synchronized count executor",
        theorem=(
            "its closed-form staircase is a THEOREM that a window's count "
            "depends only on the total integrated charge, order-independently."
        ),
    )
    refuse_saturating_membrane(
        soma_law, mechanism="the synchronized count executor",
        identity=(
            "its closed form models the membrane as a linear accumulator over "
            "the whole window."
        ),
    )
    refuse_zero_reset_window_count(
        getattr(flow, "firing_mode", None), mechanism="the synchronized count executor")
    cores = seg["cores"]
    counts_in = input_spike_train.sum(dim=0)
    buffers = [
        torch.zeros(
            batch_size, max(int(c.neurons_per_core - c.available_neurons), 1),
            device=device, dtype=COMPUTE_DTYPE,
        )
        for c in cores
    ]
    signals = [
        torch.zeros(
            batch_size, max(int(c.axons_per_core - c.available_axons), 1),
            device=device, dtype=COMPUTE_DTYPE,
        )
        for c in cores
    ]
    v0 = float(getattr(flow, "lif_membrane_init", 0.0))
    # [U1 R6] Dependency order, ties by stored latency then index: a pass pinned at the fused latencies can give a
    # producer the latency of the consumer that reads it, so the latency order alone may walk the reader first.
    reads = [{int(sp.src_core) for sp in spans if sp.kind == "core"} for spans in seg["axon_spans"]]
    order = [i for i in dependency_order(reads, [int(c.latency or 0) for c in cores]) if cores[i].latency is not None]
    for i in order:
        # Always-on axons deliver one spike per cycle => count T.
        seg["axon_fill_plans"][i].apply(
            signals[i], input_spikes=counts_in, buffers=buffers,
            on_value=float(T),
        )
        total = torch.matmul(seg["core_params"][i], signals[i].T).T
        bias = seg["hw_biases"][i]
        if bias is not None:
            total = total + bias * float(T)
        theta = seg["thresholds"][i]
        if v0:
            total = total + v0 * theta
        # The emitted count read off the window charge (``lif_window_count``): exact on the integer chip, where
        # the staircase rebuilt as ``stair * T / theta`` left the lattice beyond T <= 8 (H8's count_rebuild).
        buffers[i] = lif_window_count(total, theta, T, compare_mode=flow.thresholding_mode)
        if neuron_count_recorder is not None:
            neuron_count_recorder[i] = buffers[i].detach()

    output_counts = torch.zeros(
        batch_size, len(seg["output_sources"]), device=device,
        dtype=COMPUTE_DTYPE,
    )
    for sp in seg["output_spans"]:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
        elif sp.kind == "input":
            output_counts[:, d0:d1] = counts_in[:, int(sp.src_start):int(sp.src_end)]
        else:
            output_counts[:, d0:d1] = buffers[int(sp.src_core)][
                :, int(sp.src_start):int(sp.src_end)
            ]
    return output_counts
