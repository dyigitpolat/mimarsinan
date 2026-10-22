"""[calculus §16] the synchronized count-domain HCM reference executor."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.value_run.value_execution import dependency_order
from mimarsinan.models.spiking.cycle_policy import cycle_neuron_policy
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE
from mimarsinan.models.spiking.wire_semantics import lif_count_staircase


def runs_synchronized_counts(flow, *, recording: bool) -> bool:
    """Whether ``run_neural_segment_counts`` executes a neural segment of ``flow``: the synchronized LIF twin, unless a
    spike record is asked for (the per-core reference loop records one). One predicate for the executed path
    (``_run_neural_segment_rate``) and for the input it reads (``_encode_segment_input``): the count executor reads
    each slice's window count, so a pass's delivery delay, a shift in time, does not apply to its input."""
    policy = cycle_neuron_policy(flow.spiking_mode, flow.ttfs_cycle_schedule, flow.firing_mode)
    return (bool(getattr(flow, "lif_execution_synchronized", False)) and flow.spiking_mode == "lif"
            and not policy.single_spike_io and not recording)


def run_neural_segment_counts(
    flow, input_spike_train, *, seg, T, batch_size, device,
    neuron_count_recorder: dict | None = None,
) -> torch.Tensor:
    """The synchronized count-domain reference: per core, memb = W·counts +
    bias·T (+ V0·θ) and the emitted count is the strict staircase — bit-equal
    to two-window execution for ANY arrival (§16 theorem), and latch-correct
    for level-gapped consumers (counts have no windows)."""
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
        stair = lif_count_staircase(
            total / float(T), theta, T, compare_mode=flow.thresholding_mode,
        )
        buffers[i] = stair * float(T) / torch.clamp(
            torch.as_tensor(theta, dtype=stair.dtype, device=stair.device),
            min=1e-12,
        )
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
