"""[cert-plan W1] stage-flat streaming executor: cycles loop, cores batched.

Same per-cycle physics as the per-core reference loop (the policy's
``advance`` is the shared kernel); only the charge layout changes. Cores are
bucketed by (latency, axons, neurons) and laid out latency-sorted, so each
cycle's active set is a run of whole buckets and latency gating needs no
masking. Window-gated output accumulation equals each producer's total
in-window fires, so outputs assemble once at the end. Bit-equal to the
reference loop (locked by test_packed_cycle_equivalence)."""

from __future__ import annotations

import torch

from mimarsinan.models.spiking.hybrid.executors.packed_stage import (
    PackedStage as PackedStage,
    _Bucket as _Bucket,
    _index as _index,
    build_packed_stage,
)
from mimarsinan.models.spiking.hybrid.membrane_readout import (
    stash_membrane_readout_correction,
)
from mimarsinan.models.spiking.hybrid.carry import (
    carry_plan_for,
    record_carry,
)
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE


def run_neural_segment_packed(
    flow, input_spike_train, *, seg, stage, T, batch_size, device, policy,
    readout_corrections=None, output_train=None,
) -> torch.Tensor:
    """Stage-flat twin of the reference cycle loop (multi-spike, non-recording).

    ``output_train`` is the carry seam: pass a one-element list to also receive the
    segment's output RASTER ``(T, B, out_dim)`` in PRODUCER-LOCAL time, which is what
    a later pass of the same segment must replay verbatim. Recorded per cycle from
    the same ``fires`` the counts accumulate, so ``raster.sum(0) == counts`` holds by
    construction — the cheap invariant that says the two agree.
    """
    packed = seg.get("packed")
    if packed is None:
        packed = build_packed_stage(seg, device)
        seg["packed"] = packed

    latency = seg["latency"]
    cycles = int(latency) + T
    output_sources = seg["output_sources"]
    output_spans = seg["output_spans"]

    state = policy.make_state(batch_size, max(packed.total_neurons, 1),
                              device, COMPUTE_DTYPE)
    membrane_init = float(getattr(flow, "lif_membrane_init", 0.0))
    if membrane_init and packed.total_neurons:
        state["memb"] += membrane_init * packed.theta_flat

    fires = torch.zeros(batch_size, max(packed.total_neurons, 1),
                        device=device, dtype=COMPUTE_DTYPE)
    counts = torch.zeros_like(fires)
    zeros_in = torch.zeros(batch_size, input_spike_train.shape[2],
                           device=device, dtype=COMPUTE_DTYPE)
    train = input_spike_train.to(COMPUTE_DTYPE)

    carry = None
    carry_plan: list = []
    if output_train is not None:
        carry = torch.zeros(T, batch_size, len(output_sources),
                            device=device, dtype=COMPUTE_DTYPE)
        carry_plan = carry_plan_for(output_spans, packed, seg["cores"], T)

    for cycle in range(cycles):
        # Two-phase, mirroring the reference loop: gather every active
        # bucket's signals against the PREVIOUS cycle's fires, then advance.
        staged = []
        for bucket in packed.buckets:
            if not (bucket.latency <= cycle < bucket.latency + T):
                continue
            group = len(bucket.core_indices)
            signals = torch.zeros(
                batch_size, group * bucket.n_axons,
                device=device, dtype=COMPUTE_DTYPE,
            )
            if bucket.on_dst is not None:
                signals.index_fill_(1, bucket.on_dst, 1.0)
            if bucket.inp_dst is not None and bucket.inp_src is not None:
                local = cycle - bucket.latency
                src = train[local] if 0 <= local < T else zeros_in
                signals.index_copy_(
                    1, bucket.inp_dst, src.index_select(1, bucket.inp_src))
            if bucket.buf_dst is not None and bucket.buf_src is not None:
                signals.index_copy_(
                    1, bucket.buf_dst, fires.index_select(1, bucket.buf_src))
            staged.append((bucket, signals))
        for bucket, signals in staged:
            group = len(bucket.core_indices)
            grouped = signals.reshape(batch_size, group, bucket.n_axons)
            charge = torch.einsum("gna,bga->bgn", bucket.weights, grouped)
            if bucket.bias is not None:
                charge = charge + bucket.bias
            contribution = charge.reshape(batch_size, -1)
            n0, n1 = bucket.neuron_start, bucket.neuron_end
            slice_state = {k: v[:, n0:n1] for k, v in state.items()}
            out = policy.advance(
                slice_state, contribution, packed.theta_flat[n0:n1],
                thresholding_mode=flow.thresholding_mode,
                output_dtype=COMPUTE_DTYPE,
            )
            fires[:, n0:n1] = out
            counts[:, n0:n1] += out

        if carry is not None:
            record_carry(carry, carry_plan, cycle=cycle, fires=fires,
                         train=train, T=T)

    output_counts = torch.zeros(
        batch_size, len(output_sources), device=device, dtype=COMPUTE_DTYPE)
    input_total = train[:T].sum(dim=0)
    for sp in output_spans:
        d0, d1 = int(sp.dst_start), int(sp.dst_end)
        if sp.kind == "off":
            continue
        if sp.kind == "on":
            output_counts[:, d0:d1] = float(T)
            continue
        if sp.kind == "input":
            output_counts[:, d0:d1] = input_total[:, int(sp.src_start):int(sp.src_end)]
            continue
        src_core = int(sp.src_core)
        off = packed.neuron_offset.get(src_core)
        if off is None:
            continue
        output_counts[:, d0:d1] = counts[:, off + int(sp.src_start):off + int(sp.src_end)]

    if carry is not None:
        assert output_train is not None
        output_train.append(carry)

    if readout_corrections is not None and getattr(flow, "membrane_readout", False):
        # [C2] per-core membrane views over the flat state feed the same stash.
        cores = seg["cores"]
        per_core_states = []
        for i, c in enumerate(cores):
            n = max(int(c.neurons_per_core - c.available_neurons), 1)
            off = packed.neuron_offset.get(i)
            memb = (
                state["memb"][:, off:off + n] if off is not None
                else torch.zeros(batch_size, n, device=device, dtype=COMPUTE_DTYPE)
            )
            per_core_states.append({"memb": memb})
        stash_membrane_readout_correction(
            flow, seg=seg, stage=stage, output_counts=output_counts,
            output_spans=output_spans, neuron_states=per_core_states,
            thresholds=seg["thresholds"], single_spike=False,
            readout_corrections=readout_corrections,
        )
    return output_counts
