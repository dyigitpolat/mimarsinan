"""The hybrid flow's segment tensor cache: one stage's uploaded tensors and their byte accounting."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.packing.executed_reads import bank_read
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.models.spiking.signal_spans import SpanFillPlan
from mimarsinan.models.spiking.spiking_config import COMPUTE_DTYPE

# Retention budget for uploaded segment tensors; tier-0 vehicles fit in a few
# tens of MB (float64), large vehicles fall back to per-stage eviction.
SEGMENT_CACHE_MAX_BYTES = 256 * 1024 * 1024


def segment_entry_nbytes(entry: dict) -> int:
    """Device bytes held by one cache entry, deduped by tensor storage."""
    seen: set[int] = set()
    total = 0
    tensors: list = list(entry.get("core_params", []) or [])
    tensors += [b for b in entry.get("hw_biases", []) or [] if b is not None]
    tensors += list(entry.get("thresholds", []) or [])
    tensors += list((entry.get("bank_tensors", {}) or {}).values())
    for plan in entry.get("axon_fill_plans", []) or []:
        tensors += list(plan.tensors())
    # [F4] the packed-cycle executor memoises its PackedStage into THIS entry
    # (``seg["packed"]``); each bucket holds a DENSE stack of the per-core
    # weight views. Uncounted, the budget under-reports by exactly the bytes
    # that make large stages expensive.
    packed = entry.get("packed")
    if packed is not None:
        tensors.append(getattr(packed, "theta_flat", None))
        for bucket in getattr(packed, "buckets", []) or []:
            tensors += [
                getattr(bucket, name, None) for name in (
                    "weights", "bias", "on_dst", "inp_dst", "inp_src",
                    "buf_dst", "buf_src",
                )
            ]
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        storage = t.untyped_storage()
        key = storage.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return total


def build_segment_tensors(mapping, device: torch.device) -> dict:
    """Upload one stage's hard cores: axon/output spans, gather plans, weights, biases, thresholds."""
    cores = mapping.cores
    output_sources = mapping.output_sources
    weight_banks = getattr(mapping, "weight_banks", None) or {}

    axon_spans = []
    for c in cores:
        if hasattr(c, "get_axon_source_spans"):
            axon_spans.append(c.get_axon_source_spans())
        else:
            axon_spans.append(compress_spike_sources(c.axon_sources))
    # W3b: spans are static per core for the whole simulation, so the
    # gather plan is precomputed once and reused every core-cycle.
    axon_fill_plans = [SpanFillPlan(spans, device) for spans in axon_spans]
    if hasattr(mapping, "get_output_source_spans"):
        output_spans = mapping.get_output_source_spans()
    else:
        output_spans = compress_spike_sources(
            list(np.asarray(output_sources, dtype=object).flatten())
        )

    bank_tensors: dict[int, torch.Tensor] = {}

    def _ensure_bank_tensor(bid: int) -> torch.Tensor:
        t = bank_tensors.get(bid)
        if t is not None:
            return t
        bank_mat = weight_banks.get(int(bid))
        if bank_mat is None:
            raise KeyError(
                f"HardCoreMapping references bank_id={bid} but "
                f"mapping.weight_banks does not contain it — "
                f"ir_graph_to_soft_core_mapping must propagate the bank."
            )
        t = torch.tensor(bank_mat.T, dtype=COMPUTE_DTYPE, device=device)
        bank_tensors[int(bid)] = t
        return t

    core_params: list[torch.Tensor] = []
    hw_biases = []
    for core_idx, core in enumerate(cores):
        used_ax = used_axons(core, min_one=True)
        used_neu = used_neurons(core, min_one=True)

        core_weight: torch.Tensor | None = None

        # [W15] the bank fast path's one definition (``executed_reads.bank_read``), which the static check reads too.
        bank = bank_read(mapping, core_idx)
        if bank is not None:
            bank_t = _ensure_bank_tensor(bank.bank_id)
            core_weight = bank_t[bank.neurons[0]:bank.neurons[1], bank.axons[0]:bank.axons[1]]

        if core_weight is None:
            tile = core.get_core_matrix()[:used_ax, :used_neu]
            core_weight = torch.tensor(
                tile.T, dtype=COMPUTE_DTYPE, device=device,
            )

        core_params.append(core_weight)

        bias = getattr(core, "hardware_bias", None)
        if bias is None:
            hw_biases.append(None)
        else:
            hw_biases.append(
                torch.tensor(
                    bias[:used_neu], dtype=COMPUTE_DTYPE, device=device,
                )
            )

    thresholds = [
        torch.tensor(float(core.threshold), dtype=COMPUTE_DTYPE, device=device)
        for core in cores
    ]

    cached = dict(
        device=device,
        cores=cores,
        output_sources=output_sources,
        axon_spans=axon_spans,
        axon_fill_plans=axon_fill_plans,
        output_spans=output_spans,
        bank_tensors=bank_tensors,
        core_params=core_params,
        thresholds=thresholds,
        hw_biases=hw_biases,
        nbytes=0,
    )
    cached["nbytes"] = segment_entry_nbytes(cached)
    return cached
