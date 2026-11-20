"""Shared compute-op dispatch and segment state plumbing for simulators.

[thru:w2] The reference executors' one rule for a program of input slots (the batch pipeline): input k of a batch is
slot k mod B, and the program runs once per slot that holds a row, on that slot's rows only (``input_slot_runs``), so
a slot stage reads and writes only its slot's rows and a read of a row no earlier stage of its input wrote is refused by
name (``hybrid_stage_runner.require_inputs_produced``). The runs' rows are put back in batch order (``scatter_rows``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.chip_simulation.hybrid_run.host_compute import (
    execute_compute_op_numpy as execute_compute_op_numpy,
    execute_compute_op_torch as execute_compute_op_torch,
)
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.packing.executed_reads import op_source_counts_fresh
from mimarsinan.mapping.support.activation_scales import scalar_node_scale
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.wavefront_stages import program_slots, slot_rows
from mimarsinan.chip_simulation.hybrid_run.input_shifts import (
    apply_input_shifts_numpy as apply_input_shifts_numpy,
    compute_input_state_with_shifts as compute_input_state_with_shifts,
)


def input_slot_runs(stages: Sequence[Any], batch: int) -> List[Tuple[Optional[int], Optional[List[int]]]]:
    """[thru:w2] The runs a reference executor makes of a program on ``batch`` inputs: ``[(None, None)]`` (one run over
    every row) without input slots, else per input slot that holds a row its ``(slot, rows)``."""
    slots = program_slots(stages)
    if slots is None:
        return [(None, None)]
    return [(b, rows) for b in range(slots) if (rows := list(slot_rows(b, slots, batch)))]


def scatter_rows(parts: Sequence[Tuple[List[int], Any]], batch: int) -> Any:
    """[thru:w2] One array (tensor or ndarray, rows first) of ``batch`` rows from each run's ``(rows, value)``."""
    first = parts[0][1]
    if torch.is_tensor(first):
        out = torch.zeros((batch, *first.shape[1:]), dtype=first.dtype, device=first.device)
        for rows, value in parts:
            out[torch.as_tensor(rows, dtype=torch.long, device=first.device)] = value
        return out
    out = np.zeros((batch, *np.shape(first)[1:]), dtype=np.asarray(first).dtype)
    for rows, value in parts:
        out[np.asarray(rows, dtype=np.int64)] = value
    return out


def assemble_segment_input_torch(
    input_map,
    state_buffer: Dict[int, torch.Tensor],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a segment's composite input tensor from the state buffer."""
    total_size = max((s.offset + s.size for s in input_map), default=0)
    inp = torch.zeros(batch_size, total_size, device=device, dtype=dtype)
    for s in input_map:
        buf = state_buffer[s.node_id]
        inp[:, s.offset : s.offset + s.size] = buf[:, : s.size].to(dtype)
    return inp


def store_segment_output_torch(
    output_map,
    state_buffer: Dict[int, torch.Tensor],
    output_tensor: torch.Tensor,
) -> None:
    """Parse a segment's output tensor into the state buffer."""
    for s in output_map:
        state_buffer[s.node_id] = output_tensor[:, s.offset : s.offset + s.size]


def gather_final_output_torch(
    output_sources: np.ndarray,
    state_buffer: Dict[int, torch.Tensor],
    original_input: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble the network's final output from the state buffer."""
    flat = output_sources.flatten()
    out = torch.zeros(batch_size, len(flat), device=device, dtype=dtype)
    for idx, src in enumerate(flat):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        elif src.is_input():
            out[:, idx] = original_input[:, src.index].to(dtype)
        elif src.is_always_on():
            out[:, idx] = 1.0
        else:
            out[:, idx] = state_buffer[src.node_id][:, src.index].to(dtype)
    return out



def assemble_segment_input_numpy(
    input_map,
    state_buffer: Dict[int, np.ndarray],
    num_samples: int,
    *,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    """Numpy segment-input assembly at the declared ``dtype`` ([U-DAR] never defaulted: the record's boundary dtype,
    ``SPIKING_ARITHMETIC.boundary_np``, the HCM's, or a backend's named departure from it)."""
    total_size = max((s.offset + s.size for s in input_map), default=0)
    inp = np.zeros((num_samples, total_size), dtype=dtype)
    for s in input_map:
        buf = state_buffer[s.node_id]
        inp[:, s.offset : s.offset + s.size] = buf[:, : s.size]
    return inp


def store_segment_output_numpy(
    output_map,
    state_buffer: Dict[int, np.ndarray],
    output: np.ndarray,
) -> None:
    """Numpy analogue of :func:`store_segment_output_torch`."""
    for s in output_map:
        state_buffer[s.node_id] = output[:, s.offset : s.offset + s.size]


def gather_final_output_numpy(
    output_sources: np.ndarray,
    state_buffer: Dict[int, np.ndarray],
    original_input: np.ndarray,
    num_samples: int,
    *,
    dtype: npt.DTypeLike,
) -> np.ndarray:
    """Numpy analogue of :func:`gather_final_output_torch`, at the declared ``dtype`` ([U-DAR] the record's final-read
    dtype, ``SPIKING_ARITHMETIC.final_read_np``)."""
    flat_sources = output_sources.flatten()
    out = np.zeros((num_samples, len(flat_sources)), dtype=dtype)
    for idx, src in enumerate(flat_sources):
        if not isinstance(src, IRSource):
            continue
        if src.is_off():
            continue
        elif src.is_input():
            out[:, idx] = original_input[:, src.index]
        elif src.is_always_on():
            out[:, idx] = 1.0
        else:
            out[:, idx] = state_buffer[src.node_id][:, src.index]
    return out



def compute_op_owns_scale_domain(op) -> bool:
    """Whether the op module transcodes rate<->absolute itself (``ScaleNormalizingWrapper``).

    Such an op computes ``f(r_i * s_i) / s_out`` internally; applying the outer
    stage scales around it would apply the domain transcode twice.
    """
    params = getattr(op, "params", None) or {}
    return isinstance(params.get("module"), ScaleNormalizingWrapper)


def resolve_stage_compute_scales(
    mapping,
    op_id: int,
    *,
    apply_ttfs: bool = True,
    op=None,
) -> tuple[float, float]:
    """Return ``(input_scale, output_scale)`` for a hybrid compute stage.

    ``apply_ttfs`` selects the TTFS value-domain convention: decode the gathered
    rates by the producer scale, run the op in absolute units, re-normalize by the
    op's boundary out-scale. Pass ``op`` so wrapper-owned ops keep ``(1, 1)``.
    """
    out_scales = getattr(mapping, "node_activation_scales", {})
    in_scales = getattr(mapping, "node_input_activation_scales", out_scales)
    if not apply_ttfs:
        return 1.0, 1.0
    if op is not None and compute_op_owns_scale_domain(op):
        return 1.0, 1.0
    # kappa_fold entries may be per-channel; this stage API is scalar — the
    # collapse is the SSOT mean (scalar_node_scale), identical for scalars.
    return (
        scalar_node_scale(in_scales.get(op_id, 1.0)),
        scalar_node_scale(out_scales.get(op_id, 1.0)),
    )


def decref_consumers(
    state_buffer,
    remaining: Dict[int, int],
    src_ids: Iterable[int], state_buffer_spikes=None,
) -> None:
    """Drop state-buffer entries whose consumer refcount reaches zero."""
    for nid in src_ids:
        if nid < 0:
            continue
        r = remaining.get(nid)
        if r is None:
            continue
        r -= 1
        if r <= 0:
            remaining.pop(nid, None)
            state_buffer.pop(nid, None)
            if state_buffer_spikes is not None:  # trains share the lifetime
                state_buffer_spikes.pop(nid, None)
        else:
            remaining[nid] = r


def op_source_counts(op) -> Dict[int, int]:
    """``{producer node id: how many of the op's inputs read it}``, counted once.

    A host op carries one input source per axon (up to 605,184 on the real
    ViT), and its wiring is static after packing, so the refcount multiset is
    computed on first use and reused for every forward.
    """
    cached = getattr(op, "_source_counts_cache", None)
    if cached is not None:
        return cached
    counts = op_source_counts_fresh(op)   # [W15] the derivation the static check holds the memo to
    op._source_counts_cache = counts
    return counts


def decref_op_consumers(
    state_buffer, remaining: Dict[int, int], op, state_buffer_spikes=None,
) -> None:
    """``decref_consumers`` for one host op, by producer instead of by axon.

    Subtracting a producer's occurrence count in one step is identical to
    subtracting it one occurrence at a time: nothing observes the buffer
    between an op's own decrements.
    """
    for nid, times in op_source_counts(op).items():
        r = remaining.get(nid)
        if r is None:
            continue
        r -= times
        if r <= 0:
            remaining.pop(nid, None)
            state_buffer.pop(nid, None)
            if state_buffer_spikes is not None:
                state_buffer_spikes.pop(nid, None)
        else:
            remaining[nid] = r
