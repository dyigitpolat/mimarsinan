"""The HOST group of the physics vocabulary: the deployment host on the other side of the NeuralOps/ComputeOps
boundary, and its link to the chip. Rows as ``vocabulary.ROWS`` states them: ``(key, group, dimension,
display_unit, multiplicand, doc)``. [U-J] The edge host's rows: its idle power, its memory and the host-chip link.
[U-HOST] Every row here is admitted only from an identified host description (``hosts.admission``), else refused by
name: a host row prices a machine the toolchain does not run on, from its published data, never a measurement.
[U-HOST2] Admitted means the description's own row or a declared derivation of its rows, judged on the value; [U-HOST3]
judged against the description held under the digest it was sealed under, and printed with its declaration."""

from __future__ import annotations

import json
from typing import Tuple

from mimarsinan.deployment_record.units import DIMENSIONLESS, ENERGY, POWER, RATE, TIME

HOST = "host"
#: [U-J] Every byte that crosses the host-chip boundary per inference, both directions (alternatives of one use).
_CROSSING = "carry_in_bytes | carry_out_bytes | segment_in_bytes | segment_out_bytes"

HOST_ROWS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("p_host", HOST, POWER, "W", "latency_host_s",
     "Power drawn by the host while it executes ComputeOps (over the priced host time). Without it, moving work "
     "host-side is free and `subsume` wins every comparison by disappearing."),
    ("host_compute_rate", HOST, DIMENSIONLESS, "fraction", "declared, not multiplied",
     "Speed of the deployment host relative to the development machine. [U-HOST] No priced term reads it: the "
     "development machine's measured wall is a diagnostic (stage_timing.host_wall), never a host price."),
    ("t_host_op_overhead", HOST, TIME, "us", "compute_op_count",
     "Per-invocation wall of one host ComputeOp through the deployment's own "
     "dispatch path (segment I/O, tensor conversion) — the dominant host cost "
     "at small op sizes; [U-HOST] the deployment host's, from its description's published data "
     "(scripts/calibrate_host.py measures the development machine: a diagnostic, never this row)."),
    ("host_macs_per_s", HOST, RATE, "G/s", "host_macs",
     "Sustained forward-MAC throughput of the deployment host, pricing host time "
     "STATICALLY (host_macs / host_macs_per_s) on every plane ([U-HOST] a measured wall prices nothing) — the "
     "host model of owner decision 3. A DIVIDING constant: corners "
     "flip so the band stays monotone."),
    # [U-J] The host idles while it waits; its memory and its link to the chip move every crossing byte.
    ("p_host_idle", HOST, POWER, "W", "e2e_latency_s",
     "Power the host draws while it waits during an inference (on the chip and on the link): priced over the "
     "inference's latency less the host's own time (which p_host prices). [U-J2] Whose energy it is the host's "
     "description states (energy_plane): system-level, it is a floor under every host state, priced with the "
     "whole-system line (energy_system_mj) and never in the headline, and p_host is charged above it."),
    ("e_host_mem_per_byte", HOST, ENERGY, "pJ", _CROSSING,
     "Energy of one byte the host-chip link moves into or out of host memory (each crossing byte is written to or "
     "read from it once), above the host's idle power; a host operator's own memory traffic is inside p_host. "
     "[U-J3] The pass carry and the weight writes cross only to a host-owned backing store (hosts.store). [U-J4] "
     "With e_link_per_byte it is the whole price of a crossing byte: none of the chip's backing-store rows "
     "(e_dma_per_byte, e_readout_per_byte) is charged on it."),
    ("t_host_mem_per_byte", HOST, TIME, "ns", "host_memory_bound_bytes",
     "Host memory time per byte (the inverse sustained bandwidth): a MAC-free host operator's bytes are priced at "
     "it (memory binds an element-wise or pooling operator), and it is the memory stage of a link transfer, whose "
     "bytes move no faster than the slower of it and t_link_per_byte."),
    ("e_link_per_byte", HOST, ENERGY, "pJ", _CROSSING,
     "Energy of the host-chip link per payload byte it moves, either direction: the PHY ends and channel its host "
     "description counts (its row's note states which). [U-J3] The program's own crossing (the segment I/O, a "
     "host-sourced payload) always; the pass carry, the weight writes and a program load only to a host-owned "
     "backing store (hosts.store). [U-J4] A crossing byte pays it and e_host_mem_per_byte, never the chip's "
     "backing-store rows (e_dma_per_byte in, e_readout_per_byte out); no row prices the chip's own end of the link, "
     "so the link's energy names it unpriced (a boundary) at the corners where the host description's link_ends "
     "leave it out (U-J5)."),
    ("t_link_per_byte", HOST, TIME, "ns", _CROSSING,
     "Host-chip link time per payload byte, either direction (the inverse payload bandwidth)."),
    ("t_link_transfer", HOST, TIME, "us", "host_link_transfers",
     "Fixed latency of one host-chip transfer (a DMA's setup and round trip): every boundary a program crosses, "
     "in each direction, is one transfer."),
)


def declaration_statement(text: str) -> str:
    """[U-HOST3] What a sealed ``derived_from`` declaration (``hosts.derivation``'s one form, as its JSON text) states,
    as a derived row's printed basis opens with it: ``'derived_from: the <combine> of <factor> x <row> and ...; '``
    (empty without a declaration; a text outside the form is printed as it stands, and the admission rule refuses it)."""
    if not text:
        return ""
    try:
        data = json.loads(text)
        terms = " and ".join(f"{float(factor)!r} x {row}" for row, factor in sorted(dict(data["of"]).items()))
        return f"derived_from: the {data.get('combine', 'sum')} of {terms}; "
    except (ValueError, TypeError, KeyError):  # no JSON, no "of", or a non-mapping where one is read
        return f"derived_from: {text}; "
