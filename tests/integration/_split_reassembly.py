"""Reassemble per-perceptron, per-neuron arrays from a packed HCM record.

The hybrid hard-core mapping (HCM) packs each IR neural core into hard cores and
may *split* a core's neurons across hard cores (neuron tiling), split its axons
across partial-sum cores (psum tiling), or *coalesce* several cores into one. A
recorded run therefore scatters one perceptron's neurons across many hard cores
with bookkeeping roles. These helpers invert that packing so a test can compare
the reassembled per-neuron values against the torch NF nodes one-to-one.

A perceptron's cores are ordered by their original-layer neuron position (the
JOINT ``(ir_core_id, neuron_range)`` key: ``perceptron_output_slice[0]`` for the
IR output tile, ``neuron_range_in_original`` for an HCM neuron-split fragment),
NOT by raw ``ir_node_id`` — id assignment is not monotone in the output slice
once IR-graph compaction reorders ids, so an id-sort would scramble attribution
under coalescing+output-tiling while the deployed value stays bit-exact (GAP-1).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def hcm_per_perceptron_counts(record, hybrid) -> dict[int, np.ndarray]:
    """Per-perceptron, per-neuron HCM output spike counts.

    Concatenates each perceptron's IR cores in original-layer neuron-position
    order (the joint ``(ir_core_id, neuron_range)`` key), filtering coalescing/psum
    partial cores and reassembling neuron-split fragments by their original-neuron
    offset.
    """
    return _reassemble(record, hybrid, lambda cc, n0, n1: np.asarray(
        cc.output_spike_count[n0:n1], dtype=np.int64))


def hcm_per_perceptron_values(record, hybrid, extract) -> dict[int, np.ndarray]:
    """Generalized reassembly: ``extract(core_counts, n0, n1) -> np.ndarray``."""
    return _reassemble(record, hybrid, extract)


def _reassemble(record, hybrid, extract) -> dict[int, np.ndarray]:
    # ir_id -> (perceptron_index, tile_offset, [(orig_neuron_offset, values), ...]).
    # Neuron-split fragments of one IR core share its id; reassemble them by their
    # offset within the original layer (HCM split offset). ``tile_offset`` is the IR
    # output-tile's start in the original layer (perceptron_output_slice[0]) — the
    # JOINT (ir_core_id, neuron_range) key that orders the tiles of one perceptron.
    frags: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    perc_of: dict[int, int] = {}
    tile_offset_of: dict[int, int] = {}
    for si, seg in record.segments.items():
        placements = hybrid.stages[si].hard_core_mapping.soft_core_placements_per_hard_core
        for ci, cc in enumerate(seg.cores):
            for pl in placements[ci]:
                pi = pl.get("perceptron_index")
                if pi is None or pi < 0:
                    continue
                if pl.get("coalescing_role") not in (None, "master", "accum"):
                    continue
                if pl.get("psum_role") not in (None, "accum"):
                    continue
                n0 = pl["neuron_offset"]
                n1 = n0 + pl["neurons"]
                values = extract(cc, n0, n1)
                orig_off = (
                    int(pl["neuron_range_in_original"][0])
                    if pl.get("split_group_id") is not None else 0
                )
                ir_id = pl["ir_node_id"]
                frags[ir_id].append((orig_off, values))
                perc_of[ir_id] = pi
                out_slice = pl.get("perceptron_output_slice")
                tile_offset_of[ir_id] = int(out_slice[0]) if out_slice is not None else 0

    # Reassemble each IR core's neurons (HCM-split fragments in original-offset order).
    per_core: dict[int, np.ndarray] = {
        ir_id: np.concatenate([v for _, v in sorted(flist, key=lambda f: f[0])])
        for ir_id, flist in frags.items()
    }

    by_perc: dict[int, list[int]] = defaultdict(list)
    for ir_id in per_core:
        by_perc[perc_of[ir_id]].append(ir_id)
    # Order a perceptron's cores by their original-layer neuron position
    # (tile_offset), not by ir-id: id assignment is not monotone in the slice once
    # IR-graph compaction reorders ids, so sorted(ir_id) would scramble attribution.
    return {
        pi: np.concatenate(
            [per_core[cid] for cid in sorted(ids, key=lambda c: (tile_offset_of[c], c))]
        )
        for pi, ids in by_perc.items()
    }


def torch_lif_node_counts(forward_fn, nodes, x, T) -> dict[int, np.ndarray]:
    """Per-LIF-node, per-neuron spike counts (flattened, summed over T cycles).

    ``forward_fn`` must run the chip-aligned cycle-accurate forward (e.g.
    ``lambda: chip_aligned_segment_forward(flow, x, T)``); ``nodes`` maps a
    perceptron index to its ``LIFActivation`` (scale 1 -> binary spikes).
    """
    acc = {i: None for i in nodes}

    def mk(i):
        def hook(mod, inp, out):
            v = out.detach().reshape(out.shape[0], -1)
            acc[i] = v.clone() if acc[i] is None else acc[i] + v
        return hook

    handles = [nodes[i].register_forward_hook(mk(i)) for i in nodes]
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for h in handles:
            h.remove()
    return {i: acc[i][0].numpy().astype(np.int64) for i in acc if acc[i] is not None}
