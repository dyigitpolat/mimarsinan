"""Reassemble per-perceptron, per-neuron arrays from a packed HCM record.

The hybrid hard-core mapping (HCM) packs each IR neural core into hard cores and
may *split* a core's neurons across hard cores (neuron tiling), split its axons
across partial-sum cores (psum tiling), or *coalesce* several cores into one. A
recorded run therefore scatters one perceptron's neurons across many hard cores
with bookkeeping roles. These helpers invert that packing so a test can compare
the reassembled per-neuron values against the torch NF nodes one-to-one.

``placement.ir_node_id == IR NeuralCore.id`` (compaction preserves ids and the
segment remap rewrites only sources), so cores group + order by id directly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def hcm_per_perceptron_counts(record, hybrid) -> dict[int, np.ndarray]:
    """Per-perceptron, per-neuron HCM output spike counts.

    Concatenates each perceptron's IR cores in IR-id order (local-neuron order),
    filtering coalescing/psum partial cores and reassembling neuron-split
    fragments by their original-neuron offset.
    """
    return _reassemble(record, hybrid, lambda cc, n0, n1: np.asarray(
        cc.output_spike_count[n0:n1], dtype=np.int64))


def hcm_per_perceptron_values(record, hybrid, extract) -> dict[int, np.ndarray]:
    """Generalized reassembly: ``extract(core_counts, n0, n1) -> np.ndarray``."""
    return _reassemble(record, hybrid, extract)


def _reassemble(record, hybrid, extract) -> dict[int, np.ndarray]:
    # ir_id -> list of (orig_neuron_offset, perceptron_index, values). Neuron-split
    # fragments of one IR core share its id; reassemble them by original offset.
    frags: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
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
                frags[pl["ir_node_id"]].append((orig_off, pi, values))

    # Reassemble each IR core's neurons (fragments in original-offset order).
    per_core: dict[int, tuple[int, np.ndarray]] = {}
    for ir_id, flist in frags.items():
        flist.sort(key=lambda f: f[0])
        per_core[ir_id] = (flist[0][1], np.concatenate([v for _, _, v in flist]))

    by_perc: dict[int, list[int]] = defaultdict(list)
    for ir_id, (pi, _) in per_core.items():
        by_perc[pi].append(ir_id)
    return {
        pi: np.concatenate([per_core[cid][1] for cid in sorted(ids)])
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
